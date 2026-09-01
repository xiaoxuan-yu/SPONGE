// Production count kernel: one warp per candidate-SCI, split into
// kClusteredCountSubgroups (4) subgroups
// of 8 lanes. The 8 lanes map to the (<=8) i-clusters exactly as in the
// baseline. Each subgroup processes a strided slice of the SCI's candidate-leaf
// list; the cross-leaf dedup (processed_cluster_end) is replaced by the
// provably-equivalent local form deduped_start[k]=max(start[k], end[k-1])
// because candidate leaves are cluster-start-sorted per SCI (verified: 0
// non-monotonic of 10.9M). Per-(shift) counts accumulate via shared atomicAdd
// (order-independent => bit-exact). i-side shared data stays per-warp
// (read-only, shared by all subgroups); j-side scratch and the j-signature get
// a per-subgroup dimension. Validated bit-for-bit against the baseline count
// kernel before it is allowed to feed the build.
template <bool kParallelAccum, bool kParallelFragmentEmit,
          bool kFixedShiftLeafScreenedSpecialized = false,
          bool kLightFragmentOnly = false, bool kFixedLightSassOpt = false,
          bool kUseCandidateLeafMetadata = false, bool kFixedLightSlim = false>
static __global__ void Count_Nbnxm_Payload_From_Candidate_Leaves_Subgroup(
    const int candidate_sci_numbers, const int sci_shift_numbers,
    const int cluster_size, const int super_cluster_clusters,
    const int local_atom_numbers, const float cutoff, const LTMatrix3 cell,
    const LTMatrix3 rcell, const int* permutation, const int* cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* super_cluster_offsets, const int* cluster_to_supercluster,
    const int* sci_supercluster_ids, const VECTOR* super_cluster_centers,
    const int* candidate_leaf_offsets, const int* candidate_leaf_ids,
    const int* candidate_leaf_prev_running_max_ends,
    const int candidate_leaf_cluster_stride,
    const unsigned int* candidate_leaf_reach_masks,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, const float* cluster_radii,
    const uint64_t* cluster_molecule_signatures,
    const int* cluster_molecule_ids, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers,
    const float record_stream_cutoff,
    const VECTOR* crd, int* sci_shift_flags, int* cjpacked_group_counts,
    int* exclusion_counts, int* record_stream_source_rows,
    int* record_stream_source_counts_by_candidate,
    const bool accumulate_record_stream_source_rows_by_candidate,
    const int* active_candidate_sci_mask, const int max_leaf_cluster_span,
    CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT* count_light_source_fragments,
    const int count_source_fragment_capacity, int* count_source_fragment_cursor,
    int* count_source_fragment_overflow_rows)
{
    constexpr int kSubgroupSize = kClusteredClusterSize;  // 8
    constexpr int kCountSubgroups =
        kClusteredBuilderWarpSize / kSubgroupSize;  // 4
    static_assert(!kFixedLightSassOpt || (kFixedShiftLeafScreenedSpecialized &&
                                          kLightFragmentOnly),
                  "fixed-light SASS opt requires specialized light count path");
    static_assert(
        !kUseCandidateLeafMetadata || kFixedShiftLeafScreenedSpecialized,
        "candidate-leaf count metadata is fixed-shift specialized only");
    static_assert(
        !kFixedLightSlim ||
            (kFixedShiftLeafScreenedSpecialized && kLightFragmentOnly &&
             kFixedLightSassOpt && !kUseCandidateLeafMetadata),
        "fixed-light slim requires the peak fixed light count path");
    const int lane_id = threadIdx.x & (warpSize - 1);
    const int warp_id = threadIdx.x / warpSize;
    const int warps_per_block = blockDim.x / warpSize;
    const int subgroup = lane_id / kSubgroupSize;  // 0..3
    const int sublane =
        lane_id % kSubgroupSize;  // 0..7 == i_local / j-atom lane
    const unsigned int subgroup_mask = 0xFFu << (subgroup * kSubgroupSize);
    const int candidate_sci = blockIdx.x * warps_per_block + warp_id;
    if (candidate_sci >= candidate_sci_numbers)
    {
        return;
    }
    if (active_candidate_sci_mask != NULL &&
        active_candidate_sci_mask[candidate_sci] == 0)
    {
        return;
    }

    (void)sci_shift_numbers;
    (void)super_cluster_centers;
    (void)cluster_radii;
    if constexpr (!kUseCandidateLeafMetadata)
    {
        (void)candidate_leaf_prev_running_max_ends;
    }
    const int sci_base = candidate_sci / kClusteredShiftCount;
    const int fixed_shift_id = candidate_sci % kClusteredShiftCount;
    const VECTOR fixed_shift_vec =
        Shift_Vector_From_Id(fixed_shift_id, cell);
    const int super_i = sci_supercluster_ids[sci_base];
    const int cluster_i_start = super_cluster_offsets[super_i];
    const int cluster_i_end = super_cluster_offsets[super_i + 1];
    constexpr int kWarpsPerBlock =
        kClusteredBuilderBlockSize / kClusteredBuilderWarpSize;
    // i-side data is shared by all subgroups of a warp (read-only): per-warp.
    __shared__ unsigned int
        shared_i_local_masks[kWarpsPerBlock][kClusteredMaxSuperClusterClusters];
    __shared__ int shared_i_atom_ids[kWarpsPerBlock]
                                    [kClusteredMaxSuperClusterClusters]
                                    [kClusteredClusterSize];
    __shared__ uint64_t
        shared_i_signatures[kWarpsPerBlock][kClusteredMaxSuperClusterClusters];
    __shared__ int shared_i_molecule_ids[kWarpsPerBlock]
                                        [kClusteredMaxSuperClusterClusters]
                                        [kClusteredClusterSize];
    __shared__ float shared_i_center_x[kWarpsPerBlock]
                                      [kClusteredMaxSuperClusterClusters];
    __shared__ float shared_i_center_y[kWarpsPerBlock]
                                      [kClusteredMaxSuperClusterClusters];
    __shared__ float shared_i_center_z[kWarpsPerBlock]
                                      [kClusteredMaxSuperClusterClusters];
    __shared__ float shared_i_extent_x[kWarpsPerBlock]
                                      [kClusteredMaxSuperClusterClusters];
    __shared__ float shared_i_extent_y[kWarpsPerBlock]
                                      [kClusteredMaxSuperClusterClusters];
    __shared__ float shared_i_extent_z[kWarpsPerBlock]
                                      [kClusteredMaxSuperClusterClusters];
    // j-side scratch + j-signature are per-subgroup (each subgroup works a
    // different cluster_j concurrently).
    __shared__ int shared_j_atom_ids[kWarpsPerBlock][kCountSubgroups]
                                    [kClusteredClusterSize];
    __shared__ uint64_t shared_j_signature[kWarpsPerBlock][kCountSubgroups];
    __shared__ int shared_j_molecule_ids[kWarpsPerBlock][kCountSubgroups]
                                        [kClusteredClusterSize];
    __shared__ unsigned int
        shared_parallel_split_imasks[kWarpsPerBlock][kCountSubgroups]
                                    [kClusteredWarpSplitCount];
    __shared__ unsigned long long
        shared_parallel_exclusion_masks[kWarpsPerBlock][kCountSubgroups]
                                       [kClusteredMaxSuperClusterClusters];
    // shift counters accumulated across subgroups via atomicAdd (order-free).
    __shared__ int shared_shift_record_counts[kWarpsPerBlock]
                                             [kClusteredShiftCount];
    __shared__ int shared_shift_exclusion_counts[kWarpsPerBlock]
                                                [kClusteredShiftCount];
    __shared__ int shared_record_stream_source_counts[kWarpsPerBlock];
    const bool has_molecule_metadata =
        cluster_molecule_signatures != NULL && cluster_molecule_ids != NULL;
    const bool use_candidate_record_stream_source_count =
        record_stream_source_counts_by_candidate != NULL ||
        (accumulate_record_stream_source_rows_by_candidate &&
         record_stream_source_rows != NULL);
    const bool emit_count_light_source_fragments =
        count_light_source_fragments != NULL &&
        count_source_fragment_capacity > 0 &&
        count_source_fragment_cursor != NULL &&
        count_source_fragment_overflow_rows != NULL;
    const bool count_record_stream_source_rows =
        record_stream_source_rows != NULL ||
        record_stream_source_counts_by_candidate != NULL ||
        emit_count_light_source_fragments;
    const float record_stream_cutoff_sq =
        record_stream_cutoff * record_stream_cutoff;
    int local_record_stream_source_count = 0;
    int local_shift_record_count = 0;
    int local_shift_exclusion_count = 0;
    // Load i-cluster metadata (per-warp, all 32 lanes cooperate).
    if (lane_id < kClusteredMaxSuperClusterClusters)
    {
        const int cluster_i = cluster_i_start + lane_id;
        if (cluster_i < cluster_i_end)
        {
            const VECTOR center_i = cluster_centers[cluster_i];
            const VECTOR extent_i = cluster_extents[cluster_i];
            shared_i_local_masks[warp_id][lane_id] =
                cluster_local_masks[cluster_i];
            shared_i_center_x[warp_id][lane_id] = center_i.x;
            shared_i_center_y[warp_id][lane_id] = center_i.y;
            shared_i_center_z[warp_id][lane_id] = center_i.z;
            shared_i_extent_x[warp_id][lane_id] = extent_i.x;
            shared_i_extent_y[warp_id][lane_id] = extent_i.y;
            shared_i_extent_z[warp_id][lane_id] = extent_i.z;
            shared_i_signatures[warp_id][lane_id] =
                has_molecule_metadata ? cluster_molecule_signatures[cluster_i]
                                      : 0ull;
        }
        else
        {
            shared_i_local_masks[warp_id][lane_id] = 0u;
            shared_i_center_x[warp_id][lane_id] = 0.0f;
            shared_i_center_y[warp_id][lane_id] = 0.0f;
            shared_i_center_z[warp_id][lane_id] = 0.0f;
            shared_i_extent_x[warp_id][lane_id] = 0.0f;
            shared_i_extent_y[warp_id][lane_id] = 0.0f;
            shared_i_extent_z[warp_id][lane_id] = 0.0f;
            shared_i_signatures[warp_id][lane_id] = 0ull;
        }
    }
    for (int atom_slot = lane_id;
         atom_slot < kClusteredMaxSuperClusterClusters * kClusteredClusterSize;
         atom_slot += warpSize)
    {
        const int i_local = atom_slot / kClusteredClusterSize;
        const int atom_lane = atom_slot % kClusteredClusterSize;
        const int cluster_i = cluster_i_start + i_local;
        if (cluster_i < cluster_i_end)
        {
            const int sorted_atom_i = cluster_offsets[cluster_i] + atom_lane;
            shared_i_atom_ids[warp_id][i_local][atom_lane] =
                permutation[sorted_atom_i];
            shared_i_molecule_ids[warp_id][i_local][atom_lane] =
                has_molecule_metadata
                    ? cluster_molecule_ids[cluster_i * kClusteredClusterSize +
                                           atom_lane]
                    : -1;
        }
        else
        {
            shared_i_atom_ids[warp_id][i_local][atom_lane] = -1;
            shared_i_molecule_ids[warp_id][i_local][atom_lane] = -1;
        }
    }
    const int sci_shift_base = candidate_sci;
    if (lane_id < kClusteredShiftCount)
    {
        shared_shift_record_counts[warp_id][lane_id] = 0;
        shared_shift_exclusion_counts[warp_id][lane_id] = 0;
    }
    if (lane_id == 0)
    {
        shared_record_stream_source_counts[warp_id] = 0;
    }
    __syncwarp();
    const int active_cluster_count = cluster_i_end - cluster_i_start;
    const unsigned int active_i_lane_mask =
        active_cluster_count > 0
            ? ((1u << static_cast<unsigned int>(active_cluster_count)) - 1u)
            : 0u;

    const int leaf_begin = candidate_leaf_offsets[candidate_sci];
    const int leaf_end = candidate_leaf_offsets[candidate_sci + 1];
    // Each subgroup walks a strided slice of the candidate-leaf list. The
    // baseline dedups via a serial running max of cluster_j_end over all prior
    // leaves; deduped_start = max(this leaf start, that running max). Candidate
    // leaves are cluster-start-sorted per SCI (Hilbert SFC default;
    // structurally guaranteed by cstone::singleTraversal visiting octants in
    // SFC-key order and the shared atom-sort key). We reconstruct the exact
    // running max by a backward scan with a PROVABLY-CORRECT early stop: since
    // starts are non-decreasing along the list, going back starts are
    // non-increasing, so any earlier leaf b' < b has end[b'] <= start[b'] +
    // S_max <= start[b] + S_max. Once start[b] + S_max <= cur_max, no
    // still-earlier leaf can raise the max, so we stop. S_max =
    // max_leaf_cluster_span is the exact per-leaf cluster-span upper bound
    // computed at rebuild (no magic number; holds for any system /
    // cornerstone_leaf_size / density).
    const int s_max = max_leaf_cluster_span;
    for (int candidate_idx = leaf_begin + subgroup; candidate_idx < leaf_end;
         candidate_idx += kCountSubgroups)
    {
        const int leaf_j = candidate_leaf_ids[candidate_idx];
        const int cluster_j_start = leaf_cluster_starts[leaf_j];
        const int cluster_j_end = leaf_cluster_ends[leaf_j];
        int prev_running_max_end = 0;
        const bool use_candidate_leaf_metadata =
            kUseCandidateLeafMetadata &&
            candidate_leaf_prev_running_max_ends != NULL;
        if (use_candidate_leaf_metadata)
        {
            prev_running_max_end =
                candidate_leaf_prev_running_max_ends[candidate_idx];
        }
        else
        {
            for (int b = candidate_idx - 1; b >= leaf_begin; b -= 1)
            {
                const int b_leaf = candidate_leaf_ids[b];
                const int b_start = leaf_cluster_starts[b_leaf];
                const int b_end = leaf_cluster_ends[b_leaf];
                if (b_end > prev_running_max_end)
                {
                    prev_running_max_end = b_end;
                }
                // Provable stop: any still-earlier leaf b' has start[b'] <=
                // b_start, so end[b'] <= start[b'] + S_max <= b_start + S_max.
                // If that cannot exceed the running max we have, the running
                // max is settled.
                if (b_start + s_max <= prev_running_max_end)
                {
                    break;
                }
            }
        }
        const int leaf_mask_base =
            kFixedShiftLeafScreenedSpecialized ||
                    candidate_leaf_reach_masks != NULL
                ? candidate_idx * candidate_leaf_cluster_stride
                : 0;

        const int deduped_cluster_j_start =
            IntMax(cluster_j_start, prev_running_max_end);
        for (int cluster_j = deduped_cluster_j_start; cluster_j < cluster_j_end;
             cluster_j += 1)
        {
            unsigned int precomputed_i_mask = 0u;
            if constexpr (kFixedShiftLeafScreenedSpecialized)
            {
                if (sublane == 0)
                {
                    precomputed_i_mask = candidate_leaf_reach_masks
                        [leaf_mask_base + (cluster_j - cluster_j_start)];
                }
                precomputed_i_mask =
                    deviceShfl(subgroup_mask, precomputed_i_mask,
                               subgroup * kSubgroupSize, warpSize) &
                    active_i_lane_mask;
                if (precomputed_i_mask == 0u)
                {
                    continue;
                }
            }
            else if (candidate_leaf_reach_masks != NULL)
            {
                if (sublane == 0)
                {
                    precomputed_i_mask = candidate_leaf_reach_masks
                        [leaf_mask_base + (cluster_j - cluster_j_start)];
                }
                precomputed_i_mask =
                    deviceShfl(subgroup_mask, precomputed_i_mask,
                               subgroup * kSubgroupSize, warpSize) &
                    active_i_lane_mask;
                if (precomputed_i_mask == 0u)
                {
                    continue;
                }
            }
            unsigned int valid_mask_j = 0u;
            unsigned int local_mask_j = 0u;
            VECTOR center_j = {0.0f, 0.0f, 0.0f};
            VECTOR extent_j = {0.0f, 0.0f, 0.0f};
            int super_j = 0;
            uint64_t signature_j = 0ull;
            if (sublane == 0)
            {
                valid_mask_j = cluster_valid_masks[cluster_j];
                local_mask_j = cluster_local_masks[cluster_j];
                if (valid_mask_j != 0u)
                {
                    super_j = cluster_to_supercluster[cluster_j];
                    if (!(Clustered_Valid_Lanes_Are_All_Local(valid_mask_j,
                                                              local_mask_j) &&
                          super_j < super_i))
                    {
                        center_j = cluster_centers[cluster_j];
                        if constexpr (!kFixedLightSlim)
                        {
                            extent_j = cluster_extents[cluster_j];
                        }
                    }
                    if (has_molecule_metadata)
                    {
                        signature_j = cluster_molecule_signatures[cluster_j];
                    }
                }
            }
            const int subgroup_leader = subgroup * kSubgroupSize;
            valid_mask_j = deviceShfl(subgroup_mask, valid_mask_j,
                                      subgroup_leader, warpSize);
            local_mask_j = deviceShfl(subgroup_mask, local_mask_j,
                                      subgroup_leader, warpSize);
            super_j =
                deviceShfl(subgroup_mask, super_j, subgroup_leader, warpSize);
            center_j.x = deviceShfl(subgroup_mask, center_j.x, subgroup_leader,
                                    warpSize);
            center_j.y = deviceShfl(subgroup_mask, center_j.y, subgroup_leader,
                                    warpSize);
            center_j.z = deviceShfl(subgroup_mask, center_j.z, subgroup_leader,
                                    warpSize);
            if constexpr (!kFixedLightSlim)
            {
                extent_j.x = deviceShfl(subgroup_mask, extent_j.x,
                                        subgroup_leader, warpSize);
                extent_j.y = deviceShfl(subgroup_mask, extent_j.y,
                                        subgroup_leader, warpSize);
                extent_j.z = deviceShfl(subgroup_mask, extent_j.z,
                                        subgroup_leader, warpSize);
            }
            {
                unsigned int sig_lo = static_cast<unsigned int>(signature_j);
                unsigned int sig_hi =
                    static_cast<unsigned int>(signature_j >> 32);
                sig_lo = deviceShfl(subgroup_mask, sig_lo, subgroup_leader,
                                    warpSize);
                sig_hi = deviceShfl(subgroup_mask, sig_hi, subgroup_leader,
                                    warpSize);
                signature_j = (static_cast<uint64_t>(sig_hi) << 32) |
                              static_cast<uint64_t>(sig_lo);
            }

            if (valid_mask_j == 0u)
            {
                continue;
            }
            if (Clustered_Valid_Lanes_Are_All_Local(valid_mask_j,
                                                    local_mask_j) &&
                super_j < super_i)
            {
                continue;
            }
            if (sublane == 0)
            {
                shared_j_signature[warp_id][subgroup] = signature_j;
            }
            int pair_shift_id = -1;
            bool exclusion_candidate = false;
            if (sublane < active_cluster_count)
            {
                const int i_local = sublane;
                if constexpr (kFixedShiftLeafScreenedSpecialized)
                {
                    if ((precomputed_i_mask &
                         (1u << static_cast<unsigned int>(i_local))) != 0u)
                    {
                        exclusion_candidate =
                            !has_molecule_metadata ||
                            (shared_i_signatures[warp_id][i_local] &
                             signature_j) != 0ull;
                    }
                }
                else if (candidate_leaf_reach_masks != NULL)
                {
                    if ((precomputed_i_mask &
                         (1u << static_cast<unsigned int>(i_local))) != 0u)
                    {
                        pair_shift_id = fixed_shift_id;
                        exclusion_candidate =
                            !has_molecule_metadata ||
                            (shared_i_signatures[warp_id][i_local] &
                             signature_j) != 0ull;
                    }
                }
                else
                {
                    const unsigned int local_mask_i =
                        shared_i_local_masks[warp_id][i_local];
                    if (local_mask_i != 0u)
                    {
                        const VECTOR center_i = {
                            shared_i_center_x[warp_id][i_local],
                            shared_i_center_y[warp_id][i_local],
                            shared_i_center_z[warp_id][i_local]};
                        const VECTOR extent_i = {
                            shared_i_extent_x[warp_id][i_local],
                            shared_i_extent_y[warp_id][i_local],
                            shared_i_extent_z[warp_id][i_local]};
                        pair_shift_id = fixed_shift_id;
                        if (cluster_j >= cluster_i_start &&
                            cluster_j < cluster_i_end &&
                            (cluster_i_start + i_local) > cluster_j &&
                            Clustered_Valid_Lanes_Are_All_Local(valid_mask_j,
                                                                local_mask_j))
                        {
                            pair_shift_id = -1;
                        }
                        else if (pair_shift_id >= 0 &&
                                 !Cluster_Aabb_Overlaps_Shifted(
                                     center_i, extent_i, center_j, extent_j,
                                     cutoff,
                                     fixed_shift_vec))
                        {
                            pair_shift_id = -1;
                        }
                        exclusion_candidate =
                            pair_shift_id >= 0 &&
                            (!has_molecule_metadata ||
                             (shared_i_signatures[warp_id][i_local] &
                              signature_j) != 0ull);
                    }
                }
            }

            const unsigned int active_pair_lane_mask =
                kFixedShiftLeafScreenedSpecialized ||
                        candidate_leaf_reach_masks != NULL
                    ? (precomputed_i_mask << (subgroup * kSubgroupSize))
                    : deviceBallot(
                          subgroup_mask,
                          sublane < active_cluster_count && pair_shift_id >= 0);
            if ((active_pair_lane_mask & subgroup_mask) == 0u)
            {
                continue;
            }
            const unsigned int exclusion_candidate_lane_mask =
                deviceBallot(subgroup_mask, sublane < active_cluster_count &&
                                                exclusion_candidate);

            const bool need_j_cached_atoms =
                (exclusion_candidate_lane_mask & subgroup_mask) != 0u ||
                count_record_stream_source_rows;
            if (need_j_cached_atoms)
            {
                if (sublane < kClusteredClusterSize)
                {
                    if ((valid_mask_j & (1u << sublane)) != 0u)
                    {
                        const int sorted_atom_j =
                            cluster_offsets[cluster_j] + sublane;
                        shared_j_atom_ids[warp_id][subgroup][sublane] =
                            permutation[sorted_atom_j];
                        shared_j_molecule_ids[warp_id][subgroup][sublane] =
                            has_molecule_metadata
                                ? cluster_molecule_ids
                                      [cluster_j * kClusteredClusterSize +
                                       sublane]
                                : -1;
                    }
                    else
                    {
                        shared_j_atom_ids[warp_id][subgroup][sublane] = -1;
                        shared_j_molecule_ids[warp_id][subgroup][sublane] = -1;
                    }
                }
                deviceSyncWarp(subgroup_mask);
            }

            // The pair-lane mask uses subgroup-local bit positions (0..7) for
            // the i_local group reduction below; shift active_pair_lane_mask
            // down.
            unsigned int remaining_lane_mask =
                (active_pair_lane_mask >> (subgroup * kSubgroupSize)) & 0xFFu;
            while (remaining_lane_mask != 0u)
            {
                const int leader_sublane =
                    __ffs(static_cast<int>(remaining_lane_mask)) - 1;
                const int group_shift_id =
                    kFixedShiftLeafScreenedSpecialized
                        ? fixed_shift_id
                        : deviceShfl(subgroup_mask, pair_shift_id,
                                     subgroup * kSubgroupSize + leader_sublane,
                                     warpSize);
                const unsigned int group_lane_mask_local =
                    kFixedShiftLeafScreenedSpecialized
                        ? remaining_lane_mask
                        : ((deviceBallot(subgroup_mask,
                                         sublane < active_cluster_count &&
                                             pair_shift_id == group_shift_id) >>
                            (subgroup * kSubgroupSize)) &
                           0xFFu);
                const unsigned int group_record_imask =
                    group_lane_mask_local & active_i_lane_mask;
                if constexpr (kFixedShiftLeafScreenedSpecialized)
                {
                    remaining_lane_mask = 0u;
                }
                else
                {
                    remaining_lane_mask &= ~group_lane_mask_local;
                }
                constexpr bool kSpecializedParallelGroupAccum =
                    kParallelAccum && kFixedShiftLeafScreenedSpecialized;
                const bool use_parallel_group_accum =
                    kSpecializedParallelGroupAccum ||
                    (kParallelAccum && candidate_leaf_reach_masks != NULL);
                const int source_shift_id = fixed_shift_id;
                VECTOR source_pair_shift = {0.0f, 0.0f, 0.0f};
                if constexpr (kFixedLightSassOpt)
                {
                    source_pair_shift =
                        Shift_Vector_From_Id(source_shift_id, cell);
                }
                unsigned int parallel_split_local_imask = 0u;
                int parallel_source_rows_for_group = 0;
                if (use_parallel_group_accum &&
                    count_record_stream_source_rows && group_record_imask != 0u)
                {
                    bool split_has_source_row = false;
                    if (sublane < kClusteredWarpSplitCount)
                    {
                        unsigned int split_local_imask = 0u;
                        if constexpr (kFixedLightSassOpt)
                        {
                            split_local_imask =
                                Prune_Gmxpacked_Record_Stream_Source_Imask_With_Shift(
                                    sublane, group_record_imask, valid_mask_j,
                                    source_pair_shift, cell, rcell, crd,
                                    shared_i_center_x[warp_id],
                                    shared_i_center_y[warp_id],
                                    shared_i_center_z[warp_id], center_j,
                                    shared_i_atom_ids[warp_id],
                                    shared_j_atom_ids[warp_id][subgroup],
                                    shared_i_local_masks[warp_id],
                                    record_stream_cutoff_sq);
                        }
                        else
                        {
                            split_local_imask =
                                Prune_Gmxpacked_Record_Stream_Source_Imask(
                                    sublane, group_record_imask, valid_mask_j,
                                    source_shift_id, cell, rcell, crd,
                                    shared_i_center_x[warp_id],
                                    shared_i_center_y[warp_id],
                                    shared_i_center_z[warp_id], center_j,
                                    shared_i_atom_ids[warp_id],
                                    shared_j_atom_ids[warp_id][subgroup],
                                    shared_i_local_masks[warp_id],
                                    record_stream_cutoff_sq);
                        }
                        parallel_split_local_imask = split_local_imask;
                        split_has_source_row =
                            Clustered_Split_Has_Atoms(valid_mask_j, sublane) &&
                            split_local_imask != 0u;
                    }
                    const unsigned int source_row_lane_mask =
                        (deviceBallot(subgroup_mask, split_has_source_row) >>
                         (subgroup * kSubgroupSize)) &
                        0xFFu;
                    parallel_source_rows_for_group =
                        __popc(source_row_lane_mask);
                }
                unsigned long long parallel_lane_exclusion_mask = 0ull;
                int parallel_exclusion_count_for_group = 0;
                if (use_parallel_group_accum && need_j_cached_atoms)
                {
                    const unsigned int group_exclusion_imask =
                        (exclusion_candidate_lane_mask >>
                         (subgroup * kSubgroupSize)) &
                        group_record_imask;
                    bool has_lane_exclusion = false;
                    if (sublane < active_cluster_count &&
                        (group_exclusion_imask &
                         (1u << static_cast<unsigned int>(sublane))) != 0u)
                    {
                        const unsigned int local_mask_i =
                            shared_i_local_masks[warp_id][sublane];
                        const unsigned long long exclusion_mask =
                            Build_Exclusion_Mask_From_Cached_Atoms(
                                shared_i_atom_ids[warp_id][sublane],
                                shared_j_atom_ids[warp_id][subgroup],
                                shared_i_molecule_ids[warp_id][sublane],
                                shared_j_molecule_ids[warp_id][subgroup],
                                shared_i_signatures[warp_id][sublane],
                                shared_j_signature[warp_id][subgroup],
                                has_molecule_metadata, local_mask_i,
                                valid_mask_j, cluster_size, local_atom_numbers,
                                excluded_list_start, excluded_list,
                                excluded_numbers);
                        parallel_lane_exclusion_mask = exclusion_mask;
                        has_lane_exclusion = exclusion_mask != 0ull;
                    }
                    const unsigned int exclusion_lane_mask =
                        (deviceBallot(subgroup_mask, has_lane_exclusion) >>
                         (subgroup * kSubgroupSize)) &
                        0xFFu;
                    parallel_exclusion_count_for_group =
                        __popc(exclusion_lane_mask);
                }
                if (use_parallel_group_accum)
                {
                    if (count_record_stream_source_rows &&
                        group_record_imask != 0u &&
                        sublane < kClusteredWarpSplitCount)
                    {
                        shared_parallel_split_imasks
                            [warp_id][subgroup][sublane] =
                                parallel_split_local_imask;
                    }
                    if (need_j_cached_atoms &&
                        sublane < kClusteredMaxSuperClusterClusters)
                    {
                        shared_parallel_exclusion_masks
                            [warp_id][subgroup][sublane] =
                                parallel_lane_exclusion_mask;
                    }
                    deviceSyncWarp(subgroup_mask);
                }
                const bool use_parallel_fragment_emit =
                    kLightFragmentOnly
                        ? true
                        : (kParallelFragmentEmit && use_parallel_group_accum &&
                           emit_count_light_source_fragments &&
                           group_record_imask != 0u);
                if (use_parallel_fragment_emit)
                {
                    unsigned int split_local_imask = 0u;
                    bool split_has_source_row = false;
                    if (sublane < kClusteredWarpSplitCount)
                    {
                        split_local_imask =
                            shared_parallel_split_imasks[warp_id][subgroup]
                                                        [sublane];
                        split_has_source_row =
                            Clustered_Split_Has_Atoms(valid_mask_j, sublane) &&
                            split_local_imask != 0u;
                    }
                    const unsigned int source_row_lane_mask =
                        (deviceBallot(subgroup_mask, split_has_source_row) >>
                         (subgroup * kSubgroupSize)) &
                        0xFFu;
                    const int source_rows_for_group =
                        __popc(source_row_lane_mask);
                    int fragment_base = -1;
                    if (sublane == leader_sublane && source_rows_for_group > 0)
                    {
                        fragment_base = atomicAdd(count_source_fragment_cursor,
                                                  source_rows_for_group);
                    }
                    fragment_base = deviceShfl(
                        subgroup_mask, fragment_base,
                        subgroup * kSubgroupSize + leader_sublane, warpSize);
                    if (split_has_source_row)
                    {
                        const unsigned int lower_source_rows =
                            source_row_lane_mask &
                            ((1u << static_cast<unsigned int>(sublane)) - 1u);
                        const int fragment_idx =
                            fragment_base + __popc(lower_source_rows);
                        if (fragment_idx < count_source_fragment_capacity)
                        {
                            CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT*
                                fragment =
                                    count_light_source_fragments + fragment_idx;
                            fragment->sci_id = candidate_sci;
                            fragment->shift_id = source_shift_id;
                            fragment->supercluster_id = super_i;
                            fragment->cluster_j = cluster_j;
                            fragment->split_id = sublane;
                            fragment->imask = split_local_imask;
                            fragment->valid_mask_j = valid_mask_j;
                            fragment->local_mask_j = local_mask_j;
                            fragment->source_order = fragment_idx;
#pragma unroll
                            for (int i_local = 0;
                                 i_local < kClusteredMaxSuperClusterClusters;
                                 i_local += 1)
                            {
                                fragment->exclusion_masks[i_local] =
                                    shared_parallel_exclusion_masks[warp_id]
                                                                   [subgroup]
                                                                   [i_local];
                            }
                        }
                        else
                        {
                            atomicAdd(count_source_fragment_overflow_rows, 1);
                        }
                    }
                    if (sublane == leader_sublane)
                    {
                        const int output_shift_idx = 0;
                        if (source_rows_for_group > 0)
                        {
                            if constexpr (kFixedLightSassOpt)
                            {
                                local_record_stream_source_count +=
                                    source_rows_for_group;
                            }
                            else if (use_candidate_record_stream_source_count)
                            {
                                atomicAdd(&shared_record_stream_source_counts
                                              [warp_id],
                                          source_rows_for_group);
                            }
                            else if (record_stream_source_rows != NULL)
                            {
                                atomicAdd(record_stream_source_rows,
                                          source_rows_for_group);
                            }
                        }
                        if constexpr (kFixedLightSassOpt)
                        {
                            local_shift_record_count += 1;
                        }
                        else
                        {
                            atomicAdd(
                                &shared_shift_record_counts[warp_id]
                                                           [output_shift_idx],
                                1);
                        }
                        if (need_j_cached_atoms &&
                            parallel_exclusion_count_for_group > 0)
                        {
                            if constexpr (kFixedLightSassOpt)
                            {
                                local_shift_exclusion_count +=
                                    parallel_exclusion_count_for_group;
                            }
                            else
                            {
                                atomicAdd(&shared_shift_exclusion_counts
                                              [warp_id][output_shift_idx],
                                          parallel_exclusion_count_for_group);
                            }
                        }
                    }
                }
                if (sublane == leader_sublane && !use_parallel_fragment_emit)
                {
                    const int output_shift_idx = 0;
                    if (emit_count_light_source_fragments)
                    {
                        unsigned long long
                            exclusion_masks[kClusteredMaxSuperClusterClusters] =
                                {};
                        int exclusion_count_for_group = 0;
                        if (need_j_cached_atoms)
                        {
                            if (use_parallel_group_accum)
                            {
                                exclusion_count_for_group =
                                    parallel_exclusion_count_for_group;
#pragma unroll
                                for (int i_local = 0;
                                     i_local <
                                     kClusteredMaxSuperClusterClusters;
                                     i_local += 1)
                                {
                                    exclusion_masks[i_local] =
                                        shared_parallel_exclusion_masks
                                            [warp_id][subgroup][i_local];
                                }
                            }
                            else
                            {
                                const unsigned int group_exclusion_imask =
                                    (exclusion_candidate_lane_mask >>
                                     (subgroup * kSubgroupSize)) &
                                    group_record_imask;
                                unsigned int remaining_i =
                                    group_exclusion_imask;
                                while (remaining_i != 0u)
                                {
                                    const int i_local =
                                        __ffs(static_cast<int>(remaining_i)) -
                                        1;
                                    remaining_i &= (remaining_i - 1u);
                                    const unsigned int local_mask_i =
                                        shared_i_local_masks[warp_id][i_local];
                                    const unsigned long long exclusion_mask =
                                        Build_Exclusion_Mask_From_Cached_Atoms(
                                            shared_i_atom_ids[warp_id][i_local],
                                            shared_j_atom_ids[warp_id]
                                                             [subgroup],
                                            shared_i_molecule_ids[warp_id]
                                                                 [i_local],
                                            shared_j_molecule_ids[warp_id]
                                                                 [subgroup],
                                            shared_i_signatures[warp_id]
                                                               [i_local],
                                            shared_j_signature[warp_id]
                                                              [subgroup],
                                            has_molecule_metadata, local_mask_i,
                                            valid_mask_j, cluster_size,
                                            local_atom_numbers,
                                            excluded_list_start, excluded_list,
                                            excluded_numbers);
                                    exclusion_masks[i_local] = exclusion_mask;
                                    exclusion_count_for_group +=
                                        exclusion_mask != 0ull ? 1 : 0;
                                }
                            }
                        }
                        if (group_record_imask != 0u)
                        {
                            int source_rows_for_group = 0;
#pragma unroll
                            for (int split = 0;
                                 split < kClusteredWarpSplitCount; split += 1)
                            {
                                unsigned int split_local_imask = 0u;
                                if (use_parallel_group_accum)
                                {
                                    split_local_imask =
                                        shared_parallel_split_imasks[warp_id]
                                                                    [subgroup]
                                                                    [split];
                                }
                                else
                                {
                                    if constexpr (kFixedLightSassOpt)
                                    {
                                        split_local_imask =
                                            Prune_Gmxpacked_Record_Stream_Source_Imask_With_Shift(
                                                split, group_record_imask,
                                                valid_mask_j, source_pair_shift,
                                                cell, rcell, crd,
                                                shared_i_center_x[warp_id],
                                                shared_i_center_y[warp_id],
                                                shared_i_center_z[warp_id],
                                                center_j,
                                                shared_i_atom_ids[warp_id],
                                                shared_j_atom_ids[warp_id]
                                                                 [subgroup],
                                                shared_i_local_masks[warp_id],
                                                record_stream_cutoff_sq);
                                    }
                                    else
                                    {
                                        split_local_imask =
                                            Prune_Gmxpacked_Record_Stream_Source_Imask(
                                                split, group_record_imask,
                                                valid_mask_j, source_shift_id,
                                                cell, rcell, crd,
                                                shared_i_center_x[warp_id],
                                                shared_i_center_y[warp_id],
                                                shared_i_center_z[warp_id],
                                                center_j,
                                                shared_i_atom_ids[warp_id],
                                                shared_j_atom_ids[warp_id]
                                                                 [subgroup],
                                                shared_i_local_masks[warp_id],
                                                record_stream_cutoff_sq);
                                    }
                                }
                                if (Clustered_Split_Has_Atoms(valid_mask_j,
                                                              split) &&
                                    split_local_imask != 0u)
                                {
                                    source_rows_for_group += 1;
                                    const int fragment_idx = atomicAdd(
                                        count_source_fragment_cursor, 1);
                                    if (fragment_idx <
                                        count_source_fragment_capacity)
                                    {
                                        CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT
                                        fragment = {};
                                        fragment.sci_id = candidate_sci;
                                        fragment.shift_id = source_shift_id;
                                        fragment.supercluster_id = super_i;
                                        fragment.cluster_j = cluster_j;
                                        fragment.split_id = split;
                                        fragment.imask = split_local_imask;
                                        fragment.valid_mask_j = valid_mask_j;
                                        fragment.local_mask_j = local_mask_j;
                                        fragment.source_order = fragment_idx;
#pragma unroll
                                        for (int i_local = 0;
                                             i_local <
                                             kClusteredMaxSuperClusterClusters;
                                             i_local += 1)
                                        {
                                            fragment.exclusion_masks[i_local] =
                                                exclusion_masks[i_local];
                                        }
                                        count_light_source_fragments
                                            [fragment_idx] = fragment;
                                    }
                                    else
                                    {
                                        atomicAdd(
                                            count_source_fragment_overflow_rows,
                                            1);
                                    }
                                }
                            }
                            if (source_rows_for_group > 0)
                            {
                                if constexpr (kFixedLightSassOpt)
                                {
                                    local_record_stream_source_count +=
                                        source_rows_for_group;
                                }
                                else if (
                                    use_candidate_record_stream_source_count)
                                {
                                    atomicAdd(
                                        &shared_record_stream_source_counts
                                            [warp_id],
                                        source_rows_for_group);
                                }
                                else if (record_stream_source_rows != NULL)
                                {
                                    atomicAdd(record_stream_source_rows,
                                              source_rows_for_group);
                                }
                            }
                        }
                        if constexpr (kFixedLightSassOpt)
                        {
                            local_shift_record_count += 1;
                        }
                        else
                        {
                            atomicAdd(
                                &shared_shift_record_counts[warp_id]
                                                           [output_shift_idx],
                                1);
                        }
                        if (need_j_cached_atoms)
                        {
                            if constexpr (kFixedLightSassOpt)
                            {
                                local_shift_exclusion_count +=
                                    exclusion_count_for_group;
                            }
                            else
                            {
                                atomicAdd(&shared_shift_exclusion_counts
                                              [warp_id][output_shift_idx],
                                          exclusion_count_for_group);
                            }
                        }
                    }
                    else if (use_parallel_group_accum)
                    {
                        if (parallel_source_rows_for_group > 0)
                        {
                            if constexpr (kFixedLightSassOpt)
                            {
                                local_record_stream_source_count +=
                                    parallel_source_rows_for_group;
                            }
                            else if (use_candidate_record_stream_source_count)
                            {
                                atomicAdd(&shared_record_stream_source_counts
                                              [warp_id],
                                          parallel_source_rows_for_group);
                            }
                            else if (record_stream_source_rows != NULL)
                            {
                                atomicAdd(record_stream_source_rows,
                                          parallel_source_rows_for_group);
                            }
                        }
                        if constexpr (kFixedLightSassOpt)
                        {
                            local_shift_record_count += 1;
                        }
                        else
                        {
                            atomicAdd(
                                &shared_shift_record_counts[warp_id]
                                                           [output_shift_idx],
                                1);
                        }
                        if (need_j_cached_atoms &&
                            parallel_exclusion_count_for_group > 0)
                        {
                            if constexpr (kFixedLightSassOpt)
                            {
                                local_shift_exclusion_count +=
                                    parallel_exclusion_count_for_group;
                            }
                            else
                            {
                                atomicAdd(&shared_shift_exclusion_counts
                                              [warp_id][output_shift_idx],
                                          parallel_exclusion_count_for_group);
                            }
                        }
                    }
                    else
                    {
                        if (count_record_stream_source_rows &&
                            group_record_imask != 0u)
                        {
                            int source_rows_for_group = 0;
#pragma unroll
                            for (int split = 0;
                                 split < kClusteredWarpSplitCount; split += 1)
                            {
                                unsigned int split_local_imask = 0u;
                                if constexpr (kFixedLightSassOpt)
                                {
                                    split_local_imask =
                                        Prune_Gmxpacked_Record_Stream_Source_Imask_With_Shift(
                                            split, group_record_imask,
                                            valid_mask_j, source_pair_shift,
                                            cell, rcell, crd,
                                            shared_i_center_x[warp_id],
                                            shared_i_center_y[warp_id],
                                            shared_i_center_z[warp_id],
                                            center_j,
                                            shared_i_atom_ids[warp_id],
                                            shared_j_atom_ids[warp_id]
                                                             [subgroup],
                                            shared_i_local_masks[warp_id],
                                            record_stream_cutoff_sq);
                                }
                                else
                                {
                                    split_local_imask =
                                        Prune_Gmxpacked_Record_Stream_Source_Imask(
                                            split, group_record_imask,
                                            valid_mask_j,
                                            fixed_shift_id,
                                            cell, rcell, crd,
                                            shared_i_center_x[warp_id],
                                            shared_i_center_y[warp_id],
                                            shared_i_center_z[warp_id],
                                            center_j,
                                            shared_i_atom_ids[warp_id],
                                            shared_j_atom_ids[warp_id]
                                                             [subgroup],
                                            shared_i_local_masks[warp_id],
                                            record_stream_cutoff_sq);
                                }
                                if (Clustered_Split_Has_Atoms(valid_mask_j,
                                                              split) &&
                                    split_local_imask != 0u)
                                {
                                    source_rows_for_group += 1;
                                }
                            }
                            if (source_rows_for_group > 0)
                            {
                                if constexpr (kFixedLightSassOpt)
                                {
                                    local_record_stream_source_count +=
                                        source_rows_for_group;
                                }
                                else if (
                                    use_candidate_record_stream_source_count)
                                {
                                    atomicAdd(
                                        &shared_record_stream_source_counts
                                            [warp_id],
                                        source_rows_for_group);
                                }
                                else if (record_stream_source_rows != NULL)
                                {
                                    atomicAdd(record_stream_source_rows,
                                              source_rows_for_group);
                                }
                            }
                        }
                        if constexpr (kFixedLightSassOpt)
                        {
                            local_shift_record_count += 1;
                        }
                        else
                        {
                            atomicAdd(
                                &shared_shift_record_counts[warp_id]
                                                           [output_shift_idx],
                                1);
                        }
                        if (need_j_cached_atoms)
                        {
                            const unsigned int group_exclusion_imask =
                                (exclusion_candidate_lane_mask >>
                                 (subgroup * kSubgroupSize)) &
                                group_record_imask;
                            int exclusion_count_for_group = 0;
                            unsigned int remaining_i = group_exclusion_imask;
                            while (remaining_i != 0u)
                            {
                                const int i_local =
                                    __ffs(static_cast<int>(remaining_i)) - 1;
                                remaining_i &= (remaining_i - 1u);
                                const unsigned int local_mask_i =
                                    shared_i_local_masks[warp_id][i_local];
                                const unsigned long long exclusion_mask =
                                    Build_Exclusion_Mask_From_Cached_Atoms(
                                        shared_i_atom_ids[warp_id][i_local],
                                        shared_j_atom_ids[warp_id][subgroup],
                                        shared_i_molecule_ids[warp_id][i_local],
                                        shared_j_molecule_ids[warp_id]
                                                             [subgroup],
                                        shared_i_signatures[warp_id][i_local],
                                        shared_j_signature[warp_id][subgroup],
                                        has_molecule_metadata, local_mask_i,
                                        valid_mask_j, cluster_size,
                                        local_atom_numbers, excluded_list_start,
                                        excluded_list, excluded_numbers);
                                exclusion_count_for_group +=
                                    exclusion_mask != 0ull ? 1 : 0;
                            }
                            if constexpr (kFixedLightSassOpt)
                            {
                                local_shift_exclusion_count +=
                                    exclusion_count_for_group;
                            }
                            else
                            {
                                atomicAdd(&shared_shift_exclusion_counts
                                              [warp_id][output_shift_idx],
                                          exclusion_count_for_group);
                            }
                        }
                    }
                }
                deviceSyncWarp(subgroup_mask);
            }
        }
    }

    if constexpr (kFixedLightSassOpt)
    {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            local_record_stream_source_count += deviceShflDown(
                FULL_MASK, local_record_stream_source_count, offset, warpSize);
            local_shift_record_count += deviceShflDown(
                FULL_MASK, local_shift_record_count, offset, warpSize);
            local_shift_exclusion_count += deviceShflDown(
                FULL_MASK, local_shift_exclusion_count, offset, warpSize);
        }
        if (lane_id == 0)
        {
            shared_record_stream_source_counts[warp_id] =
                local_record_stream_source_count;
            shared_shift_record_counts[warp_id][0] = local_shift_record_count;
            shared_shift_exclusion_counts[warp_id][0] =
                local_shift_exclusion_count;
        }
    }
    __syncwarp();
    if (lane_id == 0)
    {
        const int candidate_record_stream_source_count =
            shared_record_stream_source_counts[warp_id];
        if (record_stream_source_counts_by_candidate != NULL)
        {
            record_stream_source_counts_by_candidate[candidate_sci] =
                candidate_record_stream_source_count;
        }
        if (use_candidate_record_stream_source_count &&
            record_stream_source_rows != NULL &&
            candidate_record_stream_source_count > 0)
        {
            atomicAdd(record_stream_source_rows,
                      candidate_record_stream_source_count);
        }
    }
    if (lane_id == 0)
    {
        const int record_count = shared_shift_record_counts[warp_id][0];
        sci_shift_flags[sci_shift_base] = record_count > 0 ? 1 : 0;
        cjpacked_group_counts[sci_shift_base] =
            (record_count + kClusteredMaxJGroupSize - 1) /
            kClusteredMaxJGroupSize;
        exclusion_counts[sci_shift_base] =
            shared_shift_exclusion_counts[warp_id][0];
    }
}
