static __device__ __forceinline__ CLUSTERED_GMXPACKED_CJ
Make_Empty_Gmxpacked_CjPacked();

// Production record-source fill: one warp per candidate-SCI, split into
// 4x 8-lane subgroups. Mirrors the proven count
// subgroup kernel (Count_Nbnxm_Payload_From_Candidate_Leaves_Subgroup): leaf-
// strided parallelism, S_max-bounded dedup (reconstructs the baseline
// running-max exactly; candidate leaves are cluster-start-sorted per SCI via
// Hilbert SFC), subgroup masks instead of FULL_MASK, per-subgroup j-side
// scratch, shared-atomicAdd shift/source-row counters. DIFFERENCE from count:
// the fill writes source rows into the per-SCI reserved range [offset,end);
// with 4 subgroups writing concurrently the per-warp write cursor is bumped via
// shared-mem atomicAdd (bounded by write_end). Output is order-independent
// (source_order=write_idx, sorted downstream) so the multiset of rows must
// match baseline exactly (validated by the fill compare).
static __global__ void
Fill_Gmxpacked_Record_Stream_Sources_From_Candidate_Leaves_Subgroup(
    const int candidate_sci_numbers, const int cluster_size,
    const int local_atom_numbers, const float cutoff, const LTMatrix3 cell,
    const LTMatrix3 rcell, const int* permutation, const int* cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* super_cluster_offsets, const int* cluster_to_supercluster,
    const int* sci_supercluster_ids, const int* candidate_leaf_offsets,
    const int* candidate_leaf_ids,
    const int candidate_leaf_cluster_stride,
    const unsigned int* candidate_leaf_reach_masks,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, const uint64_t* cluster_molecule_signatures,
    const int* cluster_molecule_ids, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers,
    const float record_stream_cutoff,
    const VECTOR* crd, const int record_stream_source_capacity,
    const int* record_stream_source_offsets_by_candidate,
    CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* record_stream_sources,
    int* record_stream_fill_cursor, int* record_stream_overflow_rows,
    const int* active_candidate_sci_mask, const int max_leaf_cluster_span)
{
    constexpr int kSubgroupSize = kClusteredClusterSize;  // 8
    constexpr int kCountSubgroups =
        kClusteredBuilderWarpSize / kSubgroupSize;  // 4
    const int lane_id = threadIdx.x & (warpSize - 1);
    const int warp_id = threadIdx.x / warpSize;
    const int warps_per_block = blockDim.x / warpSize;
    const int subgroup = lane_id / kSubgroupSize;  // 0..3
    const int sublane =
        lane_id % kSubgroupSize;  // 0..7 == i_local / j-atom lane
    const unsigned int subgroup_mask = 0xFFu << (subgroup * kSubgroupSize);
    const int candidate_sci = blockIdx.x * warps_per_block + warp_id;
    if (candidate_sci >= candidate_sci_numbers ||
        record_stream_source_capacity <= 0 || record_stream_sources == NULL ||
        record_stream_fill_cursor == NULL ||
        record_stream_overflow_rows == NULL)
    {
        return;
    }
    if (active_candidate_sci_mask != NULL &&
        active_candidate_sci_mask[candidate_sci] == 0)
    {
        return;
    }

    const int sci_base = candidate_sci / kClusteredShiftCount;
    const int fixed_shift_id = candidate_sci % kClusteredShiftCount;
    const VECTOR fixed_shift_vec = Shift_Vector_From_Id(fixed_shift_id, cell);
    const int super_i = sci_supercluster_ids[sci_base];
    const int cluster_i_start = super_cluster_offsets[super_i];
    const int cluster_i_end = super_cluster_offsets[super_i + 1];
    constexpr int kWarpsPerBlock =
        kClusteredBuilderBlockSize / kClusteredBuilderWarpSize;
    // i-side data shared by all subgroups of a warp (read-only): per-warp.
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
    // j-side scratch + j-signature per-subgroup (each subgroup works a
    // different cluster_j concurrently).
    __shared__ int shared_j_atom_ids[kWarpsPerBlock][kCountSubgroups]
                                    [kClusteredClusterSize];
    __shared__ uint64_t shared_j_signature[kWarpsPerBlock][kCountSubgroups];
    __shared__ int shared_j_molecule_ids[kWarpsPerBlock][kCountSubgroups]
                                        [kClusteredClusterSize];
    // Write cursor / end / written_rows are per-warp (shared by 4 subgroups),
    // bumped via shared-mem atomicAdd so concurrent subgroup writers do not
    // race.
    __shared__ int shared_record_stream_source_write_cursor[kWarpsPerBlock];
    __shared__ int shared_record_stream_source_write_end[kWarpsPerBlock];
    __shared__ int shared_record_stream_source_written_rows[kWarpsPerBlock];
    const bool has_molecule_metadata =
        cluster_molecule_signatures != NULL && cluster_molecule_ids != NULL;
    const bool use_candidate_source_offsets =
        record_stream_source_offsets_by_candidate != NULL;
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
    if (lane_id == 0)
    {
        if (use_candidate_source_offsets)
        {
            shared_record_stream_source_write_cursor[warp_id] =
                record_stream_source_offsets_by_candidate[candidate_sci];
            shared_record_stream_source_write_end[warp_id] =
                record_stream_source_offsets_by_candidate[candidate_sci + 1];
        }
        else
        {
            shared_record_stream_source_write_cursor[warp_id] = 0;
            shared_record_stream_source_write_end[warp_id] = 0;
        }
        shared_record_stream_source_written_rows[warp_id] = 0;
    }
    __syncwarp();
    const int active_cluster_count = cluster_i_end - cluster_i_start;
    const unsigned int active_i_lane_mask =
        active_cluster_count > 0
            ? ((1u << static_cast<unsigned int>(active_cluster_count)) - 1u)
            : 0u;

    const int leaf_begin = candidate_leaf_offsets[candidate_sci];
    const int leaf_end = candidate_leaf_offsets[candidate_sci + 1];
    const int s_max = max_leaf_cluster_span;
    for (int candidate_idx = leaf_begin + subgroup; candidate_idx < leaf_end;
         candidate_idx += kCountSubgroups)
    {
        const int leaf_j = candidate_leaf_ids[candidate_idx];
        const int cluster_j_start = leaf_cluster_starts[leaf_j];
        const int cluster_j_end = leaf_cluster_ends[leaf_j];
        int prev_running_max_end = 0;
        for (int b = candidate_idx - 1; b >= leaf_begin; b -= 1)
        {
            const int b_leaf = candidate_leaf_ids[b];
            const int b_start = leaf_cluster_starts[b_leaf];
            const int b_end = leaf_cluster_ends[b_leaf];
            if (b_end > prev_running_max_end)
            {
                prev_running_max_end = b_end;
            }
            if (b_start + s_max <= prev_running_max_end)
            {
                break;
            }
        }
        const int leaf_mask_base =
            candidate_leaf_reach_masks != NULL
                ? candidate_idx * candidate_leaf_cluster_stride
                : 0;

        const int deduped_cluster_j_start =
            IntMax(cluster_j_start, prev_running_max_end);
        for (int cluster_j = deduped_cluster_j_start; cluster_j < cluster_j_end;
             cluster_j += 1)
        {
            unsigned int precomputed_i_mask = 0u;
            if (candidate_leaf_reach_masks != NULL)
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
                        extent_j = cluster_extents[cluster_j];
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
            extent_j.x = deviceShfl(subgroup_mask, extent_j.x, subgroup_leader,
                                    warpSize);
            extent_j.y = deviceShfl(subgroup_mask, extent_j.y, subgroup_leader,
                                    warpSize);
            extent_j.z = deviceShfl(subgroup_mask, extent_j.z, subgroup_leader,
                                    warpSize);
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
            if (sublane < active_cluster_count)
            {
                const int i_local = sublane;
                if (candidate_leaf_reach_masks != NULL)
                {
                    if ((precomputed_i_mask &
                         (1u << static_cast<unsigned int>(i_local))) != 0u)
                    {
                        pair_shift_id = fixed_shift_id;
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
                    }
                }
            }

            const unsigned int active_pair_lane_mask =
                candidate_leaf_reach_masks != NULL
                    ? (precomputed_i_mask << (subgroup * kSubgroupSize))
                    : deviceBallot(
                          subgroup_mask,
                          sublane < active_cluster_count && pair_shift_id >= 0);
            if ((active_pair_lane_mask & subgroup_mask) == 0u)
            {
                continue;
            }

            const bool need_j_cached_atoms = true;
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

            unsigned int remaining_lane_mask =
                (active_pair_lane_mask >> (subgroup * kSubgroupSize)) & 0xFFu;
            while (remaining_lane_mask != 0u)
            {
                const int leader_sublane =
                    __ffs(static_cast<int>(remaining_lane_mask)) - 1;
                const int group_shift_id = deviceShfl(
                    subgroup_mask, pair_shift_id,
                    subgroup * kSubgroupSize + leader_sublane, warpSize);
                const unsigned int group_lane_mask_local =
                    deviceBallot(subgroup_mask,
                                 sublane < active_cluster_count &&
                                     pair_shift_id == group_shift_id) >>
                        (subgroup * kSubgroupSize) &
                    0xFFu;
                const unsigned int group_record_imask =
                    group_lane_mask_local & active_i_lane_mask;
                remaining_lane_mask &= ~group_lane_mask_local;
                if (sublane == leader_sublane)
                {
                    unsigned long long
                        exclusion_masks[kClusteredMaxSuperClusterClusters] = {};
                    if (group_record_imask != 0u)
                    {
                        unsigned int remaining_i = group_record_imask;
                        while (remaining_i != 0u)
                        {
                            const int i_local =
                                __ffs(static_cast<int>(remaining_i)) - 1;
                            remaining_i &= (remaining_i - 1u);
                            const unsigned int local_mask_i =
                                shared_i_local_masks[warp_id][i_local];
                            exclusion_masks[i_local] =
                                Build_Exclusion_Mask_From_Cached_Atoms(
                                    shared_i_atom_ids[warp_id][i_local],
                                    shared_j_atom_ids[warp_id][subgroup],
                                    shared_i_molecule_ids[warp_id][i_local],
                                    shared_j_molecule_ids[warp_id][subgroup],
                                    shared_i_signatures[warp_id][i_local],
                                    shared_j_signature[warp_id][subgroup],
                                    has_molecule_metadata, local_mask_i,
                                    valid_mask_j, cluster_size,
                                    local_atom_numbers, excluded_list_start,
                                    excluded_list, excluded_numbers);
                        }
                    }

                    const int source_shift_id = fixed_shift_id;
#pragma unroll 1
                    for (int split = 0; split < kClusteredWarpSplitCount;
                         split += 1)
                    {
                        const unsigned int split_local_imask =
                            Prune_Gmxpacked_Record_Stream_Source_Imask(
                                split, group_record_imask, valid_mask_j,
                                source_shift_id, cell, rcell, crd,
                                shared_i_center_x[warp_id],
                                shared_i_center_y[warp_id],
                                shared_i_center_z[warp_id], center_j,
                                shared_i_atom_ids[warp_id],
                                shared_j_atom_ids[warp_id][subgroup],
                                shared_i_local_masks[warp_id],
                                record_stream_cutoff * record_stream_cutoff);
                        const bool split_has_atoms =
                            Clustered_Split_Has_Atoms(valid_mask_j, split);

                        int write_idx = -1;
                        if (split_has_atoms && split_local_imask != 0u)
                        {
                            if (use_candidate_source_offsets)
                            {
                                // Shared-mem atomicAdd across the 4 subgroups
                                // writing into the same SCI's reserved range;
                                // bounded by write_end. Order-independent =>
                                // atomic interleaving is fine.
                                const int candidate_write_idx = atomicAdd(
                                    &shared_record_stream_source_write_cursor
                                        [warp_id],
                                    1);
                                const int candidate_write_end =
                                    shared_record_stream_source_write_end
                                        [warp_id];
                                if (candidate_write_idx < candidate_write_end)
                                {
                                    write_idx = candidate_write_idx;
                                }
                                else
                                {
                                    atomicAdd(record_stream_overflow_rows, 1);
                                }
                            }
                            else
                            {
                                write_idx =
                                    atomicAdd(record_stream_fill_cursor, 1);
                            }
                        }
                        if (write_idx >= 0 &&
                            write_idx < record_stream_source_capacity)
                        {
                            CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE source =
                                {};
                            source.sci_id = candidate_sci;
                            source.shift_id = source_shift_id;
                            source.supercluster_id = super_i;
                            source.cluster_j = cluster_j;
                            source.split_id = split;
                            source.imask = split_local_imask;
                            source.valid_mask_j = valid_mask_j;
                            source.local_mask_j = local_mask_j;
                            source.source_order = write_idx;
                            Fill_Gmxpacked_Record_Stream_Source_Pair_Words(
                                &source, cluster_i_start, cluster_j,
                                source_shift_id, split, valid_mask_j,
                                local_mask_j, split_local_imask,
                                shared_i_local_masks[warp_id], exclusion_masks);
                            record_stream_sources[write_idx] = source;
                            if (use_candidate_source_offsets)
                            {
                                atomicAdd(
                                    &shared_record_stream_source_written_rows
                                        [warp_id],
                                    1);
                            }
                        }
                        else if (write_idx >= 0)
                        {
                            atomicAdd(record_stream_overflow_rows, 1);
                        }
                    }
                }
                deviceSyncWarp(subgroup_mask);
            }
        }
    }

    __syncwarp();
    if (use_candidate_source_offsets && lane_id == 0 &&
        record_stream_fill_cursor != NULL &&
        shared_record_stream_source_written_rows[warp_id] > 0)
    {
        atomicAdd(record_stream_fill_cursor,
                  shared_record_stream_source_written_rows[warp_id]);
    }
}

static __global__ void
Materialize_Gmxpacked_Record_Stream_Sources_From_Light_Count_Fragments(
    const int fragment_numbers, const int candidate_sci_numbers,
    const int record_stream_source_capacity,
    const CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT* count_source_fragments,
    const int* record_stream_source_offsets_by_candidate,
    int* record_stream_source_write_cursors_by_candidate,
    const int* super_cluster_offsets, const unsigned int* cluster_local_masks,
    CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* record_stream_sources,
    int* record_stream_fill_cursor, int* record_stream_overflow_rows)
{
    if (fragment_numbers <= 0 || candidate_sci_numbers <= 0 ||
        record_stream_source_capacity <= 0 || count_source_fragments == NULL ||
        record_stream_source_offsets_by_candidate == NULL ||
        record_stream_source_write_cursors_by_candidate == NULL ||
        super_cluster_offsets == NULL || cluster_local_masks == NULL ||
        record_stream_sources == NULL || record_stream_fill_cursor == NULL ||
        record_stream_overflow_rows == NULL)
    {
        return;
    }

    const int lane_id = threadIdx.x & (warpSize - 1);
    const int warp_id = threadIdx.x / warpSize;
    const int warps_per_block = blockDim.x / warpSize;
    const int fragment_idx = blockIdx.x * warps_per_block + warp_id;
    if (fragment_idx >= fragment_numbers)
    {
        return;
    }

    const CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT fragment =
        count_source_fragments[fragment_idx];
    int write_idx = -1;
    int write_valid = 0;
    if (lane_id == 0)
    {
        const int candidate_sci = fragment.sci_id;
        if (candidate_sci < 0 || candidate_sci >= candidate_sci_numbers)
        {
            atomicAdd(record_stream_overflow_rows, 1);
        }
        else
        {
            write_idx = atomicAdd(
                record_stream_source_write_cursors_by_candidate + candidate_sci,
                1);
            const int write_end =
                record_stream_source_offsets_by_candidate[candidate_sci + 1];
            if (write_idx >= 0 && write_idx < write_end &&
                write_idx < record_stream_source_capacity)
            {
                CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* source =
                    record_stream_sources + write_idx;
                source->sci_id = candidate_sci;
                source->shift_id = fragment.shift_id;
                source->supercluster_id = fragment.supercluster_id;
                source->cluster_j = fragment.cluster_j;
                source->split_id = fragment.split_id;
                source->imask = fragment.imask;
                source->valid_mask_j = fragment.valid_mask_j;
                source->local_mask_j = fragment.local_mask_j;
                source->source_order = write_idx;
                atomicAdd(record_stream_fill_cursor, 1);
                write_valid = 1;
            }
            else
            {
                atomicAdd(record_stream_overflow_rows, 1);
            }
        }
    }
    write_idx = deviceShfl(FULL_MASK, write_idx, 0, warpSize);
    write_valid = deviceShfl(FULL_MASK, write_valid, 0, warpSize);
    if (write_valid == 0)
    {
        return;
    }

    const int pair_idx = lane_id;
    const int split_j_lane = pair_idx / kClusteredClusterSize;
    const int i_lane = pair_idx - split_j_lane * kClusteredClusterSize;
    const int j_lane =
        fragment.split_id * kClusteredSplitJClusterSize + split_j_lane;
    const bool valid_j = (fragment.valid_mask_j &
                          (1u << static_cast<unsigned int>(j_lane))) != 0u;
    const bool local_j = (fragment.local_mask_j &
                          (1u << static_cast<unsigned int>(j_lane))) != 0u;
    const int cluster_i_start = super_cluster_offsets[fragment.supercluster_id];
    unsigned int pair_word = 0u;
#pragma unroll
    for (int i_local = 0; i_local < kClusteredMaxSuperClusterClusters;
         i_local += 1)
    {
        const unsigned int source_imask_bit =
            1u << static_cast<unsigned int>(i_local);
        if ((fragment.imask & source_imask_bit) == 0u)
        {
            continue;
        }
        const int cluster_i = cluster_i_start + i_local;
        const unsigned int local_mask_i = cluster_local_masks[cluster_i];
        const bool local_i =
            (local_mask_i & (1u << static_cast<unsigned int>(i_lane))) != 0u;
        bool allow_pair = valid_j && local_i;
        if (allow_pair &&
            !Clustered_Local_I_Owns_Pair(cluster_i, i_lane, fragment.cluster_j,
                                         j_lane, local_j))
        {
            allow_pair = false;
        }
        if (allow_pair &&
            (fragment.exclusion_masks[i_local] &
             (1ull << static_cast<unsigned int>(i_lane * kClusteredClusterSize +
                                                j_lane))) != 0ull)
        {
            allow_pair = false;
        }
        if (allow_pair)
        {
            pair_word |= source_imask_bit;
        }
    }
    record_stream_sources[write_idx].pair_exclusion_words[pair_idx] = pair_word;
}
