static __global__ void Fill_Local_Supercluster_Ids(
    const int super_cluster_numbers, const int* super_cluster_has_local,
    const int* sci_offsets, int* sci_supercluster_ids)
{
    SIMPLE_DEVICE_FOR(super_i, super_cluster_numbers)
    {
        if (super_cluster_has_local[super_i] != 0)
        {
            const int sci = sci_offsets[super_i];
            sci_supercluster_ids[sci] = super_i;
        }
    }
}

static __global__ void Count_Supercluster_Candidate_Leaves_Fixed_Shift(
    const int sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    const int* super_cluster_offsets, const int* leaf_cluster_starts,
    const int* leaf_cluster_ends, const LTMatrix3 cell, const float cutoff,
    const VECTOR* cluster_centers, const VECTOR* cluster_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const CornerstoneKey* node_prefixes,
    const CornerstoneNodeIndex* child_offsets,
    const CornerstoneNodeIndex* parents,
    const CornerstoneNodeIndex* internal_to_leaf,
    const int* candidate_shift_ids, const bool leaf_screening,
    int* candidate_leaf_counts)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const int sci_base =
            candidate_shift_ids == NULL ? sci / kClusteredShiftCount : sci;
        const int candidate_shift_id = candidate_shift_ids != NULL
                                           ? candidate_shift_ids[sci]
                                           : (sci % kClusteredShiftCount);
        const int super_i = sci_supercluster_ids[sci_base];
        const VECTOR target_center = super_cluster_centers[super_i];
        const VECTOR target_size = super_cluster_sizes[super_i];
        const int cluster_i_start = super_cluster_offsets[super_i];
        const int cluster_i_end = super_cluster_offsets[super_i + 1];
        const VECTOR shift_vec = Shift_Vector_From_Id(candidate_shift_id, cell);
        int count = 0;

        auto overlaps = [=](CornerstoneNodeIndex node)
        {
            return Cornerstone_Node_Overlaps_Shifted_Box(
                node_prefixes[node], target_center, target_size,
                candidate_shift_id);
        };
        auto endpoint = [&](CornerstoneNodeIndex node)
        {
            const int leaf_j = internal_to_leaf[node];
            if (leaf_j >= 0)
            {
                if (leaf_screening &&
                    !Leaf_Has_Fixed_Shift_Candidate_Overlap(
                        cluster_i_start, cluster_i_end,
                        leaf_cluster_starts[leaf_j], leaf_cluster_ends[leaf_j],
                        candidate_shift_id, cutoff, shift_vec,
                        cluster_valid_masks, cluster_local_masks,
                        cluster_centers, cluster_extents))
                {
                    return;
                }
                count += 1;
            }
        };
        cstone::singleTraversal(child_offsets, parents, overlaps, endpoint);
        candidate_leaf_counts[sci] = count;
    }
}

static __global__ void Fill_Supercluster_Candidate_Leaves_Fixed_Shift(
    const int sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    const int* super_cluster_offsets, const int* leaf_cluster_starts,
    const int* leaf_cluster_ends, const LTMatrix3 cell, const float cutoff,
    const VECTOR* cluster_centers, const VECTOR* cluster_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const CornerstoneKey* node_prefixes,
    const CornerstoneNodeIndex* child_offsets,
    const CornerstoneNodeIndex* parents,
    const CornerstoneNodeIndex* internal_to_leaf,
    const int* candidate_shift_ids, const int* candidate_leaf_offsets,
    const bool leaf_screening, int* candidate_leaf_ids)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const int sci_base =
            candidate_shift_ids == NULL ? sci / kClusteredShiftCount : sci;
        const int candidate_shift_id = candidate_shift_ids != NULL
                                           ? candidate_shift_ids[sci]
                                           : (sci % kClusteredShiftCount);
        const int super_i = sci_supercluster_ids[sci_base];
        const VECTOR target_center = super_cluster_centers[super_i];
        const VECTOR target_size = super_cluster_sizes[super_i];
        const int cluster_i_start = super_cluster_offsets[super_i];
        const int cluster_i_end = super_cluster_offsets[super_i + 1];
        const VECTOR shift_vec = Shift_Vector_From_Id(candidate_shift_id, cell);
        int write_offset = candidate_leaf_offsets[sci];

        auto overlaps = [=](CornerstoneNodeIndex node)
        {
            return Cornerstone_Node_Overlaps_Shifted_Box(
                node_prefixes[node], target_center, target_size,
                candidate_shift_id);
        };
        auto endpoint = [&](CornerstoneNodeIndex node)
        {
            const int leaf_j = internal_to_leaf[node];
            if (leaf_j >= 0)
            {
                if (leaf_screening &&
                    !Leaf_Has_Fixed_Shift_Candidate_Overlap(
                        cluster_i_start, cluster_i_end,
                        leaf_cluster_starts[leaf_j], leaf_cluster_ends[leaf_j],
                        candidate_shift_id, cutoff, shift_vec,
                        cluster_valid_masks, cluster_local_masks,
                        cluster_centers, cluster_extents))
                {
                    return;
                }
                candidate_leaf_ids[write_offset] = leaf_j;
                write_offset += 1;
            }
        };
        cstone::singleTraversal(child_offsets, parents, overlaps, endpoint);
    }
}

static __global__ void Count_Supercluster_Candidate_Leaves_Fixed_Shift_Subgroup(
    const int sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    const int* super_cluster_offsets, const int* leaf_cluster_starts,
    const int* leaf_cluster_ends, const LTMatrix3 cell, const float cutoff,
    const VECTOR* cluster_centers, const VECTOR* cluster_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const CornerstoneKey* node_prefixes,
    const CornerstoneNodeIndex* child_offsets,
    const CornerstoneNodeIndex* parents,
    const CornerstoneNodeIndex* internal_to_leaf,
    const int* candidate_shift_ids, int* candidate_leaf_counts)
{
    static_assert(kFixedShiftCandidateLeafSubgroupSize <=
                      kClusteredMaxSuperClusterClusters,
                  "candidate leaf subgroup must cover i clusters by stride");
    static_assert(
        kClusteredBuilderWarpSize % kFixedShiftCandidateLeafSubgroupSize == 0,
        "candidate leaf subgroup must divide a warp");
    static_assert(
        kFixedShiftCandidateLeafSubgroupSize < kClusteredBuilderWarpSize,
        "candidate leaf subgroup mask assumes size < warp");
    constexpr int kGroupsPerBlock =
        kClusteredBuilderBlockSize / kFixedShiftCandidateLeafSubgroupSize;
    const int group_id = threadIdx.x / kFixedShiftCandidateLeafSubgroupSize;
    const int sublane = threadIdx.x % kFixedShiftCandidateLeafSubgroupSize;
    const int lane_id = threadIdx.x & (kClusteredBuilderWarpSize - 1);
    const int subgroup_lane_base = lane_id - sublane;
    const device_mask_t subgroup_mask =
        (((device_mask_t)1 << kFixedShiftCandidateLeafSubgroupSize) - 1u)
        << subgroup_lane_base;
    const int sci = blockIdx.x * kGroupsPerBlock + group_id;
    if (sci >= sci_numbers)
    {
        return;
    }

    const int sci_base =
        candidate_shift_ids == NULL ? sci / kClusteredShiftCount : sci;
    const int candidate_shift_id = candidate_shift_ids != NULL
                                       ? candidate_shift_ids[sci]
                                       : (sci % kClusteredShiftCount);
    const int super_i = sci_supercluster_ids[sci_base];
    const VECTOR target_center = super_cluster_centers[super_i];
    const VECTOR target_size = super_cluster_sizes[super_i];
    const int cluster_i_start = super_cluster_offsets[super_i];
    const int cluster_i_end = super_cluster_offsets[super_i + 1];
    const VECTOR shift_vec = Shift_Vector_From_Id(candidate_shift_id, cell);
    int count = 0;

    auto overlaps = [=](CornerstoneNodeIndex node)
    {
        return Cornerstone_Node_Overlaps_Shifted_Box(node_prefixes[node],
                                                     target_center, target_size,
                                                     candidate_shift_id);
    };
    auto endpoint = [&](CornerstoneNodeIndex node)
    {
        const int leaf_j = internal_to_leaf[node];
        if (leaf_j >= 0)
        {
            if (!Leaf_Has_Fixed_Shift_Candidate_Overlap_Subgroup(
                    cluster_i_start, cluster_i_end, leaf_cluster_starts[leaf_j],
                    leaf_cluster_ends[leaf_j], candidate_shift_id, cutoff,
                    shift_vec, cluster_valid_masks, cluster_local_masks,
                    cluster_centers, cluster_extents, sublane, subgroup_mask))
            {
                return;
            }
            if (sublane == 0)
            {
                count += 1;
            }
        }
    };
    cstone::singleTraversal(child_offsets, parents, overlaps, endpoint);
    if (sublane == 0)
    {
        candidate_leaf_counts[sci] = count;
    }
}

static __global__ void Fill_Supercluster_Candidate_Leaves_Fixed_Shift_Subgroup(
    const int sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    const int* super_cluster_offsets, const int* leaf_cluster_starts,
    const int* leaf_cluster_ends, const LTMatrix3 cell, const float cutoff,
    const VECTOR* cluster_centers, const VECTOR* cluster_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const CornerstoneKey* node_prefixes,
    const CornerstoneNodeIndex* child_offsets,
    const CornerstoneNodeIndex* parents,
    const CornerstoneNodeIndex* internal_to_leaf,
    const int* candidate_shift_ids, const int* candidate_leaf_offsets,
    int* candidate_leaf_ids)
{
    constexpr int kGroupsPerBlock =
        kClusteredBuilderBlockSize / kFixedShiftCandidateLeafSubgroupSize;
    const int group_id = threadIdx.x / kFixedShiftCandidateLeafSubgroupSize;
    const int sublane = threadIdx.x % kFixedShiftCandidateLeafSubgroupSize;
    const int lane_id = threadIdx.x & (kClusteredBuilderWarpSize - 1);
    const int subgroup_lane_base = lane_id - sublane;
    const device_mask_t subgroup_mask =
        (((device_mask_t)1 << kFixedShiftCandidateLeafSubgroupSize) - 1u)
        << subgroup_lane_base;
    const int sci = blockIdx.x * kGroupsPerBlock + group_id;
    if (sci >= sci_numbers)
    {
        return;
    }

    const int sci_base =
        candidate_shift_ids == NULL ? sci / kClusteredShiftCount : sci;
    const int candidate_shift_id = candidate_shift_ids != NULL
                                       ? candidate_shift_ids[sci]
                                       : (sci % kClusteredShiftCount);
    const int super_i = sci_supercluster_ids[sci_base];
    const VECTOR target_center = super_cluster_centers[super_i];
    const VECTOR target_size = super_cluster_sizes[super_i];
    const int cluster_i_start = super_cluster_offsets[super_i];
    const int cluster_i_end = super_cluster_offsets[super_i + 1];
    const VECTOR shift_vec = Shift_Vector_From_Id(candidate_shift_id, cell);
    int write_offset = candidate_leaf_offsets[sci];

    auto overlaps = [=](CornerstoneNodeIndex node)
    {
        return Cornerstone_Node_Overlaps_Shifted_Box(node_prefixes[node],
                                                     target_center, target_size,
                                                     candidate_shift_id);
    };
    auto endpoint = [&](CornerstoneNodeIndex node)
    {
        const int leaf_j = internal_to_leaf[node];
        if (leaf_j >= 0)
        {
            if (!Leaf_Has_Fixed_Shift_Candidate_Overlap_Subgroup(
                    cluster_i_start, cluster_i_end, leaf_cluster_starts[leaf_j],
                    leaf_cluster_ends[leaf_j], candidate_shift_id, cutoff,
                    shift_vec, cluster_valid_masks, cluster_local_masks,
                    cluster_centers, cluster_extents, sublane, subgroup_mask))
            {
                return;
            }
            if (sublane == 0)
            {
                candidate_leaf_ids[write_offset] = leaf_j;
                write_offset += 1;
            }
        }
    };
    cstone::singleTraversal(child_offsets, parents, overlaps, endpoint);
}

static __global__ void Scatter_Candidate_Leaves_From_Queue2_Task_Records(
    const int record_numbers, const int* task_leaf_offsets,
    const int* record_task_ids, const int* record_leaf_ranks,
    const int* record_leaf_ids, int* candidate_leaf_ids)
{
    SIMPLE_DEVICE_FOR(record_idx, record_numbers)
    {
        const int task_idx = record_task_ids[record_idx];
        const int rank = record_leaf_ranks[record_idx];
        const int write_idx = task_leaf_offsets[task_idx] + rank;
        candidate_leaf_ids[write_idx] = record_leaf_ids[record_idx];
    }
}

static __global__ void Build_Candidate_Leaf_Queue2_Task_Sort_Keys(
    const int task_count, const int* task_sci_ids, const int* task_nodes,
    uint64_t* task_sort_keys, uint64_t* task_pairs)
{
    SIMPLE_DEVICE_FOR(task_idx, task_count)
    {
        const unsigned int sci =
            static_cast<unsigned int>(task_sci_ids[task_idx]);
        const unsigned int node =
            static_cast<unsigned int>(task_nodes[task_idx]);
        task_sort_keys[task_idx] =
            (static_cast<uint64_t>(sci) << 32) | static_cast<uint64_t>(node);
        task_pairs[task_idx] =
            (static_cast<uint64_t>(sci) << 32) | static_cast<uint64_t>(node);
    }
}

static __global__ void Unpack_Candidate_Leaf_Queue2_Task_Pairs(
    const int task_count, const uint64_t* task_pairs, int* task_sci_ids,
    int* task_nodes)
{
    SIMPLE_DEVICE_FOR(task_idx, task_count)
    {
        const uint64_t pair = task_pairs[task_idx];
        task_sci_ids[task_idx] = static_cast<int>(pair >> 32);
        task_nodes[task_idx] = static_cast<int>(pair & 0xffffffffu);
    }
}

static __global__ void Build_Fixed_Shift_Candidate_Leaf_Masks(
    const int candidate_sci_numbers, const int super_cluster_clusters,
    const float cutoff, const LTMatrix3 cell, const int* sci_supercluster_ids,
    const int* super_cluster_offsets, const int* cluster_to_supercluster,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* candidate_shift_ids, const int* candidate_leaf_offsets,
    const int* candidate_leaf_ids, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, const int candidate_leaf_cluster_stride,
    unsigned int* candidate_leaf_reach_masks)
{
    const int lane_id = threadIdx.x & (warpSize - 1);
    const int warp_id = threadIdx.x / warpSize;
    const int warps_per_block = blockDim.x / warpSize;
    const int candidate_sci = blockIdx.x * warps_per_block + warp_id;
    if (candidate_sci >= candidate_sci_numbers)
    {
        return;
    }

    const int sci_base = candidate_shift_ids == NULL
                             ? candidate_sci / kClusteredShiftCount
                             : candidate_sci;
    const int fixed_shift_id = candidate_shift_ids != NULL
                                   ? candidate_shift_ids[candidate_sci]
                                   : (candidate_sci % kClusteredShiftCount);
    const VECTOR shift_vec = Shift_Vector_From_Id(fixed_shift_id, cell);
    const int super_i = sci_supercluster_ids[sci_base];
    const int cluster_i_start = super_cluster_offsets[super_i];
    const int cluster_i_end = super_cluster_offsets[super_i + 1];

    for (int candidate_idx = candidate_leaf_offsets[candidate_sci];
         candidate_idx < candidate_leaf_offsets[candidate_sci + 1];
         candidate_idx += 1)
    {
        int leaf_j = 0;
        int cluster_j_start = 0;
        int cluster_j_end = 0;
        if (lane_id == 0)
        {
            leaf_j = candidate_leaf_ids[candidate_idx];
            cluster_j_start = leaf_cluster_starts[leaf_j];
            cluster_j_end = leaf_cluster_ends[leaf_j];
        }
        leaf_j = deviceShfl(FULL_MASK, leaf_j, 0, warpSize);
        cluster_j_start = deviceShfl(FULL_MASK, cluster_j_start, 0, warpSize);
        cluster_j_end = deviceShfl(FULL_MASK, cluster_j_end, 0, warpSize);

        const int mask_base = candidate_idx * candidate_leaf_cluster_stride;
        for (int slot = lane_id; slot < candidate_leaf_cluster_stride;
             slot += warpSize)
        {
            unsigned int i_mask = 0u;
            const int cluster_j = cluster_j_start + slot;
            if (cluster_j < cluster_j_end)
            {
                const unsigned int valid_mask_j =
                    cluster_valid_masks[cluster_j];
                if (valid_mask_j != 0u)
                {
                    const int super_j = cluster_to_supercluster[cluster_j];
                    const unsigned int local_mask_j =
                        cluster_local_masks[cluster_j];
                    if (!(Clustered_Valid_Lanes_Are_All_Local(valid_mask_j,
                                                              local_mask_j) &&
                          super_j < super_i))
                    {
                        i_mask = Build_Fixed_Shift_Cluster_I_Mask(
                            cluster_i_start, cluster_i_end, cluster_j,
                            fixed_shift_id, cutoff, shift_vec,
                            cluster_valid_masks, cluster_local_masks,
                            cluster_centers, cluster_extents);
                    }
                }
            }
            candidate_leaf_reach_masks[mask_base + slot] = i_mask;
        }
    }
}
