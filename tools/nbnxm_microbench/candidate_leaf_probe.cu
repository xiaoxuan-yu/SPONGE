#include "candidate_leaf_probe.h"

#ifndef USE_CPU

#include "neighbor_list/builder/candidate_common.cuh"

namespace
{

template <bool kScreen, bool kEmit, bool kFastNodeOverlap,
          bool kCoopTraversal, bool kStats, bool kRootChildSplit>
static __global__ void Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup(
    const int sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    const int* super_cluster_offsets, const int* leaf_cluster_starts,
    const int* leaf_cluster_ends, const LTMatrix3 cell, const float cutoff,
    const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, const int onepass_capacity,
    int* probe_counts,
    LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_RECORD* probe_records,
    int* probe_cursor, int* probe_overflow,
    LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_STATS* probe_stats,
    const int* candidate_sci_index_map, const int* root_child_task_sci_ids,
    const int* root_child_task_nodes)
{
    const int groups_per_block =
        blockDim.x / kFixedShiftCandidateLeafSubgroupSize;
    const int group_id = threadIdx.x / kFixedShiftCandidateLeafSubgroupSize;
    const int sublane = threadIdx.x % kFixedShiftCandidateLeafSubgroupSize;
    const int lane_id = threadIdx.x & (kClusteredBuilderWarpSize - 1);
    const int subgroup_lane_base = lane_id - sublane;
    const device_mask_t subgroup_mask =
        (((device_mask_t)1 << kFixedShiftCandidateLeafSubgroupSize) - 1u)
        << subgroup_lane_base;
    const int logical_item = blockIdx.x * groups_per_block + group_id;
    const bool use_root_child_queue =
        root_child_task_sci_ids != NULL && root_child_task_nodes != NULL;
    const int logical_sci =
        use_root_child_queue ? logical_item
                             : (kRootChildSplit ? logical_item / 8
                                                : logical_item);
    const int root_child_slot =
        use_root_child_queue ? 0 : (kRootChildSplit ? logical_item % 8 : 0);
    if (logical_item >= sci_numbers && use_root_child_queue)
    {
        return;
    }
    if (!use_root_child_queue && logical_sci >= sci_numbers)
    {
        return;
    }
    const int sci = use_root_child_queue
                        ? root_child_task_sci_ids[logical_item]
                        : (candidate_sci_index_map != NULL
                               ? candidate_sci_index_map[logical_sci]
                               : logical_sci);

    const int sci_base =
        candidate_shift_ids == NULL ? sci / kClusteredShiftCount : sci;
    const int candidate_shift_id =
        candidate_shift_ids != NULL ? candidate_shift_ids[sci]
                                    : (sci % kClusteredShiftCount);
    const int super_i = sci_supercluster_ids[sci_base];
    const VECTOR target_center = super_cluster_centers[super_i];
    const VECTOR target_size = super_cluster_sizes[super_i];
    const int cluster_i_start = super_cluster_offsets[super_i];
    const int cluster_i_end = super_cluster_offsets[super_i + 1];
    const VECTOR shifted_target_center =
        target_center + Candidate_Shift_Fractional_From_Id(candidate_shift_id);
    const VECTOR shift_vec = Candidate_Shift_Vector_From_Id(candidate_shift_id, cell);
    const float cutoff_sq = cutoff * cutoff;
    const int cluster_i = cluster_i_start + sublane;
    const bool lane_i_valid =
        cluster_i < cluster_i_end && cluster_local_masks[cluster_i] != 0u;
    const VECTOR center_i =
        lane_i_valid ? cluster_centers[cluster_i] : VECTOR{0.0f, 0.0f, 0.0f};
    const VECTOR extent_i =
        lane_i_valid ? cluster_extents[cluster_i] : VECTOR{0.0f, 0.0f, 0.0f};
    int count = 0;
    int running_max_end = 0;
    LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_STATS stats = {};

    auto write_stats = [&]()
    {
        if constexpr (kStats)
        {
            if (sublane == 0 && probe_stats != NULL)
            {
                stats.accepted_leaves = count;
                probe_stats[logical_sci] = stats;
            }
        }
    };

    auto overlaps = [&](int node)
    {
        if constexpr (kStats)
        {
            if (sublane == 0)
            {
                stats.overlap_tests += 1;
            }
        }
        bool result = false;
        if constexpr (kFastNodeOverlap)
        {
            result = Candidate_Node_Overlaps_Preshifted_Box(
                node_prefixes[node], shifted_target_center, target_size);
        }
        else
        {
            result = Candidate_Node_Overlaps_Shifted_Box(
                node_prefixes[node], target_center, target_size,
                candidate_shift_id);
        }
        if constexpr (kStats)
        {
            if (sublane == 0 && node == 0 && !result)
            {
                stats.root_rejects += 1;
            }
        }
        return result;
    };
    auto endpoint_leaf = [&](int leaf_j)
    {
        if (leaf_j < 0)
        {
            return;
        }
        if constexpr (kStats)
        {
            if (sublane == 0)
            {
                stats.endpoint_leaves += 1;
            }
        }
        if constexpr (kScreen)
        {
            if constexpr (kStats)
            {
                if (sublane == 0)
                {
                    stats.screen_tests += 1;
                }
            }
            if (!Candidate_Leaf_Has_Fixed_Shift_Overlap_Subgroup(
                    cluster_i_start, leaf_cluster_starts[leaf_j],
                    leaf_cluster_ends[leaf_j], candidate_shift_id, cutoff_sq,
                    shift_vec, cluster_valid_masks, cluster_local_masks,
                    cluster_centers,
                    cluster_extents, cluster_i, lane_i_valid, center_i, extent_i,
                    subgroup_mask))
            {
                return;
            }
            if constexpr (kStats)
            {
                if (sublane == 0)
                {
                    stats.screen_accepts += 1;
                }
            }
        }
        if (sublane == 0)
        {
            const int rank = count;
            const int prev_running_max_end = running_max_end;
            count += 1;
            if constexpr (kEmit)
            {
                const int write_idx = atomicAdd(probe_cursor, 1);
                if (write_idx < onepass_capacity && probe_records != NULL)
                {
                    LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_RECORD record;
                    record.sci_id = sci;
                    record.rank = rank;
                    record.leaf_id = leaf_j;
                    record.prev_running_max_end = prev_running_max_end;
                    probe_records[write_idx] = record;
                }
                else if (probe_overflow != NULL)
                {
                    atomicAdd(probe_overflow, 1);
                }
            }
            running_max_end =
                Candidate_Int_Max(running_max_end, leaf_cluster_ends[leaf_j]);
        }
    };
    auto endpoint = [&](int node) { endpoint_leaf(internal_to_leaf[node]); };
    if constexpr (kCoopTraversal)
    {
        constexpr int init_node = 0;
        int root_descend = 0;
        int root_child = 0;
        int root_leaf = -1;
        if (sublane == 0)
        {
            if constexpr (kStats)
            {
                stats.node_visits += 1;
            }
            root_descend = overlaps(init_node) ? 1 : 0;
            root_child = child_offsets[init_node];
            root_leaf = internal_to_leaf[init_node];
        }
        root_descend =
            deviceShfl(subgroup_mask, root_descend, subgroup_lane_base,
                       kClusteredBuilderWarpSize);
        root_child = deviceShfl(subgroup_mask, root_child, subgroup_lane_base,
                                kClusteredBuilderWarpSize);
        root_leaf = deviceShfl(subgroup_mask, root_leaf, subgroup_lane_base,
                               kClusteredBuilderWarpSize);
        if (root_descend == 0)
        {
            if (sublane == 0 && probe_counts != NULL)
            {
                probe_counts[logical_sci] = 0;
            }
            write_stats();
            return;
        }
        if (root_child == 0)
        {
            endpoint_leaf(root_leaf);
            if (sublane == 0 && probe_counts != NULL)
            {
                probe_counts[logical_sci] = count;
            }
            write_stats();
            return;
        }

        int node = root_child;
        int backtrack = 0;
        while (node != init_node)
        {
            int endpoint_active = 0;
            int endpoint_leaf_j = -1;
            int next_node = init_node;
            int next_backtrack = 0;
            if (sublane == 0)
            {
                if constexpr (kStats)
                {
                    stats.node_visits += 1;
                }
                const bool is_leaf = child_offsets[node] == 0;
                const bool descend = !backtrack && overlaps(node);
                endpoint_active = is_leaf && descend ? 1 : 0;
                endpoint_leaf_j =
                    endpoint_active != 0 ? internal_to_leaf[node] : -1;
                const int sibling_idx = (node - 1) % 8;
                if (!is_leaf && descend)
                {
                    next_node = child_offsets[node];
                    next_backtrack = 0;
                }
                else if (sibling_idx < 7)
                {
                    next_node = node + 1;
                    next_backtrack = 0;
                }
                else
                {
                    next_node = parents[(node - 1) / 8];
                    next_backtrack = 1;
                }
            }
            endpoint_active =
                deviceShfl(subgroup_mask, endpoint_active,
                           subgroup_lane_base, kClusteredBuilderWarpSize);
            endpoint_leaf_j =
                deviceShfl(subgroup_mask, endpoint_leaf_j,
                           subgroup_lane_base, kClusteredBuilderWarpSize);
            if (endpoint_active != 0)
            {
                endpoint_leaf(endpoint_leaf_j);
            }
            next_node =
                deviceShfl(subgroup_mask, next_node, subgroup_lane_base,
                           kClusteredBuilderWarpSize);
            next_backtrack =
                deviceShfl(subgroup_mask, next_backtrack, subgroup_lane_base,
                           kClusteredBuilderWarpSize);
            node = next_node;
            backtrack = next_backtrack;
        }
    }
    else
    {
        if constexpr (kRootChildSplit)
        {
            constexpr int init_node = 0;
            const int task_root_node =
                use_root_child_queue ? root_child_task_nodes[logical_item] : -1;
            if (use_root_child_queue && task_root_node == init_node)
            {
                endpoint(init_node);
                if (sublane == 0 && probe_counts != NULL && count > 0)
                {
                    atomicAdd(&probe_counts[sci], count);
                }
                write_stats();
                return;
            }
            if (!use_root_child_queue && !overlaps(init_node))
            {
                return;
            }
            const int root_child =
                use_root_child_queue ? task_root_node : child_offsets[init_node];
            if (!use_root_child_queue && root_child == 0)
            {
                if (root_child_slot == 0)
                {
                    endpoint(init_node);
                }
            }
            else
            {
                int node = use_root_child_queue
                               ? root_child
                               : root_child + root_child_slot;
                bool backtrack = false;
                while (true)
                {
                    const bool is_leaf = child_offsets[node] == 0;
                    const bool descend = !backtrack && overlaps(node);
                    if (is_leaf && descend)
                    {
                        endpoint(node);
                    }
                    if (!is_leaf && descend)
                    {
                        node = child_offsets[node];
                        backtrack = false;
                    }
                    else if (node ==
                             (use_root_child_queue
                                  ? root_child
                                  : root_child + root_child_slot))
                    {
                        break;
                    }
                    else
                    {
                        const int sibling_idx = (node - 1) % 8;
                        if (sibling_idx < 7)
                        {
                            node += 1;
                            backtrack = false;
                        }
                        else
                        {
                            node = parents[(node - 1) / 8];
                            backtrack = true;
                        }
                    }
                }
            }
            if (sublane == 0 && probe_counts != NULL && count > 0)
            {
                atomicAdd(&probe_counts[use_root_child_queue ? sci
                                                             : logical_sci],
                          count);
            }
            write_stats();
            return;
        }
        else if constexpr (kStats)
        {
            constexpr int init_node = 0;
            if (sublane == 0)
            {
                stats.node_visits += 1;
            }
            if (!overlaps(init_node))
            {
                if (sublane == 0 && probe_counts != NULL)
                {
                    probe_counts[logical_sci] = 0;
                }
                write_stats();
                return;
            }
            if (child_offsets[init_node] == 0)
            {
                endpoint(init_node);
            }
            else
            {
                int node = child_offsets[init_node];
                bool backtrack = false;
                while (node != init_node)
                {
                    if (sublane == 0)
                    {
                        stats.node_visits += 1;
                    }
                    const bool is_leaf = child_offsets[node] == 0;
                    const bool descend = !backtrack && overlaps(node);
                    if (is_leaf && descend)
                    {
                        endpoint(node);
                    }

                    const int sibling_idx = (node - 1) % 8;
                    if (!is_leaf && descend)
                    {
                        node = child_offsets[node];
                        backtrack = false;
                    }
                    else if (sibling_idx < 7)
                    {
                        node += 1;
                        backtrack = false;
                    }
                    else
                    {
                        node = parents[(node - 1) / 8];
                        backtrack = true;
                    }
                }
            }
        }
        else
        {
            cstone::singleTraversal(child_offsets, parents, overlaps, endpoint);
        }
    }
    if (sublane == 0 && probe_counts != NULL)
    {
        probe_counts[logical_sci] = count;
    }
    write_stats();
}

} // namespace

void Launch_Clustered_Gmxpacked_Candidate_Leaf_Probe(
    ClusteredGmxpackedCandidateLeafProbeMode mode,
    int candidate_sci_blocks, int builder_block_size, int candidate_sci_numbers,
    const int* sci_supercluster_ids, const VECTOR* super_cluster_centers,
    const VECTOR* super_cluster_sizes, const int* super_cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    LTMatrix3 cell, float cutoff, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, bool use_fast_node_overlap,
    bool use_cooperative_traversal, bool use_root_child_split,
    int onepass_capacity, int* probe_counts,
    LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_RECORD* probe_records,
    int* probe_cursor, int* probe_overflow,
    LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_STATS* probe_stats,
    const int* candidate_sci_index_map, const int* root_child_task_sci_ids,
    const int* root_child_task_nodes)
{
#define LAUNCH_CANDIDATE_LEAF_PROBE(SCREEN, EMIT)                         \
    do                                                                     \
    {                                                                      \
        if (use_root_child_split)                                          \
        {                                                                  \
            auto* kernel = use_fast_node_overlap                           \
                               ? Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup<SCREEN, false, true, false, false, true> \
                               : Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup<SCREEN, false, false, false, false, true>; \
            Launch_Device_Kernel(                                         \
                kernel, candidate_sci_blocks, builder_block_size, 0, NULL, \
                candidate_sci_numbers, sci_supercluster_ids,               \
                super_cluster_centers, super_cluster_sizes,                \
                super_cluster_offsets, leaf_cluster_starts,                \
                leaf_cluster_ends, cell, cutoff,                           \
                cluster_centers, cluster_extents, cluster_valid_masks,     \
                cluster_local_masks, node_prefixes, child_offsets,         \
                parents, internal_to_leaf, candidate_shift_ids,            \
                onepass_capacity, probe_counts, NULL, NULL, NULL, NULL,    \
                candidate_sci_index_map, root_child_task_sci_ids,          \
                root_child_task_nodes);                                     \
            break;                                                         \
        }                                                                  \
        if (probe_stats != NULL)                                           \
        {                                                                  \
            auto* kernel = use_cooperative_traversal                       \
                               ? (use_fast_node_overlap                    \
                                      ? Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup<SCREEN, EMIT, true, true, true, false> \
                                      : Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup<SCREEN, EMIT, false, true, true, false>) \
                               : (use_fast_node_overlap                    \
                                      ? Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup<SCREEN, EMIT, true, false, true, false> \
                                      : Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup<SCREEN, EMIT, false, false, true, false>); \
            Launch_Device_Kernel(                                         \
                kernel, candidate_sci_blocks, builder_block_size, 0, NULL, \
                candidate_sci_numbers, sci_supercluster_ids,               \
                super_cluster_centers, super_cluster_sizes,                \
                super_cluster_offsets, leaf_cluster_starts,                \
                leaf_cluster_ends, cell, cutoff,                           \
                cluster_centers, cluster_extents, cluster_valid_masks,     \
                cluster_local_masks, node_prefixes, child_offsets,         \
                parents, internal_to_leaf, candidate_shift_ids,            \
                onepass_capacity, probe_counts, probe_records,             \
                probe_cursor, probe_overflow, probe_stats,                 \
                candidate_sci_index_map, root_child_task_sci_ids,          \
                root_child_task_nodes);                                     \
        }                                                                  \
        else                                                               \
        {                                                                  \
            auto* kernel = use_cooperative_traversal                       \
                               ? (use_fast_node_overlap                    \
                                      ? Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup<SCREEN, EMIT, true, true, false, false> \
                                      : Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup<SCREEN, EMIT, false, true, false, false>) \
                               : (use_fast_node_overlap                    \
                                      ? Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup<SCREEN, EMIT, true, false, false, false> \
                                      : Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup<SCREEN, EMIT, false, false, false, false>); \
            Launch_Device_Kernel(                                         \
                kernel, candidate_sci_blocks, builder_block_size, 0, NULL, \
                candidate_sci_numbers, sci_supercluster_ids,               \
                super_cluster_centers, super_cluster_sizes,                \
                super_cluster_offsets, leaf_cluster_starts,                \
                leaf_cluster_ends, cell, cutoff,                           \
                cluster_centers, cluster_extents, cluster_valid_masks,     \
                cluster_local_masks, node_prefixes, child_offsets,         \
                parents, internal_to_leaf, candidate_shift_ids,            \
                onepass_capacity, probe_counts, probe_records,             \
                probe_cursor, probe_overflow, probe_stats,                 \
                candidate_sci_index_map, root_child_task_sci_ids,          \
                root_child_task_nodes);                                     \
        }                                                                  \
    } while (0)
    switch (mode)
    {
    case ClusteredGmxpackedCandidateLeafProbeMode::Traversal:
    {
        LAUNCH_CANDIDATE_LEAF_PROBE(false, false);
        break;
    }
    case ClusteredGmxpackedCandidateLeafProbeMode::Screen:
    {
        LAUNCH_CANDIDATE_LEAF_PROBE(true, false);
        break;
    }
    case ClusteredGmxpackedCandidateLeafProbeMode::Emit:
    {
        LAUNCH_CANDIDATE_LEAF_PROBE(true, true);
        break;
    }
    }
#undef LAUNCH_CANDIDATE_LEAF_PROBE
}

#endif
