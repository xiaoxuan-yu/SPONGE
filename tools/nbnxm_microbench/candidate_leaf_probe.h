#pragma once

#include "neighbor_list/contract/types.h"

enum class ClusteredGmxpackedCandidateLeafProbeMode
{
    Traversal,
    Screen,
    Emit,
};

struct LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_RECORD
{
    int sci_id = -1;
    int rank = 0;
    int leaf_id = -1;
    int prev_running_max_end = 0;
};

struct LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_STATS
{
    int node_visits = 0;
    int overlap_tests = 0;
    int endpoint_leaves = 0;
    int halfshell_rejects = 0;
    int screen_tests = 0;
    int screen_accepts = 0;
    int root_rejects = 0;
    int accepted_leaves = 0;
};

void Launch_Clustered_Gmxpacked_Candidate_Leaf_Probe(
    ClusteredGmxpackedCandidateLeafProbeMode mode,
    int candidate_sci_blocks, int builder_block_size,
    int candidate_sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    const int* super_cluster_offsets, const int* leaf_cluster_starts,
    const int* leaf_cluster_ends, LTMatrix3 cell, float cutoff,
    const VECTOR* cluster_centers, const VECTOR* cluster_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, bool use_fast_node_overlap,
    bool use_cooperative_traversal, bool use_root_child_split,
    int onepass_capacity, int* probe_counts,
    LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_RECORD* probe_records,
    int* probe_cursor, int* probe_overflow,
    LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_STATS* probe_stats = nullptr,
    const int* candidate_sci_index_map = nullptr,
    const int* root_child_task_sci_ids = nullptr,
    const int* root_child_task_nodes = nullptr);
