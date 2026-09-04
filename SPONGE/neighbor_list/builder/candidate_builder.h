#pragma once

#include "../contract/traversal.cuh"

struct CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT
{
    int sci_id = -1;
    int shift_id = kClusteredCentralShiftId;
    int supercluster_id = -1;
    int cluster_j = -1;
    int split_id = 0;
    unsigned int imask = 0u;
    unsigned int valid_mask_j = 0u;
    unsigned int local_mask_j = 0u;
    int source_order = 0;
    unsigned long long exclusion_masks[kClusteredSuperClusterClusters] = {};
};

#ifndef USE_CPU
void Launch_Clustered_Gmxpacked_Candidate_Leaf_Root_Child_Task_Build(
    int task_build_blocks, int task_build_block_size, int candidate_sci_numbers,
    const int* sci_supercluster_ids, const VECTOR* super_cluster_centers,
    const VECTOR* super_cluster_sizes, const uint64_t* node_prefixes,
    const int* child_offsets, const int* candidate_shift_ids,
    bool use_fast_node_overlap, int task_capacity,
    int* task_counter, int* task_overflow, int* task_sci_ids,
    int* task_nodes, int task_split_depth = 1);

void Launch_Clustered_Gmxpacked_Candidate_Leaf_Root_Child_Device_Counter_Count(
    int collect_blocks, int builder_block_size, int candidate_sci_numbers,
    const int* sci_supercluster_ids, const VECTOR* super_cluster_centers,
    const VECTOR* super_cluster_sizes, const int* super_cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    LTMatrix3 cell, float cutoff, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, bool use_fast_node_overlap,
    int task_capacity,
    const int* task_counter, int* task_work_cursor, int* counts,
    int* task_leaf_counts, const int* root_child_task_sci_ids,
    const int* root_child_task_nodes);

void Launch_Clustered_Gmxpacked_Candidate_Leaf_Root_Child_Device_Counter_Emit(
    int collect_blocks, int builder_block_size, int candidate_sci_numbers,
    const int* sci_supercluster_ids, const VECTOR* super_cluster_centers,
    const VECTOR* super_cluster_sizes, const int* super_cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    LTMatrix3 cell, float cutoff, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, bool use_fast_node_overlap,
    int task_capacity,
    const int* task_counter, int* task_work_cursor, int* emit_counts,
    const int* candidate_leaf_offsets, int* candidate_leaf_ids,
    const int* task_leaf_offsets, int* emit_overflow,
    const int* root_child_task_sci_ids,
    const int* root_child_task_nodes);

void Launch_Clustered_Gmxpacked_Candidate_Leaf_Root_Child_Device_Counter_Fused(
    int collect_blocks, int builder_block_size, int candidate_sci_numbers,
    const int* sci_supercluster_ids, const VECTOR* super_cluster_centers,
    const VECTOR* super_cluster_sizes, const int* super_cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    LTMatrix3 cell, float cutoff, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, bool use_fast_node_overlap,
    int task_capacity,
    const int* task_counter, int* task_work_cursor, int* counts,
    int* task_leaf_counts, int fused_record_capacity, int* fused_record_cursor,
    int* fused_record_overflow, int* fused_task_ids, int* fused_leaf_ranks,
    int* fused_leaf_ids, const int* root_child_task_sci_ids,
    const int* root_child_task_nodes);

void Launch_Clustered_Gmxpacked_Count_Fixed_Light_Dedicated_Cooperative(
    int candidate_sci_blocks, int builder_block_size,
    int candidate_sci_numbers, int cluster_size, int local_atom_numbers,
    float record_stream_cutoff, LTMatrix3 cell, LTMatrix3 rcell,
    const VECTOR* crd, const int* permutation, const int* cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* super_cluster_offsets, const int* cluster_to_supercluster,
    const int* sci_supercluster_ids, const int* candidate_leaf_offsets,
    const int* candidate_leaf_ids, int candidate_leaf_cluster_stride,
    const int* candidate_leaf_prev_running_max_ends,
    const unsigned int* candidate_leaf_reach_masks,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const uint64_t* cluster_molecule_signatures,
    const int* cluster_molecule_ids, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers,
    int max_leaf_cluster_span, int* sci_shift_flags,
    int* cjpacked_group_counts, int* exclusion_counts,
    int* record_stream_source_rows,
    int* record_stream_source_counts_by_candidate,
    bool accumulate_record_stream_source_rows_by_candidate,
    CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT* count_light_source_fragments,
    int count_source_fragment_capacity, int* count_source_fragment_cursor,
    int* count_source_fragment_overflow_rows);
#endif
