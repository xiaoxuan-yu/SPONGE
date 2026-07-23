#pragma once

#include "clustered_lj.h"

enum class ClusteredGmxpackedCountFixedLightProbeMode
{
    Traversal,
    SourcePrune,
    SourceEmit,
    ExclusionEmit,
};

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

struct LJ_CLUSTERED_GMXPACKED_COUNT_EXPERIMENT_LIGHT_FRAGMENT
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

struct LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT
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

struct LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT_SLIM
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
};

void Launch_Clustered_Gmxpacked_Count_Fixed_Light_Probe(
    ClusteredGmxpackedCountFixedLightProbeMode mode,
    int candidate_sci_blocks, int builder_block_size,
    int candidate_sci_numbers, int cluster_size, int local_atom_numbers,
    float record_stream_cutoff, LTMatrix3 cell, LTMatrix3 rcell,
    const VECTOR* crd, const int* permutation, const int* cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* super_cluster_offsets, const int* cluster_to_supercluster,
    const int* sci_supercluster_ids, const int* candidate_leaf_offsets,
    const int* candidate_leaf_ids, int candidate_leaf_cluster_stride,
    const unsigned int* candidate_leaf_reach_masks,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const uint64_t* cluster_molecule_signatures,
    const int* cluster_molecule_ids, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers,
    int max_leaf_cluster_span, int* probe_counts,
    LJ_CLUSTERED_GMXPACKED_COUNT_EXPERIMENT_LIGHT_FRAGMENT* probe_fragments,
    int probe_fragment_capacity, int* probe_fragment_cursor,
    int* probe_fragment_overflow_rows);

void Launch_Clustered_Gmxpacked_Candidate_Leaf_Probe(
    ClusteredGmxpackedCandidateLeafProbeMode mode,
    int candidate_sci_blocks, int builder_block_size, int candidate_sci_numbers,
    const int* sci_supercluster_ids, const VECTOR* super_cluster_centers,
    const VECTOR* super_cluster_sizes, const int* super_cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* leaf_all_local, LTMatrix3 cell, float cutoff,
    const VECTOR* cluster_centers, const VECTOR* cluster_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, bool central_halfshell_culling,
    bool use_morton_sfc, bool use_fast_node_overlap,
    bool use_cooperative_traversal, bool use_root_child_split,
    int onepass_capacity, int* probe_counts,
    LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_RECORD* probe_records,
    int* probe_cursor, int* probe_overflow,
    LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_STATS* probe_stats = nullptr,
    const int* candidate_sci_index_map = nullptr,
    const int* root_child_task_sci_ids = nullptr,
    const int* root_child_task_nodes = nullptr);

void Launch_Clustered_Gmxpacked_Candidate_Leaf_Root_Child_Task_Build(
    int task_build_blocks, int task_build_block_size, int candidate_sci_numbers,
    const int* sci_supercluster_ids, const VECTOR* super_cluster_centers,
    const VECTOR* super_cluster_sizes, const uint64_t* node_prefixes,
    const int* child_offsets, const int* candidate_shift_ids,
    bool use_morton_sfc, bool use_fast_node_overlap, int task_capacity,
    int* task_counter, int* task_overflow, int* task_sci_ids,
    int* task_nodes, int task_split_depth = 1);

void Launch_Clustered_Gmxpacked_Candidate_Leaf_Root_Child_Device_Counter_Probe(
    int collect_blocks, int builder_block_size, int candidate_sci_numbers,
    const int* sci_supercluster_ids, const VECTOR* super_cluster_centers,
    const VECTOR* super_cluster_sizes, const int* super_cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* leaf_all_local, LTMatrix3 cell, float cutoff,
    const VECTOR* cluster_centers, const VECTOR* cluster_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, bool central_halfshell_culling,
    bool use_morton_sfc, bool use_fast_node_overlap, int task_capacity,
    const int* task_counter, int* task_work_cursor, int* probe_counts,
    int* task_leaf_counts, const int* root_child_task_sci_ids,
    const int* root_child_task_nodes);

void Launch_Clustered_Gmxpacked_Candidate_Leaf_Root_Child_Device_Counter_Emit(
    int collect_blocks, int builder_block_size, int candidate_sci_numbers,
    const int* sci_supercluster_ids, const VECTOR* super_cluster_centers,
    const VECTOR* super_cluster_sizes, const int* super_cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* leaf_all_local, LTMatrix3 cell, float cutoff,
    const VECTOR* cluster_centers, const VECTOR* cluster_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, bool central_halfshell_culling,
    bool use_morton_sfc, bool use_fast_node_overlap, int task_capacity,
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
    const int* leaf_all_local, LTMatrix3 cell, float cutoff,
    const VECTOR* cluster_centers, const VECTOR* cluster_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, bool central_halfshell_culling,
    bool use_morton_sfc, bool use_fast_node_overlap, int task_capacity,
    const int* task_counter, int* task_work_cursor, int* probe_counts,
    int* task_leaf_counts, int fused_record_capacity, int* fused_record_cursor,
    int* fused_record_overflow, int* fused_task_ids, int* fused_leaf_ranks,
    int* fused_leaf_ids, const int* root_child_task_sci_ids,
    const int* root_child_task_nodes);

void Launch_Clustered_Gmxpacked_Count_Fixed_Light_Dedicated(
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
    LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT* count_light_source_fragments,
    int count_source_fragment_capacity, int* count_source_fragment_cursor,
    int* count_source_fragment_overflow_rows);

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
    LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT* count_light_source_fragments,
    int count_source_fragment_capacity, int* count_source_fragment_cursor,
    int* count_source_fragment_overflow_rows);

void Launch_Clustered_Gmxpacked_Count_Fixed_Light_Dedicated_Dynamic(
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
    LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT* count_light_source_fragments,
    int count_source_fragment_capacity, int* count_source_fragment_cursor,
    int* count_source_fragment_overflow_rows, int* dynamic_work_counter);

void Launch_Clustered_Gmxpacked_Count_Fixed_Light_Dedicated_Slim(
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
    LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT_SLIM* count_slim_source_fragments,
    int count_source_fragment_capacity, int* count_source_fragment_cursor,
    int* count_source_fragment_overflow_rows);

void Launch_Clustered_Gmxpacked_Record_Stream_Source_Materialize_From_Gmxpacked(
    int sci_numbers, int builder_block_size,
    const LJ_CLUSTERED_GMXPACKED_SCI* gmxpacked_sci,
    const LJ_CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked,
    const LJ_CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    int source_capacity,
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    int* source_cursor, int* overflow_rows);

void Launch_Clustered_Gmxpacked_Record_Stream_Inner_Active_Count_Probe(
    int source_rows, int builder_block_size,
    const LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    const int* permutation, const int* cluster_offsets,
    const int* super_cluster_offsets, const unsigned int* cluster_local_masks,
    const VECTOR* cluster_centers, const VECTOR* crd, LTMatrix3 cell,
    LTMatrix3 rcell, float cutoff_sq, int* active_flags,
    unsigned int* active_imasks_by_source);

void Launch_Clustered_Gmxpacked_Record_Stream_Inner_Active_Fill_Probe(
    int source_rows, int builder_block_size,
    const LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    const int* permutation, const int* cluster_offsets,
    const int* super_cluster_offsets, const unsigned int* cluster_local_masks,
    const VECTOR* cluster_centers, const VECTOR* crd, LTMatrix3 cell,
    LTMatrix3 rcell, float cutoff_sq, const int* active_offsets,
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* active_sources);

void Launch_Clustered_Gmxpacked_Record_Stream_Inner_Active_Fill_Cached_Probe(
    int source_rows, int builder_block_size,
    const LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    const unsigned int* active_imasks_by_source, const int* active_offsets,
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* active_sources);
