#pragma once

#include "clustered_lj.h"

enum class ClusteredGmxpackedCountFixedLightProbeMode
{
    Traversal,
    SourcePrune,
    SourceEmit,
    ExclusionEmit,
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

void Launch_Clustered_Gmxpacked_Count_Fixed_Light_Dedicated(
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
    int max_leaf_cluster_span, int* sci_shift_flags,
    int* cjpacked_group_counts, int* exclusion_counts,
    int* record_stream_source_rows,
    int* record_stream_source_counts_by_candidate,
    bool accumulate_record_stream_source_rows_by_candidate,
    LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT* count_light_source_fragments,
    int count_source_fragment_capacity, int* count_source_fragment_cursor,
    int* count_source_fragment_overflow_rows);
