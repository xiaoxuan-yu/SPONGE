#pragma once

#include <cstdint>
#include <vector>

#include "../../common.h"
#include "../../control.h"
#include "../contract/view.h"
#include "config.h"
#ifndef USE_CPU
#include "../../third_party/cornerstone_octree/include/cstone/cuda/device_vector.h"
#include "../../third_party/cornerstone_octree/include/cstone/execution.hpp"
#endif
#include "../../third_party/cornerstone_octree/include/cstone/tree/octree.hpp"

struct CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT;
struct ClusteredTreeState;

template <typename T>
struct DeviceBuffer
{
    T* data = nullptr;
    int capacity = 0;
};

struct RawDeviceWorkspace
{
    void* data = nullptr;
    size_t bytes = 0;
};

struct ClusteredBuildWorkspace
{
    RawDeviceWorkspace sort_keys;
    RawDeviceWorkspace sort_values;
    RawDeviceWorkspace sort;
    RawDeviceWorkspace reduce;
    RawDeviceWorkspace scan;
    DeviceBuffer<int> scan_total;

    DeviceBuffer<int> leaf_cluster_span_max;
    DeviceBuffer<uint64_t> stable_sort_keys;
    DeviceBuffer<int> need_rebuild;
    DeviceBuffer<int> record_scratch_counts;
    DeviceBuffer<int> record_scratch_offsets;
    DeviceBuffer<int> record_scratch_indices;

    int candidate_leaf_onepass_record_capacity = 0;
    int candidate_leaf_onepass_high_water = 0;
    int candidate_leaf_onepass_overflow_count = 0;
    DeviceBuffer<int> candidate_leaf_onepass_sci_ids;
    DeviceBuffer<int> candidate_leaf_onepass_ranks;
    DeviceBuffer<int> candidate_leaf_onepass_leaf_ids;
    DeviceBuffer<int> candidate_leaf_onepass_prev_running_max_ends;
    DeviceBuffer<int> candidate_leaf_onepass_cursor;

    int candidate_leaf_queue2_task_capacity = 0;
    DeviceBuffer<int> candidate_leaf_queue2_task_counter;
    DeviceBuffer<int> candidate_leaf_queue2_task_overflow;
    DeviceBuffer<int> candidate_leaf_queue2_task_work_cursor;
    DeviceBuffer<int> candidate_leaf_queue2_task_sci_ids;
    DeviceBuffer<int> candidate_leaf_queue2_task_nodes;
    DeviceBuffer<uint64_t> candidate_leaf_queue2_task_sort_keys;
    DeviceBuffer<uint64_t> candidate_leaf_queue2_task_pairs;
    DeviceBuffer<int> candidate_leaf_queue2_task_leaf_counts;
    DeviceBuffer<int> candidate_leaf_queue2_task_leaf_offsets;

    DeviceBuffer<int> sci_shift_flags;
    DeviceBuffer<int> sci_shift_offsets;
    DeviceBuffer<int> cjpacked_counts;
    DeviceBuffer<int> cjpacked_group_offsets;
    DeviceBuffer<int> exclusion_counts;
    DeviceBuffer<int> exclusion_offsets;

    DeviceBuffer<CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT>
        count_light_source_fragments;
    DeviceBuffer<int> count_source_fragment_cursor;
    DeviceBuffer<int> count_source_fragment_overflow_rows;
    DeviceBuffer<int> count_source_materialize_cursors;
    DeviceBuffer<int> source_counts_by_candidate;
    DeviceBuffer<int> source_offsets_by_candidate;
    DeviceBuffer<int> source_fill_cursor;
    DeviceBuffer<int> source_overflow_rows;

    DeviceBuffer<uint64_t> endpoint_incidence_keys;
    DeviceBuffer<int> endpoint_incidence_error;
    DeviceBuffer<int> endpoint_incidence_counts;
    DeviceBuffer<int> incremental_replacement_source_counts_by_candidate;
    DeviceBuffer<int> incremental_replacement_source_offsets_by_candidate;
    DeviceBuffer<int> incremental_dirty_atoms;
    DeviceBuffer<int> incremental_dirty_clusters;
    DeviceBuffer<int> incremental_dirty_superclusters;
    DeviceBuffer<int> incremental_dirty_i_candidate_sci;
    DeviceBuffer<int> incremental_dirty_candidate_sci;
};

struct SpatialOrderingArrays
{
    DeviceBuffer<int> sort_permutation;
    DeviceBuffer<VECTOR> cached_crd;
};

struct ClusterArrays
{
    DeviceBuffer<int> offsets;
    DeviceBuffer<unsigned int> valid_masks;
    DeviceBuffer<unsigned int> local_masks;
    DeviceBuffer<VECTOR> centers;
    DeviceBuffer<VECTOR> extents;
    DeviceBuffer<VECTOR> fractional_centers;
    DeviceBuffer<VECTOR> fractional_extents;
    DeviceBuffer<float> radii;
};

struct LeafArrays
{
    DeviceBuffer<int> atom_offsets;
    DeviceBuffer<int> cluster_starts;
    DeviceBuffer<int> cluster_ends;
    int max_cluster_span = 0;
    float periodic_image_max_fractional_extent_bound = 0.0f;
};

struct SuperclusterArrays
{
    DeviceBuffer<int> offsets;
    DeviceBuffer<int> cluster_to_supercluster;
    DeviceBuffer<int> has_local;
    DeviceBuffer<VECTOR> centers;
    DeviceBuffer<VECTOR> sizes;
};

struct MoleculeMetadataArrays
{
    DeviceBuffer<int> global_atom_to_molecule;
    DeviceBuffer<int> local_atom_to_molecule;
    DeviceBuffer<uint64_t> cluster_signatures;
    DeviceBuffer<int> cluster_ids;
};

struct CandidateArrays
{
    int sci_numbers = 0;
    int candidate_sci_numbers = 0;
    int leaf_numbers = 0;
    int leaf_cluster_stride = 0;
    DeviceBuffer<int> sci_supercluster_ids;
    DeviceBuffer<int> leaf_counts;
    DeviceBuffer<int> leaf_offsets;
    DeviceBuffer<int> leaf_ids;
    DeviceBuffer<unsigned int> leaf_reach_masks;
};

struct ClusteredTreeStorage
{
    ClusteredTreeState* cornerstone_state = NULL;
};

struct ClusteredSpatialLayout
{
    int total_atom_numbers = 0;
    int padded_total_atom_numbers = 0;
    int cluster_numbers = 0;
    int local_cluster_numbers = 0;
    int super_cluster_numbers = 0;
    int leaf_numbers = 0;
    long long geometry_generation = 0;
    bool ready = false;

    SpatialOrderingArrays ordering;
    ClusterArrays clusters;
    LeafArrays leaves;
    SuperclusterArrays superclusters;
    MoleculeMetadataArrays molecules;
    CandidateArrays candidates;
    ClusteredTreeStorage tree;
};

struct ClusteredPairList
{
    int gmxpacked_sci_numbers = 0;
    int gmxpacked_cjpacked_numbers = 0;
    int gmxpacked_exclusion_numbers = 0;
    int gmxpacked_split_exclusion_numbers = 0;
    int gmxpacked_record_stream_source_numbers = 0;
    int gmxpacked_record_stream_aggregate_numbers = 0;
    float gmxpacked_inner_active_guard_cutoff = -1.0f;

    float gmxpacked_incremental_source_cutoff = -1.0f;
    int gmxpacked_incremental_source_numbers = 0;
    int gmxpacked_incremental_candidate_sci_numbers = 0;
    bool gmxpacked_incremental_source_offsets_ready = false;
    bool gmxpacked_incremental_source_cache_ready = false;

    bool gmxpacked_inner_active_anchor_ready = false;
    bool gmxpacked_inner_active_source_imasks_ready = false;
    int gmxpacked_inner_active_source_imask_numbers = 0;
    int gmxpacked_inner_active_source_rows_baseline = 0;

    bool gmxpacked_endpoint_incidence_ready = false;
    long long gmxpacked_endpoint_incidence_provider_incarnation = -1;
    long long gmxpacked_endpoint_incidence_payload_generation = -1;
    int gmxpacked_endpoint_incidence_sci_numbers = 0;
    int gmxpacked_endpoint_incidence_cjpacked_numbers = 0;
    int gmxpacked_endpoint_incidence_super_cluster_numbers = 0;
    int gmxpacked_endpoint_incidence_reference_numbers = 0;
    int gmxpacked_endpoint_incidence_offset_tail = 0;

    bool gmxpacked_pair_shift_metadata_ready = false;
    int gmxpacked_pair_shift_metadata_sci_numbers = 0;
    int gmxpacked_pair_shift_metadata_cjpacked_numbers = 0;
    int gmxpacked_pair_shift_metadata_exclusion_numbers = 0;
    long long gmxpacked_pair_shift_metadata_payload_generation = -1;
    long long gmxpacked_pair_shift_metadata_geometry_generation = -1;
    LTMatrix3 gmxpacked_pair_shift_metadata_rcell = {};

    long long gmxpacked_compact_payload_generation = 0;
    long long gmxpacked_current_source_anchor_generation = 0;
    long long gmxpacked_current_source_generation = 0;
    long long gmxpacked_incremental_source_anchor_generation = -1;
    long long gmxpacked_incremental_source_generation = -1;

    DeviceBuffer<VECTOR> gmxpacked_inner_active_anchor_crd;
    DeviceBuffer<unsigned int> gmxpacked_inner_active_source_imasks;

    DeviceBuffer<CLUSTERED_GMXPACKED_SCI> gmxpacked_sci;
    DeviceBuffer<CLUSTERED_GMXPACKED_CJ> gmxpacked_cjpacked;
    DeviceBuffer<CLUSTERED_GMXPACKED_EXCLUSION> gmxpacked_exclusions;
    DeviceBuffer<CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE>
        gmxpacked_record_stream_sources;
    DeviceBuffer<CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE>
        gmxpacked_record_stream_aggregates;

    DeviceBuffer<int> gmxpacked_incremental_source_offsets_by_candidate;
    DeviceBuffer<CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE>
        gmxpacked_incremental_record_stream_sources;

    DeviceBuffer<int> gmxpacked_endpoint_incidence_offsets;
    DeviceBuffer<CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE>
        gmxpacked_endpoint_incidence_references;
    DeviceBuffer<uint64_t> pair_shift_bits;
    DeviceBuffer<int> gmxpacked_pair_shift_sci_safe_flags;
};

struct ClusteredDomainBinding
{
    int local_atom_count = 0;
    int direct_local_atom_count = 0;
    int ghost_atom_count = 0;
    const int* atom_local = nullptr;
    const int* excluded_list_start = nullptr;
    const int* excluded_list = nullptr;
    const int* excluded_numbers = nullptr;
};

struct ClusteredBuildRequest
{
    const VECTOR* coordinates = nullptr;
    LTMatrix3 cell = {};
    LTMatrix3 reciprocal_cell = {};
    float cutoff = -1.0f;
    bool need_endpoint_incidence = false;
};

struct ClusteredTreeState
{
#ifndef USE_CPU
    cstone::DeviceVector<uint64_t> leaves;
    cstone::DeviceVector<uint64_t> tmp_leaves;
    cstone::DeviceVector<unsigned> leaf_counts;
    cstone::DeviceVector<cstone::TreeNodeIndex> work_array;
    cstone::OctreeData<uint64_t, cstone::execution::Gpu> octree;
#else
    std::vector<uint64_t> leaves;
    std::vector<unsigned> leaf_counts;
    cstone::OctreeData<uint64_t, cstone::execution::Cpu> octree;
#endif
};

#ifndef USE_CPU
__global__ void Refresh_Gmxpacked_Pair_Shift_Bits(
    int sci_numbers, const int* super_cluster_offsets,
    const VECTOR* cluster_fractional_centers,
    const VECTOR* cluster_fractional_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const CLUSTERED_GMXPACKED_SCI* gmxpacked_sci,
    const CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked,
    const CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    uint64_t* pair_shift_bits, int* sci_shift_safe_flags);
__global__ void Refresh_Gmxpacked_Pair_Shift_Bits_Unique_Image(
    int sci_numbers, const int* super_cluster_offsets,
    const VECTOR* cluster_fractional_centers,
    const VECTOR* cluster_fractional_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const CLUSTERED_GMXPACKED_SCI* gmxpacked_sci,
    const CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked,
    const CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    uint64_t* pair_shift_bits, int* sci_shift_safe_flags);
#endif
