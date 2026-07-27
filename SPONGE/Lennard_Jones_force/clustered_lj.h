#pragma once

#include <cstdint>
#include <vector>

#include "../common.h"
#include "../control.h"
#include "../neighbor_list/clustered_spatial_view.h"
#ifndef USE_CPU
#include "../third_party/cornerstone_octree/include/cstone/cuda/device_vector.h"
#include "../third_party/cornerstone_octree/include/cstone/execution.hpp"
#endif
#include "../third_party/cornerstone_octree/include/cstone/tree/octree.hpp"

struct VECTOR_LJ;
struct VECTOR_LJ_SOFT_TYPE;

struct LJ_CORNERSTONE_STATE
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
    const LJ_CLUSTERED_GMXPACKED_SCI* gmxpacked_sci,
    const LJ_CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked,
    const LJ_CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    uint64_t* pair_shift_bits, int* sci_shift_only_safe,
    int* sci_shift_safe_flags, int* sci_shift_safe_count,
    bool exact_sci_shift_flags);
#endif

struct LJ_CLUSTERED_LEGACY_NEIGHBOR_VIEW_REQUEST
{
    bool request_half = true;
    bool request_full = false;
    bool contains_non_lj_consumer = false;
    bool require_all_local_atoms = true;
    bool require_local_ghost_pairs = true;
    bool require_exclusions = true;
    int local_atom_numbers = 0;
    int ghost_numbers = 0;
    float cutoff = 0.0f;
    float skin = 0.0f;
    const int* d_atom_local = NULL;
    const int* d_excluded_list_start = NULL;
    const int* d_excluded_list = NULL;
    const int* d_excluded_numbers = NULL;
};

struct LJ_CLUSTER_LAYOUT
{
    CONTROLLER* controller = NULL;
    bool enabled = false;
    bool warn_legacy_ordered_layout = false;
    bool compression_requested = false;
    bool rebuild_dirty = true;
    bool cache_ready = false;
    TIME_RECORDER* payload_build_time_recorder = NULL;
    TIME_RECORDER* primary_payload_time_recorder = NULL;
    int primary_payload_build_step = -1;
    int primary_payload_build_count_this_step = 0;
    long long primary_payload_build_count_total = 0;
    bool runtime_gmxpacked_direct_requested = false;
    bool runtime_aux_clustered_metadata_requested = false;
    bool legacy_neighbor_view_ready = false;
    long long legacy_neighbor_view_payload_build_count = -1;
    int legacy_neighbor_view_step = -1;
    float legacy_neighbor_view_cutoff_skin = -1.0f;

    int cluster_size = 8;
    int super_cluster_clusters = 8;
    int cornerstone_max_depth = 6;
    int cornerstone_leaf_size = 32;
    int working_device = 0;
    int rebuild_refresh_interval = 0;
    int cached_build_step = -1;
    float rebuild_skin = 2.0f;
    float rebuild_skin_permit = 0.5f;
    float cached_cutoff = -1.0f;

    int local_atom_numbers = 0;
    int direct_local_atom_numbers = 0;
    int ghost_numbers = 0;
    int total_atom_numbers = 0;
    int padded_total_atom_numbers = 0;
    int cluster_numbers = 0;
    int super_cluster_numbers = 0;
    int local_cluster_numbers = 0;
    int candidate_sci_numbers = 0;
    int sci_numbers = 0;
    int cjpacked_numbers = 0;
    int gmxpacked_sci_numbers = 0;
    int gmxpacked_cjpacked_numbers = 0;
    int gmxpacked_exclusion_numbers = 0;
    int gmxpacked_split_exclusion_numbers = 0;
    int gmxpacked_record_stream_source_numbers = 0;
    int gmxpacked_record_stream_aggregate_numbers = 0;
    float gmxpacked_inner_active_guard_cutoff = -1.0f;
    float gmxpacked_incremental_source_cutoff = -1.0f;
    float gmxpacked_incremental_source_global_valid_cutoff = -1.0f;
    int gmxpacked_incremental_source_numbers = 0;
    int gmxpacked_incremental_candidate_sci_numbers = 0;
    int gmxpacked_incremental_source_patch_successes = 0;
    int gmxpacked_incremental_source_patch_fallbacks = 0;
    bool gmxpacked_pair_shift_sci_only_compatible = false;
    bool gmxpacked_pair_shift_metadata_ready = false;
    bool gmxpacked_pair_shift_sci_safe_counts_ready = false;
    int gmxpacked_pair_shift_metadata_sci_numbers = 0;
    int gmxpacked_pair_shift_metadata_cjpacked_numbers = 0;
    int gmxpacked_pair_shift_metadata_exclusion_numbers = 0;
    long long gmxpacked_pair_shift_metadata_payload_generation = -1;
    long long gmxpacked_pair_shift_metadata_geometry_generation = -1;
    int gmxpacked_pair_shift_safe_sci_numbers = 0;
    int gmxpacked_pair_shift_unsafe_sci_numbers = 0;
    LTMatrix3 gmxpacked_pair_shift_metadata_rcell = {};
    int candidate_leaf_numbers = 0;
    int candidate_leaf_cluster_stride = 0;
    int exclusion_pool_numbers = 0;
    bool grouped_sci_ready = false;
    bool gmxpacked_grouped_sci_ready = false;
    bool gmxpacked_endpoint_incidence_ready = false;
    long long gmxpacked_endpoint_incidence_provider_incarnation = -1;
    long long gmxpacked_endpoint_incidence_payload_generation = -1;
    int gmxpacked_endpoint_incidence_sci_numbers = 0;
    int gmxpacked_endpoint_incidence_cjpacked_numbers = 0;
    int gmxpacked_endpoint_incidence_super_cluster_numbers = 0;
    int gmxpacked_endpoint_incidence_reference_numbers = 0;
    int gmxpacked_endpoint_incidence_offset_tail = 0;
    bool gmxpacked_incremental_source_offsets_ready = false;
    bool gmxpacked_incremental_source_cache_ready = false;
    bool gmxpacked_outer_source_anchor_ready = false;
    bool stable_target_layout_anchor_ready = false;
    bool gmxpacked_inner_active_anchor_ready = false;
    bool gmxpacked_inner_active_source_imasks_ready = false;
    int gmxpacked_inner_active_source_imask_numbers = 0;
    int gmxpacked_inner_active_source_rows_baseline = 0;
    long long gmxpacked_inner_active_anchor_generation = 0;
    long long gmxpacked_inner_active_source_generation = 0;
    long long provider_incarnation = 1;
    long long spatial_view_lease_epoch = 0;
    long long native_payload_generation = 0;
    long long gmxpacked_compact_payload_generation = 0;
    long long geometry_generation = 0;
    long long gmxpacked_compact_payload_anchor_generation = -1;
    long long gmxpacked_compact_payload_source_generation = -1;
    long long gmxpacked_current_source_anchor_generation = 0;
    long long gmxpacked_current_source_generation = 0;
    long long gmxpacked_incremental_source_anchor_generation = -1;
    long long gmxpacked_incremental_source_generation = -1;
    int gmxpacked_current_source_patch_attempts = 0;
    int gmxpacked_current_source_patch_successes = 0;
    int gmxpacked_current_source_patch_fallbacks = 0;
    long long gmxpacked_inner_active_append_attempts = 0;
    long long gmxpacked_inner_active_append_successes = 0;
    long long gmxpacked_inner_active_append_fallbacks = 0;
    long long gmxpacked_inner_active_append_zero_rows = 0;
    long long gmxpacked_inner_active_append_rows_total = 0;
    long long gmxpacked_inner_active_append_overflow_rows_total = 0;
    int gmxpacked_inner_active_append_max_active_rows = 0;
    int gmxpacked_inner_active_append_max_limit_rows = 0;
    int gmxpacked_inner_active_append_last_active_rows = 0;

    int permutation_capacity = 0;
    int sort_key_capacity = 0;
    int cluster_capacity = 0;
    int cluster_valid_mask_capacity = 0;
    int cluster_local_mask_capacity = 0;
    int cluster_center_capacity = 0;
    int cluster_extent_capacity = 0;
    int cluster_fractional_center_capacity = 0;
    int cluster_fractional_extent_capacity = 0;
    int cluster_radius_capacity = 0;
    int leaf_capacity = 0;
    int leaf_cluster_start_capacity = 0;
    int leaf_cluster_end_capacity = 0;
    int leaf_all_local_capacity = 0;
    int super_cluster_capacity = 0;
    int super_cluster_has_local_capacity = 0;
    int super_cluster_center_capacity = 0;
    int super_cluster_size_capacity = 0;
    int sci_capacity = 0;
    int sci_candidate_leaf_count_capacity = 0;
    int sci_candidate_leaf_offset_capacity = 0;
    int candidate_leaf_capacity = 0;
    int candidate_leaf_prev_running_max_capacity = 0;
    int candidate_leaf_onepass_sci_capacity = 0;
    int candidate_leaf_onepass_rank_capacity = 0;
    int candidate_leaf_onepass_leaf_capacity = 0;
    int candidate_leaf_onepass_prev_running_max_capacity = 0;
    int candidate_leaf_onepass_cursor_capacity = 0;
    int candidate_leaf_onepass_record_capacity = 0;
    int candidate_leaf_onepass_high_water = 0;
    int candidate_leaf_onepass_overflow_count = 0;
    int candidate_leaf_mask_capacity = 0;
    int cjpacked_capacity = 0;
    int exclusion_scan_capacity = 0;
    int exclusion_capacity = 0;
    int scan_capacity = 0;
    int sci_shift_capacity = 0;
    int cjpacked_count_capacity = 0;
    int exclusion_count_capacity = 0;
    int sci_shift_offset_capacity = 0;
    int cjpacked_group_offset_capacity = 0;
    int exclusion_offset_capacity = 0;
    int candidate_offset_capacity = 0;
    int candidate_shift_capacity = 0;
    int grouped_sci_offset_capacity = 0;
    int grouped_sci_id_capacity = 0;
    int gmxpacked_grouped_sci_offset_capacity = 0;
    int gmxpacked_grouped_sci_id_capacity = 0;
    int gmxpacked_endpoint_incidence_offset_capacity = 0;
    int gmxpacked_endpoint_incidence_reference_capacity = 0;
    int gmxpacked_endpoint_incidence_key_capacity = 0;
    int gmxpacked_endpoint_incidence_error_capacity = 0;
    int cached_crd_capacity = 0;
    int rebuild_flag_capacity = 0;
    int nbnxm_sci_capacity = 0;
    int nbnxm_cjpacked_capacity = 0;
    int gmxpacked_sci_capacity = 0;
    int gmxpacked_cjpacked_capacity = 0;
    int gmxpacked_exclusion_capacity = 0;
    int gmxpacked_record_stream_source_capacity = 0;
    int gmxpacked_record_stream_aggregate_capacity = 0;
    int gmxpacked_incremental_source_offset_capacity = 0;
    int gmxpacked_incremental_source_cache_capacity = 0;
    int gmxpacked_incremental_source_valid_cutoff_capacity = 0;
    int gmxpacked_incremental_replacement_source_count_capacity = 0;
    int gmxpacked_incremental_replacement_source_offset_capacity = 0;
    int gmxpacked_incremental_replacement_source_capacity = 0;
    int gmxpacked_incremental_dirty_atom_capacity = 0;
    int gmxpacked_incremental_dirty_cluster_capacity = 0;
    int gmxpacked_incremental_dirty_supercluster_capacity = 0;
    int gmxpacked_incremental_dirty_i_candidate_capacity = 0;
    int gmxpacked_incremental_dirty_candidate_capacity = 0;
    int cjpacked_sort_index_capacity = 0;
    int cjpacked_sort_buffer_capacity = 0;
    int jentry_capacity = 0;
    int jentry_buffer_capacity = 0;
    int jentry_count_capacity = 0;
    int jentry_offset_capacity = 0;
    int jentry_index_capacity = 0;
    int nbnxm_warp_record_capacity = 0;
    int forceonly_warp_record_numbers = 0;
    int forceonly_warp_record_capacity = 0;
    int forceonly_warp_record_count_capacity = 0;
    int forceonly_warp_record_offset_capacity = 0;
    int pair_shift_capacity = 0;
    int gmxpacked_pair_shift_sci_safe_flag_capacity = 0;
    int outer_imask_capacity = 0;
    int global_atom_to_molecule_capacity = 0;
    int local_atom_to_molecule_capacity = 0;
    int cluster_molecule_signature_capacity = 0;
    int cluster_molecule_id_capacity = 0;
    int cluster_to_supercluster_capacity = 0;
    int stable_target_layout_crd_capacity = 0;
    int gmxpacked_outer_source_anchor_crd_capacity = 0;
    int gmxpacked_inner_active_anchor_crd_capacity = 0;
    int gmxpacked_inner_active_source_imask_capacity = 0;

    int* d_sort_permutation = NULL;
    uint64_t* d_sort_keys = NULL;
    VECTOR* d_cached_crd = NULL;
    VECTOR* d_stable_target_layout_crd = NULL;
    VECTOR* d_gmxpacked_outer_source_anchor_crd = NULL;
    VECTOR* d_gmxpacked_inner_active_anchor_crd = NULL;
    unsigned int* d_gmxpacked_inner_active_source_imasks = NULL;
    int* d_need_rebuild = NULL;
    const int* d_atom_local = NULL;
    int* d_global_atom_to_molecule = NULL;
    int* d_local_atom_to_molecule = NULL;

    int* d_cluster_offsets = NULL;
    unsigned int* d_cluster_valid_masks = NULL;
    unsigned int* d_cluster_local_masks = NULL;
    VECTOR* d_cluster_centers = NULL;
    VECTOR* d_cluster_extents = NULL;
    VECTOR* d_cluster_fractional_centers = NULL;
    VECTOR* d_cluster_fractional_extents = NULL;
    float* d_cluster_radii = NULL;
    uint64_t* d_cluster_molecule_signatures = NULL;
    int* d_cluster_molecule_ids = NULL;

    int* d_leaf_atom_offsets = NULL;
    int* d_leaf_cluster_starts = NULL;
    int* d_leaf_cluster_ends = NULL;
    // Phase A subgroup builder: provable upper bound on per-leaf cluster span
    // (max over leaves of cluster_j_end - cluster_j_start), used to bound the
    // backward dedup scan in the subgroup count/fill kernels. Computed once per
    // rebuild by a reduction over leaf_cluster_ends/starts; exact, not a guess.
    int max_leaf_cluster_span = 0;
    int* d_leaf_cluster_span_max_scratch = NULL;
    int leaf_cluster_span_max_scratch_capacity = 0;
    int* d_leaf_all_local = NULL;
    int* d_super_cluster_offsets = NULL;
    int* d_cluster_to_supercluster = NULL;
    int* d_super_cluster_has_local = NULL;
    VECTOR* d_super_cluster_centers = NULL;
    VECTOR* d_super_cluster_sizes = NULL;
    int* d_sci_supercluster_ids = NULL;
    int* d_sci_candidate_leaf_counts = NULL;
    int* d_sci_candidate_leaf_offsets = NULL;
    int* d_sci_candidate_leaf_ids = NULL;
    int* d_sci_candidate_leaf_prev_running_max_ends = NULL;
    int* d_candidate_leaf_onepass_sci_ids = NULL;
    int* d_candidate_leaf_onepass_ranks = NULL;
    int* d_candidate_leaf_onepass_leaf_ids = NULL;
    int* d_candidate_leaf_onepass_prev_running_max_ends = NULL;
    int* d_candidate_leaf_onepass_cursor = NULL;
    int* d_candidate_leaf_queue2_task_counter = NULL;
    int* d_candidate_leaf_queue2_task_overflow = NULL;
    int* d_candidate_leaf_queue2_task_work_cursor = NULL;
    int* d_candidate_leaf_queue2_task_sci_ids = NULL;
    int* d_candidate_leaf_queue2_task_nodes = NULL;
    uint64_t* d_candidate_leaf_queue2_task_sort_keys = NULL;
    uint64_t* d_candidate_leaf_queue2_task_pairs = NULL;
    int* d_candidate_leaf_queue2_task_leaf_counts = NULL;
    int* d_candidate_leaf_queue2_task_leaf_offsets = NULL;
    int candidate_leaf_queue2_counter_capacity = 0;
    int candidate_leaf_queue2_overflow_capacity = 0;
    int candidate_leaf_queue2_work_cursor_capacity = 0;
    int candidate_leaf_queue2_task_sci_capacity = 0;
    int candidate_leaf_queue2_task_node_capacity = 0;
    int candidate_leaf_queue2_task_sort_key_capacity = 0;
    int candidate_leaf_queue2_task_pair_capacity = 0;
    int candidate_leaf_queue2_task_leaf_count_capacity = 0;
    int candidate_leaf_queue2_task_leaf_offset_capacity = 0;
    int candidate_leaf_queue2_task_capacity = 0;
    unsigned int* d_candidate_leaf_reach_masks = NULL;
    int* d_candidate_sci_offsets = NULL;
    int* d_candidate_shift_ids = NULL;
    int* d_grouped_sci_offsets = NULL;
    int* d_grouped_sci_ids = NULL;
    int* d_gmxpacked_grouped_sci_offsets = NULL;
    int* d_gmxpacked_grouped_sci_ids = NULL;
    int* d_gmxpacked_endpoint_incidence_offsets = NULL;
    CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE*
        d_gmxpacked_endpoint_incidence_references = NULL;
    uint64_t* d_gmxpacked_endpoint_incidence_keys = NULL;
    int* d_gmxpacked_endpoint_incidence_error = NULL;
    int* d_cjpacked_counts = NULL;
    int* d_exclusion_counts = NULL;
    int* d_exclusion_offsets = NULL;
    int* d_sci_shift_flags = NULL;
    int* d_sci_shift_offsets = NULL;
    int* d_cjpacked_group_offsets = NULL;
    unsigned int* d_outer_imask = NULL;
    unsigned long long* d_exclusion_mask_pool = NULL;
    LJ_CLUSTERED_SCI* d_nbnxm_sci = NULL;
    LJ_CLUSTERED_CJ_PACKED* d_nbnxm_cjpacked = NULL;
    LJ_CLUSTERED_GMXPACKED_SCI* d_gmxpacked_sci = NULL;
    LJ_CLUSTERED_GMXPACKED_CJ* d_gmxpacked_cjpacked = NULL;
    LJ_CLUSTERED_GMXPACKED_EXCLUSION* d_gmxpacked_exclusions = NULL;
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE*
        d_gmxpacked_record_stream_sources = NULL;
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE*
        d_gmxpacked_record_stream_aggregates = NULL;
    int* d_gmxpacked_incremental_source_offsets_by_candidate = NULL;
    float* d_gmxpacked_incremental_source_valid_cutoff_by_candidate = NULL;
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE*
        d_gmxpacked_incremental_record_stream_sources = NULL;
    int* d_gmxpacked_incremental_replacement_source_counts_by_candidate = NULL;
    int* d_gmxpacked_incremental_replacement_source_offsets_by_candidate = NULL;
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE*
        d_gmxpacked_incremental_replacement_sources = NULL;
    int* d_gmxpacked_incremental_dirty_atoms = NULL;
    int* d_gmxpacked_incremental_dirty_clusters = NULL;
    int* d_gmxpacked_incremental_dirty_superclusters = NULL;
    int* d_gmxpacked_incremental_dirty_i_candidate_sci = NULL;
    int* d_gmxpacked_incremental_dirty_candidate_sci = NULL;
    int* d_cjpacked_sort_indices = NULL;
    LJ_CLUSTERED_CJ_PACKED* d_cjpacked_sort_buffer = NULL;
    LJ_CLUSTERED_J_ENTRY* d_j_entries = NULL;
    LJ_CLUSTERED_J_ENTRY* d_j_entry_buffer = NULL;
    int* d_jentry_counts = NULL;
    int* d_jentry_offsets = NULL;
    int* d_jentry_indices = NULL;
    LJ_CLUSTERED_WARP_J_RECORD* d_nbnxm_warp_j_records = NULL;
    int* d_forceonly_warp_record_counts = NULL;
    int* d_forceonly_warp_record_offsets = NULL;
    LJ_CLUSTERED_WARP_J_RECORD* d_forceonly_warp_j_records = NULL;
    uint64_t* d_pair_shift_bits = NULL;
    int* d_gmxpacked_pair_shift_sci_only_flag = NULL;
    int* d_gmxpacked_pair_shift_sci_safe_flags = NULL;
    int* d_gmxpacked_pair_shift_safe_sci_count = NULL;
    unsigned char* d_sort_key_buffer = NULL;
    unsigned char* d_sort_value_buffer = NULL;
    void* d_sort_temp_storage = NULL;
    void* d_reduce_temp_storage = NULL;
    void* d_scan_temp_storage = NULL;
    int* d_scan_total = NULL;
    size_t sort_key_buffer_bytes = 0;
    size_t sort_value_buffer_bytes = 0;
    size_t sort_temp_storage_bytes = 0;
    size_t reduce_temp_storage_bytes = 0;
    size_t scan_temp_storage_bytes = 0;

    const int* d_excluded_list_start = NULL;
    const int* d_excluded_list = NULL;
    const int* d_excluded_numbers = NULL;
    LJ_CORNERSTONE_STATE* cornerstone_state = NULL;

    void Initial(CONTROLLER* controller, const char* module_name,
                 bool ordered_layout_enabled = false);
    void Enable_Clustered_Spatial_Service();
    void Refresh_Metadata(int local_atom_numbers, int direct_local_atom_numbers,
                          int ghost_numbers,
                          const int* d_atom_local,
                          const int* d_excluded_list_start,
                          const int* d_excluded_list,
                          const int* d_excluded_numbers);
    void Build(const VECTOR* crd, LTMatrix3 cell, LTMatrix3 rcell,
               float cutoff, bool need_virial,
               bool prefer_full_warp_record, bool need_gmxpacked_payload,
               bool need_aux_clustered_metadata,
               bool runtime_gmxpacked_direct_requested);
    void Retire_Spatial_Provider_Lifetime()
    {
        gmxpacked_endpoint_incidence_ready = false;
        gmxpacked_endpoint_incidence_provider_incarnation = -1;
        gmxpacked_endpoint_incidence_payload_generation = -1;
        gmxpacked_endpoint_incidence_sci_numbers = 0;
        gmxpacked_endpoint_incidence_cjpacked_numbers = 0;
        gmxpacked_endpoint_incidence_super_cluster_numbers = 0;
        gmxpacked_endpoint_incidence_reference_numbers = 0;
        gmxpacked_endpoint_incidence_offset_tail = 0;
        provider_incarnation += 1;
        spatial_view_lease_epoch += 1;
        native_payload_generation += 1;
        gmxpacked_compact_payload_generation += 1;
        geometry_generation += 1;
    }
    void Clear();

    bool Use_Clustered_Direct() const { return enabled; }
};

struct LJ_CLUSTERED_DIRECT_CACHE
{
    bool initialized = false;
    LJ_CLUSTER_LAYOUT layout;
    TIME_RECORDER* payload_gather_time_recorder = NULL;
    TIME_RECORDER* direct_kernel_time_recorder = NULL;
    TIME_RECORDER* gmxpacked_force_scratch_memset_time_recorder = NULL;
    TIME_RECORDER* gmxpacked_kernel_launch_time_recorder = NULL;
    TIME_RECORDER* gmxpacked_sorted_force_scatter_time_recorder = NULL;
    TIME_RECORDER* gmxpacked_full_output_snapshot_time_recorder = NULL;
    int coordinate_gather_step = -1;
    int coordinate_gather_count_this_step = 0;
    long long coordinate_gather_count_total = 0;

    int sorted_atom_ids_capacity = 0;
    int* d_sorted_atom_ids = NULL;
    int sorted_xq_capacity = 0;
    float4* d_sorted_xq = NULL;
    int sorted_lj_type_capacity = 0;
    int* d_sorted_lj_type = NULL;
    int sorted_cluster_ids_capacity = 0;
    int* d_sorted_cluster_ids = NULL;
    int sorted_lj_comb_capacity = 0;
    float2* d_sorted_lj_comb = NULL;
    int sorted_frc_capacity = 0;
    VECTOR* d_sorted_frc = NULL;
    int sorted_soft_crd_capacity = 0;
    VECTOR_LJ_SOFT_TYPE* d_sorted_soft_crd = NULL;

    void Initial(CONTROLLER* controller, const char* module_name,
                 bool ordered_layout_enabled = false,
                 bool clustered_spatial_service_requested = false);
    void Refresh_Metadata(int local_atom_numbers, int direct_local_atom_numbers,
                          int ghost_numbers,
                          const int* d_atom_local,
                          const int* d_excluded_list_start,
                          const int* d_excluded_list,
                          const int* d_excluded_numbers);
    void Build(const VECTOR* crd, LTMatrix3 cell, LTMatrix3 rcell,
               float cutoff, bool need_virial,
               bool prefer_full_warp_record, bool need_gmxpacked_payload,
               bool need_aux_clustered_metadata,
               bool runtime_gmxpacked_direct_requested);
    void Gather_Plain(const VECTOR* crd, const float* charge,
                      const VECTOR_LJ* lj_type_src, LTMatrix3 cell,
                      LTMatrix3 rcell, const float2* lj_ab_packed = NULL);
    void Gather_Plain(const VECTOR_LJ* src, LTMatrix3 cell, LTMatrix3 rcell,
                      const float2* lj_ab_packed = NULL);
    void Gather_Soft_Core(const VECTOR_LJ_SOFT_TYPE* src, LTMatrix3 cell,
                          LTMatrix3 rcell);
    bool Coordinate_Gather_Ready_For_Current_Step() const;
    void Clear();

    bool Use_Clustered_Direct() const { return layout.Use_Clustered_Direct(); }
};

LJ_CLUSTERED_DIRECT_CACHE* Acquire_Shared_LJ_Clustered_Direct_Cache(
    CONTROLLER* controller, const char* module_name,
    bool ordered_layout_enabled = false,
    bool clustered_spatial_service_requested = false);
void Release_Shared_LJ_Clustered_Direct_Cache();

bool Make_Clustered_Spatial_View_From_LJ_Cache(
    const LJ_CLUSTERED_DIRECT_CACHE* cache, CLUSTERED_SPATIAL_VIEW* view,
    const char** failure_reason = NULL);

void Compare_Gmxpacked_Record_Stream_Focus_Pair_Forces(
    LJ_CLUSTERED_DIRECT_CACHE* cache, const float2* d_LJ_AB_packed,
    size_t lj_param_numbers, float cutoff, float pme_beta, LTMatrix3 cell,
    LTMatrix3 rcell);

bool Ensure_Legacy_Neighbor_View_From_Clustered_Payload(
    LJ_CLUSTERED_DIRECT_CACHE* cache,
    const LJ_CLUSTERED_LEGACY_NEIGHBOR_VIEW_REQUEST& request,
    ATOM_GROUP* d_legacy_nl, int max_neighbor_numbers,
    int* d_neighbor_list_overflow, const char** fallback_reason);
