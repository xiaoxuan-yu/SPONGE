#pragma once

#include "../common.h"
#include "../control.h"

struct VECTOR_LJ;
struct VECTOR_LJ_SOFT_TYPE;
struct LJ_CORNERSTONE_STATE;

constexpr int kClusteredClusterSize = 8;
constexpr int kClusteredSuperClusterClusters = 8;
constexpr int kClusteredWarpSplitCount = 2;
constexpr int kClusteredSplitJClusterSize =
    kClusteredClusterSize / kClusteredWarpSplitCount;
constexpr int kClusteredJGroupSize = kClusteredSplitJClusterSize;
constexpr int kClusteredGmxpackedExclusionPairCount =
    kClusteredSplitJClusterSize * kClusteredClusterSize;
constexpr int kClusteredShiftCount = 27;
constexpr int kClusteredCentralShiftId = 13;
constexpr int kClusteredPairShiftBits = 5;
constexpr uint64_t kClusteredPairShiftMask =
    (1ull << kClusteredPairShiftBits) - 1ull;
static_assert(kClusteredSuperClusterClusters * kClusteredPairShiftBits <= 64,
              "Clustered pair shift bit packing exceeds 64 bits.");

__host__ __device__ __forceinline__ VECTOR Clustered_Shift_Vector_From_Id(
    int shift_id, LTMatrix3 cell)
{
    const int sx = shift_id / 9 - 1;
    const int sy = (shift_id % 9) / 3 - 1;
    const int sz = shift_id % 3 - 1;
    return VECTOR(static_cast<float>(sx), static_cast<float>(sy),
                  static_cast<float>(sz)) *
           cell;
}

struct LJ_CLUSTERED_SCI
{
    int supercluster_id = 0;
    int shift_id = kClusteredCentralShiftId;
    int cjpacked_begin = 0;
    int cjpacked_end = 0;
};

struct LJ_CLUSTERED_IMEI
{
    unsigned int imask = 0u;
    int excl_ind[kClusteredJGroupSize * kClusteredSuperClusterClusters];
};

struct LJ_CLUSTERED_CJ_PACKED
{
    int cj[kClusteredJGroupSize];
    LJ_CLUSTERED_IMEI imei[kClusteredWarpSplitCount];
};

struct LJ_CLUSTERED_GMXPACKED_SCI
{
    int supercluster_id = 0;
    int shift_id = kClusteredCentralShiftId;
    int cjpacked_begin = 0;
    int cjpacked_end = 0;
};

struct LJ_CLUSTERED_GMXPACKED_SPLIT
{
    unsigned int imask = 0u;
    int exclusion_index = 0;
};

struct LJ_CLUSTERED_GMXPACKED_CJ
{
    int cj[kClusteredJGroupSize] = { -1, -1, -1, -1 };
    LJ_CLUSTERED_GMXPACKED_SPLIT split[kClusteredWarpSplitCount] = {};
};
static_assert(sizeof(LJ_CLUSTERED_GMXPACKED_CJ) == 32,
              "Unexpected clustered gmxpacked cj size.");

struct LJ_CLUSTERED_GMXPACKED_EXCLUSION
{
    unsigned int pair[kClusteredGmxpackedExclusionPairCount] = {};
};
static_assert(sizeof(LJ_CLUSTERED_GMXPACKED_EXCLUSION) ==
                  sizeof(unsigned int) * kClusteredGmxpackedExclusionPairCount,
              "Unexpected clustered gmxpacked exclusion size.");

// Future production-owned gmxpacked record-stream contract. Hook this stream
// after cornerstone-sorted atoms already have cluster/supercluster/
// candidate-leaf, shift, and exclusion metadata, but before finalized native
// LJ_CLUSTERED_SCI / LJ_CLUSTERED_CJ_PACKED materialization and before
// Build_Gmxpacked_Payload(). This path must consume candidate-stage metadata
// directly and must not accept finalized native CJ payload as input.
struct LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE
{
    int sci_id = -1;
    int shift_id = kClusteredCentralShiftId;
    int supercluster_id = -1;
    int cluster_j = -1;
    int split_id = 0;
    unsigned int imask = 0u;
    unsigned int valid_mask_j = 0u;
    unsigned int local_mask_j = 0u;
    unsigned int pair_exclusion_words[kClusteredGmxpackedExclusionPairCount] = {};
    int source_order = 0;
};

struct LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE
{
    int sci_id = -1;
    int shift_id = kClusteredCentralShiftId;
    int supercluster_id = -1;
    int cluster_j = -1;
    unsigned int split_imask[kClusteredWarpSplitCount] = {};
    unsigned int valid_mask_j = 0u;
    unsigned int local_mask_j = 0u;
    unsigned int pair_exclusion_words[kClusteredWarpSplitCount]
                                      [kClusteredGmxpackedExclusionPairCount] = {};
    int source_order_begin = 0;
    int source_order_end = 0;
};

struct LJ_CLUSTERED_J_ENTRY
{
    int supercluster_id = 0;
    int shift_id = kClusteredCentralShiftId;
    int cluster_j = -1;
    unsigned int imask[kClusteredWarpSplitCount] = {};
    int excl_ind[kClusteredWarpSplitCount * kClusteredSuperClusterClusters] = {};
};

constexpr int kClusteredWarpRecordPairExclBytes =
    kClusteredSplitJClusterSize * kClusteredClusterSize;

struct LJ_CLUSTERED_WARP_J_RECORD
{
    int cluster_j = -1;
    int sorted_j_base = 0;
    int pair_shift_index = -1;
    unsigned char valid_mask = 0u;
    unsigned char imask = 0u;
    unsigned char local_mask = 0u;
    unsigned char j_lane_base = 0u;
    unsigned char pair_excl[kClusteredWarpRecordPairExclBytes] = {};
};
static_assert(sizeof(LJ_CLUSTERED_WARP_J_RECORD) == 48,
              "Unexpected clustered warp-record size.");

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

__host__ __device__ __forceinline__ unsigned int
Clustered_Split_Valid_Mask(int split)
{
    return ((1u << kClusteredSplitJClusterSize) - 1u)
           << (split * kClusteredSplitJClusterSize);
}

__host__ __device__ __forceinline__ unsigned int Clustered_Jm_Imask_Shift(int jm)
{
    return static_cast<unsigned int>(jm * kClusteredSuperClusterClusters);
}

__host__ __device__ __forceinline__ unsigned int
Clustered_Jm_Imask(const LJ_CLUSTERED_IMEI& imei, int jm)
{
    return (imei.imask >> Clustered_Jm_Imask_Shift(jm)) &
           ((1u << kClusteredSuperClusterClusters) - 1u);
}

__host__ __device__ __forceinline__ int& Clustered_Exclusion_Index_Ref(
    LJ_CLUSTERED_IMEI& imei, int jm, int i_local)
{
    return imei.excl_ind[jm * kClusteredSuperClusterClusters + i_local];
}

__host__ __device__ __forceinline__ int Clustered_Exclusion_Index(
    const LJ_CLUSTERED_IMEI& imei, int jm, int i_local)
{
    return imei.excl_ind[jm * kClusteredSuperClusterClusters + i_local];
}

__host__ __device__ __forceinline__ unsigned int
Clustered_Combined_Imask(const LJ_CLUSTERED_CJ_PACKED& packed)
{
    return packed.imei[0].imask | packed.imei[1].imask;
}

__host__ __device__ __forceinline__ void Clustered_Set_Pair_Shift_Id(
    uint64_t* packed_shift_bits, int i_local, int shift_id)
{
    const uint64_t shift_mask =
        kClusteredPairShiftMask
        << (static_cast<uint64_t>(i_local) * kClusteredPairShiftBits);
    *packed_shift_bits =
        (*packed_shift_bits & ~shift_mask) |
        ((static_cast<uint64_t>(shift_id) & kClusteredPairShiftMask)
         << (static_cast<uint64_t>(i_local) * kClusteredPairShiftBits));
}

__host__ __device__ __forceinline__ int Clustered_Get_Pair_Shift_Id(
    uint64_t packed_shift_bits, int i_local)
{
    return static_cast<int>(
        (packed_shift_bits >>
         (static_cast<uint64_t>(i_local) * kClusteredPairShiftBits)) &
        kClusteredPairShiftMask);
}

__host__ __device__ __forceinline__ int Clustered_First_Exclusion_Index(
    const LJ_CLUSTERED_CJ_PACKED& packed, int jm, int i_local)
{
    for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
    {
        const int exclusion_index =
            Clustered_Exclusion_Index(packed.imei[split], jm, i_local);
        if (exclusion_index >= 0)
        {
            return exclusion_index;
        }
    }
    return -1;
}

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
    int gmxpacked_delta_sci_numbers = 0;
    int gmxpacked_delta_cjpacked_numbers = 0;
    int gmxpacked_delta_exclusion_numbers = 0;
    int gmxpacked_delta_split_exclusion_numbers = 0;
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
    int gmxpacked_pair_shift_metadata_sci_numbers = 0;
    int gmxpacked_pair_shift_metadata_cjpacked_numbers = 0;
    int gmxpacked_pair_shift_metadata_exclusion_numbers = 0;
    LTMatrix3 gmxpacked_pair_shift_metadata_rcell = {};
    int candidate_leaf_numbers = 0;
    int candidate_leaf_cluster_stride = 0;
    int exclusion_pool_numbers = 0;
    bool grouped_sci_ready = false;
    bool gmxpacked_incremental_source_offsets_ready = false;
    bool gmxpacked_incremental_source_cache_ready = false;
    bool gmxpacked_outer_source_anchor_ready = false;
    bool stable_target_layout_anchor_ready = false;
    bool gmxpacked_inner_active_anchor_ready = false;
    bool gmxpacked_inner_active_source_imasks_ready = false;
    bool gmxpacked_inner_active_compact_base_imasks_ready = false;
    int gmxpacked_inner_active_source_imask_numbers = 0;
    int gmxpacked_inner_active_compact_base_imask_numbers = 0;
    int gmxpacked_inner_active_source_rows_baseline = 0;
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
    int candidate_leaf_onepass_sci_capacity = 0;
    int candidate_leaf_onepass_rank_capacity = 0;
    int candidate_leaf_onepass_leaf_capacity = 0;
    int candidate_leaf_onepass_cursor_capacity = 0;
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
    int cached_crd_capacity = 0;
    int rebuild_flag_capacity = 0;
    int nbnxm_sci_capacity = 0;
    int nbnxm_cjpacked_capacity = 0;
    int gmxpacked_sci_capacity = 0;
    int gmxpacked_cjpacked_capacity = 0;
    int gmxpacked_exclusion_capacity = 0;
    int gmxpacked_delta_sci_capacity = 0;
    int gmxpacked_delta_cjpacked_capacity = 0;
    int gmxpacked_delta_exclusion_capacity = 0;
    int gmxpacked_delta_pair_shift_capacity = 0;
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
    int gmxpacked_inner_active_compact_base_imask_capacity = 0;
    int gmxpacked_inner_active_compact_delta_source_flag_capacity = 0;

    int* d_sort_permutation = NULL;
    uint64_t* d_sort_keys = NULL;
    VECTOR* d_cached_crd = NULL;
    VECTOR* d_stable_target_layout_crd = NULL;
    VECTOR* d_gmxpacked_outer_source_anchor_crd = NULL;
    VECTOR* d_gmxpacked_inner_active_anchor_crd = NULL;
    unsigned int* d_gmxpacked_inner_active_source_imasks = NULL;
    unsigned int* d_gmxpacked_inner_active_compact_base_imasks = NULL;
    int* d_gmxpacked_inner_active_compact_delta_source_flags = NULL;
    int* d_need_rebuild = NULL;
    const int* d_atom_local = NULL;
    int* d_global_atom_to_molecule = NULL;
    int* d_local_atom_to_molecule = NULL;

    int* d_cluster_offsets = NULL;
    unsigned int* d_cluster_valid_masks = NULL;
    unsigned int* d_cluster_local_masks = NULL;
    VECTOR* d_cluster_centers = NULL;
    VECTOR* d_cluster_extents = NULL;
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
    int* d_candidate_leaf_onepass_sci_ids = NULL;
    int* d_candidate_leaf_onepass_ranks = NULL;
    int* d_candidate_leaf_onepass_leaf_ids = NULL;
    int* d_candidate_leaf_onepass_cursor = NULL;
    unsigned int* d_candidate_leaf_reach_masks = NULL;
    int* d_candidate_sci_offsets = NULL;
    int* d_candidate_shift_ids = NULL;
    int* d_grouped_sci_offsets = NULL;
    int* d_grouped_sci_ids = NULL;
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
    LJ_CLUSTERED_GMXPACKED_SCI* d_gmxpacked_delta_sci = NULL;
    LJ_CLUSTERED_GMXPACKED_CJ* d_gmxpacked_delta_cjpacked = NULL;
    LJ_CLUSTERED_GMXPACKED_EXCLUSION* d_gmxpacked_delta_exclusions = NULL;
    uint64_t* d_gmxpacked_delta_pair_shift_bits = NULL;
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
    bool gmxpacked_sorted_force_clean = false;
    bool gmxpacked_sorted_force_clean_float4 = false;
    int gmxpacked_sorted_force_clean_capacity = 0;
    int coordinate_gather_step = -1;
    int coordinate_gather_count_this_step = 0;
    long long coordinate_gather_count_total = 0;

    int scratch_capacity = 0;
    int* d_sorted_atom_ids = NULL;
    float4* d_sorted_xq = NULL;
    int* d_sorted_lj_type = NULL;
    float2* d_sorted_lj_comb = NULL;
    VECTOR* d_sorted_frc = NULL;
    float4* d_sorted_frc4 = NULL;
    float* d_sorted_frc_x = NULL;
    float* d_sorted_frc_y = NULL;
    float* d_sorted_frc_z = NULL;
    VECTOR_LJ_SOFT_TYPE* d_sorted_soft_crd = NULL;

    void Initial(CONTROLLER* controller, const char* module_name,
                 bool ordered_layout_enabled = false);
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
    void Gather_Soft_Core(const VECTOR_LJ_SOFT_TYPE* src);
    bool Coordinate_Gather_Ready_For_Current_Step() const;
    void Clear();

    bool Use_Clustered_Direct() const { return layout.Use_Clustered_Direct(); }
};

LJ_CLUSTERED_DIRECT_CACHE* Acquire_Shared_LJ_Clustered_Direct_Cache(
    CONTROLLER* controller, const char* module_name,
    bool ordered_layout_enabled = false);
void Release_Shared_LJ_Clustered_Direct_Cache();

void Compare_Gmxpacked_Record_Stream_Focus_Pair_Forces(
    LJ_CLUSTERED_DIRECT_CACHE* cache, const float2* d_LJ_AB_packed,
    size_t lj_param_numbers, float cutoff, float pme_beta, LTMatrix3 cell,
    LTMatrix3 rcell);

bool Ensure_Legacy_Neighbor_View_From_Clustered_Payload(
    LJ_CLUSTERED_DIRECT_CACHE* cache,
    const LJ_CLUSTERED_LEGACY_NEIGHBOR_VIEW_REQUEST& request,
    ATOM_GROUP* d_legacy_nl, int max_neighbor_numbers,
    int* d_neighbor_list_overflow, const char** fallback_reason);
