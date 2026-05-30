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
constexpr int kClusteredShiftCount = 27;
constexpr int kClusteredCentralShiftId = 13;

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
    bool enabled = false;
    bool warn_legacy_ordered_layout = false;
    bool compression_requested = false;
    bool rebuild_dirty = true;
    bool cache_ready = false;
    TIME_RECORDER* payload_build_time_recorder = NULL;

    int cluster_size = 8;
    int super_cluster_clusters = 8;
    int cornerstone_max_depth = 6;
    int cornerstone_leaf_size = 32;
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
    int candidate_leaf_numbers = 0;
    int exclusion_pool_numbers = 0;

    int permutation_capacity = 0;
    int cluster_capacity = 0;
    int leaf_capacity = 0;
    int super_cluster_capacity = 0;
    int sci_capacity = 0;
    int candidate_leaf_capacity = 0;
    int cjpacked_capacity = 0;
    int exclusion_scan_capacity = 0;
    int exclusion_capacity = 0;
    int scan_capacity = 0;
    int sci_shift_capacity = 0;
    int candidate_offset_capacity = 0;
    int cached_crd_capacity = 0;
    int rebuild_flag_capacity = 0;
    int nbnxm_sci_capacity = 0;
    int nbnxm_cjpacked_capacity = 0;

    int* d_sort_permutation = NULL;
    uint64_t* d_sort_keys = NULL;
    VECTOR* d_cached_crd = NULL;
    int* d_need_rebuild = NULL;

    int* d_cluster_offsets = NULL;
    unsigned int* d_cluster_valid_masks = NULL;
    unsigned int* d_cluster_local_masks = NULL;
    VECTOR* d_cluster_centers = NULL;
    float* d_cluster_radii = NULL;

    int* d_leaf_atom_offsets = NULL;
    int* d_leaf_cluster_offsets = NULL;
    int* d_super_cluster_offsets = NULL;
    int* d_super_cluster_has_local = NULL;
    VECTOR* d_super_cluster_centers = NULL;
    VECTOR* d_super_cluster_sizes = NULL;
    int* d_sci_supercluster_ids = NULL;
    int* d_sci_candidate_leaf_counts = NULL;
    int* d_sci_candidate_leaf_offsets = NULL;
    int* d_sci_candidate_leaf_ids = NULL;
    int* d_candidate_sci_offsets = NULL;
    int* d_cjpacked_counts = NULL;
    int* d_exclusion_counts = NULL;
    int* d_exclusion_offsets = NULL;
    int* d_sci_shift_flags = NULL;
    int* d_sci_shift_offsets = NULL;
    int* d_cjpacked_group_offsets = NULL;
    unsigned long long* d_exclusion_mask_pool = NULL;
    LJ_CLUSTERED_SCI* d_nbnxm_sci = NULL;
    LJ_CLUSTERED_CJ_PACKED* d_nbnxm_cjpacked = NULL;

    const int* d_excluded_list_start = NULL;
    const int* d_excluded_list = NULL;
    const int* d_excluded_numbers = NULL;
    LJ_CORNERSTONE_STATE* cornerstone_state = NULL;

    void Initial(CONTROLLER* controller, const char* module_name,
                 bool ordered_layout_enabled = false);
    void Refresh_Metadata(int local_atom_numbers, int direct_local_atom_numbers,
                          int ghost_numbers,
                          const int* d_excluded_list_start,
                          const int* d_excluded_list,
                          const int* d_excluded_numbers);
    void Build(const VECTOR* crd, LTMatrix3 cell, LTMatrix3 rcell, float cutoff);
    void Clear();

    bool Use_Clustered_Direct() const { return enabled; }
};

struct LJ_CLUSTERED_DIRECT_CACHE
{
    bool initialized = false;
    LJ_CLUSTER_LAYOUT layout;
    TIME_RECORDER* payload_gather_time_recorder = NULL;
    TIME_RECORDER* direct_kernel_time_recorder = NULL;

    int scratch_capacity = 0;
    int* d_sorted_atom_ids = NULL;
    float4* d_sorted_xq = NULL;
    int* d_sorted_lj_type = NULL;
    VECTOR_LJ_SOFT_TYPE* d_sorted_soft_crd = NULL;

    void Initial(CONTROLLER* controller, const char* module_name,
                 bool ordered_layout_enabled = false);
    void Refresh_Metadata(int local_atom_numbers, int direct_local_atom_numbers,
                          int ghost_numbers,
                          const int* d_excluded_list_start,
                          const int* d_excluded_list,
                          const int* d_excluded_numbers);
    void Build(const VECTOR* crd, LTMatrix3 cell, LTMatrix3 rcell, float cutoff);
    void Gather_Plain(const VECTOR_LJ* src, LTMatrix3 cell, LTMatrix3 rcell);
    void Gather_Soft_Core(const VECTOR_LJ_SOFT_TYPE* src);
    void Clear();

    bool Use_Clustered_Direct() const { return layout.Use_Clustered_Direct(); }
};

LJ_CLUSTERED_DIRECT_CACHE* Acquire_Shared_LJ_Clustered_Direct_Cache(
    CONTROLLER* controller, const char* module_name,
    bool ordered_layout_enabled = false);
void Release_Shared_LJ_Clustered_Direct_Cache();
