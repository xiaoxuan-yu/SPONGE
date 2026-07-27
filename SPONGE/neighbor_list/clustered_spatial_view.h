#pragma once

#include <cstdint>
#include <vector>

#include "../common.h"

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

struct CLUSTERED_GMXPACKED_PAIR_SHIFT_CACHE_KEY
{
    long long gmxpacked_payload_generation = -1;
    long long geometry_generation = -1;
    int sci_numbers = 0;
    int cjpacked_numbers = 0;
    int exclusion_numbers = 0;
    LTMatrix3 rcell = {};
};

inline bool Clustered_Gmxpacked_Pair_Shift_Cache_Key_Matches(
    const CLUSTERED_GMXPACKED_PAIR_SHIFT_CACHE_KEY& cached,
    const CLUSTERED_GMXPACKED_PAIR_SHIFT_CACHE_KEY& current)
{
    return cached.gmxpacked_payload_generation >= 0 &&
           cached.gmxpacked_payload_generation ==
               current.gmxpacked_payload_generation &&
           cached.geometry_generation >= 0 &&
           cached.geometry_generation == current.geometry_generation &&
           cached.sci_numbers == current.sci_numbers &&
           cached.cjpacked_numbers == current.cjpacked_numbers &&
           cached.exclusion_numbers == current.exclusion_numbers &&
           cached.rcell.a11 == current.rcell.a11 &&
           cached.rcell.a21 == current.rcell.a21 &&
           cached.rcell.a22 == current.rcell.a22 &&
           cached.rcell.a31 == current.rcell.a31 &&
           cached.rcell.a32 == current.rcell.a32 &&
           cached.rcell.a33 == current.rcell.a33;
}

inline bool Clustered_Gmxpacked_Pair_Shift_Metadata_Should_Refresh(
    bool cache_enabled, bool pair_shift_storage_ready, bool metadata_ready,
    const CLUSTERED_GMXPACKED_PAIR_SHIFT_CACHE_KEY& cached,
    const CLUSTERED_GMXPACKED_PAIR_SHIFT_CACHE_KEY& current)
{
    return !cache_enabled || !pair_shift_storage_ready || !metadata_ready ||
           current.sci_numbers <= 0 || current.cjpacked_numbers <= 0 ||
           current.exclusion_numbers <= 0 ||
           !Clustered_Gmxpacked_Pair_Shift_Cache_Key_Matches(cached, current);
}

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

__host__ __device__ __forceinline__ int Clustered_Invert_Shift_Id(
    int shift_id)
{
    if (shift_id < 0 || shift_id >= kClusteredShiftCount)
    {
        return -1;
    }
    const int sx = shift_id / 9;
    const int sy = (shift_id % 9) / 3;
    const int sz = shift_id % 3;
    return (2 - sx) * 9 + (2 - sy) * 3 + (2 - sz);
}

struct CLUSTERED_SCI
{
    int supercluster_id = 0;
    int shift_id = kClusteredCentralShiftId;
    int cjpacked_begin = 0;
    int cjpacked_end = 0;
};

struct CLUSTERED_IMEI
{
    unsigned int imask = 0u;
    int excl_ind[kClusteredJGroupSize * kClusteredSuperClusterClusters];
};

struct CLUSTERED_CJ_PACKED
{
    int cj[kClusteredJGroupSize];
    CLUSTERED_IMEI imei[kClusteredWarpSplitCount];
};

struct CLUSTERED_GMXPACKED_SCI
{
    int supercluster_id = 0;
    int shift_id = kClusteredCentralShiftId;
    int cjpacked_begin = 0;
    int cjpacked_end = 0;
};

struct CLUSTERED_GMXPACKED_SPLIT
{
    unsigned int imask = 0u;
    int exclusion_index = 0;
};

struct CLUSTERED_GMXPACKED_CJ
{
    int cj[kClusteredJGroupSize] = { -1, -1, -1, -1 };
    CLUSTERED_GMXPACKED_SPLIT split[kClusteredWarpSplitCount] = {};
};
static_assert(sizeof(CLUSTERED_GMXPACKED_CJ) == 32,
              "Unexpected clustered gmxpacked cj size.");

struct CLUSTERED_GMXPACKED_EXCLUSION
{
    unsigned int pair[kClusteredGmxpackedExclusionPairCount] = {};
};
static_assert(sizeof(CLUSTERED_GMXPACKED_EXCLUSION) ==
                  sizeof(unsigned int) * kClusteredGmxpackedExclusionPairCount,
              "Unexpected clustered gmxpacked exclusion size.");

enum class CLUSTERED_ENDPOINT_ORIENTATION : unsigned char
{
    NATIVE_I = 0,
    TRANSPOSED_J = 1
};

// One structural reference addresses an authoritative CJ/jm tile. The
// i_cluster_mask retains the fixed-width cluster mask; it is never expanded
// into accepted atom-pair rows.
struct CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE
{
    int sci_id = -1;
    int cjpacked_id = -1;
    unsigned int i_cluster_mask = 0u;
    unsigned char jm = 0u;
    CLUSTERED_ENDPOINT_ORIENTATION orientation =
        CLUSTERED_ENDPOINT_ORIENTATION::NATIVE_I;
    unsigned short reserved = 0u;
};
static_assert(sizeof(CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE) == 16,
              "Unexpected clustered endpoint-reference size.");

struct CLUSTERED_GMXPACKED_ENDPOINT_INCIDENCE_HOST
{
    bool ready = false;
    long long provider_incarnation = -1;
    long long gmxpacked_payload_generation = -1;
    int super_cluster_numbers = 0;
    std::vector<int> offsets;
    std::vector<CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE> references;

    void Clear()
    {
        ready = false;
        provider_incarnation = -1;
        gmxpacked_payload_generation = -1;
        super_cluster_numbers = 0;
        offsets.clear();
        references.clear();
    }
};

bool Clustered_Build_Gmxpacked_Endpoint_Incidence_Host(
    long long provider_incarnation, long long gmxpacked_payload_generation,
    int cluster_numbers, int super_cluster_numbers,
    const int* super_cluster_offsets, int sci_numbers,
    const CLUSTERED_GMXPACKED_SCI* sci_entries, int cjpacked_numbers,
    const CLUSTERED_GMXPACKED_CJ* cjpacked_entries,
    CLUSTERED_GMXPACKED_ENDPOINT_INCIDENCE_HOST* incidence,
    const char** failure_reason = nullptr);

struct CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE
{
    int sci_id = -1;
    int shift_id = kClusteredCentralShiftId;
    int supercluster_id = -1;
    int cluster_j = -1;
    int split_id = 0;
    unsigned int imask = 0u;
    unsigned int valid_mask_j = 0u;
    unsigned int local_mask_j = 0u;
    unsigned int pair_exclusion_words[kClusteredGmxpackedExclusionPairCount] =
        {};
    int source_order = 0;
};

struct CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE
{
    int sci_id = -1;
    int shift_id = kClusteredCentralShiftId;
    int supercluster_id = -1;
    int cluster_j = -1;
    unsigned int split_imask[kClusteredWarpSplitCount] = {};
    unsigned int valid_mask_j = 0u;
    unsigned int local_mask_j = 0u;
    unsigned int pair_exclusion_words[kClusteredWarpSplitCount]
                                      [kClusteredGmxpackedExclusionPairCount] =
                                          {};
    int source_order_begin = 0;
    int source_order_end = 0;
};

struct CLUSTERED_J_ENTRY
{
    int supercluster_id = 0;
    int shift_id = kClusteredCentralShiftId;
    int cluster_j = -1;
    unsigned int imask[kClusteredWarpSplitCount] = {};
    int excl_ind[kClusteredWarpSplitCount * kClusteredSuperClusterClusters] =
        {};
};

constexpr int kClusteredWarpRecordPairExclBytes =
    kClusteredSplitJClusterSize * kClusteredClusterSize;

struct CLUSTERED_WARP_J_RECORD
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
static_assert(sizeof(CLUSTERED_WARP_J_RECORD) == 48,
              "Unexpected clustered warp-record size.");

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

__host__ __device__ __forceinline__ unsigned int Clustered_Jm_Imask(
    const CLUSTERED_IMEI& imei, int jm)
{
    return (imei.imask >> Clustered_Jm_Imask_Shift(jm)) &
           ((1u << kClusteredSuperClusterClusters) - 1u);
}

__host__ __device__ __forceinline__ int& Clustered_Exclusion_Index_Ref(
    CLUSTERED_IMEI& imei, int jm, int i_local)
{
    return imei.excl_ind[jm * kClusteredSuperClusterClusters + i_local];
}

__host__ __device__ __forceinline__ int Clustered_Exclusion_Index(
    const CLUSTERED_IMEI& imei, int jm, int i_local)
{
    return imei.excl_ind[jm * kClusteredSuperClusterClusters + i_local];
}

__host__ __device__ __forceinline__ unsigned int
Clustered_Combined_Imask(const CLUSTERED_CJ_PACKED& packed)
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

// The caller guarantees that the I lane is locally owned. A ghost J lane is
// always consumed on this rank; two local lanes use one lexicographic
// cluster/lane orientation so mixed local/ghost clusters cannot be culled as a
// whole without losing local-ghost pairs.
__host__ __device__ __forceinline__ bool
Clustered_Local_I_Owns_Pair(int cluster_i, int i_lane, int cluster_j,
                            int j_lane, bool local_j)
{
    return !local_j || cluster_i < cluster_j ||
           (cluster_i == cluster_j && i_lane < j_lane);
}

__host__ __device__ __forceinline__ int Clustered_First_Exclusion_Index(
    const CLUSTERED_CJ_PACKED& packed, int jm, int i_local)
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

enum class CLUSTERED_SPATIAL_BACKEND : int
{
    CPU = 0,
    CUDA = 1,
    HIP = 2
};

enum class CLUSTERED_SPATIAL_READINESS_SCOPE : int
{
    HOST_COMPLETE = 0,
    PRODUCER_STREAM_ORDERED = 1
};

struct CLUSTERED_SPATIAL_VIEW
{
    bool ready = false;
    CLUSTERED_SPATIAL_BACKEND backend = CLUSTERED_SPATIAL_BACKEND::CPU;
    CLUSTERED_SPATIAL_READINESS_SCOPE readiness_scope =
        CLUSTERED_SPATIAL_READINESS_SCOPE::HOST_COMPLETE;
    const void* producer_stream = nullptr;
    long long provider_incarnation = -1;
    long long lease_epoch = -1;
    long long native_payload_generation = -1;
    long long gmxpacked_payload_generation = -1;
    long long source_generation = -1;
    long long geometry_generation = -1;

    int cluster_size = 0;
    int super_cluster_clusters = 0;
    int local_atom_numbers = 0;
    int direct_local_atom_numbers = 0;
    int ghost_numbers = 0;
    int total_atom_numbers = 0;
    int padded_total_atom_numbers = 0;
    int cluster_numbers = 0;
    int super_cluster_numbers = 0;
    int sci_numbers = 0;
    int cjpacked_numbers = 0;
    int exclusion_pool_numbers = 0;
    int gmxpacked_sci_numbers = 0;
    int gmxpacked_cjpacked_numbers = 0;
    int gmxpacked_exclusion_numbers = 0;

    float cached_cutoff = -1.0f;
    float rebuild_skin = 0.0f;
    bool native_grouped_sci_ready = false;
    bool gmxpacked_grouped_sci_ready = false;
    bool gmxpacked_endpoint_incidence_ready = false;
    long long endpoint_incidence_provider_incarnation = -1;
    long long endpoint_incidence_payload_generation = -1;
    int endpoint_incidence_sci_numbers = 0;
    int endpoint_incidence_cjpacked_numbers = 0;
    int endpoint_incidence_super_cluster_numbers = 0;
    int endpoint_incidence_reference_numbers = 0;
    int endpoint_incidence_offset_tail = 0;
    bool pair_shift_metadata_ready = false;
    long long pair_shift_payload_generation = -1;
    long long pair_shift_geometry_generation = -1;
    int pair_shift_sci_numbers = 0;
    int pair_shift_cjpacked_numbers = 0;
    int pair_shift_exclusion_numbers = 0;
    LTMatrix3 pair_shift_rcell = {};

    const int* sort_permutation = nullptr;
    const int* cluster_offsets = nullptr;
    const unsigned int* cluster_valid_masks = nullptr;
    const unsigned int* cluster_local_masks = nullptr;
    const VECTOR* cluster_centers = nullptr;
    const VECTOR* cluster_extents = nullptr;
    const int* super_cluster_offsets = nullptr;
    const int* native_grouped_sci_offsets = nullptr;
    const int* native_grouped_sci_ids = nullptr;
    const int* gmxpacked_grouped_sci_offsets = nullptr;
    const int* gmxpacked_grouped_sci_ids = nullptr;
    const int* gmxpacked_endpoint_incidence_offsets = nullptr;
    const CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE*
        gmxpacked_endpoint_incidence_references = nullptr;

    const CLUSTERED_SCI* sci = nullptr;
    const CLUSTERED_CJ_PACKED* cjpacked = nullptr;
    const unsigned long long* exclusion_mask_pool = nullptr;
    const CLUSTERED_GMXPACKED_SCI* gmxpacked_sci = nullptr;
    const CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked = nullptr;
    const CLUSTERED_GMXPACKED_EXCLUSION* gmxpacked_exclusions = nullptr;
    const uint64_t* pair_shift_bits = nullptr;
    const int* gmxpacked_sci_shift_safe_flags = nullptr;
};

struct CLUSTERED_GROUPED_SCI_RANGE
{
    int begin = 0;
    int end = 0;
};

struct CLUSTERED_ENDPOINT_INCIDENCE_RANGE
{
    int begin = 0;
    int end = 0;
};

struct CLUSTERED_GMXPACKED_CENTER_CURSOR
{
    int center_cluster = -1;
    int center_supercluster = -1;
    int center_i_local = -1;
    int next_reference = 0;
    int end_reference = 0;
};

struct CLUSTERED_GMXPACKED_CENTER_TILE
{
    int sci_id = -1;
    int cjpacked_id = -1;
    int center_cluster = -1;
    int neighbor_cluster_base = -1;
    unsigned int neighbor_cluster_mask = 0u;
    unsigned char jm = 0u;
    unsigned char original_i_local = 0u;
    CLUSTERED_ENDPOINT_ORIENTATION orientation =
        CLUSTERED_ENDPOINT_ORIENTATION::NATIVE_I;
    unsigned char reserved = 0u;
};

// A grouped range is an O(SCI) structural index. It groups every SCI for one
// center supercluster, but intentionally does not imply one common shift for
// the range.
__host__ __device__ __forceinline__ CLUSTERED_GROUPED_SCI_RANGE
Clustered_Native_Grouped_Sci_Range(const CLUSTERED_SPATIAL_VIEW& view,
                                   int supercluster_id)
{
    if (!view.native_grouped_sci_ready ||
        view.native_grouped_sci_offsets == nullptr ||
        view.native_grouped_sci_ids == nullptr || supercluster_id < 0 ||
        supercluster_id >= view.super_cluster_numbers)
    {
        return {};
    }
    return {view.native_grouped_sci_offsets[supercluster_id],
            view.native_grouped_sci_offsets[supercluster_id + 1]};
}

__host__ __device__ __forceinline__ int Clustered_Native_Grouped_Sci_Id(
    const CLUSTERED_SPATIAL_VIEW& view, int grouped_index)
{
    if (!view.native_grouped_sci_ready ||
        view.native_grouped_sci_ids == nullptr ||
        grouped_index < 0 || grouped_index >= view.sci_numbers)
    {
        return -1;
    }
    return view.native_grouped_sci_ids[grouped_index];
}

__host__ __device__ __forceinline__ CLUSTERED_GROUPED_SCI_RANGE
Clustered_Gmxpacked_Grouped_Sci_Range(const CLUSTERED_SPATIAL_VIEW& view,
                                      int supercluster_id)
{
    if (!view.gmxpacked_grouped_sci_ready ||
        view.gmxpacked_grouped_sci_offsets == nullptr ||
        view.gmxpacked_grouped_sci_ids == nullptr || supercluster_id < 0 ||
        supercluster_id >= view.super_cluster_numbers)
    {
        return {};
    }
    return {view.gmxpacked_grouped_sci_offsets[supercluster_id],
            view.gmxpacked_grouped_sci_offsets[supercluster_id + 1]};
}

__host__ __device__ __forceinline__ int Clustered_Gmxpacked_Grouped_Sci_Id(
    const CLUSTERED_SPATIAL_VIEW& view, int grouped_index)
{
    if (!view.gmxpacked_grouped_sci_ready ||
        view.gmxpacked_grouped_sci_ids == nullptr || grouped_index < 0 ||
        grouped_index >= view.gmxpacked_sci_numbers)
    {
        return -1;
    }
    return view.gmxpacked_grouped_sci_ids[grouped_index];
}

__host__ __device__ __forceinline__ CLUSTERED_ENDPOINT_INCIDENCE_RANGE
Clustered_Gmxpacked_Endpoint_Incidence_Range(
    const CLUSTERED_SPATIAL_VIEW& view, int supercluster_id)
{
    if (!view.gmxpacked_endpoint_incidence_ready ||
        view.gmxpacked_endpoint_incidence_offsets == nullptr ||
        view.gmxpacked_endpoint_incidence_references == nullptr ||
        supercluster_id < 0 ||
        supercluster_id >= view.super_cluster_numbers)
    {
        return {};
    }
    const int begin =
        view.gmxpacked_endpoint_incidence_offsets[supercluster_id];
    const int end =
        view.gmxpacked_endpoint_incidence_offsets[supercluster_id + 1];
    if (begin < 0 || end < begin ||
        end > view.endpoint_incidence_reference_numbers)
    {
        return {};
    }
    return {begin, end};
}

__host__ __device__ __forceinline__
    const CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE*
    Clustered_Gmxpacked_Endpoint_Incidence_Reference(
        const CLUSTERED_SPATIAL_VIEW& view, int reference_index)
{
    if (!view.gmxpacked_endpoint_incidence_ready ||
        view.gmxpacked_endpoint_incidence_references == nullptr ||
        reference_index < 0 ||
        reference_index >= view.endpoint_incidence_reference_numbers)
    {
        return nullptr;
    }
    return view.gmxpacked_endpoint_incidence_references + reference_index;
}

__host__ __device__ __forceinline__ bool
Clustered_Gmxpacked_Center_Cursor_Begin(
    const CLUSTERED_SPATIAL_VIEW& view, int center_cluster,
    CLUSTERED_GMXPACKED_CENTER_CURSOR* cursor)
{
    if (cursor == nullptr)
    {
        return false;
    }
    *cursor = {};
    if (!view.gmxpacked_endpoint_incidence_ready ||
        view.super_cluster_offsets == nullptr || center_cluster < 0 ||
        center_cluster >= view.cluster_numbers)
    {
        return false;
    }
    int low = 0;
    int high = view.super_cluster_numbers;
    while (low < high)
    {
        const int middle = low + (high - low) / 2;
        if (view.super_cluster_offsets[middle + 1] <= center_cluster)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    if (low < 0 || low >= view.super_cluster_numbers ||
        center_cluster < view.super_cluster_offsets[low] ||
        center_cluster >= view.super_cluster_offsets[low + 1])
    {
        return false;
    }
    const CLUSTERED_ENDPOINT_INCIDENCE_RANGE range =
        Clustered_Gmxpacked_Endpoint_Incidence_Range(view, low);
    if (range.end < range.begin)
    {
        return false;
    }
    cursor->center_cluster = center_cluster;
    cursor->center_supercluster = low;
    cursor->center_i_local =
        center_cluster - view.super_cluster_offsets[low];
    cursor->next_reference = range.begin;
    cursor->end_reference = range.end;
    return true;
}

__host__ __device__ __forceinline__ bool
Clustered_Gmxpacked_Center_Cursor_Next(
    const CLUSTERED_SPATIAL_VIEW& view,
    CLUSTERED_GMXPACKED_CENTER_CURSOR* cursor,
    CLUSTERED_GMXPACKED_CENTER_TILE* tile)
{
    if (cursor == nullptr || tile == nullptr ||
        view.gmxpacked_sci == nullptr ||
        view.gmxpacked_cjpacked == nullptr ||
        view.super_cluster_offsets == nullptr)
    {
        return false;
    }
    *tile = {};
    while (cursor->next_reference < cursor->end_reference)
    {
        const auto* reference =
            Clustered_Gmxpacked_Endpoint_Incidence_Reference(
                view, cursor->next_reference);
        cursor->next_reference += 1;
        if (reference == nullptr || reference->sci_id < 0 ||
            reference->sci_id >= view.gmxpacked_sci_numbers ||
            reference->cjpacked_id < 0 ||
            reference->cjpacked_id >=
                view.gmxpacked_cjpacked_numbers ||
            reference->jm >= kClusteredJGroupSize)
        {
            continue;
        }
        const CLUSTERED_GMXPACKED_SCI& sci =
            view.gmxpacked_sci[reference->sci_id];
        const CLUSTERED_GMXPACKED_CJ& packed =
            view.gmxpacked_cjpacked[reference->cjpacked_id];
        if (sci.supercluster_id < 0 ||
            sci.supercluster_id >= view.super_cluster_numbers)
        {
            continue;
        }
        if (reference->orientation ==
            CLUSTERED_ENDPOINT_ORIENTATION::NATIVE_I)
        {
            if (sci.supercluster_id !=
                    cursor->center_supercluster ||
                cursor->center_i_local < 0 ||
                cursor->center_i_local >=
                    kClusteredSuperClusterClusters ||
                (reference->i_cluster_mask &
                 (1u << static_cast<unsigned int>(
                      cursor->center_i_local))) == 0u)
            {
                continue;
            }
            const int cluster_j = packed.cj[reference->jm];
            if (cluster_j < 0 || cluster_j >= view.cluster_numbers)
            {
                continue;
            }
            tile->sci_id = reference->sci_id;
            tile->cjpacked_id = reference->cjpacked_id;
            tile->center_cluster = cursor->center_cluster;
            tile->neighbor_cluster_base = cluster_j;
            tile->neighbor_cluster_mask = 1u;
            tile->jm = reference->jm;
            tile->original_i_local =
                static_cast<unsigned char>(
                    cursor->center_i_local);
            tile->orientation =
                CLUSTERED_ENDPOINT_ORIENTATION::NATIVE_I;
            return true;
        }
        if (reference->orientation !=
                CLUSTERED_ENDPOINT_ORIENTATION::TRANSPOSED_J ||
            packed.cj[reference->jm] != cursor->center_cluster)
        {
            continue;
        }
        tile->sci_id = reference->sci_id;
        tile->cjpacked_id = reference->cjpacked_id;
        tile->center_cluster = cursor->center_cluster;
        tile->neighbor_cluster_base =
            view.super_cluster_offsets[sci.supercluster_id];
        tile->neighbor_cluster_mask =
            reference->i_cluster_mask;
        tile->jm = reference->jm;
        tile->orientation =
            CLUSTERED_ENDPOINT_ORIENTATION::TRANSPOSED_J;
        return true;
    }
    return false;
}

// Active-view compaction can require a different periodic image for each
// i-cluster lane in a CJ record. Consumers that request pair-shift metadata
// must use this value rather than treating SCI::shift_id as pair identity.
__host__ __device__ __forceinline__ int Clustered_Gmxpacked_Pair_Shift_Id(
    const CLUSTERED_SPATIAL_VIEW& view,
    const CLUSTERED_GMXPACKED_SCI& sci_entry, int packed_index, int jm,
    int i_local)
{
    if (!view.pair_shift_metadata_ready || view.pair_shift_bits == nullptr)
    {
        return sci_entry.shift_id;
    }
    const uint64_t shift_bits =
        view.pair_shift_bits[packed_index * kClusteredJGroupSize + jm];
    return Clustered_Get_Pair_Shift_Id(shift_bits, i_local);
}

__host__ __device__ __forceinline__ int
Clustered_Gmxpacked_Center_Tile_Pair_Shift_Id(
    const CLUSTERED_SPATIAL_VIEW& view,
    const CLUSTERED_GMXPACKED_CENTER_TILE& tile,
    int neighbor_cluster_offset)
{
    if (tile.sci_id < 0 ||
        tile.sci_id >= view.gmxpacked_sci_numbers ||
        tile.cjpacked_id < 0 ||
        tile.cjpacked_id >= view.gmxpacked_cjpacked_numbers ||
        tile.jm >= kClusteredJGroupSize)
    {
        return -1;
    }
    const CLUSTERED_GMXPACKED_SCI& sci =
        view.gmxpacked_sci[tile.sci_id];
    int original_i_local = tile.original_i_local;
    if (tile.orientation ==
        CLUSTERED_ENDPOINT_ORIENTATION::TRANSPOSED_J)
    {
        original_i_local = neighbor_cluster_offset;
        if (original_i_local < 0 ||
            original_i_local >= kClusteredSuperClusterClusters ||
            (tile.neighbor_cluster_mask &
             (1u << static_cast<unsigned int>(
                  original_i_local))) == 0u)
        {
            return -1;
        }
    }
    const int shift_id = Clustered_Gmxpacked_Pair_Shift_Id(
        view, sci, tile.cjpacked_id, tile.jm, original_i_local);
    return tile.orientation ==
                   CLUSTERED_ENDPOINT_ORIENTATION::TRANSPOSED_J
               ? Clustered_Invert_Shift_Id(shift_id)
               : shift_id;
}

struct CLUSTERED_SPATIAL_VIEW_REQUIREMENTS
{
    int local_atom_numbers = -1;
    int ghost_numbers = -1;
    float cutoff = -1.0f;
    long long provider_incarnation = -1;
    long long lease_epoch = -1;
    long long native_payload_generation = -1;
    long long gmxpacked_payload_generation = -1;
    long long source_generation = -1;
    long long geometry_generation = -1;
    bool require_backend = false;
    CLUSTERED_SPATIAL_BACKEND backend = CLUSTERED_SPATIAL_BACKEND::CPU;
    bool require_same_producer_stream = false;
    const void* consumer_stream = nullptr;
    bool require_all_local_atoms = true;
    bool require_native_payload = false;
    bool require_gmxpacked_payload = false;
    bool require_native_grouped_sci = false;
    bool require_gmxpacked_grouped_sci = false;
    bool require_gmxpacked_endpoint_incidence = false;
    bool require_pair_shift_metadata = false;
    bool require_pair_shift_rcell = false;
    LTMatrix3 pair_shift_rcell = {};
};

bool Clustered_Validate_Spatial_View(
    const CLUSTERED_SPATIAL_VIEW& view,
    const CLUSTERED_SPATIAL_VIEW_REQUIREMENTS& requirements,
    const char** failure_reason = nullptr);

// Compatibility aliases keep the current LJ kernels byte-for-byte equivalent
// while ownership of the structural contract moves out of the LJ module.
using LJ_CLUSTERED_SCI = CLUSTERED_SCI;
using LJ_CLUSTERED_IMEI = CLUSTERED_IMEI;
using LJ_CLUSTERED_CJ_PACKED = CLUSTERED_CJ_PACKED;
using LJ_CLUSTERED_GMXPACKED_SCI = CLUSTERED_GMXPACKED_SCI;
using LJ_CLUSTERED_GMXPACKED_SPLIT = CLUSTERED_GMXPACKED_SPLIT;
using LJ_CLUSTERED_GMXPACKED_CJ = CLUSTERED_GMXPACKED_CJ;
using LJ_CLUSTERED_GMXPACKED_EXCLUSION = CLUSTERED_GMXPACKED_EXCLUSION;
using LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE =
    CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE;
using LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE =
    CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE;
using LJ_CLUSTERED_J_ENTRY = CLUSTERED_J_ENTRY;
using LJ_CLUSTERED_WARP_J_RECORD = CLUSTERED_WARP_J_RECORD;
