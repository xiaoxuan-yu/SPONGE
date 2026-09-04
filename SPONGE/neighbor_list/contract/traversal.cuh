#pragma once

#include "types.h"

#define Clustered_Lane_Is_Valid(valid_mask, lane)                          \
    (((valid_mask) & (1u << static_cast<unsigned int>(lane))) != 0u)

#define Clustered_Lane_Is_Local(local_mask, lane)                          \
    (((local_mask) & (1u << static_cast<unsigned int>(lane))) != 0u)

#define Clustered_Lane_Bit_Is_Set(mask, lane_bit)                          \
    (((mask) & (lane_bit)) != 0u)

// The caller guarantees a non-empty clustered layout and a sorted index in
// [cluster_offsets[0], cluster_offsets[cluster_numbers]).
__host__ __device__ __forceinline__ int
Clustered_Find_Cluster_For_Sorted_Index(int sorted_index,
                                        int cluster_numbers,
                                        const int* cluster_offsets)
{
    int low = 0;
    int high = cluster_numbers;
    while (low + 1 < high)
    {
        const int middle = (low + high) >> 1;
        if (cluster_offsets[middle] <= sorted_index)
            low = middle;
        else
            high = middle;
    }
    return low;
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

__host__ __device__ __forceinline__ void Clustered_Set_Pair_Active_I_Masks(
    uint64_t* packed_shift_bits, unsigned int split_0_mask,
    unsigned int split_1_mask)
{
    constexpr uint64_t active_mask =
        (1ull << kClusteredPairActiveMaskBits) - 1ull;
    *packed_shift_bits |=
        kClusteredPairActiveMarker |
        ((static_cast<uint64_t>(split_0_mask) & active_mask)
         << kClusteredPairActiveMaskOffset) |
        ((static_cast<uint64_t>(split_1_mask) & active_mask)
         << (kClusteredPairActiveMaskOffset +
             kClusteredPairActiveMaskBits));
}

__host__ __device__ __forceinline__ unsigned int
Clustered_Get_Pair_Active_I_Mask(uint64_t packed_shift_bits, int split)
{
    if ((packed_shift_bits & kClusteredPairActiveMarker) == 0ull)
    {
        return (1u << kClusteredSuperClusterClusters) - 1u;
    }
    return static_cast<unsigned int>(
        (packed_shift_bits >>
         (kClusteredPairActiveMaskOffset +
          split * kClusteredPairActiveMaskBits)) &
        ((1ull << kClusteredPairActiveMaskBits) - 1ull));
}

// A null exclusion table leaves the split imask unchanged. A non-null table
// must contain every non-zero exclusion index referenced by the payload.
// Keeping this layout-only decode here gives host and device consumers one
// authoritative interpretation of packed exclusions.
__host__ __device__ __forceinline__ unsigned int
Clustered_Gmxpacked_Effective_Imask(
    const CLUSTERED_GMXPACKED_CJ& packed,
    const CLUSTERED_GMXPACKED_EXCLUSION* exclusions, int split,
    int split_j_lane, int i_lane,
    int i_cluster_size = kClusteredClusterSize)
{
    const CLUSTERED_GMXPACKED_SPLIT& split_entry = packed.split[split];
    unsigned int pair_bits = 0xffffffffu;
    if (split_entry.exclusion_index != 0 && exclusions != nullptr)
    {
        pair_bits =
            exclusions[split_entry.exclusion_index]
                .pair[split_j_lane * i_cluster_size + i_lane];
    }
    return split_entry.imask & pair_bits;
}

__host__ __device__ __forceinline__ bool
Clustered_Gmxpacked_I_Entry_Is_Active(unsigned int effective_imask,
                                      uint64_t pair_shift_bits, int split,
                                      int jm, int i_local)
{
    const unsigned int packed_bit =
        1u << (Clustered_Jm_Imask_Shift(jm) +
               static_cast<unsigned int>(i_local));
    const unsigned int i_local_bit =
        1u << static_cast<unsigned int>(i_local);
    return (effective_imask & packed_bit) != 0u &&
           (Clustered_Get_Pair_Active_I_Mask(pair_shift_bits, split) &
            i_local_bit) != 0u;
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
