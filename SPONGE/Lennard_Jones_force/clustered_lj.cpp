#include "clustered_lj.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <span>
#include <vector>

#include "LJ_soft_core.h"
#include "Lennard_Jones_force.h"
#include "../third_party/cornerstone_octree/include/cstone/sfc/box.hpp"
#include "../third_party/cornerstone_octree/include/cstone/sfc/common.hpp"
#include "../third_party/cornerstone_octree/include/cstone/sfc/sfc.hpp"
#include "../third_party/cornerstone_octree/include/cstone/tree/csarray.hpp"
#include "../third_party/cornerstone_octree/include/cstone/tree/octree.hpp"
#include "../third_party/cornerstone_octree/include/cstone/traversal/boxoverlap.hpp"
#include "../third_party/cornerstone_octree/include/cstone/traversal/traversal.hpp"

#ifndef USE_CPU
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include "../third_party/cornerstone_octree/include/cstone/tree/octree_gpu.h"
#include "../third_party/cornerstone_octree/include/cstone/tree/update_gpu.cuh"
#endif

struct LJ_CORNERSTONE_STATE
{
#ifndef USE_CPU
    cstone::DeviceVector<uint64_t> leaves;
    cstone::DeviceVector<uint64_t> tmp_leaves;
    cstone::DeviceVector<unsigned> leaf_counts;
    cstone::DeviceVector<cstone::TreeNodeIndex> work_array;
    cstone::OctreeData<uint64_t, cstone::GpuTag> octree;
#else
    std::vector<uint64_t> leaves;
    std::vector<unsigned> leaf_counts;
    cstone::OctreeData<uint64_t, cstone::CpuTag> octree;
#endif
};

namespace
{

using CornerstoneKey = uint64_t;
using CornerstoneNodeIndex = cstone::TreeNodeIndex;
constexpr int kClusteredBuilderBlockSize = 128;
constexpr int kClusteredMaxSuperClusterClusters = kClusteredSuperClusterClusters;
constexpr int kClusteredMaxJGroupSize = kClusteredJGroupSize;

#ifdef USE_CPU
struct HostClusteredJRecord
{
    int cluster_j = -1;
    unsigned int imask = 0u;
    std::array<unsigned long long, kClusteredMaxSuperClusterClusters>
        exclusion_masks = {};

    bool Has_Exclusions() const
    {
        return std::any_of(exclusion_masks.begin(), exclusion_masks.end(),
                           [](unsigned long long mask)
                           { return mask != 0ull; });
    }
};

struct HostClusteredBuildInput
{
    int total_atom_numbers = 0;
    int leaf_numbers = 0;
    int super_cluster_clusters = kClusteredSuperClusterClusters;
    int local_atom_numbers = 0;
    int cluster_size = kClusteredClusterSize;
    int candidate_sci_numbers = 0;
    float cutoff = 0.0f;
    LTMatrix3 cell = {};
    LTMatrix3 rcell = {};

    std::vector<int> permutation;
    std::vector<int> cluster_offsets;
    std::vector<unsigned int> cluster_valid_masks;
    std::vector<unsigned int> cluster_local_masks;
    std::vector<VECTOR> cluster_centers;
    std::vector<float> cluster_radii;
    std::vector<int> leaf_cluster_offsets;
    std::vector<int> super_cluster_offsets;
    std::vector<VECTOR> super_cluster_centers;
    std::vector<int> sci_supercluster_ids;
    std::vector<int> candidate_leaf_offsets;
    std::vector<int> candidate_leaf_ids;
    std::vector<int> excluded_list_start;
    std::vector<int> excluded_list;
    std::vector<int> excluded_numbers;
};
#endif

struct ClusteredRecorderScope
{
    TIME_RECORDER* time_recorder;

    explicit ClusteredRecorderScope(TIME_RECORDER* recorder)
        : time_recorder(recorder)
    {
        if (time_recorder != NULL)
        {
            time_recorder->Start();
        }
    }

    ~ClusteredRecorderScope()
    {
        if (time_recorder != NULL)
        {
            time_recorder->Stop();
        }
    }
};

template <typename T>
static void Reserve_Device_Buffer(int capacity, T** pointer,
                                  int* current_capacity)
{
    if (capacity <= *current_capacity && *pointer != NULL)
    {
        return;
    }
    if (*pointer != NULL)
    {
        Free_Single_Device_Pointer((void**)pointer);
    }
    Device_Malloc_Safely((void**)pointer, sizeof(T) * capacity);
    *current_capacity = capacity;
}

#ifndef USE_CPU
template <typename Key, typename Value>
static void Stable_Sort_Device_By_Key(int count, Key* d_keys, Value* d_values)
{
    if (count <= 1 || d_keys == NULL || d_values == NULL)
    {
        return;
    }
    int current_device = 0;
    cudaGetDevice(&current_device);
    setWorkingDevice(current_device);
    thrust::device_ptr<Key> key_begin(d_keys);
    thrust::device_ptr<Value> value_begin(d_values);
    // Clustered build stays device-native on GPU; no host fallback here.
    thrust::stable_sort_by_key(thrust::device, key_begin, key_begin + count,
                               value_begin);
}
#endif

static __host__ __device__ __forceinline__ int Clamp_Shift_Component(int shift)
{
    return shift < -1 ? -1 : (shift > 1 ? 1 : shift);
}

static __host__ __device__ __forceinline__ int Encode_Shift_Id(int sx, int sy,
                                                               int sz)
{
    return (Clamp_Shift_Component(sx) + 1) * 9 +
           (Clamp_Shift_Component(sy) + 1) * 3 +
           (Clamp_Shift_Component(sz) + 1);
}

static __host__ __device__ __forceinline__ VECTOR Shift_Vector_From_Id(
    int shift_id, LTMatrix3 cell)
{
    const int sx = shift_id / 9 - 1;
    const int sy = (shift_id % 9) / 3 - 1;
    const int sz = shift_id % 3 - 1;
    VECTOR frac_shift = {static_cast<float>(sx), static_cast<float>(sy),
                         static_cast<float>(sz)};
    return frac_shift * cell;
}

static __host__ __device__ __forceinline__ int Determine_Shift_Id(
    VECTOR frac_i, VECTOR frac_j)
{
    const VECTOR dfrac = frac_j - frac_i;
    const int sx = static_cast<int>(floorf(dfrac.x + 0.5f));
    const int sy = static_cast<int>(floorf(dfrac.y + 0.5f));
    const int sz = static_cast<int>(floorf(dfrac.z + 0.5f));
    return Encode_Shift_Id(sx, sy, sz);
}

static __host__ __device__ __forceinline__ int Determine_Cluster_Shift_Id(
    VECTOR super_cluster_frac_center, VECTOR cluster_center, LTMatrix3 rcell)
{
    VECTOR cluster_frac = cluster_center * rcell;
    cluster_frac.x -= floorf(cluster_frac.x);
    cluster_frac.y -= floorf(cluster_frac.y);
    cluster_frac.z -= floorf(cluster_frac.z);
    return Determine_Shift_Id(super_cluster_frac_center, cluster_frac);
}

static __host__ __device__ __forceinline__ bool Cluster_Reach_Overlaps_Shifted(
    VECTOR center_i, VECTOR center_j, float radius_i, float radius_j,
    float cutoff, VECTOR shift_vec)
{
    const VECTOR dr = center_j - (center_i + shift_vec);
    const float reach = cutoff + radius_i + radius_j;
    return dr * dr <= reach * reach;
}

static __host__ __device__ __forceinline__ LJ_CLUSTERED_IMEI
Make_Empty_Clustered_Imei()
{
    LJ_CLUSTERED_IMEI imei;
    imei.imask = 0u;
#pragma unroll
    for (int index = 0;
         index < kClusteredMaxJGroupSize * kClusteredMaxSuperClusterClusters;
         index += 1)
    {
        imei.excl_ind[index] = -1;
    }
    return imei;
}

static __host__ __device__ __forceinline__ LJ_CLUSTERED_CJ_PACKED
Make_Empty_Clustered_CjPacked()
{
    LJ_CLUSTERED_CJ_PACKED packed;
#pragma unroll
    for (int jm = 0; jm < kClusteredMaxJGroupSize; jm += 1)
    {
        packed.cj[jm] = -1;
    }
#pragma unroll
    for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
    {
        packed.imei[split] = Make_Empty_Clustered_Imei();
    }
    return packed;
}

static __host__ __device__ __forceinline__ bool Clustered_Split_Has_Atoms(
    unsigned int valid_mask_j, int split)
{
    return (valid_mask_j & Clustered_Split_Valid_Mask(split)) != 0u;
}

static __host__ __device__ __forceinline__ void
Append_Record_To_Clustered_CjPacked(
    LJ_CLUSTERED_CJ_PACKED* packed, const int jm, const int cluster_j,
    const unsigned int valid_mask_j, const unsigned int record_imask,
    const unsigned long long* exclusion_masks, int* write_exclusion,
    unsigned long long* exclusion_mask_pool)
{
    packed->cj[jm] = cluster_j;

    int exclusion_indices[kClusteredMaxSuperClusterClusters];
#pragma unroll
    for (int i_local = 0; i_local < kClusteredMaxSuperClusterClusters;
         i_local += 1)
    {
        exclusion_indices[i_local] = -1;
        if (exclusion_masks[i_local] != 0ull)
        {
            exclusion_indices[i_local] = *write_exclusion;
            exclusion_mask_pool[*write_exclusion] = exclusion_masks[i_local];
            *write_exclusion += 1;
        }
    }

#pragma unroll
    for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
    {
        if (!Clustered_Split_Has_Atoms(valid_mask_j, split))
        {
            continue;
        }
        packed->imei[split].imask |=
            record_imask << Clustered_Jm_Imask_Shift(jm);
#pragma unroll
        for (int i_local = 0; i_local < kClusteredMaxSuperClusterClusters;
             i_local += 1)
        {
            if (exclusion_indices[i_local] >= 0)
            {
                Clustered_Exclusion_Index_Ref(packed->imei[split], jm, i_local) =
                    exclusion_indices[i_local];
            }
        }
    }
}

static __global__ void Gather_Sorted_LJ_Direct_Scratch(
    const int atom_numbers, const int cluster_numbers, const int* permutation,
    const int* cluster_offsets, const VECTOR* cluster_centers,
    const LTMatrix3 cell, const LTMatrix3 rcell, const VECTOR_LJ* src,
    int* sorted_atom_ids, float4* sorted_xq, int* sorted_lj_type)
{
    SIMPLE_DEVICE_FOR(sorted_i, atom_numbers)
    {
        int cluster_lo = 0;
        int cluster_hi = cluster_numbers;
        while (cluster_lo + 1 < cluster_hi)
        {
            const int cluster_mid = (cluster_lo + cluster_hi) >> 1;
            if (cluster_offsets[cluster_mid] <= sorted_i)
            {
                cluster_lo = cluster_mid;
            }
            else
            {
                cluster_hi = cluster_mid;
            }
        }
        const VECTOR center = cluster_centers[cluster_lo];
        const int atom_i = permutation[sorted_i];
        const VECTOR_LJ atom = src[atom_i];
        const VECTOR shifted_crd =
            center + Get_Periodic_Displacement(atom.crd, center, cell, rcell);
        sorted_atom_ids[sorted_i] = atom_i;
        sorted_xq[sorted_i] = {shifted_crd.x, shifted_crd.y, shifted_crd.z,
                               atom.charge};
        sorted_lj_type[sorted_i] = atom.LJ_type;
    }
}

static __global__ void Gather_Sorted_Soft_Core_Scratch(
    const int atom_numbers, const int* permutation,
    const VECTOR_LJ_SOFT_TYPE* src, VECTOR_LJ_SOFT_TYPE* dest)
{
    SIMPLE_DEVICE_FOR(sorted_i, atom_numbers)
    {
        dest[sorted_i] = src[permutation[sorted_i]];
    }
}

static __host__ __device__ __forceinline__ int IntMin(int a, int b)
{
    return a < b ? a : b;
}

static __host__ __device__ __forceinline__ int IntMax(int a, int b)
{
    return a > b ? a : b;
}

static __host__ __device__ __forceinline__ VECTOR Wrap_To_Box_Fractional(
    VECTOR crd, LTMatrix3 rcell)
{
    VECTOR frac = crd * rcell;
    frac.x -= floorf(frac.x);
    frac.y -= floorf(frac.y);
    frac.z -= floorf(frac.z);
    return frac;
}

static __host__ __device__ __forceinline__ uint32_t Quantize_Unit_Coordinate(
    float value, int bits)
{
    if (bits <= 0)
    {
        return 0;
    }
    const uint32_t grid = 1u << bits;
    float clamped = fmaxf(0.0f, fminf(0.99999994f, value));
    uint32_t coord = static_cast<uint32_t>(clamped * static_cast<float>(grid));
    if (coord >= grid)
    {
        coord = grid - 1;
    }
    return coord;
}

static __host__ __device__ __forceinline__ VECTOR Wrap_Unit_Coordinate(
    VECTOR frac)
{
    frac.x -= floorf(frac.x);
    frac.y -= floorf(frac.y);
    frac.z -= floorf(frac.z);
    return frac;
}

static __host__ __device__ __forceinline__ VECTOR Periodic_Unit_Displacement(
    VECTOR a, VECTOR b)
{
    VECTOR dr = a - b;
    dr.x -= floorf(dr.x + 0.5f);
    dr.y -= floorf(dr.y + 0.5f);
    dr.z -= floorf(dr.z + 0.5f);
    return dr;
}

static __host__ __device__ __forceinline__ cstone::Vec3<float> To_Cstone_Vec(
    VECTOR v)
{
    return {v.x, v.y, v.z};
}

static __host__ __device__ __forceinline__ VECTOR Fractional_Cutoff_Pad(
    float cutoff, LTMatrix3 rcell)
{
    return {
        cutoff *
            sqrtf(rcell.a11 * rcell.a11 + rcell.a21 * rcell.a21 +
                  rcell.a31 * rcell.a31),
        cutoff * sqrtf(rcell.a22 * rcell.a22 + rcell.a32 * rcell.a32),
        cutoff * fabsf(rcell.a33)};
}

template <typename KeyType>
static __host__ __device__ __forceinline__ bool Cornerstone_Node_Overlaps_Box(
    KeyType prefix, VECTOR target_center, VECTOR target_size)
{
    const auto unit_box =
        cstone::Box<float>(0.0f, 1.0f, cstone::BoundaryType::periodic);
    const KeyType start_key = cstone::decodePlaceholderBit(prefix);
    const unsigned level = cstone::decodePrefixLength(prefix) / 3;
    const auto node_ibox = cstone::hilbertIBox<KeyType>(start_key, level);
    const auto [node_center, node_size] =
        cstone::centerAndSize<KeyType>(node_ibox, unit_box);
    return cstone::overlap(node_center, node_size, To_Cstone_Vec(target_center),
                           To_Cstone_Vec(target_size), unit_box);
}

static __host__ __device__ __forceinline__ bool Accept_Leaf_Pair(
    int leaf_i, int leaf_j, const int* leaf_has_local)
{
    return !(leaf_j < leaf_i && leaf_has_local[leaf_j] != 0);
}

static __host__ __device__ __forceinline__ bool Exclusion_List_Contains(
    int atom_i, int atom_j, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers)
{
    const int exclude_start = excluded_list_start[atom_i];
    const int exclude_count = excluded_numbers[atom_i];
    for (int k = 0; k < exclude_count; k += 1)
    {
        if (excluded_list[exclude_start + k] == atom_j)
        {
            return true;
        }
    }
    return false;
}

static __host__ __device__ __forceinline__ unsigned long long
Build_Exclusion_Mask(const int* permutation, const int* cluster_offsets,
                     int cluster_i, int cluster_j, unsigned int local_mask_i,
                     unsigned int valid_mask_j, int cluster_size,
                     int local_atom_numbers, const int* excluded_list_start,
                     const int* excluded_list, const int* excluded_numbers)
{
    unsigned long long mask = 0ull;
    const int start_i = cluster_offsets[cluster_i];
    const int start_j = cluster_offsets[cluster_j];
    for (int lane_i = 0; lane_i < cluster_size; lane_i += 1)
    {
        if ((local_mask_i & (1u << lane_i)) == 0u)
        {
            continue;
        }
        const int atom_i = permutation[start_i + lane_i];
        for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
        {
            if ((valid_mask_j & (1u << lane_j)) == 0u)
            {
                continue;
            }
            const int atom_j = permutation[start_j + lane_j];
            bool excluded = Exclusion_List_Contains(
                atom_i, atom_j, excluded_list_start, excluded_list,
                excluded_numbers);
            if (!excluded && atom_j < local_atom_numbers)
            {
                excluded = Exclusion_List_Contains(
                    atom_j, atom_i, excluded_list_start, excluded_list,
                    excluded_numbers);
            }
            if (excluded)
            {
                mask |= (1ull << (lane_i * cluster_size + lane_j));
            }
        }
    }
    return mask;
}

static __host__ __device__ __forceinline__ bool Cluster_Reach_Overlaps(
    VECTOR center_i, float radius_i, VECTOR center_j, float radius_j,
    float cutoff, LTMatrix3 cell, LTMatrix3 rcell)
{
    const VECTOR dr = Get_Periodic_Displacement(center_j, center_i, cell, rcell);
    const float reach = cutoff + radius_i + radius_j;
    return dr * dr <= reach * reach;
}

static __host__ __device__ __forceinline__ int
Build_CjPacked_Cluster_Metadata(
    const int cluster_i_start, const int cluster_i_end, const int cluster_j,
    const int cluster_size, const int local_atom_numbers, const float cutoff,
    const LTMatrix3 cell, const LTMatrix3 rcell, const int* permutation,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const float* cluster_radii, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers,
    const unsigned int valid_mask_j, unsigned int* imask,
    unsigned long long* exclusion_masks)
{
    *imask = 0u;
#pragma unroll
    for (int i_local = 0; i_local < kClusteredMaxSuperClusterClusters;
         i_local += 1)
    {
        exclusion_masks[i_local] = 0ull;
    }

    int exclusion_count = 0;
    for (int cluster_i = cluster_i_start; cluster_i < cluster_i_end;
         cluster_i += 1)
    {
        const unsigned int local_mask_i = cluster_local_masks[cluster_i];
        if (local_mask_i == 0u)
        {
            continue;
        }
        if (cluster_j >= cluster_i_start && cluster_j < cluster_i_end &&
            cluster_i > cluster_j)
        {
            continue;
        }
        if (!Cluster_Reach_Overlaps(cluster_centers[cluster_i],
                                   cluster_radii[cluster_i],
                                   cluster_centers[cluster_j],
                                   cluster_radii[cluster_j], cutoff, cell,
                                   rcell))
        {
            continue;
        }

        const int i_local = cluster_i - cluster_i_start;
        *imask |= (1u << static_cast<unsigned int>(i_local));
        const unsigned long long exclusion_mask = Build_Exclusion_Mask(
            permutation, cluster_offsets, cluster_i, cluster_j, local_mask_i,
            valid_mask_j, cluster_size, local_atom_numbers,
            excluded_list_start, excluded_list, excluded_numbers);
        exclusion_masks[i_local] = exclusion_mask;
        exclusion_count += exclusion_mask != 0ull ? 1 : 0;
    }
    return exclusion_count;
}

#ifdef USE_CPU
static int Build_CjPacked_Cluster_Metadata_Shifted(
    const HostClusteredBuildInput& input, const int cluster_i_start,
    const int cluster_i_end, const int cluster_j, const int shift_id,
    const unsigned int valid_mask_j, unsigned int* imask,
    std::array<unsigned long long, kClusteredMaxSuperClusterClusters>*
        exclusion_masks)
{
    *imask = 0u;
    exclusion_masks->fill(0ull);
    const VECTOR shift_vec = Shift_Vector_From_Id(shift_id, input.cell);
    int exclusion_count = 0;

    for (int cluster_i = cluster_i_start; cluster_i < cluster_i_end;
         cluster_i += 1)
    {
        const unsigned int local_mask_i =
            input.cluster_local_masks[(size_t)cluster_i];
        if (local_mask_i == 0u)
        {
            continue;
        }
        if (shift_id == kClusteredCentralShiftId &&
            cluster_j >= cluster_i_start && cluster_j < cluster_i_end &&
            cluster_i > cluster_j)
        {
            continue;
        }
        if (!Cluster_Reach_Overlaps_Shifted(
                input.cluster_centers[(size_t)cluster_i],
                input.cluster_centers[(size_t)cluster_j],
                input.cluster_radii[(size_t)cluster_i],
                input.cluster_radii[(size_t)cluster_j], input.cutoff,
                shift_vec))
        {
            continue;
        }

        const int i_local = cluster_i - cluster_i_start;
        *imask |= (1u << static_cast<unsigned int>(i_local));
        const unsigned long long exclusion_mask = Build_Exclusion_Mask(
            input.permutation.data(), input.cluster_offsets.data(), cluster_i,
            cluster_j, local_mask_i, valid_mask_j, input.cluster_size,
            input.local_atom_numbers, input.excluded_list_start.data(),
            input.excluded_list.data(), input.excluded_numbers.data());
        (*exclusion_masks)[(size_t)i_local] = exclusion_mask;
        exclusion_count += exclusion_mask != 0ull ? 1 : 0;
    }

    return exclusion_count;
}
#endif

static __host__ __device__ __forceinline__ int
Build_CjPacked_Cluster_Metadata_Shifted_Raw(
    const int cluster_i_start, const int cluster_i_end, const int cluster_j,
    const int shift_id, const int cluster_size, const int local_atom_numbers,
    const float cutoff, const LTMatrix3 cell, const LTMatrix3 rcell,
    const int* permutation, const int* cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const float* cluster_radii, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers,
    const unsigned int valid_mask_j, unsigned int* imask,
    unsigned long long* exclusion_masks)
{
    *imask = 0u;
#pragma unroll
    for (int i_local = 0; i_local < kClusteredMaxSuperClusterClusters;
         i_local += 1)
    {
        exclusion_masks[i_local] = 0ull;
    }

    const VECTOR shift_vec = Shift_Vector_From_Id(shift_id, cell);
    int exclusion_count = 0;

    for (int cluster_i = cluster_i_start; cluster_i < cluster_i_end;
         cluster_i += 1)
    {
        const unsigned int local_mask_i = cluster_local_masks[cluster_i];
        if (local_mask_i == 0u)
        {
            continue;
        }
        if (shift_id == kClusteredCentralShiftId &&
            cluster_j >= cluster_i_start && cluster_j < cluster_i_end &&
            cluster_i > cluster_j)
        {
            continue;
        }
        if (!Cluster_Reach_Overlaps_Shifted(
                cluster_centers[cluster_i], cluster_centers[cluster_j],
                cluster_radii[cluster_i], cluster_radii[cluster_j], cutoff,
                shift_vec))
        {
            continue;
        }

        const int i_local = cluster_i - cluster_i_start;
        *imask |= (1u << static_cast<unsigned int>(i_local));
        const unsigned long long exclusion_mask = Build_Exclusion_Mask(
            permutation, cluster_offsets, cluster_i, cluster_j, local_mask_i,
            valid_mask_j, cluster_size, local_atom_numbers,
            excluded_list_start, excluded_list, excluded_numbers);
        exclusion_masks[i_local] = exclusion_mask;
        exclusion_count += exclusion_mask != 0ull ? 1 : 0;
    }

    return exclusion_count;
}

#ifdef USE_CPU
static void Build_Nbnxm_Payload_On_Host(
    const HostClusteredBuildInput& input, std::vector<LJ_CLUSTERED_SCI>* scis,
    std::vector<LJ_CLUSTERED_CJ_PACKED>* cjpacked,
    std::vector<unsigned long long>* exclusion_pool)
{
    scis->clear();
    cjpacked->clear();
    exclusion_pool->clear();
    if (input.candidate_sci_numbers <= 0)
    {
        return;
    }

    std::array<std::vector<HostClusteredJRecord>, kClusteredShiftCount>
        shift_buckets;

    for (int candidate_sci = 0; candidate_sci < input.candidate_sci_numbers;
         candidate_sci += 1)
    {
        for (auto& bucket : shift_buckets)
        {
            bucket.clear();
        }

        const int super_i =
            input.sci_supercluster_ids[(size_t)candidate_sci];
        const VECTOR super_center_frac =
            input.super_cluster_centers[(size_t)super_i];
        const int cluster_i_start =
            input.super_cluster_offsets[(size_t)super_i];
        const int cluster_i_end =
            input.super_cluster_offsets[(size_t)super_i + 1];
        for (int candidate_idx =
                 input.candidate_leaf_offsets[(size_t)candidate_sci];
             candidate_idx <
             input.candidate_leaf_offsets[(size_t)candidate_sci + 1];
             candidate_idx += 1)
        {
            const int leaf_j = input.candidate_leaf_ids[(size_t)candidate_idx];
            const int cluster_j_start =
                input.leaf_cluster_offsets[(size_t)leaf_j];
            const int cluster_j_end =
                input.leaf_cluster_offsets[(size_t)leaf_j + 1];
            for (int cluster_j = cluster_j_start; cluster_j < cluster_j_end;
                 cluster_j += 1)
            {
                const unsigned int valid_mask_j =
                    input.cluster_valid_masks[(size_t)cluster_j];
                if (valid_mask_j == 0u)
                {
                    continue;
                }
                const int super_j = cluster_j / input.super_cluster_clusters;
                if (input.cluster_local_masks[(size_t)cluster_j] != 0u &&
                    super_j < super_i)
                {
                    continue;
                }
                const int shift_id = Determine_Cluster_Shift_Id(
                    super_center_frac,
                    input.cluster_centers[(size_t)cluster_j], input.rcell);
                HostClusteredJRecord record = {};
                record.cluster_j = cluster_j;
                Build_CjPacked_Cluster_Metadata_Shifted(
                    input, cluster_i_start, cluster_i_end, cluster_j, shift_id,
                    valid_mask_j, &record.imask, &record.exclusion_masks);
                if (record.imask != 0u)
                {
                    shift_buckets[(size_t)shift_id].push_back(record);
                }
            }
        }

        for (int shift_id = 0; shift_id < kClusteredShiftCount; shift_id += 1)
        {
            auto& bucket = shift_buckets[(size_t)shift_id];
            if (bucket.empty())
            {
                continue;
            }
            std::stable_sort(
                bucket.begin(), bucket.end(),
                [](const HostClusteredJRecord& lhs,
                   const HostClusteredJRecord& rhs)
                {
                    if (lhs.Has_Exclusions() != rhs.Has_Exclusions())
                    {
                        return lhs.Has_Exclusions() > rhs.Has_Exclusions();
                    }
                    return lhs.cluster_j < rhs.cluster_j;
                });

            const int cj_begin = static_cast<int>(cjpacked->size());
            for (size_t bucket_begin = 0; bucket_begin < bucket.size();
                 bucket_begin += kClusteredMaxJGroupSize)
            {
                LJ_CLUSTERED_CJ_PACKED packed = Make_Empty_Clustered_CjPacked();
                for (int jm = 0; jm < kClusteredMaxJGroupSize; jm += 1)
                {
                    const size_t record_index = bucket_begin + (size_t)jm;
                    if (record_index >= bucket.size())
                    {
                        break;
                    }
                    const auto& record = bucket[record_index];
                    packed.cj[jm] = record.cluster_j;
                    const unsigned int valid_mask_j =
                        input.cluster_valid_masks[(size_t)record.cluster_j];
                    int exclusion_indices[kClusteredMaxSuperClusterClusters];
                    for (int i_local = 0;
                         i_local < kClusteredMaxSuperClusterClusters;
                         i_local += 1)
                    {
                        exclusion_indices[i_local] = -1;
                        const unsigned long long exclusion_mask =
                            record.exclusion_masks[(size_t)i_local];
                        if (exclusion_mask == 0ull)
                        {
                            continue;
                        }
                        exclusion_indices[i_local] =
                            static_cast<int>(exclusion_pool->size());
                        exclusion_pool->push_back(exclusion_mask);
                    }
#pragma unroll
                    for (int split = 0; split < kClusteredWarpSplitCount;
                         split += 1)
                    {
                        if (!Clustered_Split_Has_Atoms(valid_mask_j, split))
                        {
                            continue;
                        }
                        packed.imei[split].imask |=
                            record.imask << Clustered_Jm_Imask_Shift(jm);
                        for (int i_local = 0;
                             i_local < kClusteredMaxSuperClusterClusters;
                             i_local += 1)
                        {
                            if (exclusion_indices[i_local] >= 0)
                            {
                                Clustered_Exclusion_Index_Ref(
                                    packed.imei[split], jm, i_local) =
                                    exclusion_indices[i_local];
                            }
                        }
                    }
                }
                cjpacked->push_back(packed);
            }
            scis->push_back(
                {super_i, shift_id, cj_begin, static_cast<int>(cjpacked->size())});
        }
    }
}
#endif

static __global__ void Check_Clustered_Rebuild(
    const int atom_numbers, const VECTOR* crd, const VECTOR* cached_crd,
    const LTMatrix3 cell, const LTMatrix3 rcell, int* need_rebuild,
    const float permit_square)
{
    SIMPLE_DEVICE_FOR(tid, atom_numbers)
    {
        const VECTOR dr = Get_Periodic_Displacement(crd[tid], cached_crd[tid],
                                                    cell, rcell);
        if (dr * dr > permit_square)
        {
            need_rebuild[0] = 1;
        }
    }
}

static __global__ void Build_Cornerstone_Sort_Keys(const int atom_numbers,
                                                   const VECTOR* crd,
                                                   const LTMatrix3 rcell,
                                                   uint64_t* keys,
                                                   int* permutation)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        const VECTOR frac = Wrap_To_Box_Fractional(crd[i], rcell);
        constexpr int bits = cstone::maxTreeLevel<CornerstoneKey>{};
        const uint32_t x = Quantize_Unit_Coordinate(frac.x, bits);
        const uint32_t y = Quantize_Unit_Coordinate(frac.y, bits);
        const uint32_t z = Quantize_Unit_Coordinate(frac.z, bits);
        keys[i] = cstone::iHilbert<CornerstoneKey>(x, y, z);
        permutation[i] = i;
    }
}

static __global__ void Copy_UInt_To_Int(const int count, const unsigned* src,
                                        int* dest)
{
    SIMPLE_DEVICE_FOR(i, count) { dest[i] = static_cast<int>(src[i]); }
}

static __global__ void Build_Leaf_Cluster_Counts(const int leaf_numbers,
                                                 const int cluster_size,
                                                 const int* leaf_atom_offsets,
                                                 int* leaf_cluster_counts)
{
    SIMPLE_DEVICE_FOR(leaf_i, leaf_numbers)
    {
        const int atom_count =
            leaf_atom_offsets[leaf_i + 1] - leaf_atom_offsets[leaf_i];
        leaf_cluster_counts[leaf_i] =
            IntMax(0, (atom_count + cluster_size - 1) / cluster_size);
    }
}

static __global__ void Build_Leaf_Metadata_And_Clusters(
    const int leaf_numbers, const int direct_local_atom_numbers,
    const int cluster_size, const float cutoff, const int* permutation,
    const VECTOR* crd,
    const LTMatrix3 cell, const LTMatrix3 rcell, const int* leaf_atom_offsets,
    const int* leaf_cluster_offsets, int* cluster_offsets,
    unsigned int* cluster_valid_masks, unsigned int* cluster_local_masks,
    VECTOR* cluster_centers, float* cluster_radii)
{
    SIMPLE_DEVICE_FOR(leaf_i, leaf_numbers)
    {
        const int atom_start = leaf_atom_offsets[leaf_i];
        const int atom_end = leaf_atom_offsets[leaf_i + 1];
        const int atom_count = IntMax(0, atom_end - atom_start);
        const int cluster_start = leaf_cluster_offsets[leaf_i];
        const int cluster_end = leaf_cluster_offsets[leaf_i + 1];
        (void)cutoff;
        (void)rcell;

        for (int cluster_i = cluster_start; cluster_i < cluster_end; cluster_i += 1)
        {
            const int cluster_local = cluster_i - cluster_start;
            const int start = atom_start + cluster_local * cluster_size;
            const int end = IntMin(atom_end, start + cluster_size);
            const int count = IntMax(0, end - start);
            unsigned int valid_mask = 0u;
            unsigned int local_mask = 0u;
            VECTOR center = {0.0f, 0.0f, 0.0f};
            float radius = 0.0f;

            cluster_offsets[cluster_i] = start;
            if (count > 0)
            {
                const VECTOR anchor = crd[permutation[start]];
                for (int lane = 0; lane < count; lane += 1)
                {
                    valid_mask |= (1u << lane);
                    const int atom_index = permutation[start + lane];
                    if (atom_index < direct_local_atom_numbers)
                    {
                        local_mask |= (1u << lane);
                    }
                    const VECTOR pos = crd[atom_index];
                    center = center +
                             (anchor + Get_Periodic_Displacement(pos, anchor,
                                                                 cell, rcell));
                }
                center = (1.0f / static_cast<float>(count)) * center;
                center = Get_Periodic_Coordinate(center, cell, rcell);
                for (int lane = 0; lane < count; lane += 1)
                {
                    const int atom_index = permutation[start + lane];
                    const VECTOR pos = crd[atom_index];
                    const VECTOR dr =
                        Get_Periodic_Displacement(pos, center, cell, rcell);
                    radius = fmaxf(radius, norm3df(dr.x, dr.y, dr.z));
                }
            }

            cluster_valid_masks[cluster_i] = valid_mask;
            cluster_local_masks[cluster_i] = local_mask;
            cluster_centers[cluster_i] = center;
            cluster_radii[cluster_i] = radius;
        }
        cluster_offsets[cluster_end] = atom_end;
    }
}

static __global__ void Build_Fixed_Group_Offsets(const int offset_numbers,
                                                 const int group_size,
                                                 const int total_numbers,
                                                 int* offsets)
{
    SIMPLE_DEVICE_FOR(group_i, offset_numbers)
    {
        offsets[group_i] =
            IntMin(group_i * group_size, total_numbers);
    }
}

static __global__ void Build_Supercluster_Metadata(
    const int super_cluster_numbers, const int super_cluster_clusters,
    const float cutoff, const LTMatrix3 rcell,
    const int* super_cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const float* cluster_radii, int* super_cluster_has_local,
    VECTOR* super_cluster_centers, VECTOR* super_cluster_sizes)
{
    SIMPLE_DEVICE_FOR(super_i, super_cluster_numbers)
    {
        const int cluster_start = super_cluster_offsets[super_i];
        const int cluster_end = super_cluster_offsets[super_i + 1];
        int cluster_count = 0;
        int has_local = 0;
        VECTOR anchor_frac = {0.0f, 0.0f, 0.0f};
        VECTOR frac_sum = {0.0f, 0.0f, 0.0f};
        VECTOR frac_center = {0.0f, 0.0f, 0.0f};
        VECTOR frac_size = {0.0f, 0.0f, 0.0f};

        for (int cluster_i = cluster_start; cluster_i < cluster_end; cluster_i += 1)
        {
            if (cluster_valid_masks[cluster_i] == 0u)
            {
                continue;
            }
            const VECTOR frac =
                Wrap_To_Box_Fractional(cluster_centers[cluster_i], rcell);
            if (cluster_count == 0)
            {
                anchor_frac = frac;
            }
            frac_sum = frac_sum +
                       (anchor_frac +
                        Periodic_Unit_Displacement(frac, anchor_frac));
            has_local |= cluster_local_masks[cluster_i] != 0u;
            cluster_count += 1;
        }

        if (cluster_count > 0)
        {
            frac_center = (1.0f / static_cast<float>(cluster_count)) * frac_sum;
            frac_center = Wrap_Unit_Coordinate(frac_center);
            const VECTOR cutoff_pad = Fractional_Cutoff_Pad(cutoff, rcell);
            for (int cluster_i = cluster_start; cluster_i < cluster_end;
                 cluster_i += 1)
            {
                if (cluster_valid_masks[cluster_i] == 0u)
                {
                    continue;
                }
                const VECTOR frac =
                    Wrap_To_Box_Fractional(cluster_centers[cluster_i], rcell);
                const VECTOR dfrac =
                    Periodic_Unit_Displacement(frac, frac_center);
                const VECTOR radius_pad =
                    Fractional_Cutoff_Pad(cluster_radii[cluster_i], rcell);
                frac_size.x =
                    fmaxf(frac_size.x, fabsf(dfrac.x) + radius_pad.x);
                frac_size.y =
                    fmaxf(frac_size.y, fabsf(dfrac.y) + radius_pad.y);
                frac_size.z =
                    fmaxf(frac_size.z, fabsf(dfrac.z) + radius_pad.z);
            }
            frac_size.x = fminf(0.5f, frac_size.x + cutoff_pad.x);
            frac_size.y = fminf(0.5f, frac_size.y + cutoff_pad.y);
            frac_size.z = fminf(0.5f, frac_size.z + cutoff_pad.z);
        }

        (void)super_cluster_clusters;
        super_cluster_has_local[super_i] = has_local;
        super_cluster_centers[super_i] = frac_center;
        super_cluster_sizes[super_i] = frac_size;
    }
}

static __global__ void Build_Local_Supercluster_Flags(
    const int super_cluster_numbers, const int* super_cluster_has_local,
    int* sci_flags)
{
    SIMPLE_DEVICE_FOR(super_i, super_cluster_numbers)
    {
        sci_flags[super_i] = super_cluster_has_local[super_i] != 0 ? 1 : 0;
    }
}

static __global__ void Fill_Local_Supercluster_Ids(
    const int super_cluster_numbers, const int* super_cluster_has_local,
    const int* sci_offsets, int* sci_supercluster_ids)
{
    SIMPLE_DEVICE_FOR(super_i, super_cluster_numbers)
    {
        if (super_cluster_has_local[super_i] != 0)
        {
            const int sci = sci_offsets[super_i];
            sci_supercluster_ids[sci] = super_i;
        }
    }
}

static __global__ void Count_Supercluster_Candidate_Leaves(
    const int sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    const CornerstoneKey* node_prefixes,
    const CornerstoneNodeIndex* child_offsets,
    const CornerstoneNodeIndex* parents,
    const CornerstoneNodeIndex* internal_to_leaf,
    int* candidate_leaf_counts)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const int super_i = sci_supercluster_ids[sci];
        const VECTOR target_center = super_cluster_centers[super_i];
        const VECTOR target_size = super_cluster_sizes[super_i];
        int count = 0;

        auto overlaps = [=](CornerstoneNodeIndex node)
        {
            return Cornerstone_Node_Overlaps_Box(node_prefixes[node],
                                                target_center, target_size);
        };
        auto endpoint = [&](CornerstoneNodeIndex node)
        {
            const int leaf_j = internal_to_leaf[node];
            if (leaf_j >= 0)
            {
                count += 1;
            }
        };
        cstone::singleTraversal(child_offsets, parents, overlaps, endpoint);
        candidate_leaf_counts[sci] = count;
    }
}

static __global__ void Fill_Supercluster_Candidate_Leaves(
    const int sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    const CornerstoneKey* node_prefixes,
    const CornerstoneNodeIndex* child_offsets,
    const CornerstoneNodeIndex* parents,
    const CornerstoneNodeIndex* internal_to_leaf,
    const int* candidate_leaf_offsets, int* candidate_leaf_ids)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const int super_i = sci_supercluster_ids[sci];
        const VECTOR target_center = super_cluster_centers[super_i];
        const VECTOR target_size = super_cluster_sizes[super_i];
        int write_offset = candidate_leaf_offsets[sci];

        auto overlaps = [=](CornerstoneNodeIndex node)
        {
            return Cornerstone_Node_Overlaps_Box(node_prefixes[node],
                                                target_center, target_size);
        };
        auto endpoint = [&](CornerstoneNodeIndex node)
        {
            const int leaf_j = internal_to_leaf[node];
            if (leaf_j >= 0)
            {
                candidate_leaf_ids[write_offset] = leaf_j;
                write_offset += 1;
            }
        };
        cstone::singleTraversal(child_offsets, parents, overlaps, endpoint);
    }
}

static __global__ void Count_Nbnxm_Payload_From_Candidate_Leaves(
    const int candidate_sci_numbers, const int sci_shift_numbers,
    const int cluster_size, const int super_cluster_clusters,
    const int local_atom_numbers, const float cutoff, const LTMatrix3 cell,
    const LTMatrix3 rcell, const int* permutation, const int* cluster_offsets,
    const int* leaf_cluster_offsets, const int* super_cluster_offsets,
    const int* sci_supercluster_ids, const VECTOR* super_cluster_centers,
    const int* candidate_leaf_offsets, const int* candidate_leaf_ids,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const float* cluster_radii, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers,
    int* sci_shift_flags, int* cjpacked_group_counts, int* exclusion_counts)
{
    SIMPLE_DEVICE_FOR(sci_shift, sci_shift_numbers)
    {
        const int candidate_sci = sci_shift / kClusteredShiftCount;
        if (candidate_sci < candidate_sci_numbers)
        {
            const int shift_id = sci_shift % kClusteredShiftCount;
            const int super_i = sci_supercluster_ids[candidate_sci];
            const VECTOR super_center_frac = super_cluster_centers[super_i];
            const int cluster_i_start = super_cluster_offsets[super_i];
            const int cluster_i_end = super_cluster_offsets[super_i + 1];
            int record_count = 0;
            int exclusion_count = 0;

            for (int candidate_idx = candidate_leaf_offsets[candidate_sci];
                 candidate_idx < candidate_leaf_offsets[candidate_sci + 1];
                 candidate_idx += 1)
            {
                const int leaf_j = candidate_leaf_ids[candidate_idx];
                const int cluster_j_start = leaf_cluster_offsets[leaf_j];
                const int cluster_j_end = leaf_cluster_offsets[leaf_j + 1];
                for (int cluster_j = cluster_j_start; cluster_j < cluster_j_end;
                     cluster_j += 1)
                {
                    const unsigned int valid_mask_j =
                        cluster_valid_masks[cluster_j];
                    if (valid_mask_j == 0u)
                    {
                        continue;
                    }
                    const int super_j = cluster_j / super_cluster_clusters;
                    if (cluster_local_masks[cluster_j] != 0u &&
                        super_j < super_i)
                    {
                        continue;
                    }
                    if (Determine_Cluster_Shift_Id(
                            super_center_frac, cluster_centers[cluster_j],
                            rcell) != shift_id)
                    {
                        continue;
                    }

                    unsigned int imask = 0u;
                    unsigned long long exclusion_masks[kClusteredMaxSuperClusterClusters];
                    const int exclusion_count_for_cluster =
                        Build_CjPacked_Cluster_Metadata_Shifted_Raw(
                            cluster_i_start, cluster_i_end, cluster_j,
                            shift_id, cluster_size, local_atom_numbers, cutoff,
                            cell, rcell, permutation, cluster_offsets,
                            cluster_valid_masks, cluster_local_masks,
                            cluster_centers, cluster_radii,
                            excluded_list_start, excluded_list,
                            excluded_numbers, valid_mask_j, &imask,
                            exclusion_masks);
                    if (imask == 0u)
                    {
                        continue;
                    }
                    record_count += 1;
                    exclusion_count += exclusion_count_for_cluster;
                }
            }

            sci_shift_flags[sci_shift] = record_count > 0 ? 1 : 0;
            cjpacked_group_counts[sci_shift] =
                (record_count + kClusteredMaxJGroupSize - 1) /
                kClusteredMaxJGroupSize;
            exclusion_counts[sci_shift] = exclusion_count;
        }
    }
}

static __global__ void Fill_Nbnxm_Payload_From_Candidate_Leaves(
    const int candidate_sci_numbers, const int sci_shift_numbers,
    const int cluster_size, const int super_cluster_clusters,
    const int local_atom_numbers, const float cutoff, const LTMatrix3 cell,
    const LTMatrix3 rcell, const int* permutation, const int* cluster_offsets,
    const int* leaf_cluster_offsets, const int* super_cluster_offsets,
    const int* sci_supercluster_ids, const VECTOR* super_cluster_centers,
    const int* candidate_leaf_offsets, const int* candidate_leaf_ids,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const float* cluster_radii, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers,
    const int* sci_shift_flags, const int* sci_shift_offsets,
    const int* cjpacked_group_counts, const int* cjpacked_group_offsets,
    const int* exclusion_offsets, LJ_CLUSTERED_SCI* nbnxm_sci,
    LJ_CLUSTERED_CJ_PACKED* nbnxm_cjpacked,
    unsigned long long* exclusion_mask_pool)
{
    SIMPLE_DEVICE_FOR(sci_shift, sci_shift_numbers)
    {
        const int candidate_sci = sci_shift / kClusteredShiftCount;
        if (candidate_sci < candidate_sci_numbers &&
            sci_shift_flags[sci_shift] != 0)
        {
            const int shift_id = sci_shift % kClusteredShiftCount;
            const int super_i = sci_supercluster_ids[candidate_sci];
            const VECTOR super_center_frac = super_cluster_centers[super_i];
            const int cluster_i_start = super_cluster_offsets[super_i];
            const int cluster_i_end = super_cluster_offsets[super_i + 1];
            const int write_sci = sci_shift_offsets[sci_shift];
            const int cjpacked_begin = cjpacked_group_offsets[sci_shift];
            const int cjpacked_end =
                cjpacked_begin + cjpacked_group_counts[sci_shift];
            int write_packed = cjpacked_begin;
            int write_exclusion = exclusion_offsets[sci_shift];
            int jm_in_group = 0;
            LJ_CLUSTERED_CJ_PACKED packed = Make_Empty_Clustered_CjPacked();

            nbnxm_sci[write_sci] = {
                super_i, shift_id, cjpacked_begin, cjpacked_end};

            for (int pass = 0; pass < 2; pass += 1)
            {
                const bool want_with_exclusion = pass == 0;
                for (int candidate_idx = candidate_leaf_offsets[candidate_sci];
                     candidate_idx < candidate_leaf_offsets[candidate_sci + 1];
                     candidate_idx += 1)
                {
                    const int leaf_j = candidate_leaf_ids[candidate_idx];
                    const int cluster_j_start = leaf_cluster_offsets[leaf_j];
                    const int cluster_j_end =
                        leaf_cluster_offsets[leaf_j + 1];
                    for (int cluster_j = cluster_j_start;
                         cluster_j < cluster_j_end; cluster_j += 1)
                    {
                        const unsigned int valid_mask_j =
                            cluster_valid_masks[cluster_j];
                        if (valid_mask_j == 0u)
                        {
                            continue;
                        }
                        const int super_j = cluster_j / super_cluster_clusters;
                        if (cluster_local_masks[cluster_j] != 0u &&
                            super_j < super_i)
                        {
                            continue;
                        }
                        if (Determine_Cluster_Shift_Id(
                                super_center_frac, cluster_centers[cluster_j],
                                rcell) != shift_id)
                        {
                            continue;
                        }

                        unsigned int imask = 0u;
                        unsigned long long exclusion_masks[kClusteredMaxSuperClusterClusters];
                        const int exclusion_count_for_cluster =
                            Build_CjPacked_Cluster_Metadata_Shifted_Raw(
                                cluster_i_start, cluster_i_end, cluster_j,
                                shift_id, cluster_size, local_atom_numbers,
                                cutoff, cell, rcell, permutation,
                                cluster_offsets, cluster_valid_masks,
                                cluster_local_masks, cluster_centers,
                                cluster_radii, excluded_list_start,
                                excluded_list, excluded_numbers, valid_mask_j,
                                &imask, exclusion_masks);
                        if (imask == 0u)
                        {
                            continue;
                        }
                        const bool has_exclusion =
                            exclusion_count_for_cluster > 0;
                        if (has_exclusion != want_with_exclusion)
                        {
                            continue;
                        }

                        Append_Record_To_Clustered_CjPacked(
                            &packed, jm_in_group, cluster_j, valid_mask_j, imask,
                            exclusion_masks, &write_exclusion,
                            exclusion_mask_pool);

                        jm_in_group += 1;
                        if (jm_in_group == kClusteredMaxJGroupSize)
                        {
                            nbnxm_cjpacked[write_packed] = packed;
                            write_packed += 1;
                            packed = Make_Empty_Clustered_CjPacked();
                            jm_in_group = 0;
                        }
                    }
                }
            }

            if (jm_in_group > 0)
            {
                nbnxm_cjpacked[write_packed] = packed;
                write_packed += 1;
            }
        }
    }
}

static __global__ void Build_Sci_Workload_Sort_Keys(
    const int sci_numbers, const LJ_CLUSTERED_SCI* sci_entries, int* sci_sort_keys)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        sci_sort_keys[sci] =
            -(sci_entries[sci].cjpacked_end - sci_entries[sci].cjpacked_begin);
    }
}

static void Reserve_Device_Int_Buffer(int capacity, int** pointer,
                                      int* current_capacity)
{
    if (capacity <= *current_capacity && *pointer != NULL)
    {
        return;
    }
    if (*pointer != NULL)
    {
        Free_Single_Device_Pointer((void**)pointer);
    }
    Device_Malloc_Safely((void**)pointer, sizeof(int) * capacity);
    *current_capacity = capacity;
}

static void Reserve_Device_U64_Buffer(int capacity, uint64_t** pointer,
                                      int* current_capacity)
{
    if (capacity <= *current_capacity && *pointer != NULL)
    {
        return;
    }
    if (*pointer != NULL)
    {
        Free_Single_Device_Pointer((void**)pointer);
    }
    Device_Malloc_Safely((void**)pointer, sizeof(uint64_t) * capacity);
    *current_capacity = capacity;
}

static void Reserve_Device_UInt_Buffer(int capacity, unsigned int** pointer,
                                       int* current_capacity)
{
    if (capacity <= *current_capacity && *pointer != NULL)
    {
        return;
    }
    if (*pointer != NULL)
    {
        Free_Single_Device_Pointer((void**)pointer);
    }
    Device_Malloc_Safely((void**)pointer, sizeof(unsigned int) * capacity);
    *current_capacity = capacity;
}

static void Reserve_Device_PairMask_Buffer(int capacity,
                                           unsigned long long** pointer,
                                           int* current_capacity)
{
    if (capacity <= *current_capacity && *pointer != NULL)
    {
        return;
    }
    if (*pointer != NULL)
    {
        Free_Single_Device_Pointer((void**)pointer);
    }
    Device_Malloc_Safely((void**)pointer, sizeof(unsigned long long) * capacity);
    *current_capacity = capacity;
}

static void Reserve_Device_Vector_Buffer(int capacity, VECTOR** pointer,
                                         int* current_capacity)
{
    if (capacity <= *current_capacity && *pointer != NULL)
    {
        return;
    }
    if (*pointer != NULL)
    {
        Free_Single_Device_Pointer((void**)pointer);
    }
    Device_Malloc_Safely((void**)pointer, sizeof(VECTOR) * capacity);
    *current_capacity = capacity;
}

static void Reserve_Device_Float_Buffer(int capacity, float** pointer,
                                        int* current_capacity)
{
    if (capacity <= *current_capacity && *pointer != NULL)
    {
        return;
    }
    if (*pointer != NULL)
    {
        Free_Single_Device_Pointer((void**)pointer);
    }
    Device_Malloc_Safely((void**)pointer, sizeof(float) * capacity);
    *current_capacity = capacity;
}

static void Commit_Clustered_Build_Cache(LJ_CLUSTER_LAYOUT* layout,
                                         const VECTOR* crd, float cutoff)
{
    if (layout->total_atom_numbers > 0)
    {
        Reserve_Device_Vector_Buffer(layout->total_atom_numbers,
                                     &layout->d_cached_crd,
                                     &layout->cached_crd_capacity);
        deviceMemcpy(layout->d_cached_crd, crd,
                     sizeof(VECTOR) * layout->total_atom_numbers,
                     deviceMemcpyDeviceToDevice);
    }
    layout->cached_cutoff = cutoff;
    layout->cache_ready = layout->total_atom_numbers > 0;
    layout->rebuild_dirty = false;
}

static bool Clustered_Build_Is_Needed(LJ_CLUSTER_LAYOUT* layout,
                                      const VECTOR* crd, const LTMatrix3 cell,
                                      const LTMatrix3 rcell, const float cutoff)
{
    if (layout->rebuild_dirty || !layout->cache_ready ||
        layout->d_cached_crd == NULL)
    {
        return true;
    }
    if (fabsf(layout->cached_cutoff - cutoff) > 1e-6f)
    {
        return true;
    }
    if (layout->rebuild_skin <= 0.0f || layout->rebuild_skin_permit <= 0.0f ||
        layout->local_atom_numbers <= 0)
    {
        return true;
    }

    Reserve_Device_Int_Buffer(1, &layout->d_need_rebuild,
                              &layout->rebuild_flag_capacity);
    int h_need_rebuild = 0;
    deviceMemcpy(layout->d_need_rebuild, &h_need_rebuild, sizeof(int),
                 deviceMemcpyHostToDevice);
    const float permit = layout->rebuild_skin * layout->rebuild_skin_permit;
    Launch_Device_Kernel(Check_Clustered_Rebuild,
                         (layout->local_atom_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         layout->local_atom_numbers, crd, layout->d_cached_crd,
                         cell, rcell, layout->d_need_rebuild, permit * permit);
    deviceMemcpy(&h_need_rebuild, layout->d_need_rebuild, sizeof(int),
                 deviceMemcpyDeviceToHost);
    return h_need_rebuild != 0;
}

static void Initialize_Cornerstone_State(LJ_CLUSTER_LAYOUT* layout)
{
    if (layout->cornerstone_state == NULL)
    {
        layout->cornerstone_state = new LJ_CORNERSTONE_STATE();
    }
}

static void Reset_Build_Buffers(LJ_CLUSTER_LAYOUT* layout)
{
    Free_Single_Device_Pointer((void**)&layout->d_sort_permutation);
    Free_Single_Device_Pointer((void**)&layout->d_sort_keys);
    Free_Single_Device_Pointer((void**)&layout->d_cluster_offsets);
    Free_Single_Device_Pointer((void**)&layout->d_cluster_valid_masks);
    Free_Single_Device_Pointer((void**)&layout->d_cluster_local_masks);
    Free_Single_Device_Pointer((void**)&layout->d_cluster_centers);
    Free_Single_Device_Pointer((void**)&layout->d_cluster_radii);
    Free_Single_Device_Pointer((void**)&layout->d_leaf_atom_offsets);
    Free_Single_Device_Pointer((void**)&layout->d_leaf_cluster_offsets);
    Free_Single_Device_Pointer((void**)&layout->d_super_cluster_offsets);
    Free_Single_Device_Pointer((void**)&layout->d_super_cluster_has_local);
    Free_Single_Device_Pointer((void**)&layout->d_super_cluster_centers);
    Free_Single_Device_Pointer((void**)&layout->d_super_cluster_sizes);
    Free_Single_Device_Pointer((void**)&layout->d_sci_supercluster_ids);
    Free_Single_Device_Pointer((void**)&layout->d_sci_candidate_leaf_counts);
    Free_Single_Device_Pointer((void**)&layout->d_sci_candidate_leaf_offsets);
    Free_Single_Device_Pointer((void**)&layout->d_sci_candidate_leaf_ids);
    Free_Single_Device_Pointer((void**)&layout->d_cjpacked_counts);
    Free_Single_Device_Pointer((void**)&layout->d_exclusion_counts);
    Free_Single_Device_Pointer((void**)&layout->d_exclusion_offsets);
    Free_Single_Device_Pointer((void**)&layout->d_sci_shift_flags);
    Free_Single_Device_Pointer((void**)&layout->d_sci_shift_offsets);
    Free_Single_Device_Pointer((void**)&layout->d_cjpacked_group_offsets);
    Free_Single_Device_Pointer((void**)&layout->d_exclusion_mask_pool);
    Free_Single_Device_Pointer((void**)&layout->d_nbnxm_sci);
    Free_Single_Device_Pointer((void**)&layout->d_nbnxm_cjpacked);

    layout->permutation_capacity = 0;
    layout->cluster_capacity = 0;
    layout->leaf_capacity = 0;
    layout->super_cluster_capacity = 0;
    layout->sci_capacity = 0;
    layout->candidate_leaf_capacity = 0;
    layout->cjpacked_capacity = 0;
    layout->exclusion_scan_capacity = 0;
    layout->exclusion_capacity = 0;
    layout->scan_capacity = 0;
    layout->sci_shift_capacity = 0;
    layout->nbnxm_sci_capacity = 0;
    layout->nbnxm_cjpacked_capacity = 0;
}

static void Reset_Cornerstone_Root(LJ_CORNERSTONE_STATE* state,
                                   int total_atom_numbers)
{
    std::vector<CornerstoneKey> root_leaves = {0ull,
                                               cstone::nodeRange<CornerstoneKey>(
                                                   0)};
    std::vector<unsigned> root_counts = {
        static_cast<unsigned>(total_atom_numbers)};
#ifndef USE_CPU
    state->leaves = root_leaves;
    state->leaf_counts = root_counts;
    state->tmp_leaves.resize(0);
    state->work_array.resize(0);
#else
    state->leaves = std::move(root_leaves);
    state->leaf_counts = std::move(root_counts);
#endif
}

static void Build_Cornerstone_Tree(LJ_CLUSTER_LAYOUT* layout)
{
    auto* state = layout->cornerstone_state;
    if (state == NULL || layout->total_atom_numbers <= 0)
    {
        return;
    }

    if (state->leaves.empty())
    {
        Reset_Cornerstone_Root(state, layout->total_atom_numbers);
    }

#ifndef USE_CPU
    const auto* sorted_keys = layout->d_sort_keys;
    const std::span<const CornerstoneKey> key_span(
        sorted_keys, static_cast<size_t>(layout->total_atom_numbers));
    bool converged = false;
    for (int iter = 0; iter < 64 && !converged; iter += 1)
    {
        converged = cstone::updateOctreeGpu<CornerstoneKey>(
            key_span, static_cast<unsigned>(layout->cornerstone_leaf_size),
            state->leaves, state->leaf_counts, state->tmp_leaves,
            state->work_array);
    }
    state->octree.resize(static_cast<CornerstoneNodeIndex>(
        cstone::nNodes(state->leaves)));
    cstone::buildOctreeGpu(rawPtr(state->leaves), state->octree.data());
#else
    std::vector<CornerstoneKey> host_keys(
        static_cast<size_t>(layout->total_atom_numbers));
    deviceMemcpy(host_keys.data(), layout->d_sort_keys,
                 sizeof(CornerstoneKey) * layout->total_atom_numbers,
                 deviceMemcpyDeviceToHost);
    const std::span<const CornerstoneKey> key_span(host_keys.data(),
                                                   host_keys.size());
    bool converged = false;
    for (int iter = 0; iter < 64 && !converged; iter += 1)
    {
        converged = cstone::updateOctree<CornerstoneKey>(
            key_span, static_cast<unsigned>(layout->cornerstone_leaf_size),
            state->leaves, state->leaf_counts);
    }
    state->octree.resize(static_cast<CornerstoneNodeIndex>(
        cstone::nNodes(state->leaves)));
    cstone::updateInternalTree<CornerstoneKey>(
        std::span<const CornerstoneKey>(state->leaves.data(),
                                        state->leaves.size()),
        state->octree.data());
#endif
}

static int Exclusive_Scan_Counts(int count_numbers, int* d_counts, int* d_starts)
{
    if (count_numbers <= 0)
    {
        return 0;
    }
#ifndef USE_CPU
    thrust::device_ptr<int> count_begin(d_counts);
    thrust::device_ptr<int> start_begin(d_starts);
    const int total = static_cast<int>(
        thrust::reduce(thrust::device, count_begin, count_begin + count_numbers));
    thrust::exclusive_scan(thrust::device, count_begin,
                           count_begin + count_numbers, start_begin);
    deviceMemcpy(d_starts + count_numbers, &total, sizeof(int),
                 deviceMemcpyHostToDevice);
    return total;
#else
    std::vector<int> h_counts(static_cast<size_t>(count_numbers));
    std::vector<int> h_starts(static_cast<size_t>(count_numbers) + 1, 0);
    deviceMemcpy(h_counts.data(), d_counts, sizeof(int) * count_numbers,
                 deviceMemcpyDeviceToHost);
    int total = 0;
    for (int i = 0; i < count_numbers; i += 1)
    {
        h_starts[static_cast<size_t>(i)] = total;
        total += h_counts[static_cast<size_t>(i)];
    }
    h_starts[static_cast<size_t>(count_numbers)] = total;
    deviceMemcpy(d_starts, h_starts.data(),
                 sizeof(int) * (count_numbers + 1), deviceMemcpyHostToDevice);
    return total;
#endif
}

#ifndef USE_CPU
static void Stable_Sort_Sci_By_Workload(int sci_numbers,
                                        LJ_CLUSTERED_SCI* d_sci_entries,
                                        int* d_sci_sort_keys)
{
    if (sci_numbers <= 1 || d_sci_entries == NULL || d_sci_sort_keys == NULL)
    {
        return;
    }
    Launch_Device_Kernel(Build_Sci_Workload_Sort_Keys,
                         (sci_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, sci_numbers,
                         d_sci_entries, d_sci_sort_keys);
    Stable_Sort_Device_By_Key(sci_numbers, d_sci_sort_keys, d_sci_entries);
}
#endif

}  // namespace

void LJ_CLUSTER_LAYOUT::Initial(CONTROLLER* controller, const char* module_name,
                                bool ordered_layout_enabled)
{
    constexpr float kDefaultClusteredRebuildSkin = 10.0f;
    float halo_skin = 2.0f;

    enabled = false;
    warn_legacy_ordered_layout = ordered_layout_enabled;
    compression_requested = false;
    cluster_size = 8;
    super_cluster_clusters = 8;
    cornerstone_max_depth = 6;
    cornerstone_leaf_size = 32;
    rebuild_skin = 2.0f;
    rebuild_skin_permit = 0.5f;
    rebuild_dirty = true;
    cache_ready = false;
    cached_cutoff = -1.0f;
    payload_build_time_recorder =
        controller->Get_Time_Recorder("clustered payload build");

    if (controller->Command_Choice("LJ", "direct_kernel", "clustered"))
    {
        enabled = true;
    }
    else if (controller->Command_Choice("LJ", "direct_kernel", "legacy"))
    {
        enabled = false;
    }
    if (enabled)
    {
        rebuild_skin = kDefaultClusteredRebuildSkin;
        halo_skin = kDefaultClusteredRebuildSkin;
    }

    if (controller->Command_Exist("LJ", "cluster_size"))
    {
        controller->Check_Int("LJ", "cluster_size", module_name);
        cluster_size = atoi(controller->Command("LJ", "cluster_size"));
    }
    if (controller->Command_Exist("LJ", "super_cluster_clusters"))
    {
        controller->Check_Int("LJ", "super_cluster_clusters", module_name);
        super_cluster_clusters =
            atoi(controller->Command("LJ", "super_cluster_clusters"));
    }
    if (controller->Command_Exist("LJ", "cornerstone_max_depth"))
    {
        controller->Check_Int("LJ", "cornerstone_max_depth", module_name);
        cornerstone_max_depth =
            atoi(controller->Command("LJ", "cornerstone_max_depth"));
    }
    if (controller->Command_Exist("LJ", "cornerstone_leaf_size"))
    {
        controller->Check_Int("LJ", "cornerstone_leaf_size", module_name);
        cornerstone_leaf_size =
            atoi(controller->Command("LJ", "cornerstone_leaf_size"));
    }
    if (controller->Command_Exist("LJ", "clustered_compression"))
    {
        compression_requested =
            controller->Get_Bool("LJ", "clustered_compression", module_name);
    }
    if (controller->Command_Exist("LJ", "cpu_simd"))
    {
#ifndef USE_CPU
        if (enabled)
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "LJ_CLUSTER_LAYOUT::Initial",
                "Reason:\n\t clustered direct LJ on GPU requires a "
                "device-native payload build; remove the deprecated "
                "LJ.cpu_simd setting.\n");
        }
#endif
    }
    if (controller->Command_Exist("skin"))
    {
        controller->Check_Float("skin", module_name);
        halo_skin = atof(controller->Command("skin"));
        rebuild_skin = halo_skin;
    }
    if (controller->Command_Exist("LJ", "clustered_rebuild_skin"))
    {
        controller->Check_Float("LJ", "clustered_rebuild_skin", module_name);
        rebuild_skin = atof(controller->Command("LJ", "clustered_rebuild_skin"));
        if (enabled && rebuild_skin > halo_skin)
        {
            controller->printf(
                "    Clustered LJ rebuild_skin %.2f exceeds halo skin %.2f; "
                "clamping to %.2f.\n",
                rebuild_skin, halo_skin, halo_skin);
            rebuild_skin = halo_skin;
        }
    }
    if (controller->Command_Exist("neighbor_list", "skin_permit"))
    {
        controller->Check_Float("neighbor_list", "skin_permit", module_name);
        rebuild_skin_permit =
            atof(controller->Command("neighbor_list", "skin_permit"));
    }

    cluster_size = std::max(1, cluster_size);
    super_cluster_clusters = std::max(1, super_cluster_clusters);
    cornerstone_max_depth = std::max(1, std::min(21, cornerstone_max_depth));
    cornerstone_leaf_size = std::max(1, cornerstone_leaf_size);
    rebuild_skin = fmaxf(0.0f, rebuild_skin);
    rebuild_skin_permit = fmaxf(0.0f, rebuild_skin_permit);

    if (enabled && cluster_size != 8)
    {
        controller->printf(
            "    Clustered LJ currently supports only cluster_size=8; "
            "override %d is ignored.\n",
            cluster_size);
        cluster_size = 8;
    }
    if (enabled && super_cluster_clusters != 8)
    {
        controller->printf(
            "    Clustered LJ currently supports only "
            "super_cluster_clusters=8; override %d is ignored.\n",
            super_cluster_clusters);
        super_cluster_clusters = 8;
    }
    if (enabled && compression_requested)
    {
        controller->printf(
            "    Clustered LJ compression is not implemented yet; "
            "continuing with uncompressed pair storage.\n");
    }
    if (enabled && ordered_layout_enabled)
    {
        controller->printf(
            "    Clustered direct LJ ignores ordered_layout and uses backend "
            "cluster builds instead.\n");
    }
    if (enabled)
    {
        controller->printf(
            "    direct_kernel: clustered (cluster_size=%d "
            "super_cluster_clusters=%d depth=%d leaf_size=%d reuse_skin=%.2f "
            "skin_permit=%.2f)\n",
            cluster_size, super_cluster_clusters, cornerstone_max_depth,
            cornerstone_leaf_size, rebuild_skin, rebuild_skin_permit);
    }
}

void LJ_CLUSTER_LAYOUT::Refresh_Metadata(int input_local_atom_numbers,
                                         int input_direct_local_atom_numbers,
                                         int input_ghost_numbers,
                                         const int* d_input_excluded_list_start,
                                         const int* d_input_excluded_list,
                                         const int* d_input_excluded_numbers)
{
    local_atom_numbers = input_local_atom_numbers;
    direct_local_atom_numbers =
        IntMin(local_atom_numbers, IntMax(0, input_direct_local_atom_numbers));
    ghost_numbers = input_ghost_numbers;
    total_atom_numbers = local_atom_numbers + ghost_numbers;
    if (!enabled)
    {
        return;
    }
    d_excluded_list_start = d_input_excluded_list_start;
    d_excluded_list = d_input_excluded_list;
    d_excluded_numbers = d_input_excluded_numbers;
    rebuild_dirty = true;
    Initialize_Cornerstone_State(this);
}

void LJ_CLUSTER_LAYOUT::Build(const VECTOR* crd, LTMatrix3 cell,
                              LTMatrix3 rcell, float cutoff)
{
    if (!enabled)
    {
        return;
    }
    total_atom_numbers = local_atom_numbers + ghost_numbers;
    if (total_atom_numbers <= 0)
    {
        cluster_numbers = 0;
        super_cluster_numbers = 0;
        local_cluster_numbers = 0;
        sci_numbers = 0;
        cjpacked_numbers = 0;
        candidate_leaf_numbers = 0;
        exclusion_pool_numbers = 0;
        cache_ready = false;
        cached_cutoff = -1.0f;
        return;
    }

    Initialize_Cornerstone_State(this);
    if (!Clustered_Build_Is_Needed(this, crd, cell, rcell, cutoff))
    {
        return;
    }
    ClusteredRecorderScope payload_build_scope(payload_build_time_recorder);

    Reset_Build_Buffers(this);
    local_cluster_numbers = 0;
    sci_numbers = 0;
    cjpacked_numbers = 0;
    candidate_leaf_numbers = 0;
    exclusion_pool_numbers = 0;
    const float build_cutoff = cutoff + rebuild_skin;

    Reserve_Device_U64_Buffer(total_atom_numbers, &d_sort_keys,
                              &permutation_capacity);
    Reserve_Device_Int_Buffer(total_atom_numbers, &d_sort_permutation,
                              &permutation_capacity);

    Launch_Device_Kernel(Build_Cornerstone_Sort_Keys,
                         (total_atom_numbers + CONTROLLER::device_max_thread -
                          1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         total_atom_numbers, crd, rcell, d_sort_keys,
                         d_sort_permutation);

#ifndef USE_CPU
    Stable_Sort_Device_By_Key(total_atom_numbers, d_sort_keys,
                              d_sort_permutation);
#else
    std::vector<uint64_t> h_keys(static_cast<size_t>(total_atom_numbers));
    std::vector<int> h_perm(static_cast<size_t>(total_atom_numbers));
    deviceMemcpy(h_keys.data(), d_sort_keys, sizeof(uint64_t) * total_atom_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_perm.data(), d_sort_permutation,
                 sizeof(int) * total_atom_numbers, deviceMemcpyDeviceToHost);
    std::vector<int> order(static_cast<size_t>(total_atom_numbers));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int lhs, int rhs)
                     {
                         if (h_keys[static_cast<size_t>(lhs)] !=
                             h_keys[static_cast<size_t>(rhs)])
                         {
                             return h_keys[static_cast<size_t>(lhs)] <
                                    h_keys[static_cast<size_t>(rhs)];
                         }
                         return h_perm[static_cast<size_t>(lhs)] <
                                h_perm[static_cast<size_t>(rhs)];
                     });
    std::vector<uint64_t> h_keys_sorted(static_cast<size_t>(total_atom_numbers));
    std::vector<int> h_perm_sorted(static_cast<size_t>(total_atom_numbers));
    for (int i = 0; i < total_atom_numbers; i += 1)
    {
        h_keys_sorted[static_cast<size_t>(i)] =
            h_keys[static_cast<size_t>(order[static_cast<size_t>(i)])];
        h_perm_sorted[static_cast<size_t>(i)] =
            h_perm[static_cast<size_t>(order[static_cast<size_t>(i)])];
    }
    deviceMemcpy(d_sort_keys, h_keys_sorted.data(),
                 sizeof(uint64_t) * total_atom_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_sort_permutation, h_perm_sorted.data(),
                 sizeof(int) * total_atom_numbers, deviceMemcpyHostToDevice);
#endif

    Build_Cornerstone_Tree(this);
    const int leaf_numbers = cornerstone_state->octree.numLeafNodes;
    if (leaf_numbers <= 0)
    {
        cluster_numbers = 0;
        super_cluster_numbers = 0;
        Commit_Clustered_Build_Cache(this, crd, cutoff);
        return;
    }

    Reserve_Device_Int_Buffer(leaf_numbers + 1, &d_leaf_atom_offsets,
                              &leaf_capacity);
    Launch_Device_Kernel(Copy_UInt_To_Int,
                         (leaf_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         leaf_numbers, rawPtr(cornerstone_state->leaf_counts),
                         d_leaf_atom_offsets);
    Exclusive_Scan_Counts(leaf_numbers, d_leaf_atom_offsets, d_leaf_atom_offsets);

    Reserve_Device_Int_Buffer(leaf_numbers + 1, &d_leaf_cluster_offsets,
                              &leaf_capacity);
    Launch_Device_Kernel(Build_Leaf_Cluster_Counts,
                         (leaf_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         leaf_numbers, cluster_size, d_leaf_atom_offsets,
                         d_leaf_cluster_offsets);
    cluster_numbers = Exclusive_Scan_Counts(leaf_numbers, d_leaf_cluster_offsets,
                                            d_leaf_cluster_offsets);
    if (cluster_numbers <= 0)
    {
        super_cluster_numbers = 0;
        Commit_Clustered_Build_Cache(this, crd, cutoff);
        return;
    }

    Reserve_Device_Int_Buffer(cluster_numbers + 1, &d_cluster_offsets,
                              &cluster_capacity);
    Reserve_Device_UInt_Buffer(cluster_numbers, &d_cluster_valid_masks,
                               &cluster_capacity);
    Reserve_Device_UInt_Buffer(cluster_numbers, &d_cluster_local_masks,
                               &cluster_capacity);
    Reserve_Device_Vector_Buffer(cluster_numbers, &d_cluster_centers,
                                 &cluster_capacity);
    Reserve_Device_Float_Buffer(cluster_numbers, &d_cluster_radii,
                                &cluster_capacity);

    Launch_Device_Kernel(Build_Leaf_Metadata_And_Clusters,
                         (leaf_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         leaf_numbers, direct_local_atom_numbers, cluster_size,
                         build_cutoff, d_sort_permutation, crd, cell, rcell,
                         d_leaf_atom_offsets, d_leaf_cluster_offsets,
                         d_cluster_offsets, d_cluster_valid_masks,
                         d_cluster_local_masks, d_cluster_centers, d_cluster_radii);

    super_cluster_numbers =
        (cluster_numbers + super_cluster_clusters - 1) / super_cluster_clusters;
    Reserve_Device_Int_Buffer(super_cluster_numbers + 1, &d_super_cluster_offsets,
                              &super_cluster_capacity);
    Launch_Device_Kernel(Build_Fixed_Group_Offsets,
                         (super_cluster_numbers + 1 +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         super_cluster_numbers + 1, super_cluster_clusters,
                         cluster_numbers, d_super_cluster_offsets);

    Reserve_Device_Int_Buffer(super_cluster_numbers, &d_super_cluster_has_local,
                              &super_cluster_capacity);
    Reserve_Device_Vector_Buffer(super_cluster_numbers, &d_super_cluster_centers,
                                 &super_cluster_capacity);
    Reserve_Device_Vector_Buffer(super_cluster_numbers, &d_super_cluster_sizes,
                                 &super_cluster_capacity);
    Launch_Device_Kernel(Build_Supercluster_Metadata,
                         (super_cluster_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         super_cluster_numbers, super_cluster_clusters,
                         build_cutoff,
                         rcell, d_super_cluster_offsets, d_cluster_valid_masks,
                         d_cluster_local_masks, d_cluster_centers,
                         d_cluster_radii, d_super_cluster_has_local,
                         d_super_cluster_centers, d_super_cluster_sizes);

    Reserve_Device_Int_Buffer(super_cluster_numbers, &d_sci_candidate_leaf_counts,
                              &sci_capacity);
    Reserve_Device_Int_Buffer(super_cluster_numbers + 1,
                              &d_sci_candidate_leaf_offsets, &sci_capacity);
    Launch_Device_Kernel(Build_Local_Supercluster_Flags,
                         (super_cluster_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         super_cluster_numbers, d_super_cluster_has_local,
                         d_sci_candidate_leaf_counts);
    sci_numbers = Exclusive_Scan_Counts(super_cluster_numbers,
                                        d_sci_candidate_leaf_counts,
                                        d_sci_candidate_leaf_offsets);
    if (sci_numbers <= 0)
    {
        cjpacked_numbers = 0;
        candidate_leaf_numbers = 0;
        exclusion_pool_numbers = 0;
        Commit_Clustered_Build_Cache(this, crd, cutoff);
        return;
    }

    Reserve_Device_Int_Buffer(sci_numbers, &d_sci_supercluster_ids,
                              &sci_capacity);
    Launch_Device_Kernel(Fill_Local_Supercluster_Ids,
                         (super_cluster_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         super_cluster_numbers, d_super_cluster_has_local,
                         d_sci_candidate_leaf_offsets, d_sci_supercluster_ids);

    Launch_Device_Kernel(
        Count_Supercluster_Candidate_Leaves,
        (sci_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, sci_numbers,
        d_sci_supercluster_ids, d_super_cluster_centers, d_super_cluster_sizes,
        rawPtr(cornerstone_state->octree.prefixes),
        rawPtr(cornerstone_state->octree.childOffsets),
        rawPtr(cornerstone_state->octree.parents),
        rawPtr(cornerstone_state->octree.internalToLeaf),
        d_sci_candidate_leaf_counts);
    candidate_leaf_numbers = Exclusive_Scan_Counts(
        sci_numbers, d_sci_candidate_leaf_counts, d_sci_candidate_leaf_offsets);
    if (candidate_leaf_numbers <= 0)
    {
        cjpacked_numbers = 0;
        exclusion_pool_numbers = 0;
        Commit_Clustered_Build_Cache(this, crd, cutoff);
        return;
    }

    Reserve_Device_Int_Buffer(candidate_leaf_numbers, &d_sci_candidate_leaf_ids,
                              &candidate_leaf_capacity);
    Launch_Device_Kernel(
        Fill_Supercluster_Candidate_Leaves,
        (sci_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, sci_numbers,
        d_sci_supercluster_ids, d_super_cluster_centers, d_super_cluster_sizes,
        rawPtr(cornerstone_state->octree.prefixes),
        rawPtr(cornerstone_state->octree.childOffsets),
        rawPtr(cornerstone_state->octree.parents),
        rawPtr(cornerstone_state->octree.internalToLeaf),
        d_sci_candidate_leaf_offsets, d_sci_candidate_leaf_ids);

    const int candidate_sci_numbers = sci_numbers;
#ifdef USE_CPU
    HostClusteredBuildInput host_input = {};
    host_input.total_atom_numbers = total_atom_numbers;
    host_input.leaf_numbers = leaf_numbers;
    host_input.super_cluster_clusters = super_cluster_clusters;
    host_input.local_atom_numbers = local_atom_numbers;
    host_input.cluster_size = cluster_size;
    host_input.candidate_sci_numbers = candidate_sci_numbers;
    host_input.cutoff = cutoff;
    host_input.cell = cell;
    host_input.rcell = rcell;
    host_input.permutation.resize((size_t)total_atom_numbers);
    host_input.cluster_offsets.resize((size_t)cluster_numbers + 1);
    host_input.cluster_valid_masks.resize((size_t)cluster_numbers);
    host_input.cluster_local_masks.resize((size_t)cluster_numbers);
    host_input.cluster_centers.resize((size_t)cluster_numbers);
    host_input.cluster_radii.resize((size_t)cluster_numbers);
    host_input.leaf_cluster_offsets.resize((size_t)leaf_numbers + 1);
    host_input.super_cluster_offsets.resize((size_t)super_cluster_numbers + 1);
    host_input.super_cluster_centers.resize((size_t)super_cluster_numbers);
    host_input.sci_supercluster_ids.resize((size_t)candidate_sci_numbers);
    host_input.candidate_leaf_offsets.resize((size_t)candidate_sci_numbers + 1);
    host_input.candidate_leaf_ids.resize((size_t)candidate_leaf_numbers);
    host_input.excluded_list_start.resize((size_t)local_atom_numbers, 0);
    host_input.excluded_numbers.resize((size_t)local_atom_numbers, 0);

    deviceMemcpy(host_input.permutation.data(), d_sort_permutation,
                 sizeof(int) * total_atom_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.cluster_offsets.data(), d_cluster_offsets,
                 sizeof(int) * (cluster_numbers + 1),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.cluster_valid_masks.data(), d_cluster_valid_masks,
                 sizeof(unsigned int) * cluster_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.cluster_local_masks.data(), d_cluster_local_masks,
                 sizeof(unsigned int) * cluster_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.cluster_centers.data(), d_cluster_centers,
                 sizeof(VECTOR) * cluster_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.cluster_radii.data(), d_cluster_radii,
                 sizeof(float) * cluster_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.leaf_cluster_offsets.data(), d_leaf_cluster_offsets,
                 sizeof(int) * (leaf_numbers + 1),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.super_cluster_offsets.data(), d_super_cluster_offsets,
                 sizeof(int) * (super_cluster_numbers + 1),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.super_cluster_centers.data(),
                 d_super_cluster_centers,
                 sizeof(VECTOR) * super_cluster_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.sci_supercluster_ids.data(), d_sci_supercluster_ids,
                 sizeof(int) * candidate_sci_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.candidate_leaf_offsets.data(),
                 d_sci_candidate_leaf_offsets,
                 sizeof(int) * (candidate_sci_numbers + 1),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.candidate_leaf_ids.data(), d_sci_candidate_leaf_ids,
                 sizeof(int) * candidate_leaf_numbers,
                 deviceMemcpyDeviceToHost);
    if (local_atom_numbers > 0 && d_excluded_list_start != NULL &&
        d_excluded_numbers != NULL)
    {
        deviceMemcpy(host_input.excluded_list_start.data(), d_excluded_list_start,
                     sizeof(int) * local_atom_numbers,
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(host_input.excluded_numbers.data(), d_excluded_numbers,
                     sizeof(int) * local_atom_numbers,
                     deviceMemcpyDeviceToHost);
        const int total_excluded =
            host_input.excluded_list_start.back() +
            host_input.excluded_numbers.back();
        host_input.excluded_list.resize((size_t)IntMax(total_excluded, 0));
        if (total_excluded > 0 && d_excluded_list != NULL)
        {
            deviceMemcpy(host_input.excluded_list.data(), d_excluded_list,
                         sizeof(int) * total_excluded,
                         deviceMemcpyDeviceToHost);
        }
    }

    std::vector<LJ_CLUSTERED_SCI> host_scis;
    std::vector<LJ_CLUSTERED_CJ_PACKED> host_cjpacked;
    std::vector<unsigned long long> host_exclusion_pool;
    Build_Nbnxm_Payload_On_Host(host_input, &host_scis, &host_cjpacked,
                                &host_exclusion_pool);
    std::stable_sort(
        host_scis.begin(), host_scis.end(),
        [](const LJ_CLUSTERED_SCI& lhs, const LJ_CLUSTERED_SCI& rhs)
        {
            return (lhs.cjpacked_end - lhs.cjpacked_begin) >
                   (rhs.cjpacked_end - rhs.cjpacked_begin);
        });

    sci_numbers = static_cast<int>(host_scis.size());
    cjpacked_numbers = static_cast<int>(host_cjpacked.size());
    exclusion_pool_numbers = static_cast<int>(host_exclusion_pool.size());
    if (sci_numbers <= 0 || cjpacked_numbers <= 0)
    {
        Commit_Clustered_Build_Cache(this, crd, cutoff);
        return;
    }

    Reserve_Device_Buffer(sci_numbers, &d_nbnxm_sci, &nbnxm_sci_capacity);
    Reserve_Device_Buffer(cjpacked_numbers, &d_nbnxm_cjpacked,
                          &nbnxm_cjpacked_capacity);
    deviceMemcpy(d_nbnxm_sci, host_scis.data(),
                 sizeof(LJ_CLUSTERED_SCI) * sci_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_nbnxm_cjpacked, host_cjpacked.data(),
                 sizeof(LJ_CLUSTERED_CJ_PACKED) * cjpacked_numbers,
                 deviceMemcpyHostToDevice);
    if (exclusion_pool_numbers > 0)
    {
        Reserve_Device_PairMask_Buffer(exclusion_pool_numbers,
                                       &d_exclusion_mask_pool,
                                       &exclusion_capacity);
        deviceMemcpy(d_exclusion_mask_pool, host_exclusion_pool.data(),
                     sizeof(unsigned long long) * exclusion_pool_numbers,
                     deviceMemcpyHostToDevice);
    }
#else
    const int sci_shift_numbers =
        candidate_sci_numbers * kClusteredShiftCount;
    Reserve_Device_Int_Buffer(sci_shift_numbers, &d_sci_shift_flags,
                              &sci_shift_capacity);
    Reserve_Device_Int_Buffer(sci_shift_numbers, &d_cjpacked_counts,
                              &sci_shift_capacity);
    Reserve_Device_Int_Buffer(sci_shift_numbers, &d_exclusion_counts,
                              &sci_shift_capacity);
    Reserve_Device_Int_Buffer(sci_shift_numbers + 1, &d_sci_shift_offsets,
                              &sci_shift_capacity);
    Reserve_Device_Int_Buffer(sci_shift_numbers + 1, &d_cjpacked_group_offsets,
                              &sci_shift_capacity);
    Reserve_Device_Int_Buffer(sci_shift_numbers + 1, &d_exclusion_offsets,
                              &sci_shift_capacity);

    Launch_Device_Kernel(
        Count_Nbnxm_Payload_From_Candidate_Leaves,
        (sci_shift_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, candidate_sci_numbers,
        sci_shift_numbers, cluster_size, super_cluster_clusters,
        local_atom_numbers, cutoff, cell, rcell, d_sort_permutation,
        d_cluster_offsets, d_leaf_cluster_offsets, d_super_cluster_offsets,
        d_sci_supercluster_ids, d_super_cluster_centers,
        d_sci_candidate_leaf_offsets, d_sci_candidate_leaf_ids,
        d_cluster_valid_masks, d_cluster_local_masks, d_cluster_centers,
        d_cluster_radii, d_excluded_list_start, d_excluded_list,
        d_excluded_numbers, d_sci_shift_flags, d_cjpacked_counts,
        d_exclusion_counts);

    sci_numbers = Exclusive_Scan_Counts(sci_shift_numbers, d_sci_shift_flags,
                                        d_sci_shift_offsets);
    cjpacked_numbers = Exclusive_Scan_Counts(
        sci_shift_numbers, d_cjpacked_counts, d_cjpacked_group_offsets);
    exclusion_pool_numbers =
        Exclusive_Scan_Counts(sci_shift_numbers, d_exclusion_counts,
                              d_exclusion_offsets);
    if (sci_numbers <= 0 || cjpacked_numbers <= 0)
    {
        Commit_Clustered_Build_Cache(this, crd, cutoff);
        return;
    }

    Reserve_Device_Buffer(sci_numbers, &d_nbnxm_sci, &nbnxm_sci_capacity);
    Reserve_Device_Buffer(cjpacked_numbers, &d_nbnxm_cjpacked,
                          &nbnxm_cjpacked_capacity);
    if (exclusion_pool_numbers > 0)
    {
        Reserve_Device_PairMask_Buffer(exclusion_pool_numbers,
                                       &d_exclusion_mask_pool,
                                       &exclusion_capacity);
    }

    Launch_Device_Kernel(
        Fill_Nbnxm_Payload_From_Candidate_Leaves,
        (sci_shift_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, candidate_sci_numbers,
        sci_shift_numbers, cluster_size, super_cluster_clusters,
        local_atom_numbers, cutoff, cell, rcell, d_sort_permutation,
        d_cluster_offsets, d_leaf_cluster_offsets, d_super_cluster_offsets,
        d_sci_supercluster_ids, d_super_cluster_centers,
        d_sci_candidate_leaf_offsets, d_sci_candidate_leaf_ids,
        d_cluster_valid_masks, d_cluster_local_masks, d_cluster_centers,
        d_cluster_radii, d_excluded_list_start, d_excluded_list,
        d_excluded_numbers, d_sci_shift_flags, d_sci_shift_offsets,
        d_cjpacked_counts, d_cjpacked_group_offsets, d_exclusion_offsets,
        d_nbnxm_sci, d_nbnxm_cjpacked, d_exclusion_mask_pool);
    Stable_Sort_Sci_By_Workload(sci_numbers, d_nbnxm_sci, d_sci_shift_flags);
#endif
    Commit_Clustered_Build_Cache(this, crd, cutoff);
}

void LJ_CLUSTER_LAYOUT::Clear()
{
    Free_Single_Device_Pointer((void**)&d_sort_permutation);
    Free_Single_Device_Pointer((void**)&d_sort_keys);
    Free_Single_Device_Pointer((void**)&d_cached_crd);
    Free_Single_Device_Pointer((void**)&d_need_rebuild);
    Free_Single_Device_Pointer((void**)&d_cluster_offsets);
    Free_Single_Device_Pointer((void**)&d_cluster_valid_masks);
    Free_Single_Device_Pointer((void**)&d_cluster_local_masks);
    Free_Single_Device_Pointer((void**)&d_cluster_centers);
    Free_Single_Device_Pointer((void**)&d_cluster_radii);
    Free_Single_Device_Pointer((void**)&d_leaf_atom_offsets);
    Free_Single_Device_Pointer((void**)&d_leaf_cluster_offsets);
    Free_Single_Device_Pointer((void**)&d_super_cluster_offsets);
    Free_Single_Device_Pointer((void**)&d_super_cluster_has_local);
    Free_Single_Device_Pointer((void**)&d_super_cluster_centers);
    Free_Single_Device_Pointer((void**)&d_super_cluster_sizes);
    Free_Single_Device_Pointer((void**)&d_sci_supercluster_ids);
    Free_Single_Device_Pointer((void**)&d_sci_candidate_leaf_counts);
    Free_Single_Device_Pointer((void**)&d_sci_candidate_leaf_offsets);
    Free_Single_Device_Pointer((void**)&d_sci_candidate_leaf_ids);
    Free_Single_Device_Pointer((void**)&d_cjpacked_counts);
    Free_Single_Device_Pointer((void**)&d_exclusion_counts);
    Free_Single_Device_Pointer((void**)&d_exclusion_offsets);
    Free_Single_Device_Pointer((void**)&d_sci_shift_flags);
    Free_Single_Device_Pointer((void**)&d_sci_shift_offsets);
    Free_Single_Device_Pointer((void**)&d_cjpacked_group_offsets);
    Free_Single_Device_Pointer((void**)&d_exclusion_mask_pool);
    Free_Single_Device_Pointer((void**)&d_nbnxm_sci);
    Free_Single_Device_Pointer((void**)&d_nbnxm_cjpacked);

    delete cornerstone_state;
    cornerstone_state = NULL;

    permutation_capacity = 0;
    cluster_capacity = 0;
    leaf_capacity = 0;
    super_cluster_capacity = 0;
    sci_capacity = 0;
    candidate_leaf_capacity = 0;
    cjpacked_capacity = 0;
    exclusion_scan_capacity = 0;
    exclusion_capacity = 0;
    scan_capacity = 0;
    sci_shift_capacity = 0;
    nbnxm_sci_capacity = 0;
    nbnxm_cjpacked_capacity = 0;
    cached_crd_capacity = 0;
    rebuild_flag_capacity = 0;
    local_atom_numbers = 0;
    direct_local_atom_numbers = 0;
    ghost_numbers = 0;
    total_atom_numbers = 0;
    cluster_numbers = 0;
    super_cluster_numbers = 0;
    local_cluster_numbers = 0;
    sci_numbers = 0;
    cjpacked_numbers = 0;
    candidate_leaf_numbers = 0;
    exclusion_pool_numbers = 0;
    rebuild_dirty = true;
    cache_ready = false;
    cached_cutoff = -1.0f;
    d_excluded_list_start = NULL;
    d_excluded_list = NULL;
    d_excluded_numbers = NULL;
}

void LJ_CLUSTERED_DIRECT_CACHE::Initial(CONTROLLER* controller,
                                        const char* module_name,
                                        bool ordered_layout_enabled)
{
    if (initialized)
    {
        return;
    }
    initialized = true;
    layout.Initial(controller, module_name, ordered_layout_enabled);
    payload_gather_time_recorder =
        controller->Get_Time_Recorder("clustered payload gather");
    direct_kernel_time_recorder =
        controller->Get_Time_Recorder("clustered direct kernel");
}

void LJ_CLUSTERED_DIRECT_CACHE::Refresh_Metadata(
    int local_atom_numbers, int direct_local_atom_numbers, int ghost_numbers,
    const int* d_excluded_list_start, const int* d_excluded_list,
    const int* d_excluded_numbers)
{
    if (!initialized)
    {
        return;
    }
    layout.Refresh_Metadata(local_atom_numbers, direct_local_atom_numbers,
                            ghost_numbers,
                            d_excluded_list_start, d_excluded_list,
                            d_excluded_numbers);
}

void LJ_CLUSTERED_DIRECT_CACHE::Build(const VECTOR* crd, LTMatrix3 cell,
                                      LTMatrix3 rcell, float cutoff)
{
    if (!initialized)
    {
        return;
    }
    layout.Build(crd, cell, rcell, cutoff);
}

void LJ_CLUSTERED_DIRECT_CACHE::Gather_Plain(const VECTOR_LJ* src,
                                             LTMatrix3 cell, LTMatrix3 rcell)
{
    if (!initialized || !layout.Use_Clustered_Direct() ||
        layout.total_atom_numbers <= 0)
    {
        return;
    }
    ClusteredRecorderScope gather_scope(payload_gather_time_recorder);
    Reserve_Device_Buffer(layout.total_atom_numbers, &d_sorted_atom_ids,
                          &scratch_capacity);
    Reserve_Device_Buffer(layout.total_atom_numbers, &d_sorted_xq,
                          &scratch_capacity);
    Reserve_Device_Buffer(layout.total_atom_numbers, &d_sorted_lj_type,
                          &scratch_capacity);
    Launch_Device_Kernel(
        Gather_Sorted_LJ_Direct_Scratch,
        (layout.total_atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout.total_atom_numbers,
        layout.cluster_numbers, layout.d_sort_permutation,
        layout.d_cluster_offsets, layout.d_cluster_centers, cell, rcell, src,
        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type);
}

void LJ_CLUSTERED_DIRECT_CACHE::Gather_Soft_Core(
    const VECTOR_LJ_SOFT_TYPE* src)
{
    if (!initialized || !layout.Use_Clustered_Direct() ||
        layout.total_atom_numbers <= 0)
    {
        return;
    }
    ClusteredRecorderScope gather_scope(payload_gather_time_recorder);
    Reserve_Device_Buffer(layout.total_atom_numbers, &d_sorted_soft_crd,
                          &scratch_capacity);
    Launch_Device_Kernel(
        Gather_Sorted_Soft_Core_Scratch,
        (layout.total_atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout.total_atom_numbers,
        layout.d_sort_permutation, src, d_sorted_soft_crd);
}

void LJ_CLUSTERED_DIRECT_CACHE::Clear()
{
    if (!initialized)
    {
        return;
    }
    Free_Single_Device_Pointer((void**)&d_sorted_atom_ids);
    Free_Single_Device_Pointer((void**)&d_sorted_xq);
    Free_Single_Device_Pointer((void**)&d_sorted_lj_type);
    Free_Single_Device_Pointer((void**)&d_sorted_soft_crd);
    scratch_capacity = 0;
    payload_gather_time_recorder = NULL;
    direct_kernel_time_recorder = NULL;
    layout.Clear();
    initialized = false;
}

namespace
{
LJ_CLUSTERED_DIRECT_CACHE g_shared_clustered_direct_cache;
}

LJ_CLUSTERED_DIRECT_CACHE* Acquire_Shared_LJ_Clustered_Direct_Cache(
    CONTROLLER* controller, const char* module_name,
    bool ordered_layout_enabled)
{
    g_shared_clustered_direct_cache.Initial(controller, module_name,
                                            ordered_layout_enabled);
    return &g_shared_clustered_direct_cache;
}

void Release_Shared_LJ_Clustered_Direct_Cache()
{
    g_shared_clustered_direct_cache.Clear();
}
