#include "gather.h"

#include "../../MD_core/MD_core.h"
#include "../../neighbor_list/provider/pair_shift.h"
#include "../../neighbor_list/provider/runtime.h"
#include "../../neighbor_list/builder/primitives/geometry.cuh"
#include "../LJ_soft_core.h"
#include "../Lennard_Jones_force.h"

extern MD_INFORMATION md_info;

constexpr int kClusteredFusedGatherBlockSize = 256;

struct GatheredClusterGeometry
{
    int start = 0;
    int end = 0;
    VECTOR center = {};
    VECTOR extent = {};
};

static __device__ __forceinline__ VECTOR Gather_Coordinate(const VECTOR* src,
                                                            const int atom_i)
{
    return src[atom_i];
}

template <typename SrcType>
static __device__ __forceinline__ VECTOR Gather_Coordinate(const SrcType* src,
                                                            const int atom_i)
{
    return src[atom_i].crd;
}

#ifdef USE_GPU
static __device__ __forceinline__ device_mask_t Gather_Subgroup_Mask()
{
    const int lane_in_warp = threadIdx.x & (warpSize - 1);
    const int subgroup_base =
        lane_in_warp & ~(kClusteredClusterSize - 1);
    return static_cast<device_mask_t>(0xffu) << subgroup_base;
}

static __device__ __forceinline__ VECTOR Gather_Subgroup_Broadcast(
    const device_mask_t mask, VECTOR value)
{
    value.x = deviceShfl(mask, value.x, 0, kClusteredClusterSize);
    value.y = deviceShfl(mask, value.y, 0, kClusteredClusterSize);
    value.z = deviceShfl(mask, value.z, 0, kClusteredClusterSize);
    return value;
}

static __device__ __forceinline__ VECTOR Gather_Subgroup_Sum(
    const device_mask_t mask, VECTOR value)
{
    for (int delta = kClusteredClusterSize >> 1; delta > 0; delta >>= 1)
    {
        value.x += deviceShflDown(mask, value.x, delta,
                                  kClusteredClusterSize);
        value.y += deviceShflDown(mask, value.y, delta,
                                  kClusteredClusterSize);
        value.z += deviceShflDown(mask, value.z, delta,
                                  kClusteredClusterSize);
    }
    return value;
}

static __device__ __forceinline__ VECTOR Gather_Subgroup_Max(
    const device_mask_t mask, VECTOR value)
{
    for (int delta = kClusteredClusterSize >> 1; delta > 0; delta >>= 1)
    {
        value.x = fmaxf(value.x, deviceShflDown(
                                     mask, value.x, delta,
                                     kClusteredClusterSize));
        value.y = fmaxf(value.y, deviceShflDown(
                                     mask, value.y, delta,
                                     kClusteredClusterSize));
        value.z = fmaxf(value.z, deviceShflDown(
                                     mask, value.z, delta,
                                     kClusteredClusterSize));
    }
    return value;
}

template <typename SrcType>
static __device__ __forceinline__ GatheredClusterGeometry
Gather_Cluster_Geometry_Subgroup(
    const int cluster_i, const int cluster_numbers, const int sublane,
    const device_mask_t subgroup_mask, const int* permutation,
    const int* cluster_offsets, const SrcType* src, const LTMatrix3 cell,
    const LTMatrix3 rcell)
{
    GatheredClusterGeometry geometry;
    const bool valid_cluster = cluster_i < cluster_numbers;
    geometry.start = valid_cluster ? cluster_offsets[cluster_i] : 0;
    geometry.end = valid_cluster ? cluster_offsets[cluster_i + 1] : 0;
    const int count = geometry.end > geometry.start
                          ? geometry.end - geometry.start
                          : 0;
    const bool active = sublane < count;
    VECTOR anchor = {};
    if (sublane == 0 && count > 0)
    {
        anchor = Gather_Coordinate(src, permutation[geometry.start]);
    }
    anchor = Gather_Subgroup_Broadcast(subgroup_mask, anchor);
    VECTOR unwrapped = {};
    if (active)
    {
        const VECTOR pos = Gather_Coordinate(
            src, permutation[geometry.start + sublane]);
        unwrapped = anchor +
                    Get_Periodic_Displacement(pos, anchor, cell, rcell);
    }
    const VECTOR sum = Gather_Subgroup_Sum(subgroup_mask, unwrapped);
    if (sublane == 0 && count > 0)
    {
        geometry.center =
            Get_Periodic_Coordinate(
                (1.0f / static_cast<float>(count)) * sum, cell, rcell);
    }
    geometry.center =
        Gather_Subgroup_Broadcast(subgroup_mask, geometry.center);
    VECTOR lane_extent = {};
    if (active)
    {
        const VECTOR pos = Gather_Coordinate(
            src, permutation[geometry.start + sublane]);
        const VECTOR dr =
            Get_Periodic_Displacement(pos, geometry.center, cell, rcell);
        lane_extent = {fabsf(dr.x), fabsf(dr.y), fabsf(dr.z)};
    }
    geometry.extent = Gather_Subgroup_Max(subgroup_mask, lane_extent);
    return geometry;
}
#else
template <typename SrcType>
static __device__ __forceinline__ GatheredClusterGeometry
Gather_Cluster_Geometry_Sequential(
    const int cluster_i, const int* permutation, const int* cluster_offsets,
    const SrcType* src, const LTMatrix3 cell, const LTMatrix3 rcell)
{
    GatheredClusterGeometry geometry;
    geometry.start = cluster_offsets[cluster_i];
    geometry.end = cluster_offsets[cluster_i + 1];
    const int count = geometry.end > geometry.start
                          ? geometry.end - geometry.start
                          : 0;
    if (count > 0)
    {
        const VECTOR anchor =
            Gather_Coordinate(src, permutation[geometry.start]);
        for (int sorted_i = geometry.start; sorted_i < geometry.end;
             sorted_i += 1)
        {
            const VECTOR pos =
                Gather_Coordinate(src, permutation[sorted_i]);
            geometry.center =
                geometry.center +
                (anchor +
                 Get_Periodic_Displacement(pos, anchor, cell, rcell));
        }
        geometry.center = Get_Periodic_Coordinate(
            (1.0f / static_cast<float>(count)) * geometry.center, cell,
            rcell);
        for (int sorted_i = geometry.start; sorted_i < geometry.end;
             sorted_i += 1)
        {
            const VECTOR pos =
                Gather_Coordinate(src, permutation[sorted_i]);
            const VECTOR dr = Get_Periodic_Displacement(
                pos, geometry.center, cell, rcell);
            geometry.extent.x = fmaxf(geometry.extent.x, fabsf(dr.x));
            geometry.extent.y = fmaxf(geometry.extent.y, fabsf(dr.y));
            geometry.extent.z = fmaxf(geometry.extent.z, fabsf(dr.z));
        }
    }
    return geometry;
}
#endif

static __device__ __forceinline__ void Store_Gathered_Cluster_Geometry(
    const int cluster_i, const GatheredClusterGeometry geometry,
    const LTMatrix3 rcell, VECTOR* cluster_centers,
    VECTOR* cluster_fractional_centers, VECTOR* cluster_fractional_extents)
{
    cluster_centers[cluster_i] = geometry.center;
    Store_Current_Cluster_Fractional_Geometry(
        cluster_i, geometry.center, geometry.extent, rcell,
        cluster_fractional_centers, cluster_fractional_extents);
}

static __device__ __forceinline__ float2 Gather_LJ_Combination(
    const int lj_type, const float2* lj_ab_packed)
{
    float2 comb = {0.0f, 0.0f};
    if (lj_ab_packed != NULL)
    {
        const float2 self_ab = lj_ab_packed[Get_LJ_Type(lj_type, lj_type)];
        comb.x = sqrtf(fmaxf(self_ab.y, 0.0f));
        comb.y = sqrtf(fmaxf(self_ab.x, 0.0f));
    }
    return comb;
}

static __global__ void Gather_Clustered_LJ_Direct_Scratch(
    const int cluster_numbers, const int* permutation,
    const int* cluster_offsets, const LTMatrix3 cell, const LTMatrix3 rcell,
    const VECTOR_LJ* src, VECTOR* cluster_centers,
    VECTOR* cluster_fractional_centers, VECTOR* cluster_fractional_extents,
    int* sorted_atom_ids, float4* sorted_xq, int* sorted_lj_type,
    const float2* lj_ab_packed, float2* sorted_lj_comb)
{
#ifdef USE_GPU
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int cluster_i = global_thread / kClusteredClusterSize;
    const int sublane = threadIdx.x & (kClusteredClusterSize - 1);
    const GatheredClusterGeometry geometry =
        Gather_Cluster_Geometry_Subgroup(
            cluster_i, cluster_numbers, sublane, Gather_Subgroup_Mask(),
            permutation, cluster_offsets, src, cell, rcell);
    if (cluster_i < cluster_numbers && sublane == 0)
    {
        Store_Gathered_Cluster_Geometry(
            cluster_i, geometry, rcell, cluster_centers,
            cluster_fractional_centers, cluster_fractional_extents);
    }
    const int sorted_i = geometry.start + sublane;
    if (cluster_i < cluster_numbers && sorted_i < geometry.end)
    {
        const int atom_i = permutation[sorted_i];
        const VECTOR_LJ atom = src[atom_i];
        const VECTOR shifted_crd =
            geometry.center + Get_Periodic_Displacement(
                                  atom.crd, geometry.center, cell, rcell);
        sorted_atom_ids[sorted_i] = atom_i;
        sorted_xq[sorted_i] = {shifted_crd.x, shifted_crd.y, shifted_crd.z,
                               atom.charge};
        sorted_lj_type[sorted_i] = atom.LJ_type;
        if (sorted_lj_comb != NULL)
        {
            sorted_lj_comb[sorted_i] =
                Gather_LJ_Combination(atom.LJ_type, lj_ab_packed);
        }
    }
#else
    SIMPLE_DEVICE_FOR(cluster_i, cluster_numbers)
    {
        const GatheredClusterGeometry geometry =
            Gather_Cluster_Geometry_Sequential(
                cluster_i, permutation, cluster_offsets, src, cell, rcell);
        Store_Gathered_Cluster_Geometry(
            cluster_i, geometry, rcell, cluster_centers,
            cluster_fractional_centers, cluster_fractional_extents);
        for (int sorted_i = geometry.start; sorted_i < geometry.end;
             sorted_i += 1)
        {
            const int atom_i = permutation[sorted_i];
            const VECTOR_LJ atom = src[atom_i];
            const VECTOR shifted_crd =
                geometry.center +
                Get_Periodic_Displacement(atom.crd, geometry.center, cell,
                                          rcell);
            sorted_atom_ids[sorted_i] = atom_i;
            sorted_xq[sorted_i] = {shifted_crd.x, shifted_crd.y,
                                   shifted_crd.z, atom.charge};
            sorted_lj_type[sorted_i] = atom.LJ_type;
            if (sorted_lj_comb != NULL)
            {
                sorted_lj_comb[sorted_i] =
                    Gather_LJ_Combination(atom.LJ_type, lj_ab_packed);
            }
        }
    }
#endif
}

static __global__ void Gather_Clustered_LJ_Direct_Scratch_From_Plain(
    const int cluster_numbers, const int* permutation,
    const int* cluster_offsets, const LTMatrix3 cell, const LTMatrix3 rcell,
    const VECTOR* crd, const float* charge, const VECTOR_LJ* lj_type_src,
    VECTOR* cluster_centers, VECTOR* cluster_fractional_centers,
    VECTOR* cluster_fractional_extents, int* sorted_atom_ids,
    float4* sorted_xq, int* sorted_lj_type,
    const float2* lj_ab_packed, float2* sorted_lj_comb)
{
#ifdef USE_GPU
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int cluster_i = global_thread / kClusteredClusterSize;
    const int sublane = threadIdx.x & (kClusteredClusterSize - 1);
    const GatheredClusterGeometry geometry =
        Gather_Cluster_Geometry_Subgroup(
            cluster_i, cluster_numbers, sublane, Gather_Subgroup_Mask(),
            permutation, cluster_offsets, crd, cell, rcell);
    if (cluster_i < cluster_numbers && sublane == 0)
    {
        Store_Gathered_Cluster_Geometry(
            cluster_i, geometry, rcell, cluster_centers,
            cluster_fractional_centers, cluster_fractional_extents);
    }
    const int sorted_i = geometry.start + sublane;
    if (cluster_i < cluster_numbers && sorted_i < geometry.end)
    {
        const int atom_i = permutation[sorted_i];
        const VECTOR atom_crd = crd[atom_i];
        const VECTOR shifted_crd =
            geometry.center + Get_Periodic_Displacement(
                                  atom_crd, geometry.center, cell, rcell);
        sorted_atom_ids[sorted_i] = atom_i;
        sorted_xq[sorted_i] = {shifted_crd.x, shifted_crd.y, shifted_crd.z,
                               charge[atom_i]};
        const int lj_type = lj_type_src[atom_i].LJ_type;
        sorted_lj_type[sorted_i] = lj_type;
        if (sorted_lj_comb != NULL)
        {
            sorted_lj_comb[sorted_i] =
                Gather_LJ_Combination(lj_type, lj_ab_packed);
        }
    }
#else
    SIMPLE_DEVICE_FOR(cluster_i, cluster_numbers)
    {
        const GatheredClusterGeometry geometry =
            Gather_Cluster_Geometry_Sequential(
                cluster_i, permutation, cluster_offsets, crd, cell, rcell);
        Store_Gathered_Cluster_Geometry(
            cluster_i, geometry, rcell, cluster_centers,
            cluster_fractional_centers, cluster_fractional_extents);
        for (int sorted_i = geometry.start; sorted_i < geometry.end;
             sorted_i += 1)
        {
            const int atom_i = permutation[sorted_i];
            const VECTOR atom_crd = crd[atom_i];
            const VECTOR shifted_crd =
                geometry.center + Get_Periodic_Displacement(
                                      atom_crd, geometry.center, cell, rcell);
            sorted_atom_ids[sorted_i] = atom_i;
            sorted_xq[sorted_i] = {shifted_crd.x, shifted_crd.y,
                                   shifted_crd.z, charge[atom_i]};
            const int lj_type = lj_type_src[atom_i].LJ_type;
            sorted_lj_type[sorted_i] = lj_type;
            if (sorted_lj_comb != NULL)
            {
                sorted_lj_comb[sorted_i] =
                    Gather_LJ_Combination(lj_type, lj_ab_packed);
            }
        }
    }
#endif
}

namespace
{

using clustered_neighbor_runtime::Note_Clustered_Step_Counter;
using clustered_neighbor_runtime::RecorderScope;
using clustered_neighbor_runtime::Reserve_Device_Buffer;

static __global__ void Gather_Clustered_Soft_Core_Scratch(
    const int cluster_numbers, const int* permutation,
    const int* cluster_offsets, const LTMatrix3 cell, const LTMatrix3 rcell,
    const VECTOR_LJ_SOFT_TYPE* src, VECTOR* cluster_centers,
    VECTOR* cluster_fractional_centers, VECTOR* cluster_fractional_extents,
    VECTOR_LJ_SOFT_TYPE* dest)
{
#ifdef USE_GPU
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int cluster_i = global_thread / kClusteredClusterSize;
    const int sublane = threadIdx.x & (kClusteredClusterSize - 1);
    const GatheredClusterGeometry geometry =
        Gather_Cluster_Geometry_Subgroup(
            cluster_i, cluster_numbers, sublane, Gather_Subgroup_Mask(),
            permutation, cluster_offsets, src, cell, rcell);
    if (cluster_i < cluster_numbers && sublane == 0)
    {
        Store_Gathered_Cluster_Geometry(
            cluster_i, geometry, rcell, cluster_centers,
            cluster_fractional_centers, cluster_fractional_extents);
    }
    const int sorted_i = geometry.start + sublane;
    if (cluster_i < cluster_numbers && sorted_i < geometry.end)
    {
        VECTOR_LJ_SOFT_TYPE atom = src[permutation[sorted_i]];
        atom.crd = geometry.center + Get_Periodic_Displacement(
                                         atom.crd, geometry.center, cell,
                                         rcell);
        dest[sorted_i] = atom;
    }
#else
    SIMPLE_DEVICE_FOR(cluster_i, cluster_numbers)
    {
        const GatheredClusterGeometry geometry =
            Gather_Cluster_Geometry_Sequential(
                cluster_i, permutation, cluster_offsets, src, cell, rcell);
        Store_Gathered_Cluster_Geometry(
            cluster_i, geometry, rcell, cluster_centers,
            cluster_fractional_centers, cluster_fractional_extents);
        for (int sorted_i = geometry.start; sorted_i < geometry.end;
             sorted_i += 1)
        {
            VECTOR_LJ_SOFT_TYPE atom = src[permutation[sorted_i]];
            atom.crd = geometry.center + Get_Periodic_Displacement(
                                             atom.crd, geometry.center, cell,
                                             rcell);
            dest[sorted_i] = atom;
        }
    }
#endif
}

static void Reserve_Plain_Gather_Scratch(LJClusteredWorkspace* workspace,
                                         int total_atom_numbers)
{
    Reserve_Device_Buffer(total_atom_numbers, &workspace->d_sorted_atom_ids,
                          &workspace->sorted_atom_ids_capacity);
    Reserve_Device_Buffer(total_atom_numbers, &workspace->d_sorted_xq,
                          &workspace->sorted_xq_capacity);
    Reserve_Device_Buffer(total_atom_numbers, &workspace->d_sorted_lj_type,
                          &workspace->sorted_lj_type_capacity);
    Reserve_Device_Buffer(total_atom_numbers, &workspace->d_sorted_lj_comb,
                          &workspace->sorted_lj_comb_capacity);
    Reserve_Device_Buffer(total_atom_numbers, &workspace->d_sorted_frc,
                          &workspace->sorted_frc_capacity);
}

static void Refresh_Gather_Dependent_Metadata(
    ClusteredNeighborProvider* provider, ClusteredGatherBinding* spatial,
    LTMatrix3 cell, LTMatrix3 rcell)
{
    provider->PublishGatheredGeometry(spatial);
    (void)cell;
#ifndef USE_CPU
    Refresh_Gmxpacked_Pair_Shift_Metadata_If_Needed(provider, rcell);
#else
    (void)rcell;
#endif
}

}  // namespace

void LJClusteredWorkspace::Gather_Plain(const VECTOR* crd,
                                        const float* charge,
                                        const VECTOR_LJ* lj_type_src,
                                        LTMatrix3 cell, LTMatrix3 rcell,
                                        const float2* lj_ab_packed)
{
    if (provider_ == NULL)
    {
        return;
    }
    ClusteredGatherBinding spatial;
    if (!provider_->AcquireGatherBinding(&spatial) ||
        charge == NULL || lj_type_src == NULL)
    {
        return;
    }
#ifndef USE_CPU
    provider_->BindWorkingDevice();
#endif
    Note_Clustered_Step_Counter(md_info.sys.steps, &coordinate_gather_step,
                                &coordinate_gather_count_this_step,
                                &coordinate_gather_count_total);
    RecorderScope gather_scope(payload_gather_time_recorder);
    Reserve_Plain_Gather_Scratch(this, spatial.total_atom_numbers);
    Launch_Device_Kernel(
        Gather_Clustered_LJ_Direct_Scratch_From_Plain,
        (spatial.cluster_numbers * kClusteredClusterSize +
         kClusteredFusedGatherBlockSize - 1) /
            kClusteredFusedGatherBlockSize,
        kClusteredFusedGatherBlockSize, 0, NULL, spatial.cluster_numbers,
        spatial.sort_permutation, spatial.cluster_offsets, cell, rcell, crd,
        charge, lj_type_src, spatial.cluster_centers,
        spatial.cluster_fractional_centers,
        spatial.cluster_fractional_extents, d_sorted_atom_ids, d_sorted_xq,
        d_sorted_lj_type, lj_ab_packed, d_sorted_lj_comb);
    Refresh_Gather_Dependent_Metadata(provider_, &spatial, cell, rcell);
    gathered_flavor_ = GatherFlavor::PLAIN;
    gathered_provider_incarnation_ = spatial.provider_incarnation;
    gathered_geometry_generation_ = spatial.geometry_generation;
}

void LJClusteredWorkspace::Gather_Plain(const VECTOR_LJ* src,
                                        LTMatrix3 cell, LTMatrix3 rcell,
                                        const float2* lj_ab_packed)
{
    if (provider_ == NULL)
    {
        return;
    }
    ClusteredGatherBinding spatial;
    if (!provider_->AcquireGatherBinding(&spatial) || src == NULL)
    {
        return;
    }
#ifndef USE_CPU
    provider_->BindWorkingDevice();
#endif
    Note_Clustered_Step_Counter(md_info.sys.steps, &coordinate_gather_step,
                                &coordinate_gather_count_this_step,
                                &coordinate_gather_count_total);
    RecorderScope gather_scope(payload_gather_time_recorder);
    Reserve_Plain_Gather_Scratch(this, spatial.total_atom_numbers);
    Launch_Device_Kernel(
        Gather_Clustered_LJ_Direct_Scratch,
        (spatial.cluster_numbers * kClusteredClusterSize +
         kClusteredFusedGatherBlockSize - 1) /
            kClusteredFusedGatherBlockSize,
        kClusteredFusedGatherBlockSize, 0, NULL, spatial.cluster_numbers,
        spatial.sort_permutation, spatial.cluster_offsets, cell, rcell, src,
        spatial.cluster_centers, spatial.cluster_fractional_centers,
        spatial.cluster_fractional_extents, d_sorted_atom_ids, d_sorted_xq,
        d_sorted_lj_type, lj_ab_packed, d_sorted_lj_comb);
    Refresh_Gather_Dependent_Metadata(provider_, &spatial, cell, rcell);
    gathered_flavor_ = GatherFlavor::PLAIN;
    gathered_provider_incarnation_ = spatial.provider_incarnation;
    gathered_geometry_generation_ = spatial.geometry_generation;
}

void LJClusteredWorkspace::Gather_Soft_Core(const VECTOR_LJ_SOFT_TYPE* src,
                                            LTMatrix3 cell,
                                            LTMatrix3 rcell)
{
    if (provider_ == NULL)
    {
        return;
    }
    ClusteredGatherBinding spatial;
    if (!provider_->AcquireGatherBinding(&spatial) || src == NULL)
    {
        return;
    }
#ifndef USE_CPU
    provider_->BindWorkingDevice();
#endif
    Note_Clustered_Step_Counter(md_info.sys.steps, &coordinate_gather_step,
                                &coordinate_gather_count_this_step,
                                &coordinate_gather_count_total);
    RecorderScope gather_scope(payload_gather_time_recorder);
    Reserve_Device_Buffer(spatial.total_atom_numbers, &d_sorted_soft_crd,
                          &sorted_soft_crd_capacity);
    Reserve_Device_Buffer(spatial.total_atom_numbers, &d_sorted_frc,
                          &sorted_frc_capacity);
    Launch_Device_Kernel(
        Gather_Clustered_Soft_Core_Scratch,
        (spatial.cluster_numbers * kClusteredClusterSize +
         kClusteredFusedGatherBlockSize - 1) /
            kClusteredFusedGatherBlockSize,
        kClusteredFusedGatherBlockSize, 0, NULL, spatial.cluster_numbers,
        spatial.sort_permutation, spatial.cluster_offsets, cell, rcell, src,
        spatial.cluster_centers, spatial.cluster_fractional_centers,
        spatial.cluster_fractional_extents, d_sorted_soft_crd);
    Refresh_Gather_Dependent_Metadata(provider_, &spatial, cell, rcell);
    gathered_flavor_ = GatherFlavor::SOFT_CORE;
    gathered_provider_incarnation_ = spatial.provider_incarnation;
    gathered_geometry_generation_ = spatial.geometry_generation;
}

bool LJClusteredWorkspace::Gather_Ready_For_Current_Step(
    GatherFlavor flavor) const
{
    return coordinate_gather_step == md_info.sys.steps &&
           coordinate_gather_count_this_step > 0 &&
           gathered_flavor_ == flavor && provider_ != nullptr &&
           provider_->IsGatheredGeometryCurrent(
               gathered_provider_incarnation_,
               gathered_geometry_generation_);
}

bool LJClusteredWorkspace::Plain_Gather_Ready_For_Current_Step() const
{
    return Gather_Ready_For_Current_Step(GatherFlavor::PLAIN);
}

bool LJClusteredWorkspace::Soft_Core_Gather_Ready_For_Current_Step() const
{
    return Gather_Ready_For_Current_Step(GatherFlavor::SOFT_CORE);
}
