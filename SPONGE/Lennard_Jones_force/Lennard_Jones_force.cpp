#include "Lennard_Jones_force.h"

#include <array>
#include <cstdint>

#include "../Domain_decomposition/Domain_decomposition.h"
#include "../xponge/load/native/lj.hpp"
#include "../xponge/xponge.h"
// #include "assert.h"

namespace
{

struct OrderedResiduePoint
{
    int residue_index = 0;
    int atom_start = 0;
    int atom_count = 0;
    VECTOR wrapped = {0.0f, 0.0f, 0.0f};
    VECTOR normalized = {0.0f, 0.0f, 0.0f};
    uint64_t point_hilbert = 0;
};

struct CornerstoneLeaf
{
    std::vector<int> residues;
    VECTOR min_bound = {0.0f, 0.0f, 0.0f};
    VECTOR max_bound = {1.0f, 1.0f, 1.0f};
    uint64_t leaf_hilbert = 0;
};

static VECTOR Wrap_To_Box_Fractional(VECTOR crd, LTMatrix3 rcell,
                                     VECTOR box_length)
{
    VECTOR frac = crd * rcell;
    frac.x -= floorf(frac.x);
    frac.y -= floorf(frac.y);
    frac.z -= floorf(frac.z);
    return wiseproduct(frac, box_length);
}

static uint32_t Quantize_Unit_Coordinate(float value, int bits)
{
    if (bits <= 0)
    {
        return 0;
    }
    const uint32_t grid = 1u << bits;
    float clamped = std::max(0.0f, std::min(0.99999994f, value));
    uint32_t coord = static_cast<uint32_t>(clamped * grid);
    if (coord >= grid)
    {
        coord = grid - 1;
    }
    return coord;
}

static void Hilbert_Axes_To_Transpose(std::array<uint32_t, 3>* coords,
                                      int bits)
{
    if (bits <= 0)
    {
        return;
    }
    uint32_t Q = 1u << (bits - 1);
    while (Q > 1)
    {
        const uint32_t P = Q - 1;
        for (int dim = 0; dim < 3; dim += 1)
        {
            if (((*coords)[dim] & Q) != 0)
            {
                (*coords)[0] ^= P;
            }
            else
            {
                const uint32_t t = ((*coords)[0] ^ (*coords)[dim]) & P;
                (*coords)[0] ^= t;
                (*coords)[dim] ^= t;
            }
        }
        Q >>= 1;
    }

    for (int dim = 1; dim < 3; dim += 1)
    {
        (*coords)[dim] ^= (*coords)[dim - 1];
    }

    uint32_t t = 0;
    Q = 1u << (bits - 1);
    while (Q > 1)
    {
        if (((*coords)[2] & Q) != 0)
        {
            t ^= Q - 1;
        }
        Q >>= 1;
    }
    for (int dim = 0; dim < 3; dim += 1)
    {
        (*coords)[dim] ^= t;
    }
}

static uint64_t Hilbert_Index_3D(uint32_t x, uint32_t y, uint32_t z, int bits)
{
    std::array<uint32_t, 3> coords = {x, y, z};
    Hilbert_Axes_To_Transpose(&coords, bits);

    uint64_t index = 0;
    for (int bit = bits - 1; bit >= 0; bit -= 1)
    {
        for (int dim = 0; dim < 3; dim += 1)
        {
            index = (index << 1) | ((coords[dim] >> bit) & 1u);
        }
    }
    return index;
}

static uint64_t Hilbert_Index_3D(VECTOR normalized, int bits)
{
    return Hilbert_Index_3D(Quantize_Unit_Coordinate(normalized.x, bits),
                            Quantize_Unit_Coordinate(normalized.y, bits),
                            Quantize_Unit_Coordinate(normalized.z, bits), bits);
}

#ifdef USE_GPU
static __device__ __forceinline__ unsigned int Clustered_Subgroup_Mask(
    int lane, int subgroup_width)
{
    const unsigned int subgroup =
        static_cast<unsigned int>(lane / subgroup_width);
    const unsigned int width_mask =
        (1u << static_cast<unsigned int>(subgroup_width)) - 1u;
    return width_mask << (subgroup * static_cast<unsigned int>(subgroup_width));
}

static __device__ __forceinline__ VECTOR Reduce_Clustered_Subgroup_Vector(
    VECTOR value, int lane, int subgroup_width)
{
    const unsigned int subgroup_mask =
        Clustered_Subgroup_Mask(lane, subgroup_width);
    for (int delta = subgroup_width >> 1; delta > 0; delta >>= 1)
    {
        value.x +=
            deviceShflDown(subgroup_mask, value.x, delta, subgroup_width);
        value.y +=
            deviceShflDown(subgroup_mask, value.y, delta, subgroup_width);
        value.z +=
            deviceShflDown(subgroup_mask, value.z, delta, subgroup_width);
    }
    return value;
}

static __device__ __forceinline__ VECTOR Reduce_Clustered_Warp_Vector_Over_J(
    VECTOR value, int subgroup_width)
{
    for (int delta = warpSize >> 1; delta >= subgroup_width; delta >>= 1)
    {
        value.x += deviceShflDown(FULL_MASK, value.x, delta, warpSize);
        value.y += deviceShflDown(FULL_MASK, value.y, delta, warpSize);
        value.z += deviceShflDown(FULL_MASK, value.z, delta, warpSize);
    }
    return value;
}

template <typename T>
static __device__ __forceinline__ T Broadcast_Clustered_Subgroup_Value(
    T value, int lane, int subgroup_width)
{
    const unsigned int subgroup_mask =
        Clustered_Subgroup_Mask(lane, subgroup_width);
    const int subgroup_leader = lane - lane % subgroup_width;
    return deviceShfl(subgroup_mask, value, subgroup_leader, warpSize);
}

static __device__ __forceinline__ float4 Broadcast_Clustered_Subgroup_Float4(
    float4 value, int lane, int subgroup_width)
{
    value.x = Broadcast_Clustered_Subgroup_Value(value.x, lane, subgroup_width);
    value.y = Broadcast_Clustered_Subgroup_Value(value.y, lane, subgroup_width);
    value.z = Broadcast_Clustered_Subgroup_Value(value.z, lane, subgroup_width);
    value.w = Broadcast_Clustered_Subgroup_Value(value.w, lane, subgroup_width);
    return value;
}

static __device__ __forceinline__ float Reduce_Clustered_Warp_Float_Over_J(
    float value, int subgroup_width)
{
    for (int delta = warpSize >> 1; delta >= subgroup_width; delta >>= 1)
    {
        value += deviceShflDown(FULL_MASK, value, delta, warpSize);
    }
    return value;
}

static __device__ __forceinline__ LTMatrix3 Reduce_Clustered_Warp_Virial_Over_J(
    LTMatrix3 value, int subgroup_width)
{
    for (int delta = warpSize >> 1; delta >= subgroup_width; delta >>= 1)
    {
        value.a11 += deviceShflDown(FULL_MASK, value.a11, delta, warpSize);
        value.a21 += deviceShflDown(FULL_MASK, value.a21, delta, warpSize);
        value.a22 += deviceShflDown(FULL_MASK, value.a22, delta, warpSize);
        value.a31 += deviceShflDown(FULL_MASK, value.a31, delta, warpSize);
        value.a32 += deviceShflDown(FULL_MASK, value.a32, delta, warpSize);
        value.a33 += deviceShflDown(FULL_MASK, value.a33, delta, warpSize);
    }
    return value;
}

static __device__ __forceinline__ float4 Pack_Clustered_Virial_Lo(
    LTMatrix3 value)
{
    return {value.a11, value.a21, value.a22, value.a31};
}

static __device__ __forceinline__ float2 Pack_Clustered_Virial_Hi(
    LTMatrix3 value)
{
    return {value.a32, value.a33};
}

static __device__ __forceinline__ LTMatrix3 Unpack_Clustered_Virial(
    float4 lo, float2 hi)
{
    return {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y};
}
#endif

static void Build_Cornerstone_Leaves(
    const std::vector<OrderedResiduePoint>& points,
    const std::vector<int>& residue_indices, int depth, int max_depth,
    int leaf_size, VECTOR min_bound, VECTOR max_bound,
    std::vector<CornerstoneLeaf>* leaves)
{
    if (residue_indices.empty())
    {
        return;
    }
    if ((int)residue_indices.size() <= leaf_size || depth >= max_depth)
    {
        leaves->push_back(
            {residue_indices, min_bound, max_bound, static_cast<uint64_t>(0)});
        return;
    }

    const VECTOR mid = 0.5f * (min_bound + max_bound);
    std::array<std::vector<int>, 8> children;
    for (int residue_index : residue_indices)
    {
        const VECTOR& p = points[residue_index].normalized;
        int octant = 0;
        if (p.x >= mid.x)
        {
            octant |= 1;
        }
        if (p.y >= mid.y)
        {
            octant |= 2;
        }
        if (p.z >= mid.z)
        {
            octant |= 4;
        }
        children[octant].push_back(residue_index);
    }

    int non_empty_children = 0;
    for (const auto& child : children)
    {
        non_empty_children += !child.empty();
    }
    if (non_empty_children <= 1)
    {
        leaves->push_back(
            {residue_indices, min_bound, max_bound, static_cast<uint64_t>(0)});
        return;
    }

    for (int octant = 0; octant < 8; octant += 1)
    {
        if (children[octant].empty())
        {
            continue;
        }
        VECTOR child_min = min_bound;
        VECTOR child_max = max_bound;
        if ((octant & 1) != 0)
        {
            child_min.x = mid.x;
        }
        else
        {
            child_max.x = mid.x;
        }
        if ((octant & 2) != 0)
        {
            child_min.y = mid.y;
        }
        else
        {
            child_max.y = mid.y;
        }
        if ((octant & 4) != 0)
        {
            child_min.z = mid.z;
        }
        else
        {
            child_max.z = mid.z;
        }
        Build_Cornerstone_Leaves(points, children[octant], depth + 1,
                                 max_depth, leaf_size, child_min, child_max,
                                 leaves);
    }
}

}  // namespace

// 由LJ坐标和转化系数求距离
__global__ void Copy_LJ_Type_To_New_Crd(const int atom_numbers,
                                        VECTOR_LJ* new_crd, const int* LJ_type)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        new_crd[atom_i].LJ_type = LJ_type[atom_i];
    }
}

__global__ void Copy_Crd_And_Charge_To_New_Crd(const int atom_numbers,
                                               const VECTOR* crd,
                                               VECTOR_LJ* new_crd,
                                               const float* charge)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        new_crd[atom_i].crd = crd[atom_i];
        new_crd[atom_i].charge = charge[atom_i];
    }
}

__global__ void Copy_Crd_To_New_Crd(const int atom_numbers, const VECTOR* crd,
                                    VECTOR_LJ* new_crd)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        new_crd[atom_i].crd = crd[atom_i];
    }
}

static __global__ void Gather_Sorted_LJ_Crd(const int atom_numbers,
                                            const int* permutation,
                                            const VECTOR_LJ* src,
                                            VECTOR_LJ* dest)
{
    SIMPLE_DEVICE_FOR(sorted_i, atom_numbers)
    {
        dest[sorted_i] = src[permutation[sorted_i]];
    }
}

static __global__ void Gather_Sorted_LJ_Packed(const int atom_numbers,
                                               const int* permutation,
                                               const VECTOR_LJ* src,
                                               int* sorted_atom_ids,
                                               float4* sorted_xq,
                                               int* sorted_lj_type)
{
    SIMPLE_DEVICE_FOR(sorted_i, atom_numbers)
    {
        const int atom_i = permutation[sorted_i];
        const VECTOR_LJ atom = src[atom_i];
        sorted_atom_ids[sorted_i] = atom_i;
        sorted_xq[sorted_i] = {atom.crd.x, atom.crd.y, atom.crd.z, atom.charge};
        sorted_lj_type[sorted_i] = atom.LJ_type;
    }
}

static __global__ void device_add(float* variable, const float adder)
{
    variable[0] += adder;
}

static __host__ __device__ __forceinline__ VECTOR_LJ Make_Packed_LJ_Atom(
    const float4 xq, const int lj_type)
{
    VECTOR_LJ atom = {};
    atom.crd = {xq.x, xq.y, xq.z};
    atom.LJ_type = lj_type;
    atom.charge = xq.w;
    return atom;
}

static __host__ __device__ __forceinline__ VECTOR
Wrap_Clustered_Center_Fractional(const VECTOR center, const LTMatrix3 rcell)
{
    VECTOR frac = center * rcell;
    frac.x -= floorf(frac.x);
    frac.y -= floorf(frac.y);
    frac.z -= floorf(frac.z);
    return frac;
}

static __host__ __device__ __forceinline__ int Encode_Clustered_Pair_Shift_Id(
    int sx, int sy, int sz)
{
    sx = sx < -1 ? -1 : (sx > 1 ? 1 : sx);
    sy = sy < -1 ? -1 : (sy > 1 ? 1 : sy);
    sz = sz < -1 ? -1 : (sz > 1 ? 1 : sz);
    return (sx + 1) * 9 + (sy + 1) * 3 + (sz + 1);
}

static __host__ __device__ __forceinline__ int
Determine_Clustered_Pair_Shift_Id(const VECTOR center_i, const VECTOR center_j,
                                  const LTMatrix3 rcell)
{
    const VECTOR frac_i = Wrap_Clustered_Center_Fractional(center_i, rcell);
    const VECTOR frac_j = Wrap_Clustered_Center_Fractional(center_j, rcell);
    const VECTOR dfrac = frac_j - frac_i;
    return Encode_Clustered_Pair_Shift_Id(
        static_cast<int>(floorf(dfrac.x + 0.5f)),
        static_cast<int>(floorf(dfrac.y + 0.5f)),
        static_cast<int>(floorf(dfrac.z + 0.5f)));
}

static __host__ __device__ __forceinline__ VECTOR
Get_Clustered_Pair_Shift_Vector(const VECTOR center_i, const VECTOR center_j,
                                const LTMatrix3 cell, const LTMatrix3 rcell)
{
    return Clustered_Shift_Vector_From_Id(
        Determine_Clustered_Pair_Shift_Id(center_i, center_j, rcell), cell);
}

static __host__ __device__ __forceinline__ VECTOR
Get_Clustered_Shifted_Displacement(const VECTOR_LJ r2, const VECTOR_LJ r1,
                                   const VECTOR shift_vec)
{
    return (r2.crd - r1.crd) - shift_vec;
}

template <bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb>
static __global__ void Lennard_Jones_And_Direct_Coulomb_Device(
    const int local_atom_numbers, const int solvent_numbers,
    const ATOM_GROUP* nl, const VECTOR_LJ* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* LJ_type_A, const float* LJ_type_B,
    const float cutoff, VECTOR* frc, const float pme_beta, float* atom_energy,
    LTMatrix3* atom_virial, float* atom_direct_cf_energy, float* atom_LJ_ene)
{
#ifdef USE_GPU
    int atom_i = 0 + blockDim.y * blockIdx.x + threadIdx.y;
    if (atom_i < local_atom_numbers - solvent_numbers)
#else
#pragma omp parallel for schedule(dynamic)
    for (int atom_i = 0; atom_i < local_atom_numbers - solvent_numbers;
         atom_i++)
#endif
    {
        VECTOR frc_record = {0.0f, 0.0f, 0.0f};
        LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float energy_lj = 0.0f;
        float energy_coulomb = 0.0f;
        float energy_total = 0.0f;
        ATOM_GROUP nl_i = nl[atom_i];
        VECTOR_LJ r1 = crd[atom_i];
#ifdef USE_GPU
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
#else
        for (int j = 0; j < nl_i.atom_numbers; j += 1)
#endif
        {
            int atom_j = nl_i.atom_serial[j];
            float ij_factor = atom_j < local_atom_numbers ? 1.0f : 0.5f;
            VECTOR_LJ r2 = crd[atom_j];
            VECTOR dr = Get_Periodic_Displacement(r2, r1, cell, rcell);
            float dr_abs = norm3df(dr.x, dr.y, dr.z);
            if (dr_abs < cutoff)
            {
                int atom_pair_LJ_type = Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                float A = LJ_type_A[atom_pair_LJ_type];
                float B = LJ_type_B[atom_pair_LJ_type];
                if (need_force)
                {
                    float frc_abs = Get_LJ_Force(r1, r2, dr_abs, A, B);
                    if (need_coulomb)
                    {
                        float frc_cf_abs =
                            Get_Direct_Coulomb_Force(r1, r2, dr_abs, pme_beta);
                        frc_abs = frc_abs - frc_cf_abs;
                    }
                    VECTOR frc_lin = frc_abs * dr;
                    frc_record = frc_record + frc_lin;
                    if (atom_j < local_atom_numbers)
                    {
                        atomicAdd(frc + atom_j, -frc_lin);
                    }
                    if (need_virial)
                    {
                        virial = virial - ij_factor * Get_Virial_From_Force_Dis(
                                                          frc_lin, dr);
                    }
                }
                if (need_energy)
                {
                    energy_lj +=
                        ij_factor * Get_LJ_Energy(r1, r2, dr_abs, A, B);
                    if (need_coulomb)
                    {
                        energy_coulomb +=
                            ij_factor *
                            Get_Direct_Coulomb_Energy(r1, r2, dr_abs, pme_beta);
                    }
                }
            }
        }
        energy_total = energy_lj + energy_coulomb;
        if (need_force)
        {
            Warp_Sum_To(frc + atom_i, frc_record, warpSize);
        }
        if (need_energy)
        {
            Warp_Sum_To(atom_energy + atom_i, energy_total, warpSize);
            Warp_Sum_To(atom_LJ_ene + atom_i, energy_lj, warpSize);
            if (need_coulomb)
                Warp_Sum_To(atom_direct_cf_energy + atom_i, energy_coulomb,
                            warpSize);
        }
        if (need_virial)
        {
            Warp_Sum_To(atom_virial + atom_i, virial, warpSize);
        }
    }
}

template <bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb>
static __global__ void Clustered_Lennard_Jones_And_Direct_Coulomb_Device(
    const int sci_numbers, const int cluster_size,
    const int super_cluster_clusters, const int local_atom_numbers,
    const int* cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const int* super_cluster_offsets, const int* sci_supercluster_ids,
    const int* sci_offsets, const int* cjpacked_cluster_ids,
    const unsigned int* cjpacked_imasks,
    const int* cjpacked_exclusion_indices,
    const unsigned long long* exclusion_mask_pool,
    const int* sorted_atom_ids, const float4* sorted_xq,
    const int* sorted_lj_type,
    const LTMatrix3 cell, const LTMatrix3 rcell, const float* LJ_type_A,
    const float* LJ_type_B, const float cutoff, VECTOR* frc,
    const float pme_beta, float* atom_energy, LTMatrix3* atom_virial,
    float* atom_direct_cf_energy, float* atom_LJ_ene)
{
    constexpr int max_cluster_size = 8;
    constexpr int max_super_cluster_atoms = 64;
    constexpr int max_block_warps = 2;
#ifdef USE_GPU
    const int sci = blockIdx.x;
    const int tid = threadIdx.x;
    if (sci < sci_numbers &&
        tid < super_cluster_clusters * cluster_size)
#else
#pragma omp parallel for schedule(dynamic)
    for (int sci = 0; sci < sci_numbers; sci += 1)
#endif
    {
#ifndef USE_GPU
        const int super_i = sci_supercluster_ids[sci];
        const int cluster_i_start = super_cluster_offsets[super_i];
        const int cluster_i_end = super_cluster_offsets[super_i + 1];
        for (int cluster_i = cluster_i_start; cluster_i < cluster_i_end;
             cluster_i += 1)
        {
            const unsigned int valid_mask_i = cluster_valid_masks[cluster_i];
            const unsigned int local_mask_i = cluster_local_masks[cluster_i];
            const int i_local = cluster_i - cluster_i_start;
            const float cutoff_sq = cutoff * cutoff;
            for (int lane_i = 0; lane_i < cluster_size; lane_i += 1)
            {
                if ((valid_mask_i & (1u << lane_i)) == 0u ||
                    (local_mask_i & (1u << lane_i)) == 0u)
                {
                    continue;
                }
                const int start_i = cluster_offsets[cluster_i];
                const int sorted_atom_i = start_i + lane_i;
                const int atom_i = sorted_atom_ids[sorted_atom_i];
                const VECTOR_LJ r1 = Make_Packed_LJ_Atom(
                    sorted_xq[sorted_atom_i], sorted_lj_type[sorted_atom_i]);
                VECTOR frc_i = {0.0f, 0.0f, 0.0f};
                float energy_lj = 0.0f;
                float energy_coulomb = 0.0f;
                LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

                for (int cj = sci_offsets[sci]; cj < sci_offsets[sci + 1];
                     cj += 1)
                {
                    const unsigned int imask = cjpacked_imasks[cj];
                    if (imask == 0u)
                    {
                        continue;
                    }
                    if ((imask & (1u << i_local)) == 0u)
                    {
                        continue;
                    }
                    const int cluster_j = cjpacked_cluster_ids[cj];
                    const unsigned int valid_mask_j =
                        cluster_valid_masks[cluster_j];
                    const int exclusion_index =
                        cjpacked_exclusion_indices[cj * super_cluster_clusters +
                                                   i_local];
                    const unsigned long long exclusion_mask =
                        exclusion_index >= 0 ? exclusion_mask_pool[exclusion_index]
                                             : 0ull;
                    VECTOR frc_j[max_cluster_size];
                    for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
                    {
                        frc_j[lane_j] = {0.0f, 0.0f, 0.0f};
                    }
                    for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
                    {
                        if ((valid_mask_j & (1u << lane_j)) == 0u)
                        {
                            continue;
                        }
                        const int sorted_atom_j =
                            cluster_offsets[cluster_j] + lane_j;
                        const int atom_j = sorted_atom_ids[sorted_atom_j];
                        if (cluster_i == cluster_j && atom_j < local_atom_numbers &&
                            lane_j <= lane_i)
                        {
                            continue;
                        }
                        if ((exclusion_mask &
                             (1ull << (lane_i * cluster_size + lane_j))) != 0ull)
                        {
                            continue;
                        }
                        const VECTOR_LJ r2 = Make_Packed_LJ_Atom(
                            sorted_xq[sorted_atom_j],
                            sorted_lj_type[sorted_atom_j]);
                        const VECTOR dr =
                            Get_Periodic_Displacement(r2, r1, cell, rcell);
                        const float dr2 = dr * dr;
                        if (dr2 >= cutoff_sq || dr2 == 0.0f)
                        {
                            continue;
                        }
                        const float dr_abs = sqrtf(dr2);
                        const int atom_pair_LJ_type =
                            Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                        const float A = LJ_type_A[atom_pair_LJ_type];
                        const float B = LJ_type_B[atom_pair_LJ_type];
                        const float ij_factor =
                            atom_j < local_atom_numbers ? 1.0f : 0.5f;
                        if (need_force)
                        {
                            float frc_abs = Get_LJ_Force(r1, r2, dr_abs, A, B);
                            if (need_coulomb)
                            {
                                frc_abs -= Get_Direct_Coulomb_Force(
                                    r1, r2, dr_abs, pme_beta);
                            }
                            const VECTOR frc_lin = frc_abs * dr;
                            frc_i = frc_i + frc_lin;
                            if (atom_j < local_atom_numbers)
                            {
                                frc_j[lane_j] = frc_j[lane_j] - frc_lin;
                            }
                            if (need_virial)
                            {
                                virial = virial -
                                         ij_factor *
                                             Get_Virial_From_Force_Dis(frc_lin, dr);
                            }
                        }
                        if (need_energy)
                        {
                            energy_lj +=
                                ij_factor * Get_LJ_Energy(r1, r2, dr_abs, A, B);
                            if (need_coulomb)
                            {
                                energy_coulomb +=
                                    ij_factor * Get_Direct_Coulomb_Energy(
                                                    r1, r2, dr_abs, pme_beta);
                            }
                        }
                    }
                    if (need_force)
                    {
                        for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
                        {
                            if ((valid_mask_j & (1u << lane_j)) == 0u)
                            {
                                continue;
                            }
                            const int sorted_atom_j =
                                cluster_offsets[cluster_j] + lane_j;
                            const int atom_j = sorted_atom_ids[sorted_atom_j];
                            if (atom_j < local_atom_numbers)
                            {
                                atomicAdd(frc + atom_j, frc_j[lane_j]);
                            }
                        }
                    }
                }
                if (need_energy)
                {
                    atomicAdd(atom_energy + atom_i, energy_lj + energy_coulomb);
                    atomicAdd(atom_LJ_ene + atom_i, energy_lj);
                    if (need_coulomb)
                    {
                        atomicAdd(atom_direct_cf_energy + atom_i,
                                  energy_coulomb);
                    }
                }
                if (need_force)
                {
                    atomicAdd(frc + atom_i, frc_i);
                }
                if (need_virial)
                {
                    atomicAdd(atom_virial + atom_i, virial);
                }
            }
        }
#else
        __shared__ float4 shared_i_xq[max_super_cluster_atoms];
        __shared__ int shared_i_lj_type[max_super_cluster_atoms];
        __shared__ int shared_i_atom_ids[max_super_cluster_atoms];
        __shared__ float4 shared_j_xq[max_cluster_size];
        __shared__ int shared_j_lj_type[max_cluster_size];
        __shared__ int shared_j_atom_ids[max_cluster_size];
        __shared__ int shared_j_local_flags[max_cluster_size];
        __shared__ unsigned int shared_j_valid_mask;
        __shared__ VECTOR warp_j_force[max_block_warps][max_cluster_size];

        const int super_i = sci_supercluster_ids[sci];
        const int cluster_i_start = super_cluster_offsets[super_i];
        const int cluster_i_end = super_cluster_offsets[super_i + 1];
        const int i_cluster_local = tid / cluster_size;
        const int i_lane = tid % cluster_size;
        const int active_cluster_count = cluster_i_end - cluster_i_start;
        bool active_i = false;
        int cluster_i = -1;
        int atom_i = -1;
        VECTOR frc_i = {0.0f, 0.0f, 0.0f};
        float energy_lj = 0.0f;
        float energy_coulomb = 0.0f;
        LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float4 r1_xq = {0.0f, 0.0f, 0.0f, 0.0f};
        int r1_lj_type = 0;
        const float cutoff_sq = cutoff * cutoff;

        if (i_cluster_local < active_cluster_count)
        {
            cluster_i = cluster_i_start + i_cluster_local;
            if ((cluster_valid_masks[cluster_i] & (1u << i_lane)) != 0u)
            {
                const int sorted_atom_i = cluster_offsets[cluster_i] + i_lane;
                shared_i_xq[tid] = sorted_xq[sorted_atom_i];
                shared_i_lj_type[tid] = sorted_lj_type[sorted_atom_i];
                shared_i_atom_ids[tid] = sorted_atom_ids[sorted_atom_i];
                if ((cluster_local_masks[cluster_i] & (1u << i_lane)) != 0u)
                {
                    active_i = true;
                    atom_i = shared_i_atom_ids[tid];
                    r1_xq = shared_i_xq[tid];
                    r1_lj_type = shared_i_lj_type[tid];
                }
            }
        }
        __syncthreads();

        const int lane = tid & (warpSize - 1);
        const int warp_id = tid / warpSize;
        const int warp_count =
            (super_cluster_clusters * cluster_size + warpSize - 1) / warpSize;

        for (int cj = sci_offsets[sci]; cj < sci_offsets[sci + 1]; cj += 1)
        {
            const unsigned int imask = cjpacked_imasks[cj];
            if (imask == 0u)
            {
                continue;
            }
            const int cluster_j = cjpacked_cluster_ids[cj];
            const unsigned int valid_mask_j = cluster_valid_masks[cluster_j];
            if (tid == 0)
            {
                shared_j_valid_mask = valid_mask_j;
            }
            if (tid < cluster_size)
            {
                if ((valid_mask_j & (1u << tid)) != 0u)
                {
                    const int sorted_atom_j = cluster_offsets[cluster_j] + tid;
                    shared_j_xq[tid] = sorted_xq[sorted_atom_j];
                    shared_j_lj_type[tid] = sorted_lj_type[sorted_atom_j];
                    shared_j_atom_ids[tid] = sorted_atom_ids[sorted_atom_j];
                    shared_j_local_flags[tid] =
                        shared_j_atom_ids[tid] < local_atom_numbers ? 1 : 0;
                }
                else
                {
                    shared_j_atom_ids[tid] = -1;
                    shared_j_local_flags[tid] = 0;
                }
            }
            __syncthreads();
            const bool tile_active =
                active_i && ((imask & (1u << i_cluster_local)) != 0u);
            unsigned long long exclusion_mask = 0ull;
            VECTOR_LJ r1 = {};
            if (active_i)
            {
                r1 = Make_Packed_LJ_Atom(r1_xq, r1_lj_type);
            }
            if (tile_active)
            {
                const int exclusion_index =
                    cjpacked_exclusion_indices[cj * super_cluster_clusters +
                                               i_cluster_local];
                exclusion_mask =
                    exclusion_index >= 0 ? exclusion_mask_pool[exclusion_index]
                                         : 0ull;
            }

            for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
            {
                VECTOR j_force_local = {0.0f, 0.0f, 0.0f};
                if (tile_active && (shared_j_valid_mask & (1u << lane_j)) != 0u)
                {
                    const int atom_j = shared_j_atom_ids[lane_j];
                    if (!(cluster_i == cluster_j &&
                          atom_j < local_atom_numbers && lane_j <= i_lane) &&
                        (exclusion_mask &
                         (1ull << (i_lane * cluster_size + lane_j))) == 0ull)
                    {
                        const VECTOR_LJ r2 = Make_Packed_LJ_Atom(
                            shared_j_xq[lane_j], shared_j_lj_type[lane_j]);
                        const VECTOR dr =
                            Get_Periodic_Displacement(r2, r1, cell, rcell);
                        const float dr2 = dr * dr;
                        if (dr2 < cutoff_sq && dr2 != 0.0f)
                        {
                            const float dr_abs = sqrtf(dr2);
                            const int atom_pair_LJ_type =
                                Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                            const float A = LJ_type_A[atom_pair_LJ_type];
                            const float B = LJ_type_B[atom_pair_LJ_type];
                            const float ij_factor =
                                atom_j < local_atom_numbers ? 1.0f : 0.5f;
                            if (need_force)
                            {
                                float frc_abs =
                                    Get_LJ_Force(r1, r2, dr_abs, A, B);
                                if (need_coulomb)
                                {
                                    frc_abs -= Get_Direct_Coulomb_Force(
                                        r1, r2, dr_abs, pme_beta);
                                }
                                const VECTOR frc_lin = frc_abs * dr;
                                frc_i = frc_i + frc_lin;
                                if (shared_j_local_flags[lane_j] != 0)
                                {
                                    j_force_local = j_force_local - frc_lin;
                                }
                                if (need_virial)
                                {
                                    virial = virial -
                                             ij_factor *
                                                 Get_Virial_From_Force_Dis(
                                                     frc_lin, dr);
                                }
                            }
                            if (need_energy)
                            {
                                energy_lj += ij_factor *
                                             Get_LJ_Energy(r1, r2, dr_abs, A, B);
                                if (need_coulomb)
                                {
                                    energy_coulomb +=
                                        ij_factor *
                                        Get_Direct_Coulomb_Energy(
                                            r1, r2, dr_abs, pme_beta);
                                }
                            }
                        }
                    }
                }
                if (need_force)
                {
                    VECTOR reduced = j_force_local;
                    for (int delta = warpSize >> 1; delta > 0; delta >>= 1)
                    {
                        reduced.x +=
                            deviceShflDown(FULL_MASK, reduced.x, delta, warpSize);
                        reduced.y +=
                            deviceShflDown(FULL_MASK, reduced.y, delta, warpSize);
                        reduced.z +=
                            deviceShflDown(FULL_MASK, reduced.z, delta, warpSize);
                    }
                    if (lane == 0)
                    {
                        warp_j_force[warp_id][lane_j] = reduced;
                    }
                }
            }
            if (need_force)
            {
                __syncthreads();
                if (tid < cluster_size &&
                    (shared_j_valid_mask & (1u << tid)) != 0u &&
                    shared_j_local_flags[tid] != 0)
                {
                    VECTOR total = {0.0f, 0.0f, 0.0f};
                    for (int warp_i = 0; warp_i < warp_count; warp_i += 1)
                    {
                        total = total + warp_j_force[warp_i][tid];
                    }
                    atomicAdd(frc + shared_j_atom_ids[tid], total);
                }
                __syncthreads();
            }
        }

        if (active_i)
        {
            if (need_force)
            {
                atomicAdd(frc + atom_i, frc_i);
            }
            if (need_energy)
            {
                atomicAdd(atom_energy + atom_i, energy_lj + energy_coulomb);
                atomicAdd(atom_LJ_ene + atom_i, energy_lj);
                if (need_coulomb)
                {
                    atomicAdd(atom_direct_cf_energy + atom_i, energy_coulomb);
                }
            }
            if (need_virial)
            {
                atomicAdd(atom_virial + atom_i, virial);
            }
        }
#endif
    }
}

template <bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb>
static __global__ void Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Device(
    const int sci_numbers, const int cluster_size,
    const int super_cluster_clusters, const int local_atom_numbers,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const int* super_cluster_offsets,
    const LJ_CLUSTERED_SCI* sci_entries,
    const LJ_CLUSTERED_CJ_PACKED* cj_packed_entries,
    const unsigned long long* exclusion_mask_pool, const int* sorted_atom_ids,
    const float4* sorted_xq, const int* sorted_lj_type, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* LJ_type_A, const float* LJ_type_B,
    const float cutoff, VECTOR* frc, const float pme_beta, float* atom_energy,
    LTMatrix3* atom_virial, float* atom_direct_cf_energy, float* atom_LJ_ene)
{
    constexpr int max_cluster_size = kClusteredClusterSize;
    constexpr int max_super_cluster_atoms =
        kClusteredClusterSize * kClusteredSuperClusterClusters;
    constexpr int max_block_warps = 2;
#ifdef USE_GPU
    const int sci = blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (sci >= sci_numbers ||
        tid >= super_cluster_clusters * cluster_size)
    {
        return;
    }
#else
#pragma omp parallel for schedule(dynamic)
    for (int sci = 0; sci < sci_numbers; sci += 1)
#endif
    {
        const LJ_CLUSTERED_SCI sci_entry = sci_entries[sci];
        const int super_i = sci_entry.supercluster_id;
        const int cluster_i_start = super_cluster_offsets[super_i];
        const int cluster_i_end = super_cluster_offsets[super_i + 1];
        const bool sci_is_central =
            sci_entry.shift_id == kClusteredCentralShiftId;
        const float cutoff_sq = cutoff * cutoff;

#ifndef USE_GPU
        for (int cluster_i = cluster_i_start; cluster_i < cluster_i_end;
             cluster_i += 1)
        {
            const unsigned int valid_mask_i = cluster_valid_masks[cluster_i];
            const unsigned int local_mask_i = cluster_local_masks[cluster_i];
            const int i_local = cluster_i - cluster_i_start;
            for (int lane_i = 0; lane_i < cluster_size; lane_i += 1)
            {
                if ((valid_mask_i & (1u << lane_i)) == 0u ||
                    (local_mask_i & (1u << lane_i)) == 0u)
                {
                    continue;
                }
                const int sorted_atom_i = cluster_offsets[cluster_i] + lane_i;
                const int atom_i = sorted_atom_ids[sorted_atom_i];
                VECTOR_LJ r1 = Make_Packed_LJ_Atom(sorted_xq[sorted_atom_i],
                                                   sorted_lj_type[sorted_atom_i]);
                VECTOR frc_i = {0.0f, 0.0f, 0.0f};
                float energy_lj = 0.0f;
                float energy_coulomb = 0.0f;
                LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

                for (int packed_idx = sci_entry.cjpacked_begin;
                     packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
                {
                    const LJ_CLUSTERED_CJ_PACKED& packed =
                        cj_packed_entries[packed_idx];
                    for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
                    {
                        const int cluster_j = packed.cj[jm];
                        if (cluster_j < 0)
                        {
                            continue;
                        }
                        const unsigned int imask =
                            Clustered_Jm_Imask(packed.imei[0], jm) |
                            Clustered_Jm_Imask(packed.imei[1], jm);
                        if ((imask & (1u << i_local)) == 0u)
                        {
                            continue;
                        }
                        const unsigned int valid_mask_j =
                            cluster_valid_masks[cluster_j];
                        const int exclusion_index =
                            Clustered_First_Exclusion_Index(packed, jm, i_local);
                        const unsigned long long exclusion_mask =
                            exclusion_index >= 0
                                ? exclusion_mask_pool[exclusion_index]
                                : 0ull;
                        const VECTOR pair_shift_vec =
                            Get_Clustered_Pair_Shift_Vector(
                                cluster_centers[cluster_i],
                                cluster_centers[cluster_j], cell, rcell);
                        VECTOR frc_j[max_cluster_size] = {};

                        for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
                        {
                            if ((valid_mask_j & (1u << lane_j)) == 0u)
                            {
                                continue;
                            }
                            const int sorted_atom_j =
                                cluster_offsets[cluster_j] + lane_j;
                            const int atom_j = sorted_atom_ids[sorted_atom_j];
                            if (sci_is_central && cluster_i == cluster_j &&
                                atom_j < local_atom_numbers &&
                                lane_j <= lane_i)
                            {
                                continue;
                            }
                            if ((exclusion_mask &
                                 (1ull << (lane_i * cluster_size + lane_j))) !=
                                0ull)
                            {
                                continue;
                            }
                            const VECTOR_LJ r2 = Make_Packed_LJ_Atom(
                                sorted_xq[sorted_atom_j],
                                sorted_lj_type[sorted_atom_j]);
                            const VECTOR dr =
                                Get_Clustered_Shifted_Displacement(
                                    r2, r1, pair_shift_vec);
                            const float dr2 = dr * dr;
                            if (dr2 >= cutoff_sq || dr2 == 0.0f)
                            {
                                continue;
                            }
                            const float dr_abs = sqrtf(dr2);
                            const int atom_pair_LJ_type =
                                Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                            const float A = LJ_type_A[atom_pair_LJ_type];
                            const float B = LJ_type_B[atom_pair_LJ_type];
                            const float ij_factor =
                                atom_j < local_atom_numbers ? 1.0f : 0.5f;
                            if (need_force)
                            {
                                float frc_abs =
                                    Get_LJ_Force(r1, r2, dr_abs, A, B);
                                if (need_coulomb)
                                {
                                    frc_abs -= Get_Direct_Coulomb_Force(
                                        r1, r2, dr_abs, pme_beta);
                                }
                                const VECTOR frc_lin = frc_abs * dr;
                                frc_i = frc_i + frc_lin;
                                if (atom_j < local_atom_numbers)
                                {
                                    frc_j[lane_j] = frc_j[lane_j] - frc_lin;
                                }
                                if (need_virial)
                                {
                                    virial = virial -
                                             ij_factor *
                                                 Get_Virial_From_Force_Dis(
                                                     frc_lin, dr);
                                }
                            }
                            if (need_energy)
                            {
                                energy_lj +=
                                    ij_factor *
                                    Get_LJ_Energy(r1, r2, dr_abs, A, B);
                                if (need_coulomb)
                                {
                                    energy_coulomb +=
                                        ij_factor *
                                        Get_Direct_Coulomb_Energy(
                                            r1, r2, dr_abs, pme_beta);
                                }
                            }
                        }
                        if (need_force)
                        {
                            for (int lane_j = 0; lane_j < cluster_size;
                                 lane_j += 1)
                            {
                                const int sorted_atom_j =
                                    cluster_offsets[cluster_j] + lane_j;
                                const int atom_j = sorted_atom_ids[sorted_atom_j];
                                if ((valid_mask_j & (1u << lane_j)) != 0u &&
                                    atom_j < local_atom_numbers)
                                {
                                    atomicAdd(frc + atom_j, frc_j[lane_j]);
                                }
                            }
                        }
                    }
                }

                if (need_force)
                {
                    atomicAdd(frc + atom_i, frc_i);
                }
                if (need_energy)
                {
                    atomicAdd(atom_energy + atom_i, energy_lj + energy_coulomb);
                    atomicAdd(atom_LJ_ene + atom_i, energy_lj);
                    if (need_coulomb)
                    {
                        atomicAdd(atom_direct_cf_energy + atom_i,
                                  energy_coulomb);
                    }
                }
                if (need_virial)
                {
                    atomicAdd(atom_virial + atom_i, virial);
                }
            }
        }
#else
        if constexpr (need_virial)
        {
            __shared__ float4 shared_i_xq[max_super_cluster_atoms];
            __shared__ int shared_i_lj_type[max_super_cluster_atoms];
            __shared__ int shared_i_atom_ids[max_super_cluster_atoms];
            __shared__ unsigned int shared_i_valid_masks[kClusteredSuperClusterClusters];
            __shared__ unsigned int shared_i_local_masks[kClusteredSuperClusterClusters];
            __shared__ int shared_i_cluster_ids[kClusteredSuperClusterClusters];
            __shared__ float4 warp1_i_force[kClusteredSuperClusterClusters]
                                            [max_cluster_size];
            __shared__ float warp1_i_energy_lj[kClusteredSuperClusterClusters]
                                              [max_cluster_size];
            __shared__ float warp1_i_energy_coulomb
                [kClusteredSuperClusterClusters][max_cluster_size];
            __shared__ float4 warp1_i_virial_lo
                [kClusteredSuperClusterClusters][max_cluster_size];
            __shared__ float2 warp1_i_virial_hi
                [kClusteredSuperClusterClusters][max_cluster_size];
            __shared__ float4 shared_j_shift[kClusteredSuperClusterClusters];

            const int i_lane = threadIdx.x;
            const int j_lane = threadIdx.y;
            const int lane = tid & (warpSize - 1);
            const int warp_id = tid / warpSize;
            const int active_cluster_count = cluster_i_end - cluster_i_start;
            const int i_slot = j_lane * cluster_size + i_lane;

            VECTOR fci_buf[kClusteredSuperClusterClusters];
            float energy_lj_buf[kClusteredSuperClusterClusters] = {};
            float energy_coulomb_buf[kClusteredSuperClusterClusters] = {};
            LTMatrix3 virial_buf[kClusteredSuperClusterClusters];
            for (int i_local = 0; i_local < kClusteredSuperClusterClusters;
                 i_local += 1)
            {
                fci_buf[i_local] = {0.0f, 0.0f, 0.0f};
                virial_buf[i_local] = {0.0f, 0.0f, 0.0f,
                                       0.0f, 0.0f, 0.0f};
            }

            if (j_lane < active_cluster_count)
            {
                const int cluster_i = cluster_i_start + j_lane;
                if (i_lane == 0)
                {
                    shared_i_valid_masks[j_lane] = cluster_valid_masks[cluster_i];
                    shared_i_local_masks[j_lane] = cluster_local_masks[cluster_i];
                    shared_i_cluster_ids[j_lane] = cluster_i;
                }
                if ((cluster_valid_masks[cluster_i] & (1u << i_lane)) != 0u)
                {
                    const int sorted_atom_i = cluster_offsets[cluster_i] + i_lane;
                    shared_i_xq[i_slot] = sorted_xq[sorted_atom_i];
                    shared_i_lj_type[i_slot] = sorted_lj_type[sorted_atom_i];
                    shared_i_atom_ids[i_slot] = sorted_atom_ids[sorted_atom_i];
                }
                else
                {
                    shared_i_atom_ids[i_slot] = -1;
                }
            }
            else if (i_lane == 0 && j_lane < kClusteredSuperClusterClusters)
            {
                shared_i_valid_masks[j_lane] = 0u;
                shared_i_local_masks[j_lane] = 0u;
                shared_i_cluster_ids[j_lane] = -1;
            }
            __syncthreads();

            for (int packed_idx = sci_entry.cjpacked_begin;
                 packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
            {
                const LJ_CLUSTERED_CJ_PACKED packed =
                    cj_packed_entries[packed_idx];
                const LJ_CLUSTERED_IMEI imei = packed.imei[warp_id];
                for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
                {
                    const int cluster_j = packed.cj[jm];
                    const unsigned int combined_imask =
                        cluster_j >= 0
                            ? (Clustered_Jm_Imask(packed.imei[0], jm) |
                               Clustered_Jm_Imask(packed.imei[1], jm))
                            : 0u;
                    if (i_lane == 0 &&
                        j_lane < kClusteredSuperClusterClusters)
                    {
                        if (j_lane < active_cluster_count && cluster_j >= 0 &&
                            (combined_imask & (1u << j_lane)) != 0u)
                        {
                            const int cluster_i = shared_i_cluster_ids[j_lane];
                            const VECTOR pair_shift =
                                Get_Clustered_Pair_Shift_Vector(
                                    cluster_centers[cluster_i],
                                    cluster_centers[cluster_j], cell, rcell);
                            shared_j_shift[j_lane] = {
                                pair_shift.x, pair_shift.y, pair_shift.z, 0.0f};
                        }
                        else
                        {
                            shared_j_shift[j_lane] = {0.0f, 0.0f, 0.0f, 0.0f};
                        }
                    }
                    __syncthreads();

                    const unsigned int imask =
                        cluster_j >= 0 ? Clustered_Jm_Imask(imei, jm) : 0u;
                    if (cluster_j >= 0 && imask != 0u)
                    {
                        unsigned int valid_mask_j = 0u;
                        int atom_j = -1;
                        int atom_j_is_local = 0;
                        float4 r2_xq = {0.0f, 0.0f, 0.0f, 0.0f};
                        int r2_lj_type = 0;
                        if (i_lane == 0)
                        {
                            valid_mask_j = cluster_valid_masks[cluster_j];
                            if ((valid_mask_j & (1u << j_lane)) != 0u)
                            {
                                const int sorted_atom_j =
                                    cluster_offsets[cluster_j] + j_lane;
                                atom_j = sorted_atom_ids[sorted_atom_j];
                                atom_j_is_local =
                                    atom_j < local_atom_numbers ? 1 : 0;
                                r2_xq = sorted_xq[sorted_atom_j];
                                r2_lj_type = sorted_lj_type[sorted_atom_j];
                            }
                        }
                        valid_mask_j = Broadcast_Clustered_Subgroup_Value(
                            valid_mask_j, lane, cluster_size);
                        if ((valid_mask_j & (1u << j_lane)) != 0u)
                        {
                            atom_j = Broadcast_Clustered_Subgroup_Value(
                                atom_j, lane, cluster_size);
                            atom_j_is_local = Broadcast_Clustered_Subgroup_Value(
                                atom_j_is_local, lane, cluster_size);
                            r2_xq = Broadcast_Clustered_Subgroup_Float4(
                                r2_xq, lane, cluster_size);
                            r2_lj_type = Broadcast_Clustered_Subgroup_Value(
                                r2_lj_type, lane, cluster_size);
                            const VECTOR_LJ r2 =
                                Make_Packed_LJ_Atom(r2_xq, r2_lj_type);
                            VECTOR fcj_buf = {0.0f, 0.0f, 0.0f};
                            for (int i_local = 0;
                                 i_local < active_cluster_count; i_local += 1)
                            {
                                if ((imask & (1u << i_local)) == 0u)
                                {
                                    continue;
                                }
                                const unsigned int valid_mask_i =
                                    shared_i_valid_masks[i_local];
                                const unsigned int local_mask_i =
                                    shared_i_local_masks[i_local];
                                if ((valid_mask_i & (1u << i_lane)) == 0u ||
                                    (local_mask_i & (1u << i_lane)) == 0u)
                                {
                                    continue;
                                }
                                const int cluster_i = shared_i_cluster_ids[i_local];
                                const int exclusion_index =
                                    Clustered_Exclusion_Index(
                                        imei, jm, i_local);
                                const unsigned long long exclusion_mask =
                                    exclusion_index >= 0
                                        ? exclusion_mask_pool[exclusion_index]
                                        : 0ull;
                                if (sci_is_central && cluster_i == cluster_j &&
                                    atom_j < local_atom_numbers &&
                                    j_lane <= i_lane)
                                {
                                    continue;
                                }
                                if ((exclusion_mask &
                                     (1ull << (i_lane * cluster_size + j_lane))) !=
                                    0ull)
                                {
                                    continue;
                                }

                                const float4 r1_xq =
                                    shared_i_xq[i_local * cluster_size + i_lane];
                                const int r1_lj_type = shared_i_lj_type
                                    [i_local * cluster_size + i_lane];
                                const VECTOR_LJ r1 =
                                    Make_Packed_LJ_Atom(r1_xq, r1_lj_type);
                                const float4 pair_shift4 =
                                    shared_j_shift[i_local];
                                const VECTOR dr =
                                    Get_Clustered_Shifted_Displacement(
                                        r2, r1,
                                        {pair_shift4.x, pair_shift4.y,
                                         pair_shift4.z});
                                const float dr2 = dr * dr;
                                if (dr2 >= cutoff_sq || dr2 == 0.0f)
                                {
                                    continue;
                                }
                                const float dr_abs = sqrtf(dr2);
                                const int atom_pair_LJ_type =
                                    Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                                const float A = LJ_type_A[atom_pair_LJ_type];
                                const float B = LJ_type_B[atom_pair_LJ_type];
                                const float ij_factor =
                                    atom_j < local_atom_numbers ? 1.0f : 0.5f;

                                if (need_force)
                                {
                                    float frc_abs =
                                        Get_LJ_Force(r1, r2, dr_abs, A, B);
                                    if (need_coulomb)
                                    {
                                        frc_abs -= Get_Direct_Coulomb_Force(
                                            r1, r2, dr_abs, pme_beta);
                                    }
                                    const VECTOR frc_lin = frc_abs * dr;
                                    fci_buf[i_local] =
                                        fci_buf[i_local] + frc_lin;
                                    if (atom_j_is_local != 0)
                                    {
                                        fcj_buf = fcj_buf - frc_lin;
                                    }
                                    virial_buf[i_local] =
                                        virial_buf[i_local] -
                                        ij_factor *
                                            Get_Virial_From_Force_Dis(
                                                frc_lin, dr);
                                }
                                if (need_energy)
                                {
                                    energy_lj_buf[i_local] +=
                                        ij_factor *
                                        Get_LJ_Energy(r1, r2, dr_abs, A, B);
                                    if (need_coulomb)
                                    {
                                        energy_coulomb_buf[i_local] +=
                                            ij_factor *
                                            Get_Direct_Coulomb_Energy(
                                                r1, r2, dr_abs, pme_beta);
                                    }
                                }
                            }

                            if (need_force && atom_j_is_local != 0)
                            {
                                VECTOR reduced = Reduce_Clustered_Subgroup_Vector(
                                    fcj_buf, lane, cluster_size);
                                if (i_lane == 0)
                                {
                                    atomicAdd(frc + atom_j, reduced);
                                }
                            }
                        }
                    }
                    __syncthreads();
                }
            }

            for (int i_local = 0; i_local < active_cluster_count; i_local += 1)
            {
                const unsigned int valid_mask_i = shared_i_valid_masks[i_local];
                const unsigned int local_mask_i = shared_i_local_masks[i_local];
                const bool active_i =
                    (valid_mask_i & (1u << i_lane)) != 0u &&
                    (local_mask_i & (1u << i_lane)) != 0u;

                if (need_force)
                {
                    VECTOR reduced = active_i ? fci_buf[i_local]
                                              : VECTOR{0.0f, 0.0f, 0.0f};
                    reduced =
                        Reduce_Clustered_Warp_Vector_Over_J(reduced, cluster_size);
                    if (lane < cluster_size)
                    {
                        if (warp_id == 0)
                        {
                            fci_buf[i_local] = reduced;
                        }
                        else
                        {
                            warp1_i_force[i_local][lane] = {
                                reduced.x, reduced.y, reduced.z, 0.0f};
                        }
                    }
                }
                if (need_energy)
                {
                    float reduced_lj = active_i ? energy_lj_buf[i_local] : 0.0f;
                    float reduced_coulomb =
                        active_i ? energy_coulomb_buf[i_local] : 0.0f;
                    reduced_lj = Reduce_Clustered_Warp_Float_Over_J(
                        reduced_lj, cluster_size);
                    reduced_coulomb = Reduce_Clustered_Warp_Float_Over_J(
                        reduced_coulomb, cluster_size);
                    if (lane < cluster_size)
                    {
                        if (warp_id == 0)
                        {
                            energy_lj_buf[i_local] = reduced_lj;
                            energy_coulomb_buf[i_local] = reduced_coulomb;
                        }
                        else
                        {
                            warp1_i_energy_lj[i_local][lane] = reduced_lj;
                            warp1_i_energy_coulomb[i_local][lane] =
                                reduced_coulomb;
                        }
                    }
                }
                LTMatrix3 reduced = active_i ? virial_buf[i_local]
                                             : LTMatrix3(0.0f);
                reduced =
                    Reduce_Clustered_Warp_Virial_Over_J(reduced, cluster_size);
                if (lane < cluster_size)
                {
                    if (warp_id == 0)
                    {
                        virial_buf[i_local] = reduced;
                    }
                    else
                    {
                        warp1_i_virial_lo[i_local][lane] =
                            Pack_Clustered_Virial_Lo(reduced);
                        warp1_i_virial_hi[i_local][lane] =
                            Pack_Clustered_Virial_Hi(reduced);
                    }
                }
            }
            __syncthreads();

            if (warp_id == 0 && j_lane == 0)
            {
                for (int i_local = 0; i_local < active_cluster_count;
                     i_local += 1)
                {
                    const unsigned int valid_mask_i =
                        shared_i_valid_masks[i_local];
                    const unsigned int local_mask_i =
                        shared_i_local_masks[i_local];
                    const bool active_i =
                        (valid_mask_i & (1u << i_lane)) != 0u &&
                        (local_mask_i & (1u << i_lane)) != 0u;
                    if (!active_i)
                    {
                        continue;
                    }

                    const int atom_i =
                        shared_i_atom_ids[i_local * cluster_size + i_lane];
                    if (need_force)
                    {
                        const float4 warp1_force = warp1_i_force[i_local][i_lane];
                        atomicAdd(
                            frc + atom_i,
                            fci_buf[i_local] +
                                VECTOR{warp1_force.x, warp1_force.y,
                                       warp1_force.z});
                    }
                    if (need_energy)
                    {
                        const float total_energy_lj =
                            energy_lj_buf[i_local] +
                            warp1_i_energy_lj[i_local][i_lane];
                        const float total_energy_coulomb =
                            energy_coulomb_buf[i_local] +
                            warp1_i_energy_coulomb[i_local][i_lane];
                        atomicAdd(atom_energy + atom_i,
                                  total_energy_lj + total_energy_coulomb);
                        atomicAdd(atom_LJ_ene + atom_i, total_energy_lj);
                        if (need_coulomb)
                        {
                            atomicAdd(atom_direct_cf_energy + atom_i,
                                      total_energy_coulomb);
                        }
                    }
                    atomicAdd(atom_virial + atom_i,
                              virial_buf[i_local] +
                                  Unpack_Clustered_Virial(
                                      warp1_i_virial_lo[i_local][i_lane],
                                      warp1_i_virial_hi[i_local][i_lane]));
                }
            }
        }
        else
        {
            __shared__ float4 shared_i_xq[max_super_cluster_atoms];
            __shared__ int shared_i_lj_type[max_super_cluster_atoms];
            __shared__ int shared_i_atom_ids[max_super_cluster_atoms];
            __shared__ unsigned int shared_i_valid_masks[kClusteredSuperClusterClusters];
            __shared__ unsigned int shared_i_local_masks[kClusteredSuperClusterClusters];
            __shared__ int shared_i_cluster_ids[kClusteredSuperClusterClusters];
            __shared__ float4 warp1_i_force[kClusteredSuperClusterClusters]
                                            [max_cluster_size];
            __shared__ float warp1_i_energy_lj[kClusteredSuperClusterClusters]
                                              [max_cluster_size];
            __shared__ float warp1_i_energy_coulomb
                [kClusteredSuperClusterClusters][max_cluster_size];
            __shared__ float4 warp1_i_virial_lo
                [kClusteredSuperClusterClusters][max_cluster_size];
            __shared__ float2 warp1_i_virial_hi
                [kClusteredSuperClusterClusters][max_cluster_size];
            __shared__ float4 shared_j_shift[kClusteredSuperClusterClusters];

            const int i_lane = threadIdx.x;
            const int j_lane = threadIdx.y;
            const int lane = tid & (warpSize - 1);
            const int warp_id = tid / warpSize;
            const int active_cluster_count = cluster_i_end - cluster_i_start;
            const int i_slot = j_lane * cluster_size + i_lane;

            VECTOR fci_buf[kClusteredSuperClusterClusters];
            float energy_lj_buf[kClusteredSuperClusterClusters] = {};
            float energy_coulomb_buf[kClusteredSuperClusterClusters] = {};
            LTMatrix3 virial_buf[kClusteredSuperClusterClusters];
            for (int i_local = 0; i_local < kClusteredSuperClusterClusters;
                 i_local += 1)
            {
                fci_buf[i_local] = {0.0f, 0.0f, 0.0f};
                virial_buf[i_local] = {0.0f, 0.0f, 0.0f,
                                       0.0f, 0.0f, 0.0f};
            }

            if (j_lane < active_cluster_count)
            {
                const int cluster_i = cluster_i_start + j_lane;
                if (i_lane == 0)
                {
                    shared_i_valid_masks[j_lane] = cluster_valid_masks[cluster_i];
                    shared_i_local_masks[j_lane] = cluster_local_masks[cluster_i];
                    shared_i_cluster_ids[j_lane] = cluster_i;
                }
                if ((cluster_valid_masks[cluster_i] & (1u << i_lane)) != 0u)
                {
                    const int sorted_atom_i = cluster_offsets[cluster_i] + i_lane;
                    shared_i_xq[i_slot] = sorted_xq[sorted_atom_i];
                    shared_i_lj_type[i_slot] = sorted_lj_type[sorted_atom_i];
                    shared_i_atom_ids[i_slot] = sorted_atom_ids[sorted_atom_i];
                }
                else
                {
                    shared_i_atom_ids[i_slot] = -1;
                }
            }
            else if (i_lane == 0 && j_lane < kClusteredSuperClusterClusters)
            {
                shared_i_valid_masks[j_lane] = 0u;
                shared_i_local_masks[j_lane] = 0u;
                shared_i_cluster_ids[j_lane] = -1;
            }
            __syncthreads();

            for (int packed_idx = sci_entry.cjpacked_begin;
                 packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
            {
                const LJ_CLUSTERED_CJ_PACKED packed =
                    cj_packed_entries[packed_idx];
                const LJ_CLUSTERED_IMEI imei = packed.imei[warp_id];
                for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
                {
                    const int cluster_j = packed.cj[jm];
                    const unsigned int combined_imask =
                        cluster_j >= 0
                            ? (Clustered_Jm_Imask(packed.imei[0], jm) |
                               Clustered_Jm_Imask(packed.imei[1], jm))
                            : 0u;
                    if (i_lane == 0 &&
                        j_lane < kClusteredSuperClusterClusters)
                    {
                        if (j_lane < active_cluster_count && cluster_j >= 0 &&
                            (combined_imask & (1u << j_lane)) != 0u)
                        {
                            const int cluster_i = shared_i_cluster_ids[j_lane];
                            const VECTOR pair_shift =
                                Get_Clustered_Pair_Shift_Vector(
                                    cluster_centers[cluster_i],
                                    cluster_centers[cluster_j], cell, rcell);
                            shared_j_shift[j_lane] = {
                                pair_shift.x, pair_shift.y, pair_shift.z, 0.0f};
                        }
                        else
                        {
                            shared_j_shift[j_lane] = {0.0f, 0.0f, 0.0f, 0.0f};
                        }
                    }
                    __syncthreads();

                    const unsigned int imask =
                        cluster_j >= 0 ? Clustered_Jm_Imask(imei, jm) : 0u;
                    if (cluster_j >= 0 && imask != 0u)
                    {
                        unsigned int valid_mask_j = 0u;
                        int atom_j = -1;
                        int atom_j_is_local = 0;
                        float4 r2_xq = {0.0f, 0.0f, 0.0f, 0.0f};
                        int r2_lj_type = 0;
                        if (i_lane == 0)
                        {
                            valid_mask_j = cluster_valid_masks[cluster_j];
                            if ((valid_mask_j & (1u << j_lane)) != 0u)
                            {
                                const int sorted_atom_j =
                                    cluster_offsets[cluster_j] + j_lane;
                                atom_j = sorted_atom_ids[sorted_atom_j];
                                atom_j_is_local =
                                    atom_j < local_atom_numbers ? 1 : 0;
                                r2_xq = sorted_xq[sorted_atom_j];
                                r2_lj_type = sorted_lj_type[sorted_atom_j];
                            }
                        }
                        valid_mask_j = Broadcast_Clustered_Subgroup_Value(
                            valid_mask_j, lane, cluster_size);
                        if ((valid_mask_j & (1u << j_lane)) != 0u)
                        {
                            atom_j = Broadcast_Clustered_Subgroup_Value(
                                atom_j, lane, cluster_size);
                            atom_j_is_local = Broadcast_Clustered_Subgroup_Value(
                                atom_j_is_local, lane, cluster_size);
                            r2_xq = Broadcast_Clustered_Subgroup_Float4(
                                r2_xq, lane, cluster_size);
                            r2_lj_type = Broadcast_Clustered_Subgroup_Value(
                                r2_lj_type, lane, cluster_size);
                            const VECTOR_LJ r2 =
                                Make_Packed_LJ_Atom(r2_xq, r2_lj_type);
                            VECTOR fcj_buf = {0.0f, 0.0f, 0.0f};
                            for (int i_local = 0;
                                 i_local < active_cluster_count; i_local += 1)
                            {
                                if ((imask & (1u << i_local)) == 0u)
                                {
                                    continue;
                                }
                                const unsigned int valid_mask_i =
                                    shared_i_valid_masks[i_local];
                                const unsigned int local_mask_i =
                                    shared_i_local_masks[i_local];
                                if ((valid_mask_i & (1u << i_lane)) == 0u ||
                                    (local_mask_i & (1u << i_lane)) == 0u)
                                {
                                    continue;
                                }
                                const int cluster_i = shared_i_cluster_ids[i_local];
                                const int exclusion_index =
                                    Clustered_Exclusion_Index(
                                        imei, jm, i_local);
                                const unsigned long long exclusion_mask =
                                    exclusion_index >= 0
                                        ? exclusion_mask_pool[exclusion_index]
                                        : 0ull;
                                if (sci_is_central && cluster_i == cluster_j &&
                                    atom_j < local_atom_numbers &&
                                    j_lane <= i_lane)
                                {
                                    continue;
                                }
                                if ((exclusion_mask &
                                     (1ull << (i_lane * cluster_size + j_lane))) !=
                                    0ull)
                                {
                                    continue;
                                }

                                const float4 r1_xq =
                                    shared_i_xq[i_local * cluster_size + i_lane];
                                const int r1_lj_type = shared_i_lj_type
                                    [i_local * cluster_size + i_lane];
                                const VECTOR_LJ r1 =
                                    Make_Packed_LJ_Atom(r1_xq, r1_lj_type);
                                const float4 pair_shift4 =
                                    shared_j_shift[i_local];
                                const VECTOR dr =
                                    Get_Clustered_Shifted_Displacement(
                                        r2, r1,
                                        {pair_shift4.x, pair_shift4.y,
                                         pair_shift4.z});
                                const float dr2 = dr * dr;
                                if (dr2 >= cutoff_sq || dr2 == 0.0f)
                                {
                                    continue;
                                }
                                const float dr_abs = sqrtf(dr2);
                                const int atom_pair_LJ_type =
                                    Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                                const float A = LJ_type_A[atom_pair_LJ_type];
                                const float B = LJ_type_B[atom_pair_LJ_type];
                                const float ij_factor =
                                    atom_j < local_atom_numbers ? 1.0f : 0.5f;

                                if (need_force)
                                {
                                    float frc_abs =
                                        Get_LJ_Force(r1, r2, dr_abs, A, B);
                                    if (need_coulomb)
                                    {
                                        frc_abs -= Get_Direct_Coulomb_Force(
                                            r1, r2, dr_abs, pme_beta);
                                    }
                                    const VECTOR frc_lin = frc_abs * dr;
                                    fci_buf[i_local] =
                                        fci_buf[i_local] + frc_lin;
                                    if (atom_j_is_local != 0)
                                    {
                                        fcj_buf = fcj_buf - frc_lin;
                                    }
                                    if (need_virial)
                                    {
                                        virial_buf[i_local] =
                                            virial_buf[i_local] -
                                            ij_factor *
                                                Get_Virial_From_Force_Dis(
                                                    frc_lin, dr);
                                    }
                                }
                                if (need_energy)
                                {
                                    energy_lj_buf[i_local] +=
                                        ij_factor *
                                        Get_LJ_Energy(r1, r2, dr_abs, A, B);
                                    if (need_coulomb)
                                    {
                                        energy_coulomb_buf[i_local] +=
                                            ij_factor *
                                            Get_Direct_Coulomb_Energy(
                                                r1, r2, dr_abs, pme_beta);
                                    }
                                }
                            }

                            if (need_force && atom_j_is_local != 0)
                            {
                                VECTOR reduced = Reduce_Clustered_Subgroup_Vector(
                                    fcj_buf, lane, cluster_size);
                                if (i_lane == 0)
                                {
                                    atomicAdd(frc + atom_j, reduced);
                                }
                            }
                        }
                    }
                    __syncthreads();
                }
            }

            for (int i_local = 0; i_local < active_cluster_count; i_local += 1)
            {
                const unsigned int valid_mask_i = shared_i_valid_masks[i_local];
                const unsigned int local_mask_i = shared_i_local_masks[i_local];
                const bool active_i =
                    (valid_mask_i & (1u << i_lane)) != 0u &&
                    (local_mask_i & (1u << i_lane)) != 0u;

                if (need_force)
                {
                    VECTOR reduced = active_i ? fci_buf[i_local]
                                              : VECTOR{0.0f, 0.0f, 0.0f};
                    reduced =
                        Reduce_Clustered_Warp_Vector_Over_J(reduced, cluster_size);
                    if (lane < cluster_size)
                    {
                        if (warp_id == 0)
                        {
                            fci_buf[i_local] = reduced;
                        }
                        else
                        {
                            warp1_i_force[i_local][lane] = {
                                reduced.x, reduced.y, reduced.z, 0.0f};
                        }
                    }
                }
                if (need_energy)
                {
                    float reduced_lj = active_i ? energy_lj_buf[i_local] : 0.0f;
                    float reduced_coulomb =
                        active_i ? energy_coulomb_buf[i_local] : 0.0f;
                    reduced_lj = Reduce_Clustered_Warp_Float_Over_J(
                        reduced_lj, cluster_size);
                    reduced_coulomb = Reduce_Clustered_Warp_Float_Over_J(
                        reduced_coulomb, cluster_size);
                    if (lane < cluster_size)
                    {
                        if (warp_id == 0)
                        {
                            energy_lj_buf[i_local] = reduced_lj;
                            energy_coulomb_buf[i_local] = reduced_coulomb;
                        }
                        else
                        {
                            warp1_i_energy_lj[i_local][lane] = reduced_lj;
                            warp1_i_energy_coulomb[i_local][lane] =
                                reduced_coulomb;
                        }
                    }
                }
                if (need_virial)
                {
                    LTMatrix3 reduced = active_i ? virial_buf[i_local]
                                                 : LTMatrix3(0.0f);
                    reduced = Reduce_Clustered_Warp_Virial_Over_J(
                        reduced, cluster_size);
                    if (lane < cluster_size)
                    {
                        if (warp_id == 0)
                        {
                            virial_buf[i_local] = reduced;
                        }
                        else
                        {
                            warp1_i_virial_lo[i_local][lane] =
                                Pack_Clustered_Virial_Lo(reduced);
                            warp1_i_virial_hi[i_local][lane] =
                                Pack_Clustered_Virial_Hi(reduced);
                        }
                    }
                }
            }
            __syncthreads();

            if (warp_id == 0 && j_lane == 0)
            {
                for (int i_local = 0; i_local < active_cluster_count;
                     i_local += 1)
                {
                    const unsigned int valid_mask_i =
                        shared_i_valid_masks[i_local];
                    const unsigned int local_mask_i =
                        shared_i_local_masks[i_local];
                    const bool active_i =
                        (valid_mask_i & (1u << i_lane)) != 0u &&
                        (local_mask_i & (1u << i_lane)) != 0u;
                    if (!active_i)
                    {
                        continue;
                    }

                    const int atom_i =
                        shared_i_atom_ids[i_local * cluster_size + i_lane];
                    if (need_force)
                    {
                        const float4 warp1_force = warp1_i_force[i_local][i_lane];
                        atomicAdd(
                            frc + atom_i,
                            fci_buf[i_local] +
                                VECTOR{warp1_force.x, warp1_force.y,
                                       warp1_force.z});
                    }
                    if (need_energy)
                    {
                        const float total_energy_lj =
                            energy_lj_buf[i_local] +
                            warp1_i_energy_lj[i_local][i_lane];
                        const float total_energy_coulomb =
                            energy_coulomb_buf[i_local] +
                            warp1_i_energy_coulomb[i_local][i_lane];
                        atomicAdd(atom_energy + atom_i,
                                  total_energy_lj + total_energy_coulomb);
                        atomicAdd(atom_LJ_ene + atom_i, total_energy_lj);
                        if (need_coulomb)
                        {
                            atomicAdd(atom_direct_cf_energy + atom_i,
                                      total_energy_coulomb);
                        }
                    }
                    if (need_virial)
                    {
                        atomicAdd(atom_virial + atom_i,
                                  virial_buf[i_local] +
                                      Unpack_Clustered_Virial(
                                          warp1_i_virial_lo[i_local][i_lane],
                                          warp1_i_virial_hi[i_local][i_lane]));
                    }
                }
            }
        }
#endif
    }
}

// Reference GPU path for validating NBNXM metadata without the warp-split
// execution scheme. This keeps the clustered payload unchanged while aligning
// pair traversal with the scalar logic.
template <bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb>
static __global__ void Reference_Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Device(
    const int sci_numbers, const int cluster_size,
    const int super_cluster_clusters, const int local_atom_numbers,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const int* super_cluster_offsets,
    const LJ_CLUSTERED_SCI* sci_entries,
    const LJ_CLUSTERED_CJ_PACKED* cj_packed_entries,
    const unsigned long long* exclusion_mask_pool, const int* sorted_atom_ids,
    const float4* sorted_xq, const int* sorted_lj_type, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* LJ_type_A, const float* LJ_type_B,
    const float cutoff, VECTOR* frc, const float pme_beta, float* atom_energy,
    LTMatrix3* atom_virial, float* atom_direct_cf_energy, float* atom_LJ_ene)
{
    constexpr int max_cluster_size = kClusteredClusterSize;
#ifdef USE_GPU
    const int sci = blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (sci >= sci_numbers ||
        tid >= super_cluster_clusters * cluster_size)
    {
        return;
    }
#else
#pragma omp parallel for schedule(dynamic)
    for (int sci = 0; sci < sci_numbers; sci += 1)
#endif
    {
        const LJ_CLUSTERED_SCI sci_entry = sci_entries[sci];
        const int super_i = sci_entry.supercluster_id;
        const int cluster_i_start = super_cluster_offsets[super_i];
        const int cluster_i_end = super_cluster_offsets[super_i + 1];
        const VECTOR shift_vec =
            Clustered_Shift_Vector_From_Id(sci_entry.shift_id, cell);
        const float cutoff_sq = cutoff * cutoff;
        (void)rcell;

#ifndef USE_GPU
        for (int cluster_i = cluster_i_start; cluster_i < cluster_i_end;
             cluster_i += 1)
        {
            const unsigned int valid_mask_i = cluster_valid_masks[cluster_i];
            const unsigned int local_mask_i = cluster_local_masks[cluster_i];
            const int i_local = cluster_i - cluster_i_start;
            for (int lane_i = 0; lane_i < cluster_size; lane_i += 1)
            {
                if ((valid_mask_i & (1u << lane_i)) == 0u ||
                    (local_mask_i & (1u << lane_i)) == 0u)
                {
                    continue;
                }
                const int sorted_atom_i = cluster_offsets[cluster_i] + lane_i;
                const int atom_i = sorted_atom_ids[sorted_atom_i];
                VECTOR_LJ r1 = Make_Packed_LJ_Atom(sorted_xq[sorted_atom_i],
                                                   sorted_lj_type[sorted_atom_i]);
                r1.crd = r1.crd + shift_vec;
                VECTOR frc_i = {0.0f, 0.0f, 0.0f};
                float energy_lj = 0.0f;
                float energy_coulomb = 0.0f;
                LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

                for (int packed_idx = sci_entry.cjpacked_begin;
                     packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
                {
                    const LJ_CLUSTERED_CJ_PACKED& packed =
                        cj_packed_entries[packed_idx];
                    for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
                    {
                        const int cluster_j = packed.cj[jm];
                        if (cluster_j < 0)
                        {
                            continue;
                        }
                        const unsigned int imask =
                            Clustered_Jm_Imask(packed.imei[0], jm) |
                            Clustered_Jm_Imask(packed.imei[1], jm);
                        if ((imask & (1u << i_local)) == 0u)
                        {
                            continue;
                        }
                        const unsigned int valid_mask_j =
                            cluster_valid_masks[cluster_j];
                        const int exclusion_index =
                            Clustered_First_Exclusion_Index(packed, jm, i_local);
                        const unsigned long long exclusion_mask =
                            exclusion_index >= 0
                                ? exclusion_mask_pool[exclusion_index]
                                : 0ull;
                        VECTOR frc_j[max_cluster_size] = {};

                        for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
                        {
                            if ((valid_mask_j & (1u << lane_j)) == 0u)
                            {
                                continue;
                            }
                            const int sorted_atom_j =
                                cluster_offsets[cluster_j] + lane_j;
                            const int atom_j = sorted_atom_ids[sorted_atom_j];
                            if (sci_entry.shift_id == kClusteredCentralShiftId &&
                                cluster_i == cluster_j &&
                                atom_j < local_atom_numbers &&
                                lane_j <= lane_i)
                            {
                                continue;
                            }
                            if ((exclusion_mask &
                                 (1ull << (lane_i * cluster_size + lane_j))) !=
                                0ull)
                            {
                                continue;
                            }
                            const VECTOR_LJ r2 = Make_Packed_LJ_Atom(
                                sorted_xq[sorted_atom_j],
                                sorted_lj_type[sorted_atom_j]);
                            const VECTOR dr =
                                Get_Periodic_Displacement(r2, r1, cell, rcell);
                            const float dr2 = dr * dr;
                            if (dr2 >= cutoff_sq || dr2 == 0.0f)
                            {
                                continue;
                            }
                            const float dr_abs = sqrtf(dr2);
                            const int atom_pair_LJ_type =
                                Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                            const float A = LJ_type_A[atom_pair_LJ_type];
                            const float B = LJ_type_B[atom_pair_LJ_type];
                            const float ij_factor =
                                atom_j < local_atom_numbers ? 1.0f : 0.5f;
                            if (need_force)
                            {
                                float frc_abs =
                                    Get_LJ_Force(r1, r2, dr_abs, A, B);
                                if (need_coulomb)
                                {
                                    frc_abs -= Get_Direct_Coulomb_Force(
                                        r1, r2, dr_abs, pme_beta);
                                }
                                const VECTOR frc_lin = frc_abs * dr;
                                frc_i = frc_i + frc_lin;
                                if (atom_j < local_atom_numbers)
                                {
                                    frc_j[lane_j] = frc_j[lane_j] - frc_lin;
                                }
                                if (need_virial)
                                {
                                    virial = virial -
                                             ij_factor *
                                                 Get_Virial_From_Force_Dis(
                                                     frc_lin, dr);
                                }
                            }
                            if (need_energy)
                            {
                                energy_lj +=
                                    ij_factor *
                                    Get_LJ_Energy(r1, r2, dr_abs, A, B);
                                if (need_coulomb)
                                {
                                    energy_coulomb +=
                                        ij_factor *
                                        Get_Direct_Coulomb_Energy(
                                            r1, r2, dr_abs, pme_beta);
                                }
                            }
                        }
                        if (need_force)
                        {
                            for (int lane_j = 0; lane_j < cluster_size;
                                 lane_j += 1)
                            {
                                if ((valid_mask_j & (1u << lane_j)) == 0u)
                                {
                                    continue;
                                }
                                const int sorted_atom_j =
                                    cluster_offsets[cluster_j] + lane_j;
                                const int atom_j = sorted_atom_ids[sorted_atom_j];
                                if (atom_j < local_atom_numbers)
                                {
                                    atomicAdd(frc + atom_j, frc_j[lane_j]);
                                }
                            }
                        }
                    }
                }

                if (need_force)
                {
                    atomicAdd(frc + atom_i, frc_i);
                }
                if (need_energy)
                {
                    atomicAdd(atom_energy + atom_i, energy_lj + energy_coulomb);
                    atomicAdd(atom_LJ_ene + atom_i, energy_lj);
                    if (need_coulomb)
                    {
                        atomicAdd(atom_direct_cf_energy + atom_i,
                                  energy_coulomb);
                    }
                }
                if (need_virial)
                {
                    atomicAdd(atom_virial + atom_i, virial);
                }
            }
        }
#else
        const int active_cluster_count = cluster_i_end - cluster_i_start;
        const int i_local = tid / cluster_size;
        const int lane_i = tid % cluster_size;
        if (i_local >= active_cluster_count)
        {
            return;
        }
        const int cluster_i = cluster_i_start + i_local;
        const unsigned int valid_mask_i = cluster_valid_masks[cluster_i];
        const unsigned int local_mask_i = cluster_local_masks[cluster_i];
        if ((valid_mask_i & (1u << lane_i)) == 0u ||
            (local_mask_i & (1u << lane_i)) == 0u)
        {
            return;
        }

        const int sorted_atom_i = cluster_offsets[cluster_i] + lane_i;
        const int atom_i = sorted_atom_ids[sorted_atom_i];
        VECTOR_LJ r1 = Make_Packed_LJ_Atom(sorted_xq[sorted_atom_i],
                                           sorted_lj_type[sorted_atom_i]);
        r1.crd = r1.crd + shift_vec;
        VECTOR frc_i = {0.0f, 0.0f, 0.0f};
        float energy_lj = 0.0f;
        float energy_coulomb = 0.0f;
        LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        for (int packed_idx = sci_entry.cjpacked_begin;
             packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
        {
            const LJ_CLUSTERED_CJ_PACKED packed =
                cj_packed_entries[packed_idx];
            for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
            {
                const int cluster_j = packed.cj[jm];
                if (cluster_j < 0)
                {
                    continue;
                }
                const unsigned int imask =
                    Clustered_Jm_Imask(packed.imei[0], jm) |
                    Clustered_Jm_Imask(packed.imei[1], jm);
                if ((imask & (1u << i_local)) == 0u)
                {
                    continue;
                }
                const unsigned int valid_mask_j =
                    cluster_valid_masks[cluster_j];
                const int exclusion_index =
                    Clustered_First_Exclusion_Index(packed, jm, i_local);
                const unsigned long long exclusion_mask =
                    exclusion_index >= 0 ? exclusion_mask_pool[exclusion_index]
                                         : 0ull;
                VECTOR frc_j[max_cluster_size] = {};

                for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
                {
                    if ((valid_mask_j & (1u << lane_j)) == 0u)
                    {
                        continue;
                    }
                    const int sorted_atom_j =
                        cluster_offsets[cluster_j] + lane_j;
                    const int atom_j = sorted_atom_ids[sorted_atom_j];
                    if (sci_entry.shift_id == kClusteredCentralShiftId &&
                        cluster_i == cluster_j &&
                        atom_j < local_atom_numbers && lane_j <= lane_i)
                    {
                        continue;
                    }
                    if ((exclusion_mask &
                         (1ull << (lane_i * cluster_size + lane_j))) != 0ull)
                    {
                        continue;
                    }
                    const VECTOR_LJ r2 = Make_Packed_LJ_Atom(
                        sorted_xq[sorted_atom_j], sorted_lj_type[sorted_atom_j]);
                    const VECTOR dr =
                        Get_Periodic_Displacement(r2, r1, cell, rcell);
                    const float dr2 = dr * dr;
                    if (dr2 >= cutoff_sq || dr2 == 0.0f)
                    {
                        continue;
                    }
                    const float dr_abs = sqrtf(dr2);
                    const int atom_pair_LJ_type =
                        Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                    const float A = LJ_type_A[atom_pair_LJ_type];
                    const float B = LJ_type_B[atom_pair_LJ_type];
                    const float ij_factor =
                        atom_j < local_atom_numbers ? 1.0f : 0.5f;
                    if (need_force)
                    {
                        float frc_abs = Get_LJ_Force(r1, r2, dr_abs, A, B);
                        if (need_coulomb)
                        {
                            frc_abs -= Get_Direct_Coulomb_Force(
                                r1, r2, dr_abs, pme_beta);
                        }
                        const VECTOR frc_lin = frc_abs * dr;
                        frc_i = frc_i + frc_lin;
                        if (atom_j < local_atom_numbers)
                        {
                            frc_j[lane_j] = frc_j[lane_j] - frc_lin;
                        }
                        if (need_virial)
                        {
                            virial = virial -
                                     ij_factor *
                                         Get_Virial_From_Force_Dis(frc_lin, dr);
                        }
                    }
                    if (need_energy)
                    {
                        energy_lj +=
                            ij_factor * Get_LJ_Energy(r1, r2, dr_abs, A, B);
                        if (need_coulomb)
                        {
                            energy_coulomb +=
                                ij_factor *
                                Get_Direct_Coulomb_Energy(
                                    r1, r2, dr_abs, pme_beta);
                        }
                    }
                }
                if (need_force)
                {
                    for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
                    {
                        if ((valid_mask_j & (1u << lane_j)) == 0u)
                        {
                            continue;
                        }
                        const int sorted_atom_j =
                            cluster_offsets[cluster_j] + lane_j;
                        const int atom_j = sorted_atom_ids[sorted_atom_j];
                        if (atom_j < local_atom_numbers)
                        {
                            atomicAdd(frc + atom_j, frc_j[lane_j]);
                        }
                    }
                }
            }
        }

        if (need_force)
        {
            atomicAdd(frc + atom_i, frc_i);
        }
        if (need_energy)
        {
            atomicAdd(atom_energy + atom_i, energy_lj + energy_coulomb);
            atomicAdd(atom_LJ_ene + atom_i, energy_lj);
            if (need_coulomb)
            {
                atomicAdd(atom_direct_cf_energy + atom_i, energy_coulomb);
            }
        }
        if (need_virial)
        {
            atomicAdd(atom_virial + atom_i, virial);
        }
#endif
    }
}

void LENNARD_JONES_INFORMATION::LJ_Malloc()
{
    Malloc_Safely((void**)&h_atom_LJ_type, sizeof(int) * atom_numbers);
    Malloc_Safely((void**)&h_LJ_A, sizeof(float) * pair_type_numbers);
    Malloc_Safely((void**)&h_LJ_B, sizeof(float) * pair_type_numbers);
    Malloc_Safely((void**)&h_LJ_energy_atom, sizeof(float) * atom_numbers);
}

static __global__ void Total_C6_Get(int atom_numbers, int* atom_lj_type,
                                    float* d_lj_b, float* d_factor)
{
    int j;
    double temp_sum = 0;
    int x, y;
    int itype, jtype, atom_pair_LJ_type;
#ifdef USE_GPU
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers;
         i += gridDim.x * blockDim.x)
#else
#pragma omp parallel for firstprivate( \
        j, x, y, itype, jtype, atom_pair_LJ_type) reduction(+ : temp_sum)
    for (int i = 0; i < atom_numbers; i++)
#endif
    {
        itype = atom_lj_type[i];
        double temp_small_sum = 0;
#ifdef USE_GPU
        for (j = blockIdx.y * blockDim.y + threadIdx.y; j < atom_numbers;
             j += gridDim.y * blockDim.y)
#else
        for (j = 0; j < atom_numbers; j++)
#endif
        {
            jtype = atom_lj_type[j];
            y = (jtype - itype);
            x = y >> 31;
            y = (y ^ x) - x;
            x = jtype + itype;
            jtype = (x + y) >> 1;
            x = (x - y) >> 1;
            atom_pair_LJ_type = (jtype * (jtype + 1) >> 1) + x;
            temp_small_sum += d_lj_b[atom_pair_LJ_type];
        }
        temp_sum += temp_small_sum;
    }
    atomicAdd(d_factor, temp_sum);
}

void LENNARD_JONES_INFORMATION::Maybe_Apply_Ordered_Layout(
    CONTROLLER* controller, DOMAIN_INFORMATION* domain, LTMatrix3 cell,
    LTMatrix3 rcell, VECTOR box_length)
{
    (void)cell;
    if (!is_initialized || !use_ordered_layout || ordered_layout_applied ||
        domain == NULL)
    {
        return;
    }
    if (CONTROLLER::PP_MPI_size != 1 || domain->ghost_numbers != 0)
    {
        controller->printf(
            "    Skip LJ ordered layout: only single-rank local domains "
            "without ghosts are supported in this experiment.\n");
        ordered_layout_applied = 1;
        return;
    }
    if (domain->res_numbers < ordered_layout_min_residue_numbers ||
        domain->atom_numbers <= 0)
    {
        controller->printf(
            "    Skip LJ ordered layout: residue count %d is below threshold "
            "%d.\n",
            domain->res_numbers, ordered_layout_min_residue_numbers);
        ordered_layout_applied = 1;
        return;
    }
    if (box_length.x <= 0.0f || box_length.y <= 0.0f || box_length.z <= 0.0f)
    {
        controller->printf(
            "    Skip LJ ordered layout: invalid box lengths (%f, %f, %f).\n",
            box_length.x, box_length.y, box_length.z);
        ordered_layout_applied = 1;
        return;
    }

    std::vector<int> h_atom_local(domain->atom_numbers);
    std::vector<VECTOR> h_crd(domain->atom_numbers);
    std::vector<VECTOR> h_vel(domain->atom_numbers);
    std::vector<float> h_mass(domain->atom_numbers);
    std::vector<float> h_mass_inverse(domain->atom_numbers);
    std::vector<float> h_charge(domain->atom_numbers);
    std::vector<int> h_res_start(domain->res_numbers);
    std::vector<int> h_res_len(domain->res_numbers);

    deviceMemcpy(h_atom_local.data(), domain->atom_local,
                 sizeof(int) * domain->atom_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_crd.data(), domain->crd, sizeof(VECTOR) * domain->atom_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_vel.data(), domain->vel, sizeof(VECTOR) * domain->atom_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_mass.data(), domain->d_mass,
                 sizeof(float) * domain->atom_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_mass_inverse.data(), domain->d_mass_inverse,
                 sizeof(float) * domain->atom_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_charge.data(), domain->d_charge,
                 sizeof(float) * domain->atom_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_res_start.data(), domain->res_start,
                 sizeof(int) * domain->res_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_res_len.data(), domain->res_len,
                 sizeof(int) * domain->res_numbers,
                 deviceMemcpyDeviceToHost);

    std::vector<OrderedResiduePoint> points((size_t)domain->res_numbers);
    std::vector<int> residue_indices((size_t)domain->res_numbers);
    std::iota(residue_indices.begin(), residue_indices.end(), 0);
    for (int residue = 0; residue < domain->res_numbers; residue += 1)
    {
        OrderedResiduePoint point;
        point.residue_index = residue;
        point.atom_start = h_res_start[residue];
        point.atom_count = h_res_len[residue];
        point.wrapped = Wrap_To_Box_Fractional(h_crd[point.atom_start], rcell,
                                              box_length);
        point.normalized = {point.wrapped.x / box_length.x,
                            point.wrapped.y / box_length.y,
                            point.wrapped.z / box_length.z};
        point.point_hilbert =
            Hilbert_Index_3D(point.normalized, ordered_layout_max_depth);
        points[(size_t)residue] = point;
    }

    std::vector<CornerstoneLeaf> leaves;
    Build_Cornerstone_Leaves(points, residue_indices, 0, ordered_layout_max_depth,
                             ordered_layout_leaf_size, {0.0f, 0.0f, 0.0f},
                             {1.0f, 1.0f, 1.0f}, &leaves);
    for (auto& leaf : leaves)
    {
        const VECTOR center = 0.5f * (leaf.min_bound + leaf.max_bound);
        leaf.leaf_hilbert = Hilbert_Index_3D(center, ordered_layout_max_depth);
        std::stable_sort(
            leaf.residues.begin(), leaf.residues.end(),
            [&](int lhs, int rhs)
            {
                const uint64_t key_l = points[(size_t)lhs].point_hilbert;
                const uint64_t key_r = points[(size_t)rhs].point_hilbert;
                if (key_l != key_r)
                {
                    return key_l < key_r;
                }
                return lhs < rhs;
            });
    }
    std::stable_sort(
        leaves.begin(), leaves.end(),
        [](const CornerstoneLeaf& lhs, const CornerstoneLeaf& rhs)
        {
            if (lhs.leaf_hilbert != rhs.leaf_hilbert)
            {
                return lhs.leaf_hilbert < rhs.leaf_hilbert;
            }
            if (lhs.residues.empty() || rhs.residues.empty())
            {
                return lhs.residues.size() < rhs.residues.size();
            }
            return lhs.residues.front() < rhs.residues.front();
        });

    std::vector<int> residue_order;
    residue_order.reserve((size_t)domain->res_numbers);
    for (const auto& leaf : leaves)
    {
        residue_order.insert(residue_order.end(), leaf.residues.begin(),
                             leaf.residues.end());
    }
    if ((int)residue_order.size() != domain->res_numbers)
    {
        controller->printf(
            "    Skip LJ ordered layout: octree produced inconsistent residue "
            "count.\n");
        ordered_layout_applied = 1;
        return;
    }

    bool changed = false;
    for (int residue = 0; residue < domain->res_numbers; residue += 1)
    {
        if (residue_order[(size_t)residue] != residue)
        {
            changed = true;
            break;
        }
    }
    if (!changed)
    {
        controller->printf(
            "    LJ ordered layout leaves the current residue ordering "
            "unchanged.\n");
        ordered_layout_applied = 1;
        return;
    }

    std::vector<int> new_atom_local((size_t)domain->atom_numbers);
    std::vector<VECTOR> new_crd((size_t)domain->atom_numbers);
    std::vector<VECTOR> new_vel((size_t)domain->atom_numbers);
    std::vector<float> new_mass((size_t)domain->atom_numbers);
    std::vector<float> new_mass_inverse((size_t)domain->atom_numbers);
    std::vector<float> new_charge((size_t)domain->atom_numbers);
    std::vector<int> new_res_start((size_t)domain->res_numbers);
    std::vector<int> new_res_len((size_t)domain->res_numbers);
    std::vector<int> new_atom_local_id((size_t)domain->max_atom_numbers, -1);

    int write_atom = 0;
    for (int residue = 0; residue < domain->res_numbers; residue += 1)
    {
        const OrderedResiduePoint& point =
            points[(size_t)residue_order[(size_t)residue]];
        new_res_start[(size_t)residue] = write_atom;
        new_res_len[(size_t)residue] = point.atom_count;
        for (int atom = 0; atom < point.atom_count; atom += 1)
        {
            const int source = point.atom_start + atom;
            const int global_atom = h_atom_local[(size_t)source];
            new_atom_local[(size_t)write_atom] = global_atom;
            new_crd[(size_t)write_atom] = h_crd[(size_t)source];
            new_vel[(size_t)write_atom] = h_vel[(size_t)source];
            new_mass[(size_t)write_atom] = h_mass[(size_t)source];
            new_mass_inverse[(size_t)write_atom] =
                h_mass_inverse[(size_t)source];
            new_charge[(size_t)write_atom] = h_charge[(size_t)source];
            if (global_atom >= 0 &&
                global_atom < static_cast<int>(new_atom_local_id.size()))
            {
                new_atom_local_id[(size_t)global_atom] = write_atom;
            }
            write_atom += 1;
        }
    }

    deviceMemcpy(domain->atom_local, new_atom_local.data(),
                 sizeof(int) * domain->atom_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(domain->crd, new_crd.data(),
                 sizeof(VECTOR) * domain->atom_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(domain->vel, new_vel.data(),
                 sizeof(VECTOR) * domain->atom_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(domain->d_mass, new_mass.data(),
                 sizeof(float) * domain->atom_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(domain->d_mass_inverse, new_mass_inverse.data(),
                 sizeof(float) * domain->atom_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(domain->d_charge, new_charge.data(),
                 sizeof(float) * domain->atom_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(domain->res_start, new_res_start.data(),
                 sizeof(int) * domain->res_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(domain->res_len, new_res_len.data(),
                 sizeof(int) * domain->res_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(domain->atom_local_id, new_atom_local_id.data(),
                 sizeof(int) * domain->max_atom_numbers,
                 deviceMemcpyHostToDevice);

    ordered_layout_applied = 1;
    controller->printf(
        "    Applied LJ ordered layout with %d residues, %zu cornerstone "
        "leaves, depth=%d, leaf_size=%d.\n",
        domain->res_numbers, leaves.size(), ordered_layout_max_depth,
        ordered_layout_leaf_size);
}

void LENNARD_JONES_INFORMATION::Initial(CONTROLLER* controller, float cutoff,
                                        const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "LJ");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    controller->printf("START INITIALIZING LENNADR JONES INFORMATION:\n");
    const auto& lj = Xponge::system.classical_force_field.lj;
    Xponge::LennardJones local_lj;
    const Xponge::LennardJones* lj_to_use = NULL;
    if (module_name == NULL)
    {
        lj_to_use = &lj;
    }
    else if (controller->Command_Exist(this->module_name, "in_file"))
    {
        Xponge::Native_Load_LJ(&local_lj, controller, 0, this->module_name);
        lj_to_use = &local_lj;
    }
    if (lj_to_use != NULL)
    {
        atom_numbers = static_cast<int>(lj_to_use->atom_type.size());
        atom_type_numbers = lj_to_use->atom_type_numbers;
    }
    if (atom_numbers > 0)
    {
        controller->printf("    atom_numbers is %d\n", atom_numbers);
        controller->printf("    atom_LJ_type_number is %d\n",
                           atom_type_numbers);
        pair_type_numbers = atom_type_numbers * (atom_type_numbers + 1) / 2;
        LJ_Malloc();

        for (int i = 0; i < pair_type_numbers; i++)
        {
            h_LJ_A[i] = lj_to_use->pair_A[i];
            h_LJ_B[i] = lj_to_use->pair_B[i];
        }
        for (int i = 0; i < atom_numbers; i++)
        {
            h_atom_LJ_type[i] = lj_to_use->atom_type[i];
        }
        Parameter_Host_To_Device();
        is_initialized = 1;
    }
    if (is_initialized)
    {
        this->cutoff = cutoff;
        use_ordered_layout = false;
        ordered_layout_applied = 0;
        ordered_layout_max_depth = 6;
        ordered_layout_leaf_size = 32;
        ordered_layout_min_residue_numbers = 256;
        if (controller->Command_Exist(this->module_name, "ordered_layout"))
        {
            use_ordered_layout = controller->Get_Bool(
                this->module_name, "ordered_layout",
                "LENNARD_JONES_INFORMATION::Initial");
        }
        if (controller->Command_Exist(this->module_name,
                                      "ordered_layout_max_depth"))
        {
            controller->Check_Int(this->module_name,
                                  "ordered_layout_max_depth",
                                  "LENNARD_JONES_INFORMATION::Initial");
            ordered_layout_max_depth = atoi(controller->Command(
                this->module_name, "ordered_layout_max_depth"));
        }
        if (controller->Command_Exist(this->module_name,
                                      "ordered_layout_leaf_size"))
        {
            controller->Check_Int(this->module_name,
                                  "ordered_layout_leaf_size",
                                  "LENNARD_JONES_INFORMATION::Initial");
            ordered_layout_leaf_size = atoi(controller->Command(
                this->module_name, "ordered_layout_leaf_size"));
        }
        if (controller->Command_Exist(this->module_name,
                                      "ordered_layout_min_residue_numbers"))
        {
            controller->Check_Int(this->module_name,
                                  "ordered_layout_min_residue_numbers",
                                  "LENNARD_JONES_INFORMATION::Initial");
            ordered_layout_min_residue_numbers = atoi(controller->Command(
                this->module_name,
                "ordered_layout_min_residue_numbers"));
        }
        ordered_layout_max_depth = std::max(1, std::min(ordered_layout_max_depth,
                                                        21));
        ordered_layout_leaf_size = std::max(1, ordered_layout_leaf_size);
        ordered_layout_min_residue_numbers =
            std::max(1, ordered_layout_min_residue_numbers);
        controller->printf("    ordered_layout: %s\n",
                           use_ordered_layout ? "true" : "false");
        if (use_ordered_layout)
        {
            controller->printf(
                "        cornerstone octree depth=%d leaf_size=%d "
                "min_residues=%d\n",
                ordered_layout_max_depth, ordered_layout_leaf_size,
                ordered_layout_min_residue_numbers);
        }
        clustered_direct_cache = Acquire_Shared_LJ_Clustered_Direct_Cache(
            controller, this->module_name, use_ordered_layout);
        Device_Malloc_Safely((void**)&crd_with_LJ_parameters,
                             sizeof(VECTOR_LJ) * atom_numbers);
        Launch_Device_Kernel(
            Copy_LJ_Type_To_New_Crd,
            (this->atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, atom_numbers,
            crd_with_LJ_parameters, d_atom_LJ_type);
        controller->printf("    Start initializing long range LJ correction\n");
        long_range_factor = 0;

        Device_Malloc_And_Copy_Safely((void**)&d_long_range_factor,
                                      &long_range_factor, sizeof(float));
        deviceMemset(d_long_range_factor, 0, sizeof(float));

        dim3 gridSize = {(atom_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         1};
        dim3 blockSize = {
            CONTROLLER::device_warp,
            CONTROLLER::device_max_thread / CONTROLLER::device_warp};
        Launch_Device_Kernel(Total_C6_Get, gridSize, blockSize, 0, NULL,
                             atom_numbers, d_atom_LJ_type, d_LJ_B,
                             d_long_range_factor);

        deviceMemcpy(&long_range_factor, d_long_range_factor, sizeof(float),
                     deviceMemcpyDeviceToHost);
        printf("        Total C6 factor is %e\n", long_range_factor);

        long_range_factor *=
            -2.0f / 3.0f * CONSTANT_Pi / cutoff / cutoff / cutoff / 6.0f;
        controller->printf("        long range correction factor is: %e\n",
                           long_range_factor);
        controller->printf("    End initializing long range LJ correction\n");
    }
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller->Step_Print_Initial("LJ_short", "%.2f");
        controller->Step_Print_Initial("LJ_long", "%.2f");
        controller->Step_Print_Initial("LJ", "%.2f");
        is_controller_printf_initialized = 1;
        controller->printf("    structure last modify date is %d\n",
                           last_modify_date);
    }
    controller->printf("END INITIALIZING LENNADR JONES INFORMATION\n\n");
}

static __global__ void get_local_device(int* atom_local, int local_atom_numbers,
                                        int ghost_numbers, int* d_atom_LJ_type,
                                        VECTOR_LJ* crd_with_LJ_parameters_local)
{
    SIMPLE_DEVICE_FOR(i, local_atom_numbers + ghost_numbers)
    {
        int atom_i = atom_local[i];
        crd_with_LJ_parameters_local[i].LJ_type = d_atom_LJ_type[atom_i];
    }
}

void LENNARD_JONES_INFORMATION::Get_Local(int* atom_local,
                                          int local_atom_numbers,
                                          int ghost_numbers)
{
    if (!is_initialized) return;
    this->local_atom_numbers = local_atom_numbers;
    this->ghost_numbers = ghost_numbers;
    Launch_Device_Kernel(get_local_device,
                         (local_atom_numbers + ghost_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, atom_local,
                         local_atom_numbers, ghost_numbers, d_atom_LJ_type,
                         crd_with_LJ_parameters_local);
}

void LENNARD_JONES_INFORMATION::Refresh_Clustered_Metadata(
    int solvent_numbers, const int* d_excluded_list_start,
    const int* d_excluded_list,
    const int* d_excluded_numbers)
{
    if (!is_initialized) return;
    if (clustered_direct_cache != NULL)
    {
        const int capped_solvent_numbers =
            solvent_numbers > 0 ? solvent_numbers : 0;
        const int direct_local_atom_numbers =
            local_atom_numbers > capped_solvent_numbers
                ? (local_atom_numbers - capped_solvent_numbers)
                : 0;
        clustered_direct_cache->Refresh_Metadata(
            local_atom_numbers, direct_local_atom_numbers, ghost_numbers,
            d_excluded_list_start,
            d_excluded_list, d_excluded_numbers);
    }
}

static __global__ void Long_Range_Virial_Correction(LTMatrix3* d_virial,
                                                    const float factor)
{
    d_virial[0].a11 += factor;
    d_virial[0].a22 += factor;
    d_virial[0].a33 += factor;
}

void LENNARD_JONES_INFORMATION::Long_Range_Correction(int need_pressure,
                                                      LTMatrix3* d_virial,
                                                      int need_potential,
                                                      float* d_potential,
                                                      const float volume)
{
    if (is_initialized && CONTROLLER::PP_MPI_rank == 0)
    {
        if (need_pressure)
        {
            Launch_Device_Kernel(Long_Range_Virial_Correction, 1, 1, 0, 0,
                                 d_virial, 2 * long_range_factor / volume);
        }
        if (need_potential)
        {
            Launch_Device_Kernel(device_add, 1, 1, 0, 0, d_potential,
                                 long_range_factor / volume);

            h_LJ_long_energy = long_range_factor / volume;
        }
    }
}

void LENNARD_JONES_INFORMATION::Parameter_Host_To_Device()
{
    Device_Malloc_And_Copy_Safely((void**)&d_atom_LJ_type, h_atom_LJ_type,
                                  sizeof(int) * atom_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_A, h_LJ_A,
                                  sizeof(float) * pair_type_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_B, h_LJ_B,
                                  sizeof(float) * pair_type_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_energy_sum, h_LJ_energy_atom,
                                  sizeof(float));
    Device_Malloc_Safely((void**)&d_LJ_energy_atom,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&crd_with_LJ_parameters_local,
                         sizeof(VECTOR_LJ) * atom_numbers);
}

void LENNARD_JONES_INFORMATION::LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(
    const int atom_numbers, const int local_atom_numbers,
    const int solvent_numbers, const int ghost_numbers, const VECTOR* crd,
    const float* charge, VECTOR* frc, const LTMatrix3 cell,
    const LTMatrix3 rcell, const ATOM_GROUP* nl, const float pme_beta,
    const int need_atom_energy, float* atom_energy, const int need_virial,
    LTMatrix3* atom_virial, float* atom_direct_pme_energy)
{
    if (is_initialized)
    {
        if (Use_Clustered_Direct())
        {
            clustered_direct_cache->Build(crd, cell, rcell, cutoff);
        }
        Launch_Device_Kernel(
            Copy_Crd_And_Charge_To_New_Crd,
            (this->atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL,
            this->local_atom_numbers + this->ghost_numbers, crd,
            crd_with_LJ_parameters_local, charge);
        if (Use_Clustered_Direct() &&
            clustered_direct_cache->layout.total_atom_numbers > 0)
        {
            clustered_direct_cache->Gather_Plain(crd_with_LJ_parameters_local,
                                                 cell, rcell);
        }
        if (need_atom_energy)
        {
            deviceMemset(atom_direct_pme_energy, 0,
                         sizeof(float) * this->atom_numbers);
            deviceMemset(d_LJ_energy_atom, 0,
                         sizeof(float) * this->atom_numbers);
        }

        if (atom_numbers == 0 || local_atom_numbers == 0) return;

        if (Use_Clustered_Direct())
        {
            auto& clustered_layout = clustered_direct_cache->layout;
            if (clustered_layout.cjpacked_numbers == 0 ||
                clustered_layout.sci_numbers == 0)
                return;
            dim3 blockSize = {
                static_cast<unsigned int>(clustered_layout.cluster_size),
                static_cast<unsigned int>(clustered_layout.cluster_size), 1u};
            dim3 gridSize = {
                static_cast<unsigned int>(clustered_layout.sci_numbers), 1u, 1u};
            auto f = Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Device<
                true, false, false, true>;
            if (!need_atom_energy && !need_virial)
            {
                f = Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Device<
                    true, false, false, true>;
            }
            else if (need_atom_energy && !need_virial)
            {
                f = Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Device<
                    true, true, false, true>;
            }
            else if (!need_atom_energy && need_virial)
            {
                f = Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Device<
                    true, false, true, true>;
            }
            else
            {
                f = Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Device<
                    true, true, true, true>;
            }
            if (clustered_direct_cache->direct_kernel_time_recorder != NULL)
            {
                clustered_direct_cache->direct_kernel_time_recorder->Start();
            }
            Launch_Device_Kernel(
                f, gridSize, blockSize, 0, NULL, clustered_layout.sci_numbers,
                clustered_layout.cluster_size,
                clustered_layout.super_cluster_clusters, local_atom_numbers,
                clustered_layout.d_cluster_offsets,
                clustered_layout.d_cluster_valid_masks,
                clustered_layout.d_cluster_local_masks,
                clustered_layout.d_cluster_centers,
                clustered_layout.d_super_cluster_offsets,
                clustered_layout.d_nbnxm_sci,
                clustered_layout.d_nbnxm_cjpacked,
                clustered_layout.d_exclusion_mask_pool,
                clustered_direct_cache->d_sorted_atom_ids,
                clustered_direct_cache->d_sorted_xq,
                clustered_direct_cache->d_sorted_lj_type, cell, rcell, d_LJ_A,
                d_LJ_B, cutoff, frc, pme_beta, atom_energy, atom_virial,
                atom_direct_pme_energy, d_LJ_energy_atom);
            if (clustered_direct_cache->direct_kernel_time_recorder != NULL)
            {
                clustered_direct_cache->direct_kernel_time_recorder->Stop();
            }
        }
        else
        {
            dim3 blockSize = {
                CONTROLLER::device_warp,
                CONTROLLER::device_max_thread / CONTROLLER::device_warp};
            dim3 gridSize = (atom_numbers + blockSize.y - 1) / blockSize.y;
            auto f =
                Lennard_Jones_And_Direct_Coulomb_Device<true, false, false,
                                                        true>;
            if (!need_atom_energy && !need_virial)
            {
                f = Lennard_Jones_And_Direct_Coulomb_Device<
                    true, false, false, true>;
            }
            else if (need_atom_energy && !need_virial)
            {
                f = Lennard_Jones_And_Direct_Coulomb_Device<
                    true, true, false, true>;
            }
            else if (!need_atom_energy && need_virial)
            {
                f = Lennard_Jones_And_Direct_Coulomb_Device<
                    true, false, true, true>;
            }
            else
            {
                f = Lennard_Jones_And_Direct_Coulomb_Device<
                    true, true, true, true>;
            }
            Launch_Device_Kernel(
                f, gridSize, blockSize, 0, NULL, local_atom_numbers,
                solvent_numbers, nl, crd_with_LJ_parameters_local, cell, rcell,
                d_LJ_A, d_LJ_B, cutoff, frc, pme_beta, atom_energy,
                atom_virial, atom_direct_pme_energy, d_LJ_energy_atom);
        }
    }
}

void LENNARD_JONES_INFORMATION::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized || CONTROLLER::MPI_rank >= CONTROLLER::PP_MPI_size)
        return;
    Sum_Of_List(d_LJ_energy_atom, d_LJ_energy_sum, atom_numbers);
    deviceMemcpy(&h_LJ_energy_sum, d_LJ_energy_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &h_LJ_energy_sum, 1, MPI_FLOAT, MPI_SUM,
                  CONTROLLER::pp_comm);
#endif
    controller->Step_Print("LJ_short", h_LJ_energy_sum);
    controller->Step_Print("LJ_long", h_LJ_long_energy);
    controller->Step_Print("LJ", h_LJ_energy_sum + h_LJ_long_energy, true);
}
