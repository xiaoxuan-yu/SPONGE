#include "Lennard_Jones_force.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include "../xponge/load/native/lj.hpp"
#include "../xponge/xponge.h"
// #include "assert.h"

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

namespace
{

static __host__ __device__ __forceinline__ int
Clustered_Gmxpacked_Get_LJ_Type_MinMax(const int a, const int b)
{
    const int hi = a > b ? a : b;
    const int lo = a > b ? b : a;
    return (hi * (hi + 1) >> 1) + lo;
}

static bool Clustered_Gmxpacked_Lj_Comb_Table_Compatible(
    const float* lj_a, const float* lj_b, int atom_type_numbers)
{
    if (lj_a == NULL || lj_b == NULL || atom_type_numbers <= 0)
    {
        return false;
    }
    constexpr float rel_tol = 1.0e-4f;
    constexpr float abs_tol = 1.0e-4f;
    for (int i = 0; i < atom_type_numbers; i += 1)
    {
        const int self_i = Get_LJ_Type(i, i);
        if (lj_a[self_i] < 0.0f || lj_b[self_i] < 0.0f)
        {
            return false;
        }
        for (int j = 0; j < atom_type_numbers; j += 1)
        {
            const int self_j = Get_LJ_Type(j, j);
            const int pair = Get_LJ_Type(i, j);
            if (lj_a[self_j] < 0.0f || lj_b[self_j] < 0.0f ||
                lj_a[pair] < 0.0f || lj_b[pair] < 0.0f)
            {
                return false;
            }
            const float expected_a =
                sqrtf(fmaxf(lj_a[self_i], 0.0f)) *
                sqrtf(fmaxf(lj_a[self_j], 0.0f));
            const float expected_b =
                sqrtf(fmaxf(lj_b[self_i], 0.0f)) *
                sqrtf(fmaxf(lj_b[self_j], 0.0f));
            const float scale_a = fmaxf(1.0f, fabsf(lj_a[pair]));
            const float scale_b = fmaxf(1.0f, fabsf(lj_b[pair]));
            if (fabsf(expected_a - lj_a[pair]) >
                    abs_tol + rel_tol * scale_a ||
                fabsf(expected_b - lj_b[pair]) >
                    abs_tol + rel_tol * scale_b)
            {
                return false;
            }
        }
    }
    return true;
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

static __device__ __forceinline__
    void Reduce_Clustered_Subgroup_Vector_Components(float& x, float& y,
                                                     float& z, int lane,
                                                     int subgroup_width)
{
    const unsigned int subgroup_mask =
        Clustered_Subgroup_Mask(lane, subgroup_width);
    for (int delta = subgroup_width >> 1; delta > 0; delta >>= 1)
    {
        x += deviceShflDown(subgroup_mask, x, delta, subgroup_width);
        y += deviceShflDown(subgroup_mask, y, delta, subgroup_width);
        z += deviceShflDown(subgroup_mask, z, delta, subgroup_width);
    }
}

static __device__ __forceinline__
    void Reduce_Clustered_Warp_Vector_Over_J_Components(float& x, float& y,
                                                        float& z,
                                                        int subgroup_width)
{
    for (int delta = warpSize >> 1; delta >= subgroup_width; delta >>= 1)
    {
        x += deviceShflDown(FULL_MASK, x, delta, warpSize);
        y += deviceShflDown(FULL_MASK, y, delta, warpSize);
        z += deviceShflDown(FULL_MASK, z, delta, warpSize);
    }
}

template <bool enabled, int size>
struct Clustered_Energy_Buffer
{
    float unused;

    __device__ __forceinline__ float& operator[](int)
    {
        return unused;
    }

    __device__ __forceinline__ const float& operator[](int) const
    {
        return unused;
    }
};

template <int size>
struct Clustered_Energy_Buffer<true, size>
{
    float values[size];

    __device__ __forceinline__ void Clear()
    {
        for (int i = 0; i < size; i += 1)
        {
            values[i] = 0.0f;
        }
    }

    __device__ __forceinline__ float& operator[](int idx)
    {
        return values[idx];
    }

    __device__ __forceinline__ const float& operator[](int idx) const
    {
        return values[idx];
    }
};

template <bool total_output, bool need_energy, int size>
struct Clustered_Full_Record_Output_Buffer;

template <bool need_energy, int size>
struct Clustered_Full_Record_Output_Buffer<false, need_energy, size>
{
    Clustered_Energy_Buffer<need_energy, size> energy_lj;
    Clustered_Energy_Buffer<need_energy, size> energy_coulomb;
    LTMatrix3 virial[size];

    __device__ __forceinline__ Clustered_Full_Record_Output_Buffer(
        const bool store_energy, const bool store_virial)
    {
        if constexpr (need_energy)
        {
            if (store_energy)
            {
                energy_lj.Clear();
                energy_coulomb.Clear();
            }
        }
        if (store_virial)
        {
            for (int i = 0; i < size; i += 1)
            {
                virial[i] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            }
        }
    }
};

template <bool need_energy, int size>
struct Clustered_Full_Record_Output_Buffer<true, need_energy, size>
{
    float energy_lj_total;
    float energy_coulomb_total;
    LTMatrix3 virial_total;

    __device__ __forceinline__ Clustered_Full_Record_Output_Buffer(
        const bool store_energy, const bool store_virial)
    {
        if constexpr (need_energy)
        {
            if (store_energy)
            {
                energy_lj_total = 0.0f;
                energy_coulomb_total = 0.0f;
            }
        }
        if (store_virial)
        {
            virial_total = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
};

template <typename T>
static __device__ __forceinline__ T Clustered_Load_ReadOnly(const T* ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

static __device__ __forceinline__ float
Reduce_Clustered_Subgroup_Vector_To_Component(float x, float y, float z,
                                               int component_lane, int lane,
                                               int subgroup_width)
{
    const unsigned int subgroup_mask =
        Clustered_Subgroup_Mask(lane, subgroup_width);
    const int subgroup_leader = lane - lane % subgroup_width;
    Reduce_Clustered_Subgroup_Vector_Components(x, y, z, lane,
                                                subgroup_width);
    x = deviceShfl(subgroup_mask, x, subgroup_leader, warpSize);
    y = deviceShfl(subgroup_mask, y, subgroup_leader, warpSize);
    z = deviceShfl(subgroup_mask, z, subgroup_leader, warpSize);
    if (component_lane == 0)
    {
        return x;
    }
    if (component_lane == 1)
    {
        return y;
    }
    return z;
}

static __device__ __forceinline__ float
Reduce_Clustered_Warp_I_To_Component(float x, float y, float z, int i_lane,
                                      int component_lane, int subgroup_width)
{
    Reduce_Clustered_Warp_Vector_Over_J_Components(x, y, z, subgroup_width);
    x = deviceShfl(FULL_MASK, x, i_lane, warpSize);
    y = deviceShfl(FULL_MASK, y, i_lane, warpSize);
    z = deviceShfl(FULL_MASK, z, i_lane, warpSize);
    if (component_lane == 0)
    {
        return x;
    }
    if (component_lane == 1)
    {
        return y;
    }
    return z;
}

static __device__ __forceinline__ void Clustered_Atomic_Add_Force_Component(
    VECTOR* frc, int atom_index, int component, float value)
{
    float* frc_component = reinterpret_cast<float*>(frc + atom_index);
    atomicAdd(frc_component + component, value);
}

static __global__ void Scatter_Sorted_Clustered_Force(
    const int total_atom_numbers, const int* sorted_atom_ids,
    const VECTOR* sorted_frc, VECTOR* frc)
{
    SIMPLE_DEVICE_FOR(sorted_i, total_atom_numbers)
    {
        const int atom_i = sorted_atom_ids[sorted_i];
        frc[atom_i] = frc[atom_i] + sorted_frc[sorted_i];
    }
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

}  // namespace

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
Get_Clustered_Shifted_Displacement(const VECTOR_LJ r2, const VECTOR_LJ r1,
                                   const VECTOR shift_vec)
{
    return (r2.crd - r1.crd) - shift_vec;
}

static __device__ __forceinline__ float
Get_Clustered_LJ_Force_Abs(const float inv_r2, const float inv_r6,
                           const float A, const float B)
{
    return (B - A * inv_r6) * inv_r6 * inv_r2;
}

static __device__ __forceinline__ float
Get_Clustered_LJ_Energy(const float inv_r6, const float A, const float B)
{
    return (0.083333333f * A * inv_r6 - 0.166666667f * B) * inv_r6;
}

static __device__ __forceinline__ float Get_Clustered_Direct_Coulomb_Energy(
    const float charge_product, const float inv_r, const float beta_dr)
{
    return charge_product * erfcf(beta_dr) * inv_r;
}

static __host__ __device__ __forceinline__ float
Clustered_PME_Corr_F(const float z2)
{
    constexpr float FN6 = -1.7357322914161492954e-8F;
    constexpr float FN5 = 1.4703624142580877519e-6F;
    constexpr float FN4 = -0.000053401640219807709149F;
    constexpr float FN3 = 0.0010054721316683106153F;
    constexpr float FN2 = -0.019278317264888380590F;
    constexpr float FN1 = 0.069670166153766424023F;
    constexpr float FN0 = -0.75225204789749321333F;

    constexpr float FD4 = 0.0011193462567257629232F;
    constexpr float FD3 = 0.014866955030185295499F;
    constexpr float FD2 = 0.11583842382862377919F;
    constexpr float FD1 = 0.50736591960530292870F;
    constexpr float FD0 = 1.0F;

    const float z4 = z2 * z2;

    float polyFD0 = FD4 * z4 + FD2;
    const float polyFD1 = FD3 * z4 + FD1;
    polyFD0 = polyFD0 * z4 + FD0;
    polyFD0 = polyFD1 * z2 + polyFD0;
    polyFD0 = 1.0F / polyFD0;

    float polyFN0 = FN6 * z4 + FN4;
    float polyFN1 = FN5 * z4 + FN3;
    polyFN0 = polyFN0 * z4 + FN2;
    polyFN1 = polyFN1 * z4 + FN1;
    polyFN0 = polyFN0 * z4 + FN0;
    polyFN0 = polyFN1 * z2 + polyFN0;
    return polyFN0 * polyFD0;
}

static __device__ __forceinline__ float
Get_Clustered_Direct_Coulomb_Force_Abs_PME_Corr(
    const float charge_product, const float inv_r, const float inv_r2,
    const float beta2_r2, const float beta3)
{
    return charge_product *
           (inv_r * inv_r2 + Clustered_PME_Corr_F(beta2_r2) * beta3);
}

#ifdef USE_CPU
template <bool full_output>
static void Cpu_Gmxpacked_Clustered_Lennard_Jones_And_Direct_Coulomb(
    const int sci_numbers, const int cluster_size,
    const int local_atom_numbers, const int* cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const int* super_cluster_offsets,
    const LJ_CLUSTERED_GMXPACKED_SCI* sci_entries,
    const LJ_CLUSTERED_GMXPACKED_CJ* cj_packed_entries,
    const LJ_CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    const int* sorted_atom_ids, const float4* sorted_xq,
    const int* sorted_lj_type, const LTMatrix3 cell, const float* LJ_type_A,
    const float* LJ_type_B, const float cutoff, VECTOR* frc,
    const float pme_beta, float* atom_energy, LTMatrix3* atom_virial,
    float* atom_direct_cf_energy, float* atom_LJ_ene,
    const bool store_energy, const bool store_virial)
{
    constexpr int max_cluster_size = kClusteredClusterSize;
    const float cutoff_sq = cutoff * cutoff;

#pragma omp parallel for schedule(dynamic)
    for (int sci = 0; sci < sci_numbers; sci += 1)
    {
        const LJ_CLUSTERED_GMXPACKED_SCI sci_entry = sci_entries[sci];
        const int super_i = sci_entry.supercluster_id;
        const int cluster_i_start = super_cluster_offsets[super_i];
        const int cluster_i_end = super_cluster_offsets[super_i + 1];
        const VECTOR pair_shift_vec =
            Clustered_Shift_Vector_From_Id(sci_entry.shift_id, cell);

        for (int cluster_i = cluster_i_start; cluster_i < cluster_i_end;
             cluster_i += 1)
        {
            const unsigned int valid_mask_i = cluster_valid_masks[cluster_i];
            const unsigned int local_mask_i = cluster_local_masks[cluster_i];
            const int i_local = cluster_i - cluster_i_start;
            for (int lane_i = 0; lane_i < cluster_size; lane_i += 1)
            {
                if (!Clustered_Lane_Is_Valid(valid_mask_i, lane_i) ||
                    !Clustered_Lane_Is_Local(local_mask_i, lane_i))
                {
                    continue;
                }
                const int sorted_atom_i = cluster_offsets[cluster_i] + lane_i;
                const int atom_i = sorted_atom_ids[sorted_atom_i];
                const VECTOR_LJ r1 = Make_Packed_LJ_Atom(
                    sorted_xq[sorted_atom_i], sorted_lj_type[sorted_atom_i]);
                VECTOR frc_i = {0.0f, 0.0f, 0.0f};
                float energy_lj = 0.0f;
                float energy_coulomb = 0.0f;
                LTMatrix3 virial = {0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f};

                for (int packed_idx = sci_entry.cjpacked_begin;
                     packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
                {
                    const LJ_CLUSTERED_GMXPACKED_CJ& packed =
                        cj_packed_entries[packed_idx];
                    for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
                    {
                        const int cluster_j = packed.cj[jm];
                        if (cluster_j < 0)
                        {
                            continue;
                        }
                        VECTOR frc_j[max_cluster_size] = {};
                        unsigned int active_j_mask = 0u;

                        for (int split = 0;
                             split < kClusteredWarpSplitCount; split += 1)
                        {
                            const unsigned int packed_imask =
                                packed.split[split].imask;
                            const unsigned int record_imask =
                                (packed_imask >>
                                 Clustered_Jm_Imask_Shift(jm)) &
                                ((1u << kClusteredSuperClusterClusters) - 1u);
                            if ((record_imask &
                                 (1u << static_cast<unsigned int>(i_local))) ==
                                0u)
                            {
                                continue;
                            }
                            const int exclusion_index =
                                packed.split[split].exclusion_index;
                            for (int split_j_lane = 0;
                                 split_j_lane < kClusteredSplitJClusterSize;
                                 split_j_lane += 1)
                            {
                                unsigned int pair_bits = 0xffffffffu;
                                if (exclusion_index != 0)
                                {
                                    pair_bits =
                                        exclusion_entries[exclusion_index]
                                            .pair[split_j_lane * cluster_size +
                                                  lane_i];
                                }
                                if (((packed_imask & pair_bits) &
                                     (1u << static_cast<unsigned int>(
                                          Clustered_Jm_Imask_Shift(jm) +
                                          i_local))) == 0u)
                                {
                                    continue;
                                }
                                const int lane_j =
                                    split * kClusteredSplitJClusterSize +
                                    split_j_lane;
                                const int sorted_atom_j =
                                    cluster_offsets[cluster_j] + lane_j;
                                const int atom_j =
                                    sorted_atom_ids[sorted_atom_j];
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
                                float frc_abs =
                                    Get_LJ_Force(r1, r2, dr_abs, A, B);
                                frc_abs -= Get_Direct_Coulomb_Force(
                                    r1, r2, dr_abs, pme_beta);
                                const VECTOR frc_lin = frc_abs * dr;
                                frc_i = frc_i + frc_lin;
                                if (atom_j < local_atom_numbers)
                                {
                                    frc_j[lane_j] =
                                        frc_j[lane_j] - frc_lin;
                                    active_j_mask |=
                                        1u << static_cast<unsigned int>(lane_j);
                                }

                                if constexpr (full_output)
                                {
                                    virial =
                                        virial -
                                        ij_factor *
                                            Get_Virial_From_Force_Dis(frc_lin,
                                                                      dr);
                                    energy_lj +=
                                        ij_factor *
                                        Get_LJ_Energy(r1, r2, dr_abs, A, B);
                                    energy_coulomb +=
                                        ij_factor *
                                        Get_Direct_Coulomb_Energy(
                                            r1, r2, dr_abs, pme_beta);
                                }
                            }
                        }

                        for (int lane_j = 0; lane_j < cluster_size;
                             lane_j += 1)
                        {
                            if ((active_j_mask &
                                 (1u << static_cast<unsigned int>(lane_j))) ==
                                0u)
                            {
                                continue;
                            }
                            const int sorted_atom_j =
                                cluster_offsets[cluster_j] + lane_j;
                            const int atom_j = sorted_atom_ids[sorted_atom_j];
                            atomicAdd(frc + atom_j, frc_j[lane_j]);
                        }
                    }
                }

                atomicAdd(frc + atom_i, frc_i);
                if constexpr (full_output)
                {
                    if (store_energy)
                    {
                        atomicAdd(atom_energy + atom_i,
                                  energy_lj + energy_coulomb);
                        atomicAdd(atom_LJ_ene + atom_i, energy_lj);
                        atomicAdd(atom_direct_cf_energy + atom_i,
                                  energy_coulomb);
                    }
                    if (store_virial)
                    {
                        atomicAdd(atom_virial + atom_i, virial);
                    }
                }
            }
        }
    }
}
#endif

#ifndef USE_CPU
#include "clustered_lj_warp_record_kernel.cuh"

struct ClusteredRegularLJKernelInput
{
    LTMatrix3 cell;
    const float2* lj_ab_table;
    float cutoff;
    VECTOR* force;
    float pme_beta;
    float* atom_energy;
    LTMatrix3* atom_virial;
    float* atom_direct_pme_energy;
    float* atom_lj_energy;
    bool store_energy;
    bool store_virial;
};

template <typename Kernel>
static void Launch_Clustered_Regular_LJ_Kernel(
    Kernel kernel, dim3 grid_size, dim3 block_size,
    const CLUSTERED_SPATIAL_VIEW& view,
    const LJClusteredWorkspace& workspace, const float2* sorted_lj_comb,
    const uint64_t* pair_shift_bits, const int* sci_shift_flags,
    int sci_shift_only,
    const ClusteredRegularLJKernelInput& input)
{
    Launch_Device_Kernel(
        kernel, grid_size, block_size, 0, NULL, view.gmxpacked_sci_numbers,
        view.cluster_size, view.super_cluster_clusters, view.cluster_numbers,
        view.cluster_offsets, view.cluster_valid_masks, view.cluster_local_masks,
        view.super_cluster_offsets, view.gmxpacked_sci,
        view.gmxpacked_cjpacked, view.gmxpacked_exclusions, pair_shift_bits,
        sci_shift_flags, sci_shift_only, workspace.d_sorted_atom_ids,
        workspace.d_sorted_xq, workspace.d_sorted_lj_type,
        sorted_lj_comb, input.cell, input.lj_ab_table,
        input.cutoff, input.force, input.pme_beta, input.atom_energy,
        input.atom_virial, input.atom_direct_pme_energy, input.atom_lj_energy,
        input.store_energy, input.store_virial);
}

static void Launch_Clustered_Regular_LJ_Full_Dense(
    const CLUSTERED_SPATIAL_VIEW& view,
    const LJClusteredWorkspace& workspace, bool use_lj_comb,
    dim3 block_size, const ClusteredRegularLJKernelInput& input)
{
    const int* sci_shift_flags = view.gmxpacked_sci_shift_safe_flags;
    constexpr unsigned int kAbForceOnlySciWorkParts = 4u;
    constexpr unsigned int kAbFullOutputSciWorkParts = 4u;
    const bool full_output = input.store_energy || input.store_virial;
    const bool use_ab_force_only_partition = !use_lj_comb && !full_output;
    const bool use_ab_full_output_partition = !use_lj_comb && full_output;
    const unsigned int sci_work_parts =
        use_ab_force_only_partition
            ? kAbForceOnlySciWorkParts
            : (use_ab_full_output_partition ? kAbFullOutputSciWorkParts : 1u);
    const dim3 grid_size = {
        static_cast<unsigned int>(view.gmxpacked_sci_numbers) * sci_work_parts,
        1u, 1u};

#define CLUSTERED_GMXPACKED_SPLIT_FULL_DENSE_KERNEL(OUTPUT_MODE, SHIFT_MODE)                                  \
    (use_lj_comb                                                                                                         \
         ? Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<                               \
               OUTPUT_MODE, ClusteredRegularLJParameterMode::COMBINATION,                                                \
               ClusteredRegularLJLayoutMode::FULL_LOCAL_DENSE, SHIFT_MODE>                                               \
         : Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<                               \
               OUTPUT_MODE, ClusteredRegularLJParameterMode::AB_TABLE,                                                   \
               ClusteredRegularLJLayoutMode::FULL_LOCAL_DENSE, SHIFT_MODE>)
    auto fast_kernel = CLUSTERED_GMXPACKED_SPLIT_FULL_DENSE_KERNEL(
        ClusteredRegularLJOutputMode::FORCE_ONLY,
        ClusteredRegularLJShiftMode::SCI_SHIFT_ONLY);
    auto slow_kernel = CLUSTERED_GMXPACKED_SPLIT_FULL_DENSE_KERNEL(
        ClusteredRegularLJOutputMode::FORCE_ONLY,
        ClusteredRegularLJShiftMode::PAIR_SHIFT);
    if (use_ab_force_only_partition)
    {
        fast_kernel =
            Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                ClusteredRegularLJOutputMode::FORCE_ONLY,
                ClusteredRegularLJParameterMode::AB_TABLE,
                ClusteredRegularLJLayoutMode::FULL_LOCAL_DENSE,
                ClusteredRegularLJShiftMode::SCI_SHIFT_ONLY, VECTOR, 4, true>;
        slow_kernel =
            Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                ClusteredRegularLJOutputMode::FORCE_ONLY,
                ClusteredRegularLJParameterMode::AB_TABLE,
                ClusteredRegularLJLayoutMode::FULL_LOCAL_DENSE,
                ClusteredRegularLJShiftMode::PAIR_SHIFT, VECTOR, 4, true>;
    }
    else if (use_ab_full_output_partition)
    {
        fast_kernel =
            Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                ClusteredRegularLJOutputMode::FULL,
                ClusteredRegularLJParameterMode::AB_TABLE,
                ClusteredRegularLJLayoutMode::FULL_LOCAL_DENSE,
                ClusteredRegularLJShiftMode::SCI_SHIFT_ONLY, VECTOR, 4,
                false>;
        slow_kernel =
            Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                ClusteredRegularLJOutputMode::FULL,
                ClusteredRegularLJParameterMode::AB_TABLE,
                ClusteredRegularLJLayoutMode::FULL_LOCAL_DENSE,
                ClusteredRegularLJShiftMode::PAIR_SHIFT, VECTOR, 4, false>;
    }
    else if (full_output)
    {
        fast_kernel = CLUSTERED_GMXPACKED_SPLIT_FULL_DENSE_KERNEL(
            ClusteredRegularLJOutputMode::FULL,
            ClusteredRegularLJShiftMode::SCI_SHIFT_ONLY);
        slow_kernel = CLUSTERED_GMXPACKED_SPLIT_FULL_DENSE_KERNEL(
            ClusteredRegularLJOutputMode::FULL,
            ClusteredRegularLJShiftMode::PAIR_SHIFT);
    }
#undef CLUSTERED_GMXPACKED_SPLIT_FULL_DENSE_KERNEL

    Launch_Clustered_Regular_LJ_Kernel(
        fast_kernel, grid_size, block_size, view, workspace,
        workspace.d_sorted_lj_comb, NULL, sci_shift_flags, 1, input);
    Launch_Clustered_Regular_LJ_Kernel(
        slow_kernel, grid_size, block_size, view, workspace,
        workspace.d_sorted_lj_comb, view.pair_shift_bits, sci_shift_flags, 0,
        input);
}

static void Launch_Clustered_Regular_LJ_Dense_Offset(
    const CLUSTERED_SPATIAL_VIEW& view,
    const LJClusteredWorkspace& workspace, bool use_lj_comb,
    dim3 block_size, const ClusteredRegularLJKernelInput& input)
{
    constexpr unsigned int kAbSciWorkParts = 4u;
    const bool full_output = input.store_energy || input.store_virial;
    const dim3 grid_size = {
        static_cast<unsigned int>(view.gmxpacked_sci_numbers) *
            (use_lj_comb ? 1u : kAbSciWorkParts),
        1u, 1u};
    const int* sci_shift_flags = view.gmxpacked_sci_shift_safe_flags;

#define CLUSTERED_GMXPACKED_SPLIT_DENSE_OFFSET_KERNEL(OUTPUT_MODE, SHIFT_MODE)                                              \
    (use_lj_comb                                                                                                            \
         ? Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<                                  \
               OUTPUT_MODE, ClusteredRegularLJParameterMode::COMBINATION,                                                   \
               ClusteredRegularLJLayoutMode::DENSE_OFFSET, SHIFT_MODE>                                                      \
         : Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<                                  \
               OUTPUT_MODE, ClusteredRegularLJParameterMode::AB_TABLE,                                                      \
               ClusteredRegularLJLayoutMode::DENSE_OFFSET, SHIFT_MODE, VECTOR, 4>)
    auto fast_kernel = CLUSTERED_GMXPACKED_SPLIT_DENSE_OFFSET_KERNEL(
        ClusteredRegularLJOutputMode::FORCE_ONLY,
        ClusteredRegularLJShiftMode::SCI_SHIFT_ONLY);
    auto slow_kernel = CLUSTERED_GMXPACKED_SPLIT_DENSE_OFFSET_KERNEL(
        ClusteredRegularLJOutputMode::FORCE_ONLY,
        ClusteredRegularLJShiftMode::PAIR_SHIFT);
    if (full_output)
    {
        fast_kernel = CLUSTERED_GMXPACKED_SPLIT_DENSE_OFFSET_KERNEL(
            ClusteredRegularLJOutputMode::FULL,
            ClusteredRegularLJShiftMode::SCI_SHIFT_ONLY);
        slow_kernel = CLUSTERED_GMXPACKED_SPLIT_DENSE_OFFSET_KERNEL(
            ClusteredRegularLJOutputMode::FULL,
            ClusteredRegularLJShiftMode::PAIR_SHIFT);
    }
#undef CLUSTERED_GMXPACKED_SPLIT_DENSE_OFFSET_KERNEL

    Launch_Clustered_Regular_LJ_Kernel(
        fast_kernel, grid_size, block_size, view, workspace,
        workspace.d_sorted_lj_comb, NULL, sci_shift_flags, 1, input);
    Launch_Clustered_Regular_LJ_Kernel(
        slow_kernel, grid_size, block_size, view, workspace,
        workspace.d_sorted_lj_comb, view.pair_shift_bits, sci_shift_flags, 0,
        input);
}

static void Launch_Clustered_Regular_LJ_Regular(
    const CLUSTERED_SPATIAL_VIEW& view,
    const LJClusteredWorkspace& workspace, bool use_lj_comb,
    dim3 block_size, const ClusteredRegularLJKernelInput& input)
{
    const bool full_output = input.store_energy || input.store_virial;
    const dim3 grid_size = {
        static_cast<unsigned int>(view.gmxpacked_sci_numbers), 1u, 1u};
    if (use_lj_comb)
    {
        auto kernel =
            Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                ClusteredRegularLJOutputMode::FORCE_ONLY,
                ClusteredRegularLJParameterMode::COMBINATION,
                ClusteredRegularLJLayoutMode::REGULAR,
                ClusteredRegularLJShiftMode::PAIR_SHIFT>;
        if (full_output)
        {
            kernel =
                Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                    ClusteredRegularLJOutputMode::FULL,
                    ClusteredRegularLJParameterMode::COMBINATION,
                    ClusteredRegularLJLayoutMode::REGULAR,
                    ClusteredRegularLJShiftMode::PAIR_SHIFT>;
        }
        Launch_Clustered_Regular_LJ_Kernel(
            kernel, grid_size, block_size, view, workspace,
            workspace.d_sorted_lj_comb, view.pair_shift_bits, NULL, 0, input);
        return;
    }

    auto kernel =
        Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
            ClusteredRegularLJOutputMode::FORCE_ONLY,
            ClusteredRegularLJParameterMode::AB_TABLE,
            ClusteredRegularLJLayoutMode::REGULAR,
            ClusteredRegularLJShiftMode::PAIR_SHIFT>;
    if (full_output)
    {
        kernel =
            Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                ClusteredRegularLJOutputMode::FULL,
                ClusteredRegularLJParameterMode::AB_TABLE,
                ClusteredRegularLJLayoutMode::REGULAR,
                ClusteredRegularLJShiftMode::PAIR_SHIFT>;
    }
    Launch_Clustered_Regular_LJ_Kernel(
        kernel, grid_size, block_size, view, workspace, NULL,
        view.pair_shift_bits, NULL, 0, input);
}

static void Launch_Clustered_Regular_LJ(
    const CLUSTERED_SPATIAL_VIEW& view, LJClusteredWorkspace& workspace,
    bool lj_comb_table_compatible,
    const ClusteredRegularLJKernelInput& requested_input)
{
    const bool use_lj_comb = lj_comb_table_compatible;
    static bool warned_lj_comb_incompatible = false;
    if (!lj_comb_table_compatible && !warned_lj_comb_incompatible)
    {
        fprintf(stderr,
                "[clustered gmxpacked lj comb] requested but LJ pair "
                "table is not compatible with geometric comb; using "
                "AB-table parameter path\n");
        fflush(stderr);
        warned_lj_comb_incompatible = true;
    }

    const bool full_output =
        requested_input.store_energy || requested_input.store_virial;
    const bool has_payload =
        view.gmxpacked_sci_numbers > 0 &&
        view.gmxpacked_cjpacked_numbers > 0 &&
        view.gmxpacked_exclusion_numbers > 0 && view.gmxpacked_sci != NULL &&
        view.gmxpacked_cjpacked != NULL &&
        view.gmxpacked_exclusions != NULL &&
        workspace.d_sorted_atom_ids != NULL && workspace.d_sorted_xq != NULL &&
        workspace.d_sorted_lj_type != NULL &&
        (!use_lj_comb || workspace.d_sorted_lj_comb != NULL) &&
        requested_input.lj_ab_table != NULL;
    const bool has_required_force_scratch =
        !full_output || workspace.d_sorted_frc != NULL;
    const bool use_direct = has_payload && has_required_force_scratch;
    const bool fast_layout_compatible =
        view.cluster_size == kClusteredClusterSize &&
        view.super_cluster_clusters == kClusteredSuperClusterClusters &&
        view.cluster_numbers > 0;
    const bool use_fast =
        use_direct && fast_layout_compatible &&
        view.gmxpacked_sci_shift_safe_flags != NULL;
    const bool use_full_local_dense =
        use_fast && view.ghost_numbers == 0 &&
        view.local_atom_numbers == view.total_atom_numbers &&
        view.direct_local_atom_numbers == view.total_atom_numbers &&
        view.padded_total_atom_numbers ==
            view.cluster_numbers * kClusteredClusterSize &&
        view.cluster_numbers % kClusteredSuperClusterClusters == 0;

    static bool warned_fast_unavailable = false;
    if (use_direct && !use_fast && !warned_fast_unavailable)
    {
        fprintf(stderr,
                "[clustered gmxpacked fast] requested but requires "
                "dense %dx%d gmxpacked layout "
                "(cluster_size=%d super_cluster_clusters=%d "
                "lj_comb=%d); falling back to regular gmxpacked "
                "kernel\n",
                kClusteredClusterSize, kClusteredSuperClusterClusters,
                view.cluster_size, view.super_cluster_clusters,
                use_lj_comb ? 1 : 0);
        fflush(stderr);
        warned_fast_unavailable = true;
    }
    if (!use_direct)
    {
        throw std::runtime_error(
            "clustered regular LJ gmxpacked payload is unavailable");
    }

    ClusteredRegularLJKernelInput kernel_input = requested_input;
    if (full_output)
    {
        if (workspace.gmxpacked_force_scratch_memset_time_recorder != NULL)
        {
            workspace.gmxpacked_force_scratch_memset_time_recorder->Start();
        }
        deviceMemset(workspace.d_sorted_frc, 0,
                     sizeof(VECTOR) * view.total_atom_numbers);
        if (workspace.gmxpacked_force_scratch_memset_time_recorder != NULL)
        {
            workspace.gmxpacked_force_scratch_memset_time_recorder->Stop();
        }
        kernel_input.force = workspace.d_sorted_frc;
    }

    const dim3 block_size = {
        static_cast<unsigned int>(view.cluster_size),
        static_cast<unsigned int>(view.cluster_size), 1u};
    if (workspace.direct_kernel_time_recorder != NULL)
    {
        workspace.direct_kernel_time_recorder->Start();
    }
    if (workspace.gmxpacked_kernel_launch_time_recorder != NULL)
    {
        workspace.gmxpacked_kernel_launch_time_recorder->Start();
    }
    if (use_full_local_dense)
    {
        Launch_Clustered_Regular_LJ_Full_Dense(
            view, workspace, use_lj_comb, block_size, kernel_input);
    }
    else if (use_fast)
    {
        Launch_Clustered_Regular_LJ_Dense_Offset(
            view, workspace, use_lj_comb, block_size, kernel_input);
    }
    else
    {
        Launch_Clustered_Regular_LJ_Regular(
            view, workspace, use_lj_comb, block_size, kernel_input);
    }
    if (workspace.gmxpacked_kernel_launch_time_recorder != NULL)
    {
        workspace.gmxpacked_kernel_launch_time_recorder->Stop();
    }

    if (full_output)
    {
        if (workspace.gmxpacked_sorted_force_scatter_time_recorder != NULL)
        {
            workspace.gmxpacked_sorted_force_scatter_time_recorder->Start();
        }
        Launch_Device_Kernel(
            Scatter_Sorted_Clustered_Force,
            (view.total_atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, view.total_atom_numbers,
            workspace.d_sorted_atom_ids, workspace.d_sorted_frc,
            requested_input.force);
        if (workspace.gmxpacked_sorted_force_scatter_time_recorder != NULL)
        {
            workspace.gmxpacked_sorted_force_scatter_time_recorder->Stop();
        }
    }
    if (workspace.direct_kernel_time_recorder != NULL)
    {
        workspace.direct_kernel_time_recorder->Stop();
    }
}
#endif

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
        gmxpacked_lj_comb_table_compatible =
            Clustered_Gmxpacked_Lj_Comb_Table_Compatible(
                h_LJ_A, h_LJ_B, atom_type_numbers);
        Parameter_Host_To_Device();
        is_initialized = 1;
    }
    if (is_initialized)
    {
        this->cutoff = cutoff;
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
    std::vector<float2> h_LJ_AB_packed((size_t)pair_type_numbers);
    std::vector<float2> h_LJ_AB_matrix(
        (size_t)atom_type_numbers * (size_t)atom_type_numbers);
    for (int i = 0; i < pair_type_numbers; i += 1)
    {
        h_LJ_AB_packed[(size_t)i] = {h_LJ_A[i], h_LJ_B[i]};
    }
    for (int i = 0; i < atom_type_numbers; i += 1)
    {
        for (int j = 0; j < atom_type_numbers; j += 1)
        {
            const int pair_type = Get_LJ_Type(i, j);
            h_LJ_AB_matrix[(size_t)i * (size_t)atom_type_numbers + (size_t)j] =
                h_LJ_AB_packed[(size_t)pair_type];
        }
    }
    Device_Malloc_And_Copy_Safely((void**)&d_atom_LJ_type, h_atom_LJ_type,
                                  sizeof(int) * atom_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_A, h_LJ_A,
                                  sizeof(float) * pair_type_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_B, h_LJ_B,
                                  sizeof(float) * pair_type_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_AB_packed,
                                  h_LJ_AB_packed.data(),
                                  sizeof(float2) * pair_type_numbers);
    Device_Malloc_And_Copy_Safely(
        (void**)&d_LJ_AB_matrix, h_LJ_AB_matrix.data(),
        sizeof(float2) * (size_t)atom_type_numbers * (size_t)atom_type_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_energy_sum, h_LJ_energy_atom,
                                  sizeof(float));
    Device_Malloc_Safely((void**)&d_LJ_energy_atom,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&crd_with_LJ_parameters_local,
                         sizeof(VECTOR_LJ) * atom_numbers);
}

void LENNARD_JONES_INFORMATION::LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(
    const int atom_numbers, const int local_atom_numbers,
    const int ghost_numbers, const VECTOR* crd, const float* charge,
    VECTOR* frc, const LTMatrix3 cell, const LTMatrix3 rcell,
    const float pme_beta,
    const int need_atom_energy, float* atom_energy, const int need_virial,
    LTMatrix3* atom_virial, float* atom_direct_pme_energy)
{
    if (is_initialized)
    {
        if (clustered_neighbor_provider == NULL ||
            clustered_workspace == NULL)
        {
            throw std::runtime_error(
                "regular LJ requires the clustered spatial service");
        }
        CLUSTERED_SPATIAL_VIEW clustered_view = {};
        if (d_LJ_AB_packed == NULL)
        {
            throw std::runtime_error(
                "clustered regular LJ requires packed LJ parameters");
        }
        ClusteredBuildRequest request;
        request.coordinates = crd;
        request.cell = cell;
        request.reciprocal_cell = rcell;
        request.cutoff = cutoff;
        clustered_neighbor_provider->Build(request);
        if (clustered_neighbor_provider->TotalAtomNumbers() > 0)
        {
            clustered_workspace->Gather_Plain(
                crd, charge, crd_with_LJ_parameters_local, cell, rcell,
                d_LJ_AB_packed);
            CLUSTERED_SPATIAL_VIEW_REQUIREMENTS requirements;
            requirements.local_atom_numbers = local_atom_numbers;
            requirements.ghost_numbers = ghost_numbers;
            requirements.cutoff = cutoff;
            requirements.require_all_local_atoms = true;
            requirements.require_gmxpacked_payload = true;
#if defined(USE_CUDA) || defined(USE_HIP)
            requirements.require_pair_shift_metadata = true;
            requirements.require_pair_shift_rcell = true;
            requirements.pair_shift_rcell = rcell;
#endif
            const char* view_failure_reason = nullptr;
            if (!clustered_neighbor_provider->AcquireView(
                    requirements, &clustered_view, &view_failure_reason))
            {
                throw std::runtime_error(
                    view_failure_reason != nullptr
                        ? view_failure_reason
                        : "clustered regular LJ spatial view is unavailable");
            }
        }
        if (need_atom_energy)
        {
            deviceMemset(atom_direct_pme_energy, 0,
                         sizeof(float) * this->atom_numbers);
            deviceMemset(d_LJ_energy_atom, 0,
                         sizeof(float) * this->atom_numbers);
        }

        if (atom_numbers == 0 || local_atom_numbers == 0) return;

        {
            const bool clustered_gather_ready =
                clustered_workspace->Plain_Gather_Ready_For_Current_Step();
            if (!clustered_gather_ready)
                return;
#ifdef USE_CPU
            if (clustered_view.gmxpacked_sci_numbers <= 0 ||
                clustered_view.gmxpacked_cjpacked_numbers <= 0 ||
                clustered_view.gmxpacked_exclusion_numbers <= 0 ||
                clustered_view.gmxpacked_sci == NULL ||
                clustered_view.gmxpacked_cjpacked == NULL ||
                clustered_view.gmxpacked_exclusions == NULL)
            {
                return;
            }
            auto cpu_f =
                Cpu_Gmxpacked_Clustered_Lennard_Jones_And_Direct_Coulomb<
                    false>;
            if (need_atom_energy || need_virial)
            {
                cpu_f =
                    Cpu_Gmxpacked_Clustered_Lennard_Jones_And_Direct_Coulomb<
                        true>;
            }
            if (clustered_workspace->direct_kernel_time_recorder != NULL)
            {
                clustered_workspace->direct_kernel_time_recorder->Start();
            }
            cpu_f(
                clustered_view.gmxpacked_sci_numbers,
                clustered_view.cluster_size, local_atom_numbers,
                clustered_view.cluster_offsets,
                clustered_view.cluster_valid_masks,
                clustered_view.cluster_local_masks,
                clustered_view.super_cluster_offsets,
                clustered_view.gmxpacked_sci,
                clustered_view.gmxpacked_cjpacked,
                clustered_view.gmxpacked_exclusions,
                clustered_workspace->d_sorted_atom_ids,
                clustered_workspace->d_sorted_xq,
                clustered_workspace->d_sorted_lj_type, cell, d_LJ_A,
                d_LJ_B, cutoff, frc, pme_beta, atom_energy, atom_virial,
                atom_direct_pme_energy, d_LJ_energy_atom,
                need_atom_energy != 0, need_virial != 0);
            if (clustered_workspace->direct_kernel_time_recorder != NULL)
            {
                clustered_workspace->direct_kernel_time_recorder->Stop();
            }
            return;
#endif
#ifndef USE_CPU
            const ClusteredRegularLJKernelInput kernel_input = {
                cell,
                d_LJ_AB_packed,
                cutoff,
                frc,
                pme_beta,
                atom_energy,
                atom_virial,
                atom_direct_pme_energy,
                d_LJ_energy_atom,
                need_atom_energy != 0,
                need_virial != 0};
            Launch_Clustered_Regular_LJ(
                clustered_view, *clustered_workspace,
                gmxpacked_lj_comb_table_compatible, kernel_input);
#endif
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
