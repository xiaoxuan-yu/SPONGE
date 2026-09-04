#include "LJ_soft_core.h"

#include "../neighbor_list/contract/cpu_traversal.h"
#include "../xponge/load/native/lj_soft.hpp"
#include "../xponge/xponge.h"

__global__ void Copy_LJ_Type_And_Mask_To_New_Crd(const int atom_numbers,
                                                 VECTOR_LJ_SOFT_TYPE* new_crd,
                                                 const int* LJ_type_A,
                                                 const int* LJ_type_B,
                                                 const int* mask)
{
#ifdef USE_GPU
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        new_crd[atom_i].LJ_type = LJ_type_A[atom_i];
        new_crd[atom_i].LJ_type_B = LJ_type_B[atom_i];
        new_crd[atom_i].mask = mask[atom_i];
    }
}

static __global__ void device_add(float* variable, const float adder)
{
    variable[0] += adder;
}

__global__ void Copy_Crd_And_Charge_To_New_Crd(const int atom_numbers,
                                               const VECTOR* crd,
                                               VECTOR_LJ_SOFT_TYPE* new_crd,
                                               const float* charge)
{
#ifdef USE_GPU
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        new_crd[atom_i].crd = crd[atom_i];
        new_crd[atom_i].charge = charge[atom_i];
    }
}

__global__ void Copy_Crd_And_Charge_To_New_Crd(const int atom_numbers,
                                               const VECTOR* crd,
                                               VECTOR_LJ_SOFT_TYPE* new_crd,
                                               const float* charge,
                                               const float* charge_BA)
{
#ifdef USE_GPU
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        new_crd[atom_i].crd = crd[atom_i];
        new_crd[atom_i].charge = charge[atom_i];
        new_crd[atom_i].charge_BA = charge_BA[atom_i];
    }
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

static __device__ __forceinline__ VECTOR
Reduce_Clustered_Subgroup_Vector(VECTOR value, int lane, int subgroup_width)
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
#endif

#ifdef USE_GPU
static __device__ __forceinline__ float Reduce_Gmxpacked_Soft_Core_Over_J(
    float value)
{
    value += deviceShflDown(FULL_MASK, value, 16, warpSize);
    value += deviceShflDown(FULL_MASK, value, 8, warpSize);
    return value;
}

template <bool full_output>
static __global__
__launch_bounds__(kClusteredClusterSize* kClusteredSuperClusterClusters, full_output ? 8 : 10) void Nbnxm_Gmxpacked_Lennard_Jones_And_Direct_Coulomb_Soft_Core(
    const int sci_numbers, const int cluster_numbers,
    const int local_atom_numbers, const int* cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const int* super_cluster_offsets,
    const LJ_CLUSTERED_GMXPACKED_SCI* sci_entries,
    const LJ_CLUSTERED_GMXPACKED_CJ* cjpacked_entries,
    const LJ_CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    const uint64_t* pair_shift_bits, const int* sorted_atom_ids,
    const VECTOR_LJ_SOFT_TYPE* sorted_crd, const LTMatrix3 cell,
    const float* LJ_type_AA, const float* LJ_type_AB, const float* LJ_type_BA,
    const float* LJ_type_BB, const float cutoff, VECTOR* sorted_frc,
    const float pme_beta, float* atom_energy, LTMatrix3* atom_lj_virial,
    float* atom_direct_pme_energy, const float lambda, const float alpha,
    const float p, const float input_sigma_6, const float input_sigma_6_min,
    float* this_energy, const bool store_energy, const bool store_virial)
{
    constexpr int cluster_size = kClusteredClusterSize;
    constexpr int super_cluster_clusters = kClusteredSuperClusterClusters;
    const int sci = blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (sci >= sci_numbers || tid >= cluster_size * super_cluster_clusters)
    {
        return;
    }

    const int i_lane = threadIdx.x;
    const int j_lane = threadIdx.y;
    const int lane = tid & (warpSize - 1);
    const int split = tid / warpSize;
    const int split_j_lane = j_lane - split * kClusteredSplitJClusterSize;
    const int i_lane_slot = 1u << static_cast<unsigned int>(i_lane);
    const LJ_CLUSTERED_GMXPACKED_SCI sci_entry = sci_entries[sci];
    const int cluster_i_start =
        super_cluster_offsets[sci_entry.supercluster_id];
    const int cluster_i_end =
        super_cluster_offsets[sci_entry.supercluster_id + 1];
    const int active_cluster_count = cluster_i_end - cluster_i_start;
    const float cutoff_sq = cutoff * cutoff;
    const VECTOR sci_shift =
        Clustered_Shift_Vector_From_Id(sci_entry.shift_id, cell);

    constexpr int shared_i_atom_count = cluster_size * super_cluster_clusters;
    __shared__ __align__(
        16) unsigned char shared_i_atoms_storage[sizeof(VECTOR_LJ_SOFT_TYPE) *
                                                 shared_i_atom_count];
    VECTOR_LJ_SOFT_TYPE* shared_i_atoms =
        reinterpret_cast<VECTOR_LJ_SOFT_TYPE*>(shared_i_atoms_storage);
    float* shared_i_crd_x = reinterpret_cast<float*>(shared_i_atoms_storage);
    float* shared_i_crd_y = shared_i_crd_x + shared_i_atom_count;
    float* shared_i_crd_z = shared_i_crd_y + shared_i_atom_count;
    int* shared_i_lj_type_a =
        reinterpret_cast<int*>(shared_i_crd_z + shared_i_atom_count);
    int* shared_i_lj_type_b = shared_i_lj_type_a + shared_i_atom_count;
    int* shared_i_mask = shared_i_lj_type_b + shared_i_atom_count;
    float* shared_i_charge_a =
        reinterpret_cast<float*>(shared_i_mask + shared_i_atom_count);
    float* shared_i_charge_ba = shared_i_charge_a + shared_i_atom_count;
    __shared__ int shared_i_sorted_ids[cluster_size * super_cluster_clusters];
    __shared__ unsigned int shared_i_valid_masks[super_cluster_clusters];
    __shared__ unsigned int shared_i_local_masks[super_cluster_clusters];
    __shared__ __align__(16) unsigned char
        shared_split_force_storage[sizeof(VECTOR) * 2 * super_cluster_clusters *
                                   cluster_size];
    VECTOR(*shared_split_force)
    [super_cluster_clusters][cluster_size] =
        reinterpret_cast<VECTOR(*)[super_cluster_clusters][cluster_size]>(
            shared_split_force_storage);
    __shared__ float shared_split_lj_energy[2][super_cluster_clusters]
                                           [cluster_size];
    __shared__ float shared_split_coulomb_energy[2][super_cluster_clusters]
                                                [cluster_size];
    __shared__ __align__(16) unsigned char
        shared_split_virial_storage[sizeof(LTMatrix3) * 2 *
                                    super_cluster_clusters * cluster_size];
    LTMatrix3(*shared_split_virial)[super_cluster_clusters][cluster_size] =
        reinterpret_cast<LTMatrix3(*)[super_cluster_clusters][cluster_size]>(
            shared_split_virial_storage);
    VECTOR force_i[super_cluster_clusters];
    float energy_lj_i[super_cluster_clusters];
    float energy_coulomb_i[super_cluster_clusters];
    LTMatrix3 virial_i[super_cluster_clusters];
    unsigned int active_i_mask = 0u;
#pragma unroll
    for (int i_local = 0; i_local < super_cluster_clusters; i_local += 1)
    {
        force_i[i_local] = {0.0f, 0.0f, 0.0f};
        if constexpr (full_output)
        {
            energy_lj_i[i_local] = 0.0f;
            energy_coulomb_i[i_local] = 0.0f;
            virial_i[i_local] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    if (j_lane == 0)
    {
        if (i_lane < active_cluster_count)
        {
            const int cluster_i = cluster_i_start + i_lane;
            shared_i_valid_masks[i_lane] = cluster_valid_masks[cluster_i];
            shared_i_local_masks[i_lane] = cluster_local_masks[cluster_i];
        }
        else
        {
            shared_i_valid_masks[i_lane] = 0u;
            shared_i_local_masks[i_lane] = 0u;
        }
    }
    if (j_lane < active_cluster_count)
    {
        const int cluster_i = cluster_i_start + j_lane;
        const int i_slot = j_lane * cluster_size + i_lane;
        if (Clustered_Lane_Bit_Is_Set(cluster_valid_masks[cluster_i],
                                      i_lane_slot))
        {
            const int sorted_i = cluster_offsets[cluster_i] + i_lane;
            const VECTOR_LJ_SOFT_TYPE atom_i = sorted_crd[sorted_i];
            if constexpr (full_output)
            {
                shared_i_crd_x[i_slot] = atom_i.crd.x;
                shared_i_crd_y[i_slot] = atom_i.crd.y;
                shared_i_crd_z[i_slot] = atom_i.crd.z;
                shared_i_lj_type_a[i_slot] = atom_i.LJ_type;
                shared_i_lj_type_b[i_slot] = atom_i.LJ_type_B;
                shared_i_mask[i_slot] = atom_i.mask;
                shared_i_charge_a[i_slot] = atom_i.charge;
                shared_i_charge_ba[i_slot] = atom_i.charge_BA;
            }
            else
            {
                shared_i_atoms[i_slot] = atom_i;
            }
            shared_i_sorted_ids[i_slot] = sorted_i;
        }
        else
        {
            shared_i_sorted_ids[i_slot] = -1;
        }
    }
    __syncthreads();

#pragma unroll
    for (int i_local = 0; i_local < super_cluster_clusters; i_local += 1)
    {
        if (i_local < active_cluster_count &&
            Clustered_Lane_Bit_Is_Set(shared_i_valid_masks[i_local],
                                      i_lane_slot) &&
            Clustered_Lane_Bit_Is_Set(shared_i_local_masks[i_local],
                                      i_lane_slot))
        {
            active_i_mask |= 1u << static_cast<unsigned int>(i_local);
        }
    }

    for (int packed_idx = sci_entry.cjpacked_begin;
         packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
    {
        const LJ_CLUSTERED_GMXPACKED_CJ& packed = cjpacked_entries[packed_idx];
        const unsigned int imask = packed.split[split].imask;
        if (imask == 0u)
        {
            continue;
        }
        const unsigned int effective_mask = Clustered_Gmxpacked_Effective_Imask(
            packed, exclusion_entries, split, split_j_lane, i_lane,
            cluster_size);

        for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
        {
            const unsigned int jm_mask =
                ((1u << super_cluster_clusters) - 1u)
                << static_cast<unsigned int>(jm * super_cluster_clusters);
            if ((imask & jm_mask) == 0u)
            {
                continue;
            }
            const int cluster_j = packed.cj[jm];
            if (cluster_j < 0 || cluster_j >= cluster_numbers ||
                !Clustered_Lane_Is_Valid(cluster_valid_masks[cluster_j],
                                         j_lane))
            {
                continue;
            }
            const int sorted_j = cluster_offsets[cluster_j] + j_lane;
            const VECTOR_LJ_SOFT_TYPE r2 = sorted_crd[sorted_j];
            const bool j_is_local =
                Clustered_Lane_Is_Local(cluster_local_masks[cluster_j], j_lane);
            const float ij_factor = j_is_local ? 1.0f : 0.5f;
            const uint64_t shift_bits =
                pair_shift_bits != NULL
                    ? pair_shift_bits[packed_idx * kClusteredJGroupSize + jm]
                    : 0ull;
            VECTOR force_j = {0.0f, 0.0f, 0.0f};

#pragma unroll
            for (int i_local = 0; i_local < super_cluster_clusters;
                 i_local += 1)
            {
                if (i_local >= active_cluster_count)
                {
                    continue;
                }
                const unsigned int pair_bit =
                    1u << static_cast<unsigned int>(
                        jm * super_cluster_clusters + i_local);
                if ((effective_mask & pair_bit) == 0u ||
                    (active_i_mask &
                     (1u << static_cast<unsigned int>(i_local))) == 0u ||
                    (Clustered_Get_Pair_Active_I_Mask(shift_bits, split) &
                     (1u << static_cast<unsigned int>(i_local))) == 0u)
                {
                    continue;
                }
                const int i_slot = i_local * cluster_size + i_lane;
                VECTOR_LJ_SOFT_TYPE r1;
                if constexpr (full_output)
                {
                    r1 = {{shared_i_crd_x[i_slot], shared_i_crd_y[i_slot],
                           shared_i_crd_z[i_slot]},
                          shared_i_lj_type_a[i_slot],
                          shared_i_lj_type_b[i_slot],
                          shared_i_mask[i_slot],
                          shared_i_charge_a[i_slot],
                          shared_i_charge_ba[i_slot]};
                }
                else
                {
                    r1 = shared_i_atoms[i_slot];
                }
                const VECTOR pair_shift =
                    pair_shift_bits != NULL
                        ? Clustered_Shift_Vector_From_Id(
                              Clustered_Get_Pair_Shift_Id(shift_bits, i_local),
                              cell)
                        : sci_shift;
                const float dx = r2.crd.x - r1.crd.x - pair_shift.x;
                const float dy = r2.crd.y - r1.crd.y - pair_shift.y;
                const float dz = r2.crd.z - r1.crd.z - pair_shift.z;
                const float dr2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
                if (dr2 >= cutoff_sq || dr2 == 0.0f)
                {
                    continue;
                }

                const int pair_type_A = Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                const int pair_type_B = Get_LJ_Type(r1.LJ_type_B, r2.LJ_type_B);
                VECTOR pair_force = {0.0f, 0.0f, 0.0f};
                float pair_lj_energy = 0.0f;
                float pair_coulomb_energy = 0.0f;
                LTMatrix3 pair_virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                Compute_Clustered_Soft_Core_Pair<full_output>(
                    r1, r2, dx, dy, dz, dr2, LJ_type_AA[pair_type_A],
                    LJ_type_AB[pair_type_A], LJ_type_BA[pair_type_B],
                    LJ_type_BB[pair_type_B], lambda, alpha, p, input_sigma_6,
                    input_sigma_6_min, pme_beta, ij_factor, &pair_force,
                    &pair_lj_energy, &pair_coulomb_energy, &pair_virial);
                force_i[i_local] = force_i[i_local] + pair_force;
                if (j_is_local)
                {
                    force_j = force_j - pair_force;
                }
                if constexpr (full_output)
                {
                    energy_lj_i[i_local] += pair_lj_energy;
                    energy_coulomb_i[i_local] += pair_coulomb_energy;
                    virial_i[i_local] = virial_i[i_local] + pair_virial;
                }
            }

            if (j_is_local)
            {
                const VECTOR reduced_j = Reduce_Clustered_Subgroup_Vector(
                    force_j, lane, cluster_size);
                if (i_lane == 0)
                {
                    atomicAdd(sorted_frc + sorted_j, reduced_j);
                }
            }
        }
    }

#pragma unroll
    for (int i_local = 0; i_local < super_cluster_clusters; i_local += 1)
    {
        const bool active_i =
            (active_i_mask & (1u << static_cast<unsigned int>(i_local))) != 0u;
        VECTOR reduced_force =
            active_i ? force_i[i_local] : VECTOR{0.0f, 0.0f, 0.0f};
        reduced_force.x = Reduce_Gmxpacked_Soft_Core_Over_J(reduced_force.x);
        reduced_force.y = Reduce_Gmxpacked_Soft_Core_Over_J(reduced_force.y);
        reduced_force.z = Reduce_Gmxpacked_Soft_Core_Over_J(reduced_force.z);
        if (lane < cluster_size)
        {
            shared_split_force[split][i_local][i_lane] = reduced_force;
        }
        if constexpr (full_output)
        {
            const float reduced_lj = Reduce_Gmxpacked_Soft_Core_Over_J(
                active_i ? energy_lj_i[i_local] : 0.0f);
            const float reduced_coulomb = Reduce_Gmxpacked_Soft_Core_Over_J(
                active_i ? energy_coulomb_i[i_local] : 0.0f);
            LTMatrix3 reduced_virial =
                active_i ? virial_i[i_local] : LTMatrix3(0.0f);
            reduced_virial.a11 =
                Reduce_Gmxpacked_Soft_Core_Over_J(reduced_virial.a11);
            reduced_virial.a21 =
                Reduce_Gmxpacked_Soft_Core_Over_J(reduced_virial.a21);
            reduced_virial.a22 =
                Reduce_Gmxpacked_Soft_Core_Over_J(reduced_virial.a22);
            reduced_virial.a31 =
                Reduce_Gmxpacked_Soft_Core_Over_J(reduced_virial.a31);
            reduced_virial.a32 =
                Reduce_Gmxpacked_Soft_Core_Over_J(reduced_virial.a32);
            reduced_virial.a33 =
                Reduce_Gmxpacked_Soft_Core_Over_J(reduced_virial.a33);
            if (lane < cluster_size)
            {
                shared_split_lj_energy[split][i_local][i_lane] = reduced_lj;
                shared_split_coulomb_energy[split][i_local][i_lane] =
                    reduced_coulomb;
                shared_split_virial[split][i_local][i_lane] = reduced_virial;
            }
        }
    }
    __syncthreads();

    if (j_lane == 0)
    {
#pragma unroll
        for (int i_local = 0; i_local < super_cluster_clusters; i_local += 1)
        {
            if (i_local >= active_cluster_count ||
                (active_i_mask & (1u << static_cast<unsigned int>(i_local))) ==
                    0u)
            {
                continue;
            }
            const int i_slot = i_local * cluster_size + i_lane;
            const int sorted_i = shared_i_sorted_ids[i_slot];
            atomicAdd(sorted_frc + sorted_i,
                      shared_split_force[0][i_local][i_lane] +
                          shared_split_force[1][i_local][i_lane]);
            if constexpr (full_output)
            {
                const int atom_i = sorted_atom_ids[sorted_i];
                const float total_lj =
                    shared_split_lj_energy[0][i_local][i_lane] +
                    shared_split_lj_energy[1][i_local][i_lane];
                const float total_coulomb =
                    shared_split_coulomb_energy[0][i_local][i_lane] +
                    shared_split_coulomb_energy[1][i_local][i_lane];
                if (store_energy)
                {
                    atomicAdd(atom_energy + atom_i, total_lj + total_coulomb);
                    atomicAdd(this_energy + atom_i, total_lj);
                    atomicAdd(atom_direct_pme_energy + atom_i, total_coulomb);
                }
                if (store_virial)
                {
                    atomicAdd(atom_lj_virial + atom_i,
                              shared_split_virial[0][i_local][i_lane] +
                                  shared_split_virial[1][i_local][i_lane]);
                }
            }
        }
    }
}

static __global__ void Scatter_Gmxpacked_Soft_Core_Force(
    const int total_atom_numbers, const int local_atom_numbers,
    const int* sorted_atom_ids, const VECTOR* sorted_frc, VECTOR* frc)
{
    const int sorted_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_i < total_atom_numbers)
    {
        const int atom_i = sorted_atom_ids[sorted_i];
        if (atom_i >= 0 && atom_i < local_atom_numbers)
        {
            frc[atom_i] = frc[atom_i] + sorted_frc[sorted_i];
        }
    }
}
#endif

static __global__ void Total_C6_Get(int atom_numbers, int* atom_lj_type_A,
                                    int* atom_lj_type_B, float* d_lj_Ab,
                                    float* d_lj_Bb, double* d_factor,
                                    const float lambda)
{
    int j;
    double temp_sum = 0.0;
    int xA, yA, xB, yB;
    int itype_A, jtype_A, itype_B, jtype_B, atom_pair_LJ_type_A,
        atom_pair_LJ_type_B;
    float lambda_ = 1.0 - lambda;
#ifdef USE_GPU
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers;
         i += gridDim.x * blockDim.x)
#else
#pragma omp parallel for firstprivate(                         \
        j, xA, yA, xB, yB, itype_A, jtype_A, itype_B, jtype_B, \
            atom_pair_LJ_type_A, atom_pair_LJ_type_B, lambda)  \
    reduction(+ : temp_sum)
    for (int i = 0; i < atom_numbers; i++)
#endif
    {
        itype_A = atom_lj_type_A[i];
        itype_B = atom_lj_type_B[i];
        double temp_small_sum = 0.0;
#ifdef USE_GPU
        for (j = blockIdx.y * blockDim.y + threadIdx.y; j < atom_numbers;
             j += gridDim.y * blockDim.y)
#else
        for (j = 0; j < atom_numbers; j++)
#endif
        {
            jtype_A = atom_lj_type_A[j];
            jtype_B = atom_lj_type_B[j];
            yA = (jtype_A - itype_A);
            xA = yA >> 31;
            yA = (yA ^ xA) - xA;
            xA = jtype_A + itype_A;
            jtype_A = (xA + yA) >> 1;
            xA = (xA - yA) >> 1;
            atom_pair_LJ_type_A = (jtype_A * (jtype_A + 1) >> 1) + xA;

            yB = (jtype_B - itype_B);
            xB = yB >> 31;
            yB = (yB ^ xB) - xB;
            xB = jtype_B + itype_B;
            jtype_B = (xB + yB) >> 1;
            xB = (xB - yB) >> 1;
            atom_pair_LJ_type_B = (jtype_B * (jtype_B + 1) >> 1) + xB;

            temp_small_sum += lambda_ * d_lj_Ab[atom_pair_LJ_type_A];
            temp_small_sum += lambda * d_lj_Bb[atom_pair_LJ_type_B];
        }
        temp_sum += temp_small_sum;
    }
    atomicAdd(d_factor, temp_sum);
}

#ifdef USE_CPU
template <bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb>
static void Cpu_Gmxpacked_Lennard_Jones_And_Direct_Coulomb_Soft_Core(
    const CLUSTERED_SPATIAL_VIEW& view, const int local_atom_numbers,
    const int* sorted_atom_ids, const VECTOR_LJ_SOFT_TYPE* sorted_crd,
    const LTMatrix3 cell, const LTMatrix3 rcell, const float* LJ_type_AA,
    const float* LJ_type_AB, const float* LJ_type_BA, const float* LJ_type_BB,
    const float cutoff, VECTOR* frc, const float pme_beta, float* atom_energy,
    LTMatrix3* atom_lj_virial, float* atom_direct_pme_energy,
    const float lambda, const float alpha, const float p,
    const float input_sigma_6, const float input_sigma_6_min,
    float* this_energy, const bool store_energy, const bool store_virial)
{
    const float lambda_ = 1.0f - lambda;
    const float cutoff_sq = cutoff * cutoff;
    const int sci_numbers = view.gmxpacked_sci_numbers;
    const int cluster_size = view.cluster_size;
    const int* cluster_offsets = view.cluster_offsets;

#pragma omp parallel for schedule(dynamic)
    for (int sci = 0; sci < sci_numbers; sci += 1)
    {
        const LJ_CLUSTERED_GMXPACKED_SCI sci_entry = view.gmxpacked_sci[sci];
        const VECTOR shift_vec =
            Clustered_Shift_Vector_From_Id(sci_entry.shift_id, cell);
        auto consume_i =
            [&](const CLUSTERED_GMXPACKED_CPU_LOCAL_I_CANDIDATE& pair_i)
        {
            const int sorted_atom_i = pair_i.sorted_i;
            const int atom_i = sorted_atom_ids[sorted_atom_i];
            VECTOR_LJ_SOFT_TYPE r1 = sorted_crd[sorted_atom_i];
            r1.crd = r1.crd + shift_vec;
            VECTOR frc_i = {0.0f, 0.0f, 0.0f};
            float energy_lj = 0.0f;
            float energy_coulomb = 0.0f;
            LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

            auto consume_tile =
                [&](const CLUSTERED_GMXPACKED_CPU_J_TILE_CANDIDATE& tile)
            {
                const int cluster_j = tile.cluster_j;
                VECTOR frc_j[kClusteredClusterSize] = {};
                unsigned int active_force_j_mask = 0u;
                unsigned int pair_j_mask = tile.active_j_mask;
                while (pair_j_mask != 0u)
                {
                    const int lane_j = __builtin_ctz(pair_j_mask);
                    pair_j_mask &= pair_j_mask - 1u;
                    const int sorted_atom_j =
                        cluster_offsets[cluster_j] + lane_j;
                    const int atom_j = sorted_atom_ids[sorted_atom_j];
                    const VECTOR_LJ_SOFT_TYPE r2 = sorted_crd[sorted_atom_j];
                    const VECTOR dr =
                        Get_Periodic_Displacement(r2, r1, cell, rcell);
                    const float dr2 = dr * dr;
                    if (dr2 >= cutoff_sq || dr2 == 0.0f)
                    {
                        continue;
                    }
                    const float dr_abs = sqrtf(dr2);
                    const float ij_factor =
                        atom_j < local_atom_numbers ? 1.0f : 0.5f;
                    const int atom_pair_LJ_type_A =
                        Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                    const int atom_pair_LJ_type_B =
                        Get_LJ_Type(r1.LJ_type_B, r2.LJ_type_B);
                    const float AA = LJ_type_AA[atom_pair_LJ_type_A];
                    const float AB = LJ_type_AB[atom_pair_LJ_type_A];
                    const float BA = LJ_type_BA[atom_pair_LJ_type_B];
                    const float BB = LJ_type_BB[atom_pair_LJ_type_B];

                    float pair_lj_energy = 0.0f;
                    float pair_coulomb_energy = 0.0f;
                    VECTOR frc_lin = {0.0f, 0.0f, 0.0f};
                    bool active_force = false;
                    if (BA * AA != 0 || BA + AA == 0)
                    {
                        if constexpr (need_force)
                        {
                            float frc_abs =
                                lambda_ * Get_LJ_Force(r1, r2, dr_abs, AA, AB) +
                                lambda * Get_LJ_Force(r1, r2, dr_abs, BA, BB);
                            if constexpr (need_coulomb)
                            {
                                frc_abs -= Get_Direct_Coulomb_Force(
                                    r1, r2, dr_abs, pme_beta);
                            }
                            frc_lin = frc_abs * dr;
                            active_force = true;
                        }
                        pair_lj_energy =
                            ij_factor *
                            (lambda_ * Get_LJ_Energy(r1, r2, dr_abs, AA, AB) +
                             lambda * Get_LJ_Energy(r1, r2, dr_abs, BA, BB));
                        if constexpr (need_coulomb)
                        {
                            pair_coulomb_energy =
                                ij_factor * Get_Direct_Coulomb_Energy(
                                                r1, r2, dr_abs, pme_beta);
                        }
                    }
                    else
                    {
                        const float sigma_A = Get_Soft_Core_Sigma(
                            AA, AB, input_sigma_6, input_sigma_6_min);
                        const float sigma_B = Get_Soft_Core_Sigma(
                            BA, BB, input_sigma_6, input_sigma_6_min);
                        const float dr_softcore_A = Get_Soft_Core_Distance(
                            AA, AB, sigma_A, dr_abs, alpha, p, lambda);
                        const float dr_softcore_B = Get_Soft_Core_Distance(
                            BB, BA, sigma_B, dr_abs, alpha, p, 1.0f - lambda);
                        if constexpr (need_force)
                        {
                            float frc_abs =
                                lambda_ * Get_Soft_Core_LJ_Force(r1, r2, dr_abs,
                                                                 dr_softcore_A,
                                                                 AA, AB) +
                                lambda * Get_Soft_Core_LJ_Force(r1, r2, dr_abs,
                                                                dr_softcore_B,
                                                                BA, BB);
                            if constexpr (need_coulomb)
                            {
                                frc_abs -= lambda_ *
                                           Get_Soft_Core_Direct_Coulomb_Force(
                                               r1, r2, dr_abs, dr_softcore_A,
                                               pme_beta);
                                frc_abs -=
                                    lambda * Get_Soft_Core_Direct_Coulomb_Force(
                                                 r1, r2, dr_abs, dr_softcore_B,
                                                 pme_beta);
                            }
                            frc_lin = frc_abs * dr;
                            active_force = true;
                        }
                        pair_lj_energy =
                            ij_factor *
                            (lambda_ *
                                 Get_LJ_Energy(r1, r2, dr_softcore_A, AA, AB) +
                             lambda *
                                 Get_LJ_Energy(r1, r2, dr_softcore_B, BA, BB));
                        if constexpr (need_coulomb)
                        {
                            pair_coulomb_energy =
                                ij_factor *
                                (lambda_ *
                                     Get_Direct_Coulomb_Energy(
                                         r1, r2, dr_softcore_A, pme_beta) +
                                 lambda * Get_Direct_Coulomb_Energy(
                                              r1, r2, dr_softcore_B, pme_beta));
                        }
                    }

                    if constexpr (need_force)
                    {
                        if (active_force)
                        {
                            frc_i = frc_i + frc_lin;
                            if (atom_j < local_atom_numbers)
                            {
                                frc_j[lane_j] = frc_j[lane_j] - frc_lin;
                                active_force_j_mask |=
                                    1u << static_cast<unsigned int>(lane_j);
                            }
                            if constexpr (need_virial)
                            {
                                virial = virial -
                                         ij_factor * Get_Virial_From_Force_Dis(
                                                         frc_lin, dr);
                            }
                        }
                    }
                    if constexpr (need_energy)
                    {
                        energy_lj += pair_lj_energy;
                        energy_coulomb += pair_coulomb_energy;
                    }
                }
                if constexpr (need_force)
                {
                    for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
                    {
                        if ((active_force_j_mask &
                             (1u << static_cast<unsigned int>(lane_j))) == 0u)
                        {
                            continue;
                        }
                        const int sorted_atom_j =
                            cluster_offsets[cluster_j] + lane_j;
                        const int atom_j = sorted_atom_ids[sorted_atom_j];
                        atomicAdd(frc + atom_j, frc_j[lane_j]);
                    }
                }
            };
            Clustered_Gmxpacked_CPU_For_Each_J_Tile_For_Local_I(view, pair_i,
                                                                consume_tile);

            if constexpr (need_force)
            {
                atomicAdd(frc + atom_i, frc_i);
            }
            if constexpr (need_energy)
            {
                if (store_energy)
                {
                    atomicAdd(atom_energy + atom_i, energy_lj + energy_coulomb);
                    atomicAdd(this_energy + atom_i, energy_lj);
                    if constexpr (need_coulomb)
                    {
                        atomicAdd(atom_direct_pme_energy + atom_i,
                                  energy_coulomb);
                    }
                }
            }
            if constexpr (need_virial)
            {
                if (store_virial)
                {
                    atomicAdd(atom_lj_virial + atom_i, virial);
                }
            }
        };
        Clustered_Gmxpacked_CPU_For_Each_Local_I_In_SCI(view, sci, consume_i);
    }
}
#endif

void LJ_SOFT_CORE::Initial(CONTROLLER* controller, float cutoff,
                           char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "LJ_soft_core");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    controller->printf(
        "START INITIALIZING FEP SOFT CORE FOR LJ AND COULOMB:\n");
    const auto& lj_soft = Xponge::system.classical_force_field.lj_soft_core;
    Xponge::LJSoftCore local_lj_soft;
    const Xponge::LJSoftCore* lj_soft_to_use = NULL;
    if (module_name == NULL)
    {
        lj_soft_to_use = &lj_soft;
    }
    else if (controller->Command_Exist(this->module_name, "in_file"))
    {
        Xponge::Native_Load_LJ_Soft_Core(&local_lj_soft, controller,
                                         this->module_name);
        lj_soft_to_use = &local_lj_soft;
    }

    if (lj_soft_to_use != NULL && lj_soft_to_use->atom_numbers > 0)
    {
        if (controller->Command_Exist("lambda_lj"))
        {
            this->lambda = atof(controller->Command("lambda_lj"));
            controller->printf("    FEP lj lambda: %f\n", this->lambda);
        }
        else
        {
            char error_reason[CHAR_LENGTH_MAX];
            sprintf(error_reason,
                    "Reason:\n\t'lambda_lj' is required for the calculation of "
                    "LJ_soft_core\n");
            controller->Throw_SPONGE_Error(spongeErrorMissingCommand,
                                           "LJ_SOFT_CORE::Initial",
                                           error_reason);
        }

        if (controller->Command_Exist("soft_core_alpha"))
        {
            this->alpha = atof(controller->Command("soft_core_alpha"));
            controller->printf("    FEP soft core alpha: %f\n", this->alpha);
        }
        else
        {
            controller->printf(
                "    FEP soft core alpha is set to default value 0.5\n");
            this->alpha = 0.5;
        }

        if (controller->Command_Exist("soft_core_powfer"))
        {
            this->p = atof(controller->Command("soft_core_powfer"));
            controller->printf("    FEP soft core powfer: %f\n", this->p);
        }
        else
        {
            controller->printf(
                "    FEP soft core powfer is set to default value 1.0.\n");
            this->p = 1.0;
        }

        if (controller->Command_Exist("soft_core_sigma"))
        {
            this->sigma = atof(controller->Command("soft_core_sigma"));
            controller->printf("    FEP soft core sigma: %f\n", this->sigma);
        }
        else
        {
            controller->printf(
                "    FEP soft core sigma is set to default value 3.0\n");
            this->sigma = 3.0;
        }
        if (controller->Command_Exist("soft_core_sigma_min"))
        {
            this->sigma_min = atof(controller->Command("soft_core_sigma_min"));
            controller->printf("    FEP soft core sigma min: %f\n",
                               this->sigma_min);
        }
        else
        {
            controller->printf(
                "    FEP soft core sigma min is set to default value 0.0\n");
            this->sigma_min = 0.0;
        }

        atom_numbers = lj_soft_to_use->atom_numbers;
        atom_type_numbers_A = lj_soft_to_use->atom_type_numbers_A;
        atom_type_numbers_B = lj_soft_to_use->atom_type_numbers_B;
        int toscan = 0;
        controller->printf("    atom_numbers is %d\n", atom_numbers);
        controller->printf(
            "    atom_LJ_type_number_A is %d, atom_LJ_type_number_B is %d\n",
            atom_type_numbers_A, atom_type_numbers_B);
        pair_type_numbers_A =
            atom_type_numbers_A * (atom_type_numbers_A + 1) / 2;
        pair_type_numbers_B =
            atom_type_numbers_B * (atom_type_numbers_B + 1) / 2;
        LJ_Soft_Core_Malloc();

        for (int i = 0; i < pair_type_numbers_A; i++)
        {
            h_LJ_AA[i] = lj_soft_to_use->LJ_AA[i];
        }
        for (int i = 0; i < pair_type_numbers_A; i++)
        {
            h_LJ_AB[i] = lj_soft_to_use->LJ_AB[i];
        }
        for (int i = 0; i < pair_type_numbers_B; ++i)
        {
            h_LJ_BA[i] = lj_soft_to_use->LJ_BA[i];
        }
        for (int i = 0; i < pair_type_numbers_B; ++i)
        {
            h_LJ_BB[i] = lj_soft_to_use->LJ_BB[i];
        }
        for (int i = 0; i < atom_numbers; i++)
        {
            h_atom_LJ_type_A[i] = lj_soft_to_use->atom_LJ_type_A[i];
            h_atom_LJ_type_B[i] = lj_soft_to_use->atom_LJ_type_B[i];
        }

        if (!lj_soft_to_use->subsystem_division.empty())
        {
            controller->printf(
                "    Start reading subsystem division information:\n");
            for (int i = 0; i < atom_numbers; i++)
            {
                h_subsys_division[i] = lj_soft_to_use->subsystem_division[i];
            }
            controller->printf(
                "    End reading subsystem division information\n\n");
        }
        else
        {
            controller->printf("    subsystem mask is set to 0 as default\n");
            for (int i = 0; i < atom_numbers; i++)
            {
                h_subsys_division[i] = 0;
            }
        }

        Parameter_Host_To_Device();
        is_initialized = 1;
        alpha_lambda_p = alpha * powf(lambda, p);
        alpha_lambda_p_ = alpha * powf(1 - lambda, p);
        sigma_6 = powf(sigma, 6);
        sigma_6_min = powf(sigma_min, 6);
        alpha_lambda_p_1 = alpha * powf(lambda, p - 1);
        alpha_lambda_p_1_ = alpha * powf(1.0 - lambda, p - 1);
    }
    if (is_initialized)
    {
        this->cutoff = cutoff;
        Device_Malloc_Safely((void**)&crd_with_parameters,
                             sizeof(VECTOR_LJ_SOFT_TYPE) * atom_numbers);
        Launch_Device_Kernel(
            Copy_LJ_Type_And_Mask_To_New_Crd,
            (atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, atom_numbers,
            crd_with_parameters, d_atom_LJ_type_A, d_atom_LJ_type_B,
            d_subsys_division);
        controller->printf("    Start initializing long range LJ correction\n");
        long_range_factor = 0;
        double h_factor = 0.0;
        double* d_factor = NULL;
        Device_Malloc_Safely((void**)&d_factor, sizeof(double));
        deviceMemset(d_factor, 0, sizeof(double));

        dim3 gridSize = {4, 4};
        dim3 blockSize = {32, 32};
        Launch_Device_Kernel(Total_C6_Get, gridSize, blockSize, 0, NULL,
                             atom_numbers, d_atom_LJ_type_A, d_atom_LJ_type_B,
                             d_LJ_AB, d_LJ_BB, d_factor, this->lambda);

        deviceMemcpy(&h_factor, d_factor, sizeof(double),
                     deviceMemcpyDeviceToHost);
        long_range_factor = (float)h_factor;
        Free_Single_Device_Pointer((void**)&d_factor);

        long_range_factor *=
            -2.0f / 3.0f * CONSTANT_Pi / cutoff / cutoff / cutoff / 6.0f;
        controller->printf("        long range correction factor is: %e\n",
                           long_range_factor);
        controller->printf("    End initializing long range LJ correction\n");
    }
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller->Step_Print_Initial("LJ_soft", "%.2f");
        controller->Step_Print_Initial("LJ_soft_short", "%.2f");
        controller->Step_Print_Initial("LJ_soft_long", "%.2f");
        is_controller_printf_initialized = 1;
        controller->printf("    structure last modify date is %d\n",
                           last_modify_date);
    }
    controller->printf(
        "END INITIALIZING LENNADR JONES SOFT CORE INFORMATION\n\n");
}

void LJ_SOFT_CORE::LJ_Soft_Core_Malloc()
{
    Malloc_Safely((void**)&h_LJ_energy_atom, sizeof(float) * atom_numbers);
    Malloc_Safely((void**)&h_atom_LJ_type_A, sizeof(int) * atom_numbers);
    Malloc_Safely((void**)&h_atom_LJ_type_B, sizeof(int) * atom_numbers);
    Malloc_Safely((void**)&h_LJ_AA, sizeof(float) * pair_type_numbers_A);
    Malloc_Safely((void**)&h_LJ_AB, sizeof(float) * pair_type_numbers_A);
    Malloc_Safely((void**)&h_LJ_BA, sizeof(float) * pair_type_numbers_B);
    Malloc_Safely((void**)&h_LJ_BB, sizeof(float) * pair_type_numbers_B);
    Malloc_Safely((void**)&h_subsys_division, sizeof(int) * atom_numbers);

    Device_Malloc_And_Copy_Safely((void**)&d_LJ_energy_sum, &h_LJ_energy_sum,
                                  sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_energy_atom, h_LJ_energy_atom,
                                  sizeof(float) * atom_numbers);

    Malloc_Safely((void**)&h_LJ_energy_atom_intersys,
                  sizeof(float) * atom_numbers);
    Malloc_Safely((void**)&h_LJ_energy_atom_intrasys,
                  sizeof(float) * atom_numbers);

    Device_Malloc_And_Copy_Safely((void**)&d_LJ_energy_atom_intersys,
                                  h_LJ_energy_atom_intersys,
                                  sizeof(float) * atom_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_energy_atom_intrasys,
                                  h_LJ_energy_atom_intrasys,
                                  sizeof(float) * atom_numbers);

    Device_Malloc_And_Copy_Safely((void**)&d_direct_ene_sum_intersys,
                                  &h_direct_ene_sum_intersys, sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&d_direct_ene_sum_intrasys,
                                  &h_direct_ene_sum_intrasys, sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_energy_sum_intersys,
                                  &h_LJ_energy_sum_intersys, sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_energy_sum_intrasys,
                                  &h_LJ_energy_sum_intrasys, sizeof(float));
}

void LJ_SOFT_CORE::Parameter_Host_To_Device()
{
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_AA, h_LJ_AA,
                                  sizeof(float) * pair_type_numbers_A);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_AB, h_LJ_AB,
                                  sizeof(float) * pair_type_numbers_A);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_BA, h_LJ_BA,
                                  sizeof(float) * pair_type_numbers_B);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_BB, h_LJ_BB,
                                  sizeof(float) * pair_type_numbers_B);

    Device_Malloc_And_Copy_Safely((void**)&d_atom_LJ_type_A, h_atom_LJ_type_A,
                                  sizeof(int) * atom_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_atom_LJ_type_B, h_atom_LJ_type_B,
                                  sizeof(int) * atom_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_subsys_division, h_subsys_division,
                                  sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&crd_with_LJ_parameters_local,
                         sizeof(VECTOR_LJ_SOFT_TYPE) * atom_numbers);
}

void LJ_SOFT_CORE::LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy_And_Virial(
    const int atom_numbers, const int local_atom_numbers,
    const int ghost_numbers, const VECTOR* crd, const float* charge,
    VECTOR* frc, const LTMatrix3 cell, const LTMatrix3 rcell,
    const float pme_beta, const int need_atom_energy, float* atom_energy,
    const int need_virial, LTMatrix3* atom_lj_virial,
    float* atom_direct_pme_energy)
{
    if (is_initialized)
    {
        if (clustered_neighbor_provider == NULL || clustered_workspace == NULL)
        {
            throw std::runtime_error(
                "soft-LJ requires the clustered spatial service");
        }
        CLUSTERED_SPATIAL_VIEW clustered_view = {};
        ClusteredBuildRequest request;
        request.coordinates = crd;
        request.cell = cell;
        request.reciprocal_cell = rcell;
        request.cutoff = cutoff;
        clustered_neighbor_provider->Build(request);
        Launch_Device_Kernel(
            Copy_Crd_And_Charge_To_New_Crd,
            (this->atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL,
            this->local_atom_numbers + this->ghost_numbers, crd,
            crd_with_LJ_parameters_local, charge);
        if (clustered_neighbor_provider->TotalAtomNumbers() > 0)
        {
            clustered_workspace->Gather_Soft_Core(crd_with_LJ_parameters_local,
                                                  cell, rcell);
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
                        : "clustered soft-LJ spatial view is unavailable");
            }
        }

        if (need_atom_energy)
        {
            deviceMemset(d_LJ_energy_atom, 0, sizeof(float) * atom_numbers);
            deviceMemset(atom_direct_pme_energy, 0,
                         sizeof(float) * atom_numbers);
        }

        if (atom_numbers == 0 || local_atom_numbers == 0) return;

        {
#ifdef USE_CPU
            if (clustered_view.gmxpacked_cjpacked_numbers <= 0 ||
                clustered_view.gmxpacked_sci_numbers <= 0 ||
                clustered_view.gmxpacked_exclusion_numbers <= 0 ||
                clustered_view.gmxpacked_sci == NULL ||
                clustered_view.gmxpacked_cjpacked == NULL ||
                clustered_view.gmxpacked_exclusions == NULL)
            {
                return;
            }

            auto f = Cpu_Gmxpacked_Lennard_Jones_And_Direct_Coulomb_Soft_Core<
                true, false, false, true>;

            if (need_atom_energy || need_virial)
            {
                f = Cpu_Gmxpacked_Lennard_Jones_And_Direct_Coulomb_Soft_Core<
                    true, true, true, true>;
            }
            if (clustered_workspace->direct_kernel_time_recorder != NULL)
            {
                clustered_workspace->direct_kernel_time_recorder->Start();
            }
            f(clustered_view, local_atom_numbers,
              clustered_view.sort_permutation,
              clustered_workspace->d_sorted_soft_crd, cell, rcell, d_LJ_AA,
              d_LJ_AB, d_LJ_BA, d_LJ_BB, cutoff, frc, pme_beta, atom_energy,
              atom_lj_virial, atom_direct_pme_energy, lambda, alpha, p, sigma_6,
              sigma_6_min, d_LJ_energy_atom, need_atom_energy != 0,
              need_virial != 0);
            if (clustered_workspace->direct_kernel_time_recorder != NULL)
            {
                clustered_workspace->direct_kernel_time_recorder->Stop();
            }
#else
            if (clustered_view.gmxpacked_sci_numbers <= 0 ||
                clustered_view.gmxpacked_cjpacked_numbers <= 0)
            {
                return;
            }
            if (clustered_view.gmxpacked_sci == NULL ||
                clustered_view.gmxpacked_cjpacked == NULL ||
                clustered_view.gmxpacked_exclusions == NULL ||
                clustered_workspace->d_sorted_frc == NULL)
            {
                throw std::runtime_error(
                    "clustered soft-LJ requires a gmxpacked payload");
            }
            const int sorted_force_slots = clustered_view.total_atom_numbers;
            if (clustered_workspace
                    ->gmxpacked_force_scratch_memset_time_recorder != NULL)
            {
                clustered_workspace
                    ->gmxpacked_force_scratch_memset_time_recorder->Start();
            }
            deviceMemset(clustered_workspace->d_sorted_frc, 0,
                         sizeof(VECTOR) * sorted_force_slots);
            if (clustered_workspace
                    ->gmxpacked_force_scratch_memset_time_recorder != NULL)
            {
                clustered_workspace
                    ->gmxpacked_force_scratch_memset_time_recorder->Stop();
            }

            dim3 blockSize = {static_cast<unsigned int>(kClusteredClusterSize),
                              static_cast<unsigned int>(kClusteredClusterSize),
                              1u};
            dim3 gridSize = {
                static_cast<unsigned int>(clustered_view.gmxpacked_sci_numbers),
                1u, 1u};
            auto f = Nbnxm_Gmxpacked_Lennard_Jones_And_Direct_Coulomb_Soft_Core<
                false>;
            if (need_atom_energy || need_virial)
            {
                f = Nbnxm_Gmxpacked_Lennard_Jones_And_Direct_Coulomb_Soft_Core<
                    true>;
            }
            if (clustered_workspace->direct_kernel_time_recorder != NULL)
            {
                clustered_workspace->direct_kernel_time_recorder->Start();
            }
            if (clustered_workspace->gmxpacked_kernel_launch_time_recorder !=
                NULL)
            {
                clustered_workspace->gmxpacked_kernel_launch_time_recorder
                    ->Start();
            }
            Launch_Device_Kernel(
                f, gridSize, blockSize, 0, NULL,
                clustered_view.gmxpacked_sci_numbers,
                clustered_view.cluster_numbers, local_atom_numbers,
                clustered_view.cluster_offsets,
                clustered_view.cluster_valid_masks,
                clustered_view.cluster_local_masks,
                clustered_view.super_cluster_offsets,
                clustered_view.gmxpacked_sci, clustered_view.gmxpacked_cjpacked,
                clustered_view.gmxpacked_exclusions,
                clustered_view.pair_shift_bits, clustered_view.sort_permutation,
                clustered_workspace->d_sorted_soft_crd, cell, d_LJ_AA, d_LJ_AB,
                d_LJ_BA, d_LJ_BB, cutoff, clustered_workspace->d_sorted_frc,
                pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy,
                lambda, alpha, p, sigma_6, sigma_6_min, d_LJ_energy_atom,
                need_atom_energy != 0, need_virial != 0);
            if (clustered_workspace->gmxpacked_kernel_launch_time_recorder !=
                NULL)
            {
                clustered_workspace->gmxpacked_kernel_launch_time_recorder
                    ->Stop();
            }
            if (clustered_workspace
                    ->gmxpacked_sorted_force_scatter_time_recorder != NULL)
            {
                clustered_workspace
                    ->gmxpacked_sorted_force_scatter_time_recorder->Start();
            }
            Launch_Device_Kernel(Scatter_Gmxpacked_Soft_Core_Force,
                                 (clustered_view.total_atom_numbers +
                                  CONTROLLER::device_max_thread - 1) /
                                     CONTROLLER::device_max_thread,
                                 CONTROLLER::device_max_thread, 0, NULL,
                                 clustered_view.total_atom_numbers,
                                 local_atom_numbers,
                                 clustered_view.sort_permutation,
                                 clustered_workspace->d_sorted_frc, frc);
            if (clustered_workspace
                    ->gmxpacked_sorted_force_scatter_time_recorder != NULL)
            {
                clustered_workspace
                    ->gmxpacked_sorted_force_scatter_time_recorder->Stop();
            }
            if (clustered_workspace->direct_kernel_time_recorder != NULL)
            {
                clustered_workspace->direct_kernel_time_recorder->Stop();
            }
#endif
        }
    }
}

void LJ_SOFT_CORE::Step_Print(CONTROLLER* controller)
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
    controller->Step_Print("LJ_soft_short", h_LJ_energy_sum);
    controller->Step_Print("LJ_soft_long", h_LJ_long_energy);
    controller->Step_Print("LJ_soft", h_LJ_energy_sum + h_LJ_long_energy, true);
}

static __global__ void Long_Range_Virial_Correction(LTMatrix3* d_virial,
                                                    const float factor)
{
    d_virial->a11 += factor;
    d_virial->a22 += factor;
    d_virial->a33 += factor;
}

void LJ_SOFT_CORE::Long_Range_Correction(int need_pressure, LTMatrix3* d_virial,
                                         int need_potential, float* d_potential,
                                         const float volume)
{
    if (is_initialized && CONTROLLER::PP_MPI_rank == 0)
    {
        if (need_pressure > 0)
        {
            Launch_Device_Kernel(Long_Range_Virial_Correction, 1, 1, 0, NULL,
                                 d_virial, 2 * long_range_factor / volume);
        }
        if (need_potential > 0)
        {
            Launch_Device_Kernel(device_add, 1, 1, 0, NULL, d_potential,
                                 long_range_factor / volume);
            h_LJ_long_energy = long_range_factor / volume;
        }
    }
}

static __global__ void get_local_device(
    int* atom_local, int local_atom_numbers, int ghost_numbers,
    int* d_atom_LJ_type_A, int* d_atom_LJ_type_B, int* d_mask,
    VECTOR_LJ_SOFT_TYPE* crd_with_LJ_parameters_local)
{
    SIMPLE_DEVICE_FOR(i, local_atom_numbers + ghost_numbers)
    {
        int atom_i = atom_local[i];
        crd_with_LJ_parameters_local[i].LJ_type = d_atom_LJ_type_A[atom_i];
        crd_with_LJ_parameters_local[i].LJ_type_B = d_atom_LJ_type_B[atom_i];
        crd_with_LJ_parameters_local[i].mask = d_mask[atom_i];
    }
}

void LJ_SOFT_CORE::Get_Local(int* atom_local, int local_atom_numbers,
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
                         local_atom_numbers, ghost_numbers, d_atom_LJ_type_A,
                         d_atom_LJ_type_B, d_subsys_division,
                         crd_with_LJ_parameters_local);
}
