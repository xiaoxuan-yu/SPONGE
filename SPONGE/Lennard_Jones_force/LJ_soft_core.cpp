#include "LJ_soft_core.h"

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
__global__ void Copy_Crd_To_New_Crd(const int atom_numbers, const VECTOR* crd,
                                    VECTOR_LJ_SOFT_TYPE* new_crd)
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
#endif

static __global__ void Gather_Sorted_Soft_Core_Crd(
    const int atom_numbers, const int* permutation,
    const VECTOR_LJ_SOFT_TYPE* src, VECTOR_LJ_SOFT_TYPE* dest)
{
#ifdef USE_GPU
    int sorted_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (sorted_i < atom_numbers)
#else
#pragma omp parallel for
    for (int sorted_i = 0; sorted_i < atom_numbers; sorted_i++)
#endif
    {
        dest[sorted_i] = src[permutation[sorted_i]];
    }
}

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

static __global__ void Total_C6_B_A_Get(int atom_numbers, int* atom_lj_type_A,
                                        int* atom_lj_type_B, float* d_lj_Ab,
                                        float* d_lj_Bb, double* d_factor)
{
    int j;
    double temp_sum = 0.0;
    int xA, yA, xB, yB;
    int itype_A, jtype_A, itype_B, jtype_B, atom_pair_LJ_type_A,
        atom_pair_LJ_type_B;
#ifdef USE_GPU
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers;
         i += gridDim.x * blockDim.x)
#else
#pragma omp parallel for firstprivate(                         \
        j, xA, yA, xB, yB, itype_A, jtype_A, itype_B, jtype_B, \
            atom_pair_LJ_type_A, atom_pair_LJ_type_B) reduction(+ : temp_sum)
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

            temp_small_sum +=
                d_lj_Bb[atom_pair_LJ_type_B] - d_lj_Ab[atom_pair_LJ_type_A];
        }
        temp_sum += temp_small_sum;
    }
    atomicAdd(d_factor, temp_sum);
}

template <bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb, bool need_du_dlambda>
static __global__ void Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA(
    const int atom_numbers, const int solvent_numbers, const ATOM_GROUP* nl,
    const VECTOR_LJ_SOFT_TYPE* crd, const LTMatrix3 cell, const LTMatrix3 rcell,
    const float* LJ_type_AA, const float* LJ_type_AB, const float* LJ_type_BA,
    const float* LJ_type_BB, const float cutoff, VECTOR* frc,
    const float pme_beta, float* atom_energy, LTMatrix3* atom_virial,
    float* atom_direct_cf_energy, float* atom_du_dlambda_lj,
    float* atom_du_dlambda_direct, const float lambda, const float alpha,
    const float p, const float input_sigma_6, const float input_sigma_6_min,
    float* this_energy)
{
    float lambda_ = 1.0 - lambda;
    float alpha_lambda_p = alpha * powf(lambda, p);
    float alpha_lambda__p = alpha * powf(lambda_, p);
#ifdef USE_GPU
    int atom_i = blockDim.y * blockIdx.x + threadIdx.y;
    if (atom_i < atom_numbers - solvent_numbers)
#else
#pragma omp parallel for firstprivate(lambda, alpha_lambda_p, alpha_lambda__p)
    for (int atom_i = 0; atom_i < atom_numbers - solvent_numbers; atom_i++)
#endif
    {
        ATOM_GROUP nl_i = nl[atom_i];
        VECTOR_LJ_SOFT_TYPE r1 = crd[atom_i];
        VECTOR frc_record = {0., 0., 0.};
        LTMatrix3 virial_record = {0, 0, 0, 0, 0, 0};
        float energy_lj = 0.;
        float energy_coulomb = 0.;
        float du_dlambda_lj = 0.;
        float du_dlambda_direct = 0.;
#ifdef USE_GPU
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
#else
        for (int j = 0; j < nl_i.atom_numbers; j++)
#endif
        {
            int atom_j = nl_i.atom_serial[j];
            float ij_factor = atom_j < atom_numbers ? 1.0f : 0.5f;
            VECTOR_LJ_SOFT_TYPE r2 = crd[atom_j];
            VECTOR dr = Get_Periodic_Displacement(r2, r1, cell, rcell);
            float dr_abs = norm3df(dr.x, dr.y, dr.z);
            if (dr_abs < cutoff)
            {
                int atom_pair_LJ_type_A = Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                int atom_pair_LJ_type_B =
                    Get_LJ_Type(r1.LJ_type_B, r2.LJ_type_B);
                float AA = LJ_type_AA[atom_pair_LJ_type_A];
                float AB = LJ_type_AB[atom_pair_LJ_type_A];
                float BA = LJ_type_BA[atom_pair_LJ_type_B];
                float BB = LJ_type_BB[atom_pair_LJ_type_B];
                if (BA * AA != 0 || BA + AA == 0)
                {
                    if (need_force)
                    {
                        float frc_abs =
                            lambda_ * Get_LJ_Force(r1, r2, dr_abs, AA, AB) +
                            lambda * Get_LJ_Force(r1, r2, dr_abs, BA, BB);
                        if (need_coulomb)
                        {
                            float frc_cf_abs = Get_Direct_Coulomb_Force(
                                r1, r2, dr_abs, pme_beta);
                            frc_abs = frc_abs - frc_cf_abs;
                        }
                        VECTOR frc_lin = frc_abs * dr;
                        frc_record = frc_record + frc_lin;
                        if (atom_j < atom_numbers)
                            atomicAdd(frc + atom_j, -frc_lin);
                        if (need_virial)
                        {
                            virial_record =
                                virial_record -
                                ij_factor *
                                    Get_Virial_From_Force_Dis(frc_lin, dr);
                        }
                    }
                    if (need_coulomb && need_energy)
                    {
                        energy_coulomb +=
                            ij_factor *
                            Get_Direct_Coulomb_Energy(r1, r2, dr_abs, pme_beta);
                    }
                    if (need_energy)
                    {
                        energy_lj +=
                            ij_factor *
                            (lambda_ * Get_LJ_Energy(r1, r2, dr_abs, AA, AB) +
                             lambda * Get_LJ_Energy(r1, r2, dr_abs, BA, BB));
                    }
                    if (need_du_dlambda)
                    {
                        du_dlambda_lj +=
                            ij_factor * (Get_LJ_Energy(r1, r2, dr_abs, BA, BB) -
                                         Get_LJ_Energy(r1, r2, dr_abs, AA, AB));
                        if (need_coulomb)
                        {
                            du_dlambda_direct +=
                                ij_factor * Get_Direct_Coulomb_dU_dlambda(
                                                r1, r2, dr_abs, pme_beta);
                        }
                    }
                }
                else
                {
                    float sigma_A = Get_Soft_Core_Sigma(AA, AB, input_sigma_6,
                                                        input_sigma_6_min);
                    float sigma_B = Get_Soft_Core_Sigma(BA, BB, input_sigma_6,
                                                        input_sigma_6_min);
                    float dr_softcore_A = Get_Soft_Core_Distance(
                        AA, AB, sigma_A, dr_abs, alpha, p, lambda);
                    float dr_softcore_B = Get_Soft_Core_Distance(
                        BB, BA, sigma_B, dr_abs, alpha, p, 1 - lambda);
                    if (need_force)
                    {
                        float frc_abs =
                            lambda_ * Get_Soft_Core_LJ_Force(r1, r2, dr_abs,
                                                             dr_softcore_A, AA,
                                                             AB) +
                            lambda * Get_Soft_Core_LJ_Force(
                                         r1, r2, dr_abs, dr_softcore_B, BA, BB);
                        if (need_coulomb)
                        {
                            float frc_cf_abs =
                                lambda_ * Get_Soft_Core_Direct_Coulomb_Force(
                                              r1, r2, dr_abs, dr_softcore_A,
                                              pme_beta) +
                                lambda * Get_Soft_Core_Direct_Coulomb_Force(
                                             r1, r2, dr_abs, dr_softcore_B,
                                             pme_beta);
                            frc_abs = frc_abs - frc_cf_abs;
                        }
                        VECTOR frc_lin = frc_abs * dr;
                        frc_record = frc_record + frc_lin;
                        if (atom_j < atom_numbers)
                            atomicAdd(frc + atom_j, -frc_lin);
                        if (need_virial)
                        {
                            virial_record =
                                virial_record -
                                ij_factor *
                                    Get_Virial_From_Force_Dis(frc_lin, dr);
                        }
                    }
                    if (need_coulomb && need_energy)
                    {
                        energy_coulomb +=
                            ij_factor *
                            (lambda_ * Get_Direct_Coulomb_Energy(
                                           r1, r2, dr_softcore_A, pme_beta) +
                             lambda * Get_Direct_Coulomb_Energy(
                                          r1, r2, dr_softcore_B, pme_beta));
                    }
                    if (need_energy)
                    {
                        energy_lj +=
                            ij_factor *
                            (lambda_ *
                                 Get_LJ_Energy(r1, r2, dr_softcore_A, AA, AB) +
                             lambda *
                                 Get_LJ_Energy(r1, r2, dr_softcore_B, BA, BB));
                    }
                    if (need_du_dlambda)
                    {
                        du_dlambda_lj +=
                            ij_factor *
                            (Get_LJ_Energy(r1, r2, dr_softcore_B, BA, BB) -
                             Get_LJ_Energy(r1, r2, dr_softcore_A, AA, AB));
                        du_dlambda_lj +=
                            Get_Soft_Core_dU_dlambda(
                                Get_LJ_Force(r1, r2, dr_softcore_A, AA, AB),
                                sigma_A, dr_softcore_A, alpha, p, lambda) -
                            Get_Soft_Core_dU_dlambda(
                                Get_LJ_Force(r1, r2, dr_softcore_B, BA, BB),
                                sigma_B, dr_softcore_B, alpha, p, lambda_);
                        if (need_coulomb)
                        {
                            du_dlambda_direct +=
                                ij_factor *
                                (Get_Direct_Coulomb_Energy(
                                     r1, r2, dr_softcore_B, pme_beta) -
                                 Get_Direct_Coulomb_Energy(
                                     r1, r2, dr_softcore_A, pme_beta));
                            du_dlambda_direct +=
                                ij_factor *
                                (Get_Soft_Core_dU_dlambda(
                                     Get_Direct_Coulomb_Force(
                                         r1, r2, dr_softcore_B, pme_beta),
                                     sigma_B, dr_softcore_B, alpha, p,
                                     lambda_) -
                                 Get_Soft_Core_dU_dlambda(
                                     Get_Direct_Coulomb_Force(
                                         r1, r2, dr_softcore_A, pme_beta),
                                     sigma_A, dr_softcore_A, alpha, p, lambda));
                            du_dlambda_direct +=
                                ij_factor *
                                (lambda * Get_Direct_Coulomb_dU_dlambda(
                                              r1, r2, dr_softcore_B, pme_beta) +
                                 lambda_ *
                                     Get_Direct_Coulomb_dU_dlambda(
                                         r1, r2, dr_softcore_A, pme_beta));
                        }
                    }
                }
            }
        }
        if (need_force)
        {
            Warp_Sum_To(frc + atom_i, frc_record, warpSize);
        }
        if (need_energy)
        {
            float energy_total = energy_lj;
            if (need_coulomb)
            {
                energy_total += energy_coulomb;
            }
            Warp_Sum_To(atom_energy + atom_i, energy_total, warpSize);
            Warp_Sum_To(this_energy + atom_i, energy_lj, warpSize);
        }
        if (need_coulomb && need_energy)
        {
            Warp_Sum_To(atom_direct_cf_energy + atom_i, energy_coulomb,
                        warpSize);
        }
        if (need_virial)
        {
            Warp_Sum_To(atom_virial + atom_i, virial_record, warpSize);
        }
        if (need_du_dlambda)
        {
            Warp_Sum_To(atom_du_dlambda_lj, du_dlambda_lj, warpSize);
            if (need_coulomb)
            {
                Warp_Sum_To(atom_du_dlambda_direct, du_dlambda_direct,
                            warpSize);
            }
        }
    }
}

template <bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb>
static __global__ void Clustered_Lennard_Jones_And_Direct_Coulomb_Soft_Core(
    const int sci_numbers, const int cluster_size,
    const int super_cluster_clusters, const int local_atom_numbers,
    const int* permutation, const int* cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const int* super_cluster_offsets, const int* sci_supercluster_ids,
    const int* sci_offsets, const int* cjpacked_cluster_ids,
    const unsigned int* cjpacked_imasks,
    const int* cjpacked_exclusion_indices,
    const unsigned long long* exclusion_mask_pool,
    const VECTOR_LJ_SOFT_TYPE* sorted_crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* LJ_type_AA, const float* LJ_type_AB,
    const float* LJ_type_BA, const float* LJ_type_BB, const float cutoff,
    VECTOR* frc, const float pme_beta, float* atom_energy,
    LTMatrix3* atom_lj_virial, float* atom_direct_pme_energy,
    const float lambda, const float alpha, const float p,
    const float input_sigma_6, const float input_sigma_6_min,
    float* this_energy)
{
    const float lambda_ = 1.0f - lambda;
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
            for (int lane_i = 0; lane_i < cluster_size; lane_i += 1)
            {
                if ((valid_mask_i & (1u << lane_i)) == 0u ||
                    (local_mask_i & (1u << lane_i)) == 0u)
                {
                    continue;
                }
                const int start_i = cluster_offsets[cluster_i];
                const int sorted_atom_i = start_i + lane_i;
                const int atom_i = permutation[sorted_atom_i];
                const VECTOR_LJ_SOFT_TYPE r1 = sorted_crd[sorted_atom_i];
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
                    const int i_local = cluster_i - cluster_i_start;
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
                        const int atom_j = permutation[sorted_atom_j];
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
                        const VECTOR_LJ_SOFT_TYPE r2 = sorted_crd[sorted_atom_j];
                        const VECTOR dr =
                            Get_Periodic_Displacement(r2, r1, cell, rcell);
                        const float dr2 = dr * dr;
                        if (dr2 >= cutoff * cutoff || dr2 == 0.0f)
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
                            if (need_force)
                            {
                                float frc_abs =
                                    lambda_ * Get_LJ_Force(r1, r2, dr_abs, AA, AB) +
                                    lambda *
                                        Get_LJ_Force(r1, r2, dr_abs, BA, BB);
                                if (need_coulomb)
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
                            if (need_coulomb)
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
                                BB, BA, sigma_B, dr_abs, alpha, p,
                                1.0f - lambda);
                            if (need_force)
                            {
                                float frc_abs =
                                    lambda_ * Get_Soft_Core_LJ_Force(
                                                  r1, r2, dr_abs, dr_softcore_A,
                                                  AA, AB) +
                                    lambda * Get_Soft_Core_LJ_Force(
                                                 r1, r2, dr_abs, dr_softcore_B,
                                                 BA, BB);
                                if (need_coulomb)
                                {
                                    frc_abs -= lambda_ *
                                               Get_Soft_Core_Direct_Coulomb_Force(
                                                   r1, r2, dr_abs,
                                                   dr_softcore_A, pme_beta);
                                    frc_abs -= lambda *
                                               Get_Soft_Core_Direct_Coulomb_Force(
                                                   r1, r2, dr_abs,
                                                   dr_softcore_B, pme_beta);
                                }
                                frc_lin = frc_abs * dr;
                                active_force = true;
                            }
                            pair_lj_energy =
                                ij_factor *
                                (lambda_ * Get_LJ_Energy(r1, r2, dr_softcore_A, AA,
                                                         AB) +
                                 lambda * Get_LJ_Energy(r1, r2, dr_softcore_B, BA,
                                                        BB));
                            if (need_coulomb)
                            {
                                pair_coulomb_energy =
                                    ij_factor *
                                    (lambda_ * Get_Direct_Coulomb_Energy(
                                                   r1, r2, dr_softcore_A,
                                                   pme_beta) +
                                     lambda * Get_Direct_Coulomb_Energy(
                                                  r1, r2, dr_softcore_B,
                                                  pme_beta));
                            }
                        }

                        if (need_force && active_force)
                        {
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
                            energy_lj += pair_lj_energy;
                            energy_coulomb += pair_coulomb_energy;
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
                            const int atom_j = permutation[sorted_atom_j];
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
                    atomicAdd(this_energy + atom_i, energy_lj);
                    if (need_coulomb)
                    {
                        atomicAdd(atom_direct_pme_energy + atom_i,
                                  energy_coulomb);
                    }
                }
                if (need_force)
                {
                    atomicAdd(frc + atom_i, frc_i);
                }
                if (need_virial)
                {
                    atomicAdd(atom_lj_virial + atom_i, virial);
                }
            }
        }
#else
        __shared__ VECTOR_LJ_SOFT_TYPE shared_i_atoms[max_super_cluster_atoms];
        __shared__ int shared_i_atom_ids[max_super_cluster_atoms];
        __shared__ VECTOR_LJ_SOFT_TYPE shared_j_atoms[max_cluster_size];
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
        VECTOR_LJ_SOFT_TYPE r1 = {};

        if (i_cluster_local < active_cluster_count)
        {
            cluster_i = cluster_i_start + i_cluster_local;
            if ((cluster_valid_masks[cluster_i] & (1u << i_lane)) != 0u)
            {
                const int sorted_atom_i = cluster_offsets[cluster_i] + i_lane;
                shared_i_atoms[tid] = sorted_crd[sorted_atom_i];
                shared_i_atom_ids[tid] = permutation[sorted_atom_i];
                if ((cluster_local_masks[cluster_i] & (1u << i_lane)) != 0u)
                {
                    active_i = true;
                    atom_i = shared_i_atom_ids[tid];
                    r1 = shared_i_atoms[tid];
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
            if (tid == 0)
            {
                shared_j_valid_mask = cluster_valid_masks[cluster_j];
            }
            if (tid < cluster_size)
            {
                if ((cluster_valid_masks[cluster_j] & (1u << tid)) != 0u)
                {
                    const int sorted_atom_j = cluster_offsets[cluster_j] + tid;
                    shared_j_atoms[tid] = sorted_crd[sorted_atom_j];
                    shared_j_atom_ids[tid] = permutation[sorted_atom_j];
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

            VECTOR j_force_local[max_cluster_size];
            for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
            {
                j_force_local[lane_j] = {0.0f, 0.0f, 0.0f};
            }

            if (active_i)
            {
                if ((imask & (1u << i_cluster_local)) != 0u)
                {
                    const int exclusion_index =
                        cjpacked_exclusion_indices[cj * super_cluster_clusters +
                                                   i_cluster_local];
                    const unsigned long long exclusion_mask =
                        exclusion_index >= 0 ? exclusion_mask_pool[exclusion_index]
                                             : 0ull;
                    for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
                    {
                        if ((shared_j_valid_mask & (1u << lane_j)) == 0u)
                        {
                            continue;
                        }
                        const int atom_j = shared_j_atom_ids[lane_j];
                        if (cluster_i == cluster_j && atom_j < local_atom_numbers &&
                            lane_j <= i_lane)
                        {
                            continue;
                        }
                        if ((exclusion_mask &
                             (1ull << (i_lane * cluster_size + lane_j))) != 0ull)
                        {
                            continue;
                        }
                        const VECTOR_LJ_SOFT_TYPE r2 = shared_j_atoms[lane_j];
                        const VECTOR dr =
                            Get_Periodic_Displacement(r2, r1, cell, rcell);
                        const float dr2 = dr * dr;
                        if (dr2 >= cutoff * cutoff || dr2 == 0.0f)
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
                            if (need_force)
                            {
                                float frc_abs =
                                    lambda_ * Get_LJ_Force(r1, r2, dr_abs, AA, AB) +
                                    lambda *
                                        Get_LJ_Force(r1, r2, dr_abs, BA, BB);
                                if (need_coulomb)
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
                            if (need_coulomb)
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
                                BB, BA, sigma_B, dr_abs, alpha, p,
                                1.0f - lambda);
                            if (need_force)
                            {
                                float frc_abs =
                                    lambda_ * Get_Soft_Core_LJ_Force(
                                                  r1, r2, dr_abs, dr_softcore_A,
                                                  AA, AB) +
                                    lambda * Get_Soft_Core_LJ_Force(
                                                 r1, r2, dr_abs, dr_softcore_B,
                                                 BA, BB);
                                if (need_coulomb)
                                {
                                    frc_abs -= lambda_ *
                                               Get_Soft_Core_Direct_Coulomb_Force(
                                                   r1, r2, dr_abs,
                                                   dr_softcore_A, pme_beta);
                                    frc_abs -= lambda *
                                               Get_Soft_Core_Direct_Coulomb_Force(
                                                   r1, r2, dr_abs,
                                                   dr_softcore_B, pme_beta);
                                }
                                frc_lin = frc_abs * dr;
                                active_force = true;
                            }
                            pair_lj_energy =
                                ij_factor *
                                (lambda_ * Get_LJ_Energy(r1, r2, dr_softcore_A, AA,
                                                         AB) +
                                 lambda * Get_LJ_Energy(r1, r2, dr_softcore_B, BA,
                                                        BB));
                            if (need_coulomb)
                            {
                                pair_coulomb_energy =
                                    ij_factor *
                                    (lambda_ * Get_Direct_Coulomb_Energy(
                                                   r1, r2, dr_softcore_A,
                                                   pme_beta) +
                                     lambda * Get_Direct_Coulomb_Energy(
                                                  r1, r2, dr_softcore_B,
                                                  pme_beta));
                            }
                        }

                        if (need_force && active_force)
                        {
                            frc_i = frc_i + frc_lin;
                            if (shared_j_local_flags[lane_j] != 0)
                            {
                                j_force_local[lane_j] =
                                    j_force_local[lane_j] - frc_lin;
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
                            energy_lj += pair_lj_energy;
                            energy_coulomb += pair_coulomb_energy;
                        }
                    }
                }
            }

            if (need_force)
            {
                for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
                {
                    VECTOR reduced = j_force_local[lane_j];
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
                atomicAdd(this_energy + atom_i, energy_lj);
                if (need_coulomb)
                {
                    atomicAdd(atom_direct_pme_energy + atom_i, energy_coulomb);
                }
            }
            if (need_virial)
            {
                atomicAdd(atom_lj_virial + atom_i, virial);
            }
        }
#endif
    }
}

template <bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb>
static __global__ void Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Soft_Core(
    const int sci_numbers, const int cluster_size,
    const int super_cluster_clusters, const int local_atom_numbers,
    const int* sorted_atom_ids, const int* cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const int* super_cluster_offsets, const LJ_CLUSTERED_SCI* sci_entries,
    const LJ_CLUSTERED_CJ_PACKED* cj_packed_entries,
    const unsigned long long* exclusion_mask_pool,
    const VECTOR_LJ_SOFT_TYPE* sorted_crd, const LTMatrix3 cell,
    const LTMatrix3 rcell,
    const float* LJ_type_AA, const float* LJ_type_AB, const float* LJ_type_BA,
    const float* LJ_type_BB, const float cutoff, VECTOR* frc,
    const float pme_beta, float* atom_energy, LTMatrix3* atom_lj_virial,
    float* atom_direct_pme_energy, const float lambda, const float alpha,
    const float p, const float input_sigma_6, const float input_sigma_6_min,
    float* this_energy)
{
    const float lambda_ = 1.0f - lambda;
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
        const VECTOR shift_vec =
            Clustered_Shift_Vector_From_Id(sci_entry.shift_id, cell);
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
                VECTOR_LJ_SOFT_TYPE r1 = sorted_crd[sorted_atom_i];
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
                                if (need_force)
                                {
                                    float frc_abs =
                                        lambda_ * Get_LJ_Force(r1, r2, dr_abs, AA,
                                                               AB) +
                                        lambda *
                                            Get_LJ_Force(r1, r2, dr_abs, BA, BB);
                                    if (need_coulomb)
                                    {
                                        frc_abs -= Get_Direct_Coulomb_Force(
                                            r1, r2, dr_abs, pme_beta);
                                    }
                                    frc_lin = frc_abs * dr;
                                    active_force = true;
                                }
                                pair_lj_energy =
                                    ij_factor *
                                    (lambda_ *
                                         Get_LJ_Energy(r1, r2, dr_abs, AA, AB) +
                                     lambda *
                                         Get_LJ_Energy(r1, r2, dr_abs, BA, BB));
                                if (need_coulomb)
                                {
                                    pair_coulomb_energy =
                                        ij_factor *
                                        Get_Direct_Coulomb_Energy(
                                            r1, r2, dr_abs, pme_beta);
                                }
                            }
                            else
                            {
                                const float sigma_A = Get_Soft_Core_Sigma(
                                    AA, AB, input_sigma_6, input_sigma_6_min);
                                const float sigma_B = Get_Soft_Core_Sigma(
                                    BA, BB, input_sigma_6, input_sigma_6_min);
                                const float dr_softcore_A =
                                    Get_Soft_Core_Distance(AA, AB, sigma_A, dr_abs,
                                                           alpha, p, lambda);
                                const float dr_softcore_B =
                                    Get_Soft_Core_Distance(BB, BA, sigma_B, dr_abs,
                                                           alpha, p, 1.0f - lambda);
                                if (need_force)
                                {
                                    float frc_abs =
                                        lambda_ * Get_Soft_Core_LJ_Force(
                                                      r1, r2, dr_abs,
                                                      dr_softcore_A, AA, AB) +
                                        lambda * Get_Soft_Core_LJ_Force(
                                                     r1, r2, dr_abs,
                                                     dr_softcore_B, BA, BB);
                                    if (need_coulomb)
                                    {
                                        frc_abs -=
                                            lambda_ *
                                            Get_Soft_Core_Direct_Coulomb_Force(
                                                r1, r2, dr_abs, dr_softcore_A,
                                                pme_beta);
                                        frc_abs -=
                                            lambda *
                                            Get_Soft_Core_Direct_Coulomb_Force(
                                                r1, r2, dr_abs, dr_softcore_B,
                                                pme_beta);
                                    }
                                    frc_lin = frc_abs * dr;
                                    active_force = true;
                                }
                                pair_lj_energy =
                                    ij_factor *
                                    (lambda_ * Get_LJ_Energy(r1, r2, dr_softcore_A,
                                                             AA, AB) +
                                     lambda * Get_LJ_Energy(r1, r2, dr_softcore_B,
                                                            BA, BB));
                                if (need_coulomb)
                                {
                                    pair_coulomb_energy =
                                        ij_factor *
                                        (lambda_ * Get_Direct_Coulomb_Energy(
                                                       r1, r2, dr_softcore_A,
                                                       pme_beta) +
                                         lambda * Get_Direct_Coulomb_Energy(
                                                      r1, r2, dr_softcore_B,
                                                      pme_beta));
                                }
                            }

                            if (need_force && active_force)
                            {
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
                                energy_lj += pair_lj_energy;
                                energy_coulomb += pair_coulomb_energy;
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
                    atomicAdd(this_energy + atom_i, energy_lj);
                    if (need_coulomb)
                    {
                        atomicAdd(atom_direct_pme_energy + atom_i,
                                  energy_coulomb);
                    }
                }
                if (need_virial)
                {
                    atomicAdd(atom_lj_virial + atom_i, virial);
                }
            }
        }
#else
        if constexpr (need_virial)
        {
            __shared__ VECTOR_LJ_SOFT_TYPE shared_i_atoms[max_super_cluster_atoms];
            __shared__ int shared_i_atom_ids[max_super_cluster_atoms];
            __shared__ VECTOR_LJ_SOFT_TYPE shared_j_atoms[max_cluster_size];
            __shared__ int shared_j_atom_ids[max_cluster_size];
            __shared__ int shared_j_local_flags[max_cluster_size];
            __shared__ unsigned int shared_j_valid_mask;
            __shared__ VECTOR warp_j_force[max_block_warps][max_cluster_size];

            const int i_cluster_local = tid / cluster_size;
            const int i_lane = tid % cluster_size;
            const int active_cluster_count = cluster_i_end - cluster_i_start;
            const int lane = tid & (warpSize - 1);
            const int warp_id = tid / warpSize;
            const int warp_count =
                (super_cluster_clusters * cluster_size + warpSize - 1) /
                warpSize;

            bool active_i = false;
            int cluster_i = -1;
            int atom_i = -1;
            VECTOR frc_i = {0.0f, 0.0f, 0.0f};
            float energy_lj = 0.0f;
            float energy_coulomb = 0.0f;
            LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            VECTOR_LJ_SOFT_TYPE r1 = {};

            if (i_cluster_local < active_cluster_count)
            {
                cluster_i = cluster_i_start + i_cluster_local;
                if ((cluster_valid_masks[cluster_i] & (1u << i_lane)) != 0u)
                {
                    const int sorted_atom_i = cluster_offsets[cluster_i] + i_lane;
                    shared_i_atoms[tid] = sorted_crd[sorted_atom_i];
                    shared_i_atoms[tid].crd = shared_i_atoms[tid].crd + shift_vec;
                    shared_i_atom_ids[tid] = sorted_atom_ids[sorted_atom_i];
                    if ((cluster_local_masks[cluster_i] & (1u << i_lane)) != 0u)
                    {
                        active_i = true;
                        atom_i = shared_i_atom_ids[tid];
                        r1 = shared_i_atoms[tid];
                    }
                }
            }
            __syncthreads();

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
                    if (imask == 0u)
                    {
                        continue;
                    }
                    if (tid == 0)
                    {
                        shared_j_valid_mask = cluster_valid_masks[cluster_j];
                    }
                    __syncthreads();
                    if (tid < cluster_size)
                    {
                        if ((shared_j_valid_mask & (1u << tid)) != 0u)
                        {
                            const int sorted_atom_j =
                                cluster_offsets[cluster_j] + tid;
                            shared_j_atoms[tid] = sorted_crd[sorted_atom_j];
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

                    VECTOR j_force_local[max_cluster_size] = {};
                    if (active_i && ((imask & (1u << i_cluster_local)) != 0u))
                    {
                        const int exclusion_index =
                            Clustered_First_Exclusion_Index(
                                packed, jm, i_cluster_local);
                        const unsigned long long exclusion_mask =
                            exclusion_index >= 0 ? exclusion_mask_pool[exclusion_index]
                                                 : 0ull;
                        for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
                        {
                            if ((shared_j_valid_mask & (1u << lane_j)) == 0u)
                            {
                                continue;
                            }
                            const int atom_j = shared_j_atom_ids[lane_j];
                            const bool skip_self =
                                sci_entry.shift_id == kClusteredCentralShiftId &&
                                cluster_i == cluster_j &&
                                atom_j < local_atom_numbers && lane_j <= i_lane;
                            if (skip_self ||
                                (exclusion_mask &
                                 (1ull << (i_lane * cluster_size + lane_j))) !=
                                    0ull)
                            {
                                continue;
                            }
                            const VECTOR_LJ_SOFT_TYPE r2 = shared_j_atoms[lane_j];
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
                                if (need_force)
                                {
                                    float frc_abs =
                                        lambda_ * Get_LJ_Force(r1, r2, dr_abs, AA,
                                                               AB) +
                                        lambda *
                                            Get_LJ_Force(r1, r2, dr_abs, BA, BB);
                                    if (need_coulomb)
                                    {
                                        frc_abs -= Get_Direct_Coulomb_Force(
                                            r1, r2, dr_abs, pme_beta);
                                    }
                                    frc_lin = frc_abs * dr;
                                    active_force = true;
                                }
                                pair_lj_energy =
                                    ij_factor *
                                    (lambda_ *
                                         Get_LJ_Energy(r1, r2, dr_abs, AA, AB) +
                                     lambda *
                                         Get_LJ_Energy(r1, r2, dr_abs, BA, BB));
                                if (need_coulomb)
                                {
                                    pair_coulomb_energy =
                                        ij_factor *
                                        Get_Direct_Coulomb_Energy(
                                            r1, r2, dr_abs, pme_beta);
                                }
                            }
                            else
                            {
                                const float sigma_A = Get_Soft_Core_Sigma(
                                    AA, AB, input_sigma_6, input_sigma_6_min);
                                const float sigma_B = Get_Soft_Core_Sigma(
                                    BA, BB, input_sigma_6, input_sigma_6_min);
                                const float dr_softcore_A =
                                    Get_Soft_Core_Distance(AA, AB, sigma_A, dr_abs,
                                                           alpha, p, lambda);
                                const float dr_softcore_B =
                                    Get_Soft_Core_Distance(BB, BA, sigma_B, dr_abs,
                                                           alpha, p, 1.0f - lambda);
                                if (need_force)
                                {
                                    float frc_abs =
                                        lambda_ * Get_Soft_Core_LJ_Force(
                                                      r1, r2, dr_abs,
                                                      dr_softcore_A, AA, AB) +
                                        lambda * Get_Soft_Core_LJ_Force(
                                                     r1, r2, dr_abs,
                                                     dr_softcore_B, BA, BB);
                                    if (need_coulomb)
                                    {
                                        frc_abs -=
                                            lambda_ *
                                            Get_Soft_Core_Direct_Coulomb_Force(
                                                r1, r2, dr_abs, dr_softcore_A,
                                                pme_beta);
                                        frc_abs -=
                                            lambda *
                                            Get_Soft_Core_Direct_Coulomb_Force(
                                                r1, r2, dr_abs, dr_softcore_B,
                                                pme_beta);
                                    }
                                    frc_lin = frc_abs * dr;
                                    active_force = true;
                                }
                                pair_lj_energy =
                                    ij_factor *
                                    (lambda_ * Get_LJ_Energy(r1, r2, dr_softcore_A,
                                                             AA, AB) +
                                     lambda * Get_LJ_Energy(r1, r2, dr_softcore_B,
                                                            BA, BB));
                                if (need_coulomb)
                                {
                                    pair_coulomb_energy =
                                        ij_factor *
                                        (lambda_ * Get_Direct_Coulomb_Energy(
                                                       r1, r2, dr_softcore_A,
                                                       pme_beta) +
                                         lambda * Get_Direct_Coulomb_Energy(
                                                      r1, r2, dr_softcore_B,
                                                      pme_beta));
                                }
                            }

                            if (need_force && active_force)
                            {
                                frc_i = frc_i + frc_lin;
                                if (shared_j_local_flags[lane_j] != 0)
                                {
                                    j_force_local[lane_j] =
                                        j_force_local[lane_j] - frc_lin;
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
                                energy_lj += pair_lj_energy;
                                energy_coulomb += pair_coulomb_energy;
                            }
                        }
                    }

                    if (need_force)
                    {
                        for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
                        {
                            VECTOR reduced = j_force_local[lane_j];
                            for (int delta = warpSize >> 1; delta > 0; delta >>= 1)
                            {
                                reduced.x +=
                                    deviceShflDown(FULL_MASK, reduced.x, delta,
                                                   warpSize);
                                reduced.y +=
                                    deviceShflDown(FULL_MASK, reduced.y, delta,
                                                   warpSize);
                                reduced.z +=
                                    deviceShflDown(FULL_MASK, reduced.z, delta,
                                                   warpSize);
                            }
                            if (lane == 0)
                            {
                                warp_j_force[warp_id][lane_j] = reduced;
                            }
                        }
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
                    atomicAdd(this_energy + atom_i, energy_lj);
                    if (need_coulomb)
                    {
                        atomicAdd(atom_direct_pme_energy + atom_i, energy_coulomb);
                    }
                }
                if (need_virial)
                {
                    atomicAdd(atom_lj_virial + atom_i, virial);
                }
            }
        }
        else
        {
            __shared__ VECTOR_LJ_SOFT_TYPE shared_i_atoms[max_super_cluster_atoms];
            __shared__ int shared_i_atom_ids[max_super_cluster_atoms];
            __shared__ unsigned int shared_i_valid_masks[kClusteredSuperClusterClusters];
            __shared__ unsigned int shared_i_local_masks[kClusteredSuperClusterClusters];
            __shared__ int shared_i_cluster_ids[kClusteredSuperClusterClusters];
            __shared__ VECTOR warp_i_force[max_block_warps][max_cluster_size];
            __shared__ float warp_i_energy_lj[max_block_warps][max_cluster_size];
            __shared__ float warp_i_energy_coulomb[max_block_warps][max_cluster_size];

            const int i_lane = threadIdx.x;
            const int j_lane = threadIdx.y;
            const int lane = tid & (warpSize - 1);
            const int warp_id = tid / warpSize;
            const int active_cluster_count = cluster_i_end - cluster_i_start;
            const int i_slot = j_lane * cluster_size + i_lane;

            VECTOR fci_buf[kClusteredSuperClusterClusters];
            float energy_lj_buf[kClusteredSuperClusterClusters] = {};
            float energy_coulomb_buf[kClusteredSuperClusterClusters] = {};
            for (int i_local = 0; i_local < kClusteredSuperClusterClusters;
                 i_local += 1)
            {
                fci_buf[i_local] = {0.0f, 0.0f, 0.0f};
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
                    shared_i_atoms[i_slot] = sorted_crd[sorted_atom_i];
                    shared_i_atoms[i_slot].crd =
                        shared_i_atoms[i_slot].crd + shift_vec;
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
                if (imei.imask == 0u)
                {
                    continue;
                }
                for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
                {
                    const int cluster_j = packed.cj[jm];
                    if (cluster_j < 0)
                    {
                        continue;
                    }
                    const unsigned int imask = Clustered_Jm_Imask(imei, jm);
                    if (imask == 0u)
                    {
                        continue;
                    }
                    const unsigned int valid_mask_j =
                        cluster_valid_masks[cluster_j];
                    if ((valid_mask_j & (1u << j_lane)) == 0u)
                    {
                        continue;
                    }
                    const int sorted_atom_j =
                        cluster_offsets[cluster_j] + j_lane;
                    const VECTOR_LJ_SOFT_TYPE r2 = sorted_crd[sorted_atom_j];
                    const int atom_j = sorted_atom_ids[sorted_atom_j];
                    const int atom_j_is_local = atom_j < local_atom_numbers ? 1 : 0;
                    VECTOR fcj_buf = {0.0f, 0.0f, 0.0f};
                    for (int i_local = 0; i_local < active_cluster_count;
                         i_local += 1)
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
                            Clustered_Exclusion_Index(imei, jm, i_local);
                        const unsigned long long exclusion_mask =
                            exclusion_index >= 0
                                ? exclusion_mask_pool[exclusion_index]
                                : 0ull;
                        if (sci_entry.shift_id == kClusteredCentralShiftId &&
                            cluster_i == cluster_j &&
                            atom_j < local_atom_numbers && j_lane <= i_lane)
                        {
                            continue;
                        }
                        if ((exclusion_mask &
                             (1ull << (i_lane * cluster_size + j_lane))) != 0ull)
                        {
                            continue;
                        }

                        const VECTOR_LJ_SOFT_TYPE r1 =
                            shared_i_atoms[i_local * cluster_size + i_lane];
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
                            if (need_force)
                            {
                                float frc_abs =
                                    lambda_ * Get_LJ_Force(r1, r2, dr_abs, AA, AB) +
                                    lambda *
                                        Get_LJ_Force(r1, r2, dr_abs, BA, BB);
                                if (need_coulomb)
                                {
                                    frc_abs -= Get_Direct_Coulomb_Force(
                                        r1, r2, dr_abs, pme_beta);
                                }
                                frc_lin = frc_abs * dr;
                                active_force = true;
                            }
                            pair_lj_energy =
                                ij_factor *
                                (lambda_ *
                                     Get_LJ_Energy(r1, r2, dr_abs, AA, AB) +
                                 lambda *
                                     Get_LJ_Energy(r1, r2, dr_abs, BA, BB));
                            if (need_coulomb)
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
                            const float dr_softcore_A =
                                Get_Soft_Core_Distance(AA, AB, sigma_A, dr_abs,
                                                       alpha, p, lambda);
                            const float dr_softcore_B =
                                Get_Soft_Core_Distance(BB, BA, sigma_B, dr_abs,
                                                       alpha, p, 1.0f - lambda);
                            if (need_force)
                            {
                                float frc_abs =
                                    lambda_ * Get_Soft_Core_LJ_Force(
                                                  r1, r2, dr_abs,
                                                  dr_softcore_A, AA, AB) +
                                    lambda * Get_Soft_Core_LJ_Force(
                                                 r1, r2, dr_abs,
                                                 dr_softcore_B, BA, BB);
                                if (need_coulomb)
                                {
                                    frc_abs -=
                                        lambda_ *
                                        Get_Soft_Core_Direct_Coulomb_Force(
                                            r1, r2, dr_abs, dr_softcore_A,
                                            pme_beta);
                                    frc_abs -=
                                        lambda *
                                        Get_Soft_Core_Direct_Coulomb_Force(
                                            r1, r2, dr_abs, dr_softcore_B,
                                            pme_beta);
                                }
                                frc_lin = frc_abs * dr;
                                active_force = true;
                            }
                            pair_lj_energy =
                                ij_factor *
                                (lambda_ * Get_LJ_Energy(r1, r2, dr_softcore_A,
                                                         AA, AB) +
                                 lambda * Get_LJ_Energy(r1, r2, dr_softcore_B,
                                                        BA, BB));
                            if (need_coulomb)
                            {
                                pair_coulomb_energy =
                                    ij_factor *
                                    (lambda_ * Get_Direct_Coulomb_Energy(
                                                   r1, r2, dr_softcore_A,
                                                   pme_beta) +
                                     lambda * Get_Direct_Coulomb_Energy(
                                                  r1, r2, dr_softcore_B,
                                                  pme_beta));
                            }
                        }

                        if (need_force && active_force)
                        {
                            fci_buf[i_local] = fci_buf[i_local] + frc_lin;
                            if (atom_j_is_local != 0)
                            {
                                fcj_buf = fcj_buf - frc_lin;
                            }
                        }
                        if (need_energy)
                        {
                            energy_lj_buf[i_local] += pair_lj_energy;
                            energy_coulomb_buf[i_local] += pair_coulomb_energy;
                        }
                    }

                    if (need_force && atom_j_is_local != 0)
                    {
                        VECTOR reduced = fcj_buf;
                        for (int delta = cluster_size >> 1; delta > 0;
                             delta >>= 1)
                        {
                            reduced.x += deviceShflDown(FULL_MASK, reduced.x,
                                                        delta, cluster_size);
                            reduced.y += deviceShflDown(FULL_MASK, reduced.y,
                                                        delta, cluster_size);
                                reduced.z += deviceShflDown(FULL_MASK, reduced.z,
                                                            delta, cluster_size);
                        }
                        if (i_lane == 0)
                        {
                            atomicAdd(frc + atom_j, reduced);
                        }
                    }
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
                    reduced.x +=
                        deviceShflDown(FULL_MASK, reduced.x, 16, warpSize);
                    reduced.y +=
                        deviceShflDown(FULL_MASK, reduced.y, 16, warpSize);
                    reduced.z +=
                        deviceShflDown(FULL_MASK, reduced.z, 16, warpSize);
                    reduced.x += deviceShflDown(FULL_MASK, reduced.x, 8, warpSize);
                    reduced.y += deviceShflDown(FULL_MASK, reduced.y, 8, warpSize);
                    reduced.z += deviceShflDown(FULL_MASK, reduced.z, 8, warpSize);
                    if (lane < cluster_size)
                    {
                        warp_i_force[warp_id][lane] = reduced;
                    }
                }
                if (need_energy)
                {
                    float reduced_lj = active_i ? energy_lj_buf[i_local] : 0.0f;
                    float reduced_coulomb =
                        active_i ? energy_coulomb_buf[i_local] : 0.0f;
                    reduced_lj +=
                        deviceShflDown(FULL_MASK, reduced_lj, 16, warpSize);
                    reduced_coulomb +=
                        deviceShflDown(FULL_MASK, reduced_coulomb, 16, warpSize);
                    reduced_lj +=
                        deviceShflDown(FULL_MASK, reduced_lj, 8, warpSize);
                    reduced_coulomb +=
                        deviceShflDown(FULL_MASK, reduced_coulomb, 8, warpSize);
                    if (lane < cluster_size)
                    {
                        warp_i_energy_lj[warp_id][lane] = reduced_lj;
                        warp_i_energy_coulomb[warp_id][lane] = reduced_coulomb;
                    }
                }
                __syncthreads();

                if (j_lane == 0 && active_i)
                {
                    const int atom_i =
                        shared_i_atom_ids[i_local * cluster_size + i_lane];
                    if (need_force)
                    {
                        atomicAdd(frc + atom_i,
                                  warp_i_force[0][i_lane] +
                                      warp_i_force[1][i_lane]);
                    }
                    if (need_energy)
                    {
                        const float total_energy_lj =
                            warp_i_energy_lj[0][i_lane] +
                            warp_i_energy_lj[1][i_lane];
                        const float total_energy_coulomb =
                            warp_i_energy_coulomb[0][i_lane] +
                            warp_i_energy_coulomb[1][i_lane];
                        atomicAdd(atom_energy + atom_i,
                                  total_energy_lj + total_energy_coulomb);
                        atomicAdd(this_energy + atom_i, total_energy_lj);
                        if (need_coulomb)
                        {
                            atomicAdd(atom_direct_pme_energy + atom_i,
                                      total_energy_coulomb);
                        }
                    }
                }
                __syncthreads();
            }
        }
#endif
    }
}

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
        clustered_direct_cache = Acquire_Shared_LJ_Clustered_Direct_Cache(
            controller, this->module_name, false);
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
        deviceMemset(d_factor, 0, sizeof(double));

        Launch_Device_Kernel(Total_C6_B_A_Get, gridSize, blockSize, 0, NULL,
                             atom_numbers, d_atom_LJ_type_A, d_atom_LJ_type_B,
                             d_LJ_AB, d_LJ_BB, d_factor);
        deviceMemcpy(&h_factor, d_factor, sizeof(double),
                     deviceMemcpyDeviceToHost);
        long_range_factor_TI = (float)h_factor;
        Free_Single_Device_Pointer((void**)&d_factor);

        long_range_factor *=
            -2.0f / 3.0f * CONSTANT_Pi / cutoff / cutoff / cutoff / 6.0f;
        long_range_factor_TI *=
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

    Malloc_Safely((void**)&h_sigma_of_dH_dlambda_lj, sizeof(float));
    Malloc_Safely((void**)&h_sigma_of_dH_dlambda_direct, sizeof(float));

    Device_Malloc_And_Copy_Safely((void**)&d_sigma_of_dH_dlambda_lj,
                                  h_sigma_of_dH_dlambda_lj, sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&d_sigma_of_dH_dlambda_direct,
                                  h_sigma_of_dH_dlambda_direct, sizeof(float));
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
    const int solvent_numbers, const int ghost_numbers, const VECTOR* crd,
    const float* charge, VECTOR* frc, const LTMatrix3 cell,
    const LTMatrix3 rcell, const ATOM_GROUP* nl, const float pme_beta,
    const int need_atom_energy, float* atom_energy, const int need_virial,
    LTMatrix3* atom_lj_virial, float* atom_direct_pme_energy)
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
            clustered_direct_cache->Gather_Soft_Core(
                crd_with_LJ_parameters_local);
        }

        if (need_atom_energy)
        {
            deviceMemset(d_LJ_energy_atom, 0, sizeof(float) * atom_numbers);
            deviceMemset(atom_direct_pme_energy, 0,
                         sizeof(float) * atom_numbers);
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

            auto f = Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Soft_Core<
                true, false, false, true>;

            if (!need_atom_energy && !need_virial)
            {
                f = Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Soft_Core<
                    true, false, false, true>;
            }
            else if (need_atom_energy && !need_virial)
            {
                f = Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Soft_Core<
                    true, true, false, true>;
            }
            else if (!need_atom_energy && need_virial)
            {
                f = Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Soft_Core<
                    true, false, true, true>;
            }
            else
            {
                f = Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Soft_Core<
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
                clustered_layout.d_sort_permutation,
                clustered_layout.d_cluster_offsets,
                clustered_layout.d_cluster_valid_masks,
                clustered_layout.d_cluster_local_masks,
                clustered_layout.d_super_cluster_offsets,
                clustered_layout.d_nbnxm_sci,
                clustered_layout.d_nbnxm_cjpacked,
                clustered_layout.d_exclusion_mask_pool,
                clustered_direct_cache->d_sorted_soft_crd, cell, rcell, d_LJ_AA,
                d_LJ_AB, d_LJ_BA, d_LJ_BB, cutoff, frc, pme_beta,
                atom_energy, atom_lj_virial, atom_direct_pme_energy, lambda,
                alpha, p, sigma_6, sigma_6_min, d_LJ_energy_atom);
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

            auto f = Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<
                true, false, false, true, false>;

            if (!need_atom_energy && !need_virial)
            {
                f = Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<
                    true, false, false, true, false>;
            }
            else if (need_atom_energy && !need_virial)
            {
                f = Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<
                    true, true, false, true, false>;
            }
            else if (!need_atom_energy && need_virial)
            {
                f = Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<
                    true, false, true, true, false>;
            }
            else
            {
                f = Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<
                    true, true, true, true, false>;
            }
            Launch_Device_Kernel(
                f, gridSize, blockSize, 0, NULL, local_atom_numbers,
                solvent_numbers, nl, crd_with_LJ_parameters_local, cell, rcell,
                d_LJ_AA, d_LJ_AB, d_LJ_BA, d_LJ_BB, cutoff, frc, pme_beta,
                atom_energy, atom_lj_virial, atom_direct_pme_energy, NULL,
                NULL, lambda, alpha, p, sigma_6, sigma_6_min,
                d_LJ_energy_atom);
        }
    }
}

float LJ_SOFT_CORE::Get_Partial_H_Partial_Lambda_With_Columb_Direct(
    const int solvent_numbers, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* charge, const ATOM_GROUP* nl,
    const float* charge_B_A, const float pme_beta, const int charge_perturbated)
{
    if (is_initialized)
    {
        Launch_Device_Kernel(
            Copy_Crd_And_Charge_To_New_Crd,
            (this->atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL,
            this->local_atom_numbers + this->ghost_numbers, crd,
            crd_with_parameters, charge, charge_B_A);

        deviceMemset(d_sigma_of_dH_dlambda_lj, 0, sizeof(float));

        dim3 blockSize = {
            CONTROLLER::device_warp,
            CONTROLLER::device_max_thread / CONTROLLER::device_warp};
        dim3 gridSize = (atom_numbers + blockSize.y - 1) / blockSize.y;
        auto f =
            Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<false, false, false,
                                                            true, true>;

        if (charge_perturbated > 0)
        {
            deviceMemset(d_sigma_of_dH_dlambda_direct, 0, sizeof(float));
            f = Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<
                false, false, false, true, true>;
        }
        else
        {
            f = Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<
                false, false, false, false, true>;
        }
        Launch_Device_Kernel(
            f, gridSize, blockSize, 0, NULL, local_atom_numbers,
            solvent_numbers, nl, crd_with_LJ_parameters_local, cell, rcell,
            d_LJ_AA, d_LJ_AB, d_LJ_BA, d_LJ_BB, cutoff, NULL, pme_beta, NULL,
            NULL, NULL, d_sigma_of_dH_dlambda_lj, d_sigma_of_dH_dlambda_direct,
            lambda, alpha, p, sigma_6, sigma_6_min, NULL);

        deviceMemcpy(h_sigma_of_dH_dlambda_lj, d_sigma_of_dH_dlambda_lj,
                     sizeof(float), deviceMemcpyDeviceToHost);
        deviceMemcpy(h_sigma_of_dH_dlambda_direct, d_sigma_of_dH_dlambda_direct,
                     sizeof(float), deviceMemcpyDeviceToHost);
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, h_sigma_of_dH_dlambda_lj, 1, MPI_FLOAT,
                      MPI_SUM, CONTROLLER::pp_comm);
        MPI_Allreduce(MPI_IN_PLACE, h_sigma_of_dH_dlambda_direct, 1, MPI_FLOAT,
                      MPI_SUM, CONTROLLER::pp_comm);
#endif
        return *h_sigma_of_dH_dlambda_lj +
               long_range_factor_TI / cell.a11 / cell.a22 / cell.a33;
    }
    else
    {
        return NAN;
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

void LJ_SOFT_CORE::Refresh_Clustered_Metadata(int solvent_numbers,
                                              const int* d_excluded_list_start,
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
