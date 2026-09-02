#include "bond_order.h"

#ifndef USE_CPU
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#endif

static __device__ __forceinline__ float REAXFF_Raw_Bond_Order(
    float r, int type_i, int type_j, int atom_type_numbers, const float* r_s,
    const float* r_p, const float* r_pp, const float* bo_1, const float* bo_2,
    const float* bo_3, const float* bo_4, const float* bo_5, const float* bo_6,
    const float* ro_pi, const float* ro_pi2, float bo_cut)
{
    const int idx = type_i * atom_type_numbers + type_j;
    float bo_s = 0.0f;
    if (r_s[idx] > 0.0f)
        bo_s =
            (1.0f + bo_cut) * expf(bo_1[idx] * powf(r / r_s[idx], bo_2[idx]));
    float bo_p = 0.0f;
    if (ro_pi[type_i] > 0.0f && ro_pi[type_j] > 0.0f && r_p[idx] > 0.0f)
        bo_p = expf(bo_3[idx] * powf(r / r_p[idx], bo_4[idx]));
    float bo_p2 = 0.0f;
    if (ro_pi2[type_i] > 0.0f && ro_pi2[type_j] > 0.0f && r_pp[idx] > 0.0f)
        bo_p2 = expf(bo_5[idx] * powf(r / r_pp[idx], bo_6[idx]));
    const float raw = bo_s + bo_p + bo_p2;
    return raw >= bo_cut ? raw - bo_cut : -1.0f;
}

static __device__ __forceinline__ void REAXFF_Emit_Raw_Bond(
    int atom_i, int atom_j, float r, float total_bo, float* total_bond_order,
    int* pair_i, int* pair_j, float* distances, int max_pairs, int* num_pairs)
{
    atomicAdd(total_bond_order + atom_i, total_bo);
    atomicAdd(total_bond_order + atom_j, total_bo);
    const int pos = atomicAdd(num_pairs, 1);
    if (pos < max_pairs)
    {
        pair_i[pos] = atom_i < atom_j ? atom_i : atom_j;
        pair_j[pos] = atom_i < atom_j ? atom_j : atom_i;
        distances[pos] = r;
    }
}

static __global__ void REAXFF_Gather_Clustered_Coordinates(
    int total_numbers, int cluster_numbers, const int* sort_permutation,
    const int* cluster_offsets, const VECTOR* cluster_centers,
    const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell,
    VECTOR* sorted_crd)
{
#ifdef USE_GPU
    const int sorted_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (sorted_i < total_numbers)
#else
#pragma omp parallel for
    for (int sorted_i = 0; sorted_i < total_numbers; sorted_i += 1)
#endif
    {
        int lo = 0;
        int hi = cluster_numbers;
        while (lo + 1 < hi)
        {
            const int mid = (lo + hi) >> 1;
            if (cluster_offsets[mid] <= sorted_i)
                lo = mid;
            else
                hi = mid;
        }
        const int atom_i = sort_permutation[sorted_i];
        const VECTOR center = cluster_centers[lo];
        sorted_crd[sorted_i] = center + Get_Periodic_Displacement(
                                            crd[atom_i], center, cell, rcell);
    }
}

#ifdef USE_GPU
static __global__ void Calculate_Uncorrected_Bond_Orders_Clustered_Gmxpacked(
    int sci_numbers, int packed_partitions, int cluster_numbers,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const int* super_cluster_offsets,
    const CLUSTERED_GMXPACKED_SCI* sci_entries,
    const CLUSTERED_GMXPACKED_CJ* cjpacked_entries,
    const CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    const uint64_t* pair_shift_bits, const int* sorted_atom_ids,
    const VECTOR* sorted_crd, const int* atom_type, int atom_type_numbers,
    const float* r_s, const float* r_p, const float* r_pp, const float* bo_1,
    const float* bo_2, const float* bo_3, const float* bo_4, const float* bo_5,
    const float* bo_6, const float* ro_pi, const float* ro_pi2, float cutoff,
    float bo_cut, const LTMatrix3 cell, float* total_bond_order, int* pair_i,
    int* pair_j, float* distances, int max_pairs, int* num_pairs)
{
    const int sci = blockIdx.x;
    const int partition = blockIdx.y;
    const int i_lane = threadIdx.x;
    const int j_lane = threadIdx.y;
    if (sci >= sci_numbers || i_lane >= kClusteredClusterSize ||
        j_lane >= kClusteredClusterSize)
        return;
    const CLUSTERED_GMXPACKED_SCI sci_entry = sci_entries[sci];
    const int cluster_i_begin =
        super_cluster_offsets[sci_entry.supercluster_id];
    int cluster_i_end = super_cluster_offsets[sci_entry.supercluster_id + 1];
    if (cluster_i_end > cluster_numbers) cluster_i_end = cluster_numbers;
    const int split = j_lane / kClusteredSplitJClusterSize;
    const int split_j_lane = j_lane % kClusteredSplitJClusterSize;
    const float cutoff_sq = cutoff * cutoff;
    for (int packed_idx = sci_entry.cjpacked_begin + partition;
         packed_idx < sci_entry.cjpacked_end; packed_idx += packed_partitions)
    {
        const CLUSTERED_GMXPACKED_CJ packed = cjpacked_entries[packed_idx];
        const CLUSTERED_GMXPACKED_SPLIT split_entry = packed.split[split];
        unsigned int pair_bits = 0xffffffffu;
        if (split_entry.exclusion_index != 0)
            pair_bits =
                exclusion_entries[split_entry.exclusion_index]
                    .pair[split_j_lane * kClusteredClusterSize + i_lane];
        const unsigned int effective_mask = split_entry.imask & pair_bits;
        for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
        {
            const int cluster_j = packed.cj[jm];
            if (cluster_j < 0 ||
                (cluster_valid_masks[cluster_j] & (1u << j_lane)) == 0u ||
                (cluster_local_masks[cluster_j] & (1u << j_lane)) == 0u)
                continue;
            const int sorted_j = cluster_offsets[cluster_j] + j_lane;
            const int atom_j = sorted_atom_ids[sorted_j];
            const int type_j = atom_type[atom_j];
            if (type_j < 0 || type_j >= atom_type_numbers) continue;
            const uint64_t shift_bits =
                pair_shift_bits[packed_idx * kClusteredJGroupSize + jm];
            for (int i_local = 0; i_local < cluster_i_end - cluster_i_begin;
                 i_local += 1)
            {
                const unsigned int packed_bit =
                    1u << (jm * kClusteredSuperClusterClusters + i_local);
                if ((effective_mask & packed_bit) == 0u ||
                    (Clustered_Get_Pair_Active_I_Mask(shift_bits, split) &
                     (1u << i_local)) == 0u)
                    continue;
                const int cluster_i = cluster_i_begin + i_local;
                if ((cluster_valid_masks[cluster_i] & (1u << i_lane)) == 0u ||
                    (cluster_local_masks[cluster_i] & (1u << i_lane)) == 0u)
                    continue;
                const int sorted_i = cluster_offsets[cluster_i] + i_lane;
                const int atom_i = sorted_atom_ids[sorted_i];
                if (atom_i == atom_j) continue;
                const int type_i = atom_type[atom_i];
                if (type_i < 0 || type_i >= atom_type_numbers) continue;
                const VECTOR shift = Clustered_Shift_Vector_From_Id(
                    Clustered_Get_Pair_Shift_Id(shift_bits, i_local), cell);
                const VECTOR dr =
                    (sorted_crd[sorted_i] - sorted_crd[sorted_j]) + shift;
                const float r2 = dr * dr;
                if (r2 <= 0.0001f || r2 >= cutoff_sq) continue;
                const float r = sqrtf(r2);
                const float total_bo = REAXFF_Raw_Bond_Order(
                    r, type_i, type_j, atom_type_numbers, r_s, r_p, r_pp, bo_1,
                    bo_2, bo_3, bo_4, bo_5, bo_6, ro_pi, ro_pi2, bo_cut);
                if (total_bo >= 0.0f)
                    REAXFF_Emit_Raw_Bond(atom_i, atom_j, r, total_bo,
                                         total_bond_order, pair_i, pair_j,
                                         distances, max_pairs, num_pairs);
            }
        }
    }
}
#endif

#ifdef USE_CPU
static void Calculate_Uncorrected_Bond_Orders_Clustered_Gmxpacked_CPU(
    const CLUSTERED_SPATIAL_VIEW& view, const VECTOR* sorted_crd,
    const int* atom_type, int atom_type_numbers, const float* r_s,
    const float* r_p, const float* r_pp, const float* bo_1, const float* bo_2,
    const float* bo_3, const float* bo_4, const float* bo_5, const float* bo_6,
    const float* ro_pi, const float* ro_pi2, float cutoff, float bo_cut,
    const LTMatrix3 cell, float* total_bond_order, int* pair_i, int* pair_j,
    float* distances, int max_pairs, int* num_pairs)
{
    const float cutoff_sq = cutoff * cutoff;
#pragma omp parallel for schedule(dynamic)
    for (int sci = 0; sci < view.gmxpacked_sci_numbers; sci += 1)
    {
        const CLUSTERED_GMXPACKED_SCI entry = view.gmxpacked_sci[sci];
        const int ci_begin = view.super_cluster_offsets[entry.supercluster_id];
        const int ci_end =
            view.super_cluster_offsets[entry.supercluster_id + 1];
        const int ci_numbers = ci_end - ci_begin;
        const unsigned int valid_i_cluster_mask =
            (1u << static_cast<unsigned int>(ci_numbers)) - 1u;
        const VECTOR shift =
            Clustered_Shift_Vector_From_Id(entry.shift_id, cell);
        for (int p = entry.cjpacked_begin; p < entry.cjpacked_end; p += 1)
        {
            const CLUSTERED_GMXPACKED_CJ& packed = view.gmxpacked_cjpacked[p];
            for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
            {
                const int cj = packed.cj[jm];
                if (cj < 0) continue;
                const unsigned int jm_shift = static_cast<unsigned int>(
                    jm * kClusteredSuperClusterClusters);
                unsigned int active_j_lanes =
                    view.cluster_valid_masks[cj] & view.cluster_local_masks[cj];
                while (active_j_lanes != 0u)
                {
                    const int jl = __builtin_ctz(active_j_lanes);
                    active_j_lanes &= active_j_lanes - 1u;
                    const int split = jl / kClusteredSplitJClusterSize;
                    const int split_j_lane =
                        jl - split * kClusteredSplitJClusterSize;
                    const CLUSTERED_GMXPACKED_SPLIT& split_entry =
                        packed.split[split];
                    const unsigned int active_i_cluster_mask =
                        (split_entry.imask >> jm_shift) & valid_i_cluster_mask;
                    if (active_i_cluster_mask == 0u) continue;
                    const unsigned int* exclusion_pair =
                        split_entry.exclusion_index != 0
                            ? view.gmxpacked_exclusions[split_entry
                                                            .exclusion_index]
                                      .pair +
                                  split_j_lane * kClusteredClusterSize
                            : NULL;
                    const int sj = view.cluster_offsets[cj] + jl;
                    const int atom_j = view.sort_permutation[sj];
                    const int type_j = atom_type[atom_j];
                    if (type_j < 0 || type_j >= atom_type_numbers) continue;
                    float total_bond_order_j = 0.0f;
                    for (int il = 0; il < view.cluster_size; il += 1)
                    {
                        unsigned int active_i_mask = active_i_cluster_mask;
                        if (exclusion_pair != NULL)
                            active_i_mask &= exclusion_pair[il] >> jm_shift;
                        const unsigned int i_lane_bit = 1u << il;
                        while (active_i_mask != 0u)
                        {
                            const int i_local = __builtin_ctz(active_i_mask);
                            active_i_mask &= active_i_mask - 1u;
                            const int ci = ci_begin + i_local;
                            if ((view.cluster_valid_masks[ci] & i_lane_bit) ==
                                    0u ||
                                (view.cluster_local_masks[ci] & i_lane_bit) ==
                                    0u)
                                continue;
                            const int si = view.cluster_offsets[ci] + il;
                            const int atom_i = view.sort_permutation[si];
                            if (atom_i == atom_j) continue;
                            const int type_i = atom_type[atom_i];
                            if (type_i < 0 || type_i >= atom_type_numbers)
                                continue;
                            const VECTOR dr =
                                (sorted_crd[si] - sorted_crd[sj]) + shift;
                            const float r2 = dr * dr;
                            if (r2 <= 0.0001f || r2 >= cutoff_sq) continue;
                            const float r = sqrtf(r2);
                            const float total_bo = REAXFF_Raw_Bond_Order(
                                r, type_i, type_j, atom_type_numbers, r_s, r_p,
                                r_pp, bo_1, bo_2, bo_3, bo_4, bo_5, bo_6, ro_pi,
                                ro_pi2, bo_cut);
                            if (total_bo >= 0.0f)
                            {
                                atomicAdd(total_bond_order + atom_i, total_bo);
                                total_bond_order_j += total_bo;
                                const int pos = atomicAdd(num_pairs, 1);
                                if (pos < max_pairs)
                                {
                                    pair_i[pos] =
                                        atom_i < atom_j ? atom_i : atom_j;
                                    pair_j[pos] =
                                        atom_i < atom_j ? atom_j : atom_i;
                                    distances[pos] = r;
                                }
                            }
                        }
                    }
                    if (total_bond_order_j != 0.0f)
                        atomicAdd(total_bond_order + atom_j,
                                  total_bond_order_j);
                }
            }
        }
    }
}
#endif

// Writes corrected BO and derivatives directly to sparse per-bond arrays
static __global__ void Apply_Bond_Order_Corrections_Kernel(
    int num_pairs, int* pair_i, int* pair_j, float* distances,
    const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell,
    const int* atom_type, const float* r_s, const float* r_p, const float* r_pp,
    const float* bo_1, const float* bo_2, const float* bo_3, const float* bo_4,
    const float* bo_5, const float* bo_6, const float* ro_pi,
    const float* ro_pi2, const float* valency, const float* valency_val,
    const float* ovc, const float* v13cor, const float* p_boc3,
    const float* p_boc4, const float* p_boc5, const int atom_type_numbers,
    const int atom_numbers, float gp_boc1, float gp_boc2, float bo_cut,
    const float* total_bond_order, float* corrected_bo_s,
    float* corrected_bo_pi, float* corrected_bo_pi2, float* dbo_s_dr,
    float* dbo_pi_dr, float* dbo_pi2_dr, float* dbo_s_dDelta_i,
    float* dbo_pi_dDelta_i, float* dbo_pi2_dDelta_i, float* dbo_s_dDelta_j,
    float* dbo_pi_dDelta_j, float* dbo_pi2_dDelta_j, float* dbo_raw_total_dr)
{
    SIMPLE_DEVICE_FOR(idx, num_pairs)
    {
        int i = pair_i[idx];
        int j = pair_j[idx];
        float r = distances[idx];

        int type_i = atom_type[i];
        int type_j = atom_type[j];

        if (type_i >= 0 && type_i < atom_type_numbers && type_j >= 0 &&
            type_j < atom_type_numbers)
        {
            int pair_idx = type_i * atom_type_numbers + type_j;

            float ros = r_s[pair_idx];
            float bo_s_raw_val = 0.0f, dbo_s_raw_dr = 0.0f;
            if (ros > 0.0f)
            {
                float ratio = r / ros;
                float pow_ratio = powf(ratio, bo_2[pair_idx]);
                bo_s_raw_val =
                    (1.0f + bo_cut) * expf(bo_1[pair_idx] * pow_ratio);
                dbo_s_raw_dr = bo_s_raw_val * bo_1[pair_idx] * bo_2[pair_idx] *
                               powf(ratio, bo_2[pair_idx] - 1.0f) *
                               (1.0f / ros);
            }

            float bo_p_val = 0.0f, dbo_p_raw_dr = 0.0f;
            if (ro_pi[type_i] > 0.0f && ro_pi[type_j] > 0.0f)
            {
                float rop = r_p[pair_idx];
                if (rop > 0.0f)
                {
                    float ratio = r / rop;
                    float pow_ratio = powf(ratio, bo_4[pair_idx]);
                    bo_p_val = expf(bo_3[pair_idx] * pow_ratio);
                    dbo_p_raw_dr = bo_p_val * bo_3[pair_idx] * bo_4[pair_idx] *
                                   powf(ratio, bo_4[pair_idx] - 1.0f) *
                                   (1.0f / rop);
                }
            }

            float bo_p2_val = 0.0f, dbo_p2_raw_dr = 0.0f;
            if (ro_pi2[type_i] > 0.0f && ro_pi2[type_j] > 0.0f)
            {
                float rop2 = r_pp[pair_idx];
                if (rop2 > 0.0f)
                {
                    float ratio = r / rop2;
                    float pow_ratio = powf(ratio, bo_6[pair_idx]);
                    bo_p2_val = expf(bo_5[pair_idx] * pow_ratio);
                    dbo_p2_raw_dr =
                        bo_p2_val * bo_5[pair_idx] * bo_6[pair_idx] *
                        powf(ratio, bo_6[pair_idx] - 1.0f) * (1.0f / rop2);
                }
            }

            float total_bo_raw = bo_s_raw_val + bo_p_val + bo_p2_val;
            float dbo_raw_total_dr_val =
                dbo_s_raw_dr + dbo_p_raw_dr + dbo_p2_raw_dr;

            if (total_bo_raw >= bo_cut)
            {
                SADfloat<5> bo_s_raw(bo_s_raw_val, 0);
                SADfloat<5> bo_p(bo_p_val, 1);
                SADfloat<5> bo_p2(bo_p2_val, 2);
                SADfloat<5> Delta_i(total_bond_order[i], 3);
                SADfloat<5> Delta_j(total_bond_order[j], 4);

                SADfloat<5> total_bo_orig = (bo_s_raw + bo_p + bo_p2) - bo_cut;
                SADfloat<5> bo_s = bo_s_raw - bo_cut;
                if (bo_s.val < 0) bo_s = SADfloat<5>(0.0f);

                float ovc_val = ovc[pair_idx];
                float v13cor_val = v13cor[pair_idx];

                SADfloat<5> f1(1.0f);
                if (ovc_val >= 0.001f)
                {
                    SADfloat<5> Deltap_i = Delta_i - valency[type_i];
                    SADfloat<5> Deltap_j = Delta_j - valency[type_j];

                    SADfloat<5> exp_p1i = expf(-gp_boc1 * Deltap_i);
                    SADfloat<5> exp_p1j = expf(-gp_boc1 * Deltap_j);
                    SADfloat<5> f2 = exp_p1i + exp_p1j;

                    SADfloat<5> f3 =
                        -1.0f / gp_boc2 *
                        (Log_Sum_Exp(-gp_boc2 * Deltap_i, -gp_boc2 * Deltap_j) -
                         0.6931471805599453f);

                    float val_i = valency[type_i];
                    float val_j = valency[type_j];

                    f1 = 0.5f * ((val_i + f2) / (val_i + f2 + f3) +
                                 (val_j + f2) / (val_j + f2 + f3));
                }

                SADfloat<5> f4(1.0f), f5(1.0f);
                if (v13cor_val >= 0.001f)
                {
                    SADfloat<5> Deltap_boc_i = Delta_i - valency_val[type_i];
                    SADfloat<5> Deltap_boc_j = Delta_j - valency_val[type_j];

                    float p_boc3_val = p_boc3[pair_idx];
                    float p_boc4_val = p_boc4[pair_idx];
                    float p_boc5_val = p_boc5[pair_idx];

                    SADfloat<5> exp_f4 =
                        expf(-(p_boc4_val * total_bo_orig * total_bo_orig -
                               Deltap_boc_i) *
                                 p_boc3_val +
                             p_boc5_val);
                    SADfloat<5> exp_f5 =
                        expf(-(p_boc4_val * total_bo_orig * total_bo_orig -
                               Deltap_boc_j) *
                                 p_boc3_val +
                             p_boc5_val);

                    f4 = 1.0f / (1.0f + exp_f4);
                    f5 = 1.0f / (1.0f + exp_f5);
                }

                SADfloat<5> A0 = f1 * f4 * f5;

                SADfloat<5> s_corrected_bo_pi = bo_p * A0 * f1;
                SADfloat<5> s_corrected_bo_pi2 = bo_p2 * A0 * f1;
                SADfloat<5> s_corrected_bo_s =
                    total_bo_orig * A0 -
                    (s_corrected_bo_pi + s_corrected_bo_pi2);
                if (s_corrected_bo_s.val < 0)
                    s_corrected_bo_s = SADfloat<5>(0.0f);

                // Write to sparse per-bond arrays (one entry per bond)
                corrected_bo_s[idx] = s_corrected_bo_s.val;
                corrected_bo_pi[idx] = s_corrected_bo_pi.val;
                corrected_bo_pi2[idx] = s_corrected_bo_pi2.val;

                dbo_s_dr[idx] = s_corrected_bo_s.dval[0] * dbo_s_raw_dr +
                                s_corrected_bo_s.dval[1] * dbo_p_raw_dr +
                                s_corrected_bo_s.dval[2] * dbo_p2_raw_dr;

                dbo_pi_dr[idx] = s_corrected_bo_pi.dval[0] * dbo_s_raw_dr +
                                 s_corrected_bo_pi.dval[1] * dbo_p_raw_dr +
                                 s_corrected_bo_pi.dval[2] * dbo_p2_raw_dr;

                dbo_pi2_dr[idx] = s_corrected_bo_pi2.dval[0] * dbo_s_raw_dr +
                                  s_corrected_bo_pi2.dval[1] * dbo_p_raw_dr +
                                  s_corrected_bo_pi2.dval[2] * dbo_p2_raw_dr;

                // dDelta_i: derivative w.r.t. Delta of pair_i[idx]
                dbo_s_dDelta_i[idx] = s_corrected_bo_s.dval[3];
                dbo_pi_dDelta_i[idx] = s_corrected_bo_pi.dval[3];
                dbo_pi2_dDelta_i[idx] = s_corrected_bo_pi2.dval[3];

                // dDelta_j: derivative w.r.t. Delta of pair_j[idx]
                dbo_s_dDelta_j[idx] = s_corrected_bo_s.dval[4];
                dbo_pi_dDelta_j[idx] = s_corrected_bo_pi.dval[4];
                dbo_pi2_dDelta_j[idx] = s_corrected_bo_pi2.dval[4];

                dbo_raw_total_dr[idx] = dbo_raw_total_dr_val;
            }
            else
            {
                corrected_bo_s[idx] = 0.0f;
                corrected_bo_pi[idx] = 0.0f;
                corrected_bo_pi2[idx] = 0.0f;

                dbo_s_dr[idx] = 0.0f;
                dbo_pi_dr[idx] = 0.0f;
                dbo_pi2_dr[idx] = 0.0f;

                dbo_s_dDelta_i[idx] = 0.0f;
                dbo_pi_dDelta_i[idx] = 0.0f;
                dbo_pi2_dDelta_i[idx] = 0.0f;

                dbo_s_dDelta_j[idx] = 0.0f;
                dbo_pi_dDelta_j[idx] = 0.0f;
                dbo_pi2_dDelta_j[idx] = 0.0f;

                dbo_raw_total_dr[idx] = 0.0f;
            }
        }
    }
}

// Reduce corrected bond orders per atom using CSR bond list
static __global__ void Reduce_Total_Corrected_Bond_Order_Kernel(
    int atom_numbers, const int* bond_count, const int* bond_offset,
    const int* bond_idx, const float* bo_s, const float* bo_pi,
    const float* bo_pi2, float* total_bo)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        float sum = 0.0f;
        int start = bond_offset[i];
        int end = start + bond_count[i];
        for (int k = start; k < end; k++)
        {
            int b = bond_idx[k];
            sum += bo_s[b] + bo_pi[b] + bo_pi2[b];
        }
        total_bo[i] = sum;
    }
}

// --- CSR build kernels ---
static __global__ void Count_Bonds_Per_Atom_Kernel(int num_bonds,
                                                   const int* bond_i,
                                                   const int* bond_j,
                                                   int* bond_count)
{
    SIMPLE_DEVICE_FOR(b, num_bonds)
    {
        atomicAdd(&bond_count[bond_i[b]], 1);
        atomicAdd(&bond_count[bond_j[b]], 1);
    }
}

#ifdef USE_CPU
static __global__ void Exclusive_Prefix_Sum_Kernel(int n, const int* input,
                                                   int* output)
{
    // Simple sequential prefix sum (launched with 1 thread)
    SIMPLE_DEVICE_FOR(dummy, 1)
    {
        output[0] = 0;
        for (int i = 0; i < n; i++)
        {
            output[i + 1] = output[i] + input[i];
        }
    }
}
#endif

static __global__ void Fill_Bond_CSR_Kernel(int num_bonds, const int* bond_i,
                                            const int* bond_j,
                                            const int* bond_offset,
                                            int* fill_count, int* bond_nbr,
                                            int* bond_idx)
{
    SIMPLE_DEVICE_FOR(b, num_bonds)
    {
        int i = bond_i[b];
        int j = bond_j[b];
        // Entry for atom i: neighbor is j
        int pos_i = bond_offset[i] + atomicAdd(&fill_count[i], 1);
        bond_nbr[pos_i] = j;
        bond_idx[pos_i] = b;
        // Entry for atom j: neighbor is i
        int pos_j = bond_offset[j] + atomicAdd(&fill_count[j], 1);
        bond_nbr[pos_j] = i;
        bond_idx[pos_j] = b;
    }
}

// --- Force projection kernels (sparse) ---

// In sparse mode, dE_dBO is accumulated to a single bond index by all
// consumers, so no need to sum [i*N+j] + [j*N+i].
static __global__ void Calculate_CdDelta_Prime_Kernel(
    int num_pairs, const int* pair_i, const int* pair_j, const float* dE_dBO_s,
    const float* dE_dBO_pi, const float* dE_dBO_pi2, const float* CdDelta,
    const float* dbo_s_dDelta_i, const float* dbo_pi_dDelta_i,
    const float* dbo_pi2_dDelta_i, const float* dbo_s_dDelta_j,
    const float* dbo_pi_dDelta_j, const float* dbo_pi2_dDelta_j,
    float* CdDelta_prime)
{
    SIMPLE_DEVICE_FOR(idx, num_pairs)
    {
        int i = pair_i[idx];
        int j = pair_j[idx];

        float de_dbo_s_total = dE_dBO_s[idx];
        float de_dbo_pi_total = dE_dBO_pi[idx];
        float de_dbo_pi2_total = dE_dBO_pi2[idx];

        float eff_cdd = CdDelta[i] + CdDelta[j];

        float term_i = (de_dbo_s_total + eff_cdd) * dbo_s_dDelta_i[idx] +
                       (de_dbo_pi_total + eff_cdd) * dbo_pi_dDelta_i[idx] +
                       (de_dbo_pi2_total + eff_cdd) * dbo_pi2_dDelta_i[idx];
        atomicAdd(&CdDelta_prime[i], term_i);

        float term_j = (de_dbo_s_total + eff_cdd) * dbo_s_dDelta_j[idx] +
                       (de_dbo_pi_total + eff_cdd) * dbo_pi_dDelta_j[idx] +
                       (de_dbo_pi2_total + eff_cdd) * dbo_pi2_dDelta_j[idx];
        atomicAdd(&CdDelta_prime[j], term_j);
    }
}

static __global__ void REAXFF_Force_Projection_Kernel(
    int num_pairs, const int* pair_i, const int* pair_j, const float* distances,
    const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell,
    const float* dE_dBO_s, const float* dE_dBO_pi, const float* dE_dBO_pi2,
    const float* CdDelta, const float* dbo_s_dr, const float* dbo_pi_dr,
    const float* dbo_pi2_dr, const float* dbo_raw_total_dr,
    const float* CdDelta_prime, VECTOR* frc, LTMatrix3* atom_virial)
{
    SIMPLE_DEVICE_FOR(idx, num_pairs)
    {
        int i = pair_i[idx];
        int j = pair_j[idx];
        float r_val = distances[idx];
        if (r_val >= 0.0001f)
        {
            float de_dbo_s_total = dE_dBO_s[idx];
            float de_dbo_pi_total = dE_dBO_pi[idx];
            float de_dbo_pi2_total = dE_dBO_pi2[idx];

            float eff_cdd = CdDelta[i] + CdDelta[j];

            float de_dr = (de_dbo_s_total + eff_cdd) * dbo_s_dr[idx] +
                          (de_dbo_pi_total + eff_cdd) * dbo_pi_dr[idx] +
                          (de_dbo_pi2_total + eff_cdd) * dbo_pi2_dr[idx];

            de_dr +=
                (CdDelta_prime[i] + CdDelta_prime[j]) * dbo_raw_total_dr[idx];

            float force_mag = -de_dr;

            VECTOR ri = crd[i];
            VECTOR rj = crd[j];
            VECTOR drij = Get_Periodic_Displacement(ri, rj, cell, rcell);

            float fx = force_mag * drij.x / r_val;
            float fy = force_mag * drij.y / r_val;
            float fz = force_mag * drij.z / r_val;

            atomicAdd(&frc[i].x, fx);
            atomicAdd(&frc[i].y, fy);
            atomicAdd(&frc[i].z, fz);
            atomicAdd(&frc[j].x, -fx);
            atomicAdd(&frc[j].y, -fy);
            atomicAdd(&frc[j].z, -fz);

            if (atom_virial)
            {
                VECTOR fij = {fx, fy, fz};
                atomicAdd(atom_virial + i,
                          Get_Virial_From_Force_Dis(fij, drij));
            }
        }
    }
}

// ============================================================
// Implementation
// ============================================================

void REAXFF_BOND_ORDER::Initial(CONTROLLER* controller, int atom_numbers,
                                const char* parameter_in_file,
                                const char* type_in_file)
{
    if (parameter_in_file == NULL || type_in_file == NULL)
    {
        return;
    }

    this->atom_numbers = atom_numbers;
    controller->printf("START INITIALIZING REAXFF_BOND_ORDER\n");

    FILE* fp_p;
    Open_File_Safely(&fp_p, parameter_in_file, "r");
    char line[1024];
    auto throw_bad_format = [&](const char* file_name, const char* reason)
    {
        char error_msg[1024];
        sprintf(error_msg, "Reason:\n\t%s in file %s\n", reason, file_name);
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "REAXFF_BOND_ORDER::Initial", error_msg);
    };
    auto read_line_or_throw =
        [&](FILE* file, const char* file_name, const char* stage)
    {
        if (fgets(line, 1024, file) == NULL)
        {
            char reason[512];
            sprintf(reason, "failed to read %s", stage);
            throw_bad_format(file_name, reason);
        }
    };

    read_line_or_throw(fp_p, parameter_in_file, "parameter header line 1");
    read_line_or_throw(fp_p, parameter_in_file, "general parameter count line");
    int n_gen_params = 0;
    if (sscanf(line, "%d", &n_gen_params) != 1 || n_gen_params < 2)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of general parameters");
    }

    std::vector<float> gen_params(n_gen_params, 0.0f);
    for (int i = 0; i < n_gen_params; i++)
    {
        read_line_or_throw(fp_p, parameter_in_file, "general parameter block");
        if (sscanf(line, "%f", &gen_params[i]) != 1)
        {
            char reason[512];
            sprintf(reason, "failed to parse general parameter at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
    }
    gp_boc1 = gen_params[0];
    gp_boc2 = gen_params[1];
    if (n_gen_params > 29) gp_bo_cut = 0.01f * gen_params[29];
    read_line_or_throw(fp_p, parameter_in_file, "atom type count line");
    int n_atom_types = 0;
    if (sscanf(line, "%d", &n_atom_types) != 1 || n_atom_types <= 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of atom types");
    }
    this->atom_type_numbers = n_atom_types;
    read_line_or_throw(fp_p, parameter_in_file, "atom type header line 1");
    read_line_or_throw(fp_p, parameter_in_file, "atom type header line 2");
    read_line_or_throw(fp_p, parameter_in_file, "atom type header line 3");

    std::map<std::string, int> type_map;
    Malloc_Safely((void**)&h_ro_sigma, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_ro_pi, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_ro_pi2, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_valency, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_valency_val, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_b_o_131, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_b_o_132, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_b_o_133, sizeof(float) * n_atom_types);

    for (int i = 0; i < n_atom_types; i++)
    {
        read_line_or_throw(fp_p, parameter_in_file,
                           "atom type parameter line 1");
        char element_name[16];
        float ro_sigma, valency, mass, r_vdw, epsilon, gamma, ro_pi, valency_e;
        if (sscanf(line, "%s %f %f %f %f %f %f %f %f", element_name, &ro_sigma,
                   &valency, &mass, &r_vdw, &epsilon, &gamma, &ro_pi,
                   &valency_e) != 9)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 1 for type index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        type_map[std::string(element_name)] = i;
        h_ro_sigma[i] = ro_sigma;
        h_ro_pi[i] = ro_pi;
        h_valency[i] = valency;

        read_line_or_throw(fp_p, parameter_in_file,
                           "atom type parameter line 2");

        read_line_or_throw(fp_p, parameter_in_file,
                           "atom type parameter line 3");
        float ro_pi_pi, p_lp2, heat_inc, boc4_i, boc3_i, boc5_i;
        if (sscanf(line, "%f %f %f %f %f %f", &ro_pi_pi, &p_lp2, &heat_inc,
                   &boc4_i, &boc3_i, &boc5_i) != 6)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 3 for type index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        h_ro_pi2[i] = ro_pi_pi;
        h_b_o_131[i] = boc4_i;
        h_b_o_132[i] = boc3_i;
        h_b_o_133[i] = boc5_i;

        read_line_or_throw(fp_p, parameter_in_file,
                           "atom type parameter line 4");
        float p_ovun2, p_val3, unused, valency_val, p_val5;
        if (sscanf(line, "%f %f %f %f %f", &p_ovun2, &p_val3, &unused,
                   &valency_val, &p_val5) != 5)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 4 for type index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        h_valency_val[i] = valency_val;
    }
    read_line_or_throw(fp_p, parameter_in_file, "bond parameter count line");
    int n_bond_params = 0;
    if (sscanf(line, "%d", &n_bond_params) != 1 || n_bond_params < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of bond parameters");
    }
    Malloc_Safely((void**)&h_bo_1, sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_bo_2, sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_bo_3, sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_bo_4, sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_bo_5, sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_bo_6, sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_ovc, sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_v13cor,
                  sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_p_boc3,
                  sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_p_boc4,
                  sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_p_boc5,
                  sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_r_s, sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_r_p, sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_r_pp, sizeof(float) * n_atom_types * n_atom_types);

    for (int i = 0; i < n_atom_types * n_atom_types; i++)
    {
        h_bo_1[i] = 0.0f;
        h_bo_2[i] = 0.0f;
        h_bo_3[i] = 0.0f;
        h_bo_4[i] = 0.0f;
        h_bo_5[i] = 0.0f;
        h_bo_6[i] = 0.0f;
        h_ovc[i] = 0.0f;
        h_v13cor[i] = 0.0f;
        h_p_boc3[i] = 0.0f;
        h_p_boc4[i] = 0.0f;
        h_p_boc5[i] = 0.0f;
        h_r_s[i] = 0.0f;
        h_r_p[i] = 0.0f;
        h_r_pp[i] = 0.0f;
    }

    for (int i = 0; i < n_atom_types; i++)
    {
        for (int j = 0; j < n_atom_types; j++)
        {
            int idx = i * n_atom_types + j;
            h_p_boc3[idx] = sqrtf(h_b_o_132[i] * h_b_o_132[j]);
            h_p_boc4[idx] = sqrtf(h_b_o_131[i] * h_b_o_131[j]);
            h_p_boc5[idx] = sqrtf(h_b_o_133[i] * h_b_o_133[j]);
            h_r_s[idx] = 0.5f * (h_ro_sigma[i] + h_ro_sigma[j]);
            h_r_p[idx] = 0.5f * (h_ro_pi[i] + h_ro_pi[j]);
            h_r_pp[idx] = 0.5f * (h_ro_pi2[i] + h_ro_pi2[j]);
        }
    }

    read_line_or_throw(fp_p, parameter_in_file, "bond parameter header line");

    for (int i = 0; i < n_bond_params; i++)
    {
        read_line_or_throw(fp_p, parameter_in_file, "bond parameter line 1");
        int t1, t2;
        float De_s, De_p, De_pp, p_be1, p_bo5_val, v13cor_val, p_bo6_val,
            p_ovun1;
        if (sscanf(line, "%d %d %f %f %f %f %f %f %f %f", &t1, &t2, &De_s,
                   &De_p, &De_pp, &p_be1, &p_bo5_val, &v13cor_val, &p_bo6_val,
                   &p_ovun1) != 10)
        {
            char reason[512];
            sprintf(reason, "failed to parse bond parameter line 1 at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }

        int idx1 = t1 - 1;
        int idx2 = t2 - 1;

        if (idx1 < 0 || idx1 >= n_atom_types || idx2 < 0 ||
            idx2 >= n_atom_types)
        {
            char reason[512];
            sprintf(reason,
                    "invalid bond type indices %d %d (max atom type %d)", t1,
                    t2, n_atom_types);
            throw_bad_format(parameter_in_file, reason);
        }

        read_line_or_throw(fp_p, parameter_in_file, "bond parameter line 2");
        float p_be2, p_bo3_val, p_bo4_val, unused1, p_bo1_val, p_bo2_val,
            ovc_val;
        if (sscanf(line, "%f %f %f %f %f %f %f", &p_be2, &p_bo3_val, &p_bo4_val,
                   &unused1, &p_bo1_val, &p_bo2_val, &ovc_val) != 7)
        {
            char reason[512];
            sprintf(reason, "failed to parse bond parameter line 2 at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }

        h_bo_1[idx1 * n_atom_types + idx2] =
            h_bo_1[idx2 * n_atom_types + idx1] = p_bo1_val;
        h_bo_2[idx1 * n_atom_types + idx2] =
            h_bo_2[idx2 * n_atom_types + idx1] = p_bo2_val;
        h_bo_3[idx1 * n_atom_types + idx2] =
            h_bo_3[idx2 * n_atom_types + idx1] = p_bo3_val;
        h_bo_4[idx1 * n_atom_types + idx2] =
            h_bo_4[idx2 * n_atom_types + idx1] = p_bo4_val;
        h_bo_5[idx1 * n_atom_types + idx2] =
            h_bo_5[idx2 * n_atom_types + idx1] = p_bo5_val;
        h_bo_6[idx1 * n_atom_types + idx2] =
            h_bo_6[idx2 * n_atom_types + idx1] = p_bo6_val;

        h_ovc[idx1 * n_atom_types + idx2] = h_ovc[idx2 * n_atom_types + idx1] =
            ovc_val;
        h_v13cor[idx1 * n_atom_types + idx2] =
            h_v13cor[idx2 * n_atom_types + idx1] = v13cor_val;
    }

    if (fgets(line, 1024, fp_p) != NULL)
    {
        int n_off = 0;
        if (sscanf(line, "%d", &n_off) != 1 || n_off < 0)
        {
            throw_bad_format(parameter_in_file,
                             "failed to parse number of off-diagonal terms");
        }
        for (int off = 0; off < n_off; off++)
        {
            read_line_or_throw(fp_p, parameter_in_file,
                               "off-diagonal parameter line");
            int t1, t2;
            float dij = 0.0f, rvdw = 0.0f, alfa = 0.0f;
            float ro_sigma_od = -1.0f, ro_pi_od = -1.0f, ro_pipi_od = -1.0f;
            int read_cnt =
                sscanf(line, "%d %d %f %f %f %f %f %f", &t1, &t2, &dij, &rvdw,
                       &alfa, &ro_sigma_od, &ro_pi_od, &ro_pipi_od);
            if (read_cnt < 5)
            {
                char reason[512];
                sprintf(reason,
                        "failed to parse off-diagonal parameters at index %d",
                        off + 1);
                throw_bad_format(parameter_in_file, reason);
            }
            int idx1 = t1 - 1;
            int idx2 = t2 - 1;
            if (idx1 < 0 || idx1 >= n_atom_types || idx2 < 0 ||
                idx2 >= n_atom_types)
            {
                char reason[512];
                sprintf(reason,
                        "invalid off-diagonal type indices %d %d at index %d",
                        t1, t2, off + 1);
                throw_bad_format(parameter_in_file, reason);
            }
            int pair_idx = idx1 * n_atom_types + idx2;
            if (ro_sigma_od > 0.0f)
            {
                h_r_s[pair_idx] = h_r_s[idx2 * n_atom_types + idx1] =
                    ro_sigma_od;
            }
            if (ro_pi_od > 0.0f)
            {
                h_r_p[pair_idx] = h_r_p[idx2 * n_atom_types + idx1] = ro_pi_od;
            }
            if (ro_pipi_od > 0.0f)
            {
                h_r_pp[pair_idx] = h_r_pp[idx2 * n_atom_types + idx1] =
                    ro_pipi_od;
            }
        }
    }
    fclose(fp_p);

    FILE* fp_t;
    Open_File_Safely(&fp_t, type_in_file, "r");
    int check_atom_numbers = 0;
    read_line_or_throw(fp_t, type_in_file, "atom number line");
    if (sscanf(line, "%d", &check_atom_numbers) != 1)
    {
        throw_bad_format(type_in_file, "failed to parse atom numbers");
    }
    if (check_atom_numbers != atom_numbers)
    {
        char reason[512];
        sprintf(reason, "atom numbers (%d) does not match system (%d)",
                check_atom_numbers, atom_numbers);
        throw_bad_format(type_in_file, reason);
    }
    Malloc_Safely((void**)&h_atom_type, sizeof(int) * atom_numbers);
    for (int i = 0; i < atom_numbers; i++)
    {
        char type_name[16];
        read_line_or_throw(fp_t, type_in_file, "atom type entry line");
        if (sscanf(line, "%s", type_name) != 1)
        {
            char reason[512];
            sprintf(reason, "failed to parse atom type at index %d", i + 1);
            throw_bad_format(type_in_file, reason);
        }
        std::string type_str(type_name);
        if (type_map.find(type_str) != type_map.end())
        {
            h_atom_type[i] = type_map[type_str];
        }
        else
        {
            char reason[512];
            sprintf(reason, "atom type %s not found in parameter file %s",
                    type_name, parameter_in_file);
            throw_bad_format(type_in_file, reason);
        }
    }
    fclose(fp_t);

    Device_Malloc_And_Copy_Safely((void**)&d_ro_sigma, h_ro_sigma,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_ro_pi, h_ro_pi,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_ro_pi2, h_ro_pi2,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_bo_1, h_bo_1,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_bo_2, h_bo_2,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_bo_3, h_bo_3,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_bo_4, h_bo_4,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_bo_5, h_bo_5,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_bo_6, h_bo_6,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_r_s, h_r_s,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_r_p, h_r_p,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_r_pp, h_r_pp,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_valency, h_valency,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_valency_val, h_valency_val,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_ovc, h_ovc,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_v13cor, h_v13cor,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_p_boc3, h_p_boc3,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_p_boc4, h_p_boc4,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_p_boc5, h_p_boc5,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_atom_type, h_atom_type,
                                  sizeof(int) * atom_numbers);

    // Sparse bond storage: allocate for max_bonds = atom_numbers * 32
    // (supports up to 64 bonds per atom on average)
    max_bonds = atom_numbers * 32;

    // Per-atom arrays
    Device_Malloc_Safely((void**)&d_total_bond_order,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_total_corrected_bond_order,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_CdDelta_prime,
                         sizeof(float) * atom_numbers);

    // CSR structure
    Device_Malloc_Safely((void**)&d_bond_count, sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&d_bond_offset,
                         sizeof(int) * (atom_numbers + 1));
    Device_Malloc_Safely((void**)&d_bond_nbr, sizeof(int) * 2 * max_bonds);
    Device_Malloc_Safely((void**)&d_bond_idx, sizeof(int) * 2 * max_bonds);
    Device_Malloc_Safely((void**)&d_fill_count, sizeof(int) * atom_numbers);

    // Per-bond sparse arrays
    Device_Malloc_Safely((void**)&d_pair_i, sizeof(int) * max_bonds);
    Device_Malloc_Safely((void**)&d_pair_j, sizeof(int) * max_bonds);
    Device_Malloc_Safely((void**)&d_pair_distances, sizeof(float) * max_bonds);
    Device_Malloc_Safely((void**)&d_num_pairs_ptr, sizeof(int));
    Device_Malloc_Safely((void**)&d_clustered_sorted_crd,
                         sizeof(VECTOR) * atom_numbers);
    clustered_scratch_capacity = atom_numbers;

    Device_Malloc_Safely((void**)&d_corrected_bo_s, sizeof(float) * max_bonds);
    Device_Malloc_Safely((void**)&d_corrected_bo_pi, sizeof(float) * max_bonds);
    Device_Malloc_Safely((void**)&d_corrected_bo_pi2,
                         sizeof(float) * max_bonds);

    Device_Malloc_Safely((void**)&d_dE_dBO_s, sizeof(float) * max_bonds);
    Device_Malloc_Safely((void**)&d_dE_dBO_pi, sizeof(float) * max_bonds);
    Device_Malloc_Safely((void**)&d_dE_dBO_pi2, sizeof(float) * max_bonds);

    Device_Malloc_Safely((void**)&d_dbo_s_dr, sizeof(float) * max_bonds);
    Device_Malloc_Safely((void**)&d_dbo_pi_dr, sizeof(float) * max_bonds);
    Device_Malloc_Safely((void**)&d_dbo_pi2_dr, sizeof(float) * max_bonds);
    Device_Malloc_Safely((void**)&d_dbo_s_dDelta_i, sizeof(float) * max_bonds);
    Device_Malloc_Safely((void**)&d_dbo_pi_dDelta_i, sizeof(float) * max_bonds);
    Device_Malloc_Safely((void**)&d_dbo_pi2_dDelta_i,
                         sizeof(float) * max_bonds);
    Device_Malloc_Safely((void**)&d_dbo_s_dDelta_j, sizeof(float) * max_bonds);
    Device_Malloc_Safely((void**)&d_dbo_pi_dDelta_j, sizeof(float) * max_bonds);
    Device_Malloc_Safely((void**)&d_dbo_pi2_dDelta_j,
                         sizeof(float) * max_bonds);
    Device_Malloc_Safely((void**)&d_dbo_raw_total_dr,
                         sizeof(float) * max_bonds);

    is_initialized = 1;
    controller->printf("  Sparse bond storage: max_bonds = %d\n", max_bonds);
    controller->printf("END INITIALIZING REAXFF_BOND_ORDER\n\n");
}

void REAXFF_BOND_ORDER::Calculate_Corrected_Bond_Orders_GPU(
    int atom_numbers, const VECTOR* d_crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, float cutoff, int num_pairs, int* d_pair_i,
    int* d_pair_j, float* d_distances)
{
    if (!is_initialized) return;

    if (num_pairs <= 0) return;

    dim3 blockSize = {CONTROLLER::device_max_thread};
    dim3 gridSize = {(num_pairs + blockSize.x - 1) / blockSize.x};

    Launch_Device_Kernel(
        Apply_Bond_Order_Corrections_Kernel, gridSize, blockSize, 0, NULL,
        num_pairs, d_pair_i, d_pair_j, d_distances, d_crd, cell, rcell,
        d_atom_type, d_r_s, d_r_p, d_r_pp, d_bo_1, d_bo_2, d_bo_3, d_bo_4,
        d_bo_5, d_bo_6, d_ro_pi, d_ro_pi2, d_valency, d_valency_val, d_ovc,
        d_v13cor, d_p_boc3, d_p_boc4, d_p_boc5, atom_type_numbers, atom_numbers,
        gp_boc1, gp_boc2, gp_bo_cut, d_total_bond_order, d_corrected_bo_s,
        d_corrected_bo_pi, d_corrected_bo_pi2, d_dbo_s_dr, d_dbo_pi_dr,
        d_dbo_pi2_dr, d_dbo_s_dDelta_i, d_dbo_pi_dDelta_i, d_dbo_pi2_dDelta_i,
        d_dbo_s_dDelta_j, d_dbo_pi_dDelta_j, d_dbo_pi2_dDelta_j,
        d_dbo_raw_total_dr);
}

void REAXFF_BOND_ORDER::Build_Bond_CSR(int atom_numbers, int num_bonds)
{
    if (num_bonds <= 0)
    {
        deviceMemset(d_bond_count, 0, sizeof(int) * atom_numbers);
        deviceMemset(d_bond_offset, 0, sizeof(int) * (atom_numbers + 1));
        return;
    }

    dim3 blockSize = {CONTROLLER::device_max_thread};
    dim3 gridSize_bonds = {(num_bonds + blockSize.x - 1) / blockSize.x};

    // Phase 1: Count bonds per atom
    deviceMemset(d_bond_count, 0, sizeof(int) * atom_numbers);
    Launch_Device_Kernel(Count_Bonds_Per_Atom_Kernel, gridSize_bonds, blockSize,
                         0, NULL, num_bonds, d_pair_i, d_pair_j, d_bond_count);

    // Phase 2: Exclusive prefix sum
#ifdef USE_CPU
    Launch_Device_Kernel(Exclusive_Prefix_Sum_Kernel, dim3(1), dim3(1), 0, NULL,
                         atom_numbers, d_bond_count, d_bond_offset);
#else
    thrust::device_ptr<int> counts(d_bond_count);
    thrust::device_ptr<int> offsets(d_bond_offset);
    thrust::exclusive_scan(counts, counts + atom_numbers, offsets);
    deviceMemset(d_bond_offset + atom_numbers, 0, sizeof(int));
#endif

    // Phase 3: Fill CSR
    deviceMemset(d_fill_count, 0, sizeof(int) * atom_numbers);
    Launch_Device_Kernel(Fill_Bond_CSR_Kernel, gridSize_bonds, blockSize, 0,
                         NULL, num_bonds, d_pair_i, d_pair_j, d_bond_offset,
                         d_fill_count, d_bond_nbr, d_bond_idx);
}

void REAXFF_BOND_ORDER::Calculate_Corrected_Bond_Order(
    int atom_numbers, const VECTOR* d_crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, float cutoff, const CLUSTERED_SPATIAL_VIEW& view)
{
    if (!is_initialized) return;

    if (h_atom_type == NULL)
    {
        printf(
            "ERROR: REAXFF_BOND_ORDER::Calculate_Corrected_Bond_Order - "
            "h_atom_type is NULL\n");
        return;
    }

    if (view.ghost_numbers != 0 || view.local_atom_numbers != atom_numbers ||
        view.total_atom_numbers != atom_numbers ||
        clustered_scratch_capacity < atom_numbers)
        throw std::runtime_error(
            "clustered ReaxFF bond order requires a single-rank "
            "all-local spatial view");
    const char* failure_reason = NULL;
    CLUSTERED_SPATIAL_VIEW_REQUIREMENTS requirements;
    requirements.local_atom_numbers = atom_numbers;
    requirements.ghost_numbers = 0;
    requirements.cutoff = cutoff;
    requirements.require_all_local_atoms = true;
#ifdef USE_CPU
    requirements.require_backend = true;
    requirements.backend = CLUSTERED_SPATIAL_BACKEND::CPU;
    requirements.require_gmxpacked_payload = true;
#else
    requirements.require_backend = true;
#if defined(USE_CUDA)
    requirements.backend = CLUSTERED_SPATIAL_BACKEND::CUDA;
#else
    requirements.backend = CLUSTERED_SPATIAL_BACKEND::HIP;
#endif
    requirements.require_same_producer_stream = true;
    requirements.consumer_stream = NULL;
    requirements.require_gmxpacked_payload = true;
    requirements.require_pair_shift_metadata = true;
    requirements.require_pair_shift_rcell = true;
    requirements.pair_shift_rcell = rcell;
#endif
    if (!Clustered_Validate_Spatial_View(view, requirements, &failure_reason))
        throw std::runtime_error(
            std::string("clustered ReaxFF bond order rejected payload: ") +
            (failure_reason == NULL ? "unknown failure" : failure_reason));
    deviceMemset(d_total_bond_order, 0, sizeof(float) * atom_numbers);
    deviceMemset(d_num_pairs_ptr, 0, sizeof(int));
    if (view.total_atom_numbers > 0)
    {
        Launch_Device_Kernel(REAXFF_Gather_Clustered_Coordinates,
                             (view.total_atom_numbers + 255) / 256, 256, 0,
                             NULL, view.total_atom_numbers,
                             view.cluster_numbers, view.sort_permutation,
                             view.cluster_offsets, view.cluster_centers, d_crd,
                             cell, rcell, d_clustered_sorted_crd);
    }
#ifdef USE_CPU
    Calculate_Uncorrected_Bond_Orders_Clustered_Gmxpacked_CPU(
        view, d_clustered_sorted_crd, d_atom_type, atom_type_numbers, d_r_s,
        d_r_p, d_r_pp, d_bo_1, d_bo_2, d_bo_3, d_bo_4, d_bo_5, d_bo_6, d_ro_pi,
        d_ro_pi2, cutoff, gp_bo_cut, cell, d_total_bond_order, d_pair_i,
        d_pair_j, d_pair_distances, max_bonds, d_num_pairs_ptr);
#else
    constexpr int packed_partitions = 8;
    if (view.gmxpacked_sci_numbers > 0)
    {
        const dim3 block(kClusteredClusterSize, kClusteredClusterSize, 1);
        const dim3 grid(view.gmxpacked_sci_numbers, packed_partitions, 1);
        Launch_Device_Kernel(
            Calculate_Uncorrected_Bond_Orders_Clustered_Gmxpacked, grid, block,
            0, NULL, view.gmxpacked_sci_numbers, packed_partitions,
            view.cluster_numbers, view.cluster_offsets,
            view.cluster_valid_masks, view.cluster_local_masks,
            view.super_cluster_offsets, view.gmxpacked_sci,
            view.gmxpacked_cjpacked, view.gmxpacked_exclusions,
            view.pair_shift_bits, view.sort_permutation, d_clustered_sorted_crd,
            d_atom_type, atom_type_numbers, d_r_s, d_r_p, d_r_pp, d_bo_1,
            d_bo_2, d_bo_3, d_bo_4, d_bo_5, d_bo_6, d_ro_pi, d_ro_pi2, cutoff,
            gp_bo_cut, cell, d_total_bond_order, d_pair_i, d_pair_j,
            d_pair_distances, max_bonds, d_num_pairs_ptr);
    }
#endif

    deviceMemcpy(&h_num_pairs, d_num_pairs_ptr, sizeof(int),
                 deviceMemcpyDeviceToHost);
    int num_pairs = h_num_pairs;

    if (num_pairs > max_bonds)
    {
        printf(
            "WARNING: REAXFF_BOND_ORDER - num_pairs (%d) exceeds max_bonds "
            "(%d), results may be incorrect!\n",
            num_pairs, max_bonds);
        num_pairs = max_bonds;
    }
    h_num_pairs = num_pairs;

    Build_Bond_CSR(atom_numbers, num_pairs);
    deviceMemset(d_total_corrected_bond_order, 0, sizeof(float) * atom_numbers);
    if (num_pairs > 0)
    {
        Calculate_Corrected_Bond_Orders_GPU(atom_numbers, d_crd, cell, rcell,
                                            cutoff, num_pairs, d_pair_i,
                                            d_pair_j, d_pair_distances);

        // Reduce corrected BO per atom using CSR
        dim3 blockSize = {CONTROLLER::device_max_thread};
        dim3 gridSize = {(atom_numbers + blockSize.x - 1) / blockSize.x};
        Launch_Device_Kernel(Reduce_Total_Corrected_Bond_Order_Kernel, gridSize,
                             blockSize, 0, NULL, atom_numbers, d_bond_count,
                             d_bond_offset, d_bond_idx, d_corrected_bo_s,
                             d_corrected_bo_pi, d_corrected_bo_pi2,
                             d_total_corrected_bond_order);
    }
}

void REAXFF_BOND_ORDER::Calculate_Forces(int atom_numbers, const VECTOR* d_crd,
                                         VECTOR* d_frc, const LTMatrix3 cell,
                                         const LTMatrix3 rcell, float cutoff,
                                         float* d_CdDelta, int need_virial,
                                         LTMatrix3* atom_virial)
{
    if (!is_initialized || h_num_pairs <= 0) return;

    dim3 blockSize = {CONTROLLER::device_max_thread};
    dim3 gridSize = {(h_num_pairs + blockSize.x - 1) / blockSize.x};

    Launch_Device_Kernel(Calculate_CdDelta_Prime_Kernel, gridSize, blockSize, 0,
                         NULL, h_num_pairs, d_pair_i, d_pair_j, d_dE_dBO_s,
                         d_dE_dBO_pi, d_dE_dBO_pi2, d_CdDelta, d_dbo_s_dDelta_i,
                         d_dbo_pi_dDelta_i, d_dbo_pi2_dDelta_i,
                         d_dbo_s_dDelta_j, d_dbo_pi_dDelta_j,
                         d_dbo_pi2_dDelta_j, d_CdDelta_prime);

    Launch_Device_Kernel(
        REAXFF_Force_Projection_Kernel, gridSize, blockSize, 0, NULL,
        h_num_pairs, d_pair_i, d_pair_j, d_pair_distances, d_crd, cell, rcell,
        d_dE_dBO_s, d_dE_dBO_pi, d_dE_dBO_pi2, d_CdDelta, d_dbo_s_dr,
        d_dbo_pi_dr, d_dbo_pi2_dr, d_dbo_raw_total_dr, d_CdDelta_prime, d_frc,
        need_virial ? atom_virial : NULL);
}

void REAXFF_BOND_ORDER::Clear_Derivatives(int atom_numbers, float* d_CdDelta)
{
    if (!is_initialized) return;
    if (h_num_pairs > 0)
    {
        deviceMemset(d_dE_dBO_s, 0, sizeof(float) * h_num_pairs);
        deviceMemset(d_dE_dBO_pi, 0, sizeof(float) * h_num_pairs);
        deviceMemset(d_dE_dBO_pi2, 0, sizeof(float) * h_num_pairs);
    }
    if (d_CdDelta)
    {
        deviceMemset(d_CdDelta, 0, sizeof(float) * atom_numbers);
    }
    deviceMemset(d_CdDelta_prime, 0, sizeof(float) * atom_numbers);
}

void REAXFF_BOND_ORDER::Calculate_Bond_Order(
    int atom_numbers, const VECTOR* d_crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, float cutoff,
    const CLUSTERED_SPATIAL_VIEW& clustered_view)
{
    Calculate_Corrected_Bond_Order(atom_numbers, d_crd, cell, rcell, cutoff,
                                   clustered_view);
}
