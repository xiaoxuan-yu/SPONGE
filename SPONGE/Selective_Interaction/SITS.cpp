#include "SITS.h"

static __host__ __device__ __forceinline__ VECTOR_LJ
SITS_Make_Packed_LJ_Atom(const float4 xq, const int lj_type)
{
    VECTOR_LJ atom = {};
    atom.crd = {xq.x, xq.y, xq.z};
    atom.LJ_type = lj_type;
    atom.charge = xq.w;
    return atom;
}

struct SITS_CLUSTERED_PAIR_FORCE
{
    VECTOR force_i;
    bool owner_is_i;
    bool selective_owner;
    float selective_factor;
};

template <bool full_output, bool store_j_force_direct,
          bool correction_only = false>
#ifdef USE_CPU
static __host__ __forceinline__ SITS_CLUSTERED_PAIR_FORCE
#else
static __device__ __forceinline__ SITS_CLUSTERED_PAIR_FORCE
#endif
SITS_Store_Clustered_Computed_Pair(
    const int atom_i, const int atom_j, const int global_i,
    const int global_j, const bool j_is_local,
    const VECTOR dr, const VECTOR force_i, const float energy_lj,
    const float energy_coulomb, const int* atom_sys_mark,
    const float pwwp_factor, VECTOR* frc, VECTOR* frc_enhancing,
    float* atom_energy,
    float* atom_energy_enhancing, LTMatrix3* atom_virial,
    LTMatrix3* atom_virial_enhancing, float* atom_direct_cf_energy,
    float* atom_ene_lj, const bool store_energy, const bool store_virial)
{
    const bool owner_is_i = !j_is_local || global_i < global_j;
    const int owner = owner_is_i ? atom_i : atom_j;
    const float owner_force_sign = owner_is_i ? 1.0f : -1.0f;
    const float ij_factor = j_is_local ? 1.0f : 0.5f;
    const bool selective_owner = atom_sys_mark[owner] == 0;
    float selective_factor = 0.0f;
    if (selective_owner)
    {
        const int mark_sum =
            atom_sys_mark[atom_i] + atom_sys_mark[atom_j];
        selective_factor =
            mark_sum == 0 ? 1.0f : (mark_sum == 1 ? pwwp_factor : 0.0f);
    }

    if constexpr (!correction_only)
    {
        atomicAdd(frc + atom_i, force_i);
        if (j_is_local && store_j_force_direct)
        {
            atomicAdd(frc + atom_j, -force_i);
        }
    }
    if (selective_owner)
    {
        if constexpr (store_j_force_direct)
        {
            atomicAdd(frc_enhancing + owner,
                      owner_force_sign * selective_factor * force_i);
        }
        else if (owner_is_i)
        {
            atomicAdd(frc_enhancing + owner,
                      selective_factor * force_i);
        }
    }

    if constexpr (full_output)
    {
        if (store_energy)
        {
            if constexpr (!correction_only)
            {
                atomicAdd(atom_energy + owner,
                          energy_lj +
                              (selective_owner ? 0.0f : energy_coulomb));
                atomicAdd(atom_direct_cf_energy + owner, energy_coulomb);
                atomicAdd(atom_ene_lj + owner, energy_lj);
            }
            else if (selective_owner)
            {
                atomicAdd(atom_energy + owner, -energy_coulomb);
            }
            if (selective_owner)
            {
                atomicAdd(atom_energy_enhancing + owner,
                          selective_factor *
                              (energy_lj + energy_coulomb));
            }
        }
        if (store_virial)
        {
            const LTMatrix3 virial0 =
                Get_Virial_From_Force_Dis(force_i, dr);
            if constexpr (!correction_only)
            {
                if (selective_owner)
                {
                    atomicAdd(atom_virial + owner,
                              ij_factor * selective_factor * virial0);
                }
                else
                {
                    atomicAdd(atom_virial + owner, -ij_factor * virial0);
                }
            }
            else if (selective_owner)
            {
                atomicAdd(atom_virial + owner,
                          ij_factor * (selective_factor + 1.0f) *
                              virial0);
            }
            if (selective_owner)
            {
                atomicAdd(atom_virial_enhancing + owner,
                          ij_factor * selective_factor *
                              selective_factor * virial0);
            }
        }
    }
    return SITS_CLUSTERED_PAIR_FORCE{
        force_i, owner_is_i, selective_owner, selective_factor};
}

template <bool full_output, bool store_j_force_direct>
#ifdef USE_CPU
static __host__ __forceinline__ SITS_CLUSTERED_PAIR_FORCE
#else
static __device__ __forceinline__ SITS_CLUSTERED_PAIR_FORCE
#endif
REST2_Store_Clustered_Computed_Pair(
    const int atom_i, const int atom_j, const int global_i,
    const int global_j, const bool j_is_local, const VECTOR dr,
    const VECTOR force_i, const float energy_lj,
    const float energy_coulomb, const int* atom_sys_mark, VECTOR* frc,
    float* atom_energy, LTMatrix3* atom_virial,
    float* atom_direct_cf_energy, float* atom_ene_lj,
    const bool store_energy, const bool store_virial,
    float* unscaled_atom_energy, float* effective_atom_energy,
    const float lambda_m, const float sqrt_lambda_m)
{
    const bool owner_is_i = !j_is_local || global_i < global_j;
    const int owner = owner_is_i ? atom_i : atom_j;
    const float ij_factor = j_is_local ? 1.0f : 0.5f;
    const int mark_sum = atom_sys_mark[atom_i] + atom_sys_mark[atom_j];
    if (mark_sum >= 2)
    {
        return SITS_CLUSTERED_PAIR_FORCE{
            VECTOR{0.0f, 0.0f, 0.0f}, owner_is_i, false, 0.0f};
    }
    const float scale = mark_sum == 0 ? lambda_m : sqrt_lambda_m;
    const float correction_scale = scale - 1.0f;
    const VECTOR correction_force = correction_scale * force_i;
    atomicAdd(frc + atom_i, correction_force);
    if (j_is_local && store_j_force_direct)
    {
        atomicAdd(frc + atom_j, -correction_force);
    }
    if constexpr (full_output)
    {
        const float pair_energy = energy_lj + energy_coulomb;
        if (store_energy)
        {
            atomicAdd(atom_energy + owner, correction_scale * pair_energy);
            atomicAdd(atom_direct_cf_energy + owner,
                      correction_scale * energy_coulomb);
            atomicAdd(atom_ene_lj + owner, correction_scale * energy_lj);
            atomicAdd(unscaled_atom_energy + owner, pair_energy);
            atomicAdd(effective_atom_energy + owner, scale * pair_energy);
        }
        if (store_virial)
        {
            atomicAdd(atom_virial + owner,
                      -ij_factor * correction_scale *
                          Get_Virial_From_Force_Dis(force_i, dr));
        }
    }
    return SITS_CLUSTERED_PAIR_FORCE{
        correction_force, owner_is_i, true, correction_scale};
}

template <bool full_output, bool store_j_force_direct,
          bool correction_only = false, bool rest2_correction = false>
#ifdef USE_CPU
static __host__ __forceinline__ SITS_CLUSTERED_PAIR_FORCE
#else
static __device__ __forceinline__ SITS_CLUSTERED_PAIR_FORCE
#endif
SITS_Store_Clustered_Pair(
    const int atom_i, const int atom_j, const int global_i,
    const int global_j, const bool j_is_local,
    const VECTOR_LJ r1, const VECTOR_LJ r2, const VECTOR dr,
    const float2 lj_ab, const float dr_abs, const float pme_beta,
    const int* atom_sys_mark, const float pwwp_factor, VECTOR* frc,
    VECTOR* frc_enhancing, float* atom_energy,
    float* atom_energy_enhancing, LTMatrix3* atom_virial,
    LTMatrix3* atom_virial_enhancing, float* atom_direct_cf_energy,
    float* atom_ene_lj, const bool store_energy, const bool store_virial,
    float* rest2_unscaled_atom_energy = NULL,
    float* rest2_effective_atom_energy = NULL,
    const float rest2_lambda_m = 1.0f,
    const float rest2_sqrt_lambda_m = 1.0f)
{
    const float ij_factor = j_is_local ? 1.0f : 0.5f;
    float force_abs = Get_LJ_Force(r1, r2, dr_abs, lj_ab.x, lj_ab.y);
    force_abs -= Get_Direct_Coulomb_Force(r1, r2, dr_abs, pme_beta);
    const VECTOR force_i = force_abs * dr;
    float energy_lj = 0.0f;
    float energy_coulomb = 0.0f;
    if constexpr (full_output)
    {
        energy_lj =
            ij_factor * Get_LJ_Energy(r1, r2, dr_abs, lj_ab.x, lj_ab.y);
        energy_coulomb =
            ij_factor *
            Get_Direct_Coulomb_Energy(r1, r2, dr_abs, pme_beta);
    }
    if constexpr (rest2_correction)
    {
        return REST2_Store_Clustered_Computed_Pair<
            full_output, store_j_force_direct>(
            atom_i, atom_j, global_i, global_j, j_is_local, dr, force_i,
            energy_lj, energy_coulomb, atom_sys_mark, frc, atom_energy,
            atom_virial, atom_direct_cf_energy, atom_ene_lj, store_energy,
            store_virial, rest2_unscaled_atom_energy,
            rest2_effective_atom_energy, rest2_lambda_m,
            rest2_sqrt_lambda_m);
    }
    else
    {
        return SITS_Store_Clustered_Computed_Pair<
            full_output, store_j_force_direct, correction_only>(
            atom_i, atom_j, global_i, global_j, j_is_local, dr, force_i,
            energy_lj, energy_coulomb, atom_sys_mark, pwwp_factor, frc,
            frc_enhancing, atom_energy, atom_energy_enhancing, atom_virial,
            atom_virial_enhancing, atom_direct_cf_energy, atom_ene_lj,
            store_energy, store_virial);
    }
}

template <bool full_output, bool store_j_force_direct,
          bool correction_only = false>
#ifdef USE_CPU
static __host__ __forceinline__ SITS_CLUSTERED_PAIR_FORCE
#else
static __device__ __forceinline__ SITS_CLUSTERED_PAIR_FORCE
#endif
SITS_Store_Clustered_Soft_Pair(
    const int atom_i, const int atom_j, const int global_i,
    const int global_j, const bool j_is_local,
    const VECTOR_LJ_SOFT_TYPE r1, const VECTOR_LJ_SOFT_TYPE r2,
    const VECTOR dr, const float AA, const float AB, const float BA,
    const float BB, const float pme_beta, const float lambda,
    const float alpha, const float p, const float input_sigma_6,
    const float input_sigma_6_min, const int* atom_sys_mark,
    const float pwwp_factor, VECTOR* frc, VECTOR* frc_enhancing,
    float* atom_energy, float* atom_energy_enhancing,
    LTMatrix3* atom_virial, LTMatrix3* atom_virial_enhancing,
    float* atom_direct_cf_energy, float* atom_ene_lj,
    const bool store_energy, const bool store_virial)
{
    const float dr2 = dr * dr;
    const float ij_factor = j_is_local ? 1.0f : 0.5f;
    VECTOR force_i = {0.0f, 0.0f, 0.0f};
    float energy_lj = 0.0f;
    float energy_coulomb = 0.0f;
    LTMatrix3 unused_virial = {};
    Compute_Clustered_Soft_Core_Pair<full_output>(
        r1, r2, dr.x, dr.y, dr.z, dr2, AA, AB, BA, BB, lambda,
        alpha, p, input_sigma_6, input_sigma_6_min, pme_beta,
        ij_factor, &force_i, &energy_lj, &energy_coulomb,
        &unused_virial);
    return SITS_Store_Clustered_Computed_Pair<
        full_output, store_j_force_direct, correction_only>(
        atom_i, atom_j, global_i, global_j, j_is_local,
        dr, force_i, energy_lj, energy_coulomb,
        atom_sys_mark, pwwp_factor, frc, frc_enhancing, atom_energy,
        atom_energy_enhancing, atom_virial, atom_virial_enhancing,
        atom_direct_cf_energy, atom_ene_lj, store_energy, store_virial);
}

#ifdef USE_GPU
static __device__ __forceinline__ float
SITS_Reduce_Subgroup_Force_To_Component(float x, float y, float z,
                                        const int component_lane)
{
#ifdef USE_CUDA
    const unsigned int active_mask = __activemask();
#endif
    for (int delta = 4; delta >= 1; delta >>= 1)
    {
#ifdef USE_CUDA
        x += __shfl_down_sync(active_mask, x, delta, 8);
        y += __shfl_down_sync(active_mask, y, delta, 8);
        z += __shfl_down_sync(active_mask, z, delta, 8);
#else
        x += __shfl_down(x, delta, 8);
        y += __shfl_down(y, delta, 8);
        z += __shfl_down(z, delta, 8);
#endif
    }
#ifdef USE_CUDA
    x = __shfl_sync(active_mask, x, 0, 8);
    y = __shfl_sync(active_mask, y, 0, 8);
    z = __shfl_sync(active_mask, z, 0, 8);
#else
    x = __shfl(x, 0, 8);
    y = __shfl(y, 0, 8);
    z = __shfl(z, 0, 8);
#endif
    return component_lane == 0 ? x : (component_lane == 1 ? y : z);
}

static __device__ __forceinline__ void SITS_Atomic_Add_Force_Component(
    VECTOR* frc, const int atom_index, const int component,
    const float value)
{
    float* force_component = reinterpret_cast<float*>(frc + atom_index);
    atomicAdd(force_component + component, value);
}

static __device__ __forceinline__ bool
SITS_Prune_Gmxpacked_Record_For_Selection(
    const CLUSTERED_GMXPACKED_CJ source,
    const CLUSTERED_GMXPACKED_SCI sci_entry, const int cluster_numbers,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const int* super_cluster_offsets, const int* sorted_atom_ids,
    const int* atom_sys_mark,
    CLUSTERED_GMXPACKED_CJ* compact)
{
    *compact = source;
    const int cluster_i_begin =
        super_cluster_offsets[sci_entry.supercluster_id];
    bool keep_record = false;
    for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
    {
        const unsigned int source_imask = source.split[split].imask;
        unsigned int compact_imask = 0u;
        for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
        {
            const int cluster_j = source.cj[jm];
            bool selected_j = false;
            if (cluster_j >= 0 && cluster_j < cluster_numbers)
            {
                const int j_lane_begin =
                    split * kClusteredSplitJClusterSize;
                const int j_lane_end =
                    j_lane_begin + kClusteredSplitJClusterSize;
                for (int j_lane = j_lane_begin;
                     j_lane < j_lane_end; j_lane += 1)
                {
                    if ((cluster_valid_masks[cluster_j] &
                         (1u << static_cast<unsigned int>(j_lane))) == 0u)
                    {
                        continue;
                    }
                    const int atom_j = sorted_atom_ids[
                        cluster_offsets[cluster_j] + j_lane];
                    selected_j =
                        selected_j ||
                        (atom_j >= 0 && atom_sys_mark[atom_j] == 0);
                }
            }
            for (int i_local = 0;
                 i_local < kClusteredSuperClusterClusters; i_local += 1)
            {
                const unsigned int packed_bit =
                    1u << (jm * kClusteredSuperClusterClusters + i_local);
                if ((source_imask & packed_bit) == 0u)
                {
                    continue;
                }
                const int cluster_i = cluster_i_begin + i_local;
                bool selected_i = false;
                if (cluster_i >= 0 && cluster_i < cluster_numbers)
                {
                    for (int i_lane = 0;
                         i_lane < kClusteredClusterSize; i_lane += 1)
                    {
                        if ((cluster_valid_masks[cluster_i] &
                             (1u << static_cast<unsigned int>(i_lane))) == 0u)
                        {
                            continue;
                        }
                        const int atom_i = sorted_atom_ids[
                            cluster_offsets[cluster_i] + i_lane];
                        selected_i =
                            selected_i ||
                            (atom_i >= 0 && atom_sys_mark[atom_i] == 0);
                    }
                }
                if (selected_i || selected_j)
                {
                    compact_imask |= packed_bit;
                }
            }
        }
        compact->split[split].imask = compact_imask;
        keep_record = keep_record || compact_imask != 0u;
    }
    return keep_record;
}

static __global__ void SITS_Build_Sparse_Gmxpacked_View_Device(
    const int source_sci_numbers, const int cluster_numbers,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const int* super_cluster_offsets, const int* sorted_atom_ids,
    const int* atom_sys_mark,
    const CLUSTERED_GMXPACKED_SCI* source_sci,
    const CLUSTERED_GMXPACKED_CJ* source_cjpacked,
    const uint64_t* source_pair_shift_bits,
    CLUSTERED_GMXPACKED_SCI* compact_sci,
    CLUSTERED_GMXPACKED_CJ* compact_cjpacked,
    uint64_t* compact_pair_shift_bits, int* compact_counts)
{
    const int source_sci_index = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (source_sci_index >= source_sci_numbers)
    {
        return;
    }
    __shared__ int kept_records;
    __shared__ int compact_sci_index;
    __shared__ int compact_cj_begin;
    __shared__ int compact_write_cursor;
    if (lane == 0)
    {
        kept_records = 0;
        compact_sci_index = -1;
        compact_cj_begin = 0;
        compact_write_cursor = 0;
    }
    __syncthreads();

    const CLUSTERED_GMXPACKED_SCI source_sci_entry =
        source_sci[source_sci_index];
    for (int packed_index = source_sci_entry.cjpacked_begin + lane;
         packed_index < source_sci_entry.cjpacked_end;
         packed_index += static_cast<int>(blockDim.x))
    {
        CLUSTERED_GMXPACKED_CJ pruned = {};
        if (SITS_Prune_Gmxpacked_Record_For_Selection(
                source_cjpacked[packed_index], source_sci_entry,
                cluster_numbers, cluster_offsets, cluster_valid_masks,
                super_cluster_offsets, sorted_atom_ids, atom_sys_mark,
                &pruned))
        {
            atomicAdd(&kept_records, 1);
        }
    }
    __syncthreads();
    if (lane == 0 && kept_records > 0)
    {
        compact_sci_index = atomicAdd(compact_counts, 1);
        compact_cj_begin = atomicAdd(compact_counts + 1, kept_records);
        compact_write_cursor = compact_cj_begin;
        CLUSTERED_GMXPACKED_SCI output_sci = source_sci_entry;
        output_sci.cjpacked_begin = compact_cj_begin;
        output_sci.cjpacked_end = compact_cj_begin + kept_records;
        compact_sci[compact_sci_index] = output_sci;
    }
    __syncthreads();
    if (kept_records == 0)
    {
        return;
    }

    for (int packed_index = source_sci_entry.cjpacked_begin + lane;
         packed_index < source_sci_entry.cjpacked_end;
         packed_index += static_cast<int>(blockDim.x))
    {
        CLUSTERED_GMXPACKED_CJ pruned = {};
        if (!SITS_Prune_Gmxpacked_Record_For_Selection(
                source_cjpacked[packed_index], source_sci_entry,
                cluster_numbers, cluster_offsets, cluster_valid_masks,
                super_cluster_offsets, sorted_atom_ids, atom_sys_mark,
                &pruned))
        {
            continue;
        }
        const int compact_index = atomicAdd(&compact_write_cursor, 1);
        compact_cjpacked[compact_index] = pruned;
        for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
        {
            compact_pair_shift_bits[
                compact_index * kClusteredJGroupSize + jm] =
                source_pair_shift_bits[
                    packed_index * kClusteredJGroupSize + jm];
        }
    }
}

#endif

#ifndef USE_CPU
template <typename T>
static void SITS_Reserve_Sparse_Buffer(T** pointer, int* capacity,
                                       const int required)
{
    if (required <= *capacity && *pointer != NULL)
    {
        return;
    }
    Free_Single_Device_Pointer(reinterpret_cast<void**>(pointer));
    *capacity = 0;
    if (required > 0)
    {
        Device_Malloc_Safely(reinterpret_cast<void**>(pointer),
                             sizeof(T) * static_cast<size_t>(required));
        *capacity = required;
    }
}

static bool SITS_Ensure_Sparse_Gmxpacked_View(
    SITS_INFORMATION* sits, const LJ_CLUSTERED_DIRECT_CACHE* cache,
    const CLUSTERED_SPATIAL_VIEW& view, const int* atom_sys_mark)
{
    const LJ_CLUSTER_LAYOUT& layout = cache->layout;
    if (sits->clustered_sparse_provider_incarnation ==
            layout.provider_incarnation &&
        sits->clustered_sparse_payload_generation ==
            layout.gmxpacked_compact_payload_generation)
    {
        return true;
    }
    sits->clustered_sparse_sci_numbers = 0;
    sits->clustered_sparse_cjpacked_numbers = 0;
    if (view.gmxpacked_sci_numbers <= 0 ||
        view.gmxpacked_cjpacked_numbers <= 0)
    {
        sits->clustered_sparse_provider_incarnation =
            layout.provider_incarnation;
        sits->clustered_sparse_payload_generation =
            layout.gmxpacked_compact_payload_generation;
        return true;
    }

    SITS_Reserve_Sparse_Buffer(
        &sits->d_clustered_sparse_sci,
        &sits->clustered_sparse_sci_capacity,
        view.gmxpacked_sci_numbers);
    SITS_Reserve_Sparse_Buffer(
        &sits->d_clustered_sparse_cjpacked,
        &sits->clustered_sparse_cjpacked_capacity,
        view.gmxpacked_cjpacked_numbers);
    SITS_Reserve_Sparse_Buffer(
        &sits->d_clustered_sparse_pair_shift_bits,
        &sits->clustered_sparse_pair_shift_capacity,
        view.gmxpacked_cjpacked_numbers * kClusteredJGroupSize);
    if (sits->d_clustered_sparse_counts == NULL)
    {
        Device_Malloc_Safely(
            reinterpret_cast<void**>(&sits->d_clustered_sparse_counts),
            sizeof(int) * 2);
    }
    deviceMemset(sits->d_clustered_sparse_counts, 0, sizeof(int) * 2);
    const dim3 block_size(128u, 1u, 1u);
    const dim3 grid_size(
        static_cast<unsigned int>(view.gmxpacked_sci_numbers), 1u, 1u);
    Launch_Device_Kernel(
        SITS_Build_Sparse_Gmxpacked_View_Device, grid_size, block_size,
        0, NULL, view.gmxpacked_sci_numbers, view.cluster_numbers,
        view.cluster_offsets, view.cluster_valid_masks,
        view.super_cluster_offsets, view.sort_permutation, atom_sys_mark,
        view.gmxpacked_sci, view.gmxpacked_cjpacked,
        view.pair_shift_bits, sits->d_clustered_sparse_sci,
        sits->d_clustered_sparse_cjpacked,
        sits->d_clustered_sparse_pair_shift_bits,
        sits->d_clustered_sparse_counts);
    int compact_counts[2] = {0, 0};
    deviceMemcpy(compact_counts, sits->d_clustered_sparse_counts,
                 sizeof(compact_counts), deviceMemcpyDeviceToHost);
    if (compact_counts[0] < 0 ||
        compact_counts[0] > view.gmxpacked_sci_numbers ||
        compact_counts[1] < 0 ||
        compact_counts[1] > view.gmxpacked_cjpacked_numbers)
    {
        return false;
    }
    sits->clustered_sparse_sci_numbers = compact_counts[0];
    sits->clustered_sparse_cjpacked_numbers = compact_counts[1];
    sits->clustered_sparse_provider_incarnation =
        layout.provider_incarnation;
    sits->clustered_sparse_payload_generation =
        layout.gmxpacked_compact_payload_generation;
    return true;
}
#endif

template <bool full_output, bool correction_only = false,
          bool soft_core = false, bool rest2_correction = false>
static __global__ void SITS_Clustered_Gmxpacked_Direct_Device(
    const int sci_numbers, const int packed_partitions,
    const int cluster_numbers,
    const int local_atom_numbers,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const int* super_cluster_offsets,
    const CLUSTERED_GMXPACKED_SCI* sci_entries,
    const CLUSTERED_GMXPACKED_CJ* cjpacked_entries,
    const CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    const uint64_t* pair_shift_bits, const int* sorted_atom_ids,
    const float4* sorted_xq, const int* sorted_lj_type,
    const VECTOR_LJ_SOFT_TYPE* sorted_soft_crd,
    const int* atom_local, const int* atom_sys_mark, const LTMatrix3 cell,
    const float2* lj_ab_packed, const float* lj_aa,
    const float* lj_ab, const float* lj_ba, const float* lj_bb,
    const float lambda, const float alpha, const float soft_p,
    const float sigma_6, const float sigma_6_min,
    const float cutoff, VECTOR* frc,
    VECTOR* frc_enhancing, const float pme_beta, float* atom_energy,
    float* atom_energy_enhancing, LTMatrix3* atom_virial,
    LTMatrix3* atom_virial_enhancing, float* atom_direct_cf_energy,
    float* atom_ene_lj, const float pwwp_factor, const bool store_energy,
    const bool store_virial, float* rest2_unscaled_atom_energy,
    float* rest2_effective_atom_energy, const float rest2_lambda_m,
    const float rest2_sqrt_lambda_m)
{
#ifdef USE_GPU
    const int sci =
        full_output
            ? static_cast<int>(blockIdx.x)
            : static_cast<int>(blockIdx.x) / packed_partitions;
    const int packed_partition =
        full_output
            ? static_cast<int>(blockIdx.y)
            : static_cast<int>(blockIdx.x) % packed_partitions;
    const int i_lane = static_cast<int>(threadIdx.x);
    const int j_lane = static_cast<int>(threadIdx.y);
    if (sci >= sci_numbers || i_lane >= kClusteredClusterSize ||
        j_lane >= kClusteredClusterSize)
    {
        return;
    }
    const CLUSTERED_GMXPACKED_SCI sci_entry = sci_entries[sci];
    const int cluster_i_begin =
        super_cluster_offsets[sci_entry.supercluster_id];
    int cluster_i_end =
        super_cluster_offsets[sci_entry.supercluster_id + 1];
    if (cluster_i_end > cluster_numbers)
    {
        cluster_i_end = cluster_numbers;
    }
    const int split = j_lane / kClusteredSplitJClusterSize;
    const int split_j_lane =
        j_lane - split * kClusteredSplitJClusterSize;
    const unsigned int i_lane_bit = 1u << i_lane;
    const unsigned int j_lane_bit = 1u << j_lane;
    const float cutoff_sq = cutoff * cutoff;

    for (int packed_idx = sci_entry.cjpacked_begin + packed_partition;
         packed_idx < sci_entry.cjpacked_end;
         packed_idx += packed_partitions)
    {
        const CLUSTERED_GMXPACKED_CJ& packed =
            cjpacked_entries[packed_idx];
        const CLUSTERED_GMXPACKED_SPLIT& split_entry =
            packed.split[split];
        unsigned int pair_bits = 0xffffffffu;
        if (split_entry.exclusion_index != 0)
        {
            pair_bits =
                exclusion_entries[split_entry.exclusion_index]
                    .pair[split_j_lane * kClusteredClusterSize + i_lane];
        }
        const unsigned int effective_mask =
            split_entry.imask & pair_bits;
        const int jm_begin =
            full_output ? static_cast<int>(blockIdx.z) : 0;
        const int jm_end =
            full_output ? jm_begin + 1 : kClusteredJGroupSize;
        for (int jm = jm_begin; jm < jm_end; jm += 1)
        {
            const int cluster_j = packed.cj[jm];
            if (cluster_j < 0 ||
                (cluster_valid_masks[cluster_j] & j_lane_bit) == 0u)
            {
                continue;
            }
            const int sorted_j = cluster_offsets[cluster_j] + j_lane;
            const int atom_j = sorted_atom_ids[sorted_j];
            const bool j_is_local =
                (cluster_local_masks[cluster_j] & j_lane_bit) != 0u;
            VECTOR_LJ r2 = {};
            VECTOR_LJ_SOFT_TYPE soft_r2 = {};
            VECTOR r2_crd = {};
            if constexpr (soft_core)
            {
                soft_r2 = sorted_soft_crd[sorted_j];
                r2_crd = soft_r2.crd;
            }
            else
            {
                r2 = SITS_Make_Packed_LJ_Atom(
                    sorted_xq[sorted_j], sorted_lj_type[sorted_j]);
                r2_crd = r2.crd;
            }
            const uint64_t shift_bits =
                pair_shift_bits[packed_idx * kClusteredJGroupSize + jm];
            VECTOR force_j = {0.0f, 0.0f, 0.0f};
            VECTOR force_j_enhancing = {0.0f, 0.0f, 0.0f};
            for (int i_local = 0;
                 i_local < cluster_i_end - cluster_i_begin; i_local += 1)
            {
                const unsigned int packed_bit =
                    1u << (jm * kClusteredSuperClusterClusters + i_local);
                if ((effective_mask & packed_bit) == 0u ||
                    (Clustered_Get_Pair_Active_I_Mask(
                         shift_bits, split) &
                     (1u << static_cast<unsigned int>(i_local))) == 0u)
                {
                    continue;
                }
                const int cluster_i = cluster_i_begin + i_local;
                if ((cluster_valid_masks[cluster_i] & i_lane_bit) == 0u ||
                    (cluster_local_masks[cluster_i] & i_lane_bit) == 0u)
                {
                    continue;
                }
                const int sorted_i = cluster_offsets[cluster_i] + i_lane;
                const int atom_i = sorted_atom_ids[sorted_i];
                const int global_i = atom_local[atom_i];
                const int global_j = atom_local[atom_j];
                if constexpr (rest2_correction)
                {
                    if (atom_sys_mark[atom_i] + atom_sys_mark[atom_j] >= 2)
                    {
                        continue;
                    }
                }
                VECTOR_LJ r1 = {};
                VECTOR_LJ_SOFT_TYPE soft_r1 = {};
                VECTOR r1_crd = {};
                if constexpr (soft_core)
                {
                    soft_r1 = sorted_soft_crd[sorted_i];
                    r1_crd = soft_r1.crd;
                }
                else
                {
                    r1 = SITS_Make_Packed_LJ_Atom(
                        sorted_xq[sorted_i], sorted_lj_type[sorted_i]);
                    r1_crd = r1.crd;
                }
                const int shift_id =
                    Clustered_Get_Pair_Shift_Id(shift_bits, i_local);
                const VECTOR shift =
                    Clustered_Shift_Vector_From_Id(shift_id, cell);
                const VECTOR dr = (r2_crd - r1_crd) - shift;
                const float dr2 = dr * dr;
                if (dr2 <= 0.0f || dr2 >= cutoff_sq)
                {
                    continue;
                }
                SITS_CLUSTERED_PAIR_FORCE stored = {};
                if constexpr (soft_core)
                {
                    const int pair_type_a = Get_LJ_Type(
                        soft_r1.LJ_type, soft_r2.LJ_type);
                    const int pair_type_b = Get_LJ_Type(
                        soft_r1.LJ_type_B, soft_r2.LJ_type_B);
                        stored =
                            SITS_Store_Clustered_Soft_Pair<
                                full_output, false, correction_only>(
                                atom_i, atom_j, global_i, global_j,
                                j_is_local, soft_r1,
                            soft_r2, dr, lj_aa[pair_type_a],
                            lj_ab[pair_type_a], lj_ba[pair_type_b],
                            lj_bb[pair_type_b], pme_beta, lambda, alpha,
                            soft_p, sigma_6, sigma_6_min, atom_sys_mark,
                            pwwp_factor, frc, frc_enhancing, atom_energy,
                            atom_energy_enhancing, atom_virial,
                                atom_virial_enhancing,
                                atom_direct_cf_energy, atom_ene_lj,
                                store_energy, store_virial);
                }
                else
                {
                    const int pair_type =
                        Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                    stored =
                        SITS_Store_Clustered_Pair<
                            full_output, false, correction_only,
                            rest2_correction>(
                            atom_i, atom_j, global_i, global_j,
                            j_is_local, r1, r2, dr,
                            lj_ab_packed[pair_type], sqrtf(dr2),
                            pme_beta, atom_sys_mark, pwwp_factor, frc,
                            frc_enhancing, atom_energy,
                            atom_energy_enhancing, atom_virial,
                            atom_virial_enhancing,
                            atom_direct_cf_energy, atom_ene_lj,
                            store_energy, store_virial,
                            rest2_unscaled_atom_energy,
                            rest2_effective_atom_energy,
                            rest2_lambda_m, rest2_sqrt_lambda_m);
                }
                if (j_is_local)
                {
                    if constexpr (!correction_only || rest2_correction)
                    {
                        force_j = force_j - stored.force_i;
                    }
                    if constexpr (!rest2_correction)
                    {
                        if (stored.selective_owner && !stored.owner_is_i)
                        {
                            force_j_enhancing =
                                force_j_enhancing -
                                stored.selective_factor * stored.force_i;
                        }
                    }
                }
            }
            if (j_is_local)
            {
                if constexpr (!correction_only || rest2_correction)
                {
                    const float reduced_force =
                        SITS_Reduce_Subgroup_Force_To_Component(
                            force_j.x, force_j.y, force_j.z, i_lane);
                    if (i_lane < 3)
                    {
                        SITS_Atomic_Add_Force_Component(
                            frc, atom_j, i_lane, reduced_force);
                    }
                }
                if constexpr (!rest2_correction)
                {
                    if (atom_sys_mark[atom_j] == 0)
                    {
                        const float reduced_enhancing =
                            SITS_Reduce_Subgroup_Force_To_Component(
                                force_j_enhancing.x,
                                force_j_enhancing.y,
                                force_j_enhancing.z, i_lane);
                        if (i_lane < 3)
                        {
                            SITS_Atomic_Add_Force_Component(
                                frc_enhancing, atom_j, i_lane,
                                reduced_enhancing);
                        }
                    }
                }
            }
        }
    }
#else
    (void)sci_numbers;
    (void)packed_partitions;
    (void)cluster_numbers;
    (void)local_atom_numbers;
    (void)cluster_offsets;
    (void)cluster_valid_masks;
    (void)cluster_local_masks;
    (void)super_cluster_offsets;
    (void)sci_entries;
    (void)cjpacked_entries;
    (void)exclusion_entries;
    (void)pair_shift_bits;
    (void)sorted_atom_ids;
    (void)sorted_xq;
    (void)sorted_lj_type;
    (void)sorted_soft_crd;
    (void)atom_local;
    (void)atom_sys_mark;
    (void)cell;
    (void)lj_ab_packed;
    (void)lj_aa;
    (void)lj_ab;
    (void)lj_ba;
    (void)lj_bb;
    (void)lambda;
    (void)alpha;
    (void)soft_p;
    (void)sigma_6;
    (void)sigma_6_min;
    (void)cutoff;
    (void)frc;
    (void)frc_enhancing;
    (void)pme_beta;
    (void)atom_energy;
    (void)atom_energy_enhancing;
    (void)atom_virial;
    (void)atom_virial_enhancing;
    (void)atom_direct_cf_energy;
    (void)atom_ene_lj;
    (void)pwwp_factor;
    (void)store_energy;
    (void)store_virial;
#endif
}

#ifdef USE_CPU
template <bool full_output, bool correction_only = false,
          bool soft_core = false, bool rest2_correction = false>
static void SITS_Clustered_Native_Direct(
    const CLUSTERED_SPATIAL_VIEW& view,
    const int* sorted_atom_ids, const float4* sorted_xq,
    const int* sorted_lj_type,
    const VECTOR_LJ_SOFT_TYPE* sorted_soft_crd, const int* atom_local,
    const int* atom_sys_mark, const LTMatrix3 cell, const LTMatrix3 rcell,
    const float* lj_a, const float* lj_b, const float* lj_aa,
    const float* lj_ab, const float* lj_ba, const float* lj_bb,
    const float lambda, const float alpha, const float soft_p,
    const float sigma_6, const float sigma_6_min,
    const float cutoff, VECTOR* frc,
    VECTOR* frc_enhancing, const float pme_beta, float* atom_energy,
    float* atom_energy_enhancing, LTMatrix3* atom_virial,
    LTMatrix3* atom_virial_enhancing, float* atom_direct_cf_energy,
    float* atom_ene_lj, const float pwwp_factor, const bool store_energy,
    const bool store_virial, float* rest2_unscaled_atom_energy,
    float* rest2_effective_atom_energy, const float rest2_lambda_m,
    const float rest2_sqrt_lambda_m)
{
    const float cutoff_sq = cutoff * cutoff;
#pragma omp parallel for schedule(dynamic)
    for (int sci = 0; sci < view.sci_numbers; sci += 1)
    {
        const CLUSTERED_SCI sci_entry = view.sci[sci];
        const int cluster_i_begin =
            view.super_cluster_offsets[sci_entry.supercluster_id];
        const int cluster_i_end =
            view.super_cluster_offsets[sci_entry.supercluster_id + 1];
        for (int cluster_i = cluster_i_begin; cluster_i < cluster_i_end;
             cluster_i += 1)
        {
            const int i_local = cluster_i - cluster_i_begin;
            for (int i_lane = 0; i_lane < view.cluster_size; i_lane += 1)
            {
                const unsigned int i_lane_bit = 1u << i_lane;
                if ((view.cluster_valid_masks[cluster_i] & i_lane_bit) == 0u ||
                    (view.cluster_local_masks[cluster_i] & i_lane_bit) == 0u)
                {
                    continue;
                }
                const int sorted_i =
                    view.cluster_offsets[cluster_i] + i_lane;
                const int atom_i = sorted_atom_ids[sorted_i];
                VECTOR_LJ r1 = {};
                VECTOR_LJ_SOFT_TYPE soft_r1 = {};
                VECTOR r1_crd = {};
                if constexpr (soft_core)
                {
                    soft_r1 = sorted_soft_crd[sorted_i];
                    r1_crd = soft_r1.crd;
                }
                else
                {
                    r1 = SITS_Make_Packed_LJ_Atom(
                        sorted_xq[sorted_i], sorted_lj_type[sorted_i]);
                    r1_crd = r1.crd;
                }
                for (int packed_idx = sci_entry.cjpacked_begin;
                     packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
                {
                    const CLUSTERED_CJ_PACKED& packed =
                        view.cjpacked[packed_idx];
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
                        const int exclusion_index =
                            Clustered_First_Exclusion_Index(
                                packed, jm, i_local);
                        const uint64_t exclusion_mask =
                            exclusion_index >= 0 &&
                                    view.exclusion_mask_pool != NULL
                                ? view.exclusion_mask_pool[exclusion_index]
                                : 0ull;
                        const VECTOR shift =
                            Clustered_Shift_Vector_From_Id(
                                sci_entry.shift_id, cell);
                        for (int j_lane = 0; j_lane < view.cluster_size;
                             j_lane += 1)
                        {
                            const unsigned int j_lane_bit = 1u << j_lane;
                            if ((view.cluster_valid_masks[cluster_j] &
                                 j_lane_bit) == 0u ||
                                (exclusion_mask &
                                 (1ull << (i_lane * view.cluster_size +
                                           j_lane))) != 0ull)
                            {
                                continue;
                            }
                            const int sorted_j =
                                view.cluster_offsets[cluster_j] + j_lane;
                            const int atom_j = sorted_atom_ids[sorted_j];
                            const bool j_is_local =
                                (view.cluster_local_masks[cluster_j] &
                                 j_lane_bit) != 0u;
                            if constexpr (rest2_correction)
                            {
                                if (atom_sys_mark[atom_i] +
                                        atom_sys_mark[atom_j] >=
                                    2)
                                {
                                    continue;
                                }
                            }
                            const int global_i = atom_local[atom_i];
                            const int global_j = atom_local[atom_j];
                            if (sci_entry.shift_id ==
                                    kClusteredCentralShiftId &&
                                cluster_i == cluster_j && j_is_local &&
                                j_lane <= i_lane)
                            {
                                continue;
                            }
                            VECTOR_LJ r2 = {};
                            VECTOR_LJ_SOFT_TYPE soft_r2 = {};
                            VECTOR r2_crd = {};
                            if constexpr (soft_core)
                            {
                                soft_r2 = sorted_soft_crd[sorted_j];
                                r2_crd = soft_r2.crd;
                            }
                            else
                            {
                                r2 = SITS_Make_Packed_LJ_Atom(
                                    sorted_xq[sorted_j],
                                    sorted_lj_type[sorted_j]);
                                r2_crd = r2.crd;
                            }
                            const VECTOR dr = (r2_crd - r1_crd) - shift;
                            const float dr2 = dr * dr;
                            if (dr2 <= 0.0f || dr2 >= cutoff_sq)
                            {
                                continue;
                            }
                            if constexpr (soft_core)
                            {
                                const int pair_type_a = Get_LJ_Type(
                                    soft_r1.LJ_type, soft_r2.LJ_type);
                                const int pair_type_b = Get_LJ_Type(
                                    soft_r1.LJ_type_B, soft_r2.LJ_type_B);
                                SITS_Store_Clustered_Soft_Pair<
                                    full_output, true, correction_only>(
                                    atom_i, atom_j, global_i, global_j,
                                    j_is_local,
                                    soft_r1, soft_r2, dr,
                                    lj_aa[pair_type_a],
                                    lj_ab[pair_type_a],
                                    lj_ba[pair_type_b],
                                    lj_bb[pair_type_b], pme_beta,
                                    lambda, alpha, soft_p, sigma_6,
                                    sigma_6_min, atom_sys_mark,
                                    pwwp_factor, frc, frc_enhancing,
                                    atom_energy, atom_energy_enhancing,
                                    atom_virial,
                                    atom_virial_enhancing,
                                    atom_direct_cf_energy, atom_ene_lj,
                                    store_energy, store_virial);
                            }
                            else
                            {
                                const int pair_type =
                                    Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                                const float2 pair_lj_ab = {
                                    lj_a[pair_type], lj_b[pair_type]};
                                SITS_Store_Clustered_Pair<
                                    full_output, true, correction_only,
                                    rest2_correction>(
                                    atom_i, atom_j, global_i, global_j,
                                    j_is_local, r1,
                                    r2, dr, pair_lj_ab, sqrtf(dr2),
                                    pme_beta, atom_sys_mark,
                                    pwwp_factor, frc, frc_enhancing,
                                    atom_energy, atom_energy_enhancing,
                                    atom_virial,
                                    atom_virial_enhancing,
                                    atom_direct_cf_energy, atom_ene_lj,
                                    store_energy, store_virial,
                                    rest2_unscaled_atom_energy,
                                    rest2_effective_atom_energy,
                                    rest2_lambda_m,
                                    rest2_sqrt_lambda_m);
                            }
                        }
                    }
                }
            }
        }
    }
}
#endif

static __device__ float log_add_log(float a, float b)
{
    return fmaxf(a, b) + logf(1.0 + expf(-fabsf(a - b)));
}

static __global__ void SITS_Record_Ene_Device(float* ene_record,
                                              const float* enhancing_energy,
                                              const float pe_a,
                                              const float pe_b)
{
    *ene_record = pe_a * *enhancing_energy + pe_b;
}

static __global__ void SITS_Update_gf_Device(const int kn, float* gf,
                                             const float* ene_record,
                                             const float* log_nk,
                                             const float* beta_k)
{
#ifdef USE_GPU
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kn)
#else
#pragma omp parallel for
    for (int i = 0; i < kn; i++)
#endif
    {
        gf[i] = -beta_k[i] * ene_record[0] + log_nk[i];
    }
}

static __global__ void SITS_Update_gfsum_Device(const int kn, float* gfsum,
                                                const float* gf)
{
    float temp = -FLT_MAX;
    for (int i = 0; i < kn; i = i + 1)
    {
        temp = log_add_log(temp, gf[i]);
    }
    gfsum[0] = temp;
}

static __global__ void SITS_Update_log_pk_Device(const int kn, float* log_pk,
                                                 const float* gf,
                                                 const float* gfsum,
                                                 const int reset)
{
#ifdef USE_GPU
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kn)
#else
#pragma omp parallel for
    for (int i = 0; i < kn; i++)
#endif
    {
        float gfi = gf[i];
        log_pk[i] =
            ((float)reset) * gfi +
            ((float)(1 - reset)) * log_add_log(log_pk[i], gfi - gfsum[0]);
    }
}

static __global__ void SITS_Update_log_mk_inverse_Device(
    const int kn, float* log_weight, float* log_mk_inverse, float* log_norm_old,
    float* log_norm, const float* log_pk, const float* log_nk)
{
#ifdef USE_GPU
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kn - 1)
#else
#pragma omp parallel for
    for (int i = 0; i < kn - 1; i++)
#endif
    {
        log_weight[i] = (log_pk[i] + log_pk[i + 1]) * 0.5;
        log_mk_inverse[i] = log_nk[i] - log_nk[i + 1];
        log_norm_old[i] = log_norm[i];
        log_norm[i] = log_add_log(log_norm[i], log_weight[i]);
        log_mk_inverse[i] =
            log_add_log(log_mk_inverse[i] + log_norm_old[i] - log_norm[i],
                        log_pk[i + 1] - log_pk[i] + log_mk_inverse[i] +
                            log_weight[i] - log_norm[i]);
    }
}

static __global__ void SITS_Update_log_nk_inverse_Device(
    const int kn, float* log_nk_inverse, const float* log_mk_inverse)
{
    for (int i = 0; i < kn - 1; i++)
    {
        log_nk_inverse[i + 1] = log_nk_inverse[i] + log_mk_inverse[i];
    }
}

static __global__ void SITS_Update_nk_Device(const int kn, float* log_nk,
                                             float* nk,
                                             const float* log_nk_inverse)
{
#ifdef USE_GPU
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kn)
#else
#pragma omp parallel for
    for (int i = 0; i < kn; i++)
#endif
    {
        log_nk[i] = -log_nk_inverse[i];
        nk[i] = exp(log_nk[i]);
    }
}

static __global__ void SITS_For_Enhanced_Force_Calculate_NkExpBetakU_Device(
    const int k_numbers, const float* beta_k, const float* log_nk,
    float* nkexpbetaku, const float* ene, const float beta0, const float pe_a,
    const float pe_b)
{
#ifdef USE_GPU
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < k_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < k_numbers; i++)
#endif
    {
        nkexpbetaku[i] =
            -(beta_k[i] - beta0) * (pe_a * ene[0] + pe_b) + log_nk[i];
    }
}

static __global__ void SITS_For_Enhanced_Force_Sum_Of_Above_And_Below_Device(
    const int k_numbers, const float* nkexpbetaku, const float* beta_k,
    float* d_bias, float pe_a, float pe_b, float* sum_of_above,
    float* sum_of_below, float* factor, float beta0, float fb_bias,
    const float* h_enhancing_energy)
{
    float above = -FLT_MAX;
    float below = -FLT_MAX;
    for (int i = 0; i < k_numbers; i++)
    {
        above = log_add_log(above, logf(beta_k[i]) + nkexpbetaku[i]);
        below = log_add_log(below, nkexpbetaku[i]);
    }
    sum_of_above[0] = above;
    sum_of_below[0] = below;
    factor[0] = expf(above - below - logf(beta0)) + fb_bias;
    d_bias[0] =
        -below / beta0 / pe_a + fb_bias * (h_enhancing_energy[0] + pe_b / pe_a);
}

static __global__ void SITS_For_Enhanced_Force_Protein_Water_Device(
    const int atom_numbers, VECTOR* md_frc, const VECTOR* enhancing_frc,
    float* md_ene, const float* bias, const int need_pressure,
    LTMatrix3* md_virial, const LTMatrix3* virial_enhancing,
    const float factor_minus_one)
{
#ifdef USE_GPU
    if (blockIdx.x == 0 && threadIdx.x == 0)
#endif
    {
        md_ene[0] = md_ene[0] + bias[0];
        if (need_pressure)
        {
            md_virial[0] =
                md_virial[0] + factor_minus_one * virial_enhancing[0];
        }
    }
#ifdef USE_GPU
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < atom_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < atom_numbers; i++)
#endif
    {
        md_frc[i] = md_frc[i] + factor_minus_one * enhancing_frc[i];
    }
}

static __global__ void ESITS_Get_Current_Fb(const float* enhancing_energy,
                                            float* factor, const float pe_a,
                                            const float pe_b,
                                            const float beta_high,
                                            const float beta_low, float* d_bias)
{
    float ene = enhancing_energy[0];
    if (ene > pe_b)
    {
        factor[0] =
            beta_high - (beta_high - beta_low) * pe_a / (ene - pe_b + pe_a);
        d_bias[0] = -pe_a * logf(enhancing_energy[0] - pe_b + pe_a);
    }
    else
    {
        factor[0] = beta_low;
        d_bias[0] = -enhancing_energy[0];
    }
}

static __global__ void AMD_Get_Current_Fb(const float* enhancing_energy,
                                          float* factor, const float pe_a,
                                          const float pe_b, float* d_bias)
{
    float ene = enhancing_energy[0];
    if (ene < pe_b)
    {
        ene = pe_b - ene;
        factor[0] = 1.0f - ene * (2 * pe_a + ene) / (pe_a + ene) / (pe_a + ene);
        d_bias[0] = ene * ene / (pe_a + ene);
    }
    else
    {
        factor[0] = 1.0f;
        d_bias[0] = 0;
    }
}

static __global__ void GAMD_Get_Current_Fb(const float* enhancing_energy,
                                           float* factor, const float pe_a,
                                           const float pe_b, float* d_bias)
{
    float ene = enhancing_energy[0];
    if (ene < pe_b)
    {
        factor[0] = 1.0f - pe_a * ene;
        d_bias[0] = 0.5 * pe_a * ene * ene;
    }
    else
    {
        factor[0] = 1.0f;
        d_bias[0] = 0;
    }
}

static void SITS_Get_Current_Fb(const int atom_numbers,
                                const float* energy_enhancing, float* d_bias,
                                const int k_numbers, float* nkexpbetaku,
                                const float* beta_k, const float* log_nk,
                                const float beta0, float* sum_a, float* sum_b,
                                float* factor, const float fb_bias,
                                const float pe_a, const float pe_b,
                                const float pwwp_enhance_factor)
{
    Launch_Device_Kernel(SITS_For_Enhanced_Force_Calculate_NkExpBetakU_Device,
                         (k_numbers + 63) / 64, 64, 0, NULL, k_numbers, beta_k,
                         log_nk, nkexpbetaku, energy_enhancing, beta0, pe_a,
                         pe_b);

    Launch_Device_Kernel(SITS_For_Enhanced_Force_Sum_Of_Above_And_Below_Device,
                         1, 1, 0, NULL, k_numbers, nkexpbetaku, beta_k, d_bias,
                         pe_a, pe_b, sum_a, sum_b, factor, beta0, fb_bias,
                         energy_enhancing);
}

void CLASSIC_SITS_INFORMATION::Initial(CONTROLLER* controller,
                                       SITS_INFORMATION* sits)
{
    is_initialized = 1;
    sits_controller = sits;
    record_count = 0;
    fb_interval = 1;
    Device_Malloc_Safely((void**)&d_bias, sizeof(float));
    if (controller->Command_Exist(sits->module_name, "fb_interval"))
    {
        controller->Check_Int(sits->module_name, "fb_interval",
                              "CLASSIC_SITS_INFORMATION::Initial");
        fb_interval =
            atoi(controller->Command(sits->module_name, "fb_interval"));
    }
    controller->printf("    SITS fb update interval set to %d\n", fb_interval);
    if (sits->sits_mode == SITS_MODE_AMD)
    {
        if (controller->Command_Exist(sits->module_name, "pe_a"))
        {
            controller->Check_Float(sits->module_name, "pe_a",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            pe_a = atof(controller->Command(sits->module_name, "pe_a"));
        }
        else
        {
            controller->Throw_SPONGE_Error(
                spongeErrorMissingCommand, "CLASSIC_SITS_INFORMATION::Initial",
                "Reason:\n\tAlpha (pe_a) is required for the Accelerated MD");
        }
        controller->printf("    AMD alpha (pe_a) set to %f\n", pe_a);

        if (controller->Command_Exist(sits->module_name, "pe_b"))
        {
            controller->Check_Float(sits->module_name, "pe_b",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            pe_b = atof(controller->Command(sits->module_name, "pe_b"));
        }
        else
        {
            controller->Throw_SPONGE_Error(
                spongeErrorMissingCommand, "CLASSIC_SITS_INFORMATION::Initial",
                "Reason:\n\tE (pe_b) is required for the Accelerated MD");
        }
        controller->printf("    AMD E (pe_b) set to %f\n", pe_b);

        k_numbers = 0;
        nk_fix = 1;
        record_interval = 1;
        update_interval = INT_MAX;
        Memory_Allocate();
    }
    else if (sits->sits_mode == SITS_MODE_GAMD)
    {
        if (controller->Command_Exist(sits->module_name, "pe_a"))
        {
            controller->Check_Float(sits->module_name, "pe_a",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            pe_a = atof(controller->Command(sits->module_name, "pe_a"));
        }
        else
        {
            controller->Throw_SPONGE_Error(spongeErrorMissingCommand,
                                           "CLASSIC_SITS_INFORMATION::Initial",
                                           "Reason:\n\tk (pe_a) is required "
                                           "for the Gaussian Accelerated MD");
        }
        controller->printf("    GAMD k (pe_a) set to %f\n", pe_a);

        if (controller->Command_Exist(sits->module_name, "pe_b"))
        {
            controller->Check_Float(sits->module_name, "pe_b",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            pe_b = atof(controller->Command(sits->module_name, "pe_b"));
        }
        else
        {
            controller->Throw_SPONGE_Error(
                spongeErrorMissingCommand, "CLASSIC_SITS_INFORMATION::Initial",
                "Reason:\n\tE (pe_b) is required for the Accelerated MD");
        }
        controller->printf("    GAMD E (pe_b) set to %f\n", pe_b);

        k_numbers = 0;
        nk_fix = 1;
        record_interval = 1;
        update_interval = INT_MAX;
        Memory_Allocate();
    }
    else if (sits->sits_mode == SITS_MODE_EMPIRICAL)
    {
        if (controller->Command_Exist(sits->module_name, "pe_a"))
        {
            controller->Check_Float(sits->module_name, "pe_a",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            pe_a = atof(controller->Command(sits->module_name, "pe_a"));
        }
        else
        {
            pe_a = 1.0;
        }
        controller->printf("    SITS_pe_a set to %f\n", pe_a);

        if (controller->Command_Exist(sits->module_name, "pe_b"))
        {
            controller->Check_Float(sits->module_name, "pe_b",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            pe_b = atof(controller->Command(sits->module_name, "pe_b"));
        }
        else
        {
            pe_b = 0.0;
        }
        controller->printf("    SITS_pe_b set to %f\n", pe_b);

        if (!controller->Command_Exist(sits->module_name, "T_low") ||
            !controller->Command_Exist(sits->module_name, "T_high"))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorMissingCommand, "CLASSIC_SITS_INFORMATION::Initial",
                "Reason:\n\tSITS_T_high and SITS_T_low are required for "
                "empirical SITS");
        }
        controller->Check_Float(sits->module_name, "T_low",
                                "CLASSIC_SITS_INFORMATION::Initial");
        controller->Check_Float(sits->module_name, "T_high",
                                "CLASSIC_SITS_INFORMATION::Initial");
        T_low = atof(controller->Command(sits->module_name, "T_low"));
        T_high = atof(controller->Command(sits->module_name, "T_high"));
        controller->printf("    SITS_T_high set to %f\n", T_high);
        controller->printf("    SITS_T_low set to %f\n", T_low);

        k_numbers = 0;
        nk_fix = 1;
        record_interval = 1;
        update_interval = INT_MAX;
        Memory_Allocate();
    }
    else if (sits->sits_mode != SITS_MODE_OBSERVATION)
    {
        if (controller->Command_Exist(sits->module_name, "k_numbers"))
        {
            controller->Check_Int(sits->module_name, "k_numbers",
                                  "CLASSIC_SITS_INFORMATION::Initial");
            k_numbers = atoi(controller->Command("SITS_k_numbers"));
            if (k_numbers <= 0)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand,
                    "CLASSIC_SITS_INFORMATION::Initial",
                    "Reason:\n\tSITS k numbers cannot be smaller than 0\n");
            }
        }
        else
        {
            k_numbers = 40;
        }
        controller->printf("    k numbers is %d\n", k_numbers);
        Memory_Allocate();

        controller->printf("    Read %s temperature information.\n",
                           sits->module_name);
        float* beta_k_tmp;
        Malloc_Safely((void**)&beta_k_tmp, sizeof(float) * k_numbers);
        if (controller->Command_Exist(sits->module_name, "T_low"))
        {
            if (!controller->Command_Exist(sits->module_name, "T_high"))
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorMissingCommand,
                    "CLASSIC_SITS_INFORMATION::Initial",
                    "Reason:\n\tSITS T high must be explicitly given with SITS "
                    "T low in mdin\n");
            }
            controller->Check_Float(sits->module_name, "T_low",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            controller->Check_Float(sits->module_name, "T_high",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            T_low = atof(controller->Command(sits->module_name, "T_low"));
            T_high = atof(controller->Command(sits->module_name, "T_high"));
            float T_space = (T_high - T_low) / (k_numbers - 1);
            for (int i = 0; i < k_numbers; ++i)
            {
                beta_k_tmp[i] = 1.0 / (CONSTANT_kB * (T_low + T_space * i));
            }
        }
        else if (controller->Command_Exist(sits->module_name, "T"))
        {
            const char* char_pt = controller->Command(sits->module_name, "T");
            for (int i = 0; i < k_numbers; ++i)
            {
                float tmp_T;
                sscanf(char_pt, "%f", &tmp_T);
                if (i != k_numbers - 1)
                {
                    while (*char_pt != '/' && *char_pt != '\0') ++char_pt;
                    if (*char_pt == '/') ++char_pt;
                    if (*char_pt == '\0')
                    {
                        controller->Throw_SPONGE_Error(
                            spongeErrorValueErrorCommand,
                            "CLASSIC_SITS_INFORMATION::Initial",
                            "Reason:\n\tthe number of temperatures SITS_T != "
                            "SITS_k_numbers\n");
                    }
                }
                beta_k_tmp[i] = 1.0 / (CONSTANT_kB * tmp_T);
            }
        }
        else
        {
            controller->Throw_SPONGE_Error(
                spongeErrorMissingCommand, "CLASSIC_SITS_INFORMATION::Initial",
                "Reason:\n\tSITS T must be explicitly given in mdin.\n");
        }
        deviceMemcpy(beta_k, beta_k_tmp, sizeof(float) * k_numbers,
                     deviceMemcpyHostToDevice);
        free(beta_k_tmp);
        if (controller->Command_Exist(sits->module_name, "record_interval"))
        {
            controller->Check_Int(sits->module_name, "record_interval",
                                  "CLASSIC_SITS_INFORMATION::Initial");
            record_interval =
                atoi(controller->Command(sits->module_name, "record_interval"));
        }
        else
        {
            record_interval = 1;
        }
        controller->printf("    SITS record interval set to %d\n",
                           record_interval);

        if (controller->Command_Exist(sits->module_name, "update_interval"))
        {
            controller->Check_Int(sits->module_name, "update_interval",
                                  "CLASSIC_SITS_INFORMATION::Initial");
            update_interval =
                atoi(controller->Command(sits->module_name, "update_interval"));
        }
        else
        {
            update_interval = 100;
        }
        controller->printf("    SITS update interval set to %d\n",
                           update_interval);

        if (controller->Command_Exist(sits->module_name, "pe_a"))
        {
            controller->Check_Float(sits->module_name, "pe_a",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            pe_a = atof(controller->Command(sits->module_name, "pe_a"));
        }
        else
        {
            pe_a = 1.0;
        }
        controller->printf("    SITS_pe_a set to %f\n", pe_a);

        if (controller->Command_Exist(sits->module_name, "pe_b"))
        {
            controller->Check_Float(sits->module_name, "pe_b",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            pe_b = atof(controller->Command(sits->module_name, "pe_b"));
        }
        else
        {
            pe_b = 0.0;
        }
        controller->printf("    SITS_pe_b set to %f\n", pe_b);

        if (controller->Command_Exist(sits->module_name, "fb_bias"))
        {
            controller->Check_Float(sits->module_name, "fb_bias",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            fb_bias = atof(controller->Command(sits->module_name, "fb_bias"));
        }
        else
        {
            fb_bias = 0.0;
        }
        controller->printf("    SITS_fb_bias set to %f\n", fb_bias);

        reset = 1;

        int nk_rest;
        if (sits->sits_mode == SITS_MODE_ITERATION)
        {
            nk_rest = 0;
        }
        else
        {
            nk_rest = 1;
        }
        if (controller->Command_Exist(sits->module_name, "nk_rest"))
        {
            nk_rest = controller->Get_Bool(sits->module_name, "nk_rest",
                                           "CLASSIC_SITS_INFORMATION::Initial");
        }
        float* beta_lin;
        Malloc_Safely((void**)&beta_lin, sizeof(float) * k_numbers);

        for (int i = 0; i < k_numbers; ++i) beta_lin[i] = -FLT_MAX;

        deviceMemcpy(log_norm_old, beta_lin, sizeof(float) * k_numbers,
                     deviceMemcpyHostToDevice);
        deviceMemcpy(log_norm, beta_lin, sizeof(float) * k_numbers,
                     deviceMemcpyHostToDevice);
        deviceMemset(log_nk_inverse, 0, sizeof(float) * k_numbers);

        if (nk_rest == 0)
        {
            for (int i = 0; i < k_numbers; ++i)
            {
                beta_lin[i] = 0.0;
            }
        }
        else
        {
            FILE* nk_read_file;
            if (controller->Command_Exist(sits->module_name, "nk_in_file"))
            {
                controller->printf(
                    "    Read Nk from %s\n",
                    controller->Command(sits->module_name, "nk_in_file"));
                Open_File_Safely(
                    &nk_read_file,
                    controller->Command(sits->module_name, "nk_in_file"), "r");
                for (int i = 0; i < k_numbers; ++i)
                {
                    int retval = fscanf(nk_read_file, "%f", beta_lin + i);
                    beta_lin[i] = logf(beta_lin[i]);
                }
            }
            else
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorMissingCommand,
                    "CLASSIC_SITS_INFORMATION::Initial",
                    "Reason:\n\tSITS_nk_in_file must be given when "
                    "SITS_nk_rest = 1 or SITS_mode = production\n");
            }
        }
        deviceMemcpy(log_nk, beta_lin, sizeof(float) * k_numbers,
                     deviceMemcpyHostToDevice);

        for (int i = 0; i < k_numbers; ++i)
        {
            beta_lin[i] = -beta_lin[i];
        }
        deviceMemcpy(log_nk_inverse, beta_lin, sizeof(float) * k_numbers,
                     deviceMemcpyHostToDevice);

        for (int i = 0; i < k_numbers; ++i)
        {
            beta_lin[i] = expf(-beta_lin[i]);
        }
        deviceMemcpy(Nk, beta_lin, sizeof(float) * k_numbers,
                     deviceMemcpyHostToDevice);

        free(beta_lin);
        Reset_List(factor, 1.0, 1);

        if (controller->Command_Exist(sits->module_name, "nk_fix"))
        {
            nk_fix = controller->Get_Bool(sits->module_name, "nk_fix",
                                          "CLASSIC_SITS_INFORMATION::Initial");
        }
        else if (sits->sits_mode == SITS_MODE_ITERATION)
        {
            nk_fix = 0;
        }
        else
        {
            nk_fix = 1;
        }
        controller->printf("    SITS nk fix is: %d\n", nk_fix);
        if (nk_fix == 0)
        {
            if (controller->Command_Exist(sits->module_name, "nk_rest_file"))
            {
                strcpy(nk_rest_file_name,
                       controller->Command(sits->module_name, "nk_rest_file"));
            }
            else if (controller->Command_Exist("default_out_file_prefix"))
            {
                strcpy(nk_rest_file_name,
                       controller->Command("default_out_file_prefix"));
                strcat(nk_rest_file_name, "_");
                strcat(nk_rest_file_name, sits->module_name);
                strcat(nk_rest_file_name, "_nk_rest.txt");
            }
            else
            {
                strcpy(nk_rest_file_name, sits->module_name);
                strcat(nk_rest_file_name, "_nk_rest.txt");
            }
            controller->printf("    Restart Nk will be written in %s\n",
                               nk_rest_file_name);
            std::string default_name = sits->module_name;
            default_name += "_nk_traj.dat";
            if (CONTROLLER::MPI_rank == 0)
            {
                nk_traj_file = controller->Get_Output_File(
                    true, sits->module_name, "nk_traj_file", "_nk_traj.dat",
                    default_name.c_str());
            }
        }
    }
}

void CLASSIC_SITS_INFORMATION::Memory_Allocate()
{
    Malloc_Safely((void**)&nk_record_cpu, sizeof(float) * k_numbers);
    Malloc_Safely((void**)&log_norm_record_cpu, sizeof(float) * k_numbers);

    Device_Malloc_Safely((void**)&ene_recorded, sizeof(float));
    Device_Malloc_Safely((void**)&gf, sizeof(float) * k_numbers);
    Device_Malloc_Safely((void**)&gfsum, sizeof(float));
    Device_Malloc_Safely((void**)&log_weight, sizeof(float) * k_numbers);
    Device_Malloc_Safely((void**)&log_mk_inverse, sizeof(float) * k_numbers);
    Device_Malloc_Safely((void**)&log_norm_old, sizeof(float) * k_numbers);
    Device_Malloc_And_Copy_Safely((void**)&log_norm, log_norm_record_cpu,
                                  sizeof(float) * k_numbers);
    Device_Malloc_Safely((void**)&log_pk, sizeof(float) * k_numbers);
    Device_Malloc_Safely((void**)&log_nk_inverse, sizeof(float) * k_numbers);
    Device_Malloc_Safely((void**)&log_nk, sizeof(float) * k_numbers);

    Device_Malloc_Safely((void**)&beta_k, sizeof(float) * k_numbers);
    Device_Malloc_Safely((void**)&NkExpBetakU, sizeof(float) * k_numbers);
    Device_Malloc_And_Copy_Safely((void**)&Nk, nk_record_cpu,
                                  sizeof(float) * k_numbers);
    Device_Malloc_Safely((void**)&sum_a, sizeof(float));
    Device_Malloc_Safely((void**)&sum_b, sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&factor, &sits_controller->h_factor,
                                  sizeof(float));
}

void CLASSIC_SITS_INFORMATION::SITS_Record_Ene()
{
    Launch_Device_Kernel(SITS_Record_Ene_Device, 1, 1, 0, NULL, ene_recorded,
                         sits_controller->pw_select.select_energy[0], pe_a,
                         pe_b);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_gf()
{
    Launch_Device_Kernel(SITS_Update_gf_Device, (k_numbers + 63) / 64, 64, 0,
                         NULL, k_numbers, gf, ene_recorded, log_nk, beta_k);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_gfsum()
{
    Launch_Device_Kernel(SITS_Update_gfsum_Device, 1, 1, 0, NULL, k_numbers,
                         gfsum, gf);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_log_pk()
{
    Launch_Device_Kernel(SITS_Update_log_pk_Device, (k_numbers + 63) / 64, 64,
                         0, NULL, k_numbers, log_pk, gf, gfsum, reset);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_log_mk_inverse()
{
    Launch_Device_Kernel(SITS_Update_log_mk_inverse_Device,
                         (k_numbers + 63) / 64, 64, 0, NULL, k_numbers,
                         log_weight, log_mk_inverse, log_norm_old, log_norm,
                         log_pk, log_nk);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_log_nk_inverse()
{
    Launch_Device_Kernel(SITS_Update_log_nk_inverse_Device, 1, 1, 0, NULL,
                         k_numbers, log_nk_inverse, log_mk_inverse);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_nk()
{
    Launch_Device_Kernel(SITS_Update_nk_Device, (k_numbers + 63) / 64, 64, 0,
                         NULL, k_numbers, log_nk, Nk, log_nk_inverse);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_Fb(float beta_0, int step)
{
    if (!is_initialized ||
        sits_controller->sits_mode == SITS_MODE_OBSERVATION ||
        step % fb_interval != 0)
    {
        return;
    }
    if (sits_controller->sits_mode < SITS_MODE_EMPIRICAL)
    {
        SITS_Get_Current_Fb(sits_controller->atom_numbers,
                            sits_controller->pw_select.select_energy[0], d_bias,
                            k_numbers, NkExpBetakU, beta_k, log_nk, beta_0,
                            sum_a, sum_b, factor, fb_bias, pe_a, pe_b,
                            sits_controller->pwwp_enhance_factor);
        deviceMemcpy(&sits_controller->h_factor, factor, sizeof(float),
                     deviceMemcpyDeviceToHost);
    }
    else if (sits_controller->sits_mode == SITS_MODE_EMPIRICAL)
    {
        Launch_Device_Kernel(ESITS_Get_Current_Fb, 1, 1, 0, NULL,
                             sits_controller->pw_select.select_energy[0],
                             factor, pe_a, pe_b,
                             1.0f / (beta_0 * T_low * CONSTANT_kB),
                             1.0f / (beta_0 * T_high * CONSTANT_kB), d_bias);
        deviceMemcpy(&sits_controller->h_factor, factor, sizeof(float),
                     deviceMemcpyDeviceToHost);
    }
    else if (sits_controller->sits_mode == SITS_MODE_AMD)
    {
        Launch_Device_Kernel(AMD_Get_Current_Fb, 1, 1, 0, NULL,
                             sits_controller->pw_select.select_energy[0],
                             factor, pe_a, pe_b, d_bias);
        deviceMemcpy(&sits_controller->h_factor, factor, sizeof(float),
                     deviceMemcpyDeviceToHost);
    }
    else if (sits_controller->sits_mode == SITS_MODE_GAMD)
    {
        Launch_Device_Kernel(GAMD_Get_Current_Fb, 1, 1, 0, NULL,
                             sits_controller->pw_select.select_energy[0],
                             factor, pe_a, pe_b, d_bias);
        deviceMemcpy(&sits_controller->h_factor, factor, sizeof(float),
                     deviceMemcpyDeviceToHost);
    }
}

void CLASSIC_SITS_INFORMATION::SITS_Update_Common(const float beta)
{
    if (sits_controller->sits_mode != SITS_MODE_EMPIRICAL)
    {
        SITS_Record_Ene();
        SITS_Update_gf();
        SITS_Update_gfsum();
        SITS_Update_log_pk();
        reset = 0;
        record_count++;
    }
}

void CLASSIC_SITS_INFORMATION::SITS_Update_Nk()
{
    if (sits_controller->sits_mode != SITS_MODE_EMPIRICAL)
    {
        SITS_Update_log_mk_inverse();
        SITS_Update_log_nk_inverse();
        SITS_Update_nk();

        record_count = 0;
        reset = 1;

        SITS_Write_Nk_Norm();
    }
}

void CLASSIC_SITS_INFORMATION::SITS_Write_Nk_Norm()
{
#ifdef USE_MPI
    if (CONTROLLER::MPI_rank != 0) return;
#endif
    deviceMemcpy(nk_record_cpu, Nk, sizeof(float) * k_numbers,
                 deviceMemcpyDeviceToHost);
    if (nk_traj_file != NULL)
    {
        fwrite(nk_record_cpu, sizeof(float), k_numbers, nk_traj_file);
    }

    Open_File_Safely(&nk_rest_file, nk_rest_file_name, "w");
    for (int i = 0; i < k_numbers; ++i)
    {
        fprintf(nk_rest_file, "%e ", nk_record_cpu[i]);
    }
    fclose(nk_rest_file);
}

void SITS_INFORMATION::Initial(CONTROLLER* controller, int atom_numbers_,
                               const char* given_module_name)
{
    if (given_module_name == NULL)
    {
        strcpy(module_name, "SITS");
        strcpy(print_aa_kab_name, "SITS");
        strcpy(print_bias_name, "SITS");
        strcpy(print_fb_name, "SITS");
    }
    else
    {
        strcpy(module_name, given_module_name);
        strcpy(print_aa_kab_name, given_module_name);
        strcpy(print_bias_name, given_module_name);
        strcpy(print_fb_name, given_module_name);
    }
    strcat(print_aa_kab_name, "_AA_kAB");
    strcat(print_bias_name, "_bias");
    strcat(print_fb_name, "_fb");
    if (controller->Command_Exist(module_name, "mode"))
    {
        if (controller->Command_Choice(module_name, "mode", "observation"))
        {
            controller->printf(
                "START INITIALIZING %s\n    %s mode = observation\n",
                module_name, module_name);
            is_initialized = 1;
            sits_mode = SITS_MODE_OBSERVATION;
        }
        else if (controller->Command_Choice(module_name, "mode", "iteration"))
        {
            controller->printf(
                "START INITIALIZING %s\n    %s mode = iteration\n", module_name,
                module_name);
            is_initialized = 1;
            sits_mode = SITS_MODE_ITERATION;
        }
        else if (controller->Command_Choice(module_name, "mode", "production"))
        {
            controller->printf(
                "START INITIALIZING %s\n    %s mode = production\n",
                module_name, module_name);
            is_initialized = 1;
            sits_mode = SITS_MODE_PRODUCTION;
        }
        else if (controller->Command_Choice(module_name, "mode", "empirical"))
        {
            controller->printf(
                "START INITIALIZING %s\n    %s mode = empirical\n", module_name,
                module_name);
            is_initialized = 1;
            sits_mode = SITS_MODE_EMPIRICAL;
        }
        else if (controller->Command_Choice(module_name, "mode", "amd"))
        {
            controller->printf(
                "START INITIALIZING %s\n    %s mode = AMD (Accelerated MD)\n",
                module_name, module_name);
            is_initialized = 1;
            sits_mode = SITS_MODE_AMD;
        }
        else if (controller->Command_Choice(module_name, "mode", "gamd"))
        {
            controller->printf(
                "START INITIALIZING %s\n    %s mode = GAMD (Gaussian "
                "Accelerated MD)\n",
                module_name, module_name);
            is_initialized = 1;
            sits_mode = SITS_MODE_GAMD;
        }
        else
        {
            return;
        }
        atom_numbers = atom_numbers_;
        controller->printf("\tAtom numbers is %d\n", atom_numbers);
        Memory_Allocate();

        pw_select.Initial();
        pw_select.Add_One_Energy(atom_numbers);
        pw_select.Add_One_Force(atom_numbers);
        pw_select.Add_One_Virial(atom_numbers);

        if (controller->Command_Exist(module_name, "cross_enhance_factor"))
        {
            controller->Check_Float(module_name, "cross_enhance_factor",
                                    "SITS_INFORMATION::Initial");
            pwwp_enhance_factor =
                atof(controller->Command(module_name, "cross_enhance_factor"));
        }
        else
        {
            pwwp_enhance_factor = 0.5;
        }
        controller->printf("\tpwwp enhance factor set to %f\n",
                           pwwp_enhance_factor);

        this->selectively_applied = true;
        if (controller->Command_Exist(module_name, "atom_in_file") ||
            controller->Command_Exist(module_name, "atom_numbers"))
        {
            controller->printf("    Set atom atribution information\n");
            int* atom_sys_mark_cpu;
            Malloc_Safely((void**)&atom_sys_mark_cpu,
                          sizeof(int) * atom_numbers);
            if (controller->Command_Exist(module_name, "atom_in_file"))
            {
                for (int i = 0; i < atom_numbers; i++)
                {
                    atom_sys_mark_cpu[i] = 1;
                }
                controller->printf("    reading %s_atom_in_file\n",
                                   module_name);
                FILE* fr = NULL;
                int temp_atom;
                Open_File_Safely(
                    &fr, controller->Command(module_name, "atom_in_file"), "r");
                while (fscanf(fr, "%d", &temp_atom) != EOF)
                {
                    atom_sys_mark_cpu[temp_atom] = 0;
                }
                fclose(fr);
            }
            else if (strcmp(controller->Command(module_name, "atom_numbers"),
                            "ITS") == 0 ||
                     strcmp(controller->Command(module_name, "atom_numbers"),
                            "ALL") == 0)
            {
                this->selectively_applied = false;
            }
            else
            {
                controller->Check_Int(module_name, "atom_numbers",
                                      "SITS_INFORMATION::Initial");
                int protein_numbers =
                    atoi(controller->Command(module_name, "atom_numbers"));
                for (int i = 0; i < protein_numbers; i++)
                {
                    atom_sys_mark_cpu[i] = 0;
                }
                for (int i = protein_numbers; i < atom_numbers; i++)
                {
                    atom_sys_mark_cpu[i] = 1;
                }
            }
            deviceMemcpy(atom_sys_mark, atom_sys_mark_cpu,
                         sizeof(int) * atom_numbers, deviceMemcpyHostToDevice);
            free(atom_sys_mark_cpu);
        }
        else
        {
            controller->Throw_SPONGE_Error(
                spongeErrorMissingCommand, "SITS_INFORMATION::Initial",
                "Reason:\n\tAtom information must be given in the form of "
                "SITS_atom_in_file or SITS_atom_numbers\n");
        }

        classic_sits.Initial(controller, this);

        h_factor = 1.0f;

        controller->Step_Print_Initial(print_aa_kab_name, "%.2f");
        controller->Step_Print_Initial(print_bias_name, "%.4f");
        controller->Step_Print_Initial(print_fb_name, "%.4f");

        controller->printf("END INTIALIZING %s\n\n", module_name);
    }
    else
    {
        is_initialized = 0;
        return;
    }
}

void SITS_INFORMATION::Memory_Allocate()
{
    Device_Malloc_Safely((void**)&atom_sys_mark, sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&atom_sys_mark_local,
                         sizeof(int) * atom_numbers);
}

void SITS_INFORMATION::Reset_Force_Energy(int* md_need_potential)
{
    if (!is_initialized) return;
    md_need_potential[0] += 1;

    deviceMemset(pw_select.select_atom_energy[0], 0,
                 sizeof(float) * atom_numbers);
    deviceMemset(pw_select.select_energy[0], 0, sizeof(float));
    deviceMemset(pw_select.select_force[0], 0, sizeof(VECTOR) * atom_numbers);
    deviceMemset(pw_select.select_atom_virial[0], 0,
                 sizeof(float) * atom_numbers);
    deviceMemset(pw_select.select_virial[0], 0, sizeof(float));
}

void SITS_INFORMATION::Update_And_Enhance(const int step,
                                          float* d_total_potential,
                                          int need_pressure,
                                          LTMatrix3* d_total_virial,
                                          VECTOR* frc, float beta0)
{
    if (!is_initialized) return;
    if (selectively_applied)
    {
        Sum_Of_List(pw_select.select_atom_energy[0], pw_select.select_energy[0],
                    atom_numbers);
#ifdef USE_MPI
        if (CONTROLLER::PP_MPI_size != 1)
            D_MPI_Allreduce_IN_PLACE(pw_select.select_energy[0], 1, D_MPI_FLOAT,
                                     D_MPI_SUM, CONTROLLER::d_pp_comm, NULL);
#endif
        if (need_pressure)
        {
            Sum_Of_List(pw_select.select_atom_virial[0],
                        pw_select.select_virial[0], atom_numbers);
#ifdef USE_MPI
            if (CONTROLLER::PP_MPI_size != 1)
                D_MPI_Allreduce_IN_PLACE(pw_select.select_virial[0], 6,
                                         D_MPI_FLOAT, D_MPI_SUM,
                                         CONTROLLER::d_pp_comm, NULL);
#endif
        }
    }
    else
    {
        deviceMemcpy(pw_select.select_energy[0], d_total_potential,
                     sizeof(float), deviceMemcpyDeviceToDevice);
        deviceMemcpy(pw_select.select_force[0], frc,
                     sizeof(VECTOR) * atom_numbers, deviceMemcpyDeviceToDevice);
        if (need_pressure)
        {
            deviceMemcpy(pw_select.select_virial[0], d_total_virial,
                         sizeof(float), deviceMemcpyDeviceToDevice);
        }
    }
    if (sits_mode != SITS_MODE_OBSERVATION && !classic_sits.nk_fix &&
        step % classic_sits.record_interval == 0)
    {
        classic_sits.SITS_Update_Common(beta0);
        if (classic_sits.record_count % classic_sits.update_interval == 0)
        {
            classic_sits.SITS_Update_Nk();
        }
    }
    if (sits_mode != SITS_MODE_OBSERVATION)
    {
        classic_sits.SITS_Update_Fb(beta0, step);
    }
    Launch_Device_Kernel(SITS_For_Enhanced_Force_Protein_Water_Device,
                         (atom_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, atom_numbers,
                         frc, pw_select.select_force[0], d_total_potential,
                         classic_sits.d_bias, need_pressure, d_total_virial,
                         pw_select.select_virial_tensor[0], h_factor - 1);
}

bool SITS_INFORMATION::SITS_LJ_Direct_CF_Force_Clustered(
    const int atom_numbers, const int local_atom_numbers,
    const int ghost_numbers, const VECTOR* crd, const float* charge,
    LENNARD_JONES_INFORMATION* lj_info, VECTOR* md_frc,
    const LTMatrix3 cell, const LTMatrix3 rcell, const float cutoff,
    const float pme_beta, const int need_potential, float* atom_energy,
    const int need_pressure, LTMatrix3* atom_virial, float* coulomb_atom_ene,
    const char** failure_reason)
{
    if (failure_reason != NULL)
    {
        *failure_reason = NULL;
    }
    if (!is_initialized || !selectively_applied || lj_info == NULL ||
        !lj_info->is_initialized)
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                "clustered SITS requires active selective regular LJ";
        }
        return false;
    }
    LJ_CLUSTERED_DIRECT_CACHE* cache = lj_info->clustered_direct_cache;
    if (cache == NULL || !cache->Use_Clustered_Direct())
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                "clustered SITS requires the regular-LJ clustered cache";
        }
        return false;
    }
    if (lj_info->d_LJ_AB_packed == NULL ||
        cache->layout.d_atom_local == NULL || atom_sys_mark_local == NULL)
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                "clustered SITS operator fields are unavailable";
        }
        return false;
    }

    if (!cache->Coordinate_Gather_Ready_For_Current_Step())
    {
#ifdef USE_CPU
        cache->Build(crd, cell, rcell, cutoff, need_pressure != 0, false,
                     false, false, false);
#else
        cache->Build(crd, cell, rcell, cutoff, need_pressure != 0, false,
                     true, false, true);
#endif
        cache->Gather_Plain(
            crd, charge, lj_info->crd_with_LJ_parameters_local,
            cell, rcell, lj_info->d_LJ_AB_packed);
    }

    CLUSTERED_SPATIAL_VIEW view = {};
    const char* view_failure_reason = NULL;
    if (!Make_Clustered_Spatial_View_From_LJ_Cache(
            cache, &view, &view_failure_reason))
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                view_failure_reason != NULL
                    ? view_failure_reason
                    : "clustered SITS spatial view is unavailable";
        }
        return false;
    }
    CLUSTERED_SPATIAL_VIEW_REQUIREMENTS requirements;
    requirements.local_atom_numbers = local_atom_numbers;
    requirements.ghost_numbers = ghost_numbers;
    requirements.cutoff = cutoff;
    requirements.provider_incarnation =
        cache->layout.provider_incarnation;
    requirements.lease_epoch =
        cache->layout.spatial_view_lease_epoch;
    requirements.native_payload_generation =
        cache->layout.native_payload_generation;
    requirements.gmxpacked_payload_generation =
        cache->layout.gmxpacked_compact_payload_generation;
    requirements.geometry_generation =
        cache->layout.geometry_generation;
    requirements.require_all_local_atoms = true;
#ifdef USE_CPU
    requirements.require_backend = true;
    requirements.backend = CLUSTERED_SPATIAL_BACKEND::CPU;
    requirements.require_native_payload = true;
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
    if (!Clustered_Validate_Spatial_View(
            view, requirements, &view_failure_reason))
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                view_failure_reason != NULL
                    ? view_failure_reason
                    : "clustered SITS spatial view is invalid";
        }
        return false;
    }
    if (!cache->Coordinate_Gather_Ready_For_Current_Step())
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                "clustered SITS coordinate gather is stale";
        }
        return false;
    }

    if (need_potential)
    {
        deviceMemset(classic_sits.d_bias, 0, sizeof(float));
    }
    if (atom_numbers == 0 || local_atom_numbers == 0)
    {
        return true;
    }
#ifdef USE_CPU
    if (view.sci_numbers <= 0 || view.sci == NULL ||
        view.cjpacked == NULL)
    {
        return true;
    }
    auto cpu_f = SITS_Clustered_Native_Direct<false, true>;
    if (need_potential || need_pressure)
    {
        cpu_f = SITS_Clustered_Native_Direct<true, true>;
    }
    cpu_f(
        view, view.sort_permutation,
        cache->d_sorted_xq, cache->d_sorted_lj_type,
        NULL,
        cache->layout.d_atom_local, atom_sys_mark_local, cell, rcell,
        lj_info->d_LJ_A, lj_info->d_LJ_B, NULL, NULL, NULL, NULL,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, cutoff, md_frc,
        pw_select.select_force[0], pme_beta, atom_energy,
        pw_select.select_atom_energy[0], atom_virial,
        pw_select.select_atom_virial_tensor[0], coulomb_atom_ene,
        lj_info->d_LJ_energy_atom, pwwp_enhance_factor,
        need_potential != 0, need_pressure != 0, NULL, NULL, 1.0f, 1.0f);
#else
    if (view.gmxpacked_sci_numbers <= 0 || view.gmxpacked_sci == NULL ||
        view.gmxpacked_cjpacked == NULL ||
        view.gmxpacked_exclusions == NULL || view.pair_shift_bits == NULL)
    {
        return true;
    }
    if (!SITS_Ensure_Sparse_Gmxpacked_View(
            this, cache, view, atom_sys_mark_local))
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                "clustered SITS sparse gmxpacked view build failed";
        }
        return false;
    }
    if (clustered_sparse_sci_numbers <= 0 ||
        clustered_sparse_cjpacked_numbers <= 0)
    {
        return true;
    }
    auto device_f =
        SITS_Clustered_Gmxpacked_Direct_Device<false, true>;
    if (need_potential || need_pressure)
    {
        device_f =
            SITS_Clustered_Gmxpacked_Direct_Device<true, true>;
    }
    const dim3 block_size(
        static_cast<unsigned int>(kClusteredClusterSize),
        static_cast<unsigned int>(kClusteredClusterSize), 1u);
    constexpr int packed_partitions = 8;
    const dim3 grid_size =
        (need_potential || need_pressure)
            ? dim3(static_cast<unsigned int>(
                       clustered_sparse_sci_numbers),
                   static_cast<unsigned int>(packed_partitions),
                   static_cast<unsigned int>(kClusteredJGroupSize))
            : dim3(static_cast<unsigned int>(
                       clustered_sparse_sci_numbers * packed_partitions),
                   1u, 1u);
    Launch_Device_Kernel(
        device_f, grid_size, block_size, 0, NULL,
        clustered_sparse_sci_numbers, packed_partitions,
        view.cluster_numbers, local_atom_numbers,
        view.cluster_offsets, view.cluster_valid_masks,
        view.cluster_local_masks,
        view.super_cluster_offsets, d_clustered_sparse_sci,
        d_clustered_sparse_cjpacked, view.gmxpacked_exclusions,
        d_clustered_sparse_pair_shift_bits, view.sort_permutation,
        cache->d_sorted_xq, cache->d_sorted_lj_type,
        NULL,
        cache->layout.d_atom_local, atom_sys_mark_local, cell,
        lj_info->d_LJ_AB_packed, NULL, NULL, NULL, NULL,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, cutoff, md_frc,
        pw_select.select_force[0], pme_beta, atom_energy,
        pw_select.select_atom_energy[0], atom_virial,
        pw_select.select_atom_virial_tensor[0], coulomb_atom_ene,
        lj_info->d_LJ_energy_atom, pwwp_enhance_factor,
        need_potential != 0, need_pressure != 0, NULL, NULL, 1.0f, 1.0f);
#endif
    return true;
}

bool SITS_INFORMATION::REST2_LJ_Direct_CF_Correction_Clustered(
    const int atom_numbers, const int local_atom_numbers,
    const int ghost_numbers, LENNARD_JONES_INFORMATION* lj_info,
    VECTOR* md_frc, const LTMatrix3 cell, const LTMatrix3 rcell,
    const float cutoff, const float pme_beta, const int need_energy,
    float* atom_energy, const int need_pressure, LTMatrix3* atom_virial,
    float* atom_direct_cf_energy, const int* rest2_atom_sys_mark,
    const float rest2_lambda_m, const float rest2_sqrt_lambda_m,
    float* rest2_unscaled_atom_energy,
    float* rest2_effective_atom_energy, const char** failure_reason)
{
    if (failure_reason != NULL)
    {
        *failure_reason = NULL;
    }
    if (lj_info == NULL || !lj_info->is_initialized ||
        rest2_atom_sys_mark == NULL)
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                "clustered REST2 requires regular LJ and local REST2 marks";
        }
        return false;
    }
    LJ_CLUSTERED_DIRECT_CACHE* cache = lj_info->clustered_direct_cache;
    if (cache == NULL || !cache->Use_Clustered_Direct())
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                "clustered REST2 requires the regular-LJ clustered cache";
        }
        return false;
    }
    if (lj_info->d_LJ_AB_packed == NULL ||
        cache->layout.d_atom_local == NULL)
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                "clustered REST2 operator fields are unavailable";
        }
        return false;
    }
    if (!cache->Coordinate_Gather_Ready_For_Current_Step())
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                "clustered REST2 coordinate gather is stale";
        }
        return false;
    }

    CLUSTERED_SPATIAL_VIEW view = {};
    const char* view_failure_reason = NULL;
    if (!Make_Clustered_Spatial_View_From_LJ_Cache(
            cache, &view, &view_failure_reason))
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                view_failure_reason != NULL
                    ? view_failure_reason
                    : "clustered REST2 spatial view is unavailable";
        }
        return false;
    }
    CLUSTERED_SPATIAL_VIEW_REQUIREMENTS requirements;
    requirements.local_atom_numbers = local_atom_numbers;
    requirements.ghost_numbers = ghost_numbers;
    requirements.cutoff = cutoff;
    requirements.provider_incarnation =
        cache->layout.provider_incarnation;
    requirements.lease_epoch =
        cache->layout.spatial_view_lease_epoch;
    requirements.native_payload_generation =
        cache->layout.native_payload_generation;
    requirements.gmxpacked_payload_generation =
        cache->layout.gmxpacked_compact_payload_generation;
    requirements.geometry_generation =
        cache->layout.geometry_generation;
    requirements.require_all_local_atoms = true;
#ifdef USE_CPU
    requirements.require_backend = true;
    requirements.backend = CLUSTERED_SPATIAL_BACKEND::CPU;
    requirements.require_native_payload = true;
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
    if (!Clustered_Validate_Spatial_View(
            view, requirements, &view_failure_reason))
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                view_failure_reason != NULL
                    ? view_failure_reason
                    : "clustered REST2 spatial view is invalid";
        }
        return false;
    }
    if (atom_numbers == 0 || local_atom_numbers == 0)
    {
        return true;
    }
#ifdef USE_CPU
    if (view.sci_numbers <= 0 || view.sci == NULL ||
        view.cjpacked == NULL)
    {
        return true;
    }
    auto cpu_f =
        SITS_Clustered_Native_Direct<false, true, false, true>;
    if (need_energy || need_pressure)
    {
        cpu_f =
            SITS_Clustered_Native_Direct<true, true, false, true>;
    }
    cpu_f(
        view, view.sort_permutation, cache->d_sorted_xq,
        cache->d_sorted_lj_type, NULL, cache->layout.d_atom_local,
        rest2_atom_sys_mark, cell, rcell, lj_info->d_LJ_A,
        lj_info->d_LJ_B, NULL, NULL, NULL, NULL, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, cutoff, md_frc, NULL, pme_beta, atom_energy, NULL,
        atom_virial, NULL, atom_direct_cf_energy,
        lj_info->d_LJ_energy_atom, 0.0f, need_energy != 0,
        need_pressure != 0, rest2_unscaled_atom_energy,
        rest2_effective_atom_energy, rest2_lambda_m,
        rest2_sqrt_lambda_m);
#else
    if (view.gmxpacked_sci_numbers <= 0 || view.gmxpacked_sci == NULL ||
        view.gmxpacked_cjpacked == NULL ||
        view.gmxpacked_exclusions == NULL || view.pair_shift_bits == NULL)
    {
        return true;
    }
    if (!SITS_Ensure_Sparse_Gmxpacked_View(
            this, cache, view, rest2_atom_sys_mark))
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                "clustered REST2 sparse gmxpacked view build failed";
        }
        return false;
    }
    if (clustered_sparse_sci_numbers <= 0 ||
        clustered_sparse_cjpacked_numbers <= 0)
    {
        return true;
    }
    auto device_f =
        SITS_Clustered_Gmxpacked_Direct_Device<false, true, false, true>;
    if (need_energy || need_pressure)
    {
        device_f =
            SITS_Clustered_Gmxpacked_Direct_Device<true, true, false, true>;
    }
    const dim3 block_size(
        static_cast<unsigned int>(kClusteredClusterSize),
        static_cast<unsigned int>(kClusteredClusterSize), 1u);
    constexpr int packed_partitions = 8;
    const dim3 grid_size =
        (need_energy || need_pressure)
            ? dim3(static_cast<unsigned int>(
                       clustered_sparse_sci_numbers),
                   static_cast<unsigned int>(packed_partitions),
                   static_cast<unsigned int>(kClusteredJGroupSize))
            : dim3(static_cast<unsigned int>(
                       clustered_sparse_sci_numbers * packed_partitions),
                   1u, 1u);
    Launch_Device_Kernel(
        device_f, grid_size, block_size, 0, NULL,
        clustered_sparse_sci_numbers, packed_partitions,
        view.cluster_numbers, local_atom_numbers, view.cluster_offsets,
        view.cluster_valid_masks, view.cluster_local_masks,
        view.super_cluster_offsets, d_clustered_sparse_sci,
        d_clustered_sparse_cjpacked, view.gmxpacked_exclusions,
        d_clustered_sparse_pair_shift_bits, view.sort_permutation,
        cache->d_sorted_xq, cache->d_sorted_lj_type, NULL,
        cache->layout.d_atom_local, rest2_atom_sys_mark, cell,
        lj_info->d_LJ_AB_packed, NULL, NULL, NULL, NULL, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, cutoff, md_frc, NULL, pme_beta, atom_energy,
        NULL, atom_virial, NULL, atom_direct_cf_energy,
        lj_info->d_LJ_energy_atom, 0.0f, need_energy != 0,
        need_pressure != 0, rest2_unscaled_atom_energy,
        rest2_effective_atom_energy, rest2_lambda_m,
        rest2_sqrt_lambda_m);
#endif
    return true;
}

bool SITS_INFORMATION::SITS_LJ_Soft_Core_Direct_CF_Force_Clustered(
    const int atom_numbers, const int local_atom_numbers,
    const int ghost_numbers, LJ_SOFT_CORE* lj_info, VECTOR* md_frc,
    const LTMatrix3 cell, const LTMatrix3 rcell, const float cutoff,
    const float pme_beta, const int need_potential, float* atom_energy,
    const int need_pressure, LTMatrix3* atom_virial,
    float* coulomb_atom_ene, const char** failure_reason)
{
    if (failure_reason != NULL)
    {
        *failure_reason = NULL;
    }
    if (!is_initialized || !selectively_applied || lj_info == NULL ||
        !lj_info->is_initialized)
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                "clustered SITS requires active selective soft-LJ";
        }
        return false;
    }
    LJ_CLUSTERED_DIRECT_CACHE* cache = lj_info->clustered_direct_cache;
    if (cache == NULL || !cache->Use_Clustered_Direct())
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                "clustered SITS requires the soft-LJ clustered cache";
        }
        return false;
    }
    if (cache->d_sorted_soft_crd == NULL ||
        cache->layout.d_atom_local == NULL || atom_sys_mark_local == NULL ||
        lj_info->d_LJ_AA == NULL || lj_info->d_LJ_AB == NULL ||
        lj_info->d_LJ_BA == NULL || lj_info->d_LJ_BB == NULL)
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                "clustered SITS soft-LJ operator fields are unavailable";
        }
        return false;
    }
    if (!cache->Coordinate_Gather_Ready_For_Current_Step())
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                "clustered SITS soft-LJ coordinate gather is stale";
        }
        return false;
    }

    CLUSTERED_SPATIAL_VIEW view = {};
    const char* view_failure_reason = NULL;
    if (!Make_Clustered_Spatial_View_From_LJ_Cache(
            cache, &view, &view_failure_reason))
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                view_failure_reason != NULL
                    ? view_failure_reason
                    : "clustered SITS soft-LJ spatial view is unavailable";
        }
        return false;
    }
    CLUSTERED_SPATIAL_VIEW_REQUIREMENTS requirements;
    requirements.local_atom_numbers = local_atom_numbers;
    requirements.ghost_numbers = ghost_numbers;
    requirements.cutoff = cutoff;
    requirements.provider_incarnation =
        cache->layout.provider_incarnation;
    requirements.lease_epoch =
        cache->layout.spatial_view_lease_epoch;
    requirements.native_payload_generation =
        cache->layout.native_payload_generation;
    requirements.gmxpacked_payload_generation =
        cache->layout.gmxpacked_compact_payload_generation;
    requirements.geometry_generation =
        cache->layout.geometry_generation;
    requirements.require_all_local_atoms = true;
#ifdef USE_CPU
    requirements.require_backend = true;
    requirements.backend = CLUSTERED_SPATIAL_BACKEND::CPU;
    requirements.require_native_payload = true;
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
    if (!Clustered_Validate_Spatial_View(
            view, requirements, &view_failure_reason))
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                view_failure_reason != NULL
                    ? view_failure_reason
                    : "clustered SITS soft-LJ spatial view is invalid";
        }
        return false;
    }

    if (need_potential)
    {
        deviceMemset(classic_sits.d_bias, 0, sizeof(float));
    }
    if (atom_numbers == 0 || local_atom_numbers == 0)
    {
        return true;
    }
#ifdef USE_CPU
    if (view.sci_numbers <= 0 || view.sci == NULL ||
        view.cjpacked == NULL)
    {
        return true;
    }
    auto cpu_f = SITS_Clustered_Native_Direct<false, true, true>;
    if (need_potential || need_pressure)
    {
        cpu_f = SITS_Clustered_Native_Direct<true, true, true>;
    }
    cpu_f(
        view, view.sort_permutation, NULL, NULL,
        cache->d_sorted_soft_crd, cache->layout.d_atom_local,
        atom_sys_mark_local, cell, rcell, NULL, NULL,
        lj_info->d_LJ_AA, lj_info->d_LJ_AB, lj_info->d_LJ_BA,
        lj_info->d_LJ_BB, lj_info->lambda, lj_info->alpha, lj_info->p,
        lj_info->sigma_6, lj_info->sigma_6_min, cutoff, md_frc,
        pw_select.select_force[0], pme_beta, atom_energy,
        pw_select.select_atom_energy[0], atom_virial,
        pw_select.select_atom_virial_tensor[0], coulomb_atom_ene,
        lj_info->d_LJ_energy_atom, pwwp_enhance_factor,
        need_potential != 0, need_pressure != 0, NULL, NULL, 1.0f, 1.0f);
#else
    if (view.gmxpacked_sci_numbers <= 0 ||
        view.gmxpacked_sci == NULL ||
        view.gmxpacked_cjpacked == NULL ||
        view.gmxpacked_exclusions == NULL || view.pair_shift_bits == NULL)
    {
        return true;
    }
    if (!SITS_Ensure_Sparse_Gmxpacked_View(
            this, cache, view, atom_sys_mark_local))
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                "clustered SITS soft-LJ sparse gmxpacked view build failed";
        }
        return false;
    }
    if (clustered_sparse_sci_numbers <= 0 ||
        clustered_sparse_cjpacked_numbers <= 0)
    {
        return true;
    }
    auto device_f =
        SITS_Clustered_Gmxpacked_Direct_Device<false, true, true>;
    if (need_potential || need_pressure)
    {
        device_f =
            SITS_Clustered_Gmxpacked_Direct_Device<true, true, true>;
    }
    const dim3 block_size(
        static_cast<unsigned int>(kClusteredClusterSize),
        static_cast<unsigned int>(kClusteredClusterSize), 1u);
    constexpr int packed_partitions = 8;
    const dim3 grid_size =
        (need_potential || need_pressure)
            ? dim3(static_cast<unsigned int>(
                       clustered_sparse_sci_numbers),
                   static_cast<unsigned int>(packed_partitions),
                   static_cast<unsigned int>(kClusteredJGroupSize))
            : dim3(static_cast<unsigned int>(
                       clustered_sparse_sci_numbers * packed_partitions),
                   1u, 1u);
    Launch_Device_Kernel(
        device_f, grid_size, block_size, 0, NULL,
        clustered_sparse_sci_numbers, packed_partitions,
        view.cluster_numbers, local_atom_numbers,
        view.cluster_offsets, view.cluster_valid_masks,
        view.cluster_local_masks, view.super_cluster_offsets,
        d_clustered_sparse_sci, d_clustered_sparse_cjpacked,
        view.gmxpacked_exclusions, d_clustered_sparse_pair_shift_bits,
        view.sort_permutation, NULL, NULL, cache->d_sorted_soft_crd,
        cache->layout.d_atom_local, atom_sys_mark_local, cell, NULL,
        lj_info->d_LJ_AA, lj_info->d_LJ_AB, lj_info->d_LJ_BA,
        lj_info->d_LJ_BB, lj_info->lambda, lj_info->alpha, lj_info->p,
        lj_info->sigma_6, lj_info->sigma_6_min, cutoff, md_frc,
        pw_select.select_force[0], pme_beta, atom_energy,
        pw_select.select_atom_energy[0], atom_virial,
        pw_select.select_atom_virial_tensor[0], coulomb_atom_ene,
        lj_info->d_LJ_energy_atom, pwwp_enhance_factor,
        need_potential != 0, need_pressure != 0, NULL, NULL, 1.0f, 1.0f);
#endif
    return true;
}

void SITS_INFORMATION::Step_Print(CONTROLLER* controller, const float beta0)
{
    if (!is_initialized) return;
    float bias;
    deviceMemcpy(&bias, classic_sits.d_bias, sizeof(float),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(&h_enhancing_energy, pw_select.select_energy[0], sizeof(float),
                 deviceMemcpyDeviceToHost);
    controller->Step_Print(print_aa_kab_name, h_enhancing_energy);
    controller->Step_Print(print_bias_name, bias);
    controller->Step_Print(print_fb_name, h_factor);
}

void SELECT::Initial()
{
    select_atom_energy.clear();
    select_energy.clear();
    select_force.clear();
    select_atom_virial.clear();
    select_virial.clear();
}

int SELECT::Add_One_Energy(int atom_numbers)
{
    float* tmp_atom_energy;
    float* tmp_energy;
    Device_Malloc_Safely((void**)&tmp_atom_energy,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&tmp_energy, sizeof(float));
    select_atom_energy.push_back(tmp_atom_energy);
    select_energy.push_back(tmp_energy);
    return select_energy.size() - 1;
}

int SELECT::Add_One_Force(int atom_numbers)
{
    VECTOR* tmp_force;
    Device_Malloc_Safely((void**)&tmp_force, sizeof(VECTOR) * atom_numbers);
    select_force.push_back(tmp_force);
    return (select_force.size() - 1);
}

int SELECT::Add_One_Virial(int atom_numbers)
{
    float* tmp_atom_virial;
    float* tmp_virial;
    LTMatrix3* tmp_atom_virial_tensor;
    LTMatrix3* tmp_virial_tensor;
    Device_Malloc_Safely((void**)&tmp_atom_virial,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&tmp_virial, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&tmp_atom_virial_tensor,
                         sizeof(LTMatrix3) * atom_numbers);
    Device_Malloc_Safely((void**)&tmp_virial_tensor,
                         sizeof(LTMatrix3) * atom_numbers);
    select_atom_virial.push_back(tmp_atom_virial);
    select_virial.push_back(tmp_virial);
    select_atom_virial_tensor.push_back(tmp_atom_virial_tensor);
    select_virial_tensor.push_back(tmp_virial_tensor);
    return select_virial.size() - 1;
}

static __global__ void get_local_device(int* atom_local, int local_atom_numbers,
                                        int ghost_numbers, int* atom_sys_mark,
                                        int* atom_sys_mark_local)
{
    int total = local_atom_numbers + ghost_numbers;
    SIMPLE_DEVICE_FOR(i, total)
    {
        atom_sys_mark_local[i] = atom_sys_mark[atom_local[i]];
    }
}

void SITS_INFORMATION::Get_Local(int* atom_local, int local_atom_numbers_,
                                 int ghost_numbers_)
{
    if (is_initialized)
    {
        local_atom_numbers = local_atom_numbers_;
        ghost_numbers = ghost_numbers_;
        Launch_Device_Kernel(get_local_device,
                             (local_atom_numbers + ghost_numbers +
                              CONTROLLER::device_max_thread - 1) /
                                 CONTROLLER::device_max_thread,
                             CONTROLLER::device_max_thread, 0, NULL, atom_local,
                             local_atom_numbers, ghost_numbers, atom_sys_mark,
                             atom_sys_mark_local);
    }
}
