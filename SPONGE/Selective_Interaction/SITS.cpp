#include "SITS.h"

template <bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb>
static __global__ void Selective_Lennard_Jones_And_Direct_Coulomb_Device(
    const int local_atom_numbers, const int solvent_numbers,
    const ATOM_GROUP* nl, float* atom_ene_LJ, const VECTOR_LJ* crd,
    const LTMatrix3 cell, const LTMatrix3 rcell, const float* LJ_type_A,
    const float* LJ_type_B, const int* atom_sys_mark, const float cutoff,
    VECTOR* frc, VECTOR* frc_enhancing, const float pme_beta,
    float* atom_energy, float* atom_energy_enhancing, LTMatrix3* atom_virial,
    LTMatrix3* atom_virial_enhancing, float* atom_direct_cf_energy,
    const float pwwp_factor)
{
#ifdef USE_GPU
    int atom_i = blockDim.y * blockIdx.x + threadIdx.y;
    if (atom_i < local_atom_numbers - solvent_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < local_atom_numbers - solvent_numbers;
         atom_i++)
#endif
    {
        ATOM_GROUP nl_i = nl[atom_i];
        VECTOR_LJ r1 = crd[atom_i];
        int atom_mark_i = atom_sys_mark[atom_i];
        VECTOR frc_record = {0.0f, 0.0f, 0.0f},
               frc_enhancing_record = {0.0f, 0.0f, 0.0f};
        LTMatrix3 virial_record = {0, 0, 0, 0, 0, 0},
                  virial_enhancing = {0, 0, 0, 0, 0, 0};
        float energy_lj = 0.0f, energy_enhancing = 0.0f, energy_coulomb = 0.0f;
#ifdef USE_GPU
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
#else
        for (int j = 0; j < nl_i.atom_numbers; j++)
#endif
        {
            int atom_j = nl_i.atom_serial[j];
            float ij_factor = atom_j < local_atom_numbers ? 1.0f : 0.5f;
            VECTOR_LJ r2 = crd[atom_j];
            VECTOR dr = Get_Periodic_Displacement(r2, r1, cell, rcell);
            float dr_abs = norm3df(dr.x, dr.y, dr.z);
            if (dr_abs < cutoff)
            {
                int atom_mark_j = atom_sys_mark[atom_j] + atom_mark_i;
                int atom_pair_LJ_type = Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                float A = LJ_type_A[atom_pair_LJ_type];
                float B = LJ_type_B[atom_pair_LJ_type];
                float factor = 0;
                if (atom_mark_j == 0)
                {
                    factor = 1;
                }
                else if (atom_mark_j == 1)
                {
                    factor = pwwp_factor;
                }
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
                        atomicAdd(frc + atom_j, -frc_lin);
                    frc_lin = factor * frc_lin;
                    frc_enhancing_record = frc_enhancing_record + frc_lin;
                    if (need_virial)
                    {
                        LTMatrix3 virial0 =
                            Get_Virial_From_Force_Dis(frc_lin, dr);
                        virial_record = virial_record + ij_factor * virial0;
                        virial_enhancing =
                            virial_enhancing + ij_factor * factor * virial0;
                    }
                }
                if (need_coulomb && need_energy)
                {
                    float energy_lin =
                        Get_Direct_Coulomb_Energy(r1, r2, dr_abs, pme_beta);
                    energy_coulomb += ij_factor * energy_lin;
                    energy_enhancing += ij_factor * factor * energy_lin;
                }
                if (need_energy)
                {
                    float energy_lin = Get_LJ_Energy(r1, r2, dr_abs, A, B);
                    energy_lj += ij_factor * energy_lin;
                    energy_enhancing += ij_factor * factor * energy_lin;
                }
            }
        }
        if (need_force)
        {
            Warp_Sum_To(frc + atom_i, frc_record, warpSize);
            Warp_Sum_To(frc_enhancing + atom_i, frc_enhancing_record, warpSize);
        }
        if (need_coulomb && need_energy)
        {
            Warp_Sum_To(atom_direct_cf_energy + atom_i, energy_coulomb,
                        warpSize);
        }
        if (need_energy)
        {
            Warp_Sum_To(atom_energy + atom_i, energy_lj, warpSize);
#ifdef USE_GPU
            if (threadIdx.x == 0)
#endif
                atomicAdd(atom_ene_LJ + atom_i, energy_lj);
            Warp_Sum_To(atom_energy_enhancing + atom_i, energy_enhancing,
                        warpSize);
        }
        if (need_virial)
        {
            Warp_Sum_To(atom_virial + atom_i, virial_record, warpSize);
            Warp_Sum_To(atom_virial_enhancing + atom_i, virial_enhancing,
                        warpSize);
        }
    }
}

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
SITS_Store_Clustered_Pair(
    const int atom_i, const int atom_j, const int global_i,
    const int global_j, const bool j_is_local, const int selective_atom_end,
    const VECTOR_LJ r1, const VECTOR_LJ r2, const VECTOR dr,
    const float2 lj_ab, const float dr_abs, const float pme_beta,
    const int* atom_sys_mark, const float pwwp_factor, VECTOR* frc,
    VECTOR* frc_enhancing, float* atom_energy,
    float* atom_energy_enhancing, LTMatrix3* atom_virial,
    LTMatrix3* atom_virial_enhancing, float* atom_direct_cf_energy,
    float* atom_ene_lj, const bool store_energy, const bool store_virial)
{
    const bool owner_is_i = !j_is_local || global_i < global_j;
    const int owner = owner_is_i ? atom_i : atom_j;
    const float owner_force_sign = owner_is_i ? 1.0f : -1.0f;
    const float ij_factor = j_is_local ? 1.0f : 0.5f;
    const bool selective_owner = owner < selective_atom_end;
    float selective_factor = 0.0f;
    if (selective_owner)
    {
        const int mark_sum =
            atom_sys_mark[atom_i] + atom_sys_mark[atom_j];
        selective_factor =
            mark_sum == 0 ? 1.0f : (mark_sum == 1 ? pwwp_factor : 0.0f);
    }

    float force_abs = Get_LJ_Force(r1, r2, dr_abs, lj_ab.x, lj_ab.y);
    force_abs -= Get_Direct_Coulomb_Force(r1, r2, dr_abs, pme_beta);
    const VECTOR force_i = force_abs * dr;

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
        const float energy_lj =
            ij_factor * Get_LJ_Energy(r1, r2, dr_abs, lj_ab.x, lj_ab.y);
        const float energy_coulomb =
            ij_factor *
            Get_Direct_Coulomb_Energy(r1, r2, dr_abs, pme_beta);
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
    const int selective_atom_end, const int* cluster_offsets,
    const int* super_cluster_offsets, const int* sorted_atom_ids,
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
                    const int atom_j = sorted_atom_ids[
                        cluster_offsets[cluster_j] + j_lane];
                    selected_j = selected_j ||
                                 (atom_j >= 0 &&
                                  atom_j < selective_atom_end);
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
                        const int atom_i = sorted_atom_ids[
                            cluster_offsets[cluster_i] + i_lane];
                        selected_i = selected_i ||
                                     (atom_i >= 0 &&
                                      atom_i < selective_atom_end);
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
    const int selective_atom_end, const int* cluster_offsets,
    const int* super_cluster_offsets, const int* sorted_atom_ids,
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
                cluster_numbers, selective_atom_end, cluster_offsets,
                super_cluster_offsets, sorted_atom_ids, &pruned))
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
                cluster_numbers, selective_atom_end, cluster_offsets,
                super_cluster_offsets, sorted_atom_ids, &pruned))
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
    const CLUSTERED_SPATIAL_VIEW& view, const int selective_atom_end)
{
    const LJ_CLUSTER_LAYOUT& layout = cache->layout;
    if (sits->clustered_sparse_provider_incarnation ==
            layout.provider_incarnation &&
        sits->clustered_sparse_payload_generation ==
            layout.gmxpacked_compact_payload_generation &&
        sits->clustered_sparse_selective_atom_end == selective_atom_end)
    {
        return true;
    }
    sits->clustered_sparse_sci_numbers = 0;
    sits->clustered_sparse_cjpacked_numbers = 0;
    if (selective_atom_end <= 0 || view.gmxpacked_sci_numbers <= 0 ||
        view.gmxpacked_cjpacked_numbers <= 0)
    {
        sits->clustered_sparse_provider_incarnation =
            layout.provider_incarnation;
        sits->clustered_sparse_payload_generation =
            layout.gmxpacked_compact_payload_generation;
        sits->clustered_sparse_selective_atom_end = selective_atom_end;
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
        selective_atom_end, view.cluster_offsets,
        view.super_cluster_offsets, cache->d_sorted_atom_ids,
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
    sits->clustered_sparse_selective_atom_end = selective_atom_end;
    return true;
}
#endif

template <bool full_output, bool correction_only = false>
static __global__ void SITS_Clustered_Gmxpacked_Direct_Device(
    const int sci_numbers, const int packed_partitions,
    const int cluster_numbers,
    const int local_atom_numbers, const int selective_atom_end,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const int* super_cluster_offsets,
    const CLUSTERED_GMXPACKED_SCI* sci_entries,
    const CLUSTERED_GMXPACKED_CJ* cjpacked_entries,
    const CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    const uint64_t* pair_shift_bits, const int* sorted_atom_ids,
    const float4* sorted_xq, const int* sorted_lj_type,
    const int* atom_local, const int* atom_sys_mark, const LTMatrix3 cell,
    const float2* lj_ab_packed, const float cutoff, VECTOR* frc,
    VECTOR* frc_enhancing, const float pme_beta, float* atom_energy,
    float* atom_energy_enhancing, LTMatrix3* atom_virial,
    LTMatrix3* atom_virial_enhancing, float* atom_direct_cf_energy,
    float* atom_ene_lj, const float pwwp_factor, const bool store_energy,
    const bool store_virial)
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
            const VECTOR_LJ r2 = SITS_Make_Packed_LJ_Atom(
                sorted_xq[sorted_j], sorted_lj_type[sorted_j]);
            const uint64_t shift_bits =
                pair_shift_bits[packed_idx * kClusteredJGroupSize + jm];
            VECTOR force_j = {0.0f, 0.0f, 0.0f};
            VECTOR force_j_enhancing = {0.0f, 0.0f, 0.0f};
            for (int i_local = 0;
                 i_local < cluster_i_end - cluster_i_begin; i_local += 1)
            {
                const unsigned int packed_bit =
                    1u << (jm * kClusteredSuperClusterClusters + i_local);
                if ((effective_mask & packed_bit) == 0u)
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
                const VECTOR_LJ r1 = SITS_Make_Packed_LJ_Atom(
                    sorted_xq[sorted_i], sorted_lj_type[sorted_i]);
                const int shift_id =
                    Clustered_Get_Pair_Shift_Id(shift_bits, i_local);
                const VECTOR shift =
                    Clustered_Shift_Vector_From_Id(shift_id, cell);
                const VECTOR dr = (r2.crd - r1.crd) - shift;
                const float dr2 = dr * dr;
                if (dr2 <= 0.0f || dr2 >= cutoff_sq)
                {
                    continue;
                }
                const int pair_type =
                    Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                const SITS_CLUSTERED_PAIR_FORCE stored =
                    SITS_Store_Clustered_Pair<full_output, false,
                                              correction_only>(
                    atom_i, atom_j, global_i, global_j, j_is_local,
                    selective_atom_end, r1, r2, dr,
                    lj_ab_packed[pair_type], sqrtf(dr2), pme_beta,
                    atom_sys_mark, pwwp_factor, frc, frc_enhancing,
                    atom_energy, atom_energy_enhancing, atom_virial,
                    atom_virial_enhancing, atom_direct_cf_energy,
                    atom_ene_lj, store_energy, store_virial);
                if (j_is_local)
                {
                    if constexpr (!correction_only)
                    {
                        force_j = force_j - stored.force_i;
                    }
                    if (stored.selective_owner && !stored.owner_is_i)
                    {
                        force_j_enhancing =
                            force_j_enhancing -
                            stored.selective_factor * stored.force_i;
                    }
                }
            }
            if (j_is_local)
            {
                if constexpr (!correction_only)
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
                if (atom_j < selective_atom_end)
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
#else
    (void)sci_numbers;
    (void)packed_partitions;
    (void)cluster_numbers;
    (void)local_atom_numbers;
    (void)selective_atom_end;
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
    (void)atom_local;
    (void)atom_sys_mark;
    (void)cell;
    (void)lj_ab_packed;
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
template <bool full_output, bool correction_only = false>
static void SITS_Clustered_Native_Direct(
    const CLUSTERED_SPATIAL_VIEW& view, const int selective_atom_end,
    const int* sorted_atom_ids, const float4* sorted_xq,
    const int* sorted_lj_type, const int* atom_local,
    const int* atom_sys_mark, const LTMatrix3 cell, const LTMatrix3 rcell,
    const float* lj_a, const float* lj_b, const float cutoff, VECTOR* frc,
    VECTOR* frc_enhancing, const float pme_beta, float* atom_energy,
    float* atom_energy_enhancing, LTMatrix3* atom_virial,
    LTMatrix3* atom_virial_enhancing, float* atom_direct_cf_energy,
    float* atom_ene_lj, const float pwwp_factor, const bool store_energy,
    const bool store_virial)
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
                const VECTOR_LJ r1 = SITS_Make_Packed_LJ_Atom(
                    sorted_xq[sorted_i], sorted_lj_type[sorted_i]);
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
                            const int global_i = atom_local[atom_i];
                            const int global_j = atom_local[atom_j];
                            if (sci_entry.shift_id ==
                                    kClusteredCentralShiftId &&
                                cluster_i == cluster_j && j_is_local &&
                                j_lane <= i_lane)
                            {
                                continue;
                            }
                            const VECTOR_LJ r2 = SITS_Make_Packed_LJ_Atom(
                                sorted_xq[sorted_j],
                                sorted_lj_type[sorted_j]);
                            const VECTOR dr = (r2.crd - r1.crd) - shift;
                            const float dr2 = dr * dr;
                            if (dr2 <= 0.0f || dr2 >= cutoff_sq)
                            {
                                continue;
                            }
                            const int pair_type =
                                Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                            const float2 lj_ab = {
                                lj_a[pair_type], lj_b[pair_type]};
                            SITS_Store_Clustered_Pair<full_output, true,
                                                      correction_only>(
                                atom_i, atom_j, global_i, global_j, j_is_local,
                                selective_atom_end, r1, r2, dr,
                                lj_ab, sqrtf(dr2), pme_beta,
                                atom_sys_mark, pwwp_factor, frc,
                                frc_enhancing, atom_energy,
                                atom_energy_enhancing, atom_virial,
                                atom_virial_enhancing,
                                atom_direct_cf_energy, atom_ene_lj,
                                store_energy, store_virial);
                        }
                    }
                }
            }
        }
    }
}
#endif

template <bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb, bool need_du_dlambda>
static __global__ void
Selective_Lennard_Jones_And_Direct_Coulomb_Soft_Core_Device(
    const int local_atom_numbers, const int solvent_numbers,
    const ATOM_GROUP* nl, float* atom_ene_LJ, const VECTOR_LJ_SOFT_TYPE* crd,
    const LTMatrix3 cell, const LTMatrix3 rcell, const int* atom_sys_mark,
    const float* LJ_type_AA, const float* LJ_type_AB, const float* LJ_type_BA,
    const float* LJ_type_BB, const float cutoff, VECTOR* frc,
    VECTOR* frc_enhancing, const float pme_beta, float* atom_energy,
    float* atom_energy_enhancing, LTMatrix3* atom_virial,
    LTMatrix3* atom_virial_enhancing, float* atom_direct_cf_energy,
    float* atom_du_dlambda_lj, float* atom_du_dlambda_direct,
    float* atom_du_dlambda_enhancing, const float lambda, const float alpha,
    const float p, const float input_sigma_6, const float input_sigma_6_min,
    const float pwwp_factor)
{
    float lambda_ = 1.0 - lambda;
    float alpha_lambda_p = alpha * powf(lambda, p);
    float alpha_lambda__p = alpha * powf(lambda_, p);
#ifdef USE_GPU
    int atom_i = blockDim.y * blockIdx.x + threadIdx.y;
    if (atom_i < local_atom_numbers - solvent_numbers)
#else
#pragma omp parallel for firstprivate(lambda, alpha_lambda_p, alpha_lambda__p)
    for (int atom_i = 0; atom_i < local_atom_numbers - solvent_numbers;
         atom_i++)
#endif
    {
        ATOM_GROUP nl_i = nl[atom_i];
        VECTOR_LJ_SOFT_TYPE r1 = crd[atom_i];
        VECTOR frc_record = {0., 0., 0.},
               frc_enhancing_record = {0.0f, 0.0f, 0.0f};
        LTMatrix3 virial_record = {0, 0, 0, 0, 0, 0},
                  virial_enhancing = {0, 0, 0, 0, 0, 0};
        float energy_total = 0., energy_enhancing = 0.0f;
        float energy_coulomb = 0.;
        float du_dlambda_lj = 0.;
        float du_dlambda_direct = 0.;
        // float du_dlambda_enhancing = 0.0f;
        int atom_mark_i = atom_sys_mark[atom_i];
#ifdef USE_GPU
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
#else
        for (int j = 0; j < nl_i.atom_numbers; j++)
#endif
        {
            int atom_j = nl_i.atom_serial[j];
            float ij_factor = atom_j < local_atom_numbers ? 1.0f : 0.5f;
            VECTOR_LJ_SOFT_TYPE r2 = crd[atom_j];
            VECTOR dr = Get_Periodic_Displacement(r2, r1, cell, rcell);
            float dr_abs = norm3df(dr.x, dr.y, dr.z);
            if (dr_abs < cutoff)
            {
                int atom_mark_j = atom_sys_mark[atom_j] + atom_mark_i;
                float factor = 0;
                if (atom_mark_j == 0)
                {
                    factor = 1;
                }
                else if (atom_mark_j == 1)
                {
                    factor = pwwp_factor;
                }
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
                        frc_enhancing_record =
                            frc_enhancing_record + factor * frc_lin;
                        if (atom_j < local_atom_numbers)
                        {
                            atomicAdd(frc + atom_j, -frc_lin);
                            atomicAdd(frc_enhancing + atom_j,
                                      -factor * frc_lin);
                        }
                        if (need_virial)
                        {
                            LTMatrix3 virial0 =
                                Get_Virial_From_Force_Dis(frc_lin, dr);
                            virial_record = virial_record + ij_factor * virial0;
                            virial_enhancing =
                                virial_enhancing + ij_factor * factor * virial0;
                        }
                    }
                    if (need_coulomb && need_energy)
                    {
                        float ene =
                            Get_Direct_Coulomb_Energy(r1, r2, dr_abs, pme_beta);
                        energy_coulomb += ij_factor * ene;
                        energy_enhancing += ij_factor * factor * ene;
                    }
                    if (need_energy)
                    {
                        float ene =
                            lambda_ * Get_LJ_Energy(r1, r2, dr_abs, AA, AB) +
                            lambda * Get_LJ_Energy(r1, r2, dr_abs, BA, BB);
                        energy_total += ij_factor * ene;
                        energy_enhancing += ij_factor * factor * ene;
                    }
                    if (need_du_dlambda)
                    {
                        du_dlambda_lj += Get_LJ_Energy(r1, r2, dr_abs, BA, BB) -
                                         Get_LJ_Energy(r1, r2, dr_abs, AA, AB);
                        if (need_coulomb)
                        {
                            du_dlambda_direct += Get_Direct_Coulomb_dU_dlambda(
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
                        frc_enhancing_record =
                            frc_enhancing_record + factor * frc_lin;
                        if (atom_j < local_atom_numbers)
                        {
                            atomicAdd(frc + atom_j, -frc_lin);
                            atomicAdd(frc_enhancing + atom_j,
                                      -factor * frc_lin);
                        }
                        if (need_virial)
                        {
                            LTMatrix3 virial0 =
                                Get_Virial_From_Force_Dis(frc_lin, dr);
                            virial_record = virial_record + ij_factor * virial0;
                            virial_enhancing =
                                virial_enhancing + ij_factor * factor * virial0;
                        }
                    }
                    if (need_coulomb && need_energy)
                    {
                        float ene =
                            lambda_ * Get_Direct_Coulomb_Energy(
                                          r1, r2, dr_softcore_A, pme_beta) +
                            lambda * Get_Direct_Coulomb_Energy(
                                         r1, r2, dr_softcore_B, pme_beta);
                        energy_coulomb += ij_factor * ene;
                        energy_enhancing += ij_factor * factor * ene;
                    }
                    if (need_energy)
                    {
                        float ene =
                            lambda_ *
                                Get_LJ_Energy(r1, r2, dr_softcore_A, AA, AB) +
                            lambda *
                                Get_LJ_Energy(r1, r2, dr_softcore_B, BA, BB);
                        energy_total += ij_factor * ene;
                        energy_enhancing += ij_factor * factor * ene;
                    }
                    if (need_du_dlambda)
                    {
                        du_dlambda_lj +=
                            Get_LJ_Energy(r1, r2, dr_softcore_B, BA, BB) -
                            Get_LJ_Energy(r1, r2, dr_softcore_A, AA, AB);
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
                                Get_Direct_Coulomb_Energy(r1, r2, dr_softcore_B,
                                                          pme_beta) -
                                Get_Direct_Coulomb_Energy(r1, r2, dr_softcore_A,
                                                          pme_beta);
                            du_dlambda_direct +=
                                Get_Soft_Core_dU_dlambda(
                                    Get_Direct_Coulomb_Force(
                                        r1, r2, dr_softcore_B, pme_beta),
                                    sigma_B, dr_softcore_B, alpha, p, lambda_) -
                                Get_Soft_Core_dU_dlambda(
                                    Get_Direct_Coulomb_Force(
                                        r1, r2, dr_softcore_A, pme_beta),
                                    sigma_A, dr_softcore_A, alpha, p, lambda);
                            du_dlambda_direct +=
                                lambda * Get_Direct_Coulomb_dU_dlambda(
                                             r1, r2, dr_softcore_B, pme_beta) +
                                lambda_ * Get_Direct_Coulomb_dU_dlambda(
                                              r1, r2, dr_softcore_A, pme_beta);
                        }
                    }
                }
            }
        }
        if (need_force)
        {
            Warp_Sum_To(frc + atom_i, frc_record, warpSize);
            Warp_Sum_To(frc_enhancing + atom_i, frc_enhancing_record, warpSize);
        }
        if (need_coulomb && need_energy)
        {
            Warp_Sum_To(atom_direct_cf_energy + atom_i, energy_coulomb,
                        warpSize);
        }
        if (need_energy)
        {
            Warp_Sum_To(atom_energy + atom_i, energy_total, warpSize);
#ifdef USE_GPU
            if (threadIdx.x == 0)
#endif
                atomicAdd(atom_ene_LJ + atom_i, energy_total);
            Warp_Sum_To(atom_energy_enhancing + atom_i, energy_enhancing,
                        warpSize);
        }
        if (need_virial)
        {
            Warp_Sum_To(atom_virial + atom_i, virial_record, warpSize);
            Warp_Sum_To(atom_virial_enhancing + atom_i, virial_enhancing,
                        warpSize);
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

void SITS_INFORMATION::SITS_LJ_Direct_CF_Force_With_Atom_Energy_And_Virial(
    const int atom_numbers, const int local_atom_numbers,
    const int solvent_numbers, const int ghost_numbers, const VECTOR* crd,
    const float* charge, LENNARD_JONES_INFORMATION* lj_info, VECTOR* md_frc,
    const LTMatrix3 cell, const LTMatrix3 rcell, const ATOM_GROUP* nl,
    const float cutoff, const float pme_beta, const int need_potential,
    float* atom_energy, const int need_pressure, LTMatrix3* atom_virial,
    float* coulomb_atom_ene)
{
    if (is_initialized && lj_info->is_initialized)
    {
        Launch_Device_Kernel(
            Copy_Crd_And_Charge_To_New_Crd,
            (this->atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL,
            local_atom_numbers + ghost_numbers, crd,
            lj_info->crd_with_LJ_parameters_local, charge);

        if (need_potential)
        {
            deviceMemset(coulomb_atom_ene, 0,
                         sizeof(float) * (local_atom_numbers + ghost_numbers));
            deviceMemset(lj_info->d_LJ_energy_atom, 0,
                         sizeof(float) * (local_atom_numbers + ghost_numbers));
            deviceMemset(classic_sits.d_bias, 0, sizeof(float));
        }
        if (!local_atom_numbers) return;

        auto f = Selective_Lennard_Jones_And_Direct_Coulomb_Device<true, false,
                                                                   false, true>;
        dim3 blockSize = {
            CONTROLLER::device_warp,
            CONTROLLER::device_max_thread / CONTROLLER::device_warp};
        dim3 gridSize = (local_atom_numbers + blockSize.y - 1) / blockSize.y;

        if (need_potential && !need_pressure)
        {
            f = Selective_Lennard_Jones_And_Direct_Coulomb_Device<true, true,
                                                                  false, true>;
        }
        else if (need_potential && need_pressure)
        {
            f = Selective_Lennard_Jones_And_Direct_Coulomb_Device<true, true,
                                                                  true, true>;
        }
        else if (!need_potential && need_pressure)
        {
            f = Selective_Lennard_Jones_And_Direct_Coulomb_Device<true, false,
                                                                  true, true>;
        }
        else
        {
            f = Selective_Lennard_Jones_And_Direct_Coulomb_Device<true, false,
                                                                  false, true>;
        }

        Launch_Device_Kernel(
            f, gridSize, blockSize, 0, NULL, local_atom_numbers,
            solvent_numbers, nl, lj_info->d_LJ_energy_atom,
            lj_info->crd_with_LJ_parameters_local, cell, rcell, lj_info->d_LJ_A,
            lj_info->d_LJ_B, atom_sys_mark_local, cutoff, md_frc,
            pw_select.select_force[0], pme_beta, atom_energy,
            pw_select.select_atom_energy[0], atom_virial,
            pw_select.select_atom_virial_tensor[0], coulomb_atom_ene,
            pwwp_enhance_factor);
    }
}

bool SITS_INFORMATION::SITS_LJ_Direct_CF_Force_Clustered(
    const int atom_numbers, const int local_atom_numbers,
    const int solvent_numbers, const int ghost_numbers, const VECTOR* crd,
    const float* charge, LENNARD_JONES_INFORMATION* lj_info, VECTOR* md_frc,
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
    const int selective_atom_end =
        local_atom_numbers > solvent_numbers
            ? local_atom_numbers - solvent_numbers
            : 0;

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
        view, selective_atom_end, cache->d_sorted_atom_ids,
        cache->d_sorted_xq, cache->d_sorted_lj_type,
        cache->layout.d_atom_local, atom_sys_mark_local, cell, rcell,
        lj_info->d_LJ_A, lj_info->d_LJ_B, cutoff, md_frc,
        pw_select.select_force[0], pme_beta, atom_energy,
        pw_select.select_atom_energy[0], atom_virial,
        pw_select.select_atom_virial_tensor[0], coulomb_atom_ene,
        lj_info->d_LJ_energy_atom, pwwp_enhance_factor,
        need_potential != 0, need_pressure != 0);
#else
    if (view.gmxpacked_sci_numbers <= 0 || view.gmxpacked_sci == NULL ||
        view.gmxpacked_cjpacked == NULL ||
        view.gmxpacked_exclusions == NULL || view.pair_shift_bits == NULL)
    {
        return true;
    }
    if (!SITS_Ensure_Sparse_Gmxpacked_View(
            this, cache, view, selective_atom_end))
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
        view.cluster_numbers, local_atom_numbers, selective_atom_end,
        view.cluster_offsets, view.cluster_valid_masks,
        view.cluster_local_masks,
        view.super_cluster_offsets, d_clustered_sparse_sci,
        d_clustered_sparse_cjpacked, view.gmxpacked_exclusions,
        d_clustered_sparse_pair_shift_bits, cache->d_sorted_atom_ids,
        cache->d_sorted_xq, cache->d_sorted_lj_type,
        cache->layout.d_atom_local, atom_sys_mark_local, cell,
        lj_info->d_LJ_AB_packed, cutoff, md_frc,
        pw_select.select_force[0], pme_beta, atom_energy,
        pw_select.select_atom_energy[0], atom_virial,
        pw_select.select_atom_virial_tensor[0], coulomb_atom_ene,
        lj_info->d_LJ_energy_atom, pwwp_enhance_factor,
        need_potential != 0, need_pressure != 0);
#endif
    return true;
}

void SITS_INFORMATION::
    SITS_LJ_Soft_Core_Direct_CF_Force_With_Atom_Energy_And_Virial(
        const int atom_numbers, const int local_atom_numbers,
        const int solvent_numbers, const int ghost_numbers, const VECTOR* crd,
        const float* charge, LJ_SOFT_CORE* lj_info, VECTOR* md_frc,
        const LTMatrix3 cell, const LTMatrix3 rcell, const ATOM_GROUP* nl,
        const float cutoff, const float pme_beta, const int need_potential,
        float* atom_energy, const int need_pressure, LTMatrix3* atom_virial,
        float* coulomb_atom_ene)
{
    if (is_initialized && lj_info->is_initialized)
    {
        Launch_Device_Kernel(
            Copy_Crd_And_Charge_To_New_Crd,
            (this->atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL,
            local_atom_numbers + ghost_numbers, crd,
            lj_info->crd_with_LJ_parameters_local, charge);

        if (need_potential)
        {
            deviceMemset(coulomb_atom_ene, 0,
                         sizeof(float) * (local_atom_numbers + ghost_numbers));
            deviceMemset(lj_info->d_LJ_energy_atom, 0,
                         sizeof(float) * (local_atom_numbers + ghost_numbers));
            deviceMemset(classic_sits.d_bias, 0, sizeof(float));
        }
        if (!local_atom_numbers) return;

        auto f = Selective_Lennard_Jones_And_Direct_Coulomb_Soft_Core_Device<
            true, false, false, true, false>;
        dim3 blockSize = {
            CONTROLLER::device_warp,
            CONTROLLER::device_max_thread / CONTROLLER::device_warp};
        dim3 gridSize = (local_atom_numbers + blockSize.y - 1) / blockSize.y;

        if (need_potential && !need_pressure)
        {
            f = Selective_Lennard_Jones_And_Direct_Coulomb_Soft_Core_Device<
                true, true, false, true, false>;
        }
        else if (need_potential && need_pressure)
        {
            f = Selective_Lennard_Jones_And_Direct_Coulomb_Soft_Core_Device<
                true, true, true, true, false>;
        }
        else if (!need_potential && need_pressure)
        {
            f = Selective_Lennard_Jones_And_Direct_Coulomb_Soft_Core_Device<
                true, false, true, true, false>;
        }
        else
        {
            f = Selective_Lennard_Jones_And_Direct_Coulomb_Soft_Core_Device<
                true, false, false, true, false>;
        }
        Launch_Device_Kernel(
            f, gridSize, blockSize, 0, NULL, local_atom_numbers,
            solvent_numbers, nl, lj_info->d_LJ_energy_atom,
            lj_info->crd_with_LJ_parameters_local, cell, rcell,
            atom_sys_mark_local, lj_info->d_LJ_AA, lj_info->d_LJ_AB,
            lj_info->d_LJ_BA, lj_info->d_LJ_BB, cutoff, md_frc,
            pw_select.select_force[0], pme_beta, atom_energy,
            pw_select.select_atom_energy[0], atom_virial,
            pw_select.select_atom_virial_tensor[0], coulomb_atom_ene, NULL,
            NULL, NULL, lj_info->lambda, lj_info->alpha, lj_info->p,
            lj_info->sigma_6, lj_info->sigma_6_min, pwwp_enhance_factor);
    }
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

static __global__ void Check_Solvent_Atom_Included(int atom_numbers,
                                                   int solvent_numbers,
                                                   int* atom_sys_mark,
                                                   int* errored)
{
#ifdef USE_GPU
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < solvent_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < solvent_numbers; i++)
#endif
    {
        if (!atom_sys_mark[atom_numbers - solvent_numbers + i]) errored[0] = 1;
    }
}

void SITS_INFORMATION::Check_Solvent(CONTROLLER* controller, int atom_numbers,
                                     int solvent_numbers)
{
    if (!is_initialized || solvent_numbers == 0) return;
    int *errored, h_errored;
    Device_Malloc_Safely((void**)&errored, sizeof(int));
    deviceMemset(errored, 0, sizeof(int));
    Launch_Device_Kernel(Check_Solvent_Atom_Included,
                         (solvent_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, atom_numbers,
                         solvent_numbers, atom_sys_mark, errored);

    deviceMemcpy(&h_errored, errored, sizeof(int), deviceMemcpyDeviceToHost);
    if (h_errored == 1)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorConflictingCommand, "SITS_INFORMATION::Check_Solvent",
            "Reason:\n\tYou are trying to apply SITS to the solvents. If YOU "
            "KNOW WHAT YOU ARE DOING, set the command 'solvent_LJ' to 0 to run "
            "the simulation.");
    }
    Free_Single_Device_Pointer((void**)&errored);
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
