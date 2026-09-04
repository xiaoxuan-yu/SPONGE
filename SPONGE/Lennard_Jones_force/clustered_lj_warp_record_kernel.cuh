#pragma once

#ifndef USE_GPU
#error "clustered_lj_warp_record_kernel.cuh is GPU-only"
#endif

// Shared by the production LJ translation unit and NBNXM_MICROBENCH.
// Keep policy decisions and host dispatch outside this file.
enum class ClusteredRegularLJOutputMode
{
    FORCE_ONLY,
    FULL
};

enum class ClusteredRegularLJParameterMode
{
    AB_TABLE,
    COMBINATION
};

enum class ClusteredRegularLJLayoutMode
{
    REGULAR,
    DENSE_OFFSET,
    FULL_LOCAL_DENSE
};

enum class ClusteredRegularLJShiftMode
{
    PAIR_SHIFT,
    SCI_SHIFT_ONLY
};

template <ClusteredRegularLJOutputMode output_mode,
          ClusteredRegularLJParameterMode parameter_mode,
          ClusteredRegularLJLayoutMode layout_mode,
          ClusteredRegularLJShiftMode shift_mode,
          typename ForceTarget = VECTOR, int sci_work_parts = 1,
          bool contiguous_sci_work = false>
static __global__ __launch_bounds__(kClusteredClusterSize *
                                         kClusteredSuperClusterClusters,
                                     parameter_mode ==
                                             ClusteredRegularLJParameterMode::
                                                 COMBINATION
                                         ? 9
                                         : (output_mode ==
                                                        ClusteredRegularLJOutputMode::
                                                            FULL &&
                                                    sci_work_parts > 1
                                                ? 10
                                                : 13))
    void
Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device(
    const int sci_numbers, const int cluster_size,
    const int super_cluster_clusters, const int cluster_numbers,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const int* super_cluster_offsets,
    const LJ_CLUSTERED_GMXPACKED_SCI* sci_entries,
    const LJ_CLUSTERED_GMXPACKED_CJ* cjpacked_entries,
    const LJ_CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    const uint64_t* pair_shift_bits, const int* sci_shift_safe_flags,
    const int sci_shift_safe_value, const int* sorted_atom_ids,
    const float4* sorted_xq, const int* sorted_lj_type,
    const float2* sorted_lj_comb, const LTMatrix3 cell,
    const float2* LJ_type_AB_packed, const float cutoff, ForceTarget* frc,
    const float pme_beta,
    float* atom_energy, LTMatrix3* atom_virial, float* atom_direct_cf_energy,
    float* atom_LJ_ene, const bool store_energy, const bool store_virial)
{
    constexpr bool need_energy =
        output_mode == ClusteredRegularLJOutputMode::FULL;
    constexpr bool need_virial =
        output_mode == ClusteredRegularLJOutputMode::FULL;
    constexpr bool compact_force_storage =
        output_mode == ClusteredRegularLJOutputMode::FULL;
    constexpr bool use_lj_comb =
        parameter_mode == ClusteredRegularLJParameterMode::COMBINATION;
    constexpr bool dense_offsets =
        layout_mode != ClusteredRegularLJLayoutMode::REGULAR;
    constexpr bool full_local_dense =
        layout_mode == ClusteredRegularLJLayoutMode::FULL_LOCAL_DENSE;
    constexpr bool sci_shift_only =
        shift_mode == ClusteredRegularLJShiftMode::SCI_SHIFT_ONLY;
    static_assert(!full_local_dense || dense_offsets,
                  "full-local dense gmxpacked specialization requires dense offsets");
    static_assert(!sci_shift_only || dense_offsets,
                  "sci-shift-only gmxpacked specialization requires dense offsets");
    static_assert(!contiguous_sci_work || sci_work_parts > 1,
                  "contiguous SCI work requires multiple work parts");
    static_assert(sci_work_parts >= 1,
                  "gmxpacked SCI work partition count must be positive");
    static_assert(
        sci_work_parts == 1 ||
            ((!need_energy && !need_virial && !compact_force_storage) ||
             (need_virial && compact_force_storage)),
        "gmxpacked SCI work partitioning requires force-only atom output or "
        "virial compact output");
    constexpr int max_super_cluster_atoms =
        kClusteredClusterSize * kClusteredSuperClusterClusters;
    const int sci = sci_work_parts == 1
                        ? static_cast<int>(blockIdx.x)
                        : static_cast<int>(blockIdx.x) / sci_work_parts;
    const int sci_work_part = sci_work_parts == 1
                                  ? 0
                                  : static_cast<int>(blockIdx.x) %
                                        sci_work_parts;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (sci >= sci_numbers ||
        tid >= super_cluster_clusters * cluster_size)
    {
        return;
    }
    if (sci_shift_safe_flags != NULL &&
        sci_shift_safe_flags[sci] != sci_shift_safe_value)
    {
        return;
    }

    const int i_lane = threadIdx.x;
    const int j_lane = threadIdx.y;
    const int lane = tid & (warpSize - 1);
    const int split = tid / warpSize;
    const int split_j_lane = j_lane - split * kClusteredSplitJClusterSize;
    const int cluster_stride =
        dense_offsets ? kClusteredClusterSize : cluster_size;
    const int i_slot = j_lane * cluster_stride + i_lane;

    const LJ_CLUSTERED_GMXPACKED_SCI sci_entry = sci_entries[sci];
    const int super_i = sci_entry.supercluster_id;
    const int cluster_i_start =
        dense_offsets ? super_i * kClusteredSuperClusterClusters
                      : super_cluster_offsets[super_i];
    const int dense_cluster_i_end =
        cluster_i_start + kClusteredSuperClusterClusters;
    const int cluster_i_end =
        full_local_dense
            ? dense_cluster_i_end
            : (dense_offsets
                   ? (dense_cluster_i_end < cluster_numbers
                          ? dense_cluster_i_end
                          : cluster_numbers)
                   : super_cluster_offsets[super_i + 1]);
    const int active_cluster_count =
        full_local_dense ? kClusteredSuperClusterClusters
                         : cluster_i_end - cluster_i_start;
    const float cutoff_sq = cutoff * cutoff;
    const float beta2 = pme_beta * pme_beta;
    const float beta3 = beta2 * pme_beta;
    constexpr float min_distance_sq = 3.82e-07f;
    const VECTOR sci_shift =
        Clustered_Shift_Vector_From_Id(sci_entry.shift_id, cell);

#define CLUSTERED_GMXPACKED_I_LOCAL_LIST(OP) \
    OP(0)                                    \
    OP(1)                                    \
    OP(2)                                    \
    OP(3)                                    \
    OP(4)                                    \
    OP(5)                                    \
    OP(6)                                    \
    OP(7)

#define CLUSTERED_GMXPACKED_JM_LIST(OP) \
    OP(0)                               \
    OP(1)                               \
    OP(2)                               \
    OP(3)

    __shared__ float4 shared_i_xq[max_super_cluster_atoms];
    __shared__ int shared_i_lj_type[max_super_cluster_atoms];
    __shared__ float2 shared_i_lj_comb[max_super_cluster_atoms];
    __shared__ int shared_i_sorted_ids[max_super_cluster_atoms];
    __shared__ unsigned int shared_i_valid_masks[kClusteredSuperClusterClusters];
    __shared__ unsigned int shared_i_local_masks[kClusteredSuperClusterClusters];
    __shared__ float4 shared_split_virial_lo[2]
                                               [kClusteredSuperClusterClusters]
                                               [kClusteredClusterSize];
    __shared__ float2 shared_split_virial_hi[2]
                                               [kClusteredSuperClusterClusters]
                                               [kClusteredClusterSize];

#define CLUSTERED_GMXPACKED_DECLARE_FCI(I) \
    float fci_x_##I = 0.0f;               \
    float fci_y_##I = 0.0f;               \
    float fci_z_##I = 0.0f;
    CLUSTERED_GMXPACKED_I_LOCAL_LIST(CLUSTERED_GMXPACKED_DECLARE_FCI)
#undef CLUSTERED_GMXPACKED_DECLARE_FCI

    if constexpr (!full_local_dense)
    {
        if (j_lane == 0)
        {
            if (i_lane < active_cluster_count)
            {
                const int cluster_i = cluster_i_start + i_lane;
                shared_i_valid_masks[i_lane] = cluster_valid_masks[cluster_i];
                shared_i_local_masks[i_lane] = cluster_local_masks[cluster_i];
            }
            else if (i_lane < kClusteredSuperClusterClusters)
            {
                shared_i_valid_masks[i_lane] = 0u;
                shared_i_local_masks[i_lane] = 0u;
            }
        }
    }
    if (full_local_dense || j_lane < active_cluster_count)
    {
        const int cluster_i = cluster_i_start + j_lane;
        if (full_local_dense ||
            Clustered_Lane_Is_Valid(cluster_valid_masks[cluster_i], i_lane))
        {
            const int sorted_i =
                (dense_offsets ? cluster_i * kClusteredClusterSize
                               : cluster_offsets[cluster_i]) +
                i_lane;
            float4 i_xq = Clustered_Load_ReadOnly(sorted_xq + sorted_i);
            if constexpr (sci_shift_only)
            {
                i_xq.x += sci_shift.x;
                i_xq.y += sci_shift.y;
                i_xq.z += sci_shift.z;
            }
            shared_i_xq[i_slot] = i_xq;
            if constexpr (use_lj_comb)
            {
                shared_i_lj_comb[i_slot] =
                    Clustered_Load_ReadOnly(sorted_lj_comb + sorted_i);
            }
            else
            {
                shared_i_lj_type[i_slot] =
                    Clustered_Load_ReadOnly(sorted_lj_type + sorted_i);
            }
            if constexpr (!full_local_dense)
            {
                shared_i_sorted_ids[i_slot] = sorted_i;
            }
        }
        else
        {
            shared_i_sorted_ids[i_slot] = -1;
        }
    }
    __syncthreads();

    const unsigned int i_lane_mask = 1u << static_cast<unsigned int>(i_lane);
    unsigned int active_i_mask =
        full_local_dense ? ((1u << kClusteredSuperClusterClusters) - 1u) : 0u;
    if constexpr (!full_local_dense)
    {
#define CLUSTERED_GMXPACKED_CACHE_ACTIVE_I(I)                              \
    if ((I) < active_cluster_count)                                         \
    {                                                                       \
        const unsigned int valid_mask_i = shared_i_valid_masks[I];          \
        const unsigned int local_mask_i = shared_i_local_masks[I];          \
        if (Clustered_Lane_Bit_Is_Set(valid_mask_i, i_lane_mask) &&        \
            Clustered_Lane_Bit_Is_Set(local_mask_i, i_lane_mask))          \
        {                                                                   \
            active_i_mask |= (1u << static_cast<unsigned int>(I));          \
        }                                                                   \
    }
    CLUSTERED_GMXPACKED_I_LOCAL_LIST(CLUSTERED_GMXPACKED_CACHE_ACTIVE_I)
#undef CLUSTERED_GMXPACKED_CACHE_ACTIVE_I
    }

    const auto process_packed = [&](auto energy_tag, auto virial_tag) {
    constexpr bool compute_energy =
        need_energy && decltype(energy_tag)::value;
    constexpr bool compute_virial =
        need_virial && decltype(virial_tag)::value;
    Clustered_Full_Record_Output_Buffer<false, compute_energy,
                                        kClusteredSuperClusterClusters>
        output_buf(compute_energy, compute_virial);
    const int packed_count =
        sci_entry.cjpacked_end - sci_entry.cjpacked_begin;
    const int packed_begin =
        contiguous_sci_work
            ? sci_entry.cjpacked_begin +
                  packed_count * sci_work_part / sci_work_parts
            : sci_entry.cjpacked_begin + sci_work_part;
    const int packed_end =
        contiguous_sci_work
            ? sci_entry.cjpacked_begin +
                  packed_count * (sci_work_part + 1) / sci_work_parts
            : sci_entry.cjpacked_end;
    constexpr int packed_stride =
        contiguous_sci_work ? 1 : sci_work_parts;
    for (int packed_idx = packed_begin; packed_idx < packed_end;
         packed_idx += packed_stride)
    {
        const LJ_CLUSTERED_GMXPACKED_CJ* packed = cjpacked_entries + packed_idx;
        const unsigned int imask = packed->split[split].imask;
        if (imask == 0u)
        {
            continue;
        }
        const unsigned int effective_mask =
            Clustered_Gmxpacked_Effective_Imask(
                *packed, exclusion_entries, split, split_j_lane, i_lane,
                cluster_stride);
#define CLUSTERED_GMXPACKED_COMPUTE_I(I)                                    \
    {                                                                       \
        const unsigned int packed_bit =                                      \
            base_mask << static_cast<unsigned int>(I);                      \
        if ((effective_mask & packed_bit) != 0u &&                          \
            (active_i_mask & (1u << static_cast<unsigned int>(I))) != 0u && \
            (sci_shift_only || pair_shift_bits == NULL ||                  \
             (Clustered_Get_Pair_Active_I_Mask(shift_bits, split) &        \
              (1u << static_cast<unsigned int>(I))) != 0u))                \
        {                                                                   \
            const float4 r1_xq =                                            \
                shared_i_xq[(I) * cluster_stride + i_lane];                 \
            float dx = shifted_j_x - r1_xq.x;                               \
            float dy = shifted_j_y - r1_xq.y;                               \
            float dz = shifted_j_z - r1_xq.z;                               \
            if constexpr (!sci_shift_only)                                  \
            {                                                               \
                const VECTOR pair_shift =                                  \
                    pair_shift_bits != NULL                                \
                        ? Clustered_Shift_Vector_From_Id(                   \
                              Clustered_Get_Pair_Shift_Id(shift_bits, I),   \
                              cell)                                        \
                        : sci_shift;                                       \
                dx -= pair_shift.x;                                        \
                dy -= pair_shift.y;                                        \
                dz -= pair_shift.z;                                        \
            }                                                               \
            const float dr2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));          \
            if (dr2 < cutoff_sq && dr2 != 0.0f)                             \
            {                                                               \
                const float r2 = fmaxf(dr2, min_distance_sq);               \
                const float inv_r = rsqrtf(r2);                             \
                const float inv_r2 = inv_r * inv_r;                         \
                const float inv_r6 = inv_r2 * inv_r2 * inv_r2;              \
                const float beta_dr = pme_beta * (r2 * inv_r);              \
                const float charge_product = r1_xq.w * qj;                 \
                float c12 = 0.0f;                                           \
                float c6 = 0.0f;                                            \
                if constexpr (use_lj_comb)                                  \
                {                                                           \
                    const float2 ljcp_i =                                   \
                        shared_i_lj_comb[(I) * cluster_stride + i_lane];    \
                    c6 = ljcp_i.x * lj_j_x;                                 \
                    c12 = ljcp_i.y * lj_j_y;                                \
                }                                                           \
                else                                                        \
                {                                                           \
                    const int r1_lj_type =                                  \
                        shared_i_lj_type[(I) * cluster_stride + i_lane];    \
                    const int atom_pair_LJ_type =                           \
                        Clustered_Gmxpacked_Get_LJ_Type_MinMax(             \
                            r1_lj_type, r2_lj_type);                        \
                    const float2 AB = Clustered_Load_ReadOnly(              \
                        LJ_type_AB_packed + atom_pair_LJ_type);             \
                    c12 = AB.x;                                             \
                    c6 = AB.y;                                              \
                }                                                           \
                float frc_abs =                                             \
                    Get_Clustered_LJ_Force_Abs(inv_r2, inv_r6, c12, c6);    \
                frc_abs -=                                                  \
                    Get_Clustered_Direct_Coulomb_Force_Abs_PME_Corr(        \
                        charge_product, inv_r, inv_r2, beta2 * r2, beta3);  \
                const float fij_x = frc_abs * dx;                           \
                const float fij_y = frc_abs * dy;                           \
                const float fij_z = frc_abs * dz;                           \
                fci_x_##I += fij_x;                                         \
                fci_y_##I += fij_y;                                         \
                fci_z_##I += fij_z;                                         \
                if (j_is_local)                                             \
                {                                                           \
                    fcj_x -= fij_x;                                         \
                    fcj_y -= fij_y;                                         \
                    fcj_z -= fij_z;                                         \
                }                                                           \
                if constexpr (compute_virial)                               \
                {                                                           \
                    const LTMatrix3 pair_virial = {                         \
                        -ij_factor * fij_x * dx,                            \
                        -ij_factor * (fij_x * dy + fij_y * dx),             \
                        -ij_factor * fij_y * dy,                            \
                        -ij_factor * (fij_x * dz + fij_z * dx),             \
                        -ij_factor * (fij_y * dz + fij_z * dy),             \
                        -ij_factor * fij_z * dz};                           \
                    output_buf.virial[I] =                                 \
                        output_buf.virial[I] + pair_virial;                 \
                }                                                           \
                if constexpr (compute_energy)                               \
                {                                                           \
                    const float pair_energy_lj =                            \
                        ij_factor *                                         \
                        Get_Clustered_LJ_Energy(inv_r6, c12, c6);           \
                    const float pair_energy_coulomb =                       \
                        ij_factor * Get_Clustered_Direct_Coulomb_Energy(    \
                                        charge_product, inv_r, beta_dr);     \
                    output_buf.energy_lj[I] += pair_energy_lj;              \
                    output_buf.energy_coulomb[I] += pair_energy_coulomb;    \
                }                                                           \
            }                                                               \
        }                                                                   \
    }

#define CLUSTERED_GMXPACKED_PROCESS_JM(JM)                                  \
    {                                                                       \
        constexpr unsigned int base_mask =                                  \
            1u << ((JM) * kClusteredSuperClusterClusters);                  \
        constexpr unsigned int jm_mask =                                    \
            ((1u << kClusteredSuperClusterClusters) - 1u)                   \
            << ((JM) * kClusteredSuperClusterClusters);                     \
        if ((imask & jm_mask) != 0u)                                        \
        {                                                                   \
            const int cluster_j = packed->cj[JM];                           \
            if (cluster_j >= 0)                                             \
            {                                                               \
                uint64_t shift_bits = 0ull;                                 \
                if constexpr (!sci_shift_only)                              \
                {                                                           \
                    shift_bits =                                            \
                        pair_shift_bits != NULL                             \
                            ? pair_shift_bits[packed_idx *                  \
                                              kClusteredJGroupSize + (JM)]   \
                            : 0ull;                                         \
                }                                                           \
                const unsigned int valid_mask_j =                           \
                    full_local_dense ?                                      \
                        ((1u << kClusteredClusterSize) - 1u) :              \
                        cluster_valid_masks[cluster_j];                     \
                if (full_local_dense ||                                     \
                    Clustered_Lane_Is_Valid(valid_mask_j, j_lane))          \
                {                                                           \
                    const int sorted_j =                                    \
                        (dense_offsets ? cluster_j * kClusteredClusterSize  \
                                       : cluster_offsets[cluster_j]) +      \
                        j_lane;                                             \
                    const float4 r2_xq =                                    \
                        Clustered_Load_ReadOnly(sorted_xq + sorted_j);      \
                    int r2_lj_type = 0;                                     \
                    float lj_j_x = 0.0f;                                    \
                    float lj_j_y = 0.0f;                                    \
                    if constexpr (use_lj_comb)                              \
                    {                                                       \
                        const float2 r2_lj_comb =                           \
                            Clustered_Load_ReadOnly(sorted_lj_comb + sorted_j); \
                        lj_j_x = r2_lj_comb.x;                              \
                        lj_j_y = r2_lj_comb.y;                              \
                    }                                                       \
                    else                                                    \
                    {                                                       \
                        r2_lj_type =                                        \
                            Clustered_Load_ReadOnly(sorted_lj_type + sorted_j); \
                    }                                                       \
                    const float shifted_j_x = r2_xq.x;                      \
                    const float shifted_j_y = r2_xq.y;                      \
                    const float shifted_j_z = r2_xq.z;                      \
                    const float qj = r2_xq.w;                               \
                    const bool j_is_local =                                 \
                        full_local_dense ||                                 \
                        Clustered_Lane_Is_Local(                            \
                            cluster_local_masks[cluster_j], j_lane);        \
                    const float ij_factor = j_is_local ? 1.0f : 0.5f;       \
                    float fcj_x = 0.0f;                                     \
                    float fcj_y = 0.0f;                                     \
                    float fcj_z = 0.0f;                                     \
                    CLUSTERED_GMXPACKED_I_LOCAL_LIST(                       \
                        CLUSTERED_GMXPACKED_COMPUTE_I)                      \
                    if (j_is_local)                                         \
                    {                                                       \
                        const float fcj_component =                         \
                            Reduce_Clustered_Subgroup_Vector_To_Component(  \
                                fcj_x, fcj_y, fcj_z, i_lane,                \
                                lane, cluster_stride);                      \
                        if (i_lane < 3)                                     \
                        {                                                   \
                            const int force_index =                         \
                                compact_force_storage                       \
                                    ? sorted_j                              \
                                    : sorted_atom_ids[sorted_j];            \
                            Clustered_Atomic_Add_Force_Component(            \
                                frc, force_index, i_lane, fcj_component);   \
                        }                                                   \
                    }                                                       \
                }                                                           \
            }                                                               \
        }                                                                   \
    }
        CLUSTERED_GMXPACKED_JM_LIST(CLUSTERED_GMXPACKED_PROCESS_JM)
#undef CLUSTERED_GMXPACKED_PROCESS_JM
#undef CLUSTERED_GMXPACKED_COMPUTE_I
    }

#define CLUSTERED_GMXPACKED_REDUCE_I(I)                                     \
    if (full_local_dense || (I) < active_cluster_count)                     \
    {                                                                       \
        const bool active_i =                                               \
            full_local_dense ||                                             \
            (active_i_mask & (1u << static_cast<unsigned int>(I))) != 0u;   \
        float reduced_x = active_i ? fci_x_##I : 0.0f;                      \
        float reduced_y = active_i ? fci_y_##I : 0.0f;                      \
        float reduced_z = active_i ? fci_z_##I : 0.0f;                      \
        const float reduced_component =                                     \
            Reduce_Clustered_Warp_I_To_Component(                           \
                reduced_x, reduced_y, reduced_z, i_lane,                    \
                split_j_lane, cluster_stride);                              \
        if (active_i && split_j_lane < 3)                                   \
        {                                                                   \
            int sorted_i = 0;                                               \
            if constexpr (full_local_dense)                                 \
            {                                                                \
                sorted_i = (cluster_i_start + (I)) *                        \
                           kClusteredClusterSize + i_lane;                   \
            }                                                                \
            else                                                            \
            {                                                                \
                sorted_i =                                                  \
                    shared_i_sorted_ids[(I) * cluster_stride + i_lane];     \
            }                                                                \
            const int force_index =                                         \
                compact_force_storage ? sorted_i : sorted_atom_ids[sorted_i]; \
            Clustered_Atomic_Add_Force_Component(                           \
                frc, force_index, split_j_lane, reduced_component);         \
        }                                                                   \
        if constexpr (compute_virial)                                       \
        {                                                                    \
            if (store_virial)                                                \
            {                                                                \
                LTMatrix3 reduced_virial =                                   \
                    active_i ? output_buf.virial[I]                          \
                             : LTMatrix3{0.0f, 0.0f, 0.0f,                   \
                                         0.0f, 0.0f, 0.0f};                 \
                reduced_virial = Reduce_Clustered_Warp_Virial_Over_J(        \
                    reduced_virial, cluster_stride);                         \
                if constexpr (sci_work_parts == 2)                           \
                {                                                            \
                    if (lane < cluster_stride)                               \
                    {                                                        \
                        shared_split_virial_lo[split][I][i_lane] =           \
                            Pack_Clustered_Virial_Lo(reduced_virial);         \
                        shared_split_virial_hi[split][I][i_lane] =           \
                            Pack_Clustered_Virial_Hi(reduced_virial);         \
                    }                                                        \
                }                                                            \
                else if (active_i && lane < cluster_stride)                  \
                {                                                            \
                    const int sorted_i =                                     \
                        full_local_dense                                     \
                            ? (cluster_i_start + (I)) *                      \
                                  kClusteredClusterSize + i_lane             \
                            : shared_i_sorted_ids[                           \
                                  (I) * cluster_stride + i_lane];            \
                    const int atom_i = sorted_atom_ids[sorted_i];            \
                    if (atom_i >= 0)                                         \
                    {                                                        \
                        atomicAdd(atom_virial + atom_i, reduced_virial);     \
                    }                                                        \
                }                                                            \
            }                                                                \
        }                                                                    \
        if constexpr (compute_energy)                                       \
        {                                                                    \
            if (store_energy)                                                \
            {                                                                \
                float reduced_lj =                                          \
                    active_i ? output_buf.energy_lj[I] : 0.0f;              \
                float reduced_coulomb =                                     \
                    active_i ? output_buf.energy_coulomb[I] : 0.0f;          \
                reduced_lj = Reduce_Clustered_Warp_Float_Over_J(             \
                    reduced_lj, cluster_stride);                             \
                reduced_coulomb = Reduce_Clustered_Warp_Float_Over_J(        \
                    reduced_coulomb, cluster_stride);                        \
                if (active_i && lane < cluster_stride)                       \
                {                                                            \
                    const int sorted_i =                                     \
                        full_local_dense                                     \
                            ? (cluster_i_start + (I)) *                      \
                                  kClusteredClusterSize + i_lane             \
                            : shared_i_sorted_ids[                           \
                                  (I) * cluster_stride + i_lane];            \
                    const int atom_i = sorted_atom_ids[sorted_i];            \
                    if (atom_i >= 0)                                         \
                    {                                                        \
                        atomicAdd(atom_energy + atom_i,                      \
                                  reduced_lj + reduced_coulomb);             \
                        atomicAdd(atom_LJ_ene + atom_i, reduced_lj);         \
                        atomicAdd(atom_direct_cf_energy + atom_i,            \
                                  reduced_coulomb);                          \
                    }                                                        \
                }                                                            \
            }                                                                \
        }                                                                    \
    }
    CLUSTERED_GMXPACKED_I_LOCAL_LIST(CLUSTERED_GMXPACKED_REDUCE_I)
#undef CLUSTERED_GMXPACKED_REDUCE_I

    if constexpr (compute_virial && sci_work_parts == 2)
    {
        if (store_virial)
        {
            __syncthreads();
#define CLUSTERED_GMXPACKED_WRITE_MERGED_VIRIAL(I)                          \
    if (split == 0 &&                                                       \
        (full_local_dense || (I) < active_cluster_count))                   \
    {                                                                       \
        const bool active_i =                                               \
            full_local_dense ||                                             \
            (active_i_mask & (1u << static_cast<unsigned int>(I))) != 0u;   \
        if (active_i && lane < cluster_stride)                              \
        {                                                                   \
            const int sorted_i =                                            \
                full_local_dense                                            \
                    ? (cluster_i_start + (I)) * kClusteredClusterSize +     \
                          i_lane                                             \
                    : shared_i_sorted_ids[(I) * cluster_stride + i_lane];   \
            const int atom_i = sorted_atom_ids[sorted_i];                   \
            if (atom_i >= 0)                                                \
            {                                                               \
                const LTMatrix3 merged_virial =                             \
                    Unpack_Clustered_Virial(                                \
                        shared_split_virial_lo[0][I][i_lane],                \
                        shared_split_virial_hi[0][I][i_lane]) +              \
                    Unpack_Clustered_Virial(                                \
                        shared_split_virial_lo[1][I][i_lane],                \
                        shared_split_virial_hi[1][I][i_lane]);               \
                atomicAdd(atom_virial + atom_i, merged_virial);             \
            }                                                               \
        }                                                                   \
    }
        CLUSTERED_GMXPACKED_I_LOCAL_LIST(
            CLUSTERED_GMXPACKED_WRITE_MERGED_VIRIAL)
#undef CLUSTERED_GMXPACKED_WRITE_MERGED_VIRIAL
        }
    }
    };

    if constexpr (!need_energy && !need_virial)
    {
        process_packed(std::false_type{}, std::false_type{});
    }
    else if (store_energy)
    {
        // Energy requests use the full output path.  Virial-only pressure
        // updates retain a buffer-specialized path inside the same externally
        // visible full kernel variant.
        process_packed(std::true_type{}, std::true_type{});
    }
    else
    {
        process_packed(std::false_type{}, std::true_type{});
    }

#undef CLUSTERED_GMXPACKED_JM_LIST
#undef CLUSTERED_GMXPACKED_I_LOCAL_LIST
}
