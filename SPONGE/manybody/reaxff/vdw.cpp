#include "vdw.h"

static const int p_rvdw = 0;
static const int p_epsilon = 1;
static const int p_alpha = 2;
static const int p_gamma_w = 3;
static const int PARAM_STRIDE = 8;

template <int N>
__device__ __forceinline__ SADfloat<N> reax_vdw_energy_sad(SADfloat<N> r,
                                                           const float* param,
                                                           float cutoff,
                                                           float p_vdw1)
{
    float rvdw = param[p_rvdw];
    float epsilon = param[p_epsilon];
    float alpha = param[p_alpha];
    float gamma_w = param[p_gamma_w];

    if (r.val > cutoff) return SADfloat<N>(0.0f);

    SADfloat<N> x = r / SADfloat<N>(cutoff);
    SADfloat<N> x4 = x * x * x * x;
    SADfloat<N> x5 = x4 * x;
    SADfloat<N> x6 = x5 * x;
    SADfloat<N> x7 = x6 * x;

    SADfloat<N> tap = SADfloat<N>(1.0f) - SADfloat<N>(35.0f) * x4 +
                      SADfloat<N>(84.0f) * x5 - SADfloat<N>(70.0f) * x6 +
                      SADfloat<N>(20.0f) * x7;

    float inv_gamma = 1.0f / gamma_w;
    SADfloat<N> inv_gamma_p = powf(SADfloat<N>(inv_gamma), SADfloat<N>(p_vdw1));
    SADfloat<N> r_p = powf(r, SADfloat<N>(p_vdw1));
    SADfloat<N> shielded_r =
        powf(r_p + inv_gamma_p, SADfloat<N>(1.0f / p_vdw1));

    SADfloat<N> exp_term = alpha * (1.0f - shielded_r / rvdw);
    SADfloat<N> term1 = expf(exp_term);
    SADfloat<N> term2 = -2.0f * expf(0.5f * exp_term);

    return tap * epsilon * (term1 + term2);
}

static __global__ void REAXFF_VDW_Gather_Clustered_Coordinates(
    const int total_numbers, const int cluster_numbers,
    const int* sort_permutation, const int* cluster_offsets,
    const VECTOR* cluster_centers, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, VECTOR* sorted_crd)
{
#ifdef USE_GPU
    const int sorted_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (sorted_i < total_numbers)
#else
#pragma omp parallel for
    for (int sorted_i = 0; sorted_i < total_numbers; sorted_i += 1)
#endif
    {
        const int cluster_i = Clustered_Find_Cluster_For_Sorted_Index(
            sorted_i, cluster_numbers, cluster_offsets);
        const int atom_i = sort_permutation[sorted_i];
        const VECTOR center = cluster_centers[cluster_i];
        sorted_crd[sorted_i] = center + Get_Periodic_Displacement(
                                            crd[atom_i], center, cell, rcell);
    }
}

#ifdef USE_GPU
static __device__ __forceinline__ float REAXFF_VDW_Reduce_Subgroup(float value)
{
#ifdef USE_CUDA
    const unsigned int active_mask = __activemask();
#endif
    for (int delta = 4; delta >= 1; delta >>= 1)
    {
#ifdef USE_CUDA
        value += __shfl_down_sync(active_mask, value, delta, 8);
#else
        value += __shfl_down(value, delta, 8);
#endif
    }
    return value;
}

template <bool full_output>
static __global__ void REAXFF_VDW_Clustered_Gmxpacked(
    const int sci_numbers, const int packed_partitions,
    const int cluster_numbers, const int* cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const int* super_cluster_offsets,
    const CLUSTERED_GMXPACKED_SCI* sci_entries,
    const CLUSTERED_GMXPACKED_CJ* cjpacked_entries,
    const CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    const uint64_t* pair_shift_bits, const int* sorted_atom_ids,
    const VECTOR* sorted_crd, const int* atom_types, const float* params,
    const int ntypes, const float cutoff, const float p_vdw1,
    const LTMatrix3 cell, VECTOR* frc, float* atom_energy,
    LTMatrix3* atom_virial, float* energy_sum, const bool store_energy,
    const bool store_virial)
{
    const int sci = static_cast<int>(blockIdx.x);
    const int packed_partition = static_cast<int>(blockIdx.y);
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
    int cluster_i_end = super_cluster_offsets[sci_entry.supercluster_id + 1];
    if (cluster_i_end > cluster_numbers) cluster_i_end = cluster_numbers;
    const int split = j_lane / kClusteredSplitJClusterSize;
    const int split_j_lane = j_lane - split * kClusteredSplitJClusterSize;
    const unsigned int i_lane_bit = 1u << i_lane;
    const unsigned int j_lane_bit = 1u << j_lane;
    const float cutoff_sq = cutoff * cutoff;

    for (int packed_idx = sci_entry.cjpacked_begin + packed_partition;
         packed_idx < sci_entry.cjpacked_end; packed_idx += packed_partitions)
    {
        const CLUSTERED_GMXPACKED_CJ packed = cjpacked_entries[packed_idx];
        const CLUSTERED_GMXPACKED_SPLIT split_entry = packed.split[split];
        unsigned int pair_bits = 0xffffffffu;
        if (split_entry.exclusion_index != 0)
        {
            pair_bits =
                exclusion_entries[split_entry.exclusion_index]
                    .pair[split_j_lane * kClusteredClusterSize + i_lane];
        }
        const unsigned int effective_mask = split_entry.imask & pair_bits;
        for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
        {
            const int cluster_j = packed.cj[jm];
            if (cluster_j < 0 ||
                !Clustered_Lane_Bit_Is_Set(cluster_valid_masks[cluster_j],
                                           j_lane_bit) ||
                !Clustered_Lane_Bit_Is_Set(cluster_local_masks[cluster_j],
                                           j_lane_bit))
            {
                continue;
            }
            const int sorted_j = cluster_offsets[cluster_j] + j_lane;
            const int atom_j = sorted_atom_ids[sorted_j];
            const int type_j = atom_types[atom_j];
            const VECTOR rj = sorted_crd[sorted_j];
            const uint64_t shift_bits =
                pair_shift_bits[packed_idx * kClusteredJGroupSize + jm];
            VECTOR force_j = {0.0f, 0.0f, 0.0f};
            float energy_j = 0.0f;
            LTMatrix3 virial_j = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            for (int i_local = 0; i_local < cluster_i_end - cluster_i_begin;
                 i_local += 1)
            {
                const unsigned int packed_bit =
                    1u << (jm * kClusteredSuperClusterClusters + i_local);
                if ((effective_mask & packed_bit) == 0u ||
                    (Clustered_Get_Pair_Active_I_Mask(shift_bits, split) &
                     (1u << static_cast<unsigned int>(i_local))) == 0u)
                {
                    continue;
                }
                const int cluster_i = cluster_i_begin + i_local;
                if (!Clustered_Lane_Bit_Is_Set(cluster_valid_masks[cluster_i],
                                               i_lane_bit) ||
                    !Clustered_Lane_Bit_Is_Set(cluster_local_masks[cluster_i],
                                               i_lane_bit))
                {
                    continue;
                }
                const int sorted_i = cluster_offsets[cluster_i] + i_lane;
                const int atom_i = sorted_atom_ids[sorted_i];
                const int type_i = atom_types[atom_i];
                const int shift_id =
                    Clustered_Get_Pair_Shift_Id(shift_bits, i_local);
                const VECTOR shift =
                    Clustered_Shift_Vector_From_Id(shift_id, cell);
                const VECTOR drij = (sorted_crd[sorted_i] - rj) + shift;
                const float rij_sq = drij * drij;
                if (rij_sq <= 0.0f || rij_sq >= cutoff_sq) continue;
                const float rij = sqrtf(rij_sq);
                const float* param =
                    params + (type_i * ntypes + type_j) * PARAM_STRIDE;
                const SADfloat<1> energy_sad = reax_vdw_energy_sad(
                    SADfloat<1>(rij, 0), param, cutoff, p_vdw1);
                const VECTOR force_i = (-energy_sad.dval[0] / rij) * drij;
                atomicAdd(&frc[atom_i].x, force_i.x);
                atomicAdd(&frc[atom_i].y, force_i.y);
                atomicAdd(&frc[atom_i].z, force_i.z);
                force_j = force_j - force_i;
                if constexpr (full_output)
                {
                    if (store_energy)
                    {
                        energy_j += energy_sad.val;
                    }
                    if (store_virial)
                    {
                        virial_j =
                            virial_j + Get_Virial_From_Force_Dis(force_i, drij);
                    }
                }
            }
            const VECTOR reduced_force_j = {
                REAXFF_VDW_Reduce_Subgroup(force_j.x),
                REAXFF_VDW_Reduce_Subgroup(force_j.y),
                REAXFF_VDW_Reduce_Subgroup(force_j.z)};
            if (i_lane == 0)
            {
                atomicAdd(&frc[atom_j].x, reduced_force_j.x);
                atomicAdd(&frc[atom_j].y, reduced_force_j.y);
                atomicAdd(&frc[atom_j].z, reduced_force_j.z);
            }
            if constexpr (full_output)
            {
                const float reduced_energy_j =
                    REAXFF_VDW_Reduce_Subgroup(energy_j);
                const LTMatrix3 reduced_virial_j = {
                    REAXFF_VDW_Reduce_Subgroup(virial_j.a11),
                    REAXFF_VDW_Reduce_Subgroup(virial_j.a21),
                    REAXFF_VDW_Reduce_Subgroup(virial_j.a22),
                    REAXFF_VDW_Reduce_Subgroup(virial_j.a31),
                    REAXFF_VDW_Reduce_Subgroup(virial_j.a32),
                    REAXFF_VDW_Reduce_Subgroup(virial_j.a33)};
                if (i_lane == 0)
                {
                    if (store_energy)
                    {
                        atomicAdd(atom_energy + atom_j, reduced_energy_j);
                        atomicAdd(energy_sum, reduced_energy_j);
                    }
                    if (store_virial)
                    {
                        atomicAdd(atom_virial + atom_j, reduced_virial_j);
                    }
                }
            }
        }
    }
}
#endif

#ifdef USE_CPU
template <bool full_output>
static void REAXFF_VDW_Clustered_Gmxpacked_CPU(
    const CLUSTERED_SPATIAL_VIEW& view, const int* sorted_atom_ids,
    const VECTOR* sorted_crd, const int* atom_types, const float* params,
    const int ntypes, const float cutoff, const float p_vdw1,
    const LTMatrix3 cell, VECTOR* frc, float* atom_energy,
    LTMatrix3* atom_virial, float* energy_sum, const bool store_energy,
    const bool store_virial)
{
    const float cutoff_sq = cutoff * cutoff;
#pragma omp parallel for schedule(dynamic)
    for (int sci = 0; sci < view.gmxpacked_sci_numbers; sci += 1)
    {
        const CLUSTERED_GMXPACKED_SCI sci_entry = view.gmxpacked_sci[sci];
        const int cluster_i_begin =
            view.super_cluster_offsets[sci_entry.supercluster_id];
        const int cluster_i_end =
            view.super_cluster_offsets[sci_entry.supercluster_id + 1];
        const int cluster_i_numbers = cluster_i_end - cluster_i_begin;
        const VECTOR shift =
            Clustered_Shift_Vector_From_Id(sci_entry.shift_id, cell);
        for (int packed_idx = sci_entry.cjpacked_begin;
             packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
        {
            const CLUSTERED_GMXPACKED_CJ& packed =
                view.gmxpacked_cjpacked[packed_idx];
            for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
            {
                const int cluster_j = packed.cj[jm];
                if (cluster_j < 0) continue;
                const unsigned int jm_shift = static_cast<unsigned int>(
                    jm * kClusteredSuperClusterClusters);
                for (int j_lane = 0; j_lane < view.cluster_size; j_lane += 1)
                {
                    if (!Clustered_Lane_Is_Valid(
                            view.cluster_valid_masks[cluster_j], j_lane) ||
                        !Clustered_Lane_Is_Local(
                            view.cluster_local_masks[cluster_j], j_lane))
                    {
                        continue;
                    }
                    const int split = j_lane / kClusteredSplitJClusterSize;
                    const int split_j_lane =
                        j_lane - split * kClusteredSplitJClusterSize;
                    const CLUSTERED_GMXPACKED_SPLIT& split_entry =
                        packed.split[split];
                    const int sorted_j =
                        view.cluster_offsets[cluster_j] + j_lane;
                    const int atom_j = sorted_atom_ids[sorted_j];
                    const int type_j = atom_types[atom_j];
                    const VECTOR rj = sorted_crd[sorted_j];
                    VECTOR force_j = {0.0f, 0.0f, 0.0f};
                    float energy_j = 0.0f;
                    float pair_energy_sum = 0.0f;
                    LTMatrix3 virial_j = {};
                    for (int i_lane = 0; i_lane < view.cluster_size;
                         i_lane += 1)
                    {
                        unsigned int pair_bits = 0xffffffffu;
                        if (split_entry.exclusion_index != 0)
                        {
                            pair_bits =
                                view.gmxpacked_exclusions[split_entry
                                                              .exclusion_index]
                                    .pair[split_j_lane * kClusteredClusterSize +
                                          i_lane];
                        }
                        const unsigned int active_i_mask =
                            (split_entry.imask & pair_bits) >> jm_shift;
                        if (active_i_mask == 0u) continue;
                        for (int i_local = 0; i_local < cluster_i_numbers;
                             i_local += 1)
                        {
                            if ((active_i_mask &
                                 (1u << static_cast<unsigned int>(i_local))) ==
                                0u)
                            {
                                continue;
                            }
                            const int cluster_i = cluster_i_begin + i_local;
                            if (!Clustered_Lane_Is_Valid(
                                    view.cluster_valid_masks[cluster_i],
                                    i_lane) ||
                                !Clustered_Lane_Is_Local(
                                    view.cluster_local_masks[cluster_i],
                                    i_lane))
                            {
                                continue;
                            }
                            const int sorted_i =
                                view.cluster_offsets[cluster_i] + i_lane;
                            const int atom_i = sorted_atom_ids[sorted_i];
                            const int type_i = atom_types[atom_i];
                            const VECTOR drij =
                                (sorted_crd[sorted_i] - rj) + shift;
                            const float rij_sq = drij * drij;
                            if (rij_sq <= 0.0f || rij_sq >= cutoff_sq)
                            {
                                continue;
                            }
                            const float rij = sqrtf(rij_sq);
                            const float* param =
                                params +
                                (type_i * ntypes + type_j) * PARAM_STRIDE;
                            const SADfloat<1> energy_sad = reax_vdw_energy_sad(
                                SADfloat<1>(rij, 0), param, cutoff, p_vdw1);
                            const VECTOR force_i =
                                (-energy_sad.dval[0] / rij) * drij;
                            atomicAdd(frc + atom_i, force_i);
                            force_j = force_j - force_i;
                            if constexpr (full_output)
                            {
                                const int owner =
                                    atom_i < atom_j ? atom_i : atom_j;
                                if (store_energy)
                                {
                                    if (owner == atom_j)
                                    {
                                        energy_j += energy_sad.val;
                                    }
                                    else
                                    {
                                        atomicAdd(atom_energy + owner,
                                                  energy_sad.val);
                                    }
                                    pair_energy_sum += energy_sad.val;
                                }
                                if (store_virial)
                                {
                                    const LTMatrix3 pair_virial =
                                        Get_Virial_From_Force_Dis(force_i,
                                                                  drij);
                                    if (owner == atom_j)
                                    {
                                        virial_j = virial_j + pair_virial;
                                    }
                                    else
                                    {
                                        atomicAdd(atom_virial + owner,
                                                  pair_virial);
                                    }
                                }
                            }
                        }
                    }
                    atomicAdd(frc + atom_j, force_j);
                    if constexpr (full_output)
                    {
                        if (store_energy)
                        {
                            atomicAdd(atom_energy + atom_j, energy_j);
                            atomicAdd(energy_sum, pair_energy_sum);
                        }
                        if (store_virial)
                        {
                            atomicAdd(atom_virial + atom_j, virial_j);
                        }
                    }
                }
            }
        }
    }
}
#endif

void REAXFF_VDW::Initial(CONTROLLER* controller, int atom_numbers,
                         const char* module_name)
{
    if (module_name == NULL) module_name = "REAXFF";
    this->atom_numbers = atom_numbers;
    if (!controller->Command_Exist(module_name, "in_file")) return;

    controller->printf("START INITIALIZING REAXFF VDW FORCE\n");
    const char* parameter_in_file = controller->Command(module_name, "in_file");
    const char* type_in_file = controller->Command(module_name, "type_in_file");
    if (parameter_in_file == NULL || type_in_file == NULL)
    {
        controller->printf(
            "REAXFF_VDW IS NOT INITIALIZED (missing input files)\n\n");
        return;
    }

    FILE* fp_p;
    Open_File_Safely(&fp_p, parameter_in_file, "r");
    char line[1024];
    auto throw_bad_format = [&](const char* file_name, const char* reason)
    {
        char error_msg[1024];
        sprintf(error_msg, "Reason:\n\t%s in file %s\n", reason, file_name);
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "REAXFF_VDW::Initial", error_msg);
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
    if (sscanf(line, "%d", &n_gen_params) != 1 || n_gen_params < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of general parameters");
    }

    std::vector<float> gen_params;
    gen_params.reserve(n_gen_params);
    for (int i = 0; i < n_gen_params; i++)
    {
        read_line_or_throw(fp_p, parameter_in_file, "general parameter block");
        float val;
        if (sscanf(line, "%f", &val) != 1)
        {
            char reason[512];
            sprintf(reason, "failed to parse general parameter at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        gen_params.push_back(val);
    }
    if (gen_params.size() <= 28)
    {
        throw_bad_format(parameter_in_file,
                         "missing general parameter p_vdw1 at index 29");
    }
    this->p_vdw1 = gen_params[28];

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
    std::vector<float> rvdw(n_atom_types);
    std::vector<float> epsilon(n_atom_types);
    std::vector<float> alpha(n_atom_types);
    std::vector<float> gamma_w(n_atom_types);

    for (int i = 0; i < n_atom_types; i++)
    {
        read_line_or_throw(fp_p, parameter_in_file, "atom type block line 1");
        char element_name[16];
        float ro_sigma, valency, mass, rvdw_val, epsilon_val, gamma_val, ro_pi,
            valency_e;
        if (sscanf(line, "%15s %f %f %f %f %f %f %f %f", element_name,
                   &ro_sigma, &valency, &mass, &rvdw_val, &epsilon_val,
                   &gamma_val, &ro_pi, &valency_e) != 9)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 1 at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        type_map[std::string(element_name)] = i;

        rvdw[i] = rvdw_val;
        epsilon[i] = epsilon_val;

        read_line_or_throw(fp_p, parameter_in_file, "atom type block line 2");
        float alpha_val, gamma_w_val;
        if (sscanf(line, "%f %f", &alpha_val, &gamma_w_val) != 2)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 2 at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        alpha[i] = alpha_val;
        gamma_w[i] = gamma_w_val;

        read_line_or_throw(fp_p, parameter_in_file, "atom type block line 3");
        read_line_or_throw(fp_p, parameter_in_file, "atom type block line 4");
    }

    Malloc_Safely((void**)&h_twobody_params,
                  sizeof(float) * n_atom_types * n_atom_types * PARAM_STRIDE);
    memset(h_twobody_params, 0,
           sizeof(float) * n_atom_types * n_atom_types * PARAM_STRIDE);

    for (int i = 0; i < n_atom_types; i++)
    {
        for (int j = 0; j < n_atom_types; j++)
        {
            int idx = (i * n_atom_types + j) * PARAM_STRIDE;
            float rvdw_ij = 2.0f * sqrtf(rvdw[i] * rvdw[j]);
            float epsilon_ij = sqrtf(epsilon[i] * epsilon[j]);
            float alpha_ij = sqrtf(alpha[i] * alpha[j]);
            float gamma_w_ij = sqrtf(gamma_w[i] * gamma_w[j]);

            h_twobody_params[idx + p_rvdw] = rvdw_ij;
            h_twobody_params[idx + p_epsilon] = epsilon_ij;
            h_twobody_params[idx + p_alpha] = alpha_ij;
            h_twobody_params[idx + p_gamma_w] = gamma_w_ij;
        }
    }

    read_line_or_throw(fp_p, parameter_in_file, "bond parameter count line");
    int n_bond_params = 0;
    if (sscanf(line, "%d", &n_bond_params) != 1 || n_bond_params < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of bond parameters");
    }
    read_line_or_throw(fp_p, parameter_in_file, "bond parameter header line");
    for (int i = 0; i < n_bond_params; i++)
    {
        read_line_or_throw(fp_p, parameter_in_file,
                           "bond parameter block line 1");
        read_line_or_throw(fp_p, parameter_in_file,
                           "bond parameter block line 2");
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
                               "off-diagonal parameter entry");
            int t1, t2;
            float dij, rvdw_od, alfa_od, ro_sigma_od, ro_pi_od, ro_pipi_od;
            int read_cnt = sscanf(line, "%d %d %f %f %f %f %f %f", &t1, &t2,
                                  &dij, &rvdw_od, &alfa_od, &ro_sigma_od,
                                  &ro_pi_od, &ro_pipi_od);

            if (read_cnt < 8)
            {
                char reason[512];
                sprintf(
                    reason,
                    "failed to parse off-diagonal parameter entry at index %d",
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
                        "off-diagonal atom type index out of range at index %d",
                        off + 1);
                throw_bad_format(parameter_in_file, reason);
            }

            int pair_idx1 = (idx1 * n_atom_types + idx2) * PARAM_STRIDE;
            int pair_idx2 = (idx2 * n_atom_types + idx1) * PARAM_STRIDE;

            if (dij > 0.0f)
            {
                h_twobody_params[pair_idx1 + p_epsilon] =
                    h_twobody_params[pair_idx2 + p_epsilon] = dij;
            }
            if (rvdw_od > 0.0f)
            {
                h_twobody_params[pair_idx1 + p_rvdw] =
                    h_twobody_params[pair_idx2 + p_rvdw] = 2.0f * rvdw_od;
            }
            if (alfa_od > 0.0f)
            {
                h_twobody_params[pair_idx1 + p_alpha] =
                    h_twobody_params[pair_idx2 + p_alpha] = alfa_od;
            }
        }
    }
    fclose(fp_p);

    Device_Malloc_And_Copy_Safely(
        (void**)&d_twobody_params, h_twobody_params,
        sizeof(float) * n_atom_types * n_atom_types * PARAM_STRIDE);

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
        if (sscanf(line, "%15s", type_name) != 1)
        {
            char reason[512];
            sprintf(reason, "failed to parse atom type at index %d", i + 1);
            throw_bad_format(type_in_file, reason);
        }
        std::string type_str(type_name);
        auto iter = type_map.find(type_str);
        if (iter != type_map.end())
        {
            h_atom_type[i] = iter->second;
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

    Device_Malloc_And_Copy_Safely((void**)&d_atom_type, h_atom_type,
                                  sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&d_energy_sum, sizeof(float));
    Device_Malloc_Safely((void**)&d_energy_atom, sizeof(float) * atom_numbers);
    Device_Malloc_Safely(reinterpret_cast<void**>(&d_clustered_sorted_crd),
                         sizeof(VECTOR) * static_cast<size_t>(atom_numbers));
    clustered_scratch_capacity = atom_numbers;
    deviceMemset(d_energy_sum, 0, sizeof(float));
    deviceMemset(d_energy_atom, 0, sizeof(float) * atom_numbers);

    is_initialized = true;
    controller->Step_Print_Initial("REAXFF_VDW", "%14.7e");
    controller->printf("END INITIALIZING REAXFF VDW FORCE\n\n");
}

bool REAXFF_VDW::REAXFF_VDW_Force_Clustered(
    const CLUSTERED_SPATIAL_VIEW& view, const VECTOR* crd, VECTOR* frc,
    const LTMatrix3 cell, const LTMatrix3 rcell, const float cutoff,
    const int need_atom_energy, float* atom_energy, const int need_virial,
    LTMatrix3* atom_virial, const char** failure_reason)
{
    if (failure_reason != NULL) *failure_reason = NULL;
    if (!is_initialized) return true;
    auto fail = [failure_reason](const char* reason)
    {
        if (failure_reason != NULL) *failure_reason = reason;
        return false;
    };
    if (view.ghost_numbers != 0 || view.local_atom_numbers != atom_numbers ||
        view.total_atom_numbers != atom_numbers)
    {
        return fail(
            "clustered ReaxFF VDW currently requires a single-rank "
            "all-local spatial view");
    }
    if (crd == NULL || frc == NULL ||
        (need_atom_energy && atom_energy == NULL) ||
        (need_virial && atom_virial == NULL))
    {
        return fail("clustered ReaxFF VDW received null output buffers");
    }
    if (d_clustered_sorted_crd == NULL ||
        clustered_scratch_capacity < view.total_atom_numbers)
    {
        return fail("clustered ReaxFF VDW coordinate scratch is undersized");
    }

    if (need_atom_energy)
    {
        deviceMemset(d_energy_sum, 0, sizeof(float));
    }

    if (view.gmxpacked_sci_numbers <= 0) return true;
    if (view.gmxpacked_sci == NULL || view.gmxpacked_cjpacked == NULL ||
        view.gmxpacked_exclusions == NULL)
    {
        return fail("clustered ReaxFF VDW requires the gmxpacked pair payload");
    }
#ifndef USE_CPU
    if (view.pair_shift_bits == NULL)
    {
        return fail("clustered ReaxFF VDW requires pair-shift metadata");
    }
#endif
    if (view.cluster_numbers <= 0 || view.cluster_offsets == NULL ||
        view.cluster_centers == NULL || view.sort_permutation == NULL)
    {
        return fail("clustered ReaxFF VDW coordinate layout is unavailable");
    }

    Launch_Device_Kernel(REAXFF_VDW_Gather_Clustered_Coordinates,
                         (view.total_atom_numbers + 255) / 256, 256, 0, NULL,
                         view.total_atom_numbers, view.cluster_numbers,
                         view.sort_permutation, view.cluster_offsets,
                         view.cluster_centers, crd, cell, rcell,
                         d_clustered_sorted_crd);

#ifdef USE_CPU
    if (need_atom_energy || need_virial)
    {
        REAXFF_VDW_Clustered_Gmxpacked_CPU<true>(
            view, view.sort_permutation, d_clustered_sorted_crd, d_atom_type,
            d_twobody_params, atom_type_numbers, cutoff, p_vdw1, cell, frc,
            atom_energy, atom_virial, d_energy_sum, need_atom_energy != 0,
            need_virial != 0);
    }
    else
    {
        REAXFF_VDW_Clustered_Gmxpacked_CPU<false>(
            view, view.sort_permutation, d_clustered_sorted_crd, d_atom_type,
            d_twobody_params, atom_type_numbers, cutoff, p_vdw1, cell, frc,
            atom_energy, atom_virial, d_energy_sum, false, false);
    }
#else
    constexpr int packed_partitions = 8;
    const dim3 pair_block(static_cast<unsigned int>(kClusteredClusterSize),
                          static_cast<unsigned int>(kClusteredClusterSize), 1u);
    const dim3 pair_grid(static_cast<unsigned int>(view.gmxpacked_sci_numbers),
                         static_cast<unsigned int>(packed_partitions), 1u);
    auto force_kernel = REAXFF_VDW_Clustered_Gmxpacked<false>;
    if (need_atom_energy || need_virial)
    {
        force_kernel = REAXFF_VDW_Clustered_Gmxpacked<true>;
    }
    Launch_Device_Kernel(
        force_kernel, pair_grid, pair_block, 0, NULL,
        view.gmxpacked_sci_numbers, packed_partitions, view.cluster_numbers,
        view.cluster_offsets, view.cluster_valid_masks,
        view.cluster_local_masks, view.super_cluster_offsets,
        view.gmxpacked_sci, view.gmxpacked_cjpacked, view.gmxpacked_exclusions,
        view.pair_shift_bits, view.sort_permutation, d_clustered_sorted_crd,
        d_atom_type, d_twobody_params, atom_type_numbers, cutoff, p_vdw1, cell,
        frc, atom_energy, atom_virial, d_energy_sum, need_atom_energy != 0,
        need_virial != 0);
#endif
    return true;
}

void REAXFF_VDW::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    deviceMemcpy(&h_energy_sum, d_energy_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
    controller->Step_Print("REAXFF_VDW", h_energy_sum, true);
}
