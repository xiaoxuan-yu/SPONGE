#include "edip.h"

void EDIP_INFORMATION::Initial(CONTROLLER* controller,
                               const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "EDIP");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    if (!controller->Command_Exist(this->module_name, "in_file"))
    {
        controller->printf("EDIP FORCE IS NOT INITIALIZED\n\n");
        return;
    }
    controller->printf("START INITIALIZING EDIP FORCE\n");
    FILE* fp;
    Open_File_Safely(&fp, controller->Command(this->module_name, "in_file"),
                     "r");
    if (fscanf(fp, "%d %d\n", &atom_numbers, &atom_type_numbers) != 2)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
            "Reason:\n\tThe number of atoms and edip types can not be found\n");
    }
    pair_type_numbers = atom_type_numbers * atom_type_numbers;
    triple_type_numbers =
        atom_type_numbers * atom_type_numbers * atom_type_numbers;

    Malloc_Safely((void**)&h_energy_atom, sizeof(float) * (atom_numbers + 1));

    Device_Malloc_And_Copy_Safely((void**)&d_energy_sum, h_energy_atom,
                                  sizeof(float) * (atom_numbers + 1));
    d_energy_atom = d_energy_sum + 1;
    Device_Malloc_Safely((void**)&z, sizeof(float) * atom_numbers * 2);
    dE_dz = z + atom_numbers;
    Malloc_Safely(
        (void**)&h_parameters,
        sizeof(float) * (pair_type_numbers * 8 + triple_type_numbers * 9));
    Malloc_Safely((void**)&h_atom_type, sizeof(int) * atom_numbers);
    char temp[CHAR_LENGTH_MAX];
    std::map<int, bool> unrecorded;
    int type_a, type_b, type_c;
    float a, c, alpha, A, B, rho, beta, sigma, eta, gamma, lambda, Q0, mu, u1,
        u2, u3, u4;
    if (fgets(temp, CHAR_LENGTH_MAX, fp) == NULL || strlen(temp) < 1 ||
        temp[0] != '#')
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
            "Reason:\n\tThe first comment line can not be found\n");
    }
    for (int i = 0; i < pair_type_numbers; i++)
    {
        unrecorded[i] = true;
    }
    for (int i = 0; i < pair_type_numbers; i++)
    {
        if (fscanf(fp, "%d %d %f %f %f %f %f %f %f %f\n", &type_a, &type_b,
                   &alpha, &c, &a, &A, &B, &rho, &beta, &sigma) != 10)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
                "Reason:\n\tSome twobody parameters can not be found\n");
        }
        int index = type_a * atom_type_numbers + type_b;
        if (index >= pair_type_numbers)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
                "Reason:\n\tSome twobody type indexes are not right\n");
        }
        unrecorded[index] = false;
        h_parameters[8 * index + 0] = alpha;
        h_parameters[8 * index + 1] = c;
        h_parameters[8 * index + 2] = a;
        h_parameters[8 * index + 3] = A;
        h_parameters[8 * index + 4] = B;
        h_parameters[8 * index + 5] = rho;
        h_parameters[8 * index + 6] = beta;
        h_parameters[8 * index + 7] = sigma;
        cut = fmaxf(cut, a);
    }
    for (int i = 0; i < pair_type_numbers; i++)
    {
        if (unrecorded[i])
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
                "Reason:\n\tSome twobody parameters can not be found\n");
        }
    }
    if (fgets(temp, CHAR_LENGTH_MAX, fp) == NULL || strlen(temp) < 1 ||
        temp[0] != '#')
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
            "Reason:\n\tThe second comment line can not be found\n");
    }
    for (int i = 0; i < triple_type_numbers; i++)
    {
        unrecorded[i] = true;
    }
    for (int i = 0; i < triple_type_numbers; i++)
    {
        if (fscanf(fp, "%d %d %d %f %f %f %f %f %f %f %f %f\n", &type_a,
                   &type_b, &type_c, &eta, &gamma, &lambda, &Q0, &mu, &u1, &u2,
                   &u3, &u4) != 12)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
                "Reason:\n\tSome threebody parameters can not be found\n");
        }
        int index = type_a * atom_type_numbers * atom_type_numbers +
                    type_b * atom_type_numbers + type_c;
        if (index >= triple_type_numbers)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
                "Reason:\n\tSome threebody type indexes are not right\n");
        }
        unrecorded[index] = false;
        h_parameters[8 * pair_type_numbers + 9 * index + 0] = eta;
        h_parameters[8 * pair_type_numbers + 9 * index + 1] = gamma;
        h_parameters[8 * pair_type_numbers + 9 * index + 2] = lambda;
        h_parameters[8 * pair_type_numbers + 9 * index + 3] = Q0;
        h_parameters[8 * pair_type_numbers + 9 * index + 4] = mu;
        h_parameters[8 * pair_type_numbers + 9 * index + 5] = u1;
        h_parameters[8 * pair_type_numbers + 9 * index + 6] = u2;
        h_parameters[8 * pair_type_numbers + 9 * index + 7] = u3;
        h_parameters[8 * pair_type_numbers + 9 * index + 8] = u4;
    }
    for (int i = 0; i < triple_type_numbers; i++)
    {
        if (unrecorded[i])
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
                "Reason:\n\tSome threebody parameters can not be found\n");
        }
    }
    Device_Malloc_And_Copy_Safely(
        (void**)&d_parameters, h_parameters,
        sizeof(float) * (pair_type_numbers * 8 + triple_type_numbers * 9));
    if (fgets(temp, CHAR_LENGTH_MAX, fp) == NULL || strlen(temp) < 1 ||
        temp[0] != '#')
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
            "Reason:\n\tThe third comment line can not be found\n");
    }
    for (int i = 0; i < atom_numbers; i++)
    {
        if (fscanf(fp, "%d", h_atom_type + i) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
                "Reason:\n\tSome atom types can not be found\n");
        }
        if (h_atom_type[i] >= atom_type_numbers)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "EDIP_INFORMATION::Initial",
                "Reason:\n\tSome atom types are not right\n");
        }
    }
    Device_Malloc_And_Copy_Safely((void**)&d_atom_type, h_atom_type,
                                  sizeof(int) * atom_numbers);

    is_initialized = true;
    if (!is_controller_printf_initialized)
    {
        controller->Step_Print_Initial(this->module_name, "%.2f");
        is_controller_printf_initialized = true;
    }
    controller->printf("END INITIALIZING EDIP FORCE\n\n");
}

static __host__ __device__ __forceinline__ bool
EDIP_Clustered_Neighbor_Within_Cut(
    const int type_i, const int type_j,
    const float distance_squared, const float* parameters,
    const int atom_type_numbers, const float margin = 0.0f)
{
    const int pair_index = type_i * atom_type_numbers + type_j;
    const float cutoff = parameters[8 * pair_index + 2] + margin;
    return distance_squared < cutoff * cutoff;
}

template <typename T>
static void EDIP_Reserve_Clustered_Neighbor_Buffer(
    T** pointer, int* capacity, const int required)
{
    if (required <= *capacity && *pointer != NULL)
    {
        return;
    }
    Free_Single_Device_Pointer(reinterpret_cast<void**>(pointer));
    *capacity = 0;
    if (required > 0)
    {
        Device_Malloc_Safely(
            reinterpret_cast<void**>(pointer),
            sizeof(T) * static_cast<size_t>(required));
        *capacity = required;
    }
}

#ifdef USE_GPU
template <bool fill>
static __global__ void EDIP_Build_Clustered_Center_Atoms(
    const CLUSTERED_SPATIAL_VIEW view, const VECTOR* crd,
    const LTMatrix3 cell, const int* atom_types,
    const float* parameters, const int atom_type_numbers,
    const int packed_partitions, const int* neighbor_offsets,
    int* neighbor_atoms, int* neighbor_counts)
{
    const int sci = static_cast<int>(blockIdx.x);
    const int packed_partition = static_cast<int>(blockIdx.y);
    const int i_lane = static_cast<int>(threadIdx.x);
    const int j_lane = static_cast<int>(threadIdx.y);
    if (sci >= view.gmxpacked_sci_numbers ||
        i_lane >= kClusteredClusterSize ||
        j_lane >= kClusteredClusterSize)
    {
        return;
    }
    const CLUSTERED_GMXPACKED_SCI sci_entry =
        view.gmxpacked_sci[sci];
    const int cluster_i_begin =
        view.super_cluster_offsets[sci_entry.supercluster_id];
    int cluster_i_end =
        view.super_cluster_offsets[sci_entry.supercluster_id + 1];
    if (cluster_i_end > view.cluster_numbers)
    {
        cluster_i_end = view.cluster_numbers;
    }
    const int split = j_lane / kClusteredSplitJClusterSize;
    const int split_j_lane =
        j_lane - split * kClusteredSplitJClusterSize;
    const unsigned int i_lane_bit = 1u << i_lane;
    const unsigned int j_lane_bit = 1u << j_lane;

    for (int packed_idx =
             sci_entry.cjpacked_begin + packed_partition;
         packed_idx < sci_entry.cjpacked_end;
         packed_idx += packed_partitions)
    {
        const CLUSTERED_GMXPACKED_CJ packed =
            view.gmxpacked_cjpacked[packed_idx];
        const CLUSTERED_GMXPACKED_SPLIT split_entry =
            packed.split[split];
        unsigned int pair_bits = 0xffffffffu;
        if (split_entry.exclusion_index != 0)
        {
            pair_bits =
                view.gmxpacked_exclusions[split_entry.exclusion_index]
                    .pair[split_j_lane * kClusteredClusterSize +
                          i_lane];
        }
        const unsigned int effective_mask =
            split_entry.imask & pair_bits;
        for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
        {
            const int cluster_j = packed.cj[jm];
            if (cluster_j < 0 ||
                (view.cluster_valid_masks[cluster_j] &
                 j_lane_bit) == 0u ||
                (view.cluster_local_masks[cluster_j] &
                 j_lane_bit) == 0u)
            {
                continue;
            }
            const int sorted_j =
                view.cluster_offsets[cluster_j] + j_lane;
            const int atom_j =
                view.sort_permutation[sorted_j];
            const int type_j = atom_types[atom_j];
            const uint64_t shift_bits =
                view.pair_shift_bits[
                    packed_idx * kClusteredJGroupSize + jm];
            for (int i_local = 0;
                 i_local < cluster_i_end - cluster_i_begin;
                 i_local += 1)
            {
                const unsigned int packed_bit =
                    1u << (jm * kClusteredSuperClusterClusters +
                           i_local);
                if ((effective_mask & packed_bit) == 0u ||
                    (Clustered_Get_Pair_Active_I_Mask(
                         shift_bits, split) &
                     (1u << static_cast<unsigned int>(i_local))) ==
                        0u)
                {
                    continue;
                }
                const int cluster_i = cluster_i_begin + i_local;
                if ((view.cluster_valid_masks[cluster_i] &
                     i_lane_bit) == 0u ||
                    (view.cluster_local_masks[cluster_i] &
                     i_lane_bit) == 0u)
                {
                    continue;
                }
                const int sorted_i =
                    view.cluster_offsets[cluster_i] + i_lane;
                const int atom_i =
                    view.sort_permutation[sorted_i];
                if (atom_i == atom_j) continue;
                const int type_i = atom_types[atom_i];
                const int shift_id =
                    Clustered_Get_Pair_Shift_Id(
                        shift_bits, i_local);
                const VECTOR shift =
                    Clustered_Shift_Vector_From_Id(shift_id, cell);
                const VECTOR displacement =
                    (crd[atom_i] - crd[atom_j]) + shift;
                const float distance_squared =
                    displacement * displacement;
                if (EDIP_Clustered_Neighbor_Within_Cut(
                        type_i, type_j, distance_squared, parameters,
                        atom_type_numbers,
                        view.rebuild_skin))
                {
                    const int slot =
                        atomicAdd(neighbor_counts + atom_i, 1);
                    if constexpr (fill)
                    {
                        neighbor_atoms[
                            neighbor_offsets[atom_i] + slot] =
                            atom_j;
                    }
                }
                if (EDIP_Clustered_Neighbor_Within_Cut(
                        type_j, type_i, distance_squared, parameters,
                        atom_type_numbers,
                        view.rebuild_skin))
                {
                    const int slot =
                        atomicAdd(neighbor_counts + atom_j, 1);
                    if constexpr (fill)
                    {
                        neighbor_atoms[
                            neighbor_offsets[atom_j] + slot] =
                            atom_i;
                    }
                }
            }
        }
    }
}
#else
static bool EDIP_Build_Native_Center_Atoms_CPU(
    EDIP_INFORMATION* edip, const CLUSTERED_SPATIAL_VIEW& view,
    const VECTOR* crd, const LTMatrix3 cell,
    std::vector<std::vector<int>>* center_atoms)
{
    if (center_atoms == NULL) return false;
    center_atoms->assign(
        static_cast<size_t>(edip->atom_numbers), {});
    for (int sci = 0; sci < view.sci_numbers; sci += 1)
    {
        const CLUSTERED_SCI sci_entry = view.sci[sci];
        const int cluster_i_begin =
            view.super_cluster_offsets[sci_entry.supercluster_id];
        const int cluster_i_end =
            view.super_cluster_offsets[sci_entry.supercluster_id + 1];
        const VECTOR shift =
            Clustered_Shift_Vector_From_Id(
                sci_entry.shift_id, cell);
        for (int cluster_i = cluster_i_begin;
             cluster_i < cluster_i_end; cluster_i += 1)
        {
            const int i_local = cluster_i - cluster_i_begin;
            for (int packed_idx = sci_entry.cjpacked_begin;
                 packed_idx < sci_entry.cjpacked_end;
                 packed_idx += 1)
            {
                const CLUSTERED_CJ_PACKED packed =
                    view.cjpacked[packed_idx];
                for (int jm = 0; jm < kClusteredJGroupSize;
                     jm += 1)
                {
                    const int cluster_j = packed.cj[jm];
                    if (cluster_j < 0) continue;
                    const unsigned int imask =
                        Clustered_Jm_Imask(packed.imei[0], jm) |
                        Clustered_Jm_Imask(packed.imei[1], jm);
                    if ((imask & (1u << i_local)) == 0u) continue;
                    const int exclusion_index =
                        Clustered_First_Exclusion_Index(
                            packed, jm, i_local);
                    const uint64_t exclusion_mask =
                        exclusion_index >= 0 &&
                                view.exclusion_mask_pool != NULL
                            ? view.exclusion_mask_pool[exclusion_index]
                            : 0ull;
                    for (int i_lane = 0;
                         i_lane < view.cluster_size; i_lane += 1)
                    {
                        const unsigned int i_lane_bit =
                            1u << i_lane;
                        if ((view.cluster_valid_masks[cluster_i] &
                             i_lane_bit) == 0u ||
                            (view.cluster_local_masks[cluster_i] &
                             i_lane_bit) == 0u)
                        {
                            continue;
                        }
                        const int sorted_i =
                            view.cluster_offsets[cluster_i] + i_lane;
                        const int atom_i =
                            view.sort_permutation[sorted_i];
                        const int type_i =
                            edip->d_atom_type[atom_i];
                        for (int j_lane = 0;
                             j_lane < view.cluster_size;
                             j_lane += 1)
                        {
                            const unsigned int j_lane_bit =
                                1u << j_lane;
                            if ((view.cluster_valid_masks[cluster_j] &
                                 j_lane_bit) == 0u ||
                                (view.cluster_local_masks[cluster_j] &
                                 j_lane_bit) == 0u ||
                                (exclusion_mask &
                                 (1ull << (i_lane *
                                              view.cluster_size +
                                          j_lane))) != 0ull ||
                                (sci_entry.shift_id ==
                                     kClusteredCentralShiftId &&
                                 cluster_i == cluster_j &&
                                 j_lane <= i_lane))
                            {
                                continue;
                            }
                            const int sorted_j =
                                view.cluster_offsets[cluster_j] +
                                j_lane;
                            const int atom_j =
                                view.sort_permutation[sorted_j];
                            if (atom_i == atom_j) continue;
                            const int type_j =
                                edip->d_atom_type[atom_j];
                            const VECTOR displacement =
                                (crd[atom_i] - crd[atom_j]) +
                                shift;
                            const float distance_squared =
                                displacement * displacement;
                            if (EDIP_Clustered_Neighbor_Within_Cut(
                                    type_i, type_j, distance_squared,
                                    edip->d_parameters,
                                    edip->atom_type_numbers,
                                    view.rebuild_skin))
                            {
                                (*center_atoms)[atom_i].push_back(
                                    atom_j);
                            }
                            if (EDIP_Clustered_Neighbor_Within_Cut(
                                    type_j, type_i, distance_squared,
                                    edip->d_parameters,
                                    edip->atom_type_numbers,
                                    view.rebuild_skin))
                            {
                                (*center_atoms)[atom_j].push_back(
                                    atom_i);
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}
#endif

static bool EDIP_Ensure_Clustered_Center_Atoms(
    EDIP_INFORMATION* edip, const CLUSTERED_SPATIAL_VIEW& view,
    const VECTOR* crd, const LTMatrix3 cell)
{
    const long long payload_generation =
#ifdef USE_CPU
        view.native_payload_generation;
#else
        view.gmxpacked_payload_generation;
#endif
    if (edip->clustered_neighbor_provider_incarnation ==
            view.provider_incarnation &&
        edip->clustered_neighbor_payload_generation ==
            payload_generation)
    {
        return true;
    }
    EDIP_Reserve_Clustered_Neighbor_Buffer(
        &edip->d_clustered_neighbor_counts,
        &edip->clustered_neighbor_counts_capacity,
        edip->atom_numbers);
    EDIP_Reserve_Clustered_Neighbor_Buffer(
        &edip->d_clustered_neighbor_offsets,
        &edip->clustered_neighbor_offsets_capacity,
        edip->atom_numbers + 1);

#ifdef USE_CPU
    std::vector<std::vector<int>> center_atoms;
    if (!EDIP_Build_Native_Center_Atoms_CPU(
            edip, view, crd, cell, &center_atoms))
    {
        return false;
    }
    std::vector<int> host_counts(
        static_cast<size_t>(edip->atom_numbers), 0);
    for (int atom_i = 0; atom_i < edip->atom_numbers; atom_i += 1)
    {
        host_counts[atom_i] =
            static_cast<int>(center_atoms[atom_i].size());
    }
#else
    deviceMemset(
        edip->d_clustered_neighbor_counts, 0,
        sizeof(int) * static_cast<size_t>(edip->atom_numbers));
    constexpr int packed_partitions = 16;
    const dim3 pair_block(
        static_cast<unsigned int>(kClusteredClusterSize),
        static_cast<unsigned int>(kClusteredClusterSize), 1u);
    const dim3 pair_grid(
        static_cast<unsigned int>(view.gmxpacked_sci_numbers),
        static_cast<unsigned int>(packed_partitions), 1u);
    Launch_Device_Kernel(
        EDIP_Build_Clustered_Center_Atoms<false>,
        pair_grid, pair_block, 0, NULL, view, crd, cell,
        edip->d_atom_type, edip->d_parameters,
        edip->atom_type_numbers, packed_partitions, NULL, NULL,
        edip->d_clustered_neighbor_counts);
    std::vector<int> host_counts(
        static_cast<size_t>(edip->atom_numbers), 0);
    deviceMemcpy(
        host_counts.data(), edip->d_clustered_neighbor_counts,
        sizeof(int) * static_cast<size_t>(edip->atom_numbers),
        deviceMemcpyDeviceToHost);
#endif

    std::vector<int> host_offsets(
        static_cast<size_t>(edip->atom_numbers + 1), 0);
    long long total = 0;
    for (int atom_i = 0; atom_i < edip->atom_numbers; atom_i += 1)
    {
        if (host_counts[atom_i] < 0) return false;
        host_offsets[atom_i] = static_cast<int>(total);
        total += host_counts[atom_i];
        if (total > INT_MAX) return false;
    }
    host_offsets[edip->atom_numbers] = static_cast<int>(total);
    edip->clustered_neighbor_numbers = static_cast<int>(total);
    deviceMemcpy(
        edip->d_clustered_neighbor_offsets, host_offsets.data(),
        sizeof(int) *
            static_cast<size_t>(edip->atom_numbers + 1),
        deviceMemcpyHostToDevice);
    EDIP_Reserve_Clustered_Neighbor_Buffer(
        &edip->d_clustered_neighbor_atoms,
        &edip->clustered_neighbor_atoms_capacity,
        edip->clustered_neighbor_numbers);

#ifdef USE_CPU
    for (int atom_i = 0; atom_i < edip->atom_numbers; atom_i += 1)
    {
        const std::vector<int>& row = center_atoms[atom_i];
        if (!row.empty())
        {
            deviceMemcpy(
                edip->d_clustered_neighbor_atoms +
                    host_offsets[atom_i],
                row.data(), sizeof(int) * row.size(),
                deviceMemcpyHostToDevice);
        }
    }
#else
    if (edip->clustered_neighbor_numbers > 0)
    {
        deviceMemset(
            edip->d_clustered_neighbor_counts, 0,
            sizeof(int) *
                static_cast<size_t>(edip->atom_numbers));
        Launch_Device_Kernel(
            EDIP_Build_Clustered_Center_Atoms<true>,
            pair_grid, pair_block, 0, NULL, view, crd, cell,
            edip->d_atom_type, edip->d_parameters,
            edip->atom_type_numbers, packed_partitions,
            edip->d_clustered_neighbor_offsets,
            edip->d_clustered_neighbor_atoms,
            edip->d_clustered_neighbor_counts);
    }
#endif
    edip->clustered_neighbor_provider_incarnation =
        view.provider_incarnation;
    edip->clustered_neighbor_payload_generation =
        payload_generation;
    return true;
}

template <bool full_output>
static __global__
    __launch_bounds__(1024) void EDIP_Force_With_Full_Neighbor_CUDA(
        const int atom_numbers, const VECTOR* crd, VECTOR* frc,
        const LTMatrix3 cell, const LTMatrix3 rcell, float* z, float* dE_dz,
        const int* neighbor_offsets, const int* neighbor_atoms,
        float* atom_energy, LTMatrix3* atom_virial, int* atom_types,
        float* parameters, const int atom_type_numbers,
        const int pair_type_numbers, float* this_energy,
        const bool store_energy, const bool store_virial)
{
#ifdef USE_GPU
    int atom_i = threadIdx.y + blockDim.y * blockIdx.x, atom_j, atom_k;
    if (atom_i < atom_numbers)
#else
    int atom_j, atom_k;
#pragma omp parallel for firstprivate(atom_j, atom_k)
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        int type_i = atom_types[atom_i], type_j, type_k;
        const int neighbor_begin = neighbor_offsets[atom_i];
        const int neighbor_end = neighbor_offsets[atom_i + 1];
        VECTOR ri = crd[atom_i], rj, rk, drij, drik, drjk;
        float rij, rik, rjk;
        float A, B, sigma, a1, a2, a3, rho, beta, eta, gamma, lambda, Q0, mu,
            u1, u2, u3, u4;
        int pair_index_1, pair_index_2, triple_index;
        float local_energy = 0, zi = z[atom_i], zj, zk, dE_dzi = 0, dE_dzj,
              dE_dzk;
        VECTOR i_force = {0.0f, 0.0f, 0.0f}, j_force, k_force;
        VECTOR temp1_force, temp2_force;
        LTMatrix3 local_virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#ifdef USE_GPU
        for (int j = neighbor_begin + threadIdx.x;
             j < neighbor_end; j += blockDim.x)
#else
        for (int j = neighbor_begin; j < neighbor_end; j++)
#endif
        {
            j_force = {0.0f, 0.0f, 0.0f};
            dE_dzj = 0;
            atom_j = neighbor_atoms[j];
            zj = z[atom_j];
            type_j = atom_types[atom_j];
            rj = crd[atom_j];
            drij = Get_Periodic_Displacement(ri, rj, cell, rcell);
            rij = norm3df(drij.x, drij.y, drij.z);
            pair_index_1 = type_i * atom_type_numbers + type_j;
            a1 = parameters[8 * pair_index_1 + 2];
            A = parameters[8 * pair_index_1 + 3];
            B = parameters[8 * pair_index_1 + 4];
            rho = parameters[8 * pair_index_1 + 5];
            beta = parameters[8 * pair_index_1 + 6];
            sigma = parameters[8 * pair_index_1 + 7];

            bool should_compute_twobody = atom_j > atom_i;

            if (rij < a1 && should_compute_twobody)
            {
                SADfloat<3> twobody1(rij, 0);
                SADfloat<3> twobody2(zi, 1);
                SADfloat<3> twobody3(zj, 2);
                twobody2 = A * (2 * powf(B / twobody1, rho) -
                                expf(-beta * twobody2 * twobody2) -
                                expf(-beta * twobody3 * twobody3));
                twobody3 = twobody2 * expf(sigma / (twobody1 - a1));
                local_energy += twobody3.val;
                temp1_force = twobody3.dval[0] / rij * drij;
                i_force = i_force - temp1_force;
                j_force = j_force + temp1_force;
                dE_dzi += twobody3.dval[1];
                dE_dzj += twobody3.dval[2];
                if constexpr (full_output)
                {
                    if (store_virial)
                        local_virial =
                            local_virial -
                            Get_Virial_From_Force_Dis(
                                temp1_force, drij);
                }
            }

            for (int k = j + 1; k < neighbor_end; k += 1)
            {
                k_force = {0.0f, 0.0f, 0.0f};
                atom_k = neighbor_atoms[k];
                type_k = atom_types[atom_k];
                rk = crd[atom_k];
                drik = Get_Periodic_Displacement(ri, rk, cell, rcell);
                rik = norm3df(drik.x, drik.y, drik.z);
                pair_index_2 = type_i * atom_type_numbers + type_k;
                a2 = parameters[8 * pair_index_2 + 2];
                if (rik < a2 && rij < a1)
                {
                    triple_index = pair_index_1 * atom_type_numbers + type_k;
                    eta = parameters[8 * pair_type_numbers + 9 * triple_index];
                    gamma = parameters[8 * pair_type_numbers +
                                       9 * triple_index + 1];
                    lambda = parameters[8 * pair_type_numbers +
                                        9 * triple_index + 2];
                    Q0 = parameters[8 * pair_type_numbers + 9 * triple_index +
                                    3];
                    mu = parameters[8 * pair_type_numbers + 9 * triple_index +
                                    4];
                    u1 = parameters[8 * pair_type_numbers + 9 * triple_index +
                                    5];
                    u2 = parameters[8 * pair_type_numbers + 9 * triple_index +
                                    6];
                    u3 = parameters[8 * pair_type_numbers + 9 * triple_index +
                                    7];
                    u4 = parameters[8 * pair_type_numbers + 9 * triple_index +
                                    8];
                    SADvector<7> threebody1(drij, 0, 1, 2);
                    SADvector<7> threebody2(drik, 3, 4, 5);
                    SADfloat<7> threebody3(zi, 6);
                    SADfloat<7> r1 = sqrtf(threebody1 * threebody1);
                    SADfloat<7> r2 = sqrtf(threebody2 * threebody2);
                    SADfloat<7> E = threebody1 * threebody2 / r1 / r2;
                    E = u1 +
                        u2 * (u3 * expf(-u4 * threebody3) -
                              expf(-2.0f * u4 * threebody3)) +
                        E;
                    E = E * E;
                    SADfloat<7> Q = Q0 * expf(-mu * threebody3);
                    E = lambda * (1 - expf(-Q * E) + eta * Q * E);
                    E = E * expf(gamma / (r1 - a1) + gamma / (r2 - a2));
                    temp1_force = {E.dval[0], E.dval[1], E.dval[2]};
                    temp2_force = {E.dval[3], E.dval[4], E.dval[5]};
                    k_force = k_force + temp2_force;
                    j_force = j_force + temp1_force;
                    i_force = i_force - temp1_force - temp2_force;
                    dE_dzi += E.dval[6];
                    local_energy += E.val;
                    if constexpr (full_output)
                    {
                        if (store_virial)
                        {
                            local_virial =
                                local_virial -
                                Get_Virial_From_Force_Dis(
                                    temp1_force, drij) -
                                Get_Virial_From_Force_Dis(
                                    temp2_force, drik);
                        }
                    }
                }
                atomicAdd(frc + atom_k, k_force);
            }
            atomicAdd(frc + atom_j, j_force);
            atomicAdd(dE_dz + atom_j, dE_dzj);
        }
        Warp_Sum_To(frc + atom_i, i_force, warpSize);
        Warp_Sum_To(dE_dz + atom_i, dE_dzi, warpSize);
        if constexpr (full_output)
        {
            if (store_energy)
            {
                atomicAdd(atom_energy + atom_i, local_energy);
                atomicAdd(this_energy + atom_i, local_energy);
            }
            if (store_virial)
            {
                Warp_Sum_To(
                    atom_virial + atom_i, local_virial, warpSize);
            }
        }
    }
}

static __global__ void Get_Z(const int atom_numbers, const VECTOR* crd,
                             const LTMatrix3 cell, const LTMatrix3 rcell,
                             const int* neighbor_offsets,
                             const int* neighbor_atoms,
                             const float* parameters,
                             const int atom_type_numbers,
                             const int* atom_types, float* z)
{
#ifdef USE_GPU
    int atom_i = threadIdx.y + blockDim.y * blockIdx.x, atom_j;
    if (atom_i < atom_numbers)
#else
    int atom_j;
#pragma omp parallel for firstprivate(atom_j)
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        int type_i = atom_types[atom_i], type_j;
        const int neighbor_begin = neighbor_offsets[atom_i];
        const int neighbor_end = neighbor_offsets[atom_i + 1];
        float local_z = 0, a, c, alpha, r;
        VECTOR ri = crd[atom_i], dr;
#ifdef USE_GPU
        for (int j = neighbor_begin + threadIdx.x;
             j < neighbor_end; j += blockDim.x)
#else
        for (int j = neighbor_begin; j < neighbor_end; j++)
#endif
        {
            atom_j = neighbor_atoms[j];
            type_j = atom_types[atom_j];
            type_j += atom_type_numbers * type_i;
            alpha = parameters[8 * type_j];
            c = parameters[8 * type_j + 1];
            a = parameters[8 * type_j + 2];
            dr = Get_Periodic_Displacement(crd[atom_j], ri, cell, rcell);
            r = norm3df(dr.x, dr.y, dr.z);
            if (r < c)
            {
                local_z += 1.0f;
            }
            else if (r < a)
            {
                r = (r - c) / (a - c);
                r = expf(alpha / (1.0f - powf(r, -3.0f)));
                local_z += r;
            }
        }
        Warp_Sum_To(z + atom_i, local_z, warpSize);
    }
}

template <bool full_output>
static __global__ __launch_bounds__(1024) void Redistribute_Z_to_Atoms(
    const int atom_numbers, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const int* neighbor_offsets,
    const int* neighbor_atoms, const float* parameters,
    const int atom_type_numbers, const int* atom_types,
    const float* dE_dz, VECTOR* frc, LTMatrix3* atom_virial,
    const bool store_virial)
{
#ifdef USE_GPU
    int atom_i = threadIdx.y + blockDim.y * blockIdx.x, atom_j;
    if (atom_i < atom_numbers)
#else
    int atom_j;
#pragma omp parallel for firstprivate(atom_j)
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        int type_i = atom_types[atom_i], type_j;
        const int neighbor_begin = neighbor_offsets[atom_i];
        const int neighbor_end = neighbor_offsets[atom_i + 1];
        float dE_dzi = dE_dz[atom_i], dE_dzj;
        float a, c, alpha;
        SADfloat<1> r(0, 0), z;
        VECTOR ri = crd[atom_i], dr;
        VECTOR local_frc = {0.0f, 0.0f, 0.0f}, f;
        LTMatrix3 local_virial(0.0f);
#ifdef USE_GPU
        for (int j = neighbor_begin + threadIdx.x;
             j < neighbor_end; j += blockDim.x)
#else
        for (int j = neighbor_begin; j < neighbor_end; j++)
#endif
        {
            atom_j = neighbor_atoms[j];
            type_j = atom_types[atom_j];
            dE_dzj = dE_dz[atom_j] + dE_dzi;

            type_j += atom_type_numbers * type_i;
            alpha = parameters[8 * type_j];
            c = parameters[8 * type_j + 1];
            a = parameters[8 * type_j + 2];
            dr = Get_Periodic_Displacement(ri, crd[atom_j], cell, rcell);
            r.val = norm3df(dr.x, dr.y, dr.z);
            if (r < a && r > c)
            {
                z = (r - c) / (a - c);
                z = expf(alpha / (1.0f - powf(z, -3.0f)));
                f = dE_dzj * z.dval[0] / r.val * dr;
                local_frc = local_frc - f;
                if constexpr (full_output)
                {
                    if (store_virial && atom_j > atom_i)
                        local_virial =
                            local_virial -
                            Get_Virial_From_Force_Dis(f, dr);
                }
            }
        }
        Warp_Sum_To(frc + atom_i, local_frc, warpSize);
        if constexpr (full_output)
        {
            if (store_virial)
                Warp_Sum_To(
                    atom_virial + atom_i, local_virial, warpSize);
        }
    }
}

bool EDIP_INFORMATION::EDIP_Force_Clustered(
    const CLUSTERED_SPATIAL_VIEW& view, const VECTOR* crd, VECTOR* frc,
    const LTMatrix3 cell, const LTMatrix3 rcell,
    const int need_atom_energy, float* atom_energy,
    const int need_virial, LTMatrix3* atom_virial,
    const char** failure_reason)
{
    if (failure_reason != NULL) *failure_reason = NULL;
    if (!is_initialized) return true;
    CLUSTERED_SPATIAL_VIEW_REQUIREMENTS requirements;
    requirements.local_atom_numbers = atom_numbers;
    requirements.ghost_numbers = 0;
    requirements.cutoff = cut;
    requirements.provider_incarnation = view.provider_incarnation;
    requirements.lease_epoch = view.lease_epoch;
    requirements.require_all_local_atoms = true;
#ifdef USE_CPU
    requirements.native_payload_generation =
        view.native_payload_generation;
    requirements.require_backend = true;
    requirements.backend = CLUSTERED_SPATIAL_BACKEND::CPU;
    requirements.require_native_payload = true;
#else
    requirements.gmxpacked_payload_generation =
        view.gmxpacked_payload_generation;
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
            view, requirements, failure_reason))
    {
        return false;
    }
    if (crd == NULL || frc == NULL ||
        (need_atom_energy && atom_energy == NULL) ||
        (need_virial && atom_virial == NULL))
    {
        if (failure_reason != NULL)
            *failure_reason =
                "EDIP clustered force received null buffers";
        return false;
    }
    if (need_atom_energy)
        deviceMemset(
            d_energy_sum, 0,
            sizeof(float) * (this->atom_numbers + 1));
    if (!EDIP_Ensure_Clustered_Center_Atoms(
            this, view, crd, cell))
    {
        if (failure_reason != NULL)
            *failure_reason =
                "EDIP could not derive compact center-neighbor relation";
        return false;
    }

    dim3 blockSize = {CONTROLLER::device_warp,
                      CONTROLLER::device_max_thread / CONTROLLER::device_warp};
    dim3 gridSize = (atom_numbers + blockSize.y - 1) / blockSize.y;
    auto f1 = EDIP_Force_With_Full_Neighbor_CUDA<false>;
    auto f2 = Redistribute_Z_to_Atoms<false>;
    if (need_atom_energy || need_virial)
    {
        f1 = EDIP_Force_With_Full_Neighbor_CUDA<true>;
        f2 = Redistribute_Z_to_Atoms<true>;
    }

    deviceMemset(this->z, 0, sizeof(float) * atom_numbers * 2);
    Launch_Device_Kernel(
        Get_Z, gridSize, blockSize, 0, NULL, atom_numbers, crd,
        cell, rcell, d_clustered_neighbor_offsets,
        d_clustered_neighbor_atoms, this->d_parameters,
        this->atom_type_numbers, this->d_atom_type, this->z);
    Launch_Device_Kernel(f1, gridSize, blockSize, 0, NULL, atom_numbers, crd,
                         frc, cell, rcell, z, dE_dz,
                         d_clustered_neighbor_offsets,
                         d_clustered_neighbor_atoms, atom_energy,
                         atom_virial, this->d_atom_type, this->d_parameters,
                         this->atom_type_numbers, this->pair_type_numbers,
                         this->d_energy_atom,
                         need_atom_energy != 0, need_virial != 0);
    Launch_Device_Kernel(
        f2, gridSize, blockSize, 0, NULL, atom_numbers, crd,
        cell, rcell, d_clustered_neighbor_offsets,
        d_clustered_neighbor_atoms, this->d_parameters,
        this->atom_type_numbers, this->d_atom_type,
        this->dE_dz, frc, atom_virial, need_virial != 0);
    return true;
}

void EDIP_INFORMATION::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    Sum_Of_List(d_energy_atom, d_energy_sum, atom_numbers);
    deviceMemcpy(&h_energy_sum, d_energy_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
    controller->Step_Print(this->module_name, h_energy_sum, true);
}
