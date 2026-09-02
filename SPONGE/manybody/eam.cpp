#include "eam.h"

#include <algorithm>
#include <highfive/highfive.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
struct NativeEAMDefinition
{
    int atom_type_count = 0;
    int nrho = 0;
    int nr = 0;
    float drho = 0.0f;
    float dr = 0.0f;
    float cut = 0.0f;
    std::vector<float> embed;
    std::vector<float> electron_density;
    std::vector<float> pair_potential;
    std::vector<int> atom_type;
};

std::vector<float> Read_EAM_Float_Array(
    HighFive::File* file, const std::string& path,
    const std::vector<std::size_t>& expected_dimensions)
{
    HighFive::DataSet dataset = file->getDataSet(path);
    const auto dimensions = dataset.getSpace().getDimensions();
    if (dimensions != expected_dimensions)
    {
        throw std::runtime_error(path + " has invalid dimensions");
    }
    std::size_t count = 1;
    for (const std::size_t dimension : dimensions) count *= dimension;
    std::vector<float> values(count);
    if (H5Dread(dataset.getId(), H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                H5P_DEFAULT, values.data()) < 0)
    {
        throw std::runtime_error("failed to read " + path);
    }
    return values;
}

bool Read_H5_EAM_Input(CONTROLLER* controller, const char* module_name,
                       int atom_numbers, NativeEAMDefinition* definition)
{
    constexpr const char* input_key = "input_h5_topology_path";
    constexpr const char* root = "/manybody/eam";
    if (strcmp(module_name, "EAM") != 0 ||
        controller->Command_Exist(module_name, "in_file") ||
        !controller->Command_Exist(input_key))
    {
        return false;
    }
    try
    {
        HighFive::File file(controller->Command(input_key),
                            HighFive::File::ReadOnly);
        if (!file.exist(root)) return false;
        NativeEAMDefinition result;
        file.getDataSet(std::string(root) + "/atom_type_count")
            .read(result.atom_type_count);
        file.getDataSet(std::string(root) + "/nrho").read(result.nrho);
        file.getDataSet(std::string(root) + "/nr").read(result.nr);
        file.getDataSet(std::string(root) + "/drho").read(result.drho);
        file.getDataSet(std::string(root) + "/dr").read(result.dr);
        file.getDataSet(std::string(root) + "/cut").read(result.cut);
        if (result.atom_type_count <= 0 || result.nrho < 2 || result.nr < 2 ||
            result.drho <= 0.0f || result.dr <= 0.0f || result.cut <= 0.0f)
        {
            throw std::runtime_error(
                "/manybody/eam table dimensions and spacings must be "
                "positive");
        }
        const std::size_t type_count =
            static_cast<std::size_t>(result.atom_type_count);
        result.embed = Read_EAM_Float_Array(
            &file, std::string(root) + "/embed/value",
            {type_count, static_cast<std::size_t>(result.nrho)});
        result.electron_density = Read_EAM_Float_Array(
            &file, std::string(root) + "/electron_density/value",
            {type_count, static_cast<std::size_t>(result.nr)});
        result.pair_potential = Read_EAM_Float_Array(
            &file, std::string(root) + "/pair_potential/value",
            {type_count, type_count, static_cast<std::size_t>(result.nr)});
        if (file.exist(std::string(root) + "/atom_type"))
        {
            HighFive::DataSet dataset =
                file.getDataSet(std::string(root) + "/atom_type");
            const auto dimensions = dataset.getSpace().getDimensions();
            if (dimensions.size() != 1 ||
                dimensions[0] != static_cast<std::size_t>(atom_numbers))
            {
                throw std::runtime_error(
                    "/manybody/eam/atom_type must match runtime atom count");
            }
            dataset.read(result.atom_type);
        }
        else if (result.atom_type_count == 1)
        {
            result.atom_type.assign(static_cast<std::size_t>(atom_numbers), 0);
        }
        else
        {
            throw std::runtime_error(
                "multi-type /manybody/eam requires /manybody/eam/atom_type");
        }
        for (const int type : result.atom_type)
        {
            if (type < 0 || type >= result.atom_type_count)
            {
                throw std::runtime_error(
                    "/manybody/eam/atom_type contains an out-of-range type");
            }
        }
        *definition = std::move(result);
        return true;
    }
    catch (const std::exception& error)
    {
        const std::string message =
            std::string("Reason:\n\tfailed to read typed EAM input from ") +
            root + ": " + error.what() + "\n";
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "Read_H5_EAM_Input", message.c_str());
    }
    return false;
}
}  // namespace

struct EAM_TABLE_SAMPLE
{
    float value;
    float derivative;
};

static __host__ __device__ __forceinline__ EAM_TABLE_SAMPLE
EAM_Interpolate_Value_And_Derivative(const float* table, const int n,
                                     const float delta, const float x)
{
    float index = x / delta;
    int i = static_cast<int>(index);
    if (i < 0) i = 0;
    if (i >= n - 1) i = n - 2;

    const int idx0 = i > 0 ? i - 1 : 0;
    const int idx1 = i;
    const int idx2 = i + 1;
    const int idx3 = i + 2 < n ? i + 2 : n - 1;
    const float p0 = table[idx0];
    const float p1 = table[idx1];
    const float p2 = table[idx2];
    const float p3 = table[idx3];
    const float t = index - static_cast<float>(i);
    const float t2 = t * t;
    const float a1 = -p0 + p2;
    const float a2 = 2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3;
    const float a3 = -p0 + 3.0f * p1 - 3.0f * p2 + p3;
    return {0.5f * (2.0f * p1 + a1 * t + a2 * t2 + a3 * t2 * t),
            0.5f * (a1 + 2.0f * a2 * t + 3.0f * a3 * t2) / delta};
}

static __global__ void EAM_Gather_Clustered_Fields(
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
static __device__ __forceinline__ float EAM_Reduce_Subgroup_Component(
    float x, float y, float z, const int component_lane)
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

static __device__ __forceinline__ float EAM_Reduce_Subgroup_Scalar(float x)
{
#ifdef USE_CUDA
    const unsigned int active_mask = __activemask();
#endif
    for (int delta = 4; delta >= 1; delta >>= 1)
    {
#ifdef USE_CUDA
        x += __shfl_down_sync(active_mask, x, delta, 8);
#else
        x += __shfl_down(x, delta, 8);
#endif
    }
    return x;
}

static __device__ __forceinline__ void EAM_Atomic_Add_Force_Component(
    VECTOR* frc, const int atom, const int component, const float value)
{
    atomicAdd(reinterpret_cast<float*>(frc + atom) + component, value);
}

static __global__ void EAM_Clustered_Gmxpacked_Rho(
    const int sci_numbers, const int packed_partitions,
    const int cluster_numbers, const int* cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const int* super_cluster_offsets,
    const CLUSTERED_GMXPACKED_SCI* sci_entries,
    const CLUSTERED_GMXPACKED_CJ* cjpacked_entries,
    const CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    const uint64_t* pair_shift_bits, const int* sorted_atom_ids,
    const VECTOR* sorted_crd, const int* atom_types, const float* rho_table,
    const int nr, const float dr, const float cutoff, const LTMatrix3 cell,
    float* rho)
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
            float rho_j_sum = 0.0f;
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
                const float rho_i = EAM_Interpolate_Value_And_Derivative(
                                        rho_table + type_j * nr, nr, dr, rij)
                                        .value;
                const float rho_j = EAM_Interpolate_Value_And_Derivative(
                                        rho_table + type_i * nr, nr, dr, rij)
                                        .value;
                atomicAdd(rho + atom_i, rho_i);
                rho_j_sum += rho_j;
            }
            const float reduced_rho_j = EAM_Reduce_Subgroup_Scalar(rho_j_sum);
            if (i_lane == 0) atomicAdd(rho + atom_j, reduced_rho_j);
        }
    }
}

template <bool full_output>
static __global__ void EAM_Clustered_Gmxpacked_Force(
    const int sci_numbers, const int packed_partitions,
    const int cluster_numbers, const int* cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const int* super_cluster_offsets,
    const CLUSTERED_GMXPACKED_SCI* sci_entries,
    const CLUSTERED_GMXPACKED_CJ* cjpacked_entries,
    const CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    const uint64_t* pair_shift_bits, const int* sorted_atom_ids,
    const VECTOR* sorted_crd, const int* atom_types, const float* rho_table,
    const float* phi_table, const int ntypes, const int nr, const float dr,
    const float cutoff, const LTMatrix3 cell, const float* df_drho, VECTOR* frc,
    float* atom_energy, LTMatrix3* atom_virial, float* energy_sum,
    const bool store_energy, const bool store_virial)
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
            const float dfj = df_drho[atom_j];
            const uint64_t shift_bits =
                pair_shift_bits[packed_idx * kClusteredJGroupSize + jm];
            VECTOR force_j = {0.0f, 0.0f, 0.0f};
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
                const EAM_TABLE_SAMPLE phi =
                    EAM_Interpolate_Value_And_Derivative(
                        phi_table + (type_i * ntypes + type_j) * nr, nr, dr,
                        rij);
                const EAM_TABLE_SAMPLE rho_at_j =
                    EAM_Interpolate_Value_And_Derivative(
                        rho_table + type_j * nr, nr, dr, rij);
                const EAM_TABLE_SAMPLE rho_at_i =
                    EAM_Interpolate_Value_And_Derivative(
                        rho_table + type_i * nr, nr, dr, rij);
                const float radial_derivative =
                    phi.derivative + df_drho[atom_i] * rho_at_j.derivative +
                    dfj * rho_at_i.derivative;
                const VECTOR force_i = (-radial_derivative / rij) * drij;
                atomicAdd(frc + atom_i, force_i);
                force_j = force_j - force_i;
                if constexpr (full_output)
                {
                    if (store_energy)
                    {
                        const float half_phi = 0.5f * phi.value;
                        atomicAdd(atom_energy + atom_i, half_phi);
                        atomicAdd(atom_energy + atom_j, half_phi);
                        atomicAdd(energy_sum, phi.value);
                    }
                    if (store_virial)
                    {
                        const LTMatrix3 half_virial =
                            0.5f * Get_Virial_From_Force_Dis(force_i, drij);
                        atomicAdd(atom_virial + atom_i, half_virial);
                        atomicAdd(atom_virial + atom_j, half_virial);
                    }
                }
            }
            const float reduced_force = EAM_Reduce_Subgroup_Component(
                force_j.x, force_j.y, force_j.z, i_lane);
            if (i_lane < 3)
            {
                EAM_Atomic_Add_Force_Component(frc, atom_j, i_lane,
                                               reduced_force);
            }
        }
    }
}
#endif

#ifdef USE_CPU
template <bool density_stage, bool full_output>
static void EAM_Clustered_Gmxpacked_Pairs(
    const CLUSTERED_SPATIAL_VIEW& view, const int* sorted_atom_ids,
    const VECTOR* sorted_crd, const int* atom_types, const float* rho_table,
    const float* phi_table, const int ntypes, const int nr,
    const float table_dr, const float cutoff, const LTMatrix3 cell, float* rho,
    const float* df_drho, VECTOR* frc, float* atom_energy,
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
                const unsigned int jm_shift =
                    static_cast<unsigned int>(
                        jm * kClusteredSuperClusterClusters);
                for (int j_lane = 0; j_lane < view.cluster_size;
                     j_lane += 1)
                {
                    if (!Clustered_Lane_Is_Valid(
                            view.cluster_valid_masks[cluster_j], j_lane) ||
                        !Clustered_Lane_Is_Local(
                            view.cluster_local_masks[cluster_j], j_lane))
                    {
                        continue;
                    }
                    const int split =
                        j_lane / kClusteredSplitJClusterSize;
                    const int split_j_lane =
                        j_lane - split * kClusteredSplitJClusterSize;
                    const CLUSTERED_GMXPACKED_SPLIT& split_entry =
                        packed.split[split];
                    const int sorted_j =
                        view.cluster_offsets[cluster_j] + j_lane;
                    const int atom_j = sorted_atom_ids[sorted_j];
                    const int type_j = atom_types[atom_j];
                    const VECTOR rj = sorted_crd[sorted_j];
                    float rho_j_sum = 0.0f;
                    VECTOR force_j = {0.0f, 0.0f, 0.0f};
                    float energy_j_sum = 0.0f;
                    float pair_energy_sum = 0.0f;
                    LTMatrix3 virial_j_sum = {};
                    for (int i_lane = 0; i_lane < view.cluster_size;
                         i_lane += 1)
                    {
                        unsigned int pair_bits = 0xffffffffu;
                        if (split_entry.exclusion_index != 0)
                        {
                            pair_bits =
                                view.gmxpacked_exclusions
                                    [split_entry.exclusion_index]
                                        .pair[split_j_lane *
                                                  kClusteredClusterSize +
                                              i_lane];
                        }
                        const unsigned int active_i_mask =
                            (split_entry.imask & pair_bits) >> jm_shift;
                        if (active_i_mask == 0u) continue;
                        for (int i_local = 0;
                             i_local < cluster_i_numbers; i_local += 1)
                        {
                            if ((active_i_mask &
                                 (1u << static_cast<unsigned int>(i_local))) ==
                                0u)
                            {
                                continue;
                            }
                            const int cluster_i =
                                cluster_i_begin + i_local;
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
                            if (rij_sq <= 0.0f || rij_sq >= cutoff_sq) continue;
                            const float rij = sqrtf(rij_sq);
                            if constexpr (density_stage)
                            {
                                atomicAdd(rho + atom_i,
                                          EAM_Interpolate_Value_And_Derivative(
                                              rho_table + type_j * nr, nr,
                                              table_dr, rij)
                                              .value);
                                rho_j_sum +=
                                    EAM_Interpolate_Value_And_Derivative(
                                        rho_table + type_i * nr, nr, table_dr,
                                        rij)
                                        .value;
                            }
                            else
                            {
                                const EAM_TABLE_SAMPLE phi =
                                    EAM_Interpolate_Value_And_Derivative(
                                        phi_table +
                                            (type_i * ntypes + type_j) * nr,
                                        nr, table_dr, rij);
                                const EAM_TABLE_SAMPLE rho_at_j =
                                    EAM_Interpolate_Value_And_Derivative(
                                        rho_table + type_j * nr, nr, table_dr,
                                        rij);
                                const EAM_TABLE_SAMPLE rho_at_i =
                                    EAM_Interpolate_Value_And_Derivative(
                                        rho_table + type_i * nr, nr, table_dr,
                                        rij);
                                const float radial_derivative =
                                    phi.derivative +
                                    df_drho[atom_i] * rho_at_j.derivative +
                                    df_drho[atom_j] * rho_at_i.derivative;
                                const VECTOR force_i =
                                    (-radial_derivative / rij) * drij;
                                atomicAdd(frc + atom_i, force_i);
                                force_j = force_j - force_i;
                                if constexpr (full_output)
                                {
                                    if (store_energy)
                                    {
                                        const float half_phi =
                                            0.5f * phi.value;
                                        atomicAdd(atom_energy + atom_i,
                                                  half_phi);
                                        energy_j_sum += half_phi;
                                        pair_energy_sum += phi.value;
                                    }
                                    if (store_virial)
                                    {
                                        const LTMatrix3 half_virial =
                                            0.5f * Get_Virial_From_Force_Dis(
                                                       force_i, drij);
                                        atomicAdd(atom_virial + atom_i,
                                                  half_virial);
                                        virial_j_sum =
                                            virial_j_sum + half_virial;
                                    }
                                }
                            }
                        }
                    }
                    if constexpr (density_stage)
                    {
                        if (rho_j_sum != 0.0f)
                        {
                            atomicAdd(rho + atom_j, rho_j_sum);
                        }
                    }
                    else
                    {
                        atomicAdd(frc + atom_j, force_j);
                        if constexpr (full_output)
                        {
                            if (store_energy)
                            {
                                atomicAdd(atom_energy + atom_j, energy_j_sum);
                                atomicAdd(energy_sum, pair_energy_sum);
                            }
                            if (store_virial)
                            {
                                atomicAdd(atom_virial + atom_j,
                                          virial_j_sum);
                            }
                        }
                    }
                }
            }
        }
    }
}
#endif

template <bool need_energy>
static __global__ void EAM_Calculate_DF_Rho_CUDA(
    const int atom_numbers, float* embed_table, int* atom_types, int nrho,
    float drho, float* d_rho, float* d_df_drho, float* atom_energy,
    float* d_energy_sum)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        int type_i = atom_types[atom_i];
        float* current_embed_table = embed_table + type_i * nrho;

        const EAM_TABLE_SAMPLE F_i = EAM_Interpolate_Value_And_Derivative(
            current_embed_table, nrho, drho, d_rho[atom_i]);

        d_df_drho[atom_i] = F_i.derivative;
        if (need_energy)
        {
            atom_energy[atom_i] += F_i.value;
            atomicAdd(d_energy_sum, F_i.value);
        }
    }
}

bool EAM_INFORMATION::EAM_Force_Clustered(
    const CLUSTERED_SPATIAL_VIEW& view, const VECTOR* crd, VECTOR* frc,
    const LTMatrix3 cell, const LTMatrix3 rcell, const int need_atom_energy,
    float* atom_energy, const int need_virial, LTMatrix3* atom_virial,
    const char** failure_reason)
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
            "clustered EAM currently requires a single-rank all-local "
            "spatial view");
    }
    if (d_clustered_sorted_crd == NULL ||
        clustered_scratch_capacity < view.total_atom_numbers)
    {
        return fail("clustered EAM sorted-coordinate scratch is undersized");
    }

    if (view.cluster_numbers <= 0 || view.cluster_offsets == NULL ||
        view.cluster_centers == NULL || view.sort_permutation == NULL)
    {
        return fail("clustered EAM coordinate layout is unavailable");
    }

    Launch_Device_Kernel(
        EAM_Gather_Clustered_Fields, (view.total_atom_numbers + 255) / 256, 256,
        0, NULL, view.total_atom_numbers, view.cluster_numbers,
        view.sort_permutation, view.cluster_offsets, view.cluster_centers, crd,
        cell, rcell, d_clustered_sorted_crd);
    deviceMemset(d_rho, 0, sizeof(float) * atom_numbers);
    if (need_atom_energy)
    {
        deviceMemset(d_energy_sum, 0, sizeof(float));
    }
    const bool has_pair_payload = view.gmxpacked_sci_numbers > 0;

#ifdef USE_CPU
    if (has_pair_payload &&
        (view.gmxpacked_sci == NULL || view.gmxpacked_cjpacked == NULL ||
         view.gmxpacked_exclusions == NULL))
    {
        return fail("clustered EAM requires the gmxpacked pair payload");
    }
    if (has_pair_payload)
    {
        EAM_Clustered_Gmxpacked_Pairs<true, false>(
            view, view.sort_permutation, d_clustered_sorted_crd, d_atom_type,
            d_electron_density, d_pair_potential, atom_type_numbers, nr, dr,
            cut, cell, d_rho, d_df_drho, frc, atom_energy, atom_virial,
            d_energy_sum, false, false);
    }
#else
    if (has_pair_payload &&
        (view.gmxpacked_sci == NULL || view.gmxpacked_cjpacked == NULL ||
         view.gmxpacked_exclusions == NULL || view.pair_shift_bits == NULL))
    {
        return fail("clustered EAM requires the gmxpacked pair payload");
    }
    constexpr int packed_partitions = 8;
    const dim3 pair_block(static_cast<unsigned int>(kClusteredClusterSize),
                          static_cast<unsigned int>(kClusteredClusterSize), 1u);
    const dim3 pair_grid(static_cast<unsigned int>(view.gmxpacked_sci_numbers),
                         static_cast<unsigned int>(packed_partitions), 1u);
    if (has_pair_payload)
    {
        Launch_Device_Kernel(
            EAM_Clustered_Gmxpacked_Rho, pair_grid, pair_block, 0, NULL,
            view.gmxpacked_sci_numbers, packed_partitions,
            view.cluster_numbers, view.cluster_offsets,
            view.cluster_valid_masks, view.cluster_local_masks,
            view.super_cluster_offsets, view.gmxpacked_sci,
            view.gmxpacked_cjpacked, view.gmxpacked_exclusions,
            view.pair_shift_bits, view.sort_permutation,
            d_clustered_sorted_crd, d_atom_type, d_electron_density, nr, dr,
            cut, cell, d_rho);
    }
#endif

    const int threads = 256;
    const int blocks = (atom_numbers + threads - 1) / threads;
    auto df_rho_kernel = EAM_Calculate_DF_Rho_CUDA<false>;
    if (need_atom_energy)
    {
        df_rho_kernel = EAM_Calculate_DF_Rho_CUDA<true>;
    }
    Launch_Device_Kernel(df_rho_kernel, blocks, threads, 0, NULL, atom_numbers,
                         d_embed, d_atom_type, nrho, drho, d_rho, d_df_drho,
                         atom_energy, d_energy_sum);
    if (!has_pair_payload)
    {
        return true;
    }

#ifdef USE_CPU
    if (need_atom_energy || need_virial)
    {
        EAM_Clustered_Gmxpacked_Pairs<false, true>(
            view, view.sort_permutation, d_clustered_sorted_crd, d_atom_type,
            d_electron_density, d_pair_potential, atom_type_numbers, nr, dr,
            cut, cell, d_rho, d_df_drho, frc, atom_energy, atom_virial,
            d_energy_sum, need_atom_energy != 0, need_virial != 0);
    }
    else
    {
        EAM_Clustered_Gmxpacked_Pairs<false, false>(
            view, view.sort_permutation, d_clustered_sorted_crd, d_atom_type,
            d_electron_density, d_pair_potential, atom_type_numbers, nr, dr,
            cut, cell, d_rho, d_df_drho, frc, atom_energy, atom_virial,
            d_energy_sum, false, false);
    }
#else
    auto force_kernel = EAM_Clustered_Gmxpacked_Force<false>;
    if (need_atom_energy || need_virial)
    {
        force_kernel = EAM_Clustered_Gmxpacked_Force<true>;
    }
    Launch_Device_Kernel(
        force_kernel, pair_grid, pair_block, 0, NULL,
        view.gmxpacked_sci_numbers, packed_partitions, view.cluster_numbers,
        view.cluster_offsets, view.cluster_valid_masks,
        view.cluster_local_masks, view.super_cluster_offsets,
        view.gmxpacked_sci, view.gmxpacked_cjpacked, view.gmxpacked_exclusions,
        view.pair_shift_bits, view.sort_permutation, d_clustered_sorted_crd,
        d_atom_type, d_electron_density, d_pair_potential, atom_type_numbers,
        nr, dr, cut, cell, d_df_drho, frc, atom_energy, atom_virial,
        d_energy_sum, need_atom_energy != 0, need_virial != 0);
#endif
    return true;
}

void EAM_INFORMATION::Read_Funcfl(FILE* fp, CONTROLLER* controller)
{
    this->atom_type_numbers = 1;
    char line[CHAR_LENGTH_MAX];

    int atomic_number;
    float mass, lattice_constant;
    char lattice_type[CHAR_LENGTH_MAX];
    if (fscanf(fp, "%d %f %f %s\n", &atomic_number, &mass, &lattice_constant,
               lattice_type) != 4)
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "EAM_INFORMATION::Read_Funcfl",
                                       "Failed to read header info");
    }

    if (fscanf(fp, "%d %f %d %f %f\n", &nrho, &drho, &nr, &dr, &cut) != 5)
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "EAM_INFORMATION::Read_Funcfl",
                                       "Failed to read table parameters");
    }

    controller->printf(
        "    EAM (Funcfl) params: nrho=%d, drho=%f, nr=%d, dr=%f, cut=%f\n",
        nrho, drho, nr, dr, cut);

    Malloc_Safely((void**)&h_embed, sizeof(float) * nrho * atom_type_numbers);
    Malloc_Safely((void**)&h_electron_density,
                  sizeof(float) * nr * atom_type_numbers);
    Malloc_Safely((void**)&h_pair_potential,
                  sizeof(float) * nr * atom_type_numbers * atom_type_numbers);

    float* temp_Z;
    Malloc_Safely((void**)&temp_Z, sizeof(float) * nr);

    for (int i = 0; i < nrho; i++)
    {
        if (fscanf(fp, "%f", h_embed + i) != 1) break;
        h_embed[i] *= CONSTANT_EV_TO_KCAL_MOL;
    }

    for (int i = 0; i < nr; i++)
        if (fscanf(fp, "%f", temp_Z + i) != 1) break;

    for (int i = 0; i < nr; i++)
        if (fscanf(fp, "%f", h_electron_density + i) != 1) break;

    for (int i = 0; i < nr; i++)
    {
        float r = i * dr;
        if (i == 0) r = 1e-8f;
        float z = temp_Z[i];
        h_pair_potential[i] = (z * z / r) *
                              CONSTANT_HARTREE_BOHR_TO_EV_ANGSTROM *
                              CONSTANT_EV_TO_KCAL_MOL;
    }

    free(temp_Z);
}

void EAM_INFORMATION::Read_Setfl(FILE* fp, CONTROLLER* controller)
{
    char line[CHAR_LENGTH_MAX];
    fgets(line, CHAR_LENGTH_MAX, fp);
    fgets(line, CHAR_LENGTH_MAX, fp);
    fgets(line, CHAR_LENGTH_MAX, fp);
    if (fscanf(fp, "%d", &atom_type_numbers) != 1)
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "EAM_INFORMATION::Read_Setfl",
                                       "Failed to read atom_type_numbers");
    }
    fgets(line, CHAR_LENGTH_MAX, fp);
    if (fscanf(fp, "%d %f %d %f %f\n", &nrho, &drho, &nr, &dr, &cut) != 5)
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "EAM_INFORMATION::Read_Setfl",
                                       "Failed to read table parameters");
    }
    controller->printf(
        "    EAM (Setfl) params: ntypes=%d, nrho=%d, drho=%f, nr=%d, dr=%f, "
        "cut=%f\n",
        atom_type_numbers, nrho, drho, nr, dr, cut);
    Malloc_Safely((void**)&h_embed, sizeof(float) * nrho * atom_type_numbers);
    Malloc_Safely((void**)&h_electron_density,
                  sizeof(float) * nr * atom_type_numbers);
    Malloc_Safely((void**)&h_pair_potential,
                  sizeof(float) * nr * atom_type_numbers * atom_type_numbers);

    float* all_Z;
    Malloc_Safely((void**)&all_Z, sizeof(float) * nr * atom_type_numbers);

    for (int t = 0; t < atom_type_numbers; t++)
    {
        int atomic_number;
        float mass, lattice_constant;
        char lattice_type[CHAR_LENGTH_MAX];
        if (fscanf(fp, "%d %f %f %s\n", &atomic_number, &mass,
                   &lattice_constant, lattice_type) != 4)
        {
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                           "EAM_INFORMATION::Read_Setfl",
                                           "Failed to read element header");
        }
        float* this_embed = h_embed + t * nrho;
        for (int i = 0; i < nrho; i++)
        {
            if (fscanf(fp, "%f", this_embed + i) != 1) break;
            this_embed[i] *= CONSTANT_EV_TO_KCAL_MOL;
        }
        float* this_rho = h_electron_density + t * nr;
        for (int i = 0; i < nr; i++)
        {
            if (fscanf(fp, "%f", this_rho + i) != 1) break;
        }
    }
    for (int i = 0; i < atom_type_numbers; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            float val;
            float* phi_ij = h_pair_potential + (i * atom_type_numbers + j) * nr;
            float* phi_ji = h_pair_potential + (j * atom_type_numbers + i) * nr;

            for (int k = 0; k < nr; k++)
            {
                if (fscanf(fp, "%f", &val) != 1) break;
                float r = k * dr;
                if (k == 0) r = 1e-8f;
                float phi = (val / r) * CONSTANT_EV_TO_KCAL_MOL;
                phi_ij[k] = phi;
                if (i != j) phi_ji[k] = phi;
            }
        }
    }

    free(all_Z);
}

void EAM_INFORMATION::Initial(CONTROLLER* controller, const int atom_numbers,
                              const char* module_name)
{
    if (module_name == NULL)
        strcpy(this->module_name, "EAM");
    else
        strcpy(this->module_name, module_name);

    NativeEAMDefinition native_definition;
    const bool has_native_definition = Read_H5_EAM_Input(
        controller, this->module_name, atom_numbers, &native_definition);
    if (!has_native_definition &&
        !controller->Command_Exist(this->module_name, "in_file"))
    {
        controller->printf("%s FORCE IS NOT INITIALIZED\n\n",
                           this->module_name);
        return;
    }

    if (has_native_definition)
    {
        controller->printf("START INITIALIZING EAM FORCE FROM NATIVE H5\n");
        atom_type_numbers = native_definition.atom_type_count;
        nrho = native_definition.nrho;
        nr = native_definition.nr;
        drho = native_definition.drho;
        dr = native_definition.dr;
        cut = native_definition.cut;
        Malloc_Safely((void**)&h_embed,
                      sizeof(float) * native_definition.embed.size());
        Malloc_Safely(
            (void**)&h_electron_density,
            sizeof(float) * native_definition.electron_density.size());
        Malloc_Safely((void**)&h_pair_potential,
                      sizeof(float) * native_definition.pair_potential.size());
        std::copy(native_definition.embed.begin(),
                  native_definition.embed.end(), h_embed);
        std::copy(native_definition.electron_density.begin(),
                  native_definition.electron_density.end(), h_electron_density);
        std::copy(native_definition.pair_potential.begin(),
                  native_definition.pair_potential.end(), h_pair_potential);
    }
    else
    {
        controller->printf("START INITIALIZING EAM FORCE\n");
        FILE* fp;
        Open_File_Safely(&fp, controller->Command(this->module_name, "in_file"),
                         "r");
        char line[CHAR_LENGTH_MAX];
        fgets(line, CHAR_LENGTH_MAX, fp);
        long pos = ftell(fp);
        char line2[CHAR_LENGTH_MAX];
        fgets(line2, CHAR_LENGTH_MAX, fp);
        int temp_int;
        float temp_float;
        int items = sscanf(line2, "%d %f", &temp_int, &temp_float);
        fseek(fp, pos, SEEK_SET);

        if (items == 2)
        {
            controller->printf(
                "    Detected DYNAMO funcfl format (Single Element).\n");
            Read_Funcfl(fp, controller);
        }
        else
        {
            controller->printf("    Detected DYNAMO setfl format (Alloy).\n");
            fseek(fp, 0, SEEK_SET);
            Read_Setfl(fp, controller);
        }
        fclose(fp);
    }

    this->atom_numbers = atom_numbers;
    Device_Malloc_Safely((void**)&d_energy_sum, sizeof(float));

    int num_types = atom_type_numbers;
    Device_Malloc_And_Copy_Safely((void**)&d_embed, h_embed,
                                  sizeof(float) * nrho * num_types);
    Device_Malloc_And_Copy_Safely((void**)&d_electron_density,
                                  h_electron_density,
                                  sizeof(float) * nr * num_types);
    Device_Malloc_And_Copy_Safely((void**)&d_pair_potential, h_pair_potential,
                                  sizeof(float) * nr * num_types * num_types);

    Device_Malloc_Safely((void**)&d_rho, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_df_drho, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_clustered_sorted_crd,
                         sizeof(VECTOR) * atom_numbers);
    clustered_scratch_capacity = atom_numbers;
    Malloc_Safely((void**)&h_atom_type, sizeof(int) * atom_numbers);

    if (controller->Command_Exist(this->module_name, "atom_type_in_file"))
    {
        FILE* fp_type;
        Open_File_Safely(
            &fp_type,
            controller->Command(this->module_name, "atom_type_in_file"), "r");
        for (int i = 0; i < atom_numbers; i++)
        {
            int type_val;
            if (fscanf(fp_type, "%d", &type_val) != 1)
            {
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                               "EAM_INFORMATION::Initial",
                                               "Failed to read atom types");
            }
            h_atom_type[i] = type_val;
        }
        fclose(fp_type);
    }
    else if (has_native_definition)
    {
        std::copy(native_definition.atom_type.begin(),
                  native_definition.atom_type.end(), h_atom_type);
    }
    else
    {
        for (int i = 0; i < atom_numbers; i++) h_atom_type[i] = 0;
    }

    Device_Malloc_And_Copy_Safely((void**)&d_atom_type, h_atom_type,
                                  sizeof(int) * atom_numbers);


    is_initialized = true;
    if (!is_controller_printf_initialized)
    {
        controller->Step_Print_Initial(this->module_name, "%.2f");
        is_controller_printf_initialized = true;
    }
    controller->printf("END INITIALIZING EAM FORCE\n\n");
}

void EAM_INFORMATION::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    deviceMemcpy(&h_energy_sum, d_energy_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
    controller->Step_Print(this->module_name, h_energy_sum, true);
}
