#include "tersoff.h"

#include <algorithm>
#include <cmath>
#include <highfive/highfive.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
template <typename T>
hid_t Tersoff_Native_H5_Type();

template <>
hid_t Tersoff_Native_H5_Type<int>()
{
    return H5T_NATIVE_INT;
}

template <>
hid_t Tersoff_Native_H5_Type<float>()
{
    return H5T_NATIVE_FLOAT;
}

template <typename T>
std::vector<T> Read_Tersoff_H5_Array(HighFive::File* file,
                                     const std::string& dataset_path,
                                     std::size_t expected_rank,
                                     std::size_t expected_columns)
{
    HighFive::DataSet dataset = file->getDataSet(dataset_path);
    const auto dimensions = dataset.getSpace().getDimensions();
    if (dimensions.size() != expected_rank ||
        (expected_rank == 2 && dimensions[1] != expected_columns))
    {
        std::ostringstream message;
        message << dataset_path << " must have rank " << expected_rank;
        if (expected_rank == 2)
        {
            message << " and " << expected_columns << " columns";
        }
        throw std::runtime_error(message.str());
    }
    std::size_t value_count = 1;
    for (const std::size_t dimension : dimensions)
    {
        value_count *= dimension;
    }
    std::vector<T> values(value_count);
    if (value_count != 0 &&
        H5Dread(dataset.getId(), Tersoff_Native_H5_Type<T>(), H5S_ALL, H5S_ALL,
                H5P_DEFAULT, values.data()) < 0)
    {
        throw std::runtime_error("failed to read " + dataset_path);
    }
    return values;
}

std::vector<std::string> Read_Tersoff_Type_Names(HighFive::File* file,
                                                 int atom_type_count)
{
    constexpr const char* path = "/manybody/tersoff/type_name";
    HighFive::DataSet dataset = file->getDataSet(path);
    const auto dimensions = dataset.getSpace().getDimensions();
    if (dimensions.size() != 1 ||
        dimensions[0] != static_cast<std::size_t>(atom_type_count))
    {
        throw std::runtime_error(
            "/manybody/tersoff/type_name must have shape [atom_type_count]");
    }
    std::vector<std::string> names;
    dataset.read(names);
    std::set<std::string> unique_names;
    for (const std::string& name : names)
    {
        if (name.empty() || name.find_first_of(" \t\r\n") != std::string::npos)
        {
            throw std::runtime_error(
                "/manybody/tersoff/type_name contains an empty or whitespace "
                "name");
        }
        if (!unique_names.insert(name).second)
        {
            throw std::runtime_error(
                "/manybody/tersoff/type_name contains a duplicate name");
        }
    }
    return names;
}

bool Tersoff_Parameters_Agree(float actual, float expected)
{
    const float scale = std::max(1.0f, std::fabs(expected));
    return std::isfinite(actual) && std::isfinite(expected) &&
           std::fabs(actual - expected) <= 2.0e-5f * scale;
}

struct NativeTersoffDefinition
{
    int atom_type_count = 0;
    std::vector<int> atom_type;
    std::vector<int> type_map;
    std::vector<float> parameters;
};

bool Read_H5_Tersoff_Input(CONTROLLER* controller, const char* module_name,
                           int atom_numbers,
                           NativeTersoffDefinition* definition)
{
    constexpr const char* input_key = "input_h5_topology_path";
    constexpr const char* root = "/manybody/tersoff";
    if (strcmp(module_name, "TERSOFF") != 0 ||
        controller->Command_Exist(module_name, "in_file") ||
        !controller->Command_Exist(input_key))
    {
        return false;
    }

    try
    {
        HighFive::File file(controller->Command(input_key),
                            HighFive::File::ReadOnly);
        if (!file.exist(root))
        {
            return false;
        }

        int atom_type_count = 0;
        file.getDataSet(std::string(root) + "/atom_type_count")
            .read(atom_type_count);
        if (atom_type_count <= 0)
        {
            throw std::runtime_error(
                "/manybody/tersoff/atom_type_count must be positive");
        }
        const auto atom_type = Read_Tersoff_H5_Array<int>(
            &file, std::string(root) + "/atom_type", 1, 0);
        if (atom_type.size() != static_cast<std::size_t>(atom_numbers))
        {
            throw std::runtime_error(
                "/manybody/tersoff/atom_type must match runtime atom count");
        }
        for (const int type : atom_type)
        {
            if (type < 0 || type >= atom_type_count)
            {
                throw std::runtime_error(
                    "/manybody/tersoff/atom_type contains an out-of-range "
                    "type");
            }
        }

        const auto type_names = Read_Tersoff_Type_Names(&file, atom_type_count);
        const auto type_map =
            Read_Tersoff_H5_Array<int>(&file, std::string(root) + "/map", 1, 0);
        const std::size_t map_size = static_cast<std::size_t>(atom_type_count) *
                                     atom_type_count * atom_type_count;
        if (type_map.size() != map_size)
        {
            throw std::runtime_error(
                "/manybody/tersoff/map must have length atom_type_count^3");
        }

        long long declared_entry_count = 0;
        file.getDataSet(std::string(root) + "/entry/count")
            .read(declared_entry_count);
        if (declared_entry_count <= 0)
        {
            throw std::runtime_error(
                "/manybody/tersoff/entry/count must be positive");
        }
        const auto entry_type = Read_Tersoff_H5_Array<int>(
            &file, std::string(root) + "/entry/type", 2, 3);
        const auto parameters_raw = Read_Tersoff_H5_Array<float>(
            &file, std::string(root) + "/entry/parameters_raw", 2, 14);
        const auto parameters = Read_Tersoff_H5_Array<float>(
            &file, std::string(root) + "/entry/parameters", 2, 18);
        const std::size_t entry_count = entry_type.size() / 3;
        if (entry_count != static_cast<std::size_t>(declared_entry_count) ||
            parameters_raw.size() / 14 != entry_count ||
            parameters.size() / 18 != entry_count || entry_count > map_size)
        {
            throw std::runtime_error(
                "/manybody/tersoff entry count/type/parameter row mismatch");
        }
        HighFive::DataSet entry_name_dataset =
            file.getDataSet(std::string(root) + "/entry/type_name");
        const auto entry_name_dimensions =
            entry_name_dataset.getSpace().getDimensions();
        if (entry_name_dimensions.size() != 2 ||
            entry_name_dimensions[0] != entry_count ||
            entry_name_dimensions[1] != 3)
        {
            throw std::runtime_error(
                "/manybody/tersoff/entry/type_name must have shape [entry,3]");
        }
        std::vector<std::vector<std::string>> entry_type_names;
        entry_name_dataset.read(entry_type_names);

        std::vector<int> expected_map(map_size, -1);
        for (std::size_t row = 0; row < entry_count; ++row)
        {
            const int a = entry_type[3 * row];
            const int b = entry_type[3 * row + 1];
            const int c = entry_type[3 * row + 2];
            if (a < 0 || a >= atom_type_count || b < 0 ||
                b >= atom_type_count || c < 0 || c >= atom_type_count)
            {
                throw std::runtime_error(
                    "/manybody/tersoff/entry/type contains an out-of-range "
                    "type");
            }
            const std::size_t index =
                static_cast<std::size_t>(a) * atom_type_count *
                    atom_type_count +
                static_cast<std::size_t>(b) * atom_type_count + c;
            if (expected_map[index] != -1)
            {
                throw std::runtime_error(
                    "/manybody/tersoff/entry/type contains a duplicate row");
            }
            expected_map[index] = static_cast<int>(row);
            if (entry_type_names[row].size() != 3 ||
                entry_type_names[row][0] != type_names[a] ||
                entry_type_names[row][1] != type_names[b] ||
                entry_type_names[row][2] != type_names[c])
            {
                throw std::runtime_error(
                    "/manybody/tersoff/entry/type_name is inconsistent with "
                    "entry/type");
            }
            float expected_parameters[18] = {};
            for (std::size_t parameter = 0; parameter < 14; ++parameter)
            {
                const float value = parameters_raw[14 * row + parameter];
                if (!std::isfinite(value))
                {
                    throw std::runtime_error(
                        "/manybody/tersoff/entry/parameters_raw contains a "
                        "non-finite value");
                }
                expected_parameters[parameter] = value;
            }
            expected_parameters[9] *= CONSTANT_EV_TO_KCAL_MOL;
            expected_parameters[13] *= CONSTANT_EV_TO_KCAL_MOL;
            const float n = expected_parameters[6];
            if (n > 0.0f)
            {
                expected_parameters[14] = powf(2.0f * n * 1.0e-16f, -1.0f / n);
                expected_parameters[15] = powf(2.0f * n * 1.0e-8f, -1.0f / n);
                expected_parameters[16] = 1.0f / expected_parameters[15];
                expected_parameters[17] = 1.0f / expected_parameters[14];
            }
            for (std::size_t parameter = 0; parameter < 18; ++parameter)
            {
                if (!Tersoff_Parameters_Agree(parameters[18 * row + parameter],
                                              expected_parameters[parameter]))
                {
                    throw std::runtime_error(
                        "/manybody/tersoff/entry/parameters is inconsistent "
                        "with parameters_raw");
                }
            }
        }
        if (type_map != expected_map)
        {
            throw std::runtime_error(
                "/manybody/tersoff/map is inconsistent with entry/type");
        }

        NativeTersoffDefinition result;
        result.atom_type_count = atom_type_count;
        result.atom_type = atom_type;
        result.type_map = type_map;
        result.parameters = parameters;
        *definition = std::move(result);
        return true;
    }
    catch (const std::exception& error)
    {
        const std::string message =
            std::string("Reason:\n\tfailed to read typed Tersoff input from ") +
            root + ": " + error.what() + "\n";
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Read_H5_Tersoff_Input", message.c_str());
    }
    return false;
}
}  // namespace

#ifdef USE_CPU
#include "clustered_gmxpacked_cpu.h"
#endif

enum TersoffParam
{
    p_m = 0,
    p_gamma,
    p_lam3,
    p_c,
    p_d,
    p_h,
    p_n,
    p_beta,
    p_lam2,
    p_B,
    p_R,
    p_D,
    p_lam1,
    p_A,
    p_c1,
    p_c2,
    p_c3,
    p_c4,
    PARAM_STRIDE = 18
};

static __device__ __forceinline__ float ters_fc(float r, const float* param)
{
    float R = param[p_R];
    float D = param[p_D];
    if (r < R - D) return 1.0f;
    if (r > R + D) return 0.0f;
    return 0.5f * (1.0f - sinf(CONSTANT_Pi * 0.5f * (r - R) / D));
}

template <int N>
static __device__ __forceinline__ SADfloat<N> ters_fc_sad(SADfloat<N> r,
                                                          const float* param)
{
    float R = param[p_R];
    float D = param[p_D];
    if (r.val < R - D) return SADfloat<N>(1.0f);
    if (r.val > R + D) return SADfloat<N>(0.0f);
    return 0.5f * (1.0f - sinf(CONSTANT_Pi * 0.5f * (r - R) / D));
}

template <int N>
static __device__ __forceinline__ SADfloat<N> ters_gijk_sad(
    SADfloat<N> costheta, const float* param)
{
    float c = param[p_c];
    float d = param[p_d];
    float h = param[p_h];
    float gamma = param[p_gamma];
    SADfloat<N> diff = h - costheta;
    SADfloat<N> term = d * d + diff * diff;
    return gamma * (1.0f + c * c / (d * d) - c * c / term);
}

static __device__ __forceinline__ float ters_bij(float zeta, const float* param)
{
    float beta = param[p_beta];
    float n = param[p_n];
    float c1 = param[p_c1];
    float c2 = param[p_c2];
    float c3 = param[p_c3];
    float c4 = param[p_c4];
    float tmp = beta * zeta;
    if (tmp > c1) return 1.0f / sqrtf(tmp);
    if (tmp > c2) return (1.0f - powf(tmp, -n) / (2.0f * n)) / sqrtf(tmp);
    if (tmp < c4) return 1.0f;
    if (tmp < c3) return 1.0f - powf(tmp, n) / (2.0f * n);
    return powf(1.0f + powf(tmp, n), -1.0f / (2.0f * n));
}

static __device__ __forceinline__ float ters_bij_d(float zeta,
                                                   const float* param)
{
    float beta = param[p_beta];
    float n = param[p_n];
    float c1 = param[p_c1];
    float c2 = param[p_c2];
    float c3 = param[p_c3];
    float c4 = param[p_c4];
    float tmp = beta * zeta;
    if (tmp > c1) return beta * -0.5f * powf(tmp, -1.5f);
    if (tmp > c2)
        return beta * (-0.5f * powf(tmp, -1.5f) *
                       (1.0f - (1.0f + 1.0f / (2.0f * n)) * powf(tmp, -n)));
    if (tmp < c4) return 0.0f;
    if (tmp < c3) return -0.5f * beta * powf(tmp, n - 1.0f);
    float tmp_n = powf(tmp, n);
    return -0.5f * powf(1.0f + tmp_n, -1.0f - (1.0f / (2.0f * n))) * tmp_n /
           zeta;
}

static __host__ __device__ __forceinline__ bool
Tersoff_Clustered_Neighbor_Within_Cut(const int type_i, const int type_j,
                                      const float distance_squared,
                                      const float* center_cutoffs,
                                      const int atom_type_numbers,
                                      const float margin = 0.0f)
{
    const float cutoff =
        center_cutoffs[type_i * atom_type_numbers + type_j] + margin;
    return cutoff > 0.0f && distance_squared <= cutoff * cutoff;
}

#ifdef USE_GPU
template <bool fill>
static __global__ void Tersoff_Build_Clustered_Center_Atoms(
    const CLUSTERED_SPATIAL_VIEW view, const VECTOR* crd, const LTMatrix3 cell,
    const int* atom_types, const float* center_cutoffs,
    const int atom_type_numbers, const int packed_partitions,
    const int* neighbor_offsets, int* neighbor_atoms, int* neighbor_counts)
{
    const int sci = static_cast<int>(blockIdx.x);
    const int packed_partition = static_cast<int>(blockIdx.y);
    const int i_lane = static_cast<int>(threadIdx.x);
    const int j_lane = static_cast<int>(threadIdx.y);
    if (sci >= view.gmxpacked_sci_numbers || i_lane >= kClusteredClusterSize ||
        j_lane >= kClusteredClusterSize)
    {
        return;
    }
    const CLUSTERED_GMXPACKED_SCI sci_entry = view.gmxpacked_sci[sci];
    const int cluster_i_begin =
        view.super_cluster_offsets[sci_entry.supercluster_id];
    int cluster_i_end =
        view.super_cluster_offsets[sci_entry.supercluster_id + 1];
    if (cluster_i_end > view.cluster_numbers)
    {
        cluster_i_end = view.cluster_numbers;
    }
    const int split = j_lane / kClusteredSplitJClusterSize;
    const int split_j_lane = j_lane - split * kClusteredSplitJClusterSize;
    const unsigned int i_lane_bit = 1u << i_lane;
    const unsigned int j_lane_bit = 1u << j_lane;

    for (int packed_idx = sci_entry.cjpacked_begin + packed_partition;
         packed_idx < sci_entry.cjpacked_end; packed_idx += packed_partitions)
    {
        const CLUSTERED_GMXPACKED_CJ packed =
            view.gmxpacked_cjpacked[packed_idx];
        const unsigned int effective_mask = Clustered_Gmxpacked_Effective_Imask(
            packed, view.gmxpacked_exclusions, split, split_j_lane, i_lane);
        for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
        {
            const int cluster_j = packed.cj[jm];
            if (cluster_j < 0 ||
                (view.cluster_valid_masks[cluster_j] & j_lane_bit) == 0u ||
                (view.cluster_local_masks[cluster_j] & j_lane_bit) == 0u)
            {
                continue;
            }
            const int sorted_j = view.cluster_offsets[cluster_j] + j_lane;
            const int atom_j = view.sort_permutation[sorted_j];
            const int type_j = atom_types[atom_j];
            const uint64_t shift_bits =
                view.pair_shift_bits[packed_idx * kClusteredJGroupSize + jm];
            for (int i_local = 0; i_local < cluster_i_end - cluster_i_begin;
                 i_local += 1)
            {
                if (!Clustered_Gmxpacked_I_Entry_Is_Active(
                        effective_mask, shift_bits, split, jm, i_local))
                {
                    continue;
                }
                const int cluster_i = cluster_i_begin + i_local;
                if ((view.cluster_valid_masks[cluster_i] & i_lane_bit) == 0u ||
                    (view.cluster_local_masks[cluster_i] & i_lane_bit) == 0u)
                {
                    continue;
                }
                const int sorted_i = view.cluster_offsets[cluster_i] + i_lane;
                const int atom_i = view.sort_permutation[sorted_i];
                if (atom_i == atom_j) continue;
                const int type_i = atom_types[atom_i];
                const int shift_id =
                    Clustered_Get_Pair_Shift_Id(shift_bits, i_local);
                const VECTOR shift =
                    Clustered_Shift_Vector_From_Id(shift_id, cell);
                const VECTOR displacement = (crd[atom_i] - crd[atom_j]) + shift;
                const float distance_squared = displacement * displacement;
                if (Tersoff_Clustered_Neighbor_Within_Cut(
                        type_i, type_j, distance_squared, center_cutoffs,
                        atom_type_numbers, view.rebuild_skin))
                {
                    const int slot = atomicAdd(neighbor_counts + atom_i, 1);
                    if constexpr (fill)
                    {
                        neighbor_atoms[neighbor_offsets[atom_i] + slot] =
                            atom_j;
                    }
                }
                if (Tersoff_Clustered_Neighbor_Within_Cut(
                        type_j, type_i, distance_squared, center_cutoffs,
                        atom_type_numbers, view.rebuild_skin))
                {
                    const int slot = atomicAdd(neighbor_counts + atom_j, 1);
                    if constexpr (fill)
                    {
                        neighbor_atoms[neighbor_offsets[atom_j] + slot] =
                            atom_i;
                    }
                }
            }
        }
    }
}
#else
static bool Tersoff_Build_Gmxpacked_Center_Atoms_CPU(
    TERSOFF_INFORMATION* tersoff, const CLUSTERED_SPATIAL_VIEW& view,
    const VECTOR* crd, const LTMatrix3 cell,
    std::vector<std::vector<int>>* center_atoms)
{
    if (center_atoms == NULL) return false;
    center_atoms->assign(static_cast<size_t>(tersoff->atom_numbers), {});
    for (int sci = 0; sci < view.gmxpacked_sci_numbers; sci += 1)
    {
        const CLUSTERED_GMXPACKED_SCI sci_entry = view.gmxpacked_sci[sci];
        const VECTOR shift =
            Clustered_Shift_Vector_From_Id(sci_entry.shift_id, cell);
        auto consume_tile =
            [&](const CLUSTERED_GMXPACKED_CPU_I_TILE_CANDIDATE& tile)
        {
            const int cluster_i = tile.cluster_i;
            const int cluster_j = tile.cluster_j;
            const unsigned int valid_local_j_mask =
                view.cluster_valid_masks[cluster_j] &
                view.cluster_local_masks[cluster_j];
            for (int i_lane = 0; i_lane < view.cluster_size; i_lane += 1)
            {
                const unsigned int i_lane_bit = 1u << i_lane;
                if ((view.cluster_valid_masks[cluster_i] & i_lane_bit) == 0u ||
                    (view.cluster_local_masks[cluster_i] & i_lane_bit) == 0u)
                {
                    continue;
                }
                const int sorted_i = view.cluster_offsets[cluster_i] + i_lane;
                const int atom_i = view.sort_permutation[sorted_i];
                const int type_i = tersoff->d_atom_type[atom_i];
                unsigned int active_j_mask =
                    static_cast<unsigned int>(
                        tile.pair_mask >> (i_lane * kClusteredClusterSize)) &
                    valid_local_j_mask;
                while (active_j_mask != 0u)
                {
                    const int j_lane = __builtin_ctz(active_j_mask);
                    active_j_mask &= active_j_mask - 1u;
                    const int sorted_j =
                        view.cluster_offsets[cluster_j] + j_lane;
                    const int atom_j = view.sort_permutation[sorted_j];
                    if (atom_i == atom_j) continue;
                    const int type_j = tersoff->d_atom_type[atom_j];
                    const VECTOR displacement =
                        (crd[atom_i] - crd[atom_j]) + shift;
                    const float distance_squared = displacement * displacement;
                    if (Tersoff_Clustered_Neighbor_Within_Cut(
                            type_i, type_j, distance_squared,
                            tersoff->d_center_cutoffs,
                            tersoff->atom_type_numbers, view.rebuild_skin))
                    {
                        (*center_atoms)[atom_i].push_back(atom_j);
                    }
                    if (Tersoff_Clustered_Neighbor_Within_Cut(
                            type_j, type_i, distance_squared,
                            tersoff->d_center_cutoffs,
                            tersoff->atom_type_numbers, view.rebuild_skin))
                    {
                        (*center_atoms)[atom_j].push_back(atom_i);
                    }
                }
            }
        };
        Clustered_Gmxpacked_CPU_For_Each_I_Tile_In_SCI(view, sci, consume_tile);
    }
    return true;
}
#endif

static bool Tersoff_Ensure_Clustered_Center_Atoms(
    TERSOFF_INFORMATION* tersoff, const CLUSTERED_SPATIAL_VIEW& view,
    const VECTOR* crd, const LTMatrix3 cell)
{
    ClusteredCSRStorage& storage = tersoff->clustered_neighbors;
    if (tersoff->clustered_neighbor_stamp.Matches(view))
    {
        return true;
    }
    storage.ReserveCounts(tersoff->atom_numbers);
    storage.ReserveOffsets(tersoff->atom_numbers + 1);

#ifdef USE_CPU
    std::vector<std::vector<int>> center_atoms;
    if (!Tersoff_Build_Gmxpacked_Center_Atoms_CPU(tersoff, view, crd, cell,
                                                  &center_atoms))
    {
        return false;
    }
    std::vector<int> host_counts(static_cast<size_t>(tersoff->atom_numbers), 0);
    for (int atom_i = 0; atom_i < tersoff->atom_numbers; atom_i += 1)
    {
        host_counts[atom_i] = static_cast<int>(center_atoms[atom_i].size());
    }
#else
    deviceMemset(storage.counts, 0,
                 sizeof(int) * static_cast<size_t>(tersoff->atom_numbers));
    constexpr int packed_partitions = 16;
    const dim3 pair_block(static_cast<unsigned int>(kClusteredClusterSize),
                          static_cast<unsigned int>(kClusteredClusterSize), 1u);
    const dim3 pair_grid(static_cast<unsigned int>(view.gmxpacked_sci_numbers),
                         static_cast<unsigned int>(packed_partitions), 1u);
    Launch_Device_Kernel(Tersoff_Build_Clustered_Center_Atoms<false>, pair_grid,
                         pair_block, 0, NULL, view, crd, cell,
                         tersoff->d_atom_type, tersoff->d_center_cutoffs,
                         tersoff->atom_type_numbers, packed_partitions, NULL,
                         NULL, storage.counts);
#endif

#ifdef USE_CPU
    std::vector<int> host_offsets(
        static_cast<size_t>(tersoff->atom_numbers + 1), 0);
    long long total = 0;
    for (int atom_i = 0; atom_i < tersoff->atom_numbers; atom_i += 1)
    {
        if (host_counts[atom_i] < 0) return false;
        host_offsets[atom_i] = static_cast<int>(total);
        total += host_counts[atom_i];
        if (total > INT_MAX) return false;
    }
    host_offsets[tersoff->atom_numbers] = static_cast<int>(total);
    storage.item_count = static_cast<int>(total);
    deviceMemcpy(storage.offsets, host_offsets.data(),
                 sizeof(int) * static_cast<size_t>(tersoff->atom_numbers + 1),
                 deviceMemcpyHostToDevice);
#else
    if (!Clustered_CSR_Device_Exclusive_Scan(&storage, tersoff->atom_numbers))
    {
        return false;
    }
#endif
    storage.ReserveItems(storage.item_count);

#ifdef USE_CPU
    for (int atom_i = 0; atom_i < tersoff->atom_numbers; atom_i += 1)
    {
        const std::vector<int>& row = center_atoms[atom_i];
        if (!row.empty())
        {
            deviceMemcpy(storage.items + host_offsets[atom_i], row.data(),
                         sizeof(int) * row.size(), deviceMemcpyHostToDevice);
        }
    }
#else
    if (storage.item_count > 0)
    {
        deviceMemset(storage.counts, 0,
                     sizeof(int) * static_cast<size_t>(tersoff->atom_numbers));
        Launch_Device_Kernel(Tersoff_Build_Clustered_Center_Atoms<true>,
                             pair_grid, pair_block, 0, NULL, view, crd, cell,
                             tersoff->d_atom_type, tersoff->d_center_cutoffs,
                             tersoff->atom_type_numbers, packed_partitions,
                             storage.offsets, storage.items, storage.counts);
    }
#endif
    tersoff->clustered_neighbor_stamp.Capture(view);
    return true;
}

template <bool full_output>
static __global__ void Tersoff_Force_CUDA(
    const int atom_numbers, const VECTOR* crd, VECTOR* frc,
    const LTMatrix3 cell, const LTMatrix3 rcell, const int* neighbor_offsets,
    const int* neighbor_atoms, int* atom_types, float* params, int* map,
    int ntypes, float* atom_energy, LTMatrix3* atom_virial, float* d_energy_sum,
    const bool store_energy, const bool store_virial)
{
#ifdef USE_GPU
    const int i = static_cast<int>(threadIdx.y) +
                  static_cast<int>(blockDim.y * blockIdx.x);
    if (i < atom_numbers)
#else
    SIMPLE_DEVICE_FOR(i, atom_numbers)
#endif
    {
        int type_i = atom_types[i];
        VECTOR ri = crd[i];
        const int neighbor_begin = neighbor_offsets[i];
        const int neighbor_end = neighbor_offsets[i + 1];
        VECTOR fi = {0, 0, 0};
        float en_i = 0;
        LTMatrix3 vi = {0, 0, 0, 0, 0, 0};

#ifdef USE_GPU
        for (int jj = neighbor_begin + static_cast<int>(threadIdx.x);
             jj < neighbor_end; jj += static_cast<int>(blockDim.x))
#else
        for (int jj = neighbor_begin; jj < neighbor_end; jj++)
#endif
        {
            int j = neighbor_atoms[jj];
            int type_j = atom_types[j];
            VECTOR rj = crd[j];
            VECTOR drij = Get_Periodic_Displacement(ri, rj, cell, rcell);
            float rij = norm3df(drij.x, drij.y, drij.z);
            int param_idx_ij =
                map[type_i * ntypes * ntypes + type_j * ntypes + type_j];
            const float* param_ij = params + param_idx_ij * PARAM_STRIDE;
            if (rij > param_ij[p_R] + param_ij[p_D]) continue;

            SADfloat<1> rij_sad(rij, 0);
            SADfloat<1> fr = param_ij[p_A] * expf(-param_ij[p_lam1] * rij_sad) *
                             ters_fc_sad(rij_sad, param_ij);

            float f_rep = -0.5f * fr.dval[0] / rij;
            fi.x += f_rep * drij.x;
            fi.y += f_rep * drij.y;
            fi.z += f_rep * drij.z;
            if constexpr (full_output)
            {
                if (store_energy)
                {
                    float ev = 0.5f * fr.val;
                    en_i += ev;
                    atomicAdd(d_energy_sum, ev);
                }
            }
            if constexpr (full_output)
            {
                if (store_virial)
                {
                    vi = vi + Get_Virial_From_Force_Dis(f_rep * drij, drij);
                }
            }

            float zeta = 0;
            for (int kk = neighbor_begin; kk < neighbor_end; kk++)
            {
                if (jj == kk) continue;
                int k = neighbor_atoms[kk];
                int type_k = atom_types[k];
                int param_idx_ijk =
                    map[type_i * ntypes * ntypes + type_j * ntypes + type_k];
                const float* param_ijk = params + param_idx_ijk * PARAM_STRIDE;
                VECTOR rk = crd[k];
                VECTOR drik = Get_Periodic_Displacement(ri, rk, cell, rcell);
                float rik = norm3df(drik.x, drik.y, drik.z);
                if (rik > param_ijk[p_R] + param_ijk[p_D]) continue;
                float costheta =
                    (drij.x * drik.x + drij.y * drik.y + drij.z * drik.z) /
                    (rij * rik);
                float fc_ik = ters_fc(rik, param_ijk);
                float g = param_ijk[p_gamma] *
                          (1.0f +
                           param_ijk[p_c] * param_ijk[p_c] /
                               (param_ijk[p_d] * param_ijk[p_d]) -
                           param_ijk[p_c] * param_ijk[p_c] /
                               (param_ijk[p_d] * param_ijk[p_d] +
                                (param_ijk[p_h] - costheta) *
                                    (param_ijk[p_h] - costheta)));
                float diff = rij - rik;
                float arg = param_ijk[p_lam3] * diff;
                if (fabsf(param_ijk[p_m] - 3.0f) < 1e-5) arg = arg * arg * arg;
                zeta += fc_ik * g * expf(arg);
            }

            float bij = ters_bij(zeta, param_ij);
            float fa_val = -param_ij[p_B] * expf(-param_ij[p_lam2] * rij) *
                           ters_fc(rij, param_ij);
            SADfloat<1> fa_sad = -param_ij[p_B] *
                                 expf(-param_ij[p_lam2] * rij_sad) *
                                 ters_fc_sad(rij_sad, param_ij);
            float f_attr_direct = -0.5f * bij * fa_sad.dval[0] / rij;
            fi.x += f_attr_direct * drij.x;
            fi.y += f_attr_direct * drij.y;
            fi.z += f_attr_direct * drij.z;
            const float f_pair = f_rep + f_attr_direct;
            atomicAdd(&frc[j].x, -f_pair * drij.x);
            atomicAdd(&frc[j].y, -f_pair * drij.y);
            atomicAdd(&frc[j].z, -f_pair * drij.z);
            if constexpr (full_output)
            {
                if (store_energy)
                {
                    float ev = 0.5f * bij * fa_val;
                    en_i += ev;
                    atomicAdd(d_energy_sum, ev);
                }
            }
            if constexpr (full_output)
            {
                if (store_virial)
                {
                    vi = vi +
                         Get_Virial_From_Force_Dis(f_attr_direct * drij, drij);
                }
            }

            float pre = -0.5f * fa_val * ters_bij_d(zeta, param_ij);
            for (int kk = neighbor_begin; kk < neighbor_end; kk++)
            {
                if (jj == kk) continue;
                int k = neighbor_atoms[kk];
                int param_idx_ijk = map[type_i * ntypes * ntypes +
                                        type_j * ntypes + atom_types[k]];
                const float* param_ijk = params + param_idx_ijk * PARAM_STRIDE;
                VECTOR rk = crd[k];
                VECTOR drik = Get_Periodic_Displacement(ri, rk, cell, rcell);
                float rik = norm3df(drik.x, drik.y, drik.z);
                if (rik > param_ijk[p_R] + param_ijk[p_D]) continue;
                float costheta =
                    (drij.x * drik.x + drij.y * drik.y + drij.z * drik.z) /
                    (rij * rik);
                SADfloat<3> s_rij(rij, 0);
                SADfloat<3> s_rik(rik, 1);
                SADfloat<3> s_cos(costheta, 2);
                SADfloat<3> s_arg = param_ijk[p_lam3] * (s_rij - s_rik);
                if (fabsf(param_ijk[p_m] - 3.0f) < 1e-5)
                    s_arg = s_arg * s_arg * s_arg;
                SADfloat<3> s_zeta_ijk = ters_fc_sad(s_rik, param_ijk) *
                                         ters_gijk_sad(s_cos, param_ijk) *
                                         expf(s_arg);

                VECTOR rj_hat = (1.0f / rij) * drij;
                VECTOR rk_hat = (1.0f / rik) * drik;
                VECTOR dcos_dxj = (-1.0f / rij) * (rk_hat - costheta * rj_hat);
                VECTOR dcos_dxk = (-1.0f / rik) * (rj_hat - costheta * rk_hat);

                VECTOR fj_tri;
                fj_tri.x = pre * (-s_zeta_ijk.dval[0] * rj_hat.x +
                                  s_zeta_ijk.dval[2] * dcos_dxj.x);
                fj_tri.y = pre * (-s_zeta_ijk.dval[0] * rj_hat.y +
                                  s_zeta_ijk.dval[2] * dcos_dxj.y);
                fj_tri.z = pre * (-s_zeta_ijk.dval[0] * rj_hat.z +
                                  s_zeta_ijk.dval[2] * dcos_dxj.z);

                VECTOR fk_tri;
                fk_tri.x = pre * (-s_zeta_ijk.dval[1] * rk_hat.x +
                                  s_zeta_ijk.dval[2] * dcos_dxk.x);
                fk_tri.y = pre * (-s_zeta_ijk.dval[1] * rk_hat.y +
                                  s_zeta_ijk.dval[2] * dcos_dxk.y);
                fk_tri.z = pre * (-s_zeta_ijk.dval[1] * rk_hat.z +
                                  s_zeta_ijk.dval[2] * dcos_dxk.z);

                fi.x -= (fj_tri.x + fk_tri.x);
                fi.y -= (fj_tri.y + fk_tri.y);
                fi.z -= (fj_tri.z + fk_tri.z);
                atomicAdd(&frc[j].x, fj_tri.x);
                atomicAdd(&frc[j].y, fj_tri.y);
                atomicAdd(&frc[j].z, fj_tri.z);
                atomicAdd(&frc[k].x, fk_tri.x);
                atomicAdd(&frc[k].y, fk_tri.y);
                atomicAdd(&frc[k].z, fk_tri.z);
                if constexpr (full_output)
                {
                    if (store_virial)
                    {
                        vi = vi - Get_Virial_From_Force_Dis(drij, fj_tri);
                        vi = vi - Get_Virial_From_Force_Dis(drik, fk_tri);
                    }
                }
            }
        }
#ifdef USE_GPU
        Warp_Sum_To(frc + i, fi, warpSize);
#else
        atomicAdd(frc + i, fi);
#endif
        if constexpr (full_output)
        {
            if (store_energy)
            {
#ifdef USE_GPU
                Warp_Sum_To(atom_energy + i, en_i, warpSize);
#else
                atom_energy[i] += en_i;
#endif
            }
            if (store_virial)
            {
#ifdef USE_GPU
                Warp_Sum_To(atom_virial + i, vi, warpSize);
#else
                atomicAdd(atom_virial + i, vi);
#endif
            }
        }
    }
}

void TERSOFF_INFORMATION::Initial(CONTROLLER* controller, int atom_numbers,
                                  const char* module_name)
{
    if (module_name == NULL)
        strcpy(this->module_name, "TERSOFF");
    else
        strcpy(this->module_name, module_name);
    this->atom_numbers = atom_numbers;
    NativeTersoffDefinition native_definition;
    const bool has_native_definition = Read_H5_Tersoff_Input(
        controller, this->module_name, atom_numbers, &native_definition);
    if (has_native_definition)
    {
        controller->printf("START INITIALIZING TERSOFF FORCE FROM NATIVE H5\n");
        atom_type_numbers = native_definition.atom_type_count;
        n_unique_params = static_cast<int>(native_definition.parameters.size() /
                                           PARAM_STRIDE);
        const int map_size =
            atom_type_numbers * atom_type_numbers * atom_type_numbers;
        Malloc_Safely((void**)&h_params,
                      sizeof(float) * native_definition.parameters.size());
        std::copy(native_definition.parameters.begin(),
                  native_definition.parameters.end(), h_params);
        Malloc_Safely((void**)&h_map, sizeof(int) * map_size);
        std::copy(native_definition.type_map.begin(),
                  native_definition.type_map.end(), h_map);
        Malloc_Safely((void**)&h_atom_type, sizeof(int) * atom_numbers);
        std::copy(native_definition.atom_type.begin(),
                  native_definition.atom_type.end(), h_atom_type);
        Device_Malloc_And_Copy_Safely(
            (void**)&d_params, h_params,
            sizeof(float) * native_definition.parameters.size());
        Device_Malloc_And_Copy_Safely((void**)&d_map, h_map,
                                      sizeof(int) * map_size);
        Device_Malloc_Safely((void**)&d_energy_sum, sizeof(float));
        Device_Malloc_And_Copy_Safely((void**)&d_atom_type, h_atom_type,
                                      sizeof(int) * atom_numbers);
        is_initialized = true;
        controller->printf("END INITIALIZING TERSOFF FORCE\n\n");
        return;
    }
    if (!controller->Command_Exist(this->module_name, "in_file")) return;
    controller->printf("START INITIALIZING TERSOFF FORCE\n");
    FILE* fp;
    Open_File_Safely(&fp, controller->Command(this->module_name, "in_file"),
                     "r");
    if (fscanf(fp, "%d %d", &this->atom_numbers, &this->atom_type_numbers) != 2)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "TERSOFF_INFORMATION::Initial",
            "Reason:\n\tThe number of atoms and types can not be found\n");
    }
    std::vector<std::string> type_names(this->atom_type_numbers);
    for (int i = 0; i < this->atom_type_numbers; ++i)
    {
        char name[10];
        if (fscanf(fp, "%s", name) != 1) break;
        type_names[i] = name;
    }

    struct ParamEntry
    {
        std::string e1, e2, e3;
        float params[PARAM_STRIDE];
    };
    std::vector<ParamEntry> entries;
    while (true)
    {
        char e1[10], e2[10], e3[10];
        long current_pos = ftell(fp);
        if (fscanf(fp, "%s", e1) != 1) break;
        if (e1[0] == '#')
        {
            char tmp[1024];
            fgets(tmp, 1024, fp);
            continue;
        }
        // If we hit a number, it's probably the Atom types section
        if (isdigit(e1[0]))
        {
            fseek(fp, current_pos, SEEK_SET);
            break;
        }

        if (fscanf(fp, "%s %s", e2, e3) != 2) break;
        float p[14];
        int n_read = 0;
        for (int k = 0; k < 14; ++k) n_read += fscanf(fp, "%f", &p[k]);
        if (n_read != 14) break;
        ParamEntry entry;
        entry.e1 = e1;
        entry.e2 = e2;
        entry.e3 = e3;
        for (int i = 0; i < 14; ++i) entry.params[i] = p[i];
        entry.params[p_A] *= CONSTANT_EV_TO_KCAL_MOL;
        entry.params[p_B] *= CONSTANT_EV_TO_KCAL_MOL;
        float n = p[p_n];
        if (n > 0)
        {
            entry.params[p_c1] = powf(2.0f * n * 1.0e-16f, -1.0f / n);
            entry.params[p_c2] = powf(2.0f * n * 1.0e-8f, -1.0f / n);
            entry.params[p_c3] = 1.0f / entry.params[p_c2];
            entry.params[p_c4] = 1.0f / entry.params[p_c1];
        }
        else
            entry.params[p_c1] = entry.params[p_c2] = entry.params[p_c3] =
                entry.params[p_c4] = 0;
        entries.push_back(entry);
    }

    this->n_unique_params = entries.size();
    Malloc_Safely((void**)&h_params,
                  sizeof(float) * n_unique_params * PARAM_STRIDE);
    for (int i = 0; i < n_unique_params; ++i)
        for (int j = 0; j < PARAM_STRIDE; ++j)
            h_params[i * PARAM_STRIDE + j] = entries[i].params[j];

    int map_size = atom_type_numbers * atom_type_numbers * atom_type_numbers;
    Malloc_Safely((void**)&h_map, sizeof(int) * map_size);
    for (int i = 0; i < map_size; ++i) h_map[i] = -1;

    for (int i = 0; i < atom_type_numbers; ++i)
    {
        for (int j = 0; j < atom_type_numbers; ++j)
        {
            for (int k = 0; k < atom_type_numbers; ++k)
            {
                std::string e1 = type_names[i];
                std::string e2 = type_names[j];
                std::string e3 = type_names[k];
                for (int p = 0; p < n_unique_params; ++p)
                {
                    if (entries[p].e1 == e1 && entries[p].e2 == e2 &&
                        entries[p].e3 == e3)
                    {
                        h_map[i * atom_type_numbers * atom_type_numbers +
                              j * atom_type_numbers + k] = p;
                        break;
                    }
                }
            }
        }
    }

    const int center_cutoff_count = atom_type_numbers * atom_type_numbers;
    Malloc_Safely(reinterpret_cast<void**>(&h_center_cutoffs),
                  sizeof(float) * static_cast<size_t>(center_cutoff_count));
    for (int i = 0; i < center_cutoff_count; i += 1)
    {
        h_center_cutoffs[i] = 0.0f;
    }
    cut = 0.0f;
    for (int type_i = 0; type_i < atom_type_numbers; type_i += 1)
    {
        for (int type_k = 0; type_k < atom_type_numbers; type_k += 1)
        {
            float center_cutoff = 0.0f;
            for (int type_j = 0; type_j < atom_type_numbers; type_j += 1)
            {
                const int param_index =
                    h_map[type_i * atom_type_numbers * atom_type_numbers +
                          type_j * atom_type_numbers + type_k];
                if (param_index < 0) continue;
                const float* param = h_params + param_index * PARAM_STRIDE;
                center_cutoff = fmaxf(center_cutoff, param[p_R] + param[p_D]);
            }
            h_center_cutoffs[type_i * atom_type_numbers + type_k] =
                center_cutoff;
            cut = fmaxf(cut, center_cutoff);
        }
    }

    Device_Malloc_And_Copy_Safely(
        (void**)&d_params, h_params,
        sizeof(float) * n_unique_params * PARAM_STRIDE);
    Device_Malloc_And_Copy_Safely((void**)&d_map, h_map,
                                  sizeof(int) * map_size);
    Device_Malloc_And_Copy_Safely(
        reinterpret_cast<void**>(&d_center_cutoffs), h_center_cutoffs,
        sizeof(float) * static_cast<size_t>(center_cutoff_count));
    Device_Malloc_Safely((void**)&d_energy_sum, sizeof(float));

    Malloc_Safely((void**)&h_atom_type, sizeof(int) * atom_numbers);
    for (int i = 0; i < atom_numbers; i++)
    {
        if (fscanf(fp, "%d", h_atom_type + i) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "TERSOFF_INFORMATION::Initial",
                "Reason:\n\tSome atom types can not be found\n");
        }
    }
    Device_Malloc_And_Copy_Safely((void**)&d_atom_type, h_atom_type,
                                  sizeof(int) * atom_numbers);

    fclose(fp);
    is_initialized = true;
    controller->printf("END INITIALIZING TERSOFF FORCE\n\n");
}

bool TERSOFF_INFORMATION::TERSOFF_Force_Clustered(
    const CLUSTERED_SPATIAL_VIEW& view, const VECTOR* crd, VECTOR* frc,
    const LTMatrix3 cell, const LTMatrix3 rcell, const int need_atom_energy,
    float* atom_energy, const int need_virial, LTMatrix3* atom_virial,
    const char** failure_reason)
{
    if (failure_reason != NULL) *failure_reason = NULL;
    if (!is_initialized) return true;
    if (crd == NULL || frc == NULL ||
        (need_atom_energy && atom_energy == NULL) ||
        (need_virial && atom_virial == NULL))
    {
        if (failure_reason != NULL)
            *failure_reason = "Tersoff clustered force received null buffers";
        return false;
    }
    if (!Tersoff_Ensure_Clustered_Center_Atoms(this, view, crd, cell))
    {
        if (failure_reason != NULL)
            *failure_reason =
                "Tersoff could not derive compact center-neighbor relation";
        return false;
    }
    if (need_atom_energy)
    {
        deviceMemset(d_energy_sum, 0, sizeof(float));
    }
#ifdef USE_GPU
    dim3 blockSize(static_cast<unsigned int>(CONTROLLER::device_warp),
                   static_cast<unsigned int>(CONTROLLER::device_max_thread /
                                             CONTROLLER::device_warp));
    dim3 gridSize(static_cast<unsigned int>(
        (atom_numbers + static_cast<int>(blockSize.y) - 1) /
        static_cast<int>(blockSize.y)));
#else
    dim3 blockSize(128);
    dim3 gridSize((atom_numbers + blockSize.x - 1) / blockSize.x);
#endif

    auto force_kernel = Tersoff_Force_CUDA<false>;
    if (need_atom_energy || need_virial)
    {
        force_kernel = Tersoff_Force_CUDA<true>;
    }

    Launch_Device_Kernel(
        force_kernel, gridSize, blockSize, 0, NULL, atom_numbers, crd, frc,
        cell, rcell, clustered_neighbors.offsets, clustered_neighbors.items,
        d_atom_type, d_params, d_map, atom_type_numbers, atom_energy,
        atom_virial, d_energy_sum, need_atom_energy != 0, need_virial != 0);
    return true;
}

void TERSOFF_INFORMATION::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    deviceMemcpy(&h_energy_sum, d_energy_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
    controller->Step_Print(this->module_name, h_energy_sum, true);
}
