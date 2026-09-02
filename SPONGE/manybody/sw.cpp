#include "sw.h"

#include <algorithm>
#include <highfive/highfive.hpp>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
template <typename T>
hid_t Native_H5_Type();

template <>
hid_t Native_H5_Type<int>()
{
    return H5T_NATIVE_INT;
}

template <>
hid_t Native_H5_Type<float>()
{
    return H5T_NATIVE_FLOAT;
}

template <typename T>
std::vector<T> Read_H5_Array(HighFive::File* file,
                             const std::string& dataset_path,
                             std::size_t expected_rank,
                             std::size_t expected_columns,
                             const std::string& label)
{
    HighFive::DataSet dataset = file->getDataSet(dataset_path);
    const auto dimensions = dataset.getSpace().getDimensions();
    if (dimensions.size() != expected_rank ||
        (expected_rank == 2 && dimensions[1] != expected_columns))
    {
        std::ostringstream message;
        message << label << " dataset " << dataset_path << " must have rank "
                << expected_rank;
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
    if (H5Dread(dataset.getId(), Native_H5_Type<T>(), H5S_ALL, H5S_ALL,
                H5P_DEFAULT, values.data()) < 0)
    {
        throw std::runtime_error("failed to read " + label + " dataset " +
                                 dataset_path);
    }
    return values;
}

struct NativeSWDefinition
{
    int atom_type_count = 0;
    std::vector<int> atom_type;
    std::vector<float> pair_parameters;
    std::vector<float> triple_parameters;
};

bool Read_H5_SW_Input(CONTROLLER* controller, const char* module_name,
                      NativeSWDefinition* definition)
{
    constexpr const char* input_key = "input_h5_topology_path";
    constexpr const char* sw_root = "/manybody/sw";
    if (strcmp(module_name, "SW") != 0 ||
        controller->Command_Exist(module_name, "in_file") ||
        !controller->Command_Exist(input_key))
    {
        return false;
    }

    try
    {
        HighFive::File file(controller->Command(input_key),
                            HighFive::File::ReadOnly);
        if (!file.exist(sw_root))
        {
            return false;
        }

        int atom_type_count = 0;
        file.getDataSet(std::string(sw_root) + "/atom_type_count")
            .read(atom_type_count);
        if (atom_type_count <= 0)
        {
            throw std::runtime_error(
                "/manybody/sw/atom_type_count must be positive");
        }
        const auto atom_type = Read_H5_Array<int>(
            &file, std::string(sw_root) + "/atom_type", 1, 0, "SW atom type");
        const auto pair_type = Read_H5_Array<int>(
            &file, std::string(sw_root) + "/pair/type", 2, 2, "SW pair type");
        const auto pair_parameter = Read_H5_Array<float>(
            &file, std::string(sw_root) + "/pair/parameters", 2, 8,
            "SW pair parameters");
        const auto triple_type =
            Read_H5_Array<int>(&file, std::string(sw_root) + "/triple/type", 2,
                               3, "SW triple type");
        const auto triple_parameter = Read_H5_Array<float>(
            &file, std::string(sw_root) + "/triple/parameters", 2, 3,
            "SW triple parameters");
        if (atom_type.empty())
        {
            throw std::runtime_error(
                "/manybody/sw/atom_type must not be empty");
        }
        for (const int type : atom_type)
        {
            if (type < 0 || type >= atom_type_count)
            {
                throw std::runtime_error(
                    "/manybody/sw/atom_type contains an out-of-range type");
            }
        }

        const std::size_t pair_row_count = pair_type.size() / 2;
        if (pair_parameter.size() / 8 != pair_row_count)
        {
            throw std::runtime_error(
                "/manybody/sw pair type/parameter row count mismatch");
        }
        const std::size_t full_pair_count =
            static_cast<std::size_t>(atom_type_count) * atom_type_count;
        const std::size_t triangular_pair_count =
            static_cast<std::size_t>(atom_type_count) * (atom_type_count + 1) /
            2;
        if (pair_row_count != full_pair_count &&
            pair_row_count != triangular_pair_count)
        {
            throw std::runtime_error(
                "/manybody/sw pair row count must be atom_type_count^2 or "
                "triangular");
        }
        std::map<std::pair<int, int>, std::vector<float>> pair_rows;
        for (std::size_t row = 0; row < pair_row_count; ++row)
        {
            const int a = pair_type[2 * row];
            const int b = pair_type[2 * row + 1];
            if (a < 0 || a >= atom_type_count || b < 0 || b >= atom_type_count)
            {
                throw std::runtime_error(
                    "/manybody/sw/pair/type contains an out-of-range type");
            }
            if (pair_rows.count({a, b}) != 0)
            {
                throw std::runtime_error(
                    "/manybody/sw/pair/type contains a duplicate row");
            }
            const std::vector<float> parameters(
                pair_parameter.begin() + 8 * row,
                pair_parameter.begin() + 8 * row + 8);
            pair_rows[{a, b}] = parameters;
            if (pair_row_count == triangular_pair_count)
            {
                pair_rows[{b, a}] = parameters;
            }
        }

        const std::size_t triple_row_count = triple_type.size() / 3;
        const std::size_t full_triple_count =
            full_pair_count * static_cast<std::size_t>(atom_type_count);
        if (triple_parameter.size() / 3 != triple_row_count ||
            triple_row_count != full_triple_count)
        {
            throw std::runtime_error(
                "/manybody/sw triple payload must contain atom_type_count^3 "
                "matching rows");
        }
        std::vector<float> canonical_triples(full_triple_count * 3);
        std::vector<bool> triple_recorded(full_triple_count, false);
        for (std::size_t row = 0; row < triple_row_count; ++row)
        {
            const int a = triple_type[3 * row];
            const int b = triple_type[3 * row + 1];
            const int c = triple_type[3 * row + 2];
            if (a < 0 || a >= atom_type_count || b < 0 ||
                b >= atom_type_count || c < 0 || c >= atom_type_count)
            {
                throw std::runtime_error(
                    "/manybody/sw/triple/type contains an out-of-range type");
            }
            const std::size_t index =
                static_cast<std::size_t>(a) * atom_type_count *
                    atom_type_count +
                static_cast<std::size_t>(b) * atom_type_count + c;
            if (triple_recorded[index])
            {
                throw std::runtime_error(
                    "/manybody/sw/triple/type contains a duplicate row");
            }
            triple_recorded[index] = true;
            for (std::size_t parameter = 0; parameter < 3; ++parameter)
            {
                canonical_triples[3 * index + parameter] =
                    triple_parameter[3 * row + parameter];
            }
        }

        NativeSWDefinition result;
        result.atom_type_count = atom_type_count;
        result.atom_type = atom_type;
        result.pair_parameters.resize(full_pair_count * 8);
        for (int a = 0; a < atom_type_count; ++a)
        {
            for (int b = 0; b < atom_type_count; ++b)
            {
                const auto found = pair_rows.find({a, b});
                if (found == pair_rows.end())
                {
                    throw std::runtime_error(
                        "missing /manybody/sw pair parameters for type " +
                        std::to_string(a) + "," + std::to_string(b));
                }
                const std::size_t index =
                    static_cast<std::size_t>(a) * atom_type_count + b;
                std::copy(found->second.begin(), found->second.end(),
                          result.pair_parameters.begin() + 8 * index);
            }
        }
        if (std::find(triple_recorded.begin(), triple_recorded.end(), false) !=
            triple_recorded.end())
        {
            throw std::runtime_error(
                "/manybody/sw triple payload does not cover all type triples");
        }
        result.triple_parameters = std::move(canonical_triples);
        *definition = std::move(result);
        return true;
    }
    catch (const std::exception& error)
    {
        const std::string message =
            std::string("Reason:\n\tfailed to read typed SW input from ") +
            sw_root + ": " + error.what() + "\n";
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "Read_H5_SW_Input", message.c_str());
    }
    return false;
}
}  // namespace

#ifdef USE_CPU
#include "clustered_gmxpacked_cpu.h"
#endif

void STILLINGER_WEBER_INFORMATION::Initial(CONTROLLER* controller,
                                           const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "SW");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    NativeSWDefinition native_definition;
    const bool has_native_definition =
        Read_H5_SW_Input(controller, this->module_name, &native_definition);
    if (has_native_definition)
    {
        controller->printf(
            "START INITIALIZING STILLINGER WEBER FORCE FROM NATIVE H5\n");
        atom_numbers = static_cast<int>(native_definition.atom_type.size());
        atom_type_numbers = native_definition.atom_type_count;
        pair_type_numbers = atom_type_numbers * atom_type_numbers;
        triple_type_numbers =
            atom_type_numbers * atom_type_numbers * atom_type_numbers;
        Malloc_Safely((void**)&h_energy_atom, sizeof(float) * atom_numbers);
        Device_Malloc_And_Copy_Safely((void**)&d_energy_sum, h_energy_atom,
                                      sizeof(float) * (atom_numbers + 1));
        d_energy_atom = d_energy_sum + 1;
        const std::size_t parameter_count =
            static_cast<std::size_t>(pair_type_numbers) * 8 +
            static_cast<std::size_t>(triple_type_numbers) * 3;
        Malloc_Safely((void**)&h_parameters, sizeof(float) * parameter_count);
        Malloc_Safely((void**)&h_atom_type, sizeof(int) * atom_numbers);
        std::copy(native_definition.pair_parameters.begin(),
                  native_definition.pair_parameters.end(), h_parameters);
        std::copy(native_definition.triple_parameters.begin(),
                  native_definition.triple_parameters.end(),
                  h_parameters + pair_type_numbers * 8);
        std::copy(native_definition.atom_type.begin(),
                  native_definition.atom_type.end(), h_atom_type);
        Device_Malloc_And_Copy_Safely((void**)&d_parameters, h_parameters,
                                      sizeof(float) * parameter_count);
        Device_Malloc_And_Copy_Safely((void**)&d_atom_type, h_atom_type,
                                      sizeof(int) * atom_numbers);
        is_initialized = true;
        if (!is_controller_printf_initialized)
        {
            controller->Step_Print_Initial(this->module_name, "%.2f");
            is_controller_printf_initialized = true;
        }
        controller->printf("END INITIALIZING STILLINGER WEBER FORCE\n\n");
        return;
    }
    if (!controller->Command_Exist(this->module_name, "in_file"))
    {
        controller->printf("STILLINGER WEBER FORCE IS NOT INITIALIZED\n\n");
        return;
    }
    controller->printf("START INITIALIZING STILLINGER WEBER FORCE\n");
    FILE* fp;
    Open_File_Safely(&fp, controller->Command(this->module_name, "in_file"),
                     "r");
    if (fscanf(fp, "%d %d\n", &atom_numbers, &atom_type_numbers) != 2)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "STILLINGER_WEBER_INFORMATION::Initial",
            "Reason:\n\tThe number of atoms and SW types can not be found\n");
    }
    pair_type_numbers = atom_type_numbers * atom_type_numbers;
    triple_type_numbers =
        atom_type_numbers * atom_type_numbers * atom_type_numbers;
    Malloc_Safely((void**)&h_energy_atom, sizeof(float) * atom_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_energy_sum, h_energy_atom,
                                  sizeof(float) * (atom_numbers + 1));
    d_energy_atom = d_energy_sum + 1;
    Malloc_Safely(
        (void**)&h_parameters,
        sizeof(float) * (pair_type_numbers * 8 + triple_type_numbers * 3));
    Malloc_Safely((void**)&h_atom_type, sizeof(int) * atom_numbers);
    char temp[CHAR_LENGTH_MAX];
    std::map<int, bool> unrecorded;
    int type_a, type_b, type_c;
    float A, B, sigma, p, q, a, gamma;
    float lambda, epsilon, b;
    if (fgets(temp, CHAR_LENGTH_MAX, fp) == NULL || strlen(temp) < 1 ||
        temp[0] != '#')
    {
        printf("'%s'\n", temp);
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "STILLINGER_WEBER_INFORMATION::Initial",
            "Reason:\n\tThe first comment line can not be found\n");
    }
    for (int i = 0; i < pair_type_numbers; i++)
    {
        unrecorded[i] = true;
    }
    for (int i = 0; i < pair_type_numbers; i++)
    {
        if (fscanf(fp, "%d %d %f %f %f %f %f %f %f %f\n", &type_a, &type_b, &A,
                   &B, &epsilon, &p, &q, &a, &gamma, &sigma) != 10)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "STILLINGER_WEBER_INFORMATION::Initial",
                "Reason:\n\tSome twobody parameters can not be found\n");
        }
        int index = type_a * atom_type_numbers + type_b;
        if (index >= pair_type_numbers)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "STILLINGER_WEBER_INFORMATION::Initial",
                "Reason:\n\tSome twobody type indexes are not right\n");
        }
        unrecorded[index] = false;
        h_parameters[8 * index + 0] = A;
        h_parameters[8 * index + 1] = B;
        h_parameters[8 * index + 2] = epsilon;
        h_parameters[8 * index + 3] = p;
        h_parameters[8 * index + 4] = q;
        h_parameters[8 * index + 5] = a;
        h_parameters[8 * index + 6] = gamma;
        h_parameters[8 * index + 7] = sigma;
        cut = fmaxf(cut, a * sigma);
    }
    for (int i = 0; i < pair_type_numbers; i++)
    {
        if (unrecorded[i])
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "STILLINGER_WEBER_INFORMATION::Initial",
                "Reason:\n\tSome twobody parameters can not be found\n");
        }
    }
    if (fgets(temp, CHAR_LENGTH_MAX, fp) == NULL || strlen(temp) < 1 ||
        temp[0] != '#')
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "STILLINGER_WEBER_INFORMATION::Initial",
            "Reason:\n\tThe second comment line can not be found\n");
    }
    for (int i = 0; i < triple_type_numbers; i++)
    {
        unrecorded[i] = true;
    }
    for (int i = 0; i < triple_type_numbers; i++)
    {
        if (fscanf(fp, "%d %d %d %f %f %f\n", &type_a, &type_b, &type_c,
                   &lambda, &epsilon, &b) != 6)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "STILLINGER_WEBER_INFORMATION::Initial",
                "Reason:\n\tSome threebody parameters can not be found\n");
        }
        int index = type_a * atom_type_numbers * atom_type_numbers +
                    type_b * atom_type_numbers + type_c;
        if (index >= triple_type_numbers)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "STILLINGER_WEBER_INFORMATION::Initial",
                "Reason:\n\tSome threebody type indexes are not right\n");
        }
        unrecorded[index] = false;
        h_parameters[8 * pair_type_numbers + 3 * index + 0] = lambda;
        h_parameters[8 * pair_type_numbers + 3 * index + 1] = epsilon;
        h_parameters[8 * pair_type_numbers + 3 * index + 2] = b;
    }
    for (int i = 0; i < triple_type_numbers; i++)
    {
        if (unrecorded[i])
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "STILLINGER_WEBER_INFORMATION::Initial",
                "Reason:\n\tSome threebody parameters can not be found\n");
        }
    }
    Device_Malloc_And_Copy_Safely(
        (void**)&d_parameters, h_parameters,
        sizeof(float) * (pair_type_numbers * 8 + triple_type_numbers * 3));
    if (fgets(temp, CHAR_LENGTH_MAX, fp) == NULL || strlen(temp) < 1 ||
        temp[0] != '#')
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "STILLINGER_WEBER_INFORMATION::Initial",
            "Reason:\n\tThe third comment line can not be found\n");
    }
    for (int i = 0; i < atom_numbers; i++)
    {
        if (fscanf(fp, "%d", h_atom_type + i) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "STILLINGER_WEBER_INFORMATION::Initial",
                "Reason:\n\tSome atom types can not be found\n");
        }
        if (h_atom_type[i] >= atom_type_numbers)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "STILLINGER_WEBER_INFORMATION::Initial",
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
    controller->printf("END INITIALIZING STILLINGER WEBER FORCE\n\n");
}

struct SW_CLUSTERED_NEIGHBOR
{
    int atom = -1;
    VECTOR displacement = {};
    float distance = 0.0f;
};

constexpr int kSWClusteredCachedNeighborCapacity = 64;
constexpr int kSWClusteredWarpsPerBlock = 16;

struct SW_CLUSTERED_CENTER_NEIGHBOR_CURSOR
{
    CLUSTERED_GMXPACKED_CENTER_CURSOR center = {};
    CLUSTERED_GMXPACKED_CENTER_TILE tile = {};
    bool has_tile = false;
    int neighbor_cluster_offset = 0;
    int neighbor_lane = 0;
};

static __host__ __device__ __forceinline__ int
SW_Clustered_Find_Cluster(const CLUSTERED_SPATIAL_VIEW& view,
                          const int sorted_atom)
{
    int low = 0;
    int high = view.cluster_numbers;
    while (low + 1 < high)
    {
        const int middle = (low + high) >> 1;
        if (view.cluster_offsets[middle] <= sorted_atom)
            low = middle;
        else
            high = middle;
    }
    return low;
}

static __host__ __device__ __forceinline__ bool
SW_Clustered_Neighbor_Within_Cut(
    const int type_i, const int atom_neighbor, const float distance,
    const int* atom_types, const float* parameters,
    const int atom_type_numbers, const float margin = 0.0f)
{
    const int type_neighbor = atom_types[atom_neighbor];
    const int pair_index = type_i * atom_type_numbers + type_neighbor;
    const float a = parameters[8 * pair_index + 5];
    const float sigma = parameters[8 * pair_index + 7];
    return distance < a * sigma + margin;
}

static __host__ __device__ __forceinline__ float
SW_Two_Body_Analytic(
    const float distance, const float A, const float B,
    const float epsilon, const float p, const float q,
    const float a, const float sigma, float* radial_derivative)
{
    const float reduced_distance = distance / sigma;
    const float power_p = powf(reduced_distance, -p);
    const float power_q = powf(reduced_distance, -q);
    const float shape = B * power_p - power_q;
    const float inverse_gap = 1.0f / (reduced_distance - a);
    const float exponential = expf(inverse_gap);
    const float shape_derivative =
        (-p * B * power_p + q * power_q) / reduced_distance;
    *radial_derivative =
        A * epsilon * exponential *
        (shape_derivative - shape * inverse_gap * inverse_gap) /
        sigma;
    return A * epsilon * shape * exponential;
}

static __host__ __device__ __forceinline__ float
SW_Three_Body_Analytic(
    const VECTOR drij, const VECTOR drik, const float rij, const float rik,
    const float sigma1, const float a1, const float gamma1,
    const float sigma2, const float a2, const float gamma2,
    const float lambda, const float epsilon, const float b,
    VECTOR* j_force, VECTOR* k_force)
{
    const float inv_rij = 1.0f / rij;
    const float inv_rik = 1.0f / rik;
    const float inv_rij_sq = inv_rij * inv_rij;
    const float inv_rik_sq = inv_rik * inv_rik;
    const float dot =
        drij.x * drik.x + drij.y * drik.y + drij.z * drik.z;
    const float cosine = dot * inv_rij * inv_rik;
    const float angular = cosine - b;
    const float reduced_rij = rij / sigma1 - a1;
    const float reduced_rik = rik / sigma2 - a2;
    const float exponential1 = expf(gamma1 / reduced_rij);
    const float exponential2 = expf(gamma2 / reduced_rik);
    const float prefactor =
        lambda * epsilon * exponential1 * exponential2;
    const float angular_gradient = 2.0f * prefactor * angular;
    const float radial_gradient_j =
        -prefactor * angular * angular * gamma1 /
        (sigma1 * reduced_rij * reduced_rij * rij);
    const float radial_gradient_k =
        -prefactor * angular * angular * gamma2 /
        (sigma2 * reduced_rik * reduced_rik * rik);
    const float cross_j = inv_rij * inv_rik;
    const float self_j = cosine * inv_rij_sq;
    const float self_k = cosine * inv_rik_sq;
    *j_force = {
        angular_gradient * (drik.x * cross_j - drij.x * self_j) +
            radial_gradient_j * drij.x,
        angular_gradient * (drik.y * cross_j - drij.y * self_j) +
            radial_gradient_j * drij.y,
        angular_gradient * (drik.z * cross_j - drij.z * self_j) +
            radial_gradient_j * drij.z};
    *k_force = {
        angular_gradient * (drij.x * cross_j - drik.x * self_k) +
            radial_gradient_k * drik.x,
        angular_gradient * (drij.y * cross_j - drik.y * self_k) +
            radial_gradient_k * drik.y,
        angular_gradient * (drij.z * cross_j - drik.z * self_k) +
            radial_gradient_k * drik.z};
    return prefactor * angular * angular;
}

static __host__ __device__ __forceinline__ bool
SW_Clustered_Center_Neighbor_Cursor_Begin(
    const CLUSTERED_SPATIAL_VIEW& view, const int center_cluster,
    SW_CLUSTERED_CENTER_NEIGHBOR_CURSOR* cursor)
{
    if (cursor == NULL)
    {
        return false;
    }
    *cursor = {};
    if (!Clustered_Gmxpacked_Center_Cursor_Begin(
            view, center_cluster, &cursor->center))
    {
        return false;
    }
    cursor->has_tile = false;
    return true;
}

static __host__ __device__ __forceinline__ bool
SW_Clustered_Center_Tile_Atom(
    const CLUSTERED_SPATIAL_VIEW& view,
    const CLUSTERED_GMXPACKED_CENTER_TILE& tile,
    const int center_lane, const int center_atom,
    const int neighbor_cluster_offset, const int neighbor_lane,
    const bool require_pair_shift_active, int* atom_neighbor)
{
    if (atom_neighbor == NULL || center_lane < 0 ||
        center_lane >= view.cluster_size || neighbor_lane < 0 ||
        neighbor_lane >= view.cluster_size ||
        neighbor_cluster_offset < 0 ||
        neighbor_cluster_offset >= kClusteredSuperClusterClusters ||
        (tile.neighbor_cluster_mask &
         (1u << static_cast<unsigned int>(
              neighbor_cluster_offset))) == 0u)
    {
        return false;
    }
    const int neighbor_cluster =
        tile.neighbor_cluster_base + neighbor_cluster_offset;
    if (neighbor_cluster < 0 ||
        neighbor_cluster >= view.cluster_numbers)
    {
        return false;
    }
    const unsigned int neighbor_lane_bit =
        1u << static_cast<unsigned int>(neighbor_lane);
    if ((view.cluster_valid_masks[neighbor_cluster] &
         neighbor_lane_bit) == 0u ||
        (view.cluster_local_masks[neighbor_cluster] &
         neighbor_lane_bit) == 0u)
    {
        return false;
    }
    const CLUSTERED_GMXPACKED_CJ& packed =
        view.gmxpacked_cjpacked[tile.cjpacked_id];
    int original_i_local = tile.original_i_local;
    int original_i_lane = center_lane;
    int original_j_lane = neighbor_lane;
    if (tile.orientation ==
        CLUSTERED_ENDPOINT_ORIENTATION::TRANSPOSED_J)
    {
        original_i_local = neighbor_cluster_offset;
        original_i_lane = neighbor_lane;
        original_j_lane = center_lane;
    }
    const int split =
        original_j_lane / kClusteredSplitJClusterSize;
    const int split_j_lane =
        original_j_lane - split * kClusteredSplitJClusterSize;
    const CLUSTERED_GMXPACKED_SPLIT& split_entry =
        packed.split[split];
    const unsigned int packed_bit =
        1u << (static_cast<int>(tile.jm) *
                   kClusteredSuperClusterClusters +
               original_i_local);
    if ((split_entry.imask & packed_bit) == 0u)
    {
        return false;
    }
    if (require_pair_shift_active &&
        view.pair_shift_metadata_ready &&
        view.pair_shift_bits != NULL)
    {
        const uint64_t shift_bits =
            view.pair_shift_bits[
                tile.cjpacked_id * kClusteredJGroupSize + tile.jm];
        if ((Clustered_Get_Pair_Active_I_Mask(
                 shift_bits, split) &
             (1u << static_cast<unsigned int>(
                  original_i_local))) == 0u)
        {
            return false;
        }
    }
    if (split_entry.exclusion_index != 0)
    {
        const unsigned int pair_bits =
            view.gmxpacked_exclusions[split_entry.exclusion_index]
                .pair[split_j_lane * kClusteredClusterSize +
                      original_i_lane];
        if ((pair_bits & packed_bit) == 0u)
        {
            return false;
        }
    }
    const int sorted_neighbor =
        view.cluster_offsets[neighbor_cluster] + neighbor_lane;
    *atom_neighbor = view.sort_permutation[sorted_neighbor];
    return *atom_neighbor != center_atom;
}

static __host__ __device__ __forceinline__ bool
SW_Clustered_Center_Tile_Neighbor(
    const CLUSTERED_SPATIAL_VIEW& view,
    const CLUSTERED_GMXPACKED_CENTER_TILE& tile,
    const int center_lane, const int center_atom,
    const int neighbor_cluster_offset, const int neighbor_lane,
    const VECTOR* crd, const LTMatrix3 cell,
    SW_CLUSTERED_NEIGHBOR* neighbor)
{
    if (neighbor == NULL)
    {
        return false;
    }
    int atom_neighbor = -1;
    if (!SW_Clustered_Center_Tile_Atom(
            view, tile, center_lane, center_atom,
            neighbor_cluster_offset, neighbor_lane, true,
            &atom_neighbor))
    {
        return false;
    }
    const int shift_id =
        Clustered_Gmxpacked_Center_Tile_Pair_Shift_Id(
            view, tile, neighbor_cluster_offset);
    if (shift_id < 0)
    {
        return false;
    }
    const VECTOR shift =
        Clustered_Shift_Vector_From_Id(shift_id, cell);
    const VECTOR displacement =
        (crd[center_atom] - crd[atom_neighbor]) + shift;
    neighbor->atom = atom_neighbor;
    neighbor->displacement = displacement;
    neighbor->distance =
        sqrtf(displacement.x * displacement.x +
              displacement.y * displacement.y +
              displacement.z * displacement.z);
    return true;
}

static __host__ __device__ __forceinline__ bool
SW_Clustered_Center_Neighbor_Next(
    const CLUSTERED_SPATIAL_VIEW& view, const int center_lane,
    const int center_atom, SW_CLUSTERED_CENTER_NEIGHBOR_CURSOR* cursor,
    const VECTOR* crd, const LTMatrix3 cell,
    SW_CLUSTERED_NEIGHBOR* neighbor)
{
    if (neighbor == NULL || cursor == NULL || center_lane < 0 ||
        center_lane >= view.cluster_size)
    {
        return false;
    }
    while (true)
    {
        if (!cursor->has_tile)
        {
            if (!Clustered_Gmxpacked_Center_Cursor_Next(
                    view, &cursor->center, &cursor->tile))
            {
                return false;
            }
            cursor->has_tile = true;
            cursor->neighbor_cluster_offset = 0;
            cursor->neighbor_lane = 0;
        }
        const CLUSTERED_GMXPACKED_CENTER_TILE& tile = cursor->tile;
        for (int neighbor_cluster_offset =
                 cursor->neighbor_cluster_offset;
             neighbor_cluster_offset < kClusteredSuperClusterClusters;
             neighbor_cluster_offset += 1, cursor->neighbor_lane = 0)
        {
            cursor->neighbor_cluster_offset =
                neighbor_cluster_offset;
            if ((tile.neighbor_cluster_mask &
                 (1u << static_cast<unsigned int>(
                      neighbor_cluster_offset))) == 0u)
            {
                continue;
            }
            while (cursor->neighbor_lane < view.cluster_size)
            {
                const int neighbor_lane = cursor->neighbor_lane;
                cursor->neighbor_lane = neighbor_lane + 1;
                if (SW_Clustered_Center_Tile_Neighbor(
                        view, tile, center_lane, center_atom,
                        neighbor_cluster_offset, neighbor_lane,
                        crd, cell, neighbor))
                {
                    return true;
                }
            }
            cursor->neighbor_cluster_offset = neighbor_cluster_offset + 1;
        }
        cursor->has_tile = false;
    }
}

template <typename T>
static void SW_Reserve_Clustered_Neighbor_Buffer(
    T** pointer, int* capacity, const int required)
{
    if (required <= *capacity && *pointer != NULL)
    {
        return;
    }
    Free_Single_Device_Pointer(
        reinterpret_cast<void**>(pointer));
    *capacity = 0;
    if (required > 0)
    {
        Device_Malloc_Safely(
            reinterpret_cast<void**>(pointer),
            sizeof(T) * static_cast<size_t>(required));
        *capacity = required;
    }
}

template <bool fill>
static __global__ void SW_Build_Clustered_Center_Atoms(
    const CLUSTERED_SPATIAL_VIEW view, const int atom_numbers,
    const VECTOR* crd, const LTMatrix3 cell, const int* atom_types,
    const float* parameters, const int atom_type_numbers,
    const int* neighbor_offsets, int* neighbor_atoms,
    int* neighbor_counts)
{
#ifdef USE_GPU
    const int sorted_i =
        static_cast<int>(threadIdx.y) +
        static_cast<int>(blockDim.y * blockIdx.x);
    if (sorted_i < atom_numbers)
#else
#pragma omp parallel for
    for (int sorted_i = 0; sorted_i < atom_numbers; sorted_i += 1)
#endif
    {
        const int center_cluster =
            SW_Clustered_Find_Cluster(view, sorted_i);
        const int center_lane =
            sorted_i - view.cluster_offsets[center_cluster];
        const int atom_i = view.sort_permutation[sorted_i];
        const int type_i = atom_types[atom_i];
#ifdef USE_GPU
        if (threadIdx.x == 0)
        {
            neighbor_counts[sorted_i] = 0;
        }
        __syncwarp();
        CLUSTERED_GMXPACKED_CENTER_CURSOR fill_center = {};
        int fill_has_range = 0;
        if (threadIdx.x == 0)
        {
            fill_has_range =
                Clustered_Gmxpacked_Center_Cursor_Begin(
                    view, center_cluster, &fill_center)
                    ? 1
                    : 0;
        }
        fill_has_range = deviceShfl(
            FULL_MASK, fill_has_range, 0, warpSize);
        fill_center.center_cluster = deviceShfl(
            FULL_MASK, fill_center.center_cluster, 0, warpSize);
        fill_center.center_supercluster = deviceShfl(
            FULL_MASK, fill_center.center_supercluster, 0, warpSize);
        fill_center.center_i_local = deviceShfl(
            FULL_MASK, fill_center.center_i_local, 0, warpSize);
        fill_center.next_reference = deviceShfl(
            FULL_MASK, fill_center.next_reference, 0, warpSize);
        fill_center.end_reference = deviceShfl(
            FULL_MASK, fill_center.end_reference, 0, warpSize);
        if (fill_has_range)
        {
            for (int reference_index = fill_center.next_reference;
                 reference_index < fill_center.end_reference;
                 reference_index += 1)
            {
                CLUSTERED_GMXPACKED_CENTER_TILE tile = {};
                int has_tile = 0;
                if (threadIdx.x == 0)
                {
                    CLUSTERED_GMXPACKED_CENTER_CURSOR
                        reference_cursor = fill_center;
                    reference_cursor.next_reference =
                        reference_index;
                    reference_cursor.end_reference =
                        reference_index + 1;
                    has_tile =
                        Clustered_Gmxpacked_Center_Cursor_Next(
                            view, &reference_cursor, &tile)
                            ? 1
                            : 0;
                }
                has_tile = deviceShfl(
                    FULL_MASK, has_tile, 0, warpSize);
                tile.sci_id = deviceShfl(
                    FULL_MASK, tile.sci_id, 0, warpSize);
                tile.cjpacked_id = deviceShfl(
                    FULL_MASK, tile.cjpacked_id, 0, warpSize);
                tile.center_cluster = deviceShfl(
                    FULL_MASK, tile.center_cluster, 0, warpSize);
                tile.neighbor_cluster_base = deviceShfl(
                    FULL_MASK, tile.neighbor_cluster_base, 0,
                    warpSize);
                tile.neighbor_cluster_mask =
                    deviceShfl(
                        FULL_MASK, tile.neighbor_cluster_mask,
                        0, warpSize);
                const int tile_jm = deviceShfl(
                    FULL_MASK, static_cast<int>(tile.jm), 0,
                    warpSize);
                const int tile_original_i_local = deviceShfl(
                    FULL_MASK,
                    static_cast<int>(tile.original_i_local), 0,
                    warpSize);
                const int tile_orientation = deviceShfl(
                    FULL_MASK,
                    static_cast<int>(tile.orientation), 0,
                    warpSize);
                tile.jm = static_cast<unsigned char>(tile_jm);
                tile.original_i_local =
                    static_cast<unsigned char>(
                        tile_original_i_local);
                tile.orientation =
                    static_cast<CLUSTERED_ENDPOINT_ORIENTATION>(
                        tile_orientation);
                if (!has_tile)
                {
                    continue;
                }
                for (int candidate =
                         static_cast<int>(threadIdx.x);
                     candidate <
                     kClusteredSuperClusterClusters *
                         view.cluster_size;
                     candidate += static_cast<int>(blockDim.x))
                {
                    const int neighbor_cluster_offset =
                        candidate / view.cluster_size;
                    const int neighbor_lane =
                        candidate -
                        neighbor_cluster_offset *
                            view.cluster_size;
                    SW_CLUSTERED_NEIGHBOR neighbor;
                    if (!SW_Clustered_Center_Tile_Neighbor(
                            view, tile, center_lane, atom_i,
                            neighbor_cluster_offset, neighbor_lane,
                            crd, cell, &neighbor) ||
                        !SW_Clustered_Neighbor_Within_Cut(
                            type_i, neighbor.atom,
                            neighbor.distance, atom_types,
                            parameters, atom_type_numbers,
                            view.rebuild_skin))
                    {
                        continue;
                    }
                    const int slot =
                        atomicAdd(neighbor_counts + sorted_i, 1);
                    if constexpr (fill)
                    {
                        neighbor_atoms[
                            neighbor_offsets[sorted_i] + slot] =
                            neighbor.atom;
                    }
                }
            }
        }
#else
        int count = 0;
        int write_index =
            fill ? neighbor_offsets[sorted_i] : 0;
        SW_CLUSTERED_CENTER_NEIGHBOR_CURSOR cursor;
        if (SW_Clustered_Center_Neighbor_Cursor_Begin(
                view, center_cluster, &cursor))
        {
            SW_CLUSTERED_NEIGHBOR neighbor;
            while (SW_Clustered_Center_Neighbor_Next(
                view, center_lane, atom_i, &cursor, crd, cell,
                &neighbor))
            {
                if (SW_Clustered_Neighbor_Within_Cut(
                        type_i, neighbor.atom, neighbor.distance,
                        atom_types, parameters, atom_type_numbers,
                        view.rebuild_skin))
                {
                    if constexpr (fill)
                    {
                        neighbor_atoms[write_index] = neighbor.atom;
                        write_index += 1;
                    }
                    else
                    {
                        count += 1;
                    }
                }
            }
        }
        if constexpr (!fill)
        {
            neighbor_counts[sorted_i] = count;
        }
#endif
    }
}

#ifdef USE_CPU
static bool SW_Build_Gmxpacked_Center_Atoms_CPU(
    STILLINGER_WEBER_INFORMATION* sw,
    const CLUSTERED_SPATIAL_VIEW& view, const VECTOR* crd,
    const LTMatrix3 cell)
{
    std::vector<std::vector<int>> center_atoms(
        static_cast<size_t>(sw->atom_numbers));
    for (int sci = 0; sci < view.gmxpacked_sci_numbers; sci += 1)
    {
        const CLUSTERED_GMXPACKED_SCI& sci_entry =
            view.gmxpacked_sci[sci];
        const int cluster_i_begin =
            view.super_cluster_offsets[sci_entry.supercluster_id];
        const int cluster_i_end =
            view.super_cluster_offsets[sci_entry.supercluster_id + 1];
        const int cluster_i_numbers = cluster_i_end - cluster_i_begin;
        const VECTOR shift =
            Clustered_Shift_Vector_From_Id(sci_entry.shift_id, cell);
        for (int i_local = 0; i_local < cluster_i_numbers; i_local += 1)
        {
            const int cluster_i = cluster_i_begin + i_local;
            const unsigned int valid_local_i_mask =
                view.cluster_valid_masks[cluster_i] &
                view.cluster_local_masks[cluster_i];
            for (int packed_index = sci_entry.cjpacked_begin;
                 packed_index < sci_entry.cjpacked_end; packed_index += 1)
            {
                const CLUSTERED_GMXPACKED_CJ& packed =
                    view.gmxpacked_cjpacked[packed_index];
                for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
                {
                    const int cluster_j = packed.cj[jm];
                    if (cluster_j < 0)
                    {
                        continue;
                    }
                    const unsigned int packed_bit =
                        1u << (jm * kClusteredSuperClusterClusters + i_local);
                    const uint64_t pair_mask =
                        Clustered_Gmxpacked_CPU_Pair_Mask(
                            packed, packed_bit, view.gmxpacked_exclusions);
                    if (pair_mask == 0ull)
                    {
                        continue;
                    }
                    const unsigned int valid_local_j_mask =
                        view.cluster_valid_masks[cluster_j] &
                        view.cluster_local_masks[cluster_j];
                    unsigned int active_i_mask = valid_local_i_mask;
                    while (active_i_mask != 0u)
                    {
                        const int i_lane = __builtin_ctz(active_i_mask);
                        active_i_mask &= active_i_mask - 1u;
                        const int sorted_i =
                            view.cluster_offsets[cluster_i] + i_lane;
                        const int atom_i = view.sort_permutation[sorted_i];
                        const int type_i = sw->d_atom_type[atom_i];
                        unsigned int active_j_mask =
                            static_cast<unsigned int>(
                                pair_mask >>
                                (i_lane * kClusteredClusterSize)) &
                            valid_local_j_mask;
                        while (active_j_mask != 0u)
                        {
                            const int j_lane = __builtin_ctz(active_j_mask);
                            active_j_mask &= active_j_mask - 1u;
                            const int sorted_j =
                                view.cluster_offsets[cluster_j] + j_lane;
                            const int atom_j = view.sort_permutation[sorted_j];
                            if (atom_i == atom_j)
                            {
                                continue;
                            }
                            const VECTOR displacement =
                                (crd[atom_i] - crd[atom_j]) + shift;
                            const float distance =
                                sqrtf(displacement * displacement);
                            if (SW_Clustered_Neighbor_Within_Cut(
                                    type_i, atom_j, distance,
                                    sw->d_atom_type, sw->d_parameters,
                                    sw->atom_type_numbers,
                                    view.rebuild_skin))
                            {
                                center_atoms[atom_i].push_back(atom_j);
                            }
                            const int type_j = sw->d_atom_type[atom_j];
                            if (SW_Clustered_Neighbor_Within_Cut(
                                    type_j, atom_i, distance,
                                    sw->d_atom_type, sw->d_parameters,
                                    sw->atom_type_numbers,
                                    view.rebuild_skin))
                            {
                                center_atoms[atom_j].push_back(atom_i);
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<int> offsets(
        static_cast<size_t>(sw->atom_numbers + 1), 0);
    long long total = 0;
    for (int sorted_i = 0; sorted_i < sw->atom_numbers; sorted_i += 1)
    {
        const int atom_i = view.sort_permutation[sorted_i];
        offsets[sorted_i] = static_cast<int>(total);
        total += center_atoms[atom_i].size();
        if (total > INT_MAX)
        {
            return false;
        }
    }
    offsets[sw->atom_numbers] = static_cast<int>(total);
    sw->clustered_neighbor_numbers = static_cast<int>(total);
    SW_Reserve_Clustered_Neighbor_Buffer(
        &sw->d_clustered_neighbor_offsets,
        &sw->clustered_neighbor_offsets_capacity,
        sw->atom_numbers + 1);
    SW_Reserve_Clustered_Neighbor_Buffer(
        &sw->d_clustered_neighbor_atoms,
        &sw->clustered_neighbor_atoms_capacity,
        sw->clustered_neighbor_numbers);
    deviceMemcpy(
        sw->d_clustered_neighbor_offsets, offsets.data(),
        sizeof(int) * static_cast<size_t>(sw->atom_numbers + 1),
        deviceMemcpyHostToDevice);
    for (int sorted_i = 0; sorted_i < sw->atom_numbers; sorted_i += 1)
    {
        const int atom_i = view.sort_permutation[sorted_i];
        const std::vector<int>& row = center_atoms[atom_i];
        if (!row.empty())
        {
            deviceMemcpy(
                sw->d_clustered_neighbor_atoms + offsets[sorted_i],
                row.data(), sizeof(int) * row.size(),
                deviceMemcpyHostToDevice);
        }
    }
    return true;
}
#endif

static bool SW_Ensure_Clustered_Center_Atoms(
    STILLINGER_WEBER_INFORMATION* sw,
    const CLUSTERED_SPATIAL_VIEW& view, const VECTOR* crd,
    const LTMatrix3 cell)
{
    const long long payload_generation = view.gmxpacked_payload_generation;
    if (sw->clustered_neighbor_provider_incarnation ==
            view.provider_incarnation &&
        sw->clustered_neighbor_payload_generation ==
            payload_generation)
    {
        return true;
    }
#ifdef USE_CPU
    if (!SW_Build_Gmxpacked_Center_Atoms_CPU(sw, view, crd, cell))
    {
        return false;
    }
#else
    SW_Reserve_Clustered_Neighbor_Buffer(
        &sw->d_clustered_neighbor_counts,
        &sw->clustered_neighbor_counts_capacity,
        sw->atom_numbers);
    SW_Reserve_Clustered_Neighbor_Buffer(
        &sw->d_clustered_neighbor_offsets,
        &sw->clustered_neighbor_offsets_capacity,
        sw->atom_numbers + 1);
    constexpr int kBuilderWarpsPerBlock = 8;
    const dim3 block_size(
        static_cast<unsigned int>(CONTROLLER::device_warp),
        static_cast<unsigned int>(kBuilderWarpsPerBlock), 1u);
    const dim3 grid_size(
        static_cast<unsigned int>(
            (sw->atom_numbers + kBuilderWarpsPerBlock - 1) /
            kBuilderWarpsPerBlock),
        1u, 1u);
    Launch_Device_Kernel(
        SW_Build_Clustered_Center_Atoms<false>,
        grid_size, block_size,
        0, NULL, view, sw->atom_numbers, crd, cell,
        sw->d_atom_type, sw->d_parameters,
        sw->atom_type_numbers, NULL, NULL,
        sw->d_clustered_neighbor_counts);

    std::vector<int> host_counts(
        static_cast<size_t>(sw->atom_numbers));
    std::vector<int> host_offsets(
        static_cast<size_t>(sw->atom_numbers + 1), 0);
    deviceMemcpy(
        host_counts.data(), sw->d_clustered_neighbor_counts,
        sizeof(int) * static_cast<size_t>(sw->atom_numbers),
        deviceMemcpyDeviceToHost);
    long long total = 0;
    for (int i = 0; i < sw->atom_numbers; i += 1)
    {
        if (host_counts[i] < 0)
        {
            return false;
        }
        host_offsets[i] = static_cast<int>(total);
        total += host_counts[i];
        if (total > INT_MAX)
        {
            return false;
        }
    }
    host_offsets[sw->atom_numbers] = static_cast<int>(total);
    sw->clustered_neighbor_numbers = static_cast<int>(total);
    deviceMemcpy(
        sw->d_clustered_neighbor_offsets, host_offsets.data(),
        sizeof(int) * static_cast<size_t>(sw->atom_numbers + 1),
        deviceMemcpyHostToDevice);
    SW_Reserve_Clustered_Neighbor_Buffer(
        &sw->d_clustered_neighbor_atoms,
        &sw->clustered_neighbor_atoms_capacity,
        sw->clustered_neighbor_numbers);
    if (sw->clustered_neighbor_numbers > 0)
    {
        Launch_Device_Kernel(
            SW_Build_Clustered_Center_Atoms<true>,
            grid_size, block_size,
            0, NULL, view, sw->atom_numbers, crd, cell,
            sw->d_atom_type, sw->d_parameters,
            sw->atom_type_numbers,
            sw->d_clustered_neighbor_offsets,
            sw->d_clustered_neighbor_atoms,
            sw->d_clustered_neighbor_counts);
    }
#endif
    sw->clustered_neighbor_provider_incarnation =
        view.provider_incarnation;
    sw->clustered_neighbor_payload_generation =
        payload_generation;
    return true;
}

#ifdef USE_GPU
template <bool full_output>
static __device__ __forceinline__ void
SW_Clustered_Process_Cached_J(
    const int atom_i, const int type_i, const int j, const int lane,
    const int cached_neighbor_count, const int* cached_atoms,
    const int* cached_types,
    const float* cached_dx, const float* cached_dy,
    const float* cached_dz, const float* cached_distances,
    const float* parameters, const int atom_type_numbers,
    const int pair_type_numbers,
    const bool store_virial, VECTOR* neighbor_force, VECTOR* i_force,
    float* local_energy, LTMatrix3* local_virial)
{
    const int atom_j = cached_atoms[j];
    const int type_j = cached_types[j];
    const VECTOR drij = {cached_dx[j], cached_dy[j], cached_dz[j]};
    const float rij = cached_distances[j];
    const int pair_index_1 = type_i * atom_type_numbers + type_j;
    const float A = parameters[8 * pair_index_1];
    const float B = parameters[8 * pair_index_1 + 1];
    float epsilon = parameters[8 * pair_index_1 + 2];
    const float p = parameters[8 * pair_index_1 + 3];
    const float q = parameters[8 * pair_index_1 + 4];
    const float a1 = parameters[8 * pair_index_1 + 5];
    const float gamma1 = parameters[8 * pair_index_1 + 6];
    const float sigma1 = parameters[8 * pair_index_1 + 7];
    if (lane == j && atom_j > atom_i && rij < a1 * sigma1)
    {
        float radial_derivative = 0.0f;
        *local_energy += SW_Two_Body_Analytic(
            rij, A, B, epsilon, p, q, a1, sigma1,
            &radial_derivative);
        const VECTOR pair_force =
            radial_derivative / rij * drij;
        *i_force = *i_force - pair_force;
        *neighbor_force = *neighbor_force + pair_force;
        if constexpr (full_output)
        {
            if (store_virial)
            {
                *local_virial =
                    *local_virial -
                    Get_Virial_From_Force_Dis(pair_force, drij);
            }
        }
    }
    VECTOR j_three_force = {};
    if (lane > j && lane < cached_neighbor_count)
    {
        const int type_k = cached_types[lane];
        const VECTOR drik = {
            cached_dx[lane], cached_dy[lane], cached_dz[lane]};
        const float rik = cached_distances[lane];
        const int pair_index_2 = type_i * atom_type_numbers + type_k;
        const float a2 = parameters[8 * pair_index_2 + 5];
        const float sigma2 = parameters[8 * pair_index_2 + 7];
        if (rij < a1 * sigma1 && rik < a2 * sigma2)
        {
            const float gamma2 = parameters[8 * pair_index_2 + 6];
            const int triple_index =
                type_i * atom_type_numbers * atom_type_numbers +
                type_j * atom_type_numbers + type_k;
            const float lambda =
                parameters[8 * pair_type_numbers + 3 * triple_index];
            epsilon =
                parameters[8 * pair_type_numbers + 3 * triple_index + 1];
            const float b =
                parameters[8 * pair_type_numbers + 3 * triple_index + 2];
            VECTOR k_three_force;
            const float three_body_energy = SW_Three_Body_Analytic(
                drij, drik, rij, rik, sigma1, a1, gamma1,
                sigma2, a2, gamma2, lambda, epsilon, b,
                &j_three_force, &k_three_force);
            *neighbor_force = *neighbor_force + k_three_force;
            *i_force = *i_force - j_three_force - k_three_force;
            *local_energy += three_body_energy;
            if constexpr (full_output)
            {
                if (store_virial)
                {
                    *local_virial =
                        *local_virial -
                        Get_Virial_From_Force_Dis(
                            j_three_force, drij) -
                        Get_Virial_From_Force_Dis(
                            k_three_force, drik);
                }
            }
        }
    }
    for (int delta = warpSize / 2; delta > 0; delta >>= 1)
    {
        j_three_force.x +=
            deviceShflDown(FULL_MASK, j_three_force.x, delta, warpSize);
        j_three_force.y +=
            deviceShflDown(FULL_MASK, j_three_force.y, delta, warpSize);
        j_three_force.z +=
            deviceShflDown(FULL_MASK, j_three_force.z, delta, warpSize);
    }
    const VECTOR total_j_three_force = {
        deviceShfl(FULL_MASK, j_three_force.x, 0, warpSize),
        deviceShfl(FULL_MASK, j_three_force.y, 0, warpSize),
        deviceShfl(FULL_MASK, j_three_force.z, 0, warpSize)};
    if (lane == j)
    {
        *neighbor_force = *neighbor_force + total_j_three_force;
    }
}
#endif

#ifdef USE_CPU
template <bool full_output>
static void SW_Clustered_Process_Cached_Center_CPU(
    const int atom_i, const int type_i, const int neighbor_count,
    const int* neighbor_atoms, const VECTOR* crd, VECTOR* frc,
    const LTMatrix3 cell, const LTMatrix3 rcell, const int* atom_types,
    const float* parameters, const int atom_type_numbers,
    const int pair_type_numbers, const bool store_virial,
    VECTOR* i_force, float* local_energy, LTMatrix3* local_virial)
{
    const VECTOR ri = crd[atom_i];
    for (int j = 0; j < neighbor_count; j += 1)
    {
        const int atom_j = neighbor_atoms[j];
        const int type_j = atom_types[atom_j];
        const VECTOR drij =
            Get_Periodic_Displacement(ri, crd[atom_j], cell, rcell);
        const float rij = sqrtf(drij * drij);
        const int pair_index_1 =
            type_i * atom_type_numbers + type_j;
        const float A = parameters[8 * pair_index_1];
        const float B = parameters[8 * pair_index_1 + 1];
        float epsilon = parameters[8 * pair_index_1 + 2];
        const float p = parameters[8 * pair_index_1 + 3];
        const float q = parameters[8 * pair_index_1 + 4];
        const float a1 = parameters[8 * pair_index_1 + 5];
        const float gamma1 = parameters[8 * pair_index_1 + 6];
        const float sigma1 = parameters[8 * pair_index_1 + 7];
        VECTOR j_force = {};
        if (atom_j > atom_i && rij < a1 * sigma1)
        {
            float radial_derivative = 0.0f;
            *local_energy += SW_Two_Body_Analytic(
                rij, A, B, epsilon, p, q, a1, sigma1,
                &radial_derivative);
            const VECTOR pair_force =
                radial_derivative / rij * drij;
            *i_force = *i_force - pair_force;
            j_force = j_force + pair_force;
            if constexpr (full_output)
            {
                if (store_virial)
                {
                    *local_virial =
                        *local_virial -
                        Get_Virial_From_Force_Dis(pair_force, drij);
                }
            }
        }
        for (int k = j + 1; k < neighbor_count; k += 1)
        {
            const int atom_k = neighbor_atoms[k];
            const int type_k = atom_types[atom_k];
            const VECTOR drik =
                Get_Periodic_Displacement(
                    ri, crd[atom_k], cell, rcell);
            const float rik = sqrtf(drik * drik);
            const int pair_index_2 =
                type_i * atom_type_numbers + type_k;
            const float a2 = parameters[8 * pair_index_2 + 5];
            const float sigma2 = parameters[8 * pair_index_2 + 7];
            VECTOR k_force = {};
            if (rij < a1 * sigma1 && rik < a2 * sigma2)
            {
                const float gamma2 =
                    parameters[8 * pair_index_2 + 6];
                const int triple_index =
                    type_i * atom_type_numbers * atom_type_numbers +
                    type_j * atom_type_numbers + type_k;
                const float lambda =
                    parameters[8 * pair_type_numbers +
                               3 * triple_index];
                epsilon =
                    parameters[8 * pair_type_numbers +
                               3 * triple_index + 1];
                const float b =
                    parameters[8 * pair_type_numbers +
                               3 * triple_index + 2];
                VECTOR j_three_force;
                VECTOR k_three_force;
                const float three_body_energy =
                    SW_Three_Body_Analytic(
                        drij, drik, rij, rik, sigma1, a1, gamma1,
                        sigma2, a2, gamma2, lambda, epsilon, b,
                        &j_three_force, &k_three_force);
                j_force = j_force + j_three_force;
                k_force = k_force + k_three_force;
                *i_force =
                    *i_force - j_three_force - k_three_force;
                *local_energy += three_body_energy;
                if constexpr (full_output)
                {
                    if (store_virial)
                    {
                        *local_virial =
                            *local_virial -
                            Get_Virial_From_Force_Dis(
                                j_three_force, drij) -
                            Get_Virial_From_Force_Dis(
                                k_three_force, drik);
                    }
                }
            }
            atomicAdd(frc + atom_k, k_force);
        }
        atomicAdd(frc + atom_j, j_force);
    }
}
#endif

template <bool full_output>
static __global__ __launch_bounds__(512) void
SW_Clustered_Center_Cached(
    const CLUSTERED_SPATIAL_VIEW view, const int atom_numbers,
    const int* neighbor_offsets, const int* neighbor_atoms,
    const VECTOR* crd, VECTOR* frc, const LTMatrix3 cell,
    const LTMatrix3 rcell, float* atom_energy,
    LTMatrix3* atom_virial, const int* atom_types,
    const float* parameters, const int atom_type_numbers,
    const int pair_type_numbers, float* this_energy,
    const bool store_energy, const bool store_virial)
{
#ifdef USE_GPU
    __shared__ int cached_neighbor_atoms
        [kSWClusteredWarpsPerBlock][kSWClusteredCachedNeighborCapacity];
    __shared__ int cached_neighbor_types
        [kSWClusteredWarpsPerBlock][kSWClusteredCachedNeighborCapacity];
    __shared__ float cached_neighbor_dx
        [kSWClusteredWarpsPerBlock][kSWClusteredCachedNeighborCapacity];
    __shared__ float cached_neighbor_dy
        [kSWClusteredWarpsPerBlock][kSWClusteredCachedNeighborCapacity];
    __shared__ float cached_neighbor_dz
        [kSWClusteredWarpsPerBlock][kSWClusteredCachedNeighborCapacity];
    __shared__ float cached_neighbor_distances
        [kSWClusteredWarpsPerBlock][kSWClusteredCachedNeighborCapacity];
#endif
#ifdef USE_GPU
    const int sorted_i =
        static_cast<int>(threadIdx.y) +
        static_cast<int>(blockDim.y * blockIdx.x);
    if (sorted_i < atom_numbers)
#else
#pragma omp parallel for
    for (int sorted_i = 0; sorted_i < atom_numbers; sorted_i += 1)
#endif
    {
        const int center_cluster =
            SW_Clustered_Find_Cluster(view, sorted_i);
        const int center_lane =
            sorted_i - view.cluster_offsets[center_cluster];
        const int atom_i = view.sort_permutation[sorted_i];
        const int type_i = atom_types[atom_i];
        float local_energy = 0.0f;
        VECTOR i_force = {0.0f, 0.0f, 0.0f};
        LTMatrix3 local_virial = {};
#ifdef USE_GPU
        const int j_begin = static_cast<int>(threadIdx.x);
        const int j_stride = static_cast<int>(blockDim.x);
#else
        const int j_begin = 0;
        const int j_stride = 1;
#endif
#ifdef USE_GPU
        const int warp_slot = static_cast<int>(threadIdx.y);
        const VECTOR ri = crd[atom_i];
        const int neighbor_begin = neighbor_offsets[sorted_i];
        const int neighbor_end = neighbor_offsets[sorted_i + 1];
        int cached_neighbor_total = 0;
        for (int neighbor_base = neighbor_begin;
             neighbor_base < neighbor_end;
             neighbor_base += static_cast<int>(blockDim.x))
        {
            const int neighbor_index =
                neighbor_base + static_cast<int>(threadIdx.x);
            int atom_neighbor = -1;
            VECTOR displacement = {};
            float distance = 0.0f;
            bool keep_neighbor = false;
            if (neighbor_index < neighbor_end)
            {
                atom_neighbor = neighbor_atoms[neighbor_index];
                displacement = Get_Periodic_Displacement(
                    ri, crd[atom_neighbor], cell, rcell);
                distance =
                    sqrtf(displacement.x * displacement.x +
                          displacement.y * displacement.y +
                          displacement.z * displacement.z);
                keep_neighbor = SW_Clustered_Neighbor_Within_Cut(
                    type_i, atom_neighbor, distance, atom_types,
                    parameters, atom_type_numbers);
            }
            const device_mask_t keep_mask =
                deviceBallot(FULL_MASK, keep_neighbor);
            const unsigned int lane_prefix_mask =
                threadIdx.x == 0
                    ? 0u
                    : (1u << static_cast<unsigned int>(threadIdx.x)) -
                          1u;
            const int cached_slot =
                cached_neighbor_total +
                devicePopCount(keep_mask & lane_prefix_mask);
            if (keep_neighbor &&
                cached_slot < kSWClusteredCachedNeighborCapacity)
            {
                cached_neighbor_atoms[warp_slot][cached_slot] =
                    atom_neighbor;
                cached_neighbor_types[warp_slot][cached_slot] =
                    atom_types[atom_neighbor];
                cached_neighbor_dx[warp_slot][cached_slot] =
                    displacement.x;
                cached_neighbor_dy[warp_slot][cached_slot] =
                    displacement.y;
                cached_neighbor_dz[warp_slot][cached_slot] =
                    displacement.z;
                cached_neighbor_distances[warp_slot][cached_slot] =
                    distance;
            }
            cached_neighbor_total += devicePopCount(keep_mask);
        }
        __syncwarp();
        const int cached_neighbor_count =
            cached_neighbor_total <
                    kSWClusteredCachedNeighborCapacity
                ? cached_neighbor_total
                : kSWClusteredCachedNeighborCapacity;
        const bool use_cached_neighbors =
            cached_neighbor_total <=
            kSWClusteredCachedNeighborCapacity;
        if (use_cached_neighbors)
        {
            const int lane = static_cast<int>(threadIdx.x);
            VECTOR neighbor_force = {};
            for (int j = 0; j < cached_neighbor_count; j += 1)
            {
                SW_Clustered_Process_Cached_J<full_output>(
                    atom_i, type_i, j, lane, cached_neighbor_count,
                    cached_neighbor_atoms[warp_slot],
                    cached_neighbor_types[warp_slot],
                    cached_neighbor_dx[warp_slot],
                    cached_neighbor_dy[warp_slot],
                    cached_neighbor_dz[warp_slot],
                    cached_neighbor_distances[warp_slot],
                    parameters, atom_type_numbers,
                    pair_type_numbers, store_virial,
                    &neighbor_force, &i_force, &local_energy,
                    &local_virial);
            }
            if (lane < cached_neighbor_count)
            {
                atomicAdd(
                    frc + cached_neighbor_atoms[warp_slot][lane],
                    neighbor_force);
            }
        }
        if (!use_cached_neighbors)
#endif
        {
#ifdef USE_GPU
        SW_CLUSTERED_CENTER_NEIGHBOR_CURSOR j_cursor;
        if (SW_Clustered_Center_Neighbor_Cursor_Begin(
                view, center_cluster, &j_cursor))
        {
            int j_ordinal = 0;
            SW_CLUSTERED_NEIGHBOR neighbor_j;
            while (SW_Clustered_Center_Neighbor_Next(
                view, center_lane, atom_i, &j_cursor, crd, cell,
                &neighbor_j))
            {
                if (!SW_Clustered_Neighbor_Within_Cut(
                        type_i, neighbor_j.atom, neighbor_j.distance,
                        atom_types, parameters, atom_type_numbers))
                {
                    continue;
                }
                const bool lane_owns_j =
                    j_ordinal >= j_begin &&
                    ((j_ordinal - j_begin) % j_stride) == 0;
                if (!lane_owns_j)
                {
                    j_ordinal += 1;
                    continue;
                }
                const int atom_j = neighbor_j.atom;
                const int type_j = atom_types[atom_j];
                const VECTOR drij = neighbor_j.displacement;
                const float rij = neighbor_j.distance;
                const int pair_index_1 =
                    type_i * atom_type_numbers + type_j;
                const float A = parameters[8 * pair_index_1];
                const float B = parameters[8 * pair_index_1 + 1];
                float epsilon = parameters[8 * pair_index_1 + 2];
                const float p = parameters[8 * pair_index_1 + 3];
                const float q = parameters[8 * pair_index_1 + 4];
                const float a1 = parameters[8 * pair_index_1 + 5];
                const float gamma1 = parameters[8 * pair_index_1 + 6];
                const float sigma1 = parameters[8 * pair_index_1 + 7];
                VECTOR j_force = {};
                if (atom_j > atom_i && rij < a1 * sigma1)
                {
                    float radial_derivative = 0.0f;
                    local_energy += SW_Two_Body_Analytic(
                        rij, A, B, epsilon, p, q, a1, sigma1,
                        &radial_derivative);
                    const VECTOR pair_force =
                        radial_derivative / rij * drij;
                    i_force = i_force - pair_force;
                    j_force = j_force + pair_force;
                    if constexpr (full_output)
                    {
                        if (store_virial)
                        {
                            local_virial =
                                local_virial -
                                Get_Virial_From_Force_Dis(
                                    pair_force, drij);
                        }
                    }
                }
                SW_CLUSTERED_CENTER_NEIGHBOR_CURSOR k_cursor = j_cursor;
                SW_CLUSTERED_NEIGHBOR neighbor_k;
                while (SW_Clustered_Center_Neighbor_Next(
                    view, center_lane, atom_i, &k_cursor, crd, cell,
                    &neighbor_k))
                {
                    if (!SW_Clustered_Neighbor_Within_Cut(
                            type_i, neighbor_k.atom,
                            neighbor_k.distance, atom_types,
                            parameters, atom_type_numbers))
                    {
                        continue;
                    }
                    const int atom_k = neighbor_k.atom;
                    const int type_k = atom_types[atom_k];
                    const VECTOR drik = neighbor_k.displacement;
                    const float rik = neighbor_k.distance;
                    const int pair_index_2 =
                        type_i * atom_type_numbers + type_k;
                    const float a2 =
                        parameters[8 * pair_index_2 + 5];
                    const float sigma2 =
                        parameters[8 * pair_index_2 + 7];
                    VECTOR k_force = {};
                    if (rij < a1 * sigma1 && rik < a2 * sigma2)
                    {
                        const float gamma2 =
                            parameters[8 * pair_index_2 + 6];
                        const int triple_index =
                            type_i * atom_type_numbers *
                                atom_type_numbers +
                            type_j * atom_type_numbers + type_k;
                        const float lambda =
                            parameters[8 * pair_type_numbers +
                                       3 * triple_index];
                        epsilon =
                            parameters[8 * pair_type_numbers +
                                       3 * triple_index + 1];
                        const float b =
                            parameters[8 * pair_type_numbers +
                                       3 * triple_index + 2];
                        VECTOR j_three_force;
                        VECTOR k_three_force;
                        const float three_body_energy =
                            SW_Three_Body_Analytic(
                                drij, drik, rij, rik, sigma1, a1,
                                gamma1, sigma2, a2, gamma2, lambda,
                                epsilon, b, &j_three_force,
                                &k_three_force);
                        k_force = k_force + k_three_force;
                        j_force = j_force + j_three_force;
                        i_force =
                            i_force - j_three_force - k_three_force;
                        local_energy += three_body_energy;
                        if constexpr (full_output)
                        {
                            if (store_virial)
                            {
                                local_virial =
                                    local_virial -
                                    Get_Virial_From_Force_Dis(
                                        j_three_force, drij) -
                                    Get_Virial_From_Force_Dis(
                                        k_three_force, drik);
                            }
                        }
                    }
                    atomicAdd(frc + atom_k, k_force);
                }
                atomicAdd(frc + atom_j, j_force);
                j_ordinal += 1;
            }
        }
#else
        const int neighbor_begin = neighbor_offsets[sorted_i];
        const int neighbor_end = neighbor_offsets[sorted_i + 1];
        SW_Clustered_Process_Cached_Center_CPU<full_output>(
            atom_i, type_i, neighbor_end - neighbor_begin,
            neighbor_atoms + neighbor_begin, crd, frc, cell, rcell,
            atom_types, parameters, atom_type_numbers,
            pair_type_numbers, store_virial, &i_force, &local_energy,
            &local_virial);
#endif
        }
#ifdef USE_GPU
        Warp_Sum_To(frc + atom_i, i_force, warpSize);
        if constexpr (full_output)
        {
            if (store_energy)
            {
                Warp_Sum_To(atom_energy + atom_i, local_energy,
                            warpSize);
                if (threadIdx.x == 0)
                {
                    atomicAdd(this_energy + atom_i, local_energy);
                }
            }
            if (store_virial)
            {
                Warp_Sum_To(atom_virial + atom_i, local_virial,
                            warpSize);
            }
        }
#else
        atomicAdd(frc + atom_i, i_force);
        if constexpr (full_output)
        {
            if (store_energy)
            {
                atomicAdd(atom_energy + atom_i, local_energy);
                atomicAdd(this_energy + atom_i, local_energy);
            }
            if (store_virial)
            {
                atomicAdd(atom_virial + atom_i, local_virial);
            }
        }
#endif
    }
}

bool STILLINGER_WEBER_INFORMATION::SW_Force_Clustered(
    const CLUSTERED_SPATIAL_VIEW& view, const VECTOR* crd, VECTOR* frc,
    const LTMatrix3 cell, const LTMatrix3 rcell, const int need_atom_energy,
    float* atom_energy, const int need_virial, LTMatrix3* atom_virial,
    const char** failure_reason)
{
    if (failure_reason != NULL)
    {
        *failure_reason = NULL;
    }
    if (!is_initialized)
    {
        return true;
    }
    if (crd == NULL || frc == NULL ||
        (need_atom_energy && atom_energy == NULL) ||
        (need_virial && atom_virial == NULL))
    {
        if (failure_reason != NULL)
        {
            *failure_reason = "SW clustered force received null buffers";
        }
        return false;
    }

    if (need_atom_energy)
    {
        deviceMemset(d_energy_sum, 0, sizeof(float) * (atom_numbers + 1));
    }
    if (!SW_Ensure_Clustered_Center_Atoms(this, view, crd, cell))
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                "SW could not derive compact center-neighbor adjacency";
        }
        return false;
    }

    dim3 blockSize = {
        CONTROLLER::device_warp, kSWClusteredWarpsPerBlock};
    dim3 gridSize = (atom_numbers + blockSize.y - 1) / blockSize.y;
    auto f = SW_Clustered_Center_Cached<false>;
    if (need_atom_energy || need_virial)
    {
        f = SW_Clustered_Center_Cached<true>;
    }
    Launch_Device_Kernel(f, gridSize, blockSize, 0, NULL, view,
                         atom_numbers, d_clustered_neighbor_offsets,
                         d_clustered_neighbor_atoms, crd, frc, cell,
                         rcell, atom_energy, atom_virial,
                         this->d_atom_type, this->d_parameters,
                         this->atom_type_numbers, this->pair_type_numbers,
                         this->d_energy_atom, need_atom_energy != 0,
                         need_virial != 0);
    if (failure_reason != NULL)
    {
        *failure_reason = NULL;
    }
    return true;
}

void STILLINGER_WEBER_INFORMATION::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    Sum_Of_List(d_energy_atom, d_energy_sum, atom_numbers);
    deviceMemcpy(&h_energy_sum, d_energy_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
    controller->Step_Print(this->module_name, h_energy_sum, true);
}
