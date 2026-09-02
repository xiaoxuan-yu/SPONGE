#include "pairwise_force.h"

#include <cmath>
#include <filesystem>
#include <limits>

#include "../utils/h5md/topology_custom_force_h5_materializer.hpp"

static __global__ void pairwise_force_scatter_types(
    const int total_numbers, const int* atom_local,
    const int* global_pairwise_types, int* local_pairwise_types)
{
#ifdef USE_GPU
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_numbers)
#else
#pragma omp parallel for
    for (int idx = 0; idx < total_numbers; idx++)
#endif
    {
        int atom = atom_local[idx];
        local_pairwise_types[idx] = global_pairwise_types[atom];
    }
}

static __global__ void pairwise_force_gather_clustered_fields(
    const int total_numbers, const int cluster_numbers,
    const int* sort_permutation, const int* cluster_offsets,
    const VECTOR* cluster_centers, const VECTOR* crd, const float* charge,
    const int* local_pairwise_types, const LTMatrix3 cell,
    const LTMatrix3 rcell, int* sorted_atom_ids, VECTOR* sorted_crd,
    float* sorted_charge, int* sorted_pairwise_types)
{
#ifdef USE_GPU
    int sorted_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (sorted_i < total_numbers)
#else
#pragma omp parallel for
    for (int sorted_i = 0; sorted_i < total_numbers; sorted_i++)
#endif
    {
        const int cluster_i = Clustered_Find_Cluster_For_Sorted_Index(
            sorted_i, cluster_numbers, cluster_offsets);
        const int atom_i = sort_permutation[sorted_i];
        const VECTOR center = cluster_centers[cluster_i];
        const VECTOR atom_crd = crd[atom_i];
        sorted_atom_ids[sorted_i] = atom_i;
        sorted_crd[sorted_i] =
            center + Get_Periodic_Displacement(atom_crd, center, cell, rcell);
        sorted_charge[sorted_i] = charge == NULL ? 0.0f : charge[atom_i];
        sorted_pairwise_types[sorted_i] = local_pairwise_types[atom_i];
    }
}

static bool pairwise_force_clustered_fail(const char* reason,
                                          const char** failure_reason)
{
    if (failure_reason != NULL)
    {
        *failure_reason = reason;
    }
    return false;
}

static int pairwise_force_clustered_cj_partitions(int sci_numbers,
                                                  int cjpacked_numbers)
{
#ifdef USE_CPU
    (void)sci_numbers;
    (void)cjpacked_numbers;
    return 1;
#else
    constexpr int target_cj_per_partition = 1;
    constexpr int maximum_partitions = 8;
    if (sci_numbers <= 0 || cjpacked_numbers <= 0)
    {
        return 1;
    }
    const int average_cj_per_sci =
        (cjpacked_numbers + sci_numbers - 1) / sci_numbers;
    const int partitions = (average_cj_per_sci + target_cj_per_partition - 1) /
                           target_cj_per_partition;
    return partitions < 1
               ? 1
               : (partitions > maximum_partitions ? maximum_partitions
                                                  : partitions);
#endif
}

void PAIRWISE_FORCE::Initial(CONTROLLER* controller, const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "pairwise_force");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    bool loaded_native = false;
    if (controller->Command_Exist("input_h5_topology_path"))
    {
        SpongeH5MD::TopologyCustomForceH5Materializer reader;
        if (!reader.Open(controller->Command("input_h5_topology_path")))
        {
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                           "PAIRWISE_FORCE::Initial",
                                           reader.Last_Error().c_str());
        }
        if (reader.Has_Pairwise())
        {
            SpongeH5MD::NativePairwiseForceDefinition definition;
            if (!reader.Read_Pairwise(&definition))
            {
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                               "PAIRWISE_FORCE::Initial",
                                               reader.Last_Error().c_str());
            }
            const bool legacy_descriptor_available =
                controller->Command_Exist(this->module_name, "in_file") &&
                std::filesystem::is_regular_file(
                    controller->Command(this->module_name, "in_file"));
            const bool legacy_data_available =
                controller->Command_Exist(definition.name.c_str(), "in_file") &&
                std::filesystem::is_regular_file(
                    controller->Command(definition.name.c_str(), "in_file"));
            if (!legacy_descriptor_available || !legacy_data_available)
            {
                controller->printf(
                    "START INITIALIZING PAIRWISE FORCE FROM NATIVE H5:\n");
                force_name = definition.name;
                source_code = definition.potential;
                parameter_type = definition.parameter_types;
                parameter_name = definition.parameter_names;
                n_ij_parameter = definition.ij_parameter_count;
                with_ele = definition.with_ele;
                if (with_ele)
                {
                    ele_code = definition.electrostatic_potential.empty()
                                   ? "E_ele = charge_i * charge_j * erfc(beta "
                                     "* r_ij) / r_ij;"
                                   : definition.electrostatic_potential;
                }
                atom_numbers = definition.atom_count;
                type_numbers = definition.type_count;
                native_parameter_values = definition.parameter_values;
                native_atom_types = definition.atom_type;
                has_native_parameters = true;
                JIT_Compile(controller);
                Real_Initial(controller);
                loaded_native = true;
            }
        }
    }
    if (!loaded_native &&
        controller->Command_Exist(this->module_name, "in_file"))
    {
        controller->printf("START INITIALIZING PAIRWISE FORCE:\n");
        this->Read_Configuration(controller);
        this->JIT_Compile(controller);
        this->Real_Initial(controller);
    }
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller->Step_Print_Initial(this->force_name.c_str(), "%.2f");
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n",
                             last_modify_date);
    }
    if (is_initialized)
    {
        controller[0].printf("END INITIALIZING PAIRWISE FORCE\n\n");
    }
    else
    {
        controller->printf("PAIRWISE FORCE IS NOT INITIALIZED\n\n");
    }
}

void PAIRWISE_FORCE::Read_Configuration(CONTROLLER* controller)
{
    Configuration_Reader cfg;
    cfg.Open(controller->Command(this->module_name, "in_file"));
    cfg.Close();
    if (!cfg.error_reason.empty())
    {
        cfg.error_reason = "Reason:\n\t" + cfg.error_reason;
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "PAIRWISE_FORCE::Initial",
                                       cfg.error_reason.c_str());
    }
    if (cfg.sections.size() > 1)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "PAIRWISE_FORCE::Initial",
            "Reason:\n\tOnly one pairwise force can be used\n");
    }
    force_name = cfg.sections[0];
    if (!cfg.Key_Exist(force_name, "potential"))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "PAIRWISE_FORCE::Initial",
            string_format("Reason:\n\tThe potential of the pairwise force "
                          "%FORCE% is required ([[ potential ]])\n",
                          {{"FORCE", force_name}})
                .c_str());
    }
    source_code = cfg.Get_Value(force_name, "potential");
    if (!cfg.Key_Exist(force_name, "parameters"))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "PAIRWISE_FORCE::Initial",
            string_format("Reason:\n\tThe parameters of the pairwise force "
                          "%FORCE% are required ([[ parameter ]])\n",
                          {{"FORCE", force_name}})
                .c_str());
    }
    std::string parameter_strings = cfg.Get_Value(force_name, "parameters");
    std::vector<std::string> parameter_and_types =
        string_split(parameter_strings, ",");
    for (std::string s : parameter_and_types)
    {
        std::vector<std::string> parameter_and_type =
            string_split(string_strip(s), " ");
        if (parameter_and_type[0] != "int" && parameter_and_type[0] != "float")
        {
            controller->Throw_SPONGE_Error(
                spongeErrorTypeErrorCommand,
                "PAIRWISE_FORCE::Initialize_Parameters",
                "Reason:\n\tOnly 'int' or 'float' parameter is acceptable\n");
        }
        this->parameter_type.push_back(parameter_and_type[0]);
        this->parameter_name.push_back(parameter_and_type[1]);
    }
    n_ij_parameter = 0;
    for (auto s : this->parameter_name)
    {
        if (s.rfind("_ij") == s.length() - 3)
        {
            n_ij_parameter -= 1;
        }
        else if (n_ij_parameter < 0)
        {
            n_ij_parameter *= -1;
        }
        else
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "PAIRWISE_FORCE::Initialize_Parameters",
                "Reason:\n\tPairwise parameters should be placed in front of "
                "atomic parameters");
        }
    }
    n_ij_parameter = abs(n_ij_parameter);
    with_ele = true;
    if (cfg.Key_Exist(force_name, "with_ele"))
    {
        std::string with_ele_choice = cfg.Get_Value(force_name, "with_ele");
        if (!is_str_equal(with_ele_choice.c_str(), "true") &&
            !is_str_equal(with_ele_choice.c_str(), "false") &&
            !is_str_int(with_ele_choice.c_str()))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "PAIRWISE_FORCE::Initialize_Parameters",
                "Reason:\n\tPairwise [[ with_ele ]] should be 'true', 'false' "
                "or integers (0 for 'false' and others for 'true')");
        }
        if (is_str_equal(with_ele_choice.c_str(), "true") ||
            (is_str_int(with_ele_choice.c_str()) &&
             atoi(with_ele_choice.c_str())))
        {
            with_ele = true;
        }
        else
        {
            with_ele = false;
        }
    }
    if (with_ele)
    {
        ele_code = "E_ele = charge_i * charge_j * erfc(beta * r_ij) / r_ij;";
        if (cfg.Key_Exist(force_name, "electrostatic_potential"))
        {
            ele_code = cfg.Get_Value(force_name, "electrostatic_potential");
        }
    }
    else
    {
        ele_code.clear();
    }
    for (auto s : cfg.value_unused)
    {
        std::string error_reason = string_format(
            "Reason:\n\t[[ %s% ]] should not be one of the keys of the "
            "pairwise force input file",
            {{"s", s.second}});
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "PAIRWISE_FORCE::Read_Configuration",
                                       error_reason.c_str());
    }
}

void PAIRWISE_FORCE::JIT_Compile(CONTROLLER* controller)
{
    if (with_ele)
        controller->printf(
            "        %s will be calculated with direct part of the "
            "electrostatic potential\n",
            this->force_name.c_str());
    else
        controller->printf(
            "        %s will not be calculated with direct part of the "
            "electrostatic potential\n",
            this->force_name.c_str());
    std::string PARM_ARGS = string_join("const %0%* %1%_list", ", ",
                                        {parameter_type, parameter_name});
    std::string clustered_source_code = R"JIT(#if defined(__CUDACC__)
#ifndef USE_GPU
#define USE_GPU
#endif
#ifndef USE_CUDA
#define USE_CUDA
#endif
#elif defined(__HIPCC__) || defined(__HIPCC_RTC__)
#ifndef USE_GPU
#define USE_GPU
#endif
#ifndef USE_HIP
#define USE_HIP
#endif
#endif
#include "common.h"

#define Clustered_Lane_Is_Valid(valid_mask, lane) \
    (((valid_mask) & (1u << static_cast<unsigned int>(lane))) != 0u)
#define Clustered_Lane_Is_Local(local_mask, lane) \
    (((local_mask) & (1u << static_cast<unsigned int>(lane))) != 0u)

#ifndef USE_GPU
__forceinline__ float atomicAdd(float* x, float y)
{
    float x0;
#ifdef _WIN32
#pragma omp critical(sponge_jit_clustered_atomic_add_float)
#else
#pragma omp atomic capture
#endif
    {
        x0 = *x;
        *x += y;
    }
    return x0;
}
#endif

__device__ __forceinline__ unsigned int
Clustered_Get_Pair_Active_I_Mask(unsigned long long packed_shift_bits,
                                 int split)
{
    const unsigned long long active_marker = 1ull << 56;
    if ((packed_shift_bits & active_marker) == 0ull)
    {
        return 0xffu;
    }
    return static_cast<unsigned int>(
        (packed_shift_bits >> (40 + split * 8)) & 0xffull);
}

struct CLUSTERED_PAIRWISE_SCI
{
    int supercluster_id;
    int shift_id;
    int cjpacked_begin;
    int cjpacked_end;
};

struct CLUSTERED_PAIRWISE_SPLIT
{
    unsigned int imask;
    int exclusion_index;
};

struct CLUSTERED_PAIRWISE_CJ
{
    int cj[4];
    CLUSTERED_PAIRWISE_SPLIT split[2];
};

struct CLUSTERED_PAIRWISE_EXCLUSION
{
    unsigned int pair[32];
};

struct CLUSTERED_PAIRWISE_EVALUATION
{
    float energy;
    float coulomb_energy;
    float force_over_r;
};

struct CLUSTERED_PAIRWISE_STORED
{
    VECTOR force;
    float pair_energy;
    float coulomb_energy;
};

#ifndef CLUSTERED_PAIRWISE_ENABLE_FULL_OUTPUT
#define CLUSTERED_PAIRWISE_ENABLE_FULL_OUTPUT 1
#endif

__device__ __forceinline__ int Clustered_Pairwise_Type(int a, int b)
{
    int y = b - a;
    int x = y >> 31;
    y = (y ^ x) - x;
    x = b + a;
    int z = (x + y) >> 1;
    x = (x - y) >> 1;
    return (z * (z + 1) >> 1) + x;
}

__device__ __forceinline__ VECTOR Clustered_Pairwise_Shift_Vector(
    int shift_id, LTMatrix3 cell)
{
    const int sx = shift_id / 9 - 1;
    const int sy = (shift_id % 9) / 3 - 1;
    const int sz = shift_id % 3 - 1;
    const VECTOR shift_index = {
        static_cast<float>(sx), static_cast<float>(sy),
        static_cast<float>(sz)};
    return shift_index * cell;
}

__device__ __forceinline__ CLUSTERED_PAIRWISE_EVALUATION
Evaluate_Clustered_Pairwise(%PARM_ARGS%, int pairwise_type_i,
                            int pairwise_type_j, float charge_i,
                            float charge_j, float r_value, float beta,
                            bool do_coulomb)
{
    const int atom_pairwise_type =
        Clustered_Pairwise_Type(pairwise_type_i, pairwise_type_j);
    %PARM_DEC_CLUSTERED%
    SADfloat<1> r_ij(r_value, 0);
    SADfloat<1> E, E_ele;
    %SOURCE_CODE%
    CLUSTERED_PAIRWISE_EVALUATION result = {
        E.val, 0.0f, E.dval[0] / r_value};
    if (do_coulomb)
    {
        %COULOMB_CODE%
        result.coulomb_energy = E_ele.val;
        result.force_over_r += E_ele.dval[0] / r_value;
    }
    return result;
}

#ifdef USE_GPU
__device__ __forceinline__ bool
Clustered_Pairwise_Subgroup_Has_Work(bool lane_has_work, int j_lane)
{
    const int subgroup_in_warp = j_lane % (warpSize / 8);
    const unsigned long long subgroup_mask =
        0xffull << static_cast<unsigned int>(subgroup_in_warp * 8);
#ifdef USE_CUDA
    const unsigned int active_mask = __activemask();
    const unsigned int work_mask =
        __ballot_sync(active_mask, lane_has_work);
    return (static_cast<unsigned long long>(work_mask) & subgroup_mask) !=
           0ull;
#else
    const unsigned long long work_mask = __ballot(lane_has_work);
    return (work_mask & subgroup_mask) != 0ull;
#endif
}

__device__ __forceinline__ float
Reduce_Clustered_Pairwise_Warp_I_To_Component(float x, float y, float z,
                                                int i_lane,
                                                int component_lane)
{
    for (int delta = warpSize >> 1; delta >= 8; delta >>= 1)
    {
#ifdef USE_CUDA
        x += __shfl_down_sync(0xffffffffu, x, delta, warpSize);
        y += __shfl_down_sync(0xffffffffu, y, delta, warpSize);
        z += __shfl_down_sync(0xffffffffu, z, delta, warpSize);
#else
        x += __shfl_down(x, delta, warpSize);
        y += __shfl_down(y, delta, warpSize);
        z += __shfl_down(z, delta, warpSize);
#endif
    }
#ifdef USE_CUDA
    x = __shfl_sync(0xffffffffu, x, i_lane, warpSize);
    y = __shfl_sync(0xffffffffu, y, i_lane, warpSize);
    z = __shfl_sync(0xffffffffu, z, i_lane, warpSize);
#else
    x = __shfl(x, i_lane, warpSize);
    y = __shfl(y, i_lane, warpSize);
    z = __shfl(z, i_lane, warpSize);
#endif
    if (component_lane == 0)
    {
        return x;
    }
    if (component_lane == 1)
    {
        return y;
    }
    return z;
}

__device__ __forceinline__ float
Reduce_Clustered_Pairwise_Subgroup_J_To_Component(float x, float y, float z,
                                                   int component_lane)
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
    if (component_lane == 0)
    {
        return x;
    }
    if (component_lane == 1)
    {
        return y;
    }
    return z;
}

#if CLUSTERED_PAIRWISE_ENABLE_FULL_OUTPUT
__device__ __forceinline__ int Clustered_Pairwise_Full_Output_Index(
    int field, int i_local, int j_lane, int i_lane)
{
    return (((field * 8 + i_local) * 8 + j_lane) * 8 + i_lane);
}

__device__ __forceinline__ float Clustered_Pairwise_Reduce_Full_Output_Field(
    const float* shared_full_output, int field, int i_local, int i_lane)
{
    float value = 0.0f;
#pragma unroll
    for (int source_j_lane = 0; source_j_lane < 8; source_j_lane++)
    {
        value += shared_full_output[Clustered_Pairwise_Full_Output_Index(
            field, i_local, source_j_lane, i_lane)];
    }
    return value;
}
#endif

__device__ __forceinline__ void Clustered_Pairwise_Atomic_Add_Force_Component(
    VECTOR* frc, int atom_index, int component, float value)
{
    float* force_component = reinterpret_cast<float*>(frc + atom_index);
    atomicAdd(force_component + component, value);
}
#endif

__device__ __forceinline__ CLUSTERED_PAIRWISE_STORED Store_Clustered_Pairwise(
    %PARM_ARGS%, int atom_i, int atom_j, int pairwise_type_i,
    int pairwise_type_j, float charge_i, float charge_j, float dx, float dy,
    float dz, float r_value, float beta, float ij_factor, bool j_is_local,
    VECTOR* frc, float* atom_energy, LTMatrix3* atom_virial,
    float* pme_atom_energy, float* listed_item_energy, int need_atom_energy,
    int need_virial)
{
    const bool do_coulomb = pme_atom_energy != nullptr;
    const CLUSTERED_PAIRWISE_EVALUATION evaluation =
        Evaluate_Clustered_Pairwise(%PARM_CALL%, pairwise_type_i,
                                    pairwise_type_j, charge_i, charge_j,
                                    r_value, beta, do_coulomb);
    const float fij_x = evaluation.force_over_r * dx;
    const float fij_y = evaluation.force_over_r * dy;
    const float fij_z = evaluation.force_over_r * dz;
    if (frc != nullptr)
    {
#ifndef USE_GPU
        atomicAdd(&frc[atom_i].x, fij_x);
        atomicAdd(&frc[atom_i].y, fij_y);
        atomicAdd(&frc[atom_i].z, fij_z);
        if (j_is_local)
        {
            atomicAdd(&frc[atom_j].x, -fij_x);
            atomicAdd(&frc[atom_j].y, -fij_y);
            atomicAdd(&frc[atom_j].z, -fij_z);
        }
#endif
    }
    const float pair_energy = ij_factor * evaluation.energy;
    const float coulomb_energy = ij_factor * evaluation.coulomb_energy;
#ifndef USE_GPU
    if (listed_item_energy != nullptr)
    {
        atomicAdd(listed_item_energy + atom_i, pair_energy);
    }
    if (need_atom_energy && atom_energy != nullptr)
    {
        atomicAdd(atom_energy + atom_i, pair_energy);
    }
    if (pme_atom_energy != nullptr && (need_atom_energy || need_virial))
    {
        atomicAdd(pme_atom_energy + atom_i, coulomb_energy);
    }
    if (need_virial && atom_virial != nullptr)
    {
        const float factor = -ij_factor;
        atomicAdd(&atom_virial[atom_i].a11, factor * fij_x * dx);
        atomicAdd(&atom_virial[atom_i].a21,
                  factor * (fij_x * dy + fij_y * dx));
        atomicAdd(&atom_virial[atom_i].a22, factor * fij_y * dy);
        atomicAdd(&atom_virial[atom_i].a31,
                  factor * (fij_x * dz + fij_z * dx));
        atomicAdd(&atom_virial[atom_i].a32,
                  factor * (fij_y * dz + fij_z * dy));
        atomicAdd(&atom_virial[atom_i].a33, factor * fij_z * dz);
    }
#endif
    return CLUSTERED_PAIRWISE_STORED{
        VECTOR{fij_x, fij_y, fij_z}, pair_energy, coulomb_energy};
}

__device__ __forceinline__ void Traverse_Clustered_Pairwise_Lane(
    %PARM_ARGS%, int sci, int i_lane, int j_lane, int packed_partition,
    int packed_partitions, int cluster_numbers,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const int* super_cluster_offsets,
    const CLUSTERED_PAIRWISE_SCI* sci_entries,
    const CLUSTERED_PAIRWISE_CJ* cjpacked_entries,
    const CLUSTERED_PAIRWISE_EXCLUSION* exclusion_entries,
    const unsigned long long* pair_shift_bits, const int* sorted_atom_ids,
    const VECTOR* sorted_crd, const float* sorted_charge,
    const int* sorted_pairwise_types, float* shared_full_output,
    LTMatrix3 cell, float cutoff, float pme_beta, VECTOR* frc,
    float* atom_energy,
    LTMatrix3* atom_virial, float* pme_atom_energy,
    float* listed_item_energy, int need_atom_energy, int need_virial)
{
    const CLUSTERED_PAIRWISE_SCI sci_entry = sci_entries[sci];
    const int super_i = sci_entry.supercluster_id;
    const int cluster_i_begin = super_cluster_offsets[super_i];
    int cluster_i_end = super_cluster_offsets[super_i + 1];
    if (cluster_i_end > cluster_numbers)
    {
        cluster_i_end = cluster_numbers;
    }
    const int split = j_lane / 4;
    const int split_j_lane = j_lane - split * 4;
    const float cutoff_sq = cutoff * cutoff;

#ifdef USE_GPU
#define CLUSTERED_PAIRWISE_I_LOCAL_LIST(OP) \
    OP(0)                                   \
    OP(1)                                   \
    OP(2)                                   \
    OP(3)                                   \
    OP(4)                                   \
    OP(5)                                   \
    OP(6)                                   \
    OP(7)

#define CLUSTERED_PAIRWISE_DECLARE_FCI(I) \
    float fci_x_##I = 0.0f;              \
    float fci_y_##I = 0.0f;              \
    float fci_z_##I = 0.0f;
    CLUSTERED_PAIRWISE_I_LOCAL_LIST(CLUSTERED_PAIRWISE_DECLARE_FCI)
#undef CLUSTERED_PAIRWISE_DECLARE_FCI

    __shared__ VECTOR shared_i_crd[8 * 8];
    __shared__ float shared_i_charge[8 * 8];
    __shared__ int shared_i_pairwise_type[8 * 8];
    __shared__ int shared_i_atom_id[8 * 8];
#if CLUSTERED_PAIRWISE_ENABLE_FULL_OUTPUT
    const bool full_output =
        listed_item_energy != nullptr || need_atom_energy ||
        (pme_atom_energy != nullptr &&
         (need_atom_energy || need_virial)) ||
        need_virial;
#endif
    const int i_cache_index = j_lane * 8 + i_lane;
    const int cache_cluster_i = cluster_i_begin + j_lane;
    if (cache_cluster_i < cluster_i_end &&
        Clustered_Lane_Is_Valid(cluster_valid_masks[cache_cluster_i],
                                i_lane) &&
        Clustered_Lane_Is_Local(cluster_local_masks[cache_cluster_i], i_lane))
    {
        const int cache_sorted_i =
            cluster_offsets[cache_cluster_i] + i_lane;
        shared_i_crd[i_cache_index] = sorted_crd[cache_sorted_i];
        shared_i_charge[i_cache_index] = sorted_charge[cache_sorted_i];
        shared_i_pairwise_type[i_cache_index] =
            sorted_pairwise_types[cache_sorted_i];
        shared_i_atom_id[i_cache_index] =
            sorted_atom_ids[cache_sorted_i];
    }
    else
    {
        shared_i_crd[i_cache_index] = VECTOR{0.0f, 0.0f, 0.0f};
        shared_i_charge[i_cache_index] = 0.0f;
        shared_i_pairwise_type[i_cache_index] = 0;
        shared_i_atom_id[i_cache_index] = -1;
    }
#if CLUSTERED_PAIRWISE_ENABLE_FULL_OUTPUT
    if (full_output)
    {
        for (int output_index = i_cache_index; output_index < 8 * 8 * 8 * 8;
             output_index += 8 * 8)
        {
            shared_full_output[output_index] = 0.0f;
        }
    }
#endif
    __syncthreads();
#endif

    for (int packed_idx = sci_entry.cjpacked_begin + packed_partition;
         packed_idx < sci_entry.cjpacked_end;
         packed_idx += packed_partitions)
    {
        const CLUSTERED_PAIRWISE_CJ* packed =
            cjpacked_entries + packed_idx;
        const unsigned int imask = packed->split[split].imask;
        if (imask == 0u)
        {
            continue;
        }
        unsigned int pair_bits = 0xffffffffu;
        const int exclusion_index =
            packed->split[split].exclusion_index;
        if (exclusion_index != 0 && exclusion_entries != nullptr)
        {
            pair_bits = exclusion_entries[exclusion_index]
                            .pair[split_j_lane * 8 + i_lane];
        }
        const unsigned int effective_mask = imask & pair_bits;
        for (int jm = 0; jm < 4; jm++)
        {
            const unsigned int jm_mask = 0xffu << (jm * 8);
#ifdef USE_GPU
            if (!Clustered_Pairwise_Subgroup_Has_Work(
                    (effective_mask & jm_mask) != 0u, j_lane))
            {
                continue;
            }
#else
            if ((effective_mask & jm_mask) == 0u)
            {
                continue;
            }
#endif
            const int cluster_j = packed->cj[jm];
            if (cluster_j < 0 || !Clustered_Lane_Is_Valid(
                                     cluster_valid_masks[cluster_j], j_lane))
            {
                continue;
            }
            const int sorted_j = cluster_offsets[cluster_j] + j_lane;
            const int atom_j = sorted_atom_ids[sorted_j];
            const VECTOR rj = sorted_crd[sorted_j];
            const bool j_is_local = Clustered_Lane_Is_Local(
                cluster_local_masks[cluster_j], j_lane);
            const float ij_factor = j_is_local ? 1.0f : 0.5f;
            const unsigned long long shift_bits =
                pair_shift_bits != nullptr
                    ? pair_shift_bits[packed_idx * 4 + jm]
                    : 0ull;

#ifdef USE_GPU
            float fcj_x = 0.0f;
            float fcj_y = 0.0f;
            float fcj_z = 0.0f;
#if CLUSTERED_PAIRWISE_ENABLE_FULL_OUTPUT
#define CLUSTERED_PAIRWISE_ACCUMULATE_FULL_OUTPUT(I)                      \
    if (listed_item_energy != nullptr || need_atom_energy)                \
    {                                                                     \
        shared_full_output[                                               \
            Clustered_Pairwise_Full_Output_Index(                         \
                0, (I), j_lane, i_lane)] += stored.pair_energy;           \
    }                                                                     \
    if (pme_atom_energy != nullptr &&                                    \
        (need_atom_energy || need_virial))                                \
    {                                                                     \
        shared_full_output[                                               \
            Clustered_Pairwise_Full_Output_Index(                         \
                1, (I), j_lane, i_lane)] += stored.coulomb_energy;        \
    }                                                                     \
    if (need_virial)                                                      \
    {                                                                     \
        const float virial_factor = -ij_factor;                           \
        shared_full_output[                                               \
            Clustered_Pairwise_Full_Output_Index(                         \
                2, (I), j_lane, i_lane)] +=                               \
            virial_factor * force_ij.x * dx;                              \
        shared_full_output[                                               \
            Clustered_Pairwise_Full_Output_Index(                         \
                3, (I), j_lane, i_lane)] +=                               \
            virial_factor * (force_ij.x * dy + force_ij.y * dx);          \
        shared_full_output[                                               \
            Clustered_Pairwise_Full_Output_Index(                         \
                4, (I), j_lane, i_lane)] +=                               \
            virial_factor * force_ij.y * dy;                              \
        shared_full_output[                                               \
            Clustered_Pairwise_Full_Output_Index(                         \
                5, (I), j_lane, i_lane)] +=                               \
            virial_factor * (force_ij.x * dz + force_ij.z * dx);          \
        shared_full_output[                                               \
            Clustered_Pairwise_Full_Output_Index(                         \
                6, (I), j_lane, i_lane)] +=                               \
            virial_factor * (force_ij.y * dz + force_ij.z * dy);          \
        shared_full_output[                                               \
            Clustered_Pairwise_Full_Output_Index(                         \
                7, (I), j_lane, i_lane)] +=                               \
            virial_factor * force_ij.z * dz;                              \
    }
#else
#define CLUSTERED_PAIRWISE_ACCUMULATE_FULL_OUTPUT(I)
#endif
#define CLUSTERED_PAIRWISE_PROCESS_I(I)                                    \
    if ((I) < cluster_i_end - cluster_i_begin)                             \
    {                                                                      \
        const unsigned int packed_bit =                                    \
            1u << static_cast<unsigned int>(jm * 8 + (I));                 \
        if ((effective_mask & packed_bit) != 0u &&                         \
            (Clustered_Get_Pair_Active_I_Mask(shift_bits, split) &        \
             (1u << static_cast<unsigned int>(I))) != 0u)                 \
        {                                                                  \
            const int cache_index = (I) * 8 + i_lane;                      \
            const int atom_i = shared_i_atom_id[cache_index];              \
            if (atom_i >= 0)                                               \
            {                                                              \
                const int shift_id = static_cast<int>(                     \
                    (shift_bits >> static_cast<unsigned int>((I) * 5)) &   \
                    31ull);                                                \
                const VECTOR pair_shift =                                  \
                    Clustered_Pairwise_Shift_Vector(shift_id, cell);       \
                const VECTOR ri = shared_i_crd[cache_index];               \
                const float dx = rj.x - ri.x - pair_shift.x;               \
                const float dy = rj.y - ri.y - pair_shift.y;               \
                const float dz = rj.z - ri.z - pair_shift.z;               \
                const float dr2 = dx * dx + dy * dy + dz * dz;             \
                if (dr2 > 0.0f && dr2 < cutoff_sq)                         \
                {                                                          \
                    const float r_value = sqrtf(dr2);                       \
                    const CLUSTERED_PAIRWISE_STORED stored =              \
                        Store_Clustered_Pairwise(                          \
                        %PARM_CALL%, atom_i, atom_j,                        \
                        shared_i_pairwise_type[cache_index],                \
                        sorted_pairwise_types[sorted_j],                    \
                        shared_i_charge[cache_index],                       \
                        sorted_charge[sorted_j], dx, dy, dz, r_value,      \
                        pme_beta, ij_factor, j_is_local, frc, atom_energy,  \
                        atom_virial, pme_atom_energy, listed_item_energy,   \
                        need_atom_energy, need_virial);                     \
                    const VECTOR force_ij = stored.force;                  \
                    if (frc != nullptr)                                    \
                    {                                                      \
                        fci_x_##I += force_ij.x;                            \
                        fci_y_##I += force_ij.y;                            \
                        fci_z_##I += force_ij.z;                            \
                        if (j_is_local)                                    \
                        {                                                  \
                            fcj_x -= force_ij.x;                           \
                            fcj_y -= force_ij.y;                           \
                            fcj_z -= force_ij.z;                           \
                        }                                                  \
                    }                                                      \
                    CLUSTERED_PAIRWISE_ACCUMULATE_FULL_OUTPUT(I)           \
                }                                                          \
            }                                                              \
        }                                                                  \
    }
            CLUSTERED_PAIRWISE_I_LOCAL_LIST(CLUSTERED_PAIRWISE_PROCESS_I)
#undef CLUSTERED_PAIRWISE_PROCESS_I
#undef CLUSTERED_PAIRWISE_ACCUMULATE_FULL_OUTPUT
            if (frc != nullptr && j_is_local)
            {
                const float reduced_component =
                    Reduce_Clustered_Pairwise_Subgroup_J_To_Component(
                        fcj_x, fcj_y, fcj_z, i_lane);
                if (i_lane < 3)
                {
                    Clustered_Pairwise_Atomic_Add_Force_Component(
                        frc, atom_j, i_lane, reduced_component);
                }
            }
#else
            for (int i_local = 0;
                 i_local < cluster_i_end - cluster_i_begin; i_local++)
            {
                const unsigned int packed_bit =
                    1u << static_cast<unsigned int>(jm * 8 + i_local);
                if ((effective_mask & packed_bit) == 0u ||
                    (Clustered_Get_Pair_Active_I_Mask(
                         shift_bits, split) &
                     (1u << static_cast<unsigned int>(i_local))) == 0u)
                {
                    continue;
                }
                const int cluster_i = cluster_i_begin + i_local;
                if (!Clustered_Lane_Is_Valid(
                        cluster_valid_masks[cluster_i], i_lane) ||
                    !Clustered_Lane_Is_Local(
                        cluster_local_masks[cluster_i], i_lane))
                {
                    continue;
                }
                const int sorted_i = cluster_offsets[cluster_i] + i_lane;
                const int atom_i = sorted_atom_ids[sorted_i];
                const int shift_id =
                    pair_shift_bits != nullptr
                        ? static_cast<int>(
                              (shift_bits >> static_cast<unsigned int>(
                                   i_local * 5)) &
                              31ull)
                        : sci_entry.shift_id;
                const VECTOR pair_shift =
                    Clustered_Pairwise_Shift_Vector(shift_id, cell);
                const VECTOR ri = sorted_crd[sorted_i];
                const float dx = rj.x - ri.x - pair_shift.x;
                const float dy = rj.y - ri.y - pair_shift.y;
                const float dz = rj.z - ri.z - pair_shift.z;
                const float dr2 =
                    dx * dx + dy * dy + dz * dz;
                if (dr2 > 0.0f && dr2 < cutoff_sq)
                {
                    const float r_value = sqrtf(dr2);
                    (void)Store_Clustered_Pairwise(
                        %PARM_CALL%, atom_i, atom_j,
                        sorted_pairwise_types[sorted_i],
                        sorted_pairwise_types[sorted_j],
                        sorted_charge[sorted_i],
                        sorted_charge[sorted_j], dx, dy, dz, r_value,
                        pme_beta, ij_factor, j_is_local, frc, atom_energy,
                        atom_virial, pme_atom_energy, listed_item_energy,
                        need_atom_energy, need_virial);
                }
            }
#endif
        }
    }

#ifdef USE_GPU
#if CLUSTERED_PAIRWISE_ENABLE_FULL_OUTPUT
    if (full_output)
    {
        __syncthreads();
        const int atom_i = shared_i_atom_id[i_cache_index];
        if (atom_i >= 0)
        {
            float output_value =
                Clustered_Pairwise_Reduce_Full_Output_Field(
                    shared_full_output, 0, j_lane, i_lane);
            if (listed_item_energy != nullptr)
            {
                atomicAdd(listed_item_energy + atom_i, output_value);
            }
            if (need_atom_energy && atom_energy != nullptr)
            {
                atomicAdd(atom_energy + atom_i, output_value);
            }
            if (pme_atom_energy != nullptr &&
                (need_atom_energy || need_virial))
            {
                output_value =
                    Clustered_Pairwise_Reduce_Full_Output_Field(
                        shared_full_output, 1, j_lane, i_lane);
                atomicAdd(pme_atom_energy + atom_i, output_value);
            }
            if (need_virial && atom_virial != nullptr)
            {
                output_value =
                    Clustered_Pairwise_Reduce_Full_Output_Field(
                        shared_full_output, 2, j_lane, i_lane);
                atomicAdd(&atom_virial[atom_i].a11, output_value);
                output_value =
                    Clustered_Pairwise_Reduce_Full_Output_Field(
                        shared_full_output, 3, j_lane, i_lane);
                atomicAdd(&atom_virial[atom_i].a21, output_value);
                output_value =
                    Clustered_Pairwise_Reduce_Full_Output_Field(
                        shared_full_output, 4, j_lane, i_lane);
                atomicAdd(&atom_virial[atom_i].a22, output_value);
                output_value =
                    Clustered_Pairwise_Reduce_Full_Output_Field(
                        shared_full_output, 5, j_lane, i_lane);
                atomicAdd(&atom_virial[atom_i].a31, output_value);
                output_value =
                    Clustered_Pairwise_Reduce_Full_Output_Field(
                        shared_full_output, 6, j_lane, i_lane);
                atomicAdd(&atom_virial[atom_i].a32, output_value);
                output_value =
                    Clustered_Pairwise_Reduce_Full_Output_Field(
                        shared_full_output, 7, j_lane, i_lane);
                atomicAdd(&atom_virial[atom_i].a33, output_value);
            }
        }
    }
#endif

    const int j_lanes_per_warp = warpSize / 8;
    const int component_lane = j_lane % j_lanes_per_warp;

#define CLUSTERED_PAIRWISE_REDUCE_I(I)                                    \
    if ((I) < cluster_i_end - cluster_i_begin)                            \
    {                                                                     \
        const int cache_index = (I) * 8 + i_lane;                         \
        const int atom_i = shared_i_atom_id[cache_index];                 \
        const bool active_i = atom_i >= 0;                                \
        const float reduced_component =                                  \
            Reduce_Clustered_Pairwise_Warp_I_To_Component(                \
                active_i ? fci_x_##I : 0.0f,                              \
                active_i ? fci_y_##I : 0.0f,                              \
                active_i ? fci_z_##I : 0.0f, i_lane, component_lane);     \
        if (frc != nullptr && active_i && component_lane < 3)             \
        {                                                                 \
            Clustered_Pairwise_Atomic_Add_Force_Component(                \
                frc, atom_i, component_lane, reduced_component);          \
        }                                                                 \
    }
    CLUSTERED_PAIRWISE_I_LOCAL_LIST(CLUSTERED_PAIRWISE_REDUCE_I)
#undef CLUSTERED_PAIRWISE_REDUCE_I
#undef CLUSTERED_PAIRWISE_I_LOCAL_LIST
#endif
}

extern "C" __global__ void clustered_pairwise_force_energy_and_virial(
    %PARM_ARGS%, int sci_numbers, int cluster_numbers,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const int* super_cluster_offsets,
    const CLUSTERED_PAIRWISE_SCI* sci_entries,
    const CLUSTERED_PAIRWISE_CJ* cjpacked_entries,
    const CLUSTERED_PAIRWISE_EXCLUSION* exclusion_entries,
    const unsigned long long* pair_shift_bits, const int* sorted_atom_ids,
    const VECTOR* sorted_crd, const float* sorted_charge,
    const int* sorted_pairwise_types, LTMatrix3 cell, float cutoff,
    float pme_beta, VECTOR* frc,
    float* atom_energy,
    LTMatrix3* atom_virial, float* pme_atom_energy,
    float* listed_item_energy, int need_atom_energy, int need_virial)
{
#ifdef USE_GPU
    extern __shared__ float shared_full_output[];
    const int sci = static_cast<int>(blockIdx.x);
    const int i_lane = static_cast<int>(threadIdx.x);
    const int j_lane = static_cast<int>(threadIdx.y);
    if (sci < sci_numbers && i_lane < 8 && j_lane < 8)
    {
        Traverse_Clustered_Pairwise_Lane(
            %PARM_CALL%, sci, i_lane, j_lane,
            static_cast<int>(blockIdx.y), static_cast<int>(gridDim.y),
            cluster_numbers,
            cluster_offsets, cluster_valid_masks, cluster_local_masks,
            super_cluster_offsets, sci_entries, cjpacked_entries,
            exclusion_entries, pair_shift_bits, sorted_atom_ids,
            sorted_crd, sorted_charge, sorted_pairwise_types,
            shared_full_output, cell, cutoff, pme_beta, frc, atom_energy,
            atom_virial, pme_atom_energy,
            listed_item_energy, need_atom_energy, need_virial);
    }
#else
#pragma omp parallel for
    for (int sci = 0; sci < sci_numbers; sci++)
    {
        for (int j_lane = 0; j_lane < 8; j_lane++)
        {
            for (int i_lane = 0; i_lane < 8; i_lane++)
            {
                Traverse_Clustered_Pairwise_Lane(
                    %PARM_CALL%, sci, i_lane, j_lane, 0, 1,
                    cluster_numbers, cluster_offsets, cluster_valid_masks,
                    cluster_local_masks, super_cluster_offsets, sci_entries,
                    cjpacked_entries, exclusion_entries, pair_shift_bits,
                    sorted_atom_ids, sorted_crd, sorted_charge,
                    sorted_pairwise_types, nullptr, cell, cutoff, pme_beta,
                    frc, atom_energy, atom_virial, pme_atom_energy,
                    listed_item_energy, need_atom_energy, need_virial);
            }
        }
    }
#endif
}
)JIT";
    const std::string PARM_CALL =
        string_join("%1%_list", ", ", {parameter_type, parameter_name});
    const std::string PARM_DEC_CLUSTERED =
        string_join("    const %0% %1% = %1%_list[atom_pairwise_type];", "\n",
                    {parameter_type, parameter_name});
    clustered_source_code = string_format(
        clustered_source_code, {{"PARM_ARGS", PARM_ARGS},
                                {"PARM_CALL", PARM_CALL},
                                {"PARM_DEC_CLUSTERED", PARM_DEC_CLUSTERED},
                                {"SOURCE_CODE", source_code},
                                {"COULOMB_CODE", ele_code}});
#ifdef USE_GPU
    const std::string clustered_force_source =
        "#define CLUSTERED_PAIRWISE_ENABLE_FULL_OUTPUT 0\n" +
        clustered_source_code;
    clustered_force_function.Compile(clustered_force_source);
#else
    clustered_force_function.Compile(clustered_source_code);
#endif
    if (!clustered_force_function.error_reason.empty())
    {
        clustered_force_function.error_reason =
            "Reason:\n" + clustered_force_function.error_reason;
        controller->Throw_SPONGE_Error(
            spongeErrorMallocFailed, "PAIRWISE_FORCE::JIT_Compile",
            clustered_force_function.error_reason.c_str());
    }
#ifdef USE_GPU
    const std::string clustered_full_source =
        "#define CLUSTERED_PAIRWISE_ENABLE_FULL_OUTPUT 1\n" +
        clustered_source_code;
    clustered_full_force_function.Compile(clustered_full_source);
    if (!clustered_full_force_function.error_reason.empty())
    {
        clustered_full_force_function.error_reason =
            "Reason:\n" + clustered_full_force_function.error_reason;
        controller->Throw_SPONGE_Error(
            spongeErrorMallocFailed, "PAIRWISE_FORCE::JIT_Compile",
            clustered_full_force_function.error_reason.c_str());
    }
#endif
}

void PAIRWISE_FORCE::Real_Initial(CONTROLLER* controller)
{
    FILE* fp = NULL;
    if (!has_native_parameters &&
        !controller->Command_Exist(this->force_name.c_str(), "in_file"))
    {
        std::string error_reason = "Reason:\n\tlisted force '" +
                                   this->force_name + "' is defined, but " +
                                   this->force_name +
                                   "_in_file is not provided\n";
        controller->Throw_SPONGE_Error(spongeErrorMissingCommand,
                                       "PAIRWISE_FORCE::Initial",
                                       error_reason.c_str());
    }
    controller->printf("    Initializing %s\n", this->force_name.c_str());
    if (!has_native_parameters)
    {
        Open_File_Safely(
            &fp, controller->Command(this->force_name.c_str(), "in_file"), "r");
    }
    if (!has_native_parameters &&
        fscanf(fp, "%d %d", &atom_numbers, &type_numbers) != 2)
    {
        std::string error_reason =
            "Reason:\n\tFail to read the number of atoms and/or types of the "
            "pairwise force '" +
            this->force_name + "'\n";
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "PAIRWISE_FORCE::Initial",
                                       error_reason.c_str());
    }
    int total_type_pairwise_numbers = type_numbers * (type_numbers + 1) / 2;
    if (has_native_parameters &&
        (atom_numbers <= 0 || type_numbers <= 0 ||
         native_atom_types.size() != static_cast<std::size_t>(atom_numbers) ||
         native_parameter_values.size() !=
             static_cast<std::size_t>(n_ij_parameter) *
                 total_type_pairwise_numbers))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "PAIRWISE_FORCE::Initial",
            "Reason:\n\tnative pairwise parameter dimensions are invalid\n");
    }
    Malloc_Safely((void**)&cpu_parameters,
                  sizeof(void*) * parameter_name.size());
    Malloc_Safely((void**)&gpu_parameters,
                  sizeof(void*) * parameter_name.size());
    clustered_launch_args = std::vector<void*>(parameter_name.size() + 24);
    Malloc_Safely((void**)&cpu_pairwise_types, sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&item_energy, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&sum_energy, sizeof(float));
    Device_Malloc_Safely((void**)&gpu_pairwise_types_local,
                         sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&clustered_sorted_atom_ids,
                         sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&clustered_sorted_crd,
                         sizeof(VECTOR) * atom_numbers);
    Device_Malloc_Safely((void**)&clustered_sorted_charge,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&clustered_sorted_pairwise_types,
                         sizeof(int) * atom_numbers);
    for (int j = 0; j < n_ij_parameter; j++)
    {
        if (parameter_type[j] == "int")
        {
            Malloc_Safely((void**)cpu_parameters + j,
                          sizeof(int) * total_type_pairwise_numbers);
        }
        else
        {
            Malloc_Safely((void**)cpu_parameters + j,
                          sizeof(float) * total_type_pairwise_numbers);
        }
        clustered_launch_args[j] = gpu_parameters + j;
    }
    for (int j = 0; j < n_ij_parameter; j++)
    {
        for (int i = 0; i < total_type_pairwise_numbers; i++)
        {
            int scanf_ret = 1;
            if (has_native_parameters)
            {
                const float value =
                    native_parameter_values[static_cast<std::size_t>(j) *
                                                total_type_pairwise_numbers +
                                            i];
                if (!std::isfinite(value)) scanf_ret = 0;
                if (parameter_type[j] == "int")
                {
                    if (std::trunc(value) != value ||
                        value < std::numeric_limits<int>::min() ||
                        value > std::numeric_limits<int>::max())
                    {
                        scanf_ret = 0;
                    }
                    else
                    {
                        ((int*)cpu_parameters[j])[i] = static_cast<int>(value);
                    }
                }
                else
                {
                    ((float*)cpu_parameters[j])[i] = value;
                }
            }
            else if (parameter_type[j] == "int")
            {
                scanf_ret = fscanf(fp, "%d", ((int*)cpu_parameters[j]) + i);
            }
            else
            {
                scanf_ret = fscanf(fp, "%f", ((float*)cpu_parameters[j]) + i);
            }
            if (scanf_ret != 1)
            {
                std::string error_reason =
                    "Reason:\n\tFail to read the parameters of the pairwise "
                    "force '" +
                    this->force_name + "'\n";
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                               "PAIRWISE_FORCE::Initial",
                                               error_reason.c_str());
            }
        }
    }
    for (int i = 0; i < atom_numbers; i++)
    {
        int scanf_ret = 1;
        if (has_native_parameters)
        {
            cpu_pairwise_types[i] = native_atom_types[i];
        }
        else
        {
            scanf_ret = fscanf(fp, "%d", cpu_pairwise_types + i);
        }
        if (scanf_ret != 1)
        {
            std::string error_reason =
                "Reason:\n\tFail to read the types of the pairwise force '" +
                this->force_name + "'\n";
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                           "PAIRWISE_FORCE::Initial",
                                           error_reason.c_str());
        }
    }
    if (fp != NULL) fclose(fp);
    for (int j = 0; j < n_ij_parameter; j++)
    {
        if (parameter_type[j] == "int")
        {
            Device_Malloc_And_Copy_Safely(
                (void**)gpu_parameters + j, cpu_parameters[j],
                sizeof(int) * total_type_pairwise_numbers);
        }
        else
        {
            Device_Malloc_And_Copy_Safely(
                (void**)gpu_parameters + j, cpu_parameters[j],
                sizeof(float) * total_type_pairwise_numbers);
        }
    }
    Device_Malloc_And_Copy_Safely((void**)&gpu_pairwise_types,
                                  cpu_pairwise_types,
                                  sizeof(int) * atom_numbers);
    deviceMemcpy(gpu_pairwise_types_local, gpu_pairwise_types,
                 sizeof(int) * atom_numbers, deviceMemcpyDeviceToDevice);
    local_atom_numbers = atom_numbers;
    total_local_numbers = atom_numbers;
    this->is_initialized = 1;
}

static void Launch_Clustered_Pairwise_Jit(
    PAIRWISE_FORCE* pairwise, const CLUSTERED_SPATIAL_VIEW& view,
    LTMatrix3 cell, float cutoff, float pme_beta, VECTOR* frc, int need_energy,
    float* atom_energy, int need_virial, LTMatrix3* atom_virial,
    float* pme_atom_energy, float* listed_item_energy)
{
    for (size_t j = 0; j < pairwise->parameter_name.size(); j++)
    {
        pairwise->clustered_launch_args[j] = pairwise->gpu_parameters + j;
    }

    float* null_float = NULL;
    VECTOR* null_vector = NULL;
    LTMatrix3* null_virial = NULL;
    const size_t base = pairwise->parameter_name.size();
    int sci_numbers = view.gmxpacked_sci_numbers;
    int cluster_numbers = view.cluster_numbers;
    const int* cluster_offsets = view.cluster_offsets;
    const unsigned int* cluster_valid_masks = view.cluster_valid_masks;
    const unsigned int* cluster_local_masks = view.cluster_local_masks;
    const int* super_cluster_offsets = view.super_cluster_offsets;
    const CLUSTERED_GMXPACKED_SCI* sci_entries = view.gmxpacked_sci;
    const CLUSTERED_GMXPACKED_CJ* cjpacked_entries = view.gmxpacked_cjpacked;
    const CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries =
        view.gmxpacked_exclusions;
    const unsigned long long* pair_shift_bits =
        reinterpret_cast<const unsigned long long*>(view.pair_shift_bits);
    int need_atom_energy_flag = need_energy ? 1 : 0;
    int need_virial_flag = need_virial ? 1 : 0;

    pairwise->clustered_launch_args[base] = &sci_numbers;
    pairwise->clustered_launch_args[base + 1] = &cluster_numbers;
    pairwise->clustered_launch_args[base + 2] = &cluster_offsets;
    pairwise->clustered_launch_args[base + 3] = &cluster_valid_masks;
    pairwise->clustered_launch_args[base + 4] = &cluster_local_masks;
    pairwise->clustered_launch_args[base + 5] = &super_cluster_offsets;
    pairwise->clustered_launch_args[base + 6] = &sci_entries;
    pairwise->clustered_launch_args[base + 7] = &cjpacked_entries;
    pairwise->clustered_launch_args[base + 8] = &exclusion_entries;
    pairwise->clustered_launch_args[base + 9] = &pair_shift_bits;
    pairwise->clustered_launch_args[base + 10] =
        &pairwise->clustered_sorted_atom_ids;
    pairwise->clustered_launch_args[base + 11] =
        &pairwise->clustered_sorted_crd;
    pairwise->clustered_launch_args[base + 12] =
        &pairwise->clustered_sorted_charge;
    pairwise->clustered_launch_args[base + 13] =
        &pairwise->clustered_sorted_pairwise_types;
    pairwise->clustered_launch_args[base + 14] = &cell;
    pairwise->clustered_launch_args[base + 15] = &cutoff;
    pairwise->clustered_launch_args[base + 16] = &pme_beta;
    pairwise->clustered_launch_args[base + 17] =
        frc != NULL ? &frc : &null_vector;
    pairwise->clustered_launch_args[base + 18] =
        need_energy ? &atom_energy : &null_float;
    pairwise->clustered_launch_args[base + 19] =
        need_virial ? &atom_virial : &null_virial;
    pairwise->clustered_launch_args[base + 20] = &pme_atom_energy;
    pairwise->clustered_launch_args[base + 21] =
        listed_item_energy != NULL ? &listed_item_energy : &null_float;
    pairwise->clustered_launch_args[base + 22] = &need_atom_energy_flag;
    pairwise->clustered_launch_args[base + 23] = &need_virial_flag;

    const int cj_partitions = pairwise_force_clustered_cj_partitions(
        sci_numbers, view.gmxpacked_cjpacked_numbers);
    const dim3 grid_size = {static_cast<unsigned int>(sci_numbers),
                            static_cast<unsigned int>(cj_partitions)};
    const dim3 block_size = {kClusteredClusterSize, kClusteredClusterSize};
    const size_t dynamic_shared_bytes =
        (need_energy || need_virial) ? sizeof(float) * 8 * 8 * 8 * 8 : 0;
#ifdef USE_GPU
    JIT_Function& launch_function =
        (need_energy || need_virial) ? pairwise->clustered_full_force_function
                                     : pairwise->clustered_force_function;
#else
    JIT_Function& launch_function = pairwise->clustered_force_function;
#endif
    launch_function(grid_size, block_size, 0, dynamic_shared_bytes,
                    pairwise->clustered_launch_args);
}

bool PAIRWISE_FORCE::Compute_Force_Clustered(
    const CLUSTERED_SPATIAL_VIEW& view, const VECTOR* crd, LTMatrix3 cell,
    LTMatrix3 rcell, float cutoff, float pme_beta, float* charge, VECTOR* frc,
    int need_energy, float* atom_energy, int need_virial,
    LTMatrix3* atom_virial, float* pme_direct_atom_energy,
    const char** failure_reason)
{
    if (!this->is_initialized || total_local_numbers <= 0)
    {
        if (failure_reason != NULL)
        {
            *failure_reason = NULL;
        }
        return true;
    }
    if (failure_reason != NULL)
    {
        *failure_reason = NULL;
    }
    if (view.gmxpacked_sci_numbers < 0 || view.gmxpacked_cjpacked_numbers < 0 ||
        view.gmxpacked_exclusion_numbers < 0 ||
        (view.gmxpacked_sci_numbers > 0 && view.gmxpacked_sci == NULL) ||
        (view.gmxpacked_cjpacked_numbers > 0 &&
         view.gmxpacked_cjpacked == NULL) ||
        (view.gmxpacked_exclusion_numbers > 0 &&
         view.gmxpacked_exclusions == NULL))
    {
        return pairwise_force_clustered_fail(
            "custom pairwise requires a valid gmxpacked clustered payload",
            failure_reason);
    }
    if (view.total_atom_numbers > atom_numbers ||
        view.total_atom_numbers != total_local_numbers)
    {
        return pairwise_force_clustered_fail(
            "custom pairwise clustered scratch capacity does not match the "
            "local/ghost domain",
            failure_reason);
    }

    Launch_Device_Kernel(
        pairwise_force_gather_clustered_fields,
        (view.total_atom_numbers + 255) / 256, 256, 0, NULL,
        view.total_atom_numbers, view.cluster_numbers, view.sort_permutation,
        view.cluster_offsets, view.cluster_centers, crd, charge,
        gpu_pairwise_types_local, cell, rcell, clustered_sorted_atom_ids,
        clustered_sorted_crd, clustered_sorted_charge,
        clustered_sorted_pairwise_types);

    float* listed_item_energy = need_energy ? this->item_energy : NULL;
    if (listed_item_energy != NULL)
    {
        deviceMemset(this->item_energy, 0, sizeof(float) * local_atom_numbers);
    }
    float* pme_ptr = NULL;
    if (this->with_ele && pme_direct_atom_energy != NULL)
    {
        pme_ptr = pme_direct_atom_energy;
        deviceMemset(pme_ptr, 0, sizeof(float) * local_atom_numbers);
    }

    if (view.gmxpacked_sci_numbers > 0 && view.gmxpacked_cjpacked_numbers > 0)
    {
        Launch_Clustered_Pairwise_Jit(this, view, cell, cutoff, pme_beta, frc,
                                      need_energy, atom_energy, need_virial,
                                      atom_virial, pme_ptr, listed_item_energy);
    }

    if (need_energy)
    {
        Sum_Of_List(item_energy, sum_energy, local_atom_numbers);
        deviceMemcpy(&last_energy, sum_energy, sizeof(float),
                     deviceMemcpyDeviceToHost);
    }
    else
    {
        last_energy = 0.0f;
    }
    if (failure_reason != NULL)
    {
        *failure_reason = NULL;
    }
    return true;
}

void PAIRWISE_FORCE::Get_Local(int* atom_local, int local_atom_numbers,
                               int ghost_numbers, char* atom_local_label,
                               int* atom_local_id)
{
    (void)atom_local_label;
    (void)atom_local_id;
    if (!is_initialized) return;
    int total = local_atom_numbers + ghost_numbers;
    if (total <= 0) return;
    this->local_atom_numbers = local_atom_numbers;
    this->total_local_numbers = total;
    Launch_Device_Kernel(pairwise_force_scatter_types, (total + 255) / 256, 256,
                         0, NULL, total, atom_local, gpu_pairwise_types,
                         gpu_pairwise_types_local);
}

void PAIRWISE_FORCE::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized || !is_controller_printf_initialized) return;
    if (CONTROLLER::MPI_rank >= CONTROLLER::PP_MPI_size) return;
    h_energy = last_energy;
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &h_energy, 1, MPI_FLOAT, MPI_SUM,
                  CONTROLLER::pp_comm);
#endif
    controller->Step_Print(this->force_name.c_str(), &h_energy, true);
}
