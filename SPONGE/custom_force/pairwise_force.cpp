#include "pairwise_force.h"

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
        int cluster_lo = 0;
        int cluster_hi = cluster_numbers;
        while (cluster_lo + 1 < cluster_hi)
        {
            const int cluster_mid = (cluster_lo + cluster_hi) >> 1;
            if (cluster_offsets[cluster_mid] <= sorted_i)
            {
                cluster_lo = cluster_mid;
            }
            else
            {
                cluster_hi = cluster_mid;
            }
        }
        const int atom_i = sort_permutation[sorted_i];
        const VECTOR center = cluster_centers[cluster_lo];
        const VECTOR atom_crd = crd[atom_i];
        sorted_atom_ids[sorted_i] = atom_i;
        sorted_crd[sorted_i] =
            center +
            Get_Periodic_Displacement(atom_crd, center, cell, rcell);
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
    const int partitions =
        (average_cj_per_sci + target_cj_per_partition - 1) /
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
    if (controller->Command_Exist(this->module_name, "in_file"))
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
    std::string full_source_code = R"JIT(#if defined(__CUDACC__)
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
#ifndef USE_GPU
__forceinline__ float atomicAdd(float* x, float y)
{
    float x0;
#ifdef _WIN32
#pragma omp critical(sponge_jit_atomic_add_float)
#else
#pragma omp atomic capture
#endif
    {
        x0 = *x;
        *x += y;
    }
    return x0;
}
__forceinline__ int atomicAdd(int* x, int y)
{
    int x0;
#ifdef _WIN32
#pragma omp critical(sponge_jit_atomic_add_int)
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
__device__ __forceinline__ int Get_Pairwise_Type(int a, int b)
{
    int y = (b - a);
    int x = y >> 31;
    y = (y ^ x) - x;
    x = b + a;
    int z = (x + y) >> 1;
    x = (x - y) >> 1;
    return (z * (z + 1) >> 1) + x;
}
extern "C" __global__ void pairwise_force_energy_and_virial(%PARM_ARGS%,
    const float* charge, const float pme_beta, ATOM_GROUP* nl, const int* pairwise_types,
    const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell, const float cutoff,
    VECTOR* frc, float* atom_energy, LTMatrix3* atom_virial, float* pme_atom_energy,
    float* listed_item_energy, const int local_atom_numbers, int need_atom_energy,
    int need_virial, int atom_numbers)
{
#ifdef USE_GPU
    int atom_i = blockDim.y * blockIdx.x + threadIdx.y;
    if (atom_i < atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
#ifdef USE_GPU
        if (atom_i >= local_atom_numbers) return;
#else
        if (atom_i >= local_atom_numbers) continue;
#endif
        ATOM_GROUP nl_i = nl[atom_i];
        VECTOR r1 = crd[atom_i];
        int pairwise_type_i = pairwise_types[atom_i];
        VECTOR frc_record = { 0.0f, 0.0f, 0.0f };
        LTMatrix3 virial_record = {0, 0, 0, 0, 0, 0};
        float energy_total = 0.0f;
        float energy_coulomb = 0.0f;
        float charge_i = 0;
        float charge_j = 0;
        if (pme_atom_energy != nullptr)
        {
            charge_i = charge[atom_i];
        }
#ifdef USE_GPU
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
#else
        for (int j = 0; j < nl_i.atom_numbers; j++)
#endif
        {
            int atom_j = nl_i.atom_serial[j];
            float ij_factor = atom_j < local_atom_numbers ? 1.0f : 0.5f;
            VECTOR vector_dr = Get_Periodic_Displacement(crd[atom_j], r1, cell, rcell);
            float float_dr_ij =
                sqrtf(vector_dr.x * vector_dr.x + vector_dr.y * vector_dr.y +
                      vector_dr.z * vector_dr.z);
            if (float_dr_ij < cutoff)
            {
                int atom_pairwise_type = Get_Pairwise_Type(pairwise_type_i, pairwise_types[atom_j]);
                %PARM_DEC%
                SADfloat<1> r_ij(float_dr_ij, 0);
                SADfloat<1> E, E_ele;
                %SOURCE_CODE%
                energy_total += ij_factor * E.val;
                if (pme_atom_energy != nullptr)
                {
                    charge_j = charge[atom_j];
                    %COULOMB_CODE%
                    energy_coulomb += ij_factor * E_ele.val;
                }
                float frc_abs = E.dval[0] / float_dr_ij;
                if (pme_atom_energy != nullptr)
                {
                    frc_abs += E_ele.dval[0] / float_dr_ij;
                }
                VECTOR frc_temp = frc_abs * vector_dr;
                if (frc != nullptr)
                {
                    frc_record = frc_record + frc_temp;
                    if (atom_j < local_atom_numbers)
                        atomicAdd(frc + atom_j, -frc_temp);
                }
                if (need_virial && atom_virial != nullptr)
                {
                    virial_record = virial_record -
                        ij_factor * Get_Virial_From_Force_Dis(frc_temp, vector_dr);
                }
            }
        }
        if (frc != nullptr)
        {
            Warp_Sum_To(frc + atom_i, frc_record, warpSize);
        }
        if (pme_atom_energy != nullptr && (need_atom_energy || need_virial))
        {
            float energy_coulomb_sum = energy_coulomb;
            Warp_Sum_To(pme_atom_energy + atom_i, energy_coulomb_sum, warpSize);
        }
        if (listed_item_energy != nullptr)
        {
            float listed_energy_sum = energy_total;
            Warp_Sum_To(listed_item_energy + atom_i, listed_energy_sum, warpSize);
        }
        if (need_atom_energy && atom_energy != nullptr)
        {
            float atom_energy_sum = energy_total;
            Warp_Sum_To(atom_energy + atom_i, atom_energy_sum, warpSize);
        }
        if (need_virial && atom_virial != nullptr)
        {
            Warp_Sum_To(&(atom_virial + atom_i)->a11, virial_record.a11, warpSize);
            Warp_Sum_To(&(atom_virial + atom_i)->a21, virial_record.a21, warpSize);
            Warp_Sum_To(&(atom_virial + atom_i)->a22, virial_record.a22, warpSize);
            Warp_Sum_To(&(atom_virial + atom_i)->a31, virial_record.a31, warpSize);
            Warp_Sum_To(&(atom_virial + atom_i)->a32, virial_record.a32, warpSize);
            Warp_Sum_To(&(atom_virial + atom_i)->a33, virial_record.a33, warpSize);
        }
    }
}
)JIT";
    std::string PARM_ARGS = string_join("const %0%* %1%_list", ", ",
                                        {parameter_type, parameter_name});
    std::string PARM_DEC = string_join(
        "                const %0% %1% = %1%_list[atom_pairwise_type];", "\n",
        {parameter_type, parameter_name});
    full_source_code =
        string_format(full_source_code, {{"PARM_ARGS", PARM_ARGS},
                                         {"PARM_DEC", PARM_DEC},
                                         {"SOURCE_CODE", source_code},
                                         {"COULOMB_CODE", ele_code}});
    force_function.Compile(full_source_code);
    if (!force_function.error_reason.empty())
    {
        force_function.error_reason = "Reason:\n" + force_function.error_reason;
        controller->Throw_SPONGE_Error(spongeErrorMallocFailed,
                                       "PAIRWISE_FORCE::JIT_Compile",
                                       force_function.error_reason.c_str());
    }

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

struct CLUSTERED_PAIRWISE_NATIVE_IMEI
{
    unsigned int imask;
    int exclusion_index[32];
};

struct CLUSTERED_PAIRWISE_NATIVE_CJ
{
    int cj[4];
    CLUSTERED_PAIRWISE_NATIVE_IMEI imei[2];
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
    const unsigned int i_lane_bit =
        1u << static_cast<unsigned int>(i_lane);
    const unsigned int j_lane_bit =
        1u << static_cast<unsigned int>(j_lane);
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
        (cluster_valid_masks[cache_cluster_i] & i_lane_bit) != 0u &&
        (cluster_local_masks[cache_cluster_i] & i_lane_bit) != 0u)
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
            if (cluster_j < 0 ||
                (cluster_valid_masks[cluster_j] & j_lane_bit) == 0u)
            {
                continue;
            }
            const int sorted_j = cluster_offsets[cluster_j] + j_lane;
            const int atom_j = sorted_atom_ids[sorted_j];
            const VECTOR rj = sorted_crd[sorted_j];
            const bool j_is_local =
                (cluster_local_masks[cluster_j] & j_lane_bit) != 0u;
            const float ij_factor = j_is_local ? 1.0f : 0.5f;
            const unsigned long long shift_bits =
                pair_shift_bits[packed_idx * 4 + jm];

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
        if ((effective_mask & packed_bit) != 0u)                           \
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
                const int shift_id = static_cast<int>(
                    (shift_bits >> static_cast<unsigned int>(i_local * 5)) &
                    31ull);
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

__device__ __forceinline__ void Traverse_Clustered_Pairwise_Native_Sci(
    %PARM_ARGS%, int sci, int cluster_numbers,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const int* super_cluster_offsets,
    const CLUSTERED_PAIRWISE_SCI* sci_entries,
    const CLUSTERED_PAIRWISE_NATIVE_CJ* cjpacked_entries,
    const unsigned long long* exclusion_mask_pool,
    const int* sorted_atom_ids, const VECTOR* sorted_crd,
    const float* sorted_charge, const int* sorted_pairwise_types,
    LTMatrix3 cell, float cutoff, float pme_beta,
    VECTOR* frc, float* atom_energy, LTMatrix3* atom_virial,
    float* pme_atom_energy, float* listed_item_energy,
    int need_atom_energy, int need_virial)
{
    const CLUSTERED_PAIRWISE_SCI sci_entry = sci_entries[sci];
    const int super_i = sci_entry.supercluster_id;
    const int cluster_i_begin = super_cluster_offsets[super_i];
    int cluster_i_end = super_cluster_offsets[super_i + 1];
    if (cluster_i_end > cluster_numbers)
    {
        cluster_i_end = cluster_numbers;
    }
    const bool central_sci = sci_entry.shift_id == 13;
    const float cutoff_sq = cutoff * cutoff;

    for (int cluster_i = cluster_i_begin; cluster_i < cluster_i_end;
         cluster_i++)
    {
        const int i_local = cluster_i - cluster_i_begin;
        const unsigned int valid_mask_i =
            cluster_valid_masks[cluster_i];
        const unsigned int local_mask_i =
            cluster_local_masks[cluster_i];
        for (int i_lane = 0; i_lane < 8; i_lane++)
        {
            const unsigned int i_lane_bit =
                1u << static_cast<unsigned int>(i_lane);
            if ((valid_mask_i & i_lane_bit) == 0u ||
                (local_mask_i & i_lane_bit) == 0u)
            {
                continue;
            }
            const int sorted_i = cluster_offsets[cluster_i] + i_lane;
            const int atom_i = sorted_atom_ids[sorted_i];
            const VECTOR ri = sorted_crd[sorted_i];

            for (int packed_idx = sci_entry.cjpacked_begin;
                 packed_idx < sci_entry.cjpacked_end; packed_idx++)
            {
                const CLUSTERED_PAIRWISE_NATIVE_CJ packed =
                    cjpacked_entries[packed_idx];
                for (int jm = 0; jm < 4; jm++)
                {
                    const int cluster_j = packed.cj[jm];
                    if (cluster_j < 0)
                    {
                        continue;
                    }
                    const unsigned int imask =
                        ((packed.imei[0].imask |
                          packed.imei[1].imask) >>
                         static_cast<unsigned int>(jm * 8)) &
                        0xffu;
                    if ((imask &
                         (1u << static_cast<unsigned int>(i_local))) == 0u)
                    {
                        continue;
                    }
                    int exclusion_index = -1;
                    for (int split = 0; split < 2; split++)
                    {
                        const int candidate =
                            packed.imei[split]
                                .exclusion_index[jm * 8 + i_local];
                        if (candidate >= 0)
                        {
                            exclusion_index = candidate;
                            break;
                        }
                    }
                    const unsigned long long exclusion_mask =
                        exclusion_index >= 0 &&
                                exclusion_mask_pool != nullptr
                            ? exclusion_mask_pool[exclusion_index]
                            : 0ull;
                    const VECTOR pair_shift =
                        Clustered_Pairwise_Shift_Vector(
                            sci_entry.shift_id, cell);
                    const unsigned int valid_mask_j =
                        cluster_valid_masks[cluster_j];
                    const unsigned int local_mask_j =
                        cluster_local_masks[cluster_j];
                    for (int j_lane = 0; j_lane < 8; j_lane++)
                    {
                        const unsigned int j_lane_bit =
                            1u << static_cast<unsigned int>(j_lane);
                        if ((valid_mask_j & j_lane_bit) == 0u)
                        {
                            continue;
                        }
                        const bool j_is_local =
                            (local_mask_j & j_lane_bit) != 0u;
                        if (central_sci && cluster_i == cluster_j &&
                            j_is_local && j_lane <= i_lane)
                        {
                            continue;
                        }
                        if ((exclusion_mask &
                             (1ull << static_cast<unsigned int>(
                                  i_lane * 8 + j_lane))) != 0ull)
                        {
                            continue;
                        }
                        const int sorted_j =
                            cluster_offsets[cluster_j] + j_lane;
                        const int atom_j = sorted_atom_ids[sorted_j];
                        const VECTOR rj = sorted_crd[sorted_j];
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
                                sorted_charge[sorted_j], dx, dy, dz,
                                r_value, pme_beta,
                                j_is_local ? 1.0f : 0.5f, j_is_local, frc,
                                atom_energy, atom_virial, pme_atom_energy,
                                listed_item_energy, need_atom_energy,
                                need_virial);
                        }
                    }
                }
            }
        }
    }
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
    const int* sorted_pairwise_types,
    const CLUSTERED_PAIRWISE_SCI* native_sci_entries,
    const CLUSTERED_PAIRWISE_NATIVE_CJ* native_cjpacked_entries,
    const unsigned long long* native_exclusion_mask_pool,
    LTMatrix3 cell, int use_gmxpacked, float cutoff, float pme_beta, VECTOR* frc,
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
        if (use_gmxpacked)
        {
            for (int j_lane = 0; j_lane < 8; j_lane++)
            {
                for (int i_lane = 0; i_lane < 8; i_lane++)
                {
                    Traverse_Clustered_Pairwise_Lane(
                        %PARM_CALL%, sci, i_lane, j_lane, 0, 1,
                        cluster_numbers,
                        cluster_offsets, cluster_valid_masks,
                        cluster_local_masks, super_cluster_offsets,
                        sci_entries, cjpacked_entries, exclusion_entries,
                        pair_shift_bits, sorted_atom_ids, sorted_crd,
                        sorted_charge, sorted_pairwise_types, nullptr, cell,
                        cutoff, pme_beta, frc, atom_energy, atom_virial,
                        pme_atom_energy, listed_item_energy,
                        need_atom_energy, need_virial);
                }
            }
        }
        else
        {
            Traverse_Clustered_Pairwise_Native_Sci(
                %PARM_CALL%, sci, cluster_numbers, cluster_offsets,
                cluster_valid_masks, cluster_local_masks,
                super_cluster_offsets, native_sci_entries,
                native_cjpacked_entries, native_exclusion_mask_pool,
                sorted_atom_ids, sorted_crd, sorted_charge,
                sorted_pairwise_types, cell, cutoff, pme_beta, frc,
                atom_energy, atom_virial, pme_atom_energy,
                listed_item_energy, need_atom_energy, need_virial);
        }
    }
#endif
}
)JIT";
    const std::string PARM_CALL =
        string_join("%1%_list", ", ", {parameter_type, parameter_name});
    const std::string PARM_DEC_CLUSTERED = string_join(
        "    const %0% %1% = %1%_list[atom_pairwise_type];", "\n",
        {parameter_type, parameter_name});
    clustered_source_code = string_format(
        clustered_source_code,
        {{"PARM_ARGS", PARM_ARGS},
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
    FILE* fp;
    if (!controller->Command_Exist(this->force_name.c_str(), "in_file"))
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
    Open_File_Safely(
        &fp, controller->Command(this->force_name.c_str(), "in_file"), "r");
    if (fscanf(fp, "%d %d", &atom_numbers, &type_numbers) != 2)
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
    Malloc_Safely((void**)&cpu_parameters,
                  sizeof(void*) * parameter_name.size());
    Malloc_Safely((void**)&gpu_parameters,
                  sizeof(void*) * parameter_name.size());
    launch_args = std::vector<void*>(parameter_name.size() + 17);
    clustered_launch_args =
        std::vector<void*>(parameter_name.size() + 28);
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
        launch_args[j] = gpu_parameters + j;
        clustered_launch_args[j] = gpu_parameters + j;
    }
    for (int j = 0; j < n_ij_parameter; j++)
    {
        for (int i = 0; i < total_type_pairwise_numbers; i++)
        {
            int scanf_ret = 0;
            if (parameter_type[j] == "int")
            {
                scanf_ret = fscanf(fp, "%d", ((int*)cpu_parameters[j]) + i);
            }
            else
            {
                scanf_ret = fscanf(fp, "%f", ((float*)cpu_parameters[j]) + i);
            }
            if (scanf_ret == 0)
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
        if (fscanf(fp, "%d", cpu_pairwise_types + i) != 1)
        {
            std::string error_reason =
                "Reason:\n\tFail to read the types of the pairwise force '" +
                this->force_name + "'\n";
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                           "PAIRWISE_FORCE::Initial",
                                           error_reason.c_str());
        }
    }
    fclose(fp);
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

void PAIRWISE_FORCE::Compute_Force(ATOM_GROUP* nl, const VECTOR* crd,
                                   LTMatrix3 cell, LTMatrix3 rcell,
                                   float cutoff, float pme_beta, float* charge,
                                   VECTOR* frc, int need_energy,
                                   float* atom_energy, int need_virial,
                                   LTMatrix3* atom_virial,
                                   float* pme_direct_atom_energy)
{
    if (!this->is_initialized || total_local_numbers <= 0) return;

    float* listed_item_energy = need_energy ? this->item_energy : NULL;
    if (listed_item_energy != NULL)
    {
        deviceMemset(this->item_energy, 0, sizeof(float) * local_atom_numbers);
    }
    float* NULLPTR = NULL;
    LTMatrix3* NULL_VIRIAL = NULL;
    launch_args[parameter_name.size()] = &charge;
    launch_args[parameter_name.size() + 1] = &pme_beta;
    launch_args[parameter_name.size() + 2] = &nl;
    launch_args[parameter_name.size() + 3] = &gpu_pairwise_types_local;
    launch_args[parameter_name.size() + 4] = &crd;
    launch_args[parameter_name.size() + 5] = &cell;
    launch_args[parameter_name.size() + 6] = &rcell;
    launch_args[parameter_name.size() + 7] = &cutoff;
    launch_args[parameter_name.size() + 8] = &frc;
    launch_args[parameter_name.size() + 9] =
        need_energy ? &atom_energy : &NULLPTR;
    launch_args[parameter_name.size() + 10] =
        need_virial ? &atom_virial : &NULL_VIRIAL;
    float* pme_ptr = NULLPTR;
    if (this->with_ele && pme_direct_atom_energy != NULL)
    {
        pme_ptr = pme_direct_atom_energy;
        deviceMemset(pme_ptr, 0, sizeof(float) * local_atom_numbers);
    }
    launch_args[parameter_name.size() + 11] = &pme_ptr;
    if (listed_item_energy != NULL)
    {
        launch_args[parameter_name.size() + 12] = &listed_item_energy;
    }
    else
    {
        launch_args[parameter_name.size() + 12] = &NULLPTR;
    }
    int local_atom_numbers_flag = local_atom_numbers;
    int need_atom_energy_flag = need_energy ? 1 : 0;
    int need_virial_flag = need_virial ? 1 : 0;
    int total_numbers_flag = total_local_numbers;
    launch_args[parameter_name.size() + 13] = &local_atom_numbers_flag;
    launch_args[parameter_name.size() + 14] = &need_atom_energy_flag;
    launch_args[parameter_name.size() + 15] = &need_virial_flag;
    launch_args[parameter_name.size() + 16] = &total_numbers_flag;

    dim3 blockSize = {CONTROLLER::device_warp,
                      CONTROLLER::device_max_thread / CONTROLLER::device_warp};
    dim3 gridSize = (total_local_numbers + blockSize.y - 1) / blockSize.y;
    force_function(gridSize, blockSize, 0, 0, launch_args);

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

    CLUSTERED_SPATIAL_VIEW_REQUIREMENTS requirements;
    requirements.local_atom_numbers = local_atom_numbers;
    requirements.ghost_numbers =
        total_local_numbers - local_atom_numbers;
    requirements.cutoff = cutoff;
    requirements.provider_incarnation = view.provider_incarnation;
    requirements.lease_epoch = view.lease_epoch;
    requirements.require_all_local_atoms = true;
#ifdef USE_CPU
    requirements.native_payload_generation =
        view.native_payload_generation;
    requirements.require_backend = true;
    requirements.backend = CLUSTERED_SPATIAL_BACKEND::CPU;
    requirements.require_native_payload = true;
    requirements.require_pair_shift_metadata = false;
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
    requirements.consumer_stream = nullptr;
    requirements.require_gmxpacked_payload = true;
    requirements.require_pair_shift_metadata = true;
#endif
    if (!Clustered_Validate_Spatial_View(view, requirements, failure_reason))
    {
        return false;
    }
#ifdef USE_CPU
    if (view.sci_numbers <= 0 || view.cjpacked_numbers <= 0 ||
        view.sci == NULL || view.cjpacked == NULL ||
        (view.exclusion_pool_numbers > 0 &&
         view.exclusion_mask_pool == NULL))
    {
        return pairwise_force_clustered_fail(
            "CPU custom pairwise requires the native clustered payload",
            failure_reason);
    }
#else
    if (view.gmxpacked_sci_numbers <= 0 ||
        view.gmxpacked_cjpacked_numbers <= 0 ||
        view.gmxpacked_sci == NULL || view.gmxpacked_cjpacked == NULL ||
        view.gmxpacked_exclusions == NULL)
    {
        return pairwise_force_clustered_fail(
            "custom pairwise requires the gmxpacked clustered payload",
            failure_reason);
    }
#endif
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
        deviceMemset(this->item_energy, 0,
                     sizeof(float) * local_atom_numbers);
    }
    float* null_float = NULL;
    VECTOR* null_vector = NULL;
    LTMatrix3* null_virial = NULL;
    float* pme_ptr = NULL;
    if (this->with_ele && pme_direct_atom_energy != NULL)
    {
        pme_ptr = pme_direct_atom_energy;
        deviceMemset(pme_ptr, 0, sizeof(float) * local_atom_numbers);
    }

    for (size_t j = 0; j < parameter_name.size(); j++)
    {
        clustered_launch_args[j] = gpu_parameters + j;
    }
    const size_t base = parameter_name.size();
#ifdef USE_CPU
    int sci_numbers_flag = view.sci_numbers;
    int use_gmxpacked_flag = 0;
#else
    int sci_numbers_flag = view.gmxpacked_sci_numbers;
    int use_gmxpacked_flag = 1;
#endif
    int cluster_numbers_flag = view.cluster_numbers;
    const int* cluster_offsets = view.cluster_offsets;
    const unsigned int* cluster_valid_masks = view.cluster_valid_masks;
    const unsigned int* cluster_local_masks = view.cluster_local_masks;
    const int* super_cluster_offsets = view.super_cluster_offsets;
    const CLUSTERED_GMXPACKED_SCI* sci_entries = view.gmxpacked_sci;
    const CLUSTERED_GMXPACKED_CJ* cjpacked_entries =
        view.gmxpacked_cjpacked;
    const CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries =
        view.gmxpacked_exclusions;
    const unsigned long long* pair_shift_bits =
        reinterpret_cast<const unsigned long long*>(view.pair_shift_bits);
    const CLUSTERED_SCI* native_sci_entries = view.sci;
    const CLUSTERED_CJ_PACKED* native_cjpacked_entries = view.cjpacked;
    const unsigned long long* native_exclusion_mask_pool =
        view.exclusion_mask_pool;
    int need_atom_energy_flag = need_energy ? 1 : 0;
    int need_virial_flag = need_virial ? 1 : 0;

    clustered_launch_args[base] = &sci_numbers_flag;
    clustered_launch_args[base + 1] = &cluster_numbers_flag;
    clustered_launch_args[base + 2] = &cluster_offsets;
    clustered_launch_args[base + 3] = &cluster_valid_masks;
    clustered_launch_args[base + 4] = &cluster_local_masks;
    clustered_launch_args[base + 5] = &super_cluster_offsets;
    clustered_launch_args[base + 6] = &sci_entries;
    clustered_launch_args[base + 7] = &cjpacked_entries;
    clustered_launch_args[base + 8] = &exclusion_entries;
    clustered_launch_args[base + 9] = &pair_shift_bits;
    clustered_launch_args[base + 10] = &clustered_sorted_atom_ids;
    clustered_launch_args[base + 11] = &clustered_sorted_crd;
    clustered_launch_args[base + 12] = &clustered_sorted_charge;
    clustered_launch_args[base + 13] =
        &clustered_sorted_pairwise_types;
    clustered_launch_args[base + 14] = &native_sci_entries;
    clustered_launch_args[base + 15] = &native_cjpacked_entries;
    clustered_launch_args[base + 16] = &native_exclusion_mask_pool;
    clustered_launch_args[base + 17] = &cell;
    clustered_launch_args[base + 18] = &use_gmxpacked_flag;
    clustered_launch_args[base + 19] = &cutoff;
    clustered_launch_args[base + 20] = &pme_beta;
    clustered_launch_args[base + 21] =
        frc != NULL ? &frc : &null_vector;
    clustered_launch_args[base + 22] =
        need_energy ? &atom_energy : &null_float;
    clustered_launch_args[base + 23] =
        need_virial ? &atom_virial : &null_virial;
    clustered_launch_args[base + 24] = &pme_ptr;
    clustered_launch_args[base + 25] =
        listed_item_energy != NULL ? &listed_item_energy : &null_float;
    clustered_launch_args[base + 26] = &need_atom_energy_flag;
    clustered_launch_args[base + 27] = &need_virial_flag;

    const dim3 block_size = {kClusteredClusterSize,
                             kClusteredClusterSize};
    const int cj_partitions = pairwise_force_clustered_cj_partitions(
        sci_numbers_flag, view.gmxpacked_cjpacked_numbers);
    const dim3 grid_size = {static_cast<unsigned int>(sci_numbers_flag),
                            static_cast<unsigned int>(cj_partitions)};
    const size_t dynamic_shared_bytes =
        (need_energy || need_virial) ? sizeof(float) * 8 * 8 * 8 * 8 : 0;
#ifdef USE_GPU
    JIT_Function& clustered_launch_function =
        (need_energy || need_virial) ? clustered_full_force_function
                                     : clustered_force_function;
#else
    JIT_Function& clustered_launch_function = clustered_force_function;
#endif
    clustered_launch_function(grid_size, block_size, 0, dynamic_shared_bytes,
                              clustered_launch_args);

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

float PAIRWISE_FORCE::Get_Energy(ATOM_GROUP* nl, const VECTOR* crd,
                                 LTMatrix3 cell, LTMatrix3 rcell, float cutoff,
                                 float pme_beta, float* charge,
                                 float* pme_direct_atom_energy)
{
    if (!this->is_initialized || total_local_numbers <= 0) return 0;

    deviceMemset(this->item_energy, 0, sizeof(float) * local_atom_numbers);
    float* NULLPTR = NULL;
    LTMatrix3* NULL_VIRIAL = NULL;
    launch_args[parameter_name.size()] = &charge;
    launch_args[parameter_name.size() + 1] = &pme_beta;
    launch_args[parameter_name.size() + 2] = &nl;
    launch_args[parameter_name.size() + 3] = &gpu_pairwise_types_local;
    launch_args[parameter_name.size() + 4] = &crd;
    launch_args[parameter_name.size() + 5] = &cell;
    launch_args[parameter_name.size() + 6] = &rcell;
    launch_args[parameter_name.size() + 7] = &cutoff;
    launch_args[parameter_name.size() + 8] = &NULLPTR;
    launch_args[parameter_name.size() + 9] = &item_energy;
    launch_args[parameter_name.size() + 10] = &NULL_VIRIAL;
    float* pme_ptr = NULLPTR;
    if (this->with_ele && pme_direct_atom_energy != NULL)
    {
        pme_ptr = pme_direct_atom_energy;
        deviceMemset(pme_ptr, 0, sizeof(float) * local_atom_numbers);
    }
    launch_args[parameter_name.size() + 11] = &pme_ptr;
    launch_args[parameter_name.size() + 12] = &item_energy;
    int local_atom_numbers_flag = local_atom_numbers;
    int need_atom_energy_flag = 1;
    int need_virial_flag = 0;
    int total_numbers_flag = total_local_numbers;
    launch_args[parameter_name.size() + 13] = &local_atom_numbers_flag;
    launch_args[parameter_name.size() + 14] = &need_atom_energy_flag;
    launch_args[parameter_name.size() + 15] = &need_virial_flag;
    launch_args[parameter_name.size() + 16] = &total_numbers_flag;
    dim3 blockSize = {CONTROLLER::device_warp,
                      CONTROLLER::device_max_thread / CONTROLLER::device_warp};
    dim3 gridSize = (total_local_numbers + blockSize.y - 1) / blockSize.y;
    force_function(gridSize, blockSize, 0, 0, launch_args);
    Sum_Of_List(item_energy, sum_energy, local_atom_numbers);
    float h_energy = NAN;
    deviceMemcpy(&h_energy, sum_energy, sizeof(float),
                 deviceMemcpyDeviceToHost);
    return h_energy;
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
