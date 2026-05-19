#include "REST2.h"

namespace
{

bool REST2_Mode_Is_Enabled(CONTROLLER* controller)
{
    if (!controller->Command_Exist("REST2", "mode"))
    {
        return false;
    }
    if (controller->Command_Choice("REST2", "mode", "off") ||
        controller->Command_Choice("REST2", "mode", "none") ||
        controller->Command_Choice("REST2", "mode", "false"))
    {
        return false;
    }
    return true;
}

static __global__ void REST2_Get_Local_Device(int* atom_local,
                                              int local_atom_numbers,
                                              int ghost_numbers,
                                              int* atom_sys_mark,
                                              int* atom_sys_mark_local)
{
    int total = local_atom_numbers + ghost_numbers;
    SIMPLE_DEVICE_FOR(i, total)
    {
        atom_sys_mark_local[i] = atom_sys_mark[atom_local[i]];
    }
}

template <bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb>
static __global__ void REST2_Lennard_Jones_And_Direct_Coulomb_Device(
    const int local_atom_numbers, const int solvent_numbers,
    const ATOM_GROUP* nl, const VECTOR_LJ* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* LJ_type_A, const float* LJ_type_B,
    const int* atom_sys_mark, const float cutoff, VECTOR* frc,
    const float pme_beta, float* atom_energy, LTMatrix3* atom_virial,
    float* atom_direct_cf_energy, float* atom_LJ_ene,
    float* rest2_unscaled_atom_energy, float* rest2_effective_atom_energy,
    const float lambda_m, const float sqrt_lambda_m)
{
#ifdef USE_GPU
    int atom_i = blockDim.y * blockIdx.x + threadIdx.y;
    if (atom_i < local_atom_numbers - solvent_numbers)
#else
#pragma omp parallel for schedule(dynamic)
    for (int atom_i = 0; atom_i < local_atom_numbers - solvent_numbers;
         atom_i++)
#endif
    {
        VECTOR frc_record = {0.0f, 0.0f, 0.0f};
        LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float energy_lj = 0.0f;
        float energy_coulomb = 0.0f;
        float rest2_unscaled = 0.0f;
        float rest2_effective = 0.0f;
        ATOM_GROUP nl_i = nl[atom_i];
        VECTOR_LJ r1 = crd[atom_i];
        int atom_mark_i = atom_sys_mark[atom_i];
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
                int atom_pair_LJ_type = Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                float A = LJ_type_A[atom_pair_LJ_type];
                float B = LJ_type_B[atom_pair_LJ_type];
                int mark_sum = atom_mark_i + atom_sys_mark[atom_j];
                float scale = 1.0f;
                if (mark_sum == 0)
                {
                    scale = lambda_m;
                }
                else if (mark_sum == 1)
                {
                    scale = sqrt_lambda_m;
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
                    VECTOR frc_lin = scale * frc_abs * dr;
                    frc_record = frc_record + frc_lin;
                    if (atom_j < local_atom_numbers)
                    {
                        atomicAdd(frc + atom_j, -frc_lin);
                    }
                    if (need_virial)
                    {
                        virial = virial - ij_factor * Get_Virial_From_Force_Dis(
                                                          frc_lin, dr);
                    }
                }
                if (need_energy)
                {
                    float pair_lj = Get_LJ_Energy(r1, r2, dr_abs, A, B);
                    float pair_coulomb = 0.0f;
                    if (need_coulomb)
                    {
                        pair_coulomb =
                            Get_Direct_Coulomb_Energy(r1, r2, dr_abs, pme_beta);
                    }
                    float pair_total = pair_lj + pair_coulomb;
                    energy_lj += ij_factor * scale * pair_lj;
                    energy_coulomb += ij_factor * scale * pair_coulomb;
                    if (mark_sum < 2)
                    {
                        rest2_unscaled += ij_factor * pair_total;
                        rest2_effective += ij_factor * scale * pair_total;
                    }
                }
            }
        }
        if (need_force)
        {
            Warp_Sum_To(frc + atom_i, frc_record, warpSize);
        }
        if (need_energy)
        {
            float energy_total = energy_lj + energy_coulomb;
            Warp_Sum_To(atom_energy + atom_i, energy_total, warpSize);
            Warp_Sum_To(atom_LJ_ene + atom_i, energy_lj, warpSize);
            if (need_coulomb)
            {
                Warp_Sum_To(atom_direct_cf_energy + atom_i, energy_coulomb,
                            warpSize);
            }
            Warp_Sum_To(rest2_unscaled_atom_energy + atom_i, rest2_unscaled,
                        warpSize);
            Warp_Sum_To(rest2_effective_atom_energy + atom_i, rest2_effective,
                        warpSize);
        }
        if (need_virial)
        {
            Warp_Sum_To(atom_virial + atom_i, virial, warpSize);
        }
    }
}

}  // namespace

void REST2_INFORMATION::Initial(CONTROLLER* controller, int atom_numbers_)
{
    is_initialized = 0;
    if (!REST2_Mode_Is_Enabled(controller))
    {
        return;
    }
    atom_numbers = atom_numbers_;
    if (!controller->Command_Exist("REST2", "lambda_m"))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorMissingCommand, "REST2_INFORMATION::Initial",
            "Reason:\n\tREST2_lambda_m is required when REST2 is enabled.\n");
    }
    controller->Check_Float("REST2", "lambda_m",
                            "REST2_INFORMATION::Initial");
    lambda_m = atof(controller->Command("REST2", "lambda_m"));
    if (lambda_m <= 0.0f)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "REST2_INFORMATION::Initial",
            "Reason:\n\tREST2_lambda_m must be positive.\n");
    }
    sqrt_lambda_m = sqrtf(lambda_m);

    Memory_Allocate();
    std::vector<int> atom_sys_mark_cpu(atom_numbers, 1);
    if (controller->Command_Exist("REST2", "atom_in_file"))
    {
        FILE* fr = NULL;
        int temp_atom;
        Open_File_Safely(&fr, controller->Command("REST2", "atom_in_file"),
                         "r");
        while (fscanf(fr, "%d", &temp_atom) != EOF)
        {
            if (temp_atom < 0 || temp_atom >= atom_numbers)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand, "REST2_INFORMATION::Initial",
                    "Reason:\n\tREST2_atom_in_file contains an atom id outside "
                    "the valid range.\n");
            }
            atom_sys_mark_cpu[temp_atom] = 0;
        }
        fclose(fr);
    }
    else if (controller->Command_Exist("REST2", "atom_numbers"))
    {
        if (strcmp(controller->Command("REST2", "atom_numbers"), "ALL") == 0 ||
            strcmp(controller->Command("REST2", "atom_numbers"), "ITS") == 0)
        {
            std::fill(atom_sys_mark_cpu.begin(), atom_sys_mark_cpu.end(), 0);
        }
        else
        {
            controller->Check_Int("REST2", "atom_numbers",
                                  "REST2_INFORMATION::Initial");
            int hot_atom_numbers =
                atoi(controller->Command("REST2", "atom_numbers"));
            if (hot_atom_numbers < 0 || hot_atom_numbers > atom_numbers)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand, "REST2_INFORMATION::Initial",
                    "Reason:\n\tREST2_atom_numbers is outside the valid "
                    "range.\n");
            }
            for (int i = 0; i < hot_atom_numbers; i++)
            {
                atom_sys_mark_cpu[i] = 0;
            }
        }
    }
    else
    {
        controller->Throw_SPONGE_Error(
            spongeErrorMissingCommand, "REST2_INFORMATION::Initial",
            "Reason:\n\tREST2_atom_in_file or REST2_atom_numbers is required "
            "when REST2 is enabled.\n");
    }
    deviceMemcpy(atom_sys_mark, atom_sys_mark_cpu.data(),
                 sizeof(int) * atom_numbers, deviceMemcpyHostToDevice);

    controller->Step_Print_Initial("REST2_lambda_m", "%.6f");
    controller->Step_Print_Initial("REST2_unscaled", "%.4f");
    controller->Step_Print_Initial("REST2_effective", "%.4f");
    controller->Step_Print_Initial("REST2_bias", "%.4f");
    controller->printf("START INITIALIZING REST2\n");
    controller->printf("    REST2 lambda_m set to %f\n", lambda_m);
    controller->printf("END INITIALIZING REST2\n\n");
    is_initialized = 1;
}

void REST2_INFORMATION::Memory_Allocate()
{
    Device_Malloc_Safely((void**)&atom_sys_mark, sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&atom_sys_mark_local,
                         sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&d_unscaled_atom_energy,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_effective_atom_energy,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_unscaled_energy, sizeof(float));
    Device_Malloc_Safely((void**)&d_effective_energy, sizeof(float));
}

void REST2_INFORMATION::Reset_Force_Energy(int* md_need_potential)
{
    if (!is_initialized) return;
    md_need_potential[0] += 1;
    deviceMemset(d_unscaled_atom_energy, 0, sizeof(float) * atom_numbers);
    deviceMemset(d_effective_atom_energy, 0, sizeof(float) * atom_numbers);
    deviceMemset(d_unscaled_energy, 0, sizeof(float));
    deviceMemset(d_effective_energy, 0, sizeof(float));
}

void REST2_INFORMATION::Get_Local(int* atom_local, int local_atom_numbers_,
                                  int ghost_numbers_)
{
    if (!is_initialized) return;
    local_atom_numbers = local_atom_numbers_;
    ghost_numbers = ghost_numbers_;
    Launch_Device_Kernel(
        REST2_Get_Local_Device,
        (local_atom_numbers + ghost_numbers + CONTROLLER::device_max_thread -
         1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, atom_local, local_atom_numbers,
        ghost_numbers, atom_sys_mark, atom_sys_mark_local);
}

void REST2_INFORMATION::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    Sum_Of_List(d_unscaled_atom_energy, d_unscaled_energy, atom_numbers);
    Sum_Of_List(d_effective_atom_energy, d_effective_energy, atom_numbers);
#ifdef USE_MPI
    if (CONTROLLER::PP_MPI_size != 1)
    {
        D_MPI_Allreduce_IN_PLACE(d_unscaled_energy, 1, D_MPI_FLOAT, D_MPI_SUM,
                                 CONTROLLER::d_pp_comm, NULL);
        D_MPI_Allreduce_IN_PLACE(d_effective_energy, 1, D_MPI_FLOAT, D_MPI_SUM,
                                 CONTROLLER::d_pp_comm, NULL);
    }
#endif
    deviceMemcpy(&h_unscaled_energy, d_unscaled_energy, sizeof(float),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(&h_effective_energy, d_effective_energy, sizeof(float),
                 deviceMemcpyDeviceToHost);
    h_bias_energy = h_effective_energy - h_unscaled_energy;
    controller->Step_Print("REST2_lambda_m", lambda_m);
    controller->Step_Print("REST2_unscaled", h_unscaled_energy);
    controller->Step_Print("REST2_effective", h_effective_energy);
    controller->Step_Print("REST2_bias", h_bias_energy);
}

void REST2_INFORMATION::LJ_Direct_CF_Force_With_Atom_Energy_And_Virial(
    const int atom_numbers, const int local_atom_numbers,
    const int solvent_numbers, const int ghost_numbers, const VECTOR* crd,
    const float* charge, LENNARD_JONES_INFORMATION* lj_info, VECTOR* md_frc,
    const LTMatrix3 cell, const LTMatrix3 rcell, const ATOM_GROUP* nl,
    const float cutoff, const float pme_beta, const int need_energy,
    float* atom_energy_ww, const int need_pressure,
    LTMatrix3* atom_virial_ww, float* elect_atom_ene)
{
    if (!is_initialized || !lj_info->is_initialized) return;
    Launch_Device_Kernel(
        Copy_Crd_And_Charge_To_New_Crd,
        (this->atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL,
        local_atom_numbers + ghost_numbers, crd,
        lj_info->crd_with_LJ_parameters_local, charge);
    if (need_energy)
    {
        deviceMemset(elect_atom_ene, 0,
                     sizeof(float) * (local_atom_numbers + ghost_numbers));
        deviceMemset(lj_info->d_LJ_energy_atom, 0,
                     sizeof(float) * this->atom_numbers);
    }
    if (atom_numbers == 0 || local_atom_numbers == 0) return;

    auto f =
        REST2_Lennard_Jones_And_Direct_Coulomb_Device<true, false, false, true>;
    dim3 blockSize = {CONTROLLER::device_warp,
                      CONTROLLER::device_max_thread / CONTROLLER::device_warp};
    dim3 gridSize = (atom_numbers + blockSize.y - 1) / blockSize.y;
    if (need_energy && !need_pressure)
    {
        f = REST2_Lennard_Jones_And_Direct_Coulomb_Device<true, true, false,
                                                          true>;
    }
    else if (!need_energy && need_pressure)
    {
        f = REST2_Lennard_Jones_And_Direct_Coulomb_Device<true, false, true,
                                                          true>;
    }
    else if (need_energy && need_pressure)
    {
        f = REST2_Lennard_Jones_And_Direct_Coulomb_Device<true, true, true,
                                                          true>;
    }
    Launch_Device_Kernel(
        f, gridSize, blockSize, 0, NULL, local_atom_numbers, solvent_numbers,
        nl, lj_info->crd_with_LJ_parameters_local, cell, rcell,
        lj_info->d_LJ_A, lj_info->d_LJ_B, atom_sys_mark_local, cutoff, md_frc,
        pme_beta, atom_energy_ww, atom_virial_ww, elect_atom_ene,
        lj_info->d_LJ_energy_atom, d_unscaled_atom_energy,
        d_effective_atom_energy, lambda_m, sqrt_lambda_m);
}

bool REST2_INFORMATION::Is_Probe_Safe() const { return true; }
