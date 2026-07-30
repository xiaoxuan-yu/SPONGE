#include "REST2.h"
#include "SITS.h"

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
    controller->Check_Float("REST2", "lambda_m", "REST2_INFORMATION::Initial");
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
    Launch_Device_Kernel(REST2_Get_Local_Device,
                         (local_atom_numbers + ghost_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, atom_local,
                         local_atom_numbers, ghost_numbers, atom_sys_mark,
                         atom_sys_mark_local);
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

bool REST2_INFORMATION::LJ_Direct_CF_Force_Clustered(
    SITS_INFORMATION* clustered_workspace,
    const int atom_numbers, const int local_atom_numbers,
    const int ghost_numbers, const VECTOR* crd,
    const float* charge, LENNARD_JONES_INFORMATION* lj_info, VECTOR* md_frc,
    const LTMatrix3 cell, const LTMatrix3 rcell,
    const float cutoff, const float pme_beta, const int need_energy,
    float* atom_energy_ww, const int need_pressure, LTMatrix3* atom_virial_ww,
    float* elect_atom_ene, const char** failure_reason)
{
    if (failure_reason != NULL)
    {
        *failure_reason = NULL;
    }
    if (!is_initialized || lj_info == NULL || !lj_info->is_initialized)
    {
        return true;
    }
    if (clustered_workspace == NULL || !lj_info->Use_Clustered_Direct())
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                "REST2 requires the clustered regular-LJ production path";
        }
        return false;
    }
    lj_info->LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(
        atom_numbers, local_atom_numbers, ghost_numbers, crd, charge, md_frc,
        cell, rcell, pme_beta, need_energy, atom_energy_ww,
        need_pressure, atom_virial_ww, elect_atom_ene);
    return clustered_workspace->REST2_LJ_Direct_CF_Correction_Clustered(
        atom_numbers, local_atom_numbers, ghost_numbers, lj_info, md_frc,
        cell, rcell, cutoff, pme_beta, need_energy, atom_energy_ww,
        need_pressure, atom_virial_ww, elect_atom_ene,
        atom_sys_mark_local, lambda_m, sqrt_lambda_m,
        d_unscaled_atom_energy, d_effective_atom_energy, failure_reason);
}

bool REST2_INFORMATION::Is_Probe_Safe() const { return true; }
