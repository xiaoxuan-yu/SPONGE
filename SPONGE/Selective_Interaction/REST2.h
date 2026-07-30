#pragma once

#include "../Lennard_Jones_force/LJ_soft_core.h"
#include "../Lennard_Jones_force/Lennard_Jones_force.h"
#include "../common.h"
#include "../control.h"

struct SITS_INFORMATION;

struct REST2_INFORMATION
{
    int is_initialized = 0;
    int atom_numbers = 0;
    int local_atom_numbers = 0;
    int ghost_numbers = 0;
    int* atom_sys_mark = NULL;
    int* atom_sys_mark_local = NULL;

    float lambda_m = 1.0f;
    float sqrt_lambda_m = 1.0f;
    float h_unscaled_energy = 0.0f;
    float h_effective_energy = 0.0f;
    float h_bias_energy = 0.0f;

    float* d_unscaled_atom_energy = NULL;
    float* d_effective_atom_energy = NULL;
    float* d_unscaled_energy = NULL;
    float* d_effective_energy = NULL;

    void Initial(CONTROLLER* controller, int atom_numbers_);
    void Memory_Allocate();
    void Reset_Force_Energy(int* md_need_potential);
    void Get_Local(int* atom_local, int local_atom_numbers_,
                   int ghost_numbers_);
    void Step_Print(CONTROLLER* controller);

    bool LJ_Direct_CF_Force_Clustered(
        SITS_INFORMATION* clustered_workspace,
        const int atom_numbers, const int local_atom_numbers,
        const int ghost_numbers, const VECTOR* crd,
        const float* charge, LENNARD_JONES_INFORMATION* lj_info, VECTOR* md_frc,
        const LTMatrix3 cell, const LTMatrix3 rcell,
        const float cutoff, const float pme_beta, const int need_energy,
        float* atom_energy_ww, const int need_pressure,
        LTMatrix3* atom_virial_ww, float* elect_atom_ene,
        const char** failure_reason);

    bool Is_Probe_Safe() const;
};
