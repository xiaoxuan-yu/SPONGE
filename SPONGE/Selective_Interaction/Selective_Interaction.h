#pragma once

#include "REST2.h"
#include "SITS.h"

struct SELECTIVE_INTERACTION
{
    SITS_INFORMATION sits;
    REST2_INFORMATION rest2;
    int is_initialized = 0;

    void Initial(CONTROLLER* controller, int atom_numbers);
    void Clear_Clustered_Sparse_Product();
    void Reset_Force_Energy(int* md_need_potential);
    void Update_And_Enhance(const int step, float* d_total_potential,
                            const int need_pressure, LTMatrix3* d_total_virial,
                            VECTOR* frc, float beta0);
    void Get_Local(int* atom_local, int local_atom_numbers, int ghost_numbers);
    void Step_Print(CONTROLLER* controller, float beta0);

    bool Is_Initialized() const;
    bool Is_Probe_Safe() const;
    bool Is_Selectively_Applied() const;
    bool Has_Direct_LJ_Coulomb() const;
    bool Has_SITS_Direct_LJ_Coulomb() const;
    bool Has_REST2_Direct_LJ_Coulomb() const;
    bool Uses_SITS_Listed_Forces() const;

    VECTOR* Select_Force();
    float* Select_Atom_Energy();
    LTMatrix3* Select_Atom_Virial_Tensor();

    bool LJ_Direct_CF_Force_Clustered(
        const int atom_numbers, const int local_atom_numbers,
        const int ghost_numbers, const VECTOR* crd, const float* charge,
        LENNARD_JONES_INFORMATION* lj_info, VECTOR* md_frc,
        const LTMatrix3 cell, const LTMatrix3 rcell, const float cutoff,
        const float pme_beta, const int need_energy, float* atom_energy_ww,
        const int need_pressure, LTMatrix3* atom_virial_ww,
        float* elect_atom_ene, const char** failure_reason);

    bool LJ_Soft_Core_Direct_CF_Force_Clustered(
        const int atom_numbers, const int local_atom_numbers,
        const int ghost_numbers, LJ_SOFT_CORE* lj_info, VECTOR* frc,
        const LTMatrix3 cell, const LTMatrix3 rcell, const float cutoff,
        const float pme_beta, const int need_energy, float* atom_energy_ww,
        const int need_pressure, LTMatrix3* atom_virial_ww,
        float* elect_atom_ene, const char** failure_reason);

    void LJ_Soft_Core_Direct_CF_Force_With_Atom_Energy_And_Virial(
        const int atom_numbers, const int local_atom_numbers,
        const int ghost_numbers, const VECTOR* crd, const float* charge,
        LJ_SOFT_CORE* lj_info, VECTOR* frc,
        const LTMatrix3 cell, const LTMatrix3 rcell,
        const float pme_beta, const int need_energy, float* atom_energy_ww,
        const int need_pressure,
        LTMatrix3* atom_virial_ww, float* elect_atom_ene);
};
