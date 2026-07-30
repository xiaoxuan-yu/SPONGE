#include "Selective_Interaction.h"

void SELECTIVE_INTERACTION::Initial(CONTROLLER* controller, int atom_numbers)
{
    sits.Initial(controller, atom_numbers);
    rest2.Initial(controller, atom_numbers);
    if (sits.is_initialized && rest2.is_initialized)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "SELECTIVE_INTERACTION::Initial",
            "Reason:\n\tSITS and REST2 cannot be enabled together.\n");
    }
    is_initialized = sits.is_initialized || rest2.is_initialized;
}

void SELECTIVE_INTERACTION::Check_Solvent(CONTROLLER* controller,
                                          int atom_numbers, int solvent_numbers)
{
    sits.Check_Solvent(controller, atom_numbers, solvent_numbers);
}

void SELECTIVE_INTERACTION::Reset_Force_Energy(int* md_need_potential)
{
    sits.Reset_Force_Energy(md_need_potential);
    rest2.Reset_Force_Energy(md_need_potential);
}

void SELECTIVE_INTERACTION::Update_And_Enhance(const int step,
                                               float* d_total_potential,
                                               const int need_pressure,
                                               LTMatrix3* d_total_virial,
                                               VECTOR* frc, float beta0)
{
    sits.Update_And_Enhance(step, d_total_potential, need_pressure,
                            d_total_virial, frc, beta0);
}

void SELECTIVE_INTERACTION::Get_Local(int* atom_local, int local_atom_numbers,
                                      int ghost_numbers)
{
    sits.Get_Local(atom_local, local_atom_numbers, ghost_numbers);
    rest2.Get_Local(atom_local, local_atom_numbers, ghost_numbers);
}

void SELECTIVE_INTERACTION::Step_Print(CONTROLLER* controller, float beta0)
{
    sits.Step_Print(controller, beta0);
    rest2.Step_Print(controller);
}

bool SELECTIVE_INTERACTION::Is_Initialized() const { return is_initialized; }

bool SELECTIVE_INTERACTION::Is_Probe_Safe() const
{
    return !sits.is_initialized && rest2.Is_Probe_Safe();
}

bool SELECTIVE_INTERACTION::Is_Selectively_Applied() const
{
    return sits.is_initialized && sits.selectively_applied;
}

bool SELECTIVE_INTERACTION::Has_Direct_LJ_Coulomb() const
{
    return rest2.is_initialized ||
           (sits.is_initialized && sits.selectively_applied);
}

bool SELECTIVE_INTERACTION::Has_SITS_Direct_LJ_Coulomb() const
{
    return sits.is_initialized && sits.selectively_applied;
}

bool SELECTIVE_INTERACTION::Has_REST2_Direct_LJ_Coulomb() const
{
    return rest2.is_initialized;
}

bool SELECTIVE_INTERACTION::Uses_SITS_Listed_Forces() const
{
    return sits.is_initialized && sits.selectively_applied;
}

VECTOR* SELECTIVE_INTERACTION::Select_Force()
{
    return sits.pw_select.select_force[0];
}

float* SELECTIVE_INTERACTION::Select_Atom_Energy()
{
    return sits.pw_select.select_atom_energy[0];
}

LTMatrix3* SELECTIVE_INTERACTION::Select_Atom_Virial_Tensor()
{
    return sits.pw_select.select_atom_virial_tensor[0];
}

bool SELECTIVE_INTERACTION::LJ_Direct_CF_Force_Clustered(
    const int atom_numbers, const int local_atom_numbers,
    const int ghost_numbers, const VECTOR* crd, const float* charge,
    LENNARD_JONES_INFORMATION* lj_info, VECTOR* md_frc,
    const LTMatrix3 cell, const LTMatrix3 rcell, const float cutoff,
    const float pme_beta, const int need_energy, float* atom_energy_ww,
    const int need_pressure, LTMatrix3* atom_virial_ww,
    float* elect_atom_ene, const char** failure_reason)
{
    if (rest2.is_initialized)
    {
        return rest2.LJ_Direct_CF_Force_Clustered(
            &sits, atom_numbers, local_atom_numbers, ghost_numbers, crd,
            charge, lj_info, md_frc, cell, rcell, cutoff, pme_beta,
            need_energy, atom_energy_ww, need_pressure, atom_virial_ww,
            elect_atom_ene, failure_reason);
    }
    if (!Has_SITS_Direct_LJ_Coulomb())
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                "clustered selective direct dispatch requires SITS or REST2";
        }
        return false;
    }
    return sits.SITS_LJ_Direct_CF_Force_Clustered(
        atom_numbers, local_atom_numbers, ghost_numbers, crd, charge, lj_info,
        md_frc, cell, rcell, cutoff, pme_beta, need_energy, atom_energy_ww,
        need_pressure, atom_virial_ww, elect_atom_ene, failure_reason);
}

bool SELECTIVE_INTERACTION::LJ_Soft_Core_Direct_CF_Force_Clustered(
    const int atom_numbers, const int local_atom_numbers,
    const int ghost_numbers, LJ_SOFT_CORE* lj_info, VECTOR* frc,
    const LTMatrix3 cell, const LTMatrix3 rcell, const float cutoff,
    const float pme_beta, const int need_energy, float* atom_energy_ww,
    const int need_pressure, LTMatrix3* atom_virial_ww,
    float* elect_atom_ene, const char** failure_reason)
{
    if (!Has_SITS_Direct_LJ_Coulomb() || rest2.is_initialized)
    {
        if (failure_reason != NULL)
        {
            *failure_reason =
                "clustered selective soft-LJ dispatch is SITS-only";
        }
        return false;
    }
    return sits.SITS_LJ_Soft_Core_Direct_CF_Force_Clustered(
        atom_numbers, local_atom_numbers, ghost_numbers, lj_info, frc, cell,
        rcell, cutoff, pme_beta, need_energy, atom_energy_ww, need_pressure,
        atom_virial_ww, elect_atom_ene, failure_reason);
}

void SELECTIVE_INTERACTION::
    LJ_Soft_Core_Direct_CF_Force_With_Atom_Energy_And_Virial(
        const int atom_numbers, const int local_atom_numbers,
        const int solvent_numbers, const int ghost_numbers, const VECTOR* crd,
        const float* charge, LJ_SOFT_CORE* lj_info, VECTOR* frc,
        const LTMatrix3 cell, const LTMatrix3 rcell,
        const float pme_beta, const int need_energy, float* atom_energy_ww,
        const int need_pressure,
        LTMatrix3* atom_virial_ww, float* elect_atom_ene)
{
    lj_info->LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy_And_Virial(
        atom_numbers, local_atom_numbers, solvent_numbers, ghost_numbers, crd,
        charge, frc, cell, rcell, NULL, pme_beta, need_energy, atom_energy_ww,
        need_pressure, atom_virial_ww, elect_atom_ene);
}
