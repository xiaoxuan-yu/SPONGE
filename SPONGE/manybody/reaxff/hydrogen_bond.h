#ifndef REAXFF_HYDROGEN_BOND_H
#define REAXFF_HYDROGEN_BOND_H

#include "../../MD_core/MD_core.h"
#include "../../common.h"
#include "../../control.h"
#include "../../neighbor_list/contract/view.h"
#include "bond_order.h"

inline constexpr float kReaxffHydrogenBondCutoff = 7.5f;

struct REAXFF_HB_Entry
{
    float r0_hb;
    float p_hb1;
    float p_hb2;
    float p_hb3;
};

struct REAXFF_HB_Info
{
    int start_idx;
    int entry_count;
};

struct REAXFF_HYDROGEN_BOND
{
    int is_initialized = 0;
    int atom_numbers = 0;
    int atom_type_numbers = 0;

    // Per-atom-type parameters
    int* h_atom_type = NULL;
    int* d_atom_type = NULL;
    int* h_is_hydrogen = NULL;
    int* d_is_hydrogen = NULL;
    int hydrogen_numbers = 0;
    int* d_hydrogen_atoms = NULL;
    int* d_clustered_atom_to_sorted = NULL;

    // HB parameters [donor][hydrogen][acceptor]
    REAXFF_HB_Info* h_hb_info = NULL;
    REAXFF_HB_Info* d_hb_info = NULL;
    REAXFF_HB_Entry* h_hb_entries = NULL;
    REAXFF_HB_Entry* d_hb_entries = NULL;

    // Energy variables
    float* d_energy_hb_sum = NULL;
    float h_energy_hb = 0.0f;

    float* d_dE_dBO_s = NULL;
    float* d_dE_dBO_pi = NULL;
    float* d_dE_dBO_pi2 = NULL;

    void Initial(CONTROLLER* controller, int atom_numbers,
                 const char* module_name);
    bool Calculate_HB_Energy_And_Force_Clustered(
        const CLUSTERED_SPATIAL_VIEW& view, int atom_numbers, const VECTOR* crd,
        VECTOR* frc, const LTMatrix3 cell, const LTMatrix3 rcell,
        REAXFF_BOND_ORDER* bo_module, const int need_atom_energy,
        float* atom_energy, const int need_virial, LTMatrix3* atom_virial,
        const char** failure_reason = NULL);

    void Step_Print(CONTROLLER* controller);
};

#endif
