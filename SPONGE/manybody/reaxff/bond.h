#ifndef REAXFF_BOND_H
#define REAXFF_BOND_H

#include "../../common.h"
#include "../../control.h"

struct REAXFF_BOND
{
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    int atom_numbers = 0;
    int atom_type_numbers = 0;

    // Parameters
    int* h_atom_type = NULL;
    int* d_atom_type = NULL;

    // ReaxFF general parameters
    float* h_general_params = NULL;
    float* d_general_params = NULL;

    // Two-body parameters for bonds
    float* h_twobody_params = NULL;  // indexed by [type_i * ntypes + type_j]
    float* d_twobody_params = NULL;

    // Bond order data (sparse per-bond arrays from bond_order module)
    float* d_bo_s = NULL;
    float* d_bo_pi = NULL;
    float* d_bo_pi2 = NULL;

    float* d_dE_dBO_s = NULL;
    float* d_dE_dBO_pi = NULL;
    float* d_dE_dBO_pi2 = NULL;

    // Bond order derivatives (shared pointers from bond_order module)
    float* d_dbo_s_dr = NULL;
    float* d_dbo_pi_dr = NULL;
    float* d_dbo_pi2_dr = NULL;
    float* d_dbo_s_dDelta_i = NULL;
    float* d_dbo_pi_dDelta_i = NULL;
    float* d_dbo_pi2_dDelta_i = NULL;
    float* d_dbo_s_dDelta_j = NULL;
    float* d_dbo_pi_dDelta_j = NULL;
    float* d_dbo_pi2_dDelta_j = NULL;
    float* d_dbo_raw_total_dr = NULL;
    float* d_CdDelta = NULL;

    // Canonical bond pairs (set from bond_order module)
    int* d_pair_i = NULL;
    int* d_pair_j = NULL;

    float h_energy_sum = 0;
    float* d_energy_sum = NULL;

    void Initial(CONTROLLER* controller, int atom_numbers,
                 const char* module_name = NULL);

    void Calculate_Bond_Energy_And_Derivatives(
        int bond_numbers, int need_atom_energy, float* atom_energy);

    void Step_Print(CONTROLLER* controller);
};

#endif
