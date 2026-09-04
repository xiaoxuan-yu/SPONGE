#ifndef TERSOFF_FORCE_H
#define TERSOFF_FORCE_H
#include "../common.h"
#include "../control.h"
#include "../neighbor_list/contract/view.h"
#include "clustered_csr.h"

// J. Tersoff, Phys. Rev. B 37, 6991 (1988)
struct TERSOFF_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    int atom_numbers = 0;
    int atom_type_numbers = 0;
    float cut = 0.0f;

    int* h_atom_type = NULL;
    int* d_atom_type = NULL;

    int n_unique_params = 0;
    int* h_map = NULL;
    int* d_map = NULL;

    float* h_params = NULL;
    float* d_params = NULL;
    float* h_center_cutoffs = NULL;
    float* d_center_cutoffs = NULL;

    float* h_energy_atom = NULL;
    float h_energy_sum = 0;
    float* d_energy_atom = NULL;
    float* d_energy_sum = NULL;

    ClusteredPayloadStamp clustered_neighbor_stamp = {};
    ClusteredCSRStorage clustered_neighbors = {};

    void Initial(CONTROLLER* controller, int atom_numbers,
                 const char* module_name = NULL);

    bool TERSOFF_Force_Clustered(const CLUSTERED_SPATIAL_VIEW& view,
                                 const VECTOR* crd, VECTOR* frc,
                                 const LTMatrix3 cell, const LTMatrix3 rcell,
                                 const int need_atom_energy, float* atom_energy,
                                 const int need_virial, LTMatrix3* atom_virial,
                                 const char** failure_reason = NULL);

    void Step_Print(CONTROLLER* controller);
};
#endif
