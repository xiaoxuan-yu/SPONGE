#ifndef EAM_FORCE_H
#define EAM_FORCE_H
#include "../common.h"
#include "../control.h"
#include "../neighbor_list/clustered_spatial_view.h"

// Daw and Baskes, Phys. Rev. B 29, 6443 (1984)
struct EAM_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    int atom_numbers = 0;
    int atom_type_numbers = 0;

    int* h_atom_type = NULL;
    int* d_atom_type = NULL;
    VECTOR* d_clustered_sorted_crd = NULL;
    int clustered_scratch_capacity = 0;

    int nrho, nr;
    float drho, dr;
    float cut;

    float *h_embed = NULL, *d_embed = NULL;
    float *h_electron_density = NULL, *d_electron_density = NULL;
    float *h_pair_potential = NULL, *d_pair_potential = NULL;

    float *h_rho = NULL, *d_rho = NULL;
    float *h_df_drho = NULL, *d_df_drho = NULL;

    float *h_energy_atom = NULL, *d_energy_atom = NULL;
    float h_energy_sum = 0, *d_energy_sum = NULL;

    void Initial(CONTROLLER* controller, const int atom_numbers,
                 const char* module_name = NULL);
    void Read_Funcfl(FILE* fp, CONTROLLER* controller);
    void Read_Setfl(FILE* fp, CONTROLLER* controller);

    bool EAM_Force_Clustered(const CLUSTERED_SPATIAL_VIEW& view,
                             const VECTOR* crd, VECTOR* frc,
                             const LTMatrix3 cell, const LTMatrix3 rcell,
                             const int need_atom_energy, float* atom_energy,
                             const int need_virial, LTMatrix3* atom_virial,
                             const char** failure_reason = NULL);

    void Step_Print(CONTROLLER* controller);
};

#endif
