#ifndef STILLINGER_WEBER_FORCE_H
#define STILLINGER_WEBER_FORCE_H
#include "../common.h"
#include "../control.h"
#include "../neighbor_list/clustered_spatial_view.h"

// Stillinger and Weber, Phys Rev B, 31, 5262 (1985).
struct STILLINGER_WEBER_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    int atom_numbers = 0;
    int atom_type_numbers = 0;
    int pair_type_numbers = 0;
    int triple_type_numbers = 0;
    float cut = 0.0f;

    int* h_atom_type = NULL;
    int* d_atom_type = NULL;

    float *h_parameters, *d_parameters;

    float* h_energy_atom = NULL;
    float h_energy_sum = 0;
    float* d_energy_atom = NULL;
    float* d_energy_sum = NULL;

    long long clustered_neighbor_provider_incarnation = -1;
    long long clustered_neighbor_payload_generation = -1;
    int clustered_neighbor_numbers = 0;
    int* d_clustered_neighbor_counts = NULL;
    int clustered_neighbor_counts_capacity = 0;
    int* d_clustered_neighbor_offsets = NULL;
    int clustered_neighbor_offsets_capacity = 0;
    int* d_clustered_neighbor_atoms = NULL;
    int clustered_neighbor_atoms_capacity = 0;

    void Initial(CONTROLLER* controller, const char* module_name = NULL);

    bool SW_Force_Clustered(
        const CLUSTERED_SPATIAL_VIEW& view, const VECTOR* crd, VECTOR* frc,
        const LTMatrix3 cell, const LTMatrix3 rcell,
        const int need_atom_energy, float* atom_energy,
        const int need_virial, LTMatrix3* atom_virial,
        const char** failure_reason = NULL);

    void Step_Print(CONTROLLER* controller);
};
#endif
