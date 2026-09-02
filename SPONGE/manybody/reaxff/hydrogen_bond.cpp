#include "hydrogen_bond.h"

static __device__ __forceinline__ float REAXFF_Consume_HB_Acceptor(
    int h, int a, const VECTOR* crd, const int* atom_type,
    const REAXFF_HB_Info* hb_info, const REAXFF_HB_Entry* hb_entries,
    int atom_type_numbers, const float* bo_s, const float* bo_pi,
    const float* bo_pi2, float* d_dE_dBO_s, float* d_dE_dBO_pi,
    float* d_dE_dBO_pi2, const LTMatrix3 cell, const LTMatrix3 rcell,
    VECTOR* frc, LTMatrix3* atom_virial, const int* bond_count,
    const int* bond_offset, const int* bond_nbr, const int* bond_idx_arr)
{
    const int type_h = atom_type[h];
    const VECTOR rh = crd[h];
    const VECTOR ra = crd[a];
    const VECTOR dah = Get_Periodic_Displacement(ra, rh, cell, rcell);
    const float r_ah = norm3df(dah.x, dah.y, dah.z);
    if (r_ah <= 0.0f || r_ah > kReaxffHydrogenBondCutoff) return 0.0f;

    float en_hb = 0.0f;
    const int bc_h = bond_count[h];
    const int bo_h = bond_offset[h];
    for (int pd = 0; pd < bc_h; pd++)
    {
        const int b_dh = bond_idx_arr[bo_h + pd];
        const int d = bond_nbr[bo_h + pd];
        if (a == d) continue;
        const float bo_dh_val = bo_s[b_dh] + bo_pi[b_dh] + bo_pi2[b_dh];
        if (bo_dh_val < 0.01f) continue;

        const int type_d = atom_type[d];
        const int type_a = atom_type[a];
        const int hb_idx =
            ((type_d * atom_type_numbers + type_h) * atom_type_numbers +
             type_a);
        const REAXFF_HB_Info info = hb_info[hb_idx];
        if (info.entry_count == 0) continue;

        const VECTOR rd = crd[d];
        const VECTOR ddh = Get_Periodic_Displacement(rd, rh, cell, rcell);
        const float r_dh = norm3df(ddh.x, ddh.y, ddh.z);
        if (r_dh <= 0.0f) continue;

        float cos_theta =
            (ddh.x * dah.x + ddh.y * dah.y + ddh.z * dah.z) / (r_dh * r_ah);
        if (cos_theta > 1.0f) cos_theta = 1.0f;
        if (cos_theta < -1.0f) cos_theta = -1.0f;
        const float one_minus_cos = 1.0f - cos_theta;
        const float sin_p4 = 0.25f * one_minus_cos * one_minus_cos;

        for (int e = 0; e < info.entry_count; e++)
        {
            const REAXFF_HB_Entry* param = &hb_entries[info.start_idx + e];

            SADfloat<1> s_bo_dh(bo_dh_val, 0);
            SADfloat<1> s_f_hb = 1.0f - expf(-(float)param->p_hb2 * s_bo_dh);

            const float exp_hb3 = expf(-(float)param->p_hb3 *
                                       ((float)param->r0_hb / r_ah +
                                        r_ah / (float)param->r0_hb - 2.0f));

            SADfloat<1> s_en_total =
                (float)param->p_hb1 * s_f_hb * exp_hb3 * sin_p4;

            atomicAdd(&d_dE_dBO_s[b_dh], s_en_total.dval[0]);
            atomicAdd(&d_dE_dBO_pi[b_dh], s_en_total.dval[0]);
            atomicAdd(&d_dE_dBO_pi2[b_dh], s_en_total.dval[0]);

            const float dE_dr_ah = (float)param->p_hb1 * s_f_hb.val * sin_p4 *
                                   exp_hb3 * (-(float)param->p_hb3) *
                                   (-(float)param->r0_hb / (r_ah * r_ah) +
                                    1.0f / (float)param->r0_hb);

            const float f_ah = -dE_dr_ah;
            const VECTOR f_a_rad = {f_ah * dah.x / r_ah, f_ah * dah.y / r_ah,
                                    f_ah * dah.z / r_ah};
            atomicAdd(&frc[a].x, f_a_rad.x);
            atomicAdd(&frc[a].y, f_a_rad.y);
            atomicAdd(&frc[a].z, f_a_rad.z);
            atomicAdd(&frc[h].x, -f_a_rad.x);
            atomicAdd(&frc[h].y, -f_a_rad.y);
            atomicAdd(&frc[h].z, -f_a_rad.z);

            const float dE_dsinp4 = (float)param->p_hb1 * s_f_hb.val * exp_hb3;
            const float dE_dcos = -0.5f * dE_dsinp4 * one_minus_cos;

            const float inv_rdh = 1.0f / r_dh;
            const float inv_rah = 1.0f / r_ah;

            VECTOR fd, fa, fh;
            fd.x = dE_dcos * (dah.x * inv_rdh * inv_rah -
                              ddh.x * cos_theta * inv_rdh * inv_rdh);
            fd.y = dE_dcos * (dah.y * inv_rdh * inv_rah -
                              ddh.y * cos_theta * inv_rdh * inv_rdh);
            fd.z = dE_dcos * (dah.z * inv_rdh * inv_rah -
                              ddh.z * cos_theta * inv_rdh * inv_rdh);

            fa.x = dE_dcos * (ddh.x * inv_rdh * inv_rah -
                              dah.x * cos_theta * inv_rah * inv_rah);
            fa.y = dE_dcos * (ddh.y * inv_rdh * inv_rah -
                              dah.y * cos_theta * inv_rah * inv_rah);
            fa.z = dE_dcos * (ddh.z * inv_rdh * inv_rah -
                              dah.z * cos_theta * inv_rah * inv_rah);

            fh.x = -(fd.x + fa.x);
            fh.y = -(fd.y + fa.y);
            fh.z = -(fd.z + fa.z);

            atomicAdd(&frc[d].x, -fd.x);
            atomicAdd(&frc[d].y, -fd.y);
            atomicAdd(&frc[d].z, -fd.z);
            atomicAdd(&frc[a].x, -fa.x);
            atomicAdd(&frc[a].y, -fa.y);
            atomicAdd(&frc[a].z, -fa.z);
            atomicAdd(&frc[h].x, -fh.x);
            atomicAdd(&frc[h].y, -fh.y);
            atomicAdd(&frc[h].z, -fh.z);
            if (atom_virial)
            {
                const VECTOR f_d = {-fd.x, -fd.y, -fd.z};
                const VECTOR f_a = {-fa.x, -fa.y, -fa.z};
                const LTMatrix3 v = Get_Virial_From_Force_Dis(f_a_rad, dah) +
                                    Get_Virial_From_Force_Dis(f_d, ddh) +
                                    Get_Virial_From_Force_Dis(f_a, dah);
                atomicAdd(atom_virial + h, v);
            }

            en_hb += s_en_total.val;
        }
    }
    return en_hb;
}

static __device__ __forceinline__ void REAXFF_Consume_HB_Pair(
    int atom_i, int atom_j, const VECTOR* crd, const int* atom_type,
    const int* is_hydrogen, const REAXFF_HB_Info* hb_info,
    const REAXFF_HB_Entry* hb_entries, int atom_type_numbers, const float* bo_s,
    const float* bo_pi, const float* bo_pi2, float* d_dE_dBO_s,
    float* d_dE_dBO_pi, float* d_dE_dBO_pi2, const LTMatrix3 cell,
    const LTMatrix3 rcell, float* atom_energy, VECTOR* frc,
    LTMatrix3* atom_virial, float* energy_sum, const int* bond_count,
    const int* bond_offset, const int* bond_nbr, const int* bond_idx_arr)
{
    if (is_hydrogen[atom_i])
    {
        const float en = REAXFF_Consume_HB_Acceptor(
            atom_i, atom_j, crd, atom_type, hb_info, hb_entries,
            atom_type_numbers, bo_s, bo_pi, bo_pi2, d_dE_dBO_s, d_dE_dBO_pi,
            d_dE_dBO_pi2, cell, rcell, frc, atom_virial, bond_count,
            bond_offset, bond_nbr, bond_idx_arr);
        atomicAdd(energy_sum, en);
        if (atom_energy) atomicAdd(atom_energy + atom_i, en);
    }
    if (is_hydrogen[atom_j])
    {
        const float en = REAXFF_Consume_HB_Acceptor(
            atom_j, atom_i, crd, atom_type, hb_info, hb_entries,
            atom_type_numbers, bo_s, bo_pi, bo_pi2, d_dE_dBO_s, d_dE_dBO_pi,
            d_dE_dBO_pi2, cell, rcell, frc, atom_virial, bond_count,
            bond_offset, bond_nbr, bond_idx_arr);
        atomicAdd(energy_sum, en);
        if (atom_energy) atomicAdd(atom_energy + atom_j, en);
    }
}

#ifdef USE_GPU
static __device__ __forceinline__ int REAXFF_HB_Clustered_Find_Cluster(
    const CLUSTERED_SPATIAL_VIEW& view, const int sorted_atom)
{
    int low = 0;
    int high = view.cluster_numbers;
    while (low + 1 < high)
    {
        const int middle = (low + high) >> 1;
        if (view.cluster_offsets[middle] <= sorted_atom)
            low = middle;
        else
            high = middle;
    }
    return low;
}

static __device__ __forceinline__ bool REAXFF_HB_Clustered_Tile_Atom(
    const CLUSTERED_SPATIAL_VIEW& view,
    const CLUSTERED_GMXPACKED_CENTER_TILE& tile, const int center_lane,
    const int center_atom, const int neighbor_cluster_offset,
    const int neighbor_lane, int* atom_neighbor)
{
    if (neighbor_cluster_offset < 0 ||
        neighbor_cluster_offset >= kClusteredSuperClusterClusters ||
        (tile.neighbor_cluster_mask &
         (1u << static_cast<unsigned int>(neighbor_cluster_offset))) == 0u)
        return false;

    const int neighbor_cluster =
        tile.neighbor_cluster_base + neighbor_cluster_offset;
    if (neighbor_cluster < 0 || neighbor_cluster >= view.cluster_numbers)
        return false;
    if (!Clustered_Lane_Is_Valid(view.cluster_valid_masks[neighbor_cluster],
                                 neighbor_lane) ||
        !Clustered_Lane_Is_Local(view.cluster_local_masks[neighbor_cluster],
                                 neighbor_lane))
        return false;

    const CLUSTERED_GMXPACKED_CJ& packed =
        view.gmxpacked_cjpacked[tile.cjpacked_id];
    int original_i_local = tile.original_i_local;
    int original_i_lane = center_lane;
    int original_j_lane = neighbor_lane;
    if (tile.orientation == CLUSTERED_ENDPOINT_ORIENTATION::TRANSPOSED_J)
    {
        original_i_local = neighbor_cluster_offset;
        original_i_lane = neighbor_lane;
        original_j_lane = center_lane;
    }
    const int split = original_j_lane / kClusteredSplitJClusterSize;
    const int split_j_lane =
        original_j_lane - split * kClusteredSplitJClusterSize;
    const CLUSTERED_GMXPACKED_SPLIT& split_entry = packed.split[split];
    const unsigned int packed_bit =
        1u << (static_cast<int>(tile.jm) * kClusteredSuperClusterClusters +
               original_i_local);
    if ((split_entry.imask & packed_bit) == 0u) return false;

    const uint64_t shift_bits =
        view.pair_shift_bits[tile.cjpacked_id * kClusteredJGroupSize + tile.jm];
    if ((Clustered_Get_Pair_Active_I_Mask(shift_bits, split) &
         (1u << static_cast<unsigned int>(original_i_local))) == 0u)
        return false;
    if (split_entry.exclusion_index != 0)
    {
        const unsigned int pair_bits =
            view.gmxpacked_exclusions[split_entry.exclusion_index]
                .pair[split_j_lane * kClusteredClusterSize + original_i_lane];
        if ((pair_bits & packed_bit) == 0u) return false;
    }

    const int sorted_neighbor =
        view.cluster_offsets[neighbor_cluster] + neighbor_lane;
    *atom_neighbor = view.sort_permutation[sorted_neighbor];
    return *atom_neighbor != center_atom;
}

static __global__ void REAXFF_HB_Build_Atom_To_Sorted(
    const int atom_numbers, const int* sorted_atom_ids, int* atom_to_sorted)
{
    SIMPLE_DEVICE_FOR(sorted_atom, atom_numbers)
    {
        atom_to_sorted[sorted_atom_ids[sorted_atom]] = sorted_atom;
    }
}

static __global__ void Calculate_HB_Clustered_Gmxpacked(
    const CLUSTERED_SPATIAL_VIEW view, const int hydrogen_numbers,
    const int* hydrogen_atoms, const int* atom_to_sorted, const VECTOR* crd,
    const int* atom_type, const REAXFF_HB_Info* hb_info,
    const REAXFF_HB_Entry* hb_entries, int atom_type_numbers, const float* bo_s,
    const float* bo_pi, const float* bo_pi2, float* d_dE_dBO_s,
    float* d_dE_dBO_pi, float* d_dE_dBO_pi2, const LTMatrix3 cell,
    const LTMatrix3 rcell, float* atom_energy, VECTOR* frc,
    LTMatrix3* atom_virial, float* energy_sum, const int* bond_count,
    const int* bond_offset, const int* bond_nbr, const int* bond_idx_arr)
{
    constexpr int lanes_per_hydrogen = 16;
    const int global_thread =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int hydrogen_index = global_thread / lanes_per_hydrogen;
    if (hydrogen_index >= hydrogen_numbers) return;
    const int hydrogen_lane = global_thread % lanes_per_hydrogen;
    const int h = hydrogen_atoms[hydrogen_index];
    const int sorted_h = atom_to_sorted[h];
    const int center_cluster = REAXFF_HB_Clustered_Find_Cluster(view, sorted_h);
    const int center_lane = sorted_h - view.cluster_offsets[center_cluster];
    float en_hb = 0.0f;

    CLUSTERED_GMXPACKED_CENTER_CURSOR cursor = {};
    if (Clustered_Gmxpacked_Center_Cursor_Begin(view, center_cluster, &cursor))
    {
        CLUSTERED_GMXPACKED_CENTER_TILE tile = {};
        while (Clustered_Gmxpacked_Center_Cursor_Next(view, &cursor, &tile))
        {
            const int neighbor_ordinal_count =
                tile.orientation == CLUSTERED_ENDPOINT_ORIENTATION::NATIVE_I
                    ? kClusteredClusterSize
                    : kClusteredSuperClusterClusters * kClusteredClusterSize;
            for (int ordinal = hydrogen_lane; ordinal < neighbor_ordinal_count;
                 ordinal += lanes_per_hydrogen)
            {
                const int neighbor_cluster_offset =
                    ordinal / kClusteredClusterSize;
                const int neighbor_lane =
                    ordinal - neighbor_cluster_offset * kClusteredClusterSize;
                int a = -1;
                if (!REAXFF_HB_Clustered_Tile_Atom(view, tile, center_lane, h,
                                                   neighbor_cluster_offset,
                                                   neighbor_lane, &a))
                    continue;
                en_hb += REAXFF_Consume_HB_Acceptor(
                    h, a, crd, atom_type, hb_info, hb_entries,
                    atom_type_numbers, bo_s, bo_pi, bo_pi2, d_dE_dBO_s,
                    d_dE_dBO_pi, d_dE_dBO_pi2, cell, rcell, frc, atom_virial,
                    bond_count, bond_offset, bond_nbr, bond_idx_arr);
            }
        }
    }
    if (en_hb != 0.0f)
    {
        atomicAdd(energy_sum, en_hb);
        if (atom_energy) atomicAdd(atom_energy + h, en_hb);
    }
}
#endif

#ifdef USE_CPU
static void Calculate_HB_Clustered_Gmxpacked_CPU(
    const CLUSTERED_SPATIAL_VIEW& view, const VECTOR* crd, const int* atom_type,
    const int* is_hydrogen, const REAXFF_HB_Info* hb_info,
    const REAXFF_HB_Entry* hb_entries, int atom_type_numbers, const float* bo_s,
    const float* bo_pi, const float* bo_pi2, float* d_dE_dBO_s,
    float* d_dE_dBO_pi, float* d_dE_dBO_pi2, const LTMatrix3 cell,
    const LTMatrix3 rcell, float* atom_energy, VECTOR* frc,
    LTMatrix3* atom_virial, float* energy_sum, const int* bond_count,
    const int* bond_offset, const int* bond_nbr, const int* bond_idx_arr)
{
#pragma omp parallel for schedule(dynamic)
    for (int sci = 0; sci < view.gmxpacked_sci_numbers; sci += 1)
    {
        const CLUSTERED_GMXPACKED_SCI entry = view.gmxpacked_sci[sci];
        const int ci_begin = view.super_cluster_offsets[entry.supercluster_id];
        const int ci_end =
            view.super_cluster_offsets[entry.supercluster_id + 1];
        const int ci_numbers = ci_end - ci_begin;
        const unsigned int valid_i_cluster_mask =
            (1u << static_cast<unsigned int>(ci_numbers)) - 1u;
        for (int p = entry.cjpacked_begin; p < entry.cjpacked_end; p += 1)
        {
            const CLUSTERED_GMXPACKED_CJ& packed = view.gmxpacked_cjpacked[p];
            for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
            {
                const int cj = packed.cj[jm];
                if (cj < 0) continue;
                const unsigned int jm_shift = static_cast<unsigned int>(
                    jm * kClusteredSuperClusterClusters);
                unsigned int active_j_lanes =
                    view.cluster_valid_masks[cj] & view.cluster_local_masks[cj];
                while (active_j_lanes != 0u)
                {
                    const int jl = __builtin_ctz(active_j_lanes);
                    active_j_lanes &= active_j_lanes - 1u;
                    const int split = jl / kClusteredSplitJClusterSize;
                    const int split_j_lane =
                        jl - split * kClusteredSplitJClusterSize;
                    const CLUSTERED_GMXPACKED_SPLIT& split_entry =
                        packed.split[split];
                    const unsigned int active_i_cluster_mask =
                        (split_entry.imask >> jm_shift) & valid_i_cluster_mask;
                    if (active_i_cluster_mask == 0u) continue;
                    const unsigned int* exclusion_pair =
                        split_entry.exclusion_index != 0
                            ? view.gmxpacked_exclusions[split_entry
                                                            .exclusion_index]
                                      .pair +
                                  split_j_lane * kClusteredClusterSize
                            : NULL;
                    const int atom_j =
                        view.sort_permutation[view.cluster_offsets[cj] + jl];
                    for (int il = 0; il < view.cluster_size; il += 1)
                    {
                        unsigned int active_i_mask = active_i_cluster_mask;
                        if (exclusion_pair != NULL)
                            active_i_mask &= exclusion_pair[il] >> jm_shift;
                        while (active_i_mask != 0u)
                        {
                            const int i_local = __builtin_ctz(active_i_mask);
                            active_i_mask &= active_i_mask - 1u;
                            const int ci = ci_begin + i_local;
                            if (!Clustered_Lane_Is_Valid(
                                    view.cluster_valid_masks[ci], il) ||
                                !Clustered_Lane_Is_Local(
                                    view.cluster_local_masks[ci], il))
                                continue;
                            const int atom_i =
                                view.sort_permutation[view.cluster_offsets[ci] +
                                                      il];
                            if (atom_i == atom_j) continue;
                            REAXFF_Consume_HB_Pair(
                                atom_i, atom_j, crd, atom_type, is_hydrogen,
                                hb_info, hb_entries, atom_type_numbers, bo_s,
                                bo_pi, bo_pi2, d_dE_dBO_s, d_dE_dBO_pi,
                                d_dE_dBO_pi2, cell, rcell, atom_energy, frc,
                                atom_virial, energy_sum, bond_count,
                                bond_offset, bond_nbr, bond_idx_arr);
                        }
                    }
                }
            }
        }
    }
}
#endif

void REAXFF_HYDROGEN_BOND::Initial(CONTROLLER* controller, int atom_numbers,
                                   const char* module_name)
{
    if (module_name == NULL) module_name = "REAXFF";
    this->atom_numbers = atom_numbers;
    if (!controller->Command_Exist(module_name, "in_file")) return;

    const char* parameter_in_file = controller->Command(module_name, "in_file");
    const char* type_in_file = controller->Command(module_name, "type_in_file");
    if (parameter_in_file == NULL || type_in_file == NULL)
    {
        controller->printf(
            "REAXFF_HYDROGEN_BOND IS NOT INITIALIZED (missing input "
            "files)\n\n");
        return;
    }
    controller->Step_Print_Initial("REAXFF_HB", "%14.7e");

    FILE* fp;
    Open_File_Safely(&fp, parameter_in_file, "r");
    char line[1024];
    auto throw_bad_format = [&](const char* file_name, const char* reason)
    {
        char error_msg[1024];
        sprintf(error_msg, "Reason:\n\t%s in file %s\n", reason, file_name);
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "REAXFF_HYDROGEN_BOND::Initial",
                                       error_msg);
    };
    auto read_line_or_throw =
        [&](FILE* file, const char* file_name, const char* stage)
    {
        if (fgets(line, 1024, file) == NULL)
        {
            char reason[512];
            sprintf(reason, "failed to read %s", stage);
            throw_bad_format(file_name, reason);
        }
    };

    read_line_or_throw(fp, parameter_in_file, "parameter header line 1");
    read_line_or_throw(fp, parameter_in_file, "general parameter count line");
    int n_gp = 0;
    if (sscanf(line, "%d", &n_gp) != 1 || n_gp < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of general parameters");
    }
    for (int i = 0; i < n_gp; i++)
    {
        read_line_or_throw(fp, parameter_in_file, "general parameter block");
    }

    read_line_or_throw(fp, parameter_in_file, "atom type count line");
    int n_atom_types = 0;
    if (sscanf(line, "%d", &n_atom_types) != 1 || n_atom_types <= 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of atom types");
    }
    this->atom_type_numbers = n_atom_types;
    read_line_or_throw(fp, parameter_in_file, "atom type header line 1");
    read_line_or_throw(fp, parameter_in_file, "atom type header line 2");
    read_line_or_throw(fp, parameter_in_file, "atom type header line 3");

    std::map<std::string, int> type_map;
    for (int i = 0; i < n_atom_types; i++)
    {
        read_line_or_throw(fp, parameter_in_file, "atom type block line 1");
        char name[16];
        if (sscanf(line, "%15s", name) != 1)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 1 at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        type_map[name] = i;
        read_line_or_throw(fp, parameter_in_file, "atom type block line 2");
        read_line_or_throw(fp, parameter_in_file, "atom type block line 3");
        read_line_or_throw(fp, parameter_in_file, "atom type block line 4");
    }

    read_line_or_throw(fp, parameter_in_file, "bond parameter count line");
    int n_bond = 0;
    if (sscanf(line, "%d", &n_bond) != 1 || n_bond < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of bond parameters");
    }
    read_line_or_throw(fp, parameter_in_file, "bond parameter header line");
    for (int i = 0; i < n_bond * 2; i++)
    {
        read_line_or_throw(fp, parameter_in_file, "bond parameter block");
    }

    read_line_or_throw(fp, parameter_in_file, "off-diagonal count line");
    int n_off = 0;
    if (sscanf(line, "%d", &n_off) != 1 || n_off < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of off-diagonal parameters");
    }
    for (int i = 0; i < n_off; i++)
    {
        read_line_or_throw(fp, parameter_in_file,
                           "off-diagonal parameter entry");
    }

    read_line_or_throw(fp, parameter_in_file, "angle parameter count line");
    int n_thb = 0;
    if (sscanf(line, "%d", &n_thb) != 1 || n_thb < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of angle parameters");
    }
    for (int i = 0; i < n_thb; i++)
    {
        read_line_or_throw(fp, parameter_in_file, "angle parameter entry");
    }

    read_line_or_throw(fp, parameter_in_file, "torsion parameter count line");
    int n_tor = 0;
    if (sscanf(line, "%d", &n_tor) != 1 || n_tor < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of torsion parameters");
    }
    for (int i = 0; i < n_tor; i++)
    {
        read_line_or_throw(fp, parameter_in_file, "torsion parameter entry");
    }

    read_line_or_throw(fp, parameter_in_file, "hydrogen bond count line");
    int n_hb = 0;
    if (sscanf(line, "%d", &n_hb) != 1 || n_hb < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of hydrogen bond parameters");
    }
    std::vector<REAXFF_HB_Entry> all_entries;
    std::map<int, std::vector<int>> triplet_to_entries;
    for (int i = 0; i < n_hb; i++)
    {
        read_line_or_throw(fp, parameter_in_file,
                           "hydrogen bond parameter entry");
        int t1, t2, t3;
        float r0, p1, p2, p3;
        int read_cnt = sscanf(line, "%d %d %d %f %f %f %f", &t1, &t2, &t3, &r0,
                              &p1, &p2, &p3);
        if (read_cnt != 7)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse hydrogen bond parameter entry at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        int idx1 = t1 - 1;
        int idx2 = t2 - 1;
        int idx3 = t3 - 1;
        if (idx1 < 0 || idx1 >= n_atom_types || idx2 < 0 ||
            idx2 >= n_atom_types || idx3 < 0 || idx3 >= n_atom_types)
        {
            char reason[512];
            sprintf(reason,
                    "hydrogen bond atom type index out of range at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        REAXFF_HB_Entry entry = {r0, p1, p2, p3};
        int entry_idx = all_entries.size();
        all_entries.push_back(entry);
        int tri_idx = (idx1 * n_atom_types + idx2) * n_atom_types + idx3;
        triplet_to_entries[tri_idx].push_back(entry_idx);
    }
    fclose(fp);

    Malloc_Safely((void**)&h_hb_info, sizeof(REAXFF_HB_Info) * n_atom_types *
                                          n_atom_types * n_atom_types);
    memset(h_hb_info, 0,
           sizeof(REAXFF_HB_Info) * n_atom_types * n_atom_types * n_atom_types);
    std::vector<REAXFF_HB_Entry> sorted_entries;
    for (int i = 0; i < n_atom_types * n_atom_types * n_atom_types; i++)
    {
        if (triplet_to_entries.count(i))
        {
            h_hb_info[i].start_idx = sorted_entries.size();
            h_hb_info[i].entry_count = triplet_to_entries[i].size();
            for (int idx : triplet_to_entries[i])
                sorted_entries.push_back(all_entries[idx]);
        }
    }
    Malloc_Safely((void**)&h_hb_entries,
                  sizeof(REAXFF_HB_Entry) * sorted_entries.size());
    for (size_t i = 0; i < sorted_entries.size(); i++)
        h_hb_entries[i] = sorted_entries[i];

    Open_File_Safely(&fp, type_in_file, "r");
    int check_atom_numbers = 0;
    read_line_or_throw(fp, type_in_file, "atom number line");
    if (sscanf(line, "%d", &check_atom_numbers) != 1)
    {
        throw_bad_format(type_in_file, "failed to parse atom numbers");
    }
    if (check_atom_numbers != atom_numbers)
    {
        char reason[512];
        sprintf(reason, "atom numbers (%d) does not match system (%d)",
                check_atom_numbers, atom_numbers);
        throw_bad_format(type_in_file, reason);
    }
    Malloc_Safely((void**)&h_atom_type, sizeof(int) * atom_numbers);
    Malloc_Safely((void**)&h_is_hydrogen, sizeof(int) * atom_numbers);
    std::vector<int> hydrogen_atoms;
    for (int i = 0; i < atom_numbers; i++)
    {
        read_line_or_throw(fp, type_in_file, "atom type entry line");
        char name[16];
        if (sscanf(line, "%15s", name) != 1)
        {
            char reason[512];
            sprintf(reason, "failed to parse atom type at index %d", i + 1);
            throw_bad_format(type_in_file, reason);
        }
        auto iter = type_map.find(std::string(name));
        if (iter == type_map.end())
        {
            char reason[512];
            sprintf(reason, "atom type %s not found in parameter file %s", name,
                    parameter_in_file);
            throw_bad_format(type_in_file, reason);
        }
        h_atom_type[i] = iter->second;
        h_is_hydrogen[i] = (std::string(name) == "H");
        if (h_is_hydrogen[i]) hydrogen_atoms.push_back(i);
    }
    fclose(fp);

    Device_Malloc_And_Copy_Safely((void**)&d_atom_type, h_atom_type,
                                  sizeof(int) * atom_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_is_hydrogen, h_is_hydrogen,
                                  sizeof(int) * atom_numbers);
    hydrogen_numbers = static_cast<int>(hydrogen_atoms.size());
    if (hydrogen_numbers > 0)
    {
        Device_Malloc_And_Copy_Safely((void**)&d_hydrogen_atoms,
                                      hydrogen_atoms.data(),
                                      sizeof(int) * hydrogen_numbers);
    }
    Device_Malloc_Safely((void**)&d_clustered_atom_to_sorted,
                         sizeof(int) * atom_numbers);
    Device_Malloc_And_Copy_Safely(
        (void**)&d_hb_info, h_hb_info,
        sizeof(REAXFF_HB_Info) * n_atom_types * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely(
        (void**)&d_hb_entries, h_hb_entries,
        sizeof(REAXFF_HB_Entry) * sorted_entries.size());
    Device_Malloc_Safely((void**)&d_energy_hb_sum, sizeof(float));
    is_initialized = 1;
}

bool REAXFF_HYDROGEN_BOND::Calculate_HB_Energy_And_Force_Clustered(
    const CLUSTERED_SPATIAL_VIEW& view, int atom_numbers, const VECTOR* crd,
    VECTOR* frc, const LTMatrix3 cell, const LTMatrix3 rcell,
    REAXFF_BOND_ORDER* bo_module, const int need_atom_energy,
    float* atom_energy, const int need_virial, LTMatrix3* atom_virial,
    const char** failure_reason)
{
    if (failure_reason != NULL) *failure_reason = NULL;
    if (!is_initialized) return true;
    auto fail = [failure_reason](const char* reason)
    {
        if (failure_reason != NULL) *failure_reason = reason;
        return false;
    };
    if (crd == NULL || frc == NULL || bo_module == NULL ||
        (need_atom_energy && atom_energy == NULL) ||
        (need_virial && atom_virial == NULL))
    {
        return fail(
            "clustered ReaxFF hydrogen bond received null input or output "
            "buffers");
    }

    const char* view_failure_reason = NULL;
    CLUSTERED_SPATIAL_VIEW_REQUIREMENTS requirements;
    requirements.local_atom_numbers = atom_numbers;
    requirements.ghost_numbers = 0;
    requirements.cutoff = kReaxffHydrogenBondCutoff;
    requirements.provider_incarnation = view.provider_incarnation;
    requirements.lease_epoch = view.lease_epoch;
    requirements.gmxpacked_payload_generation =
        view.gmxpacked_payload_generation;
    requirements.require_all_local_atoms = true;
    requirements.require_backend = true;
    requirements.require_gmxpacked_payload = true;
#ifdef USE_CPU
    requirements.backend = CLUSTERED_SPATIAL_BACKEND::CPU;
#else
#if defined(USE_CUDA)
    requirements.backend = CLUSTERED_SPATIAL_BACKEND::CUDA;
#else
    requirements.backend = CLUSTERED_SPATIAL_BACKEND::HIP;
#endif
    requirements.require_same_producer_stream = true;
    requirements.consumer_stream = NULL;
    requirements.require_gmxpacked_endpoint_incidence = true;
    requirements.require_pair_shift_metadata = true;
    requirements.require_pair_shift_rcell = true;
    requirements.pair_shift_rcell = rcell;
#endif
    if (!Clustered_Validate_Spatial_View(view, requirements,
                                         &view_failure_reason))
    {
        if (failure_reason != NULL) *failure_reason = view_failure_reason;
        return false;
    }

    deviceMemset(d_energy_hb_sum, 0, sizeof(float));
#ifdef USE_CPU
    if (view.gmxpacked_sci_numbers <= 0) return true;
    Calculate_HB_Clustered_Gmxpacked_CPU(
        view, crd, d_atom_type, d_is_hydrogen, d_hb_info, d_hb_entries,
        atom_type_numbers, bo_module->d_corrected_bo_s,
        bo_module->d_corrected_bo_pi, bo_module->d_corrected_bo_pi2, d_dE_dBO_s,
        d_dE_dBO_pi, d_dE_dBO_pi2, cell, rcell,
        need_atom_energy ? atom_energy : NULL, frc,
        need_virial ? atom_virial : NULL, d_energy_hb_sum,
        bo_module->d_bond_count, bo_module->d_bond_offset,
        bo_module->d_bond_nbr, bo_module->d_bond_idx);
#else
    if (hydrogen_numbers <= 0 || view.gmxpacked_sci_numbers <= 0) return true;
    const dim3 map_block(256, 1, 1);
    const dim3 map_grid((atom_numbers + static_cast<int>(map_block.x) - 1) /
                            static_cast<int>(map_block.x),
                        1, 1);
    Launch_Device_Kernel(REAXFF_HB_Build_Atom_To_Sorted, map_grid, map_block, 0,
                         NULL, atom_numbers, view.sort_permutation,
                         d_clustered_atom_to_sorted);

    constexpr int lanes_per_hydrogen = 16;
    const dim3 block(128, 1, 1);
    const dim3 grid((hydrogen_numbers * lanes_per_hydrogen +
                     static_cast<int>(block.x) - 1) /
                        static_cast<int>(block.x),
                    1, 1);
    Launch_Device_Kernel(
        Calculate_HB_Clustered_Gmxpacked, grid, block, 0, NULL, view,
        hydrogen_numbers, d_hydrogen_atoms, d_clustered_atom_to_sorted, crd,
        d_atom_type, d_hb_info, d_hb_entries, atom_type_numbers,
        bo_module->d_corrected_bo_s, bo_module->d_corrected_bo_pi,
        bo_module->d_corrected_bo_pi2, d_dE_dBO_s, d_dE_dBO_pi, d_dE_dBO_pi2,
        cell, rcell, need_atom_energy ? atom_energy : NULL, frc,
        need_virial ? atom_virial : NULL, d_energy_hb_sum,
        bo_module->d_bond_count, bo_module->d_bond_offset,
        bo_module->d_bond_nbr, bo_module->d_bond_idx);
#endif
    return true;
}

void REAXFF_HYDROGEN_BOND::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    deviceMemcpy(&h_energy_hb, d_energy_hb_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
    controller->Step_Print("REAXFF_HB", h_energy_hb, true);
}
