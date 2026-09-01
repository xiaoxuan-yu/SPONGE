#pragma once

#include "candidate_builder.h"

#include "../../third_party/cornerstone_octree/include/cstone/sfc/box.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/sfc/common.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/sfc/sfc.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/traversal/boxoverlap.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/traversal/traversal.hpp"

namespace
{

constexpr int kClusteredBuilderBlockSize = 128;
constexpr int kClusteredBuilderWarpSize = 32;
constexpr int kClusteredMaxSuperClusterClusters =
    kClusteredSuperClusterClusters;
constexpr int kFixedShiftCandidateLeafSubgroupSize =
    kClusteredMaxSuperClusterClusters;
constexpr int kClusteredMaxJGroupSize = kClusteredJGroupSize;

static __host__ __device__ __forceinline__ int Candidate_Int_Max(int a, int b)
{
    return a > b ? a : b;
}

static __host__ __device__ __forceinline__ VECTOR Candidate_Shift_Fractional_From_Id(
    int shift_id)
{
    return {
        static_cast<float>(shift_id / 9 - 1),
        static_cast<float>((shift_id % 9) / 3 - 1),
        static_cast<float>(shift_id % 3 - 1)};
}

static __host__ __device__ __forceinline__ VECTOR Candidate_Shift_Vector_From_Id(
    int shift_id, LTMatrix3 cell)
{
    return Candidate_Shift_Fractional_From_Id(shift_id) * cell;
}

static __host__ __device__ __forceinline__ cstone::Vec3<float>
Candidate_To_Cstone_Vec(VECTOR v)
{
    return {v.x, v.y, v.z};
}

template <typename KeyType>
static __host__ __device__ __forceinline__ bool
Candidate_Node_Overlaps_Shifted_Box(KeyType prefix,
                                            VECTOR target_center,
                                            VECTOR target_size,
                                            int shift_id)
{
    const auto unit_box =
        cstone::Box<float>(0.0f, 1.0f, cstone::BoundaryType::open);
    const KeyType start_key = cstone::decodePlaceholderBit(prefix);
    const unsigned level = cstone::decodePrefixLength(prefix) / 3;
    const auto node_ibox = cstone::hilbertIBox<KeyType>(start_key, level);
    const auto [node_center, node_size] =
        cstone::centerAndSize<KeyType>(node_ibox, unit_box);
    constexpr float kFullPeriodicReach = 0.5f - 1.0e-6f;
    if (target_size.x >= kFullPeriodicReach ||
        target_size.y >= kFullPeriodicReach ||
        target_size.z >= kFullPeriodicReach)
    {
        return true;
    }
    const VECTOR shifted_center =
        target_center + Candidate_Shift_Fractional_From_Id(shift_id);
    return cstone::overlap(node_center, node_size,
                           Candidate_To_Cstone_Vec(shifted_center),
                           Candidate_To_Cstone_Vec(target_size), unit_box);
}

template <typename KeyType>
static __host__ __device__ __forceinline__ bool
Candidate_Node_Overlaps_Preshifted_Box(KeyType prefix,
                                               VECTOR shifted_target_center,
                                               VECTOR target_size)
{
    const auto unit_box =
        cstone::Box<float>(0.0f, 1.0f, cstone::BoundaryType::open);
    const KeyType start_key = cstone::decodePlaceholderBit(prefix);
    const unsigned level = cstone::decodePrefixLength(prefix) / 3;
    const auto node_ibox = cstone::hilbertIBox<KeyType>(start_key, level);
    const auto [node_center, node_size] =
        cstone::centerAndSize<KeyType>(node_ibox, unit_box);
    return cstone::overlap(node_center, node_size,
                           Candidate_To_Cstone_Vec(shifted_target_center),
                           Candidate_To_Cstone_Vec(target_size), unit_box);
}

static __host__ __device__ __forceinline__ bool
Candidate_Cluster_Aabb_Overlaps_Shifted_CutoffSq(VECTOR center_i, VECTOR extent_i,
                                             VECTOR center_j, VECTOR extent_j,
                                             float cutoff_sq, VECTOR shift_vec)
{
    const VECTOR dr = center_j - (center_i + shift_vec);
    const float gap_x =
        fmaxf(fabsf(dr.x) - (extent_i.x + extent_j.x), 0.0f);
    const float gap_y =
        fmaxf(fabsf(dr.y) - (extent_i.y + extent_j.y), 0.0f);
    const float gap_z =
        fmaxf(fabsf(dr.z) - (extent_i.z + extent_j.z), 0.0f);
    return gap_x * gap_x + gap_y * gap_y + gap_z * gap_z <= cutoff_sq;
}

static __host__ __device__ __forceinline__ bool
Candidate_Valid_Lanes_Are_All_Local(unsigned int valid_mask,
                                unsigned int local_mask)
{
    return valid_mask != 0u && (valid_mask & ~local_mask) == 0u;
}

static __device__ __forceinline__ bool
Candidate_Leaf_Has_Fixed_Shift_Overlap_Subgroup(
    int cluster_i_start, int leaf_cluster_start, int leaf_cluster_end,
    int fixed_shift_id, float cutoff_sq, VECTOR shift_vec,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, int cluster_i, bool lane_i_valid,
    VECTOR center_i, VECTOR extent_i, device_mask_t subgroup_mask)
{
    bool leaf_overlap = false;
    for (int cluster_j = leaf_cluster_start; cluster_j < leaf_cluster_end;
         cluster_j += 1)
    {
        bool lane_overlap = false;
        if (lane_i_valid && cluster_valid_masks[cluster_j] != 0u)
        {
            const bool reverse_local_cluster_pair =
                cluster_j >= cluster_i_start && cluster_i > cluster_j &&
                Candidate_Valid_Lanes_Are_All_Local(
                    cluster_valid_masks[cluster_j],
                    cluster_local_masks[cluster_j]);
            if (!reverse_local_cluster_pair)
            {
                lane_overlap = Candidate_Cluster_Aabb_Overlaps_Shifted_CutoffSq(
                    center_i, extent_i, cluster_centers[cluster_j],
                    cluster_extents[cluster_j], cutoff_sq, shift_vec);
            }
        }
        if (deviceBallot(subgroup_mask, lane_overlap) != 0u)
        {
            leaf_overlap = true;
            break;
        }
    }
    return leaf_overlap;
}

static __host__ __device__ __forceinline__ VECTOR
Candidate_Shift_Atom_Into_Sorted_XQ_Frame(const VECTOR atom_crd,
                                                const VECTOR cluster_center,
                                                const LTMatrix3 cell,
                                                const LTMatrix3 rcell)
{
    return cluster_center +
           Get_Periodic_Displacement(atom_crd, cluster_center, cell, rcell);
}

static __host__ __device__ __forceinline__ bool
Candidate_Split_Has_Atoms(unsigned int valid_mask_j, int split)
{
    return (valid_mask_j & Clustered_Split_Valid_Mask(split)) != 0u;
}

static __device__ __forceinline__ bool Candidate_Exclusion_List_Contains(
    int atom_i, int atom_j, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers)
{
    const int exclude_start = excluded_list_start[atom_i];
    const int exclude_count = excluded_numbers[atom_i];
    for (int k = 0; k < exclude_count; k += 1)
    {
        if (excluded_list[exclude_start + k] == atom_j)
        {
            return true;
        }
    }
    return false;
}

static __device__ __forceinline__ bool Candidate_Cached_Clusters_May_Share_Molecule(
    const int* molecule_ids_i, const int* molecule_ids_j,
    uint64_t signature_i, uint64_t signature_j, unsigned int local_mask_i,
    unsigned int valid_mask_j, int cluster_size, bool has_molecule_metadata)
{
    if (!has_molecule_metadata)
    {
        return true;
    }
    if ((signature_i & signature_j) == 0ull)
    {
        return false;
    }
    for (int lane_i = 0; lane_i < cluster_size; lane_i += 1)
    {
        if ((local_mask_i & (1u << lane_i)) == 0u)
        {
            continue;
        }
        const int mol_i = molecule_ids_i[lane_i];
        if (mol_i < 0)
        {
            return true;
        }
        for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
        {
            if ((valid_mask_j & (1u << lane_j)) == 0u)
            {
                continue;
            }
            const int mol_j = molecule_ids_j[lane_j];
            if (mol_j < 0 || mol_i == mol_j)
            {
                return true;
            }
        }
    }
    return false;
}

static __device__ __forceinline__ unsigned long long
Candidate_Build_Exclusion_Mask_From_Cached_Atoms(
    const int* atom_ids_i, const int* atom_ids_j, const int* molecule_ids_i,
    const int* molecule_ids_j, uint64_t signature_i, uint64_t signature_j,
    bool has_molecule_metadata, unsigned int local_mask_i,
    unsigned int valid_mask_j, int cluster_size, int local_atom_numbers,
    const int* excluded_list_start, const int* excluded_list,
    const int* excluded_numbers)
{
    if (excluded_list_start == NULL || excluded_list == NULL ||
        excluded_numbers == NULL)
    {
        return 0ull;
    }
    if (!Candidate_Cached_Clusters_May_Share_Molecule(
            molecule_ids_i, molecule_ids_j, signature_i, signature_j,
            local_mask_i, valid_mask_j, cluster_size, has_molecule_metadata))
    {
        return 0ull;
    }

    unsigned long long mask = 0ull;
    for (int lane_i = 0; lane_i < cluster_size; lane_i += 1)
    {
        if ((local_mask_i & (1u << lane_i)) == 0u)
        {
            continue;
        }
        const int atom_i = atom_ids_i[lane_i];
        if (atom_i < 0)
        {
            continue;
        }
        for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
        {
            if ((valid_mask_j & (1u << lane_j)) == 0u)
            {
                continue;
            }
            const int atom_j = atom_ids_j[lane_j];
            if (atom_j < 0)
            {
                continue;
            }
            bool excluded = Candidate_Exclusion_List_Contains(
                atom_i, atom_j, excluded_list_start, excluded_list,
                excluded_numbers);
            if (!excluded && atom_j < local_atom_numbers)
            {
                excluded = Candidate_Exclusion_List_Contains(
                    atom_j, atom_i, excluded_list_start, excluded_list,
                    excluded_numbers);
            }
            if (excluded)
            {
                mask |= (1ull << (lane_i * cluster_size + lane_j));
            }
        }
    }
    return mask;
}

static __device__ __forceinline__ unsigned int
Candidate_Prune_Source_Imask_With_Shift(
    const int split_id, const unsigned int record_imask,
    const unsigned int valid_mask_j, const VECTOR pair_shift,
    const LTMatrix3 cell, const LTMatrix3 rcell, const VECTOR* crd,
    const float* shared_i_center_x, const float* shared_i_center_y,
    const float* shared_i_center_z, const VECTOR center_j,
    const int shared_i_atom_ids[kClusteredMaxSuperClusterClusters]
                               [kClusteredClusterSize],
    const int* shared_j_atom_ids, const unsigned int* shared_i_local_masks,
    const float cutoff_sq)
{
    if (record_imask == 0u || crd == NULL)
    {
        return 0u;
    }
    if (cutoff_sq <= 0.0f)
    {
        return record_imask;
    }

    const int j_lane_base = split_id * kClusteredSplitJClusterSize;
    unsigned int pruned_imask = 0u;
    for (int i_local = 0; i_local < kClusteredMaxSuperClusterClusters;
         i_local += 1)
    {
        const unsigned int i_local_bit = 1u << i_local;
        if ((record_imask & i_local_bit) == 0u)
        {
            continue;
        }
        const unsigned int local_mask_i = shared_i_local_masks[i_local];
        if (local_mask_i == 0u)
        {
            continue;
        }

        bool any_in_range = false;
        for (int i_lane = 0; i_lane < kClusteredClusterSize; i_lane += 1)
        {
            if ((local_mask_i & (1u << i_lane)) == 0u)
            {
                continue;
            }
            const int atom_i = shared_i_atom_ids[i_local][i_lane];
            if (atom_i < 0)
            {
                continue;
            }
            const VECTOR center_i = {shared_i_center_x[i_local],
                                     shared_i_center_y[i_local],
                                     shared_i_center_z[i_local]};
            const VECTOR shifted_i =
                Candidate_Shift_Atom_Into_Sorted_XQ_Frame(
                    crd[atom_i], center_i, cell, rcell);
            for (int split_j_lane = 0;
                 split_j_lane < kClusteredSplitJClusterSize; split_j_lane += 1)
            {
                const int j_lane = j_lane_base + split_j_lane;
                if ((valid_mask_j & (1u << j_lane)) == 0u)
                {
                    continue;
                }
                const int atom_j = shared_j_atom_ids[j_lane];
                if (atom_j < 0)
                {
                    continue;
                }
                const VECTOR shifted_j =
                    Candidate_Shift_Atom_Into_Sorted_XQ_Frame(
                        crd[atom_j], center_j, cell, rcell);
                const float dr_x = shifted_j.x - shifted_i.x - pair_shift.x;
                const float dr_y = shifted_j.y - shifted_i.y - pair_shift.y;
                const float dr_z = shifted_j.z - shifted_i.z - pair_shift.z;
                const float dr2 = dr_x * dr_x + dr_y * dr_y + dr_z * dr_z;
                if (dr2 < cutoff_sq && dr2 != 0.0f)
                {
                    any_in_range = true;
                    break;
                }
            }
            if (any_in_range)
            {
                break;
            }
        }
        if (any_in_range)
        {
            pruned_imask |= i_local_bit;
        }
    }
    return pruned_imask;
}

static __device__ __forceinline__ unsigned int
Candidate_Prune_Source_Imask_With_Shift_Cooperative(
    const int subgroup, const int sublane, const device_mask_t subgroup_mask,
    const unsigned int record_imask, const unsigned int valid_mask_j,
    const VECTOR pair_shift, const LTMatrix3 cell, const LTMatrix3 rcell,
    const VECTOR* crd, const float* shared_i_center_x,
    const float* shared_i_center_y, const float* shared_i_center_z,
    const int shared_i_atom_ids[kClusteredMaxSuperClusterClusters]
                               [kClusteredClusterSize],
    const float* shared_j_shifted_x, const float* shared_j_shifted_y,
    const float* shared_j_shifted_z,
    const unsigned int* shared_i_local_masks,
    const float cutoff_sq)
{
    unsigned int lane_split_hits = 0u;
    const unsigned int i_local_bit = 1u << static_cast<unsigned int>(sublane);

    if (record_imask != 0u && crd != NULL && cutoff_sq <= 0.0f)
    {
        lane_split_hits = (record_imask & i_local_bit) != 0u ? 0x3u : 0u;
    }

    if ((record_imask & i_local_bit) != 0u && crd != NULL && cutoff_sq > 0.0f)
    {
        const unsigned int local_mask_i = shared_i_local_masks[sublane];
        const VECTOR center_i = {shared_i_center_x[sublane],
                                 shared_i_center_y[sublane],
                                 shared_i_center_z[sublane]};

        for (int i_lane = 0; i_lane < kClusteredClusterSize; i_lane += 1)
        {
            const bool active_i_atom =
                (local_mask_i & (1u << i_lane)) != 0u &&
                shared_i_atom_ids[sublane][i_lane] >= 0;
            if (!active_i_atom)
            {
                continue;
            }
            const VECTOR shifted_i =
                Candidate_Shift_Atom_Into_Sorted_XQ_Frame(
                    crd[shared_i_atom_ids[sublane][i_lane]], center_i, cell,
                    rcell);

            for (int j_lane = 0; j_lane < kClusteredClusterSize; j_lane += 1)
            {
                if ((valid_mask_j & (1u << j_lane)) == 0u)
                {
                    continue;
                }
                const VECTOR shifted_j = {shared_j_shifted_x[j_lane],
                                          shared_j_shifted_y[j_lane],
                                          shared_j_shifted_z[j_lane]};
                const float dr_x = shifted_j.x - shifted_i.x - pair_shift.x;
                const float dr_y = shifted_j.y - shifted_i.y - pair_shift.y;
                const float dr_z = shifted_j.z - shifted_i.z - pair_shift.z;
                const float dr2 = dr_x * dr_x + dr_y * dr_y + dr_z * dr_z;
                if (dr2 < cutoff_sq && dr2 != 0.0f)
                {
                    lane_split_hits |=
                        1u << (j_lane / kClusteredSplitJClusterSize);
                }
            }
        }
    }

    const unsigned int split0_imask =
        (deviceBallot(subgroup_mask, (lane_split_hits & 0x1u) != 0u) >>
         (subgroup * kClusteredClusterSize)) &
        0xFFu;
    const unsigned int split1_imask =
        (deviceBallot(subgroup_mask, (lane_split_hits & 0x2u) != 0u) >>
         (subgroup * kClusteredClusterSize)) &
        0xFFu;
    return split0_imask | (split1_imask << kClusteredClusterSize);
}

} // namespace
