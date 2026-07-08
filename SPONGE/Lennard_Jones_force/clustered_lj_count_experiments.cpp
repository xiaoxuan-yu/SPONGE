#include "clustered_lj_count_experiments.h"

#include <cstdint>

#include "../third_party/cornerstone_octree/include/cstone/sfc/box.hpp"
#include "../third_party/cornerstone_octree/include/cstone/sfc/common.hpp"
#include "../third_party/cornerstone_octree/include/cstone/sfc/sfc.hpp"
#include "../third_party/cornerstone_octree/include/cstone/traversal/boxoverlap.hpp"
#include "../third_party/cornerstone_octree/include/cstone/traversal/traversal.hpp"

#ifndef USE_CPU

namespace
{

constexpr int kClusteredBuilderBlockSize = 128;
constexpr int kClusteredBuilderWarpSize = 32;
constexpr int kClusteredMaxSuperClusterClusters =
    kClusteredSuperClusterClusters;
constexpr int kFixedShiftCandidateLeafSubgroupSize =
    kClusteredMaxSuperClusterClusters;
constexpr int kClusteredMaxJGroupSize = kClusteredJGroupSize;

static __host__ __device__ __forceinline__ int IntMaxProbe(int a, int b)
{
    return a > b ? a : b;
}

static __host__ __device__ __forceinline__ VECTOR Shift_Fractional_From_Id_Probe(
    int shift_id)
{
    return {
        static_cast<float>(shift_id / 9 - 1),
        static_cast<float>((shift_id % 9) / 3 - 1),
        static_cast<float>(shift_id % 3 - 1)};
}

static __host__ __device__ __forceinline__ VECTOR Shift_Vector_From_Id_Probe(
    int shift_id, LTMatrix3 cell)
{
    return Shift_Fractional_From_Id_Probe(shift_id) * cell;
}

static __host__ __device__ __forceinline__ cstone::Vec3<float>
To_Cstone_Vec_Probe(VECTOR v)
{
    return {v.x, v.y, v.z};
}

template <typename KeyType>
static __host__ __device__ __forceinline__ bool
Cornerstone_Node_Overlaps_Shifted_Box_Probe(KeyType prefix,
                                            VECTOR target_center,
                                            VECTOR target_size,
                                            int shift_id,
                                            bool use_morton_sfc)
{
    const auto unit_box =
        cstone::Box<float>(0.0f, 1.0f, cstone::BoundaryType::open);
    const KeyType start_key = cstone::decodePlaceholderBit(prefix);
    const unsigned level = cstone::decodePrefixLength(prefix) / 3;
    const auto node_ibox =
        use_morton_sfc ? cstone::mortonIBox<KeyType>(start_key, level)
                       : cstone::hilbertIBox<KeyType>(start_key, level);
    const auto [node_center, node_size] =
        cstone::centerAndSize<KeyType>(node_ibox, unit_box);
    const VECTOR shifted_center =
        target_center + Shift_Fractional_From_Id_Probe(shift_id);
    return cstone::overlap(node_center, node_size,
                           To_Cstone_Vec_Probe(shifted_center),
                           To_Cstone_Vec_Probe(target_size), unit_box);
}

template <typename KeyType>
static __host__ __device__ __forceinline__ bool
Cornerstone_Node_Overlaps_Preshifted_Box_Probe(KeyType prefix,
                                               VECTOR shifted_target_center,
                                               VECTOR target_size,
                                               bool use_morton_sfc)
{
    const auto unit_box =
        cstone::Box<float>(0.0f, 1.0f, cstone::BoundaryType::open);
    const KeyType start_key = cstone::decodePlaceholderBit(prefix);
    const unsigned level = cstone::decodePrefixLength(prefix) / 3;
    const auto node_ibox =
        use_morton_sfc ? cstone::mortonIBox<KeyType>(start_key, level)
                       : cstone::hilbertIBox<KeyType>(start_key, level);
    const auto [node_center, node_size] =
        cstone::centerAndSize<KeyType>(node_ibox, unit_box);
    return cstone::overlap(node_center, node_size,
                           To_Cstone_Vec_Probe(shifted_target_center),
                           To_Cstone_Vec_Probe(target_size), unit_box);
}

static __host__ __device__ __forceinline__ bool
Cluster_Aabb_Overlaps_Shifted_CutoffSq_Probe(VECTOR center_i, VECTOR extent_i,
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

static __device__ __forceinline__ bool
Leaf_Has_Fixed_Shift_Candidate_Overlap_Subgroup_Probe(
    int cluster_i_start, int leaf_cluster_start, int leaf_cluster_end,
    int fixed_shift_id, float cutoff_sq, VECTOR shift_vec,
    const unsigned int* cluster_valid_masks, const VECTOR* cluster_centers,
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
            const bool central_self_pair =
                fixed_shift_id == kClusteredCentralShiftId &&
                cluster_j >= cluster_i_start && cluster_i > cluster_j;
            if (!central_self_pair)
            {
                lane_overlap = Cluster_Aabb_Overlaps_Shifted_CutoffSq_Probe(
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
Shift_Clustered_Atom_Into_Sorted_XQ_Frame_Probe(const VECTOR atom_crd,
                                                const VECTOR cluster_center,
                                                const LTMatrix3 cell,
                                                const LTMatrix3 rcell)
{
    return cluster_center +
           Get_Periodic_Displacement(atom_crd, cluster_center, cell, rcell);
}

static __host__ __device__ __forceinline__ bool
Clustered_Split_Has_Atoms_Probe(unsigned int valid_mask_j, int split)
{
    return (valid_mask_j & Clustered_Split_Valid_Mask(split)) != 0u;
}

static __device__ __forceinline__ bool Exclusion_List_Contains_Probe(
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

static __device__ __forceinline__ unsigned int
Prune_Record_Stream_Source_Imask_From_Layout_Probe(
    const LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE& source,
    const int* permutation, const int* cluster_offsets,
    const int* super_cluster_offsets, const unsigned int* cluster_local_masks,
    const VECTOR* cluster_centers, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float cutoff_sq)
{
    if (source.imask == 0u || permutation == NULL || cluster_offsets == NULL ||
        super_cluster_offsets == NULL || cluster_local_masks == NULL ||
        cluster_centers == NULL || crd == NULL)
    {
        return 0u;
    }
    if (cutoff_sq <= 0.0f)
    {
        return source.imask;
    }

    const int cluster_i_start =
        super_cluster_offsets[source.supercluster_id];
    const int cluster_j = source.cluster_j;
    const VECTOR center_j = cluster_centers[cluster_j];
    const VECTOR pair_shift = Shift_Vector_From_Id_Probe(source.shift_id, cell);
    const int j_lane_base = source.split_id * kClusteredSplitJClusterSize;
    unsigned int pruned_imask = 0u;
#pragma unroll
    for (int i_local = 0; i_local < kClusteredMaxSuperClusterClusters;
         i_local += 1)
    {
        const unsigned int i_local_bit =
            1u << static_cast<unsigned int>(i_local);
        if ((source.imask & i_local_bit) == 0u)
        {
            continue;
        }
        const int cluster_i = cluster_i_start + i_local;
        const unsigned int local_mask_i = cluster_local_masks[cluster_i];
        if (local_mask_i == 0u)
        {
            continue;
        }

        bool any_in_range = false;
        const VECTOR center_i = cluster_centers[cluster_i];
#pragma unroll
        for (int i_lane = 0; i_lane < kClusteredClusterSize; i_lane += 1)
        {
            if ((local_mask_i & (1u << static_cast<unsigned int>(i_lane))) ==
                0u)
            {
                continue;
            }
            const int atom_i = permutation[cluster_offsets[cluster_i] + i_lane];
            const VECTOR shifted_i =
                Shift_Clustered_Atom_Into_Sorted_XQ_Frame_Probe(
                    crd[atom_i], center_i, cell, rcell);
#pragma unroll
            for (int split_j_lane = 0;
                 split_j_lane < kClusteredSplitJClusterSize; split_j_lane += 1)
            {
                const int j_lane = j_lane_base + split_j_lane;
                if ((source.valid_mask_j &
                     (1u << static_cast<unsigned int>(j_lane))) == 0u)
                {
                    continue;
                }
                const int atom_j =
                    permutation[cluster_offsets[cluster_j] + j_lane];
                const VECTOR shifted_j =
                    Shift_Clustered_Atom_Into_Sorted_XQ_Frame_Probe(
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

static __device__ __forceinline__ bool Cached_Clusters_May_Share_Molecule_Probe(
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
Build_Exclusion_Mask_From_Cached_Atoms_Probe(
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
    if (!Cached_Clusters_May_Share_Molecule_Probe(
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
            bool excluded = Exclusion_List_Contains_Probe(
                atom_i, atom_j, excluded_list_start, excluded_list,
                excluded_numbers);
            if (!excluded && atom_j < local_atom_numbers)
            {
                excluded = Exclusion_List_Contains_Probe(
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

static __global__ void Materialize_Record_Stream_Sources_From_Gmxpacked_Probe(
    const int sci_numbers, const LJ_CLUSTERED_GMXPACKED_SCI* gmxpacked_sci,
    const LJ_CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked,
    const LJ_CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    const int source_capacity,
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    int* source_cursor, int* overflow_rows)
{
    const int sci = blockIdx.x;
    if (sci >= sci_numbers || gmxpacked_sci == NULL ||
        gmxpacked_cjpacked == NULL || sources == NULL ||
        source_cursor == NULL || overflow_rows == NULL)
    {
        return;
    }
    const LJ_CLUSTERED_GMXPACKED_SCI sci_entry = gmxpacked_sci[sci];
    const int packed_count = sci_entry.cjpacked_end - sci_entry.cjpacked_begin;
    const int total_records =
        packed_count * kClusteredJGroupSize * kClusteredWarpSplitCount;
    for (int record = threadIdx.x; record < total_records;
         record += blockDim.x)
    {
        const int local_packed =
            record / (kClusteredJGroupSize * kClusteredWarpSplitCount);
        const int rem =
            record - local_packed * kClusteredJGroupSize *
                         kClusteredWarpSplitCount;
        const int jm = rem / kClusteredWarpSplitCount;
        const int split = rem - jm * kClusteredWarpSplitCount;
        const int packed_idx = sci_entry.cjpacked_begin + local_packed;
        const LJ_CLUSTERED_GMXPACKED_CJ packed = gmxpacked_cjpacked[packed_idx];
        const int cluster_j = packed.cj[jm];
        if (cluster_j < 0)
        {
            continue;
        }
        const LJ_CLUSTERED_GMXPACKED_SPLIT split_entry = packed.split[split];
        const unsigned int imask =
            (split_entry.imask >> Clustered_Jm_Imask_Shift(jm)) &
            ((1u << kClusteredSuperClusterClusters) - 1u);
        if (imask == 0u)
        {
            continue;
        }
        const int write_idx = atomicAdd(source_cursor, 1);
        if (write_idx < 0 || write_idx >= source_capacity)
        {
            atomicAdd(overflow_rows, 1);
            continue;
        }
        LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE source = {};
        source.sci_id = sci;
        source.shift_id = sci_entry.shift_id;
        source.supercluster_id = sci_entry.supercluster_id;
        source.cluster_j = cluster_j;
        source.split_id = split;
        source.imask = imask;
        source.valid_mask_j = 0xffu;
        source.local_mask_j = 0xffu;
        source.source_order = write_idx;
        const int excl_index = split_entry.exclusion_index;
#pragma unroll
        for (int pair_idx = 0;
             pair_idx < kClusteredGmxpackedExclusionPairCount; pair_idx += 1)
        {
            unsigned int pair_word = 0xffffffffu;
            if (excl_index != 0 && exclusion_entries != NULL)
            {
                pair_word = exclusion_entries[excl_index].pair[pair_idx];
            }
            source.pair_exclusion_words[pair_idx] = pair_word & imask;
        }
        sources[write_idx] = source;
    }
}

static __global__ void Count_Record_Stream_Inner_Active_Sources_Probe(
    const int source_rows,
    const LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    const int* permutation, const int* cluster_offsets,
    const int* super_cluster_offsets, const unsigned int* cluster_local_masks,
    const VECTOR* cluster_centers, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float cutoff_sq, int* active_flags,
    unsigned int* active_imasks_by_source)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (source_idx >= source_rows)
    {
        return;
    }
    const unsigned int pruned_imask =
        Prune_Record_Stream_Source_Imask_From_Layout_Probe(
            sources[source_idx], permutation, cluster_offsets,
            super_cluster_offsets, cluster_local_masks, cluster_centers, crd,
            cell, rcell, cutoff_sq);
    active_flags[source_idx] = pruned_imask != 0u ? 1 : 0;
    if (active_imasks_by_source != NULL)
    {
        active_imasks_by_source[source_idx] = pruned_imask;
    }
}

static __global__ void Fill_Record_Stream_Inner_Active_Sources_Probe(
    const int source_rows,
    const LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    const int* permutation, const int* cluster_offsets,
    const int* super_cluster_offsets, const unsigned int* cluster_local_masks,
    const VECTOR* cluster_centers, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float cutoff_sq, const int* active_offsets,
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* active_sources)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (source_idx >= source_rows)
    {
        return;
    }
    const unsigned int pruned_imask =
        Prune_Record_Stream_Source_Imask_From_Layout_Probe(
            sources[source_idx], permutation, cluster_offsets,
            super_cluster_offsets, cluster_local_masks, cluster_centers, crd,
            cell, rcell, cutoff_sq);
    if (pruned_imask == 0u)
    {
        return;
    }
    const int write_idx =
        active_offsets != NULL ? active_offsets[source_idx] : source_idx;
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE source = sources[source_idx];
    source.imask = pruned_imask;
    source.source_order = write_idx;
#pragma unroll
    for (int pair_idx = 0; pair_idx < kClusteredGmxpackedExclusionPairCount;
         pair_idx += 1)
    {
        source.pair_exclusion_words[pair_idx] &= pruned_imask;
    }
    active_sources[write_idx] = source;
}

static __global__ void Fill_Record_Stream_Inner_Active_Sources_Cached_Probe(
    const int source_rows,
    const LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    const unsigned int* active_imasks_by_source, const int* active_offsets,
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* active_sources)
{
    const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (source_idx >= source_rows || active_imasks_by_source == NULL)
    {
        return;
    }
    const unsigned int pruned_imask = active_imasks_by_source[source_idx];
    if (pruned_imask == 0u)
    {
        return;
    }
    const int write_idx =
        active_offsets != NULL ? active_offsets[source_idx] : source_idx;
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE source = sources[source_idx];
    source.imask = pruned_imask;
    source.source_order = write_idx;
#pragma unroll
    for (int pair_idx = 0; pair_idx < kClusteredGmxpackedExclusionPairCount;
         pair_idx += 1)
    {
        source.pair_exclusion_words[pair_idx] &= pruned_imask;
    }
    active_sources[write_idx] = source;
}

static __device__ __forceinline__ unsigned int
Prune_Source_Imask_With_Shift_Probe(
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
                Shift_Clustered_Atom_Into_Sorted_XQ_Frame_Probe(
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
                    Shift_Clustered_Atom_Into_Sorted_XQ_Frame_Probe(
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
Prune_Source_Imask_With_Shift_Cooperative_Probe(
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
                Shift_Clustered_Atom_Into_Sorted_XQ_Frame_Probe(
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

template <bool kSourcePrune, bool kEmitFragments, bool kBuildExclusions>
static __global__ void Probe_Count_Nbnxm_Payload_Fixed_Light(
    const int candidate_sci_numbers, const int cluster_size,
    const int local_atom_numbers, const float record_stream_cutoff,
    const LTMatrix3 cell, const LTMatrix3 rcell, const VECTOR* crd,
    const int* permutation, const int* cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* super_cluster_offsets, const int* cluster_to_supercluster,
    const int* sci_supercluster_ids, const int* candidate_leaf_offsets,
    const int* candidate_leaf_ids, const int candidate_leaf_cluster_stride,
    const unsigned int* candidate_leaf_reach_masks,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const uint64_t* cluster_molecule_signatures,
    const int* cluster_molecule_ids, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers,
    const int max_leaf_cluster_span, int* probe_counts,
    LJ_CLUSTERED_GMXPACKED_COUNT_EXPERIMENT_LIGHT_FRAGMENT* probe_fragments,
    const int probe_fragment_capacity, int* probe_fragment_cursor,
    int* probe_fragment_overflow_rows)
{
    constexpr int kSubgroupSize = kClusteredClusterSize;
    constexpr int kCountSubgroups =
        kClusteredBuilderWarpSize / kSubgroupSize;
    constexpr int kWarpsPerBlock =
        kClusteredBuilderBlockSize / kClusteredBuilderWarpSize;
    const int lane_id = threadIdx.x & (warpSize - 1);
    const int warp_id = threadIdx.x / warpSize;
    const int warps_per_block = blockDim.x / warpSize;
    const int subgroup = lane_id / kSubgroupSize;
    const int sublane = lane_id % kSubgroupSize;
    const device_mask_t subgroup_mask =
        static_cast<device_mask_t>(0xFFu) << (subgroup * kSubgroupSize);
    const int candidate_sci = blockIdx.x * warps_per_block + warp_id;
    if (candidate_sci >= candidate_sci_numbers)
    {
        return;
    }

    __shared__ int shared_probe_counts[kWarpsPerBlock];
    __shared__ unsigned int shared_i_local_masks[kWarpsPerBlock]
                                                [kClusteredMaxSuperClusterClusters];
    __shared__ int shared_i_atom_ids[kWarpsPerBlock]
                                    [kClusteredMaxSuperClusterClusters]
                                    [kClusteredClusterSize];
    __shared__ int shared_i_molecule_ids[kWarpsPerBlock]
                                        [kClusteredMaxSuperClusterClusters]
                                        [kClusteredClusterSize];
    __shared__ uint64_t shared_i_signatures[kWarpsPerBlock]
                                           [kClusteredMaxSuperClusterClusters];
    __shared__ float shared_i_center_x[kWarpsPerBlock]
                                      [kClusteredMaxSuperClusterClusters];
    __shared__ float shared_i_center_y[kWarpsPerBlock]
                                      [kClusteredMaxSuperClusterClusters];
    __shared__ float shared_i_center_z[kWarpsPerBlock]
                                      [kClusteredMaxSuperClusterClusters];
    __shared__ int shared_j_atom_ids[kWarpsPerBlock][kCountSubgroups]
                                    [kClusteredClusterSize];
    __shared__ int shared_j_molecule_ids[kWarpsPerBlock][kCountSubgroups]
                                        [kClusteredClusterSize];
    __shared__ uint64_t shared_j_signature[kWarpsPerBlock][kCountSubgroups];

    if (lane_id == 0)
    {
        shared_probe_counts[warp_id] = 0;
    }
    __syncwarp();

    const bool has_molecule_metadata =
        cluster_molecule_signatures != NULL && cluster_molecule_ids != NULL;
    const int fixed_shift_id = candidate_sci % kClusteredShiftCount;
    const VECTOR fixed_shift_vec =
        Shift_Vector_From_Id_Probe(fixed_shift_id, cell);
    const float record_stream_cutoff_sq =
        record_stream_cutoff * record_stream_cutoff;
    const int sci_base = candidate_sci / kClusteredShiftCount;
    const int super_i = sci_supercluster_ids[sci_base];
    const int cluster_i_start = super_cluster_offsets[super_i];
    const int cluster_i_end = super_cluster_offsets[super_i + 1];
    const int active_cluster_count = cluster_i_end - cluster_i_start;
    const unsigned int active_i_lane_mask =
        active_cluster_count > 0
            ? ((1u << static_cast<unsigned int>(active_cluster_count)) - 1u)
            : 0u;

    if constexpr (kSourcePrune || kBuildExclusions)
    {
        if (lane_id < kClusteredMaxSuperClusterClusters)
        {
            const int cluster_i = cluster_i_start + lane_id;
            if (cluster_i < cluster_i_end)
            {
                const VECTOR center_i = cluster_centers[cluster_i];
                shared_i_local_masks[warp_id][lane_id] =
                    cluster_local_masks[cluster_i];
                shared_i_center_x[warp_id][lane_id] = center_i.x;
                shared_i_center_y[warp_id][lane_id] = center_i.y;
                shared_i_center_z[warp_id][lane_id] = center_i.z;
                shared_i_signatures[warp_id][lane_id] =
                    has_molecule_metadata
                        ? cluster_molecule_signatures[cluster_i]
                        : 0ull;
            }
            else
            {
                shared_i_local_masks[warp_id][lane_id] = 0u;
                shared_i_center_x[warp_id][lane_id] = 0.0f;
                shared_i_center_y[warp_id][lane_id] = 0.0f;
                shared_i_center_z[warp_id][lane_id] = 0.0f;
                shared_i_signatures[warp_id][lane_id] = 0ull;
            }
        }
        for (int atom_slot = lane_id;
             atom_slot < kClusteredMaxSuperClusterClusters *
                             kClusteredClusterSize;
             atom_slot += warpSize)
        {
            const int i_local = atom_slot / kClusteredClusterSize;
            const int atom_lane = atom_slot % kClusteredClusterSize;
            const int cluster_i = cluster_i_start + i_local;
            if (cluster_i < cluster_i_end)
            {
                const int sorted_atom_i =
                    cluster_offsets[cluster_i] + atom_lane;
                shared_i_atom_ids[warp_id][i_local][atom_lane] =
                    permutation[sorted_atom_i];
                shared_i_molecule_ids[warp_id][i_local][atom_lane] =
                    has_molecule_metadata
                        ? cluster_molecule_ids[cluster_i *
                                                   kClusteredClusterSize +
                                               atom_lane]
                        : -1;
            }
            else
            {
                shared_i_atom_ids[warp_id][i_local][atom_lane] = -1;
                shared_i_molecule_ids[warp_id][i_local][atom_lane] = -1;
            }
        }
        __syncwarp();
    }

    const int leaf_begin = candidate_leaf_offsets[candidate_sci];
    const int leaf_end = candidate_leaf_offsets[candidate_sci + 1];
    const int s_max = max_leaf_cluster_span;
    int local_probe_count = 0;

    for (int candidate_idx = leaf_begin + subgroup; candidate_idx < leaf_end;
         candidate_idx += kCountSubgroups)
    {
        const int leaf_j = candidate_leaf_ids[candidate_idx];
        const int cluster_j_start = leaf_cluster_starts[leaf_j];
        const int cluster_j_end = leaf_cluster_ends[leaf_j];
        int prev_running_max_end = 0;
        for (int b = candidate_idx - 1; b >= leaf_begin; b -= 1)
        {
            const int b_leaf = candidate_leaf_ids[b];
            const int b_start = leaf_cluster_starts[b_leaf];
            const int b_end = leaf_cluster_ends[b_leaf];
            if (b_end > prev_running_max_end)
            {
                prev_running_max_end = b_end;
            }
            if (b_start + s_max <= prev_running_max_end)
            {
                break;
            }
        }
        const int leaf_mask_base =
            candidate_idx * candidate_leaf_cluster_stride;
        const int deduped_cluster_j_start =
            IntMaxProbe(cluster_j_start, prev_running_max_end);
        for (int cluster_j = deduped_cluster_j_start; cluster_j < cluster_j_end;
             cluster_j += 1)
        {
            unsigned int precomputed_i_mask = 0u;
            if (sublane == 0)
            {
                precomputed_i_mask =
                    candidate_leaf_reach_masks[leaf_mask_base +
                                               (cluster_j - cluster_j_start)];
            }
            precomputed_i_mask =
                deviceShfl(subgroup_mask, precomputed_i_mask,
                           subgroup * kSubgroupSize, warpSize) &
                active_i_lane_mask;
            if (precomputed_i_mask == 0u)
            {
                continue;
            }

            unsigned int valid_mask_j = 0u;
            unsigned int local_mask_j = 0u;
            int super_j = 0;
            uint64_t signature_j = 0ull;
            VECTOR center_j = {0.0f, 0.0f, 0.0f};
            if (sublane == 0)
            {
                valid_mask_j = cluster_valid_masks[cluster_j];
                local_mask_j = cluster_local_masks[cluster_j];
                if (valid_mask_j != 0u)
                {
                    super_j = cluster_to_supercluster[cluster_j];
                    center_j = cluster_centers != NULL
                                   ? cluster_centers[cluster_j]
                                   : VECTOR{0.0f, 0.0f, 0.0f};
                    if (has_molecule_metadata)
                    {
                        signature_j = cluster_molecule_signatures[cluster_j];
                    }
                }
            }
            const int subgroup_leader = subgroup * kSubgroupSize;
            valid_mask_j =
                deviceShfl(subgroup_mask, valid_mask_j, subgroup_leader,
                           warpSize);
            local_mask_j =
                deviceShfl(subgroup_mask, local_mask_j, subgroup_leader,
                           warpSize);
            super_j =
                deviceShfl(subgroup_mask, super_j, subgroup_leader, warpSize);
            center_j.x = deviceShfl(subgroup_mask, center_j.x,
                                    subgroup_leader, warpSize);
            center_j.y = deviceShfl(subgroup_mask, center_j.y,
                                    subgroup_leader, warpSize);
            center_j.z = deviceShfl(subgroup_mask, center_j.z,
                                    subgroup_leader, warpSize);
            unsigned int sig_lo = static_cast<unsigned int>(signature_j);
            unsigned int sig_hi =
                static_cast<unsigned int>(signature_j >> 32);
            sig_lo =
                deviceShfl(subgroup_mask, sig_lo, subgroup_leader, warpSize);
            sig_hi =
                deviceShfl(subgroup_mask, sig_hi, subgroup_leader, warpSize);
            signature_j = (static_cast<uint64_t>(sig_hi) << 32) |
                          static_cast<uint64_t>(sig_lo);
            if (valid_mask_j == 0u)
            {
                continue;
            }
            if (local_mask_j != 0u && super_j < super_i)
            {
                continue;
            }

            if constexpr (!kSourcePrune)
            {
                if (sublane == 0)
                {
                    local_probe_count += 1;
                }
                continue;
            }

            if (sublane < kClusteredClusterSize)
            {
                if ((valid_mask_j & (1u << sublane)) != 0u)
                {
                    const int sorted_atom_j = cluster_offsets[cluster_j] + sublane;
                    shared_j_atom_ids[warp_id][subgroup][sublane] =
                        permutation[sorted_atom_j];
                    shared_j_molecule_ids[warp_id][subgroup][sublane] =
                        has_molecule_metadata
                            ? cluster_molecule_ids[cluster_j *
                                                       kClusteredClusterSize +
                                                   sublane]
                            : -1;
                }
                else
                {
                    shared_j_atom_ids[warp_id][subgroup][sublane] = -1;
                    shared_j_molecule_ids[warp_id][subgroup][sublane] = -1;
                }
            }
            if (sublane == 0)
            {
                shared_j_signature[warp_id][subgroup] = signature_j;
            }
            deviceSyncWarp(subgroup_mask);

            if (sublane == 0)
            {
                unsigned long long
                    exclusion_masks[kClusteredMaxSuperClusterClusters] = {};
                if constexpr (kBuildExclusions)
                {
                    unsigned int remaining_i = precomputed_i_mask;
                    while (remaining_i != 0u)
                    {
                        const int i_local =
                            __ffs(static_cast<int>(remaining_i)) - 1;
                        remaining_i &= (remaining_i - 1u);
                        const unsigned int local_mask_i =
                            shared_i_local_masks[warp_id][i_local];
                        const bool exclusion_candidate =
                            !has_molecule_metadata ||
                            (shared_i_signatures[warp_id][i_local] &
                             shared_j_signature[warp_id][subgroup]) != 0ull;
                        if (!exclusion_candidate)
                        {
                            continue;
                        }
                        exclusion_masks[i_local] =
                            Build_Exclusion_Mask_From_Cached_Atoms_Probe(
                                shared_i_atom_ids[warp_id][i_local],
                                shared_j_atom_ids[warp_id][subgroup],
                                shared_i_molecule_ids[warp_id][i_local],
                                shared_j_molecule_ids[warp_id][subgroup],
                                shared_i_signatures[warp_id][i_local],
                                shared_j_signature[warp_id][subgroup],
                                has_molecule_metadata, local_mask_i,
                                valid_mask_j, cluster_size, local_atom_numbers,
                                excluded_list_start, excluded_list,
                                excluded_numbers);
                    }
                }

                for (int split = 0; split < kClusteredWarpSplitCount;
                     split += 1)
                {
                    const unsigned int split_local_imask =
                        Prune_Source_Imask_With_Shift_Probe(
                            split, precomputed_i_mask, valid_mask_j,
                            fixed_shift_vec, cell, rcell, crd,
                            shared_i_center_x[warp_id],
                            shared_i_center_y[warp_id],
                            shared_i_center_z[warp_id], center_j,
                            shared_i_atom_ids[warp_id],
                            shared_j_atom_ids[warp_id][subgroup],
                            shared_i_local_masks[warp_id],
                            record_stream_cutoff_sq);
                    if (!Clustered_Split_Has_Atoms_Probe(valid_mask_j, split) ||
                        split_local_imask == 0u)
                    {
                        continue;
                    }
                    local_probe_count += 1;
                    if constexpr (kEmitFragments)
                    {
                        const int fragment_idx =
                            atomicAdd(probe_fragment_cursor, 1);
                        if (fragment_idx < probe_fragment_capacity)
                        {
                            LJ_CLUSTERED_GMXPACKED_COUNT_EXPERIMENT_LIGHT_FRAGMENT
                                fragment = {};
                            fragment.sci_id = candidate_sci;
                            fragment.shift_id = fixed_shift_id;
                            fragment.supercluster_id = super_i;
                            fragment.cluster_j = cluster_j;
                            fragment.split_id = split;
                            fragment.imask = split_local_imask;
                            fragment.valid_mask_j = valid_mask_j;
                            fragment.local_mask_j = local_mask_j;
                            fragment.source_order = fragment_idx;
                            if constexpr (kBuildExclusions)
                            {
                                for (int i_local = 0;
                                     i_local <
                                     kClusteredMaxSuperClusterClusters;
                                     i_local += 1)
                                {
                                    fragment.exclusion_masks[i_local] =
                                        exclusion_masks[i_local];
                                }
                            }
                            probe_fragments[fragment_idx] = fragment;
                        }
                        else
                        {
                            atomicAdd(probe_fragment_overflow_rows, 1);
                        }
                    }
                }
            }
            deviceSyncWarp(subgroup_mask);
        }
    }
    if (sublane == 0 && local_probe_count > 0)
    {
        atomicAdd(&shared_probe_counts[warp_id], local_probe_count);
    }
    __syncwarp();
    if (lane_id == 0 && probe_counts != NULL)
    {
        probe_counts[candidate_sci] = shared_probe_counts[warp_id];
    }
}

template <bool kScreen, bool kEmit, bool kFastNodeOverlap,
          bool kCoopTraversal, bool kStats, bool kRootChildSplit>
static __global__ void Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup(
    const int sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    const int* super_cluster_offsets, const int* leaf_cluster_starts,
    const int* leaf_cluster_ends, const int* leaf_all_local,
    const LTMatrix3 cell, const float cutoff, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, const bool central_halfshell_culling,
    const bool use_morton_sfc, const int onepass_capacity,
    int* probe_counts,
    LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_RECORD* probe_records,
    int* probe_cursor, int* probe_overflow,
    LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_STATS* probe_stats,
    const int* candidate_sci_index_map, const int* root_child_task_sci_ids,
    const int* root_child_task_nodes)
{
    const int groups_per_block =
        blockDim.x / kFixedShiftCandidateLeafSubgroupSize;
    const int group_id = threadIdx.x / kFixedShiftCandidateLeafSubgroupSize;
    const int sublane = threadIdx.x % kFixedShiftCandidateLeafSubgroupSize;
    const int lane_id = threadIdx.x & (kClusteredBuilderWarpSize - 1);
    const int subgroup_lane_base = lane_id - sublane;
    const device_mask_t subgroup_mask =
        (((device_mask_t)1 << kFixedShiftCandidateLeafSubgroupSize) - 1u)
        << subgroup_lane_base;
    const int logical_item = blockIdx.x * groups_per_block + group_id;
    const bool use_root_child_queue =
        root_child_task_sci_ids != NULL && root_child_task_nodes != NULL;
    const int logical_sci =
        use_root_child_queue ? logical_item
                             : (kRootChildSplit ? logical_item / 8
                                                : logical_item);
    const int root_child_slot =
        use_root_child_queue ? 0 : (kRootChildSplit ? logical_item % 8 : 0);
    if (logical_item >= sci_numbers && use_root_child_queue)
    {
        return;
    }
    if (!use_root_child_queue && logical_sci >= sci_numbers)
    {
        return;
    }
    const int sci = use_root_child_queue
                        ? root_child_task_sci_ids[logical_item]
                        : (candidate_sci_index_map != NULL
                               ? candidate_sci_index_map[logical_sci]
                               : logical_sci);

    const int sci_base =
        candidate_shift_ids == NULL ? sci / kClusteredShiftCount : sci;
    const int candidate_shift_id =
        candidate_shift_ids != NULL ? candidate_shift_ids[sci]
                                    : (sci % kClusteredShiftCount);
    const int super_i = sci_supercluster_ids[sci_base];
    const VECTOR target_center = super_cluster_centers[super_i];
    const VECTOR target_size = super_cluster_sizes[super_i];
    const int cluster_i_start = super_cluster_offsets[super_i];
    const int cluster_i_end = super_cluster_offsets[super_i + 1];
    const VECTOR shifted_target_center =
        target_center + Shift_Fractional_From_Id_Probe(candidate_shift_id);
    const VECTOR shift_vec = Shift_Vector_From_Id_Probe(candidate_shift_id, cell);
    const float cutoff_sq = cutoff * cutoff;
    const int cluster_i = cluster_i_start + sublane;
    const bool lane_i_valid =
        cluster_i < cluster_i_end && cluster_local_masks[cluster_i] != 0u;
    const VECTOR center_i =
        lane_i_valid ? cluster_centers[cluster_i] : VECTOR{0.0f, 0.0f, 0.0f};
    const VECTOR extent_i =
        lane_i_valid ? cluster_extents[cluster_i] : VECTOR{0.0f, 0.0f, 0.0f};
    int count = 0;
    int running_max_end = 0;
    LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_STATS stats = {};

    auto write_stats = [&]()
    {
        if constexpr (kStats)
        {
            if (sublane == 0 && probe_stats != NULL)
            {
                stats.accepted_leaves = count;
                probe_stats[logical_sci] = stats;
            }
        }
    };

    auto overlaps = [&](int node)
    {
        if constexpr (kStats)
        {
            if (sublane == 0)
            {
                stats.overlap_tests += 1;
            }
        }
        bool result = false;
        if constexpr (kFastNodeOverlap)
        {
            result = Cornerstone_Node_Overlaps_Preshifted_Box_Probe(
                node_prefixes[node], shifted_target_center, target_size,
                use_morton_sfc);
        }
        else
        {
            result = Cornerstone_Node_Overlaps_Shifted_Box_Probe(
                node_prefixes[node], target_center, target_size,
                candidate_shift_id, use_morton_sfc);
        }
        if constexpr (kStats)
        {
            if (sublane == 0 && node == 0 && !result)
            {
                stats.root_rejects += 1;
            }
        }
        return result;
    };
    auto endpoint_leaf = [&](int leaf_j)
    {
        if (leaf_j < 0)
        {
            return;
        }
        if constexpr (kStats)
        {
            if (sublane == 0)
            {
                stats.endpoint_leaves += 1;
            }
        }
        if (central_halfshell_culling &&
            candidate_shift_id == kClusteredCentralShiftId &&
            leaf_all_local[leaf_j] != 0 &&
            leaf_cluster_ends[leaf_j] <= cluster_i_start)
        {
            if constexpr (kStats)
            {
                if (sublane == 0)
                {
                    stats.halfshell_rejects += 1;
                }
            }
            return;
        }
        if constexpr (kScreen)
        {
            if constexpr (kStats)
            {
                if (sublane == 0)
                {
                    stats.screen_tests += 1;
                }
            }
            if (!Leaf_Has_Fixed_Shift_Candidate_Overlap_Subgroup_Probe(
                    cluster_i_start, leaf_cluster_starts[leaf_j],
                    leaf_cluster_ends[leaf_j], candidate_shift_id, cutoff_sq,
                    shift_vec, cluster_valid_masks, cluster_centers,
                    cluster_extents, cluster_i, lane_i_valid, center_i, extent_i,
                    subgroup_mask))
            {
                return;
            }
            if constexpr (kStats)
            {
                if (sublane == 0)
                {
                    stats.screen_accepts += 1;
                }
            }
        }
        if (sublane == 0)
        {
            const int rank = count;
            const int prev_running_max_end = running_max_end;
            count += 1;
            if constexpr (kEmit)
            {
                const int write_idx = atomicAdd(probe_cursor, 1);
                if (write_idx < onepass_capacity && probe_records != NULL)
                {
                    LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_RECORD record;
                    record.sci_id = sci;
                    record.rank = rank;
                    record.leaf_id = leaf_j;
                    record.prev_running_max_end = prev_running_max_end;
                    probe_records[write_idx] = record;
                }
                else if (probe_overflow != NULL)
                {
                    atomicAdd(probe_overflow, 1);
                }
            }
            running_max_end =
                IntMaxProbe(running_max_end, leaf_cluster_ends[leaf_j]);
        }
    };
    auto endpoint = [&](int node) { endpoint_leaf(internal_to_leaf[node]); };
    if constexpr (kCoopTraversal)
    {
        constexpr int init_node = 0;
        int root_descend = 0;
        int root_child = 0;
        int root_leaf = -1;
        if (sublane == 0)
        {
            if constexpr (kStats)
            {
                stats.node_visits += 1;
            }
            root_descend = overlaps(init_node) ? 1 : 0;
            root_child = child_offsets[init_node];
            root_leaf = internal_to_leaf[init_node];
        }
        root_descend =
            deviceShfl(subgroup_mask, root_descend, subgroup_lane_base,
                       kClusteredBuilderWarpSize);
        root_child = deviceShfl(subgroup_mask, root_child, subgroup_lane_base,
                                kClusteredBuilderWarpSize);
        root_leaf = deviceShfl(subgroup_mask, root_leaf, subgroup_lane_base,
                               kClusteredBuilderWarpSize);
        if (root_descend == 0)
        {
            if (sublane == 0 && probe_counts != NULL)
            {
                probe_counts[logical_sci] = 0;
            }
            write_stats();
            return;
        }
        if (root_child == 0)
        {
            endpoint_leaf(root_leaf);
            if (sublane == 0 && probe_counts != NULL)
            {
                probe_counts[logical_sci] = count;
            }
            write_stats();
            return;
        }

        int node = root_child;
        int backtrack = 0;
        while (node != init_node)
        {
            int endpoint_active = 0;
            int endpoint_leaf_j = -1;
            int next_node = init_node;
            int next_backtrack = 0;
            if (sublane == 0)
            {
                if constexpr (kStats)
                {
                    stats.node_visits += 1;
                }
                const bool is_leaf = child_offsets[node] == 0;
                const bool descend = !backtrack && overlaps(node);
                endpoint_active = is_leaf && descend ? 1 : 0;
                endpoint_leaf_j =
                    endpoint_active != 0 ? internal_to_leaf[node] : -1;
                const int sibling_idx = (node - 1) % 8;
                if (!is_leaf && descend)
                {
                    next_node = child_offsets[node];
                    next_backtrack = 0;
                }
                else if (sibling_idx < 7)
                {
                    next_node = node + 1;
                    next_backtrack = 0;
                }
                else
                {
                    next_node = parents[(node - 1) / 8];
                    next_backtrack = 1;
                }
            }
            endpoint_active =
                deviceShfl(subgroup_mask, endpoint_active,
                           subgroup_lane_base, kClusteredBuilderWarpSize);
            endpoint_leaf_j =
                deviceShfl(subgroup_mask, endpoint_leaf_j,
                           subgroup_lane_base, kClusteredBuilderWarpSize);
            if (endpoint_active != 0)
            {
                endpoint_leaf(endpoint_leaf_j);
            }
            next_node =
                deviceShfl(subgroup_mask, next_node, subgroup_lane_base,
                           kClusteredBuilderWarpSize);
            next_backtrack =
                deviceShfl(subgroup_mask, next_backtrack, subgroup_lane_base,
                           kClusteredBuilderWarpSize);
            node = next_node;
            backtrack = next_backtrack;
        }
    }
    else
    {
        if constexpr (kRootChildSplit)
        {
            constexpr int init_node = 0;
            const int task_root_node =
                use_root_child_queue ? root_child_task_nodes[logical_item] : -1;
            if (use_root_child_queue && task_root_node == init_node)
            {
                endpoint(init_node);
                if (sublane == 0 && probe_counts != NULL && count > 0)
                {
                    atomicAdd(&probe_counts[sci], count);
                }
                write_stats();
                return;
            }
            if (!use_root_child_queue && !overlaps(init_node))
            {
                return;
            }
            const int root_child =
                use_root_child_queue ? task_root_node : child_offsets[init_node];
            if (!use_root_child_queue && root_child == 0)
            {
                if (root_child_slot == 0)
                {
                    endpoint(init_node);
                }
            }
            else
            {
                int node = use_root_child_queue
                               ? root_child
                               : root_child + root_child_slot;
                bool backtrack = false;
                while (true)
                {
                    const bool is_leaf = child_offsets[node] == 0;
                    const bool descend = !backtrack && overlaps(node);
                    if (is_leaf && descend)
                    {
                        endpoint(node);
                    }
                    if (!is_leaf && descend)
                    {
                        node = child_offsets[node];
                        backtrack = false;
                    }
                    else if (node ==
                             (use_root_child_queue
                                  ? root_child
                                  : root_child + root_child_slot))
                    {
                        break;
                    }
                    else
                    {
                        const int sibling_idx = (node - 1) % 8;
                        if (sibling_idx < 7)
                        {
                            node += 1;
                            backtrack = false;
                        }
                        else
                        {
                            node = parents[(node - 1) / 8];
                            backtrack = true;
                        }
                    }
                }
            }
            if (sublane == 0 && probe_counts != NULL && count > 0)
            {
                atomicAdd(&probe_counts[use_root_child_queue ? sci
                                                             : logical_sci],
                          count);
            }
            write_stats();
            return;
        }
        else if constexpr (kStats)
        {
            constexpr int init_node = 0;
            if (sublane == 0)
            {
                stats.node_visits += 1;
            }
            if (!overlaps(init_node))
            {
                if (sublane == 0 && probe_counts != NULL)
                {
                    probe_counts[logical_sci] = 0;
                }
                write_stats();
                return;
            }
            if (child_offsets[init_node] == 0)
            {
                endpoint(init_node);
            }
            else
            {
                int node = child_offsets[init_node];
                bool backtrack = false;
                while (node != init_node)
                {
                    if (sublane == 0)
                    {
                        stats.node_visits += 1;
                    }
                    const bool is_leaf = child_offsets[node] == 0;
                    const bool descend = !backtrack && overlaps(node);
                    if (is_leaf && descend)
                    {
                        endpoint(node);
                    }

                    const int sibling_idx = (node - 1) % 8;
                    if (!is_leaf && descend)
                    {
                        node = child_offsets[node];
                        backtrack = false;
                    }
                    else if (sibling_idx < 7)
                    {
                        node += 1;
                        backtrack = false;
                    }
                    else
                    {
                        node = parents[(node - 1) / 8];
                        backtrack = true;
                    }
                }
            }
        }
        else
        {
            cstone::singleTraversal(child_offsets, parents, overlaps, endpoint);
        }
    }
    if (sublane == 0 && probe_counts != NULL)
    {
        probe_counts[logical_sci] = count;
    }
    write_stats();
}

template <bool kFastNodeOverlap>
static __global__ void Probe_Candidate_Leaf_Root_Child_Device_Counter_Subgroup(
    const int candidate_sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    const int* super_cluster_offsets, const int* leaf_cluster_starts,
    const int* leaf_cluster_ends, const int* leaf_all_local,
    const LTMatrix3 cell, const float cutoff, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, const bool central_halfshell_culling,
    const bool use_morton_sfc, const int task_capacity,
    const int* task_counter, int* task_work_cursor, int* probe_counts,
    int* task_leaf_counts, const int* root_child_task_sci_ids,
    const int* root_child_task_nodes)
{
    const int group_id = threadIdx.x / kFixedShiftCandidateLeafSubgroupSize;
    const int sublane = threadIdx.x % kFixedShiftCandidateLeafSubgroupSize;
    const int lane_id = threadIdx.x & (kClusteredBuilderWarpSize - 1);
    const int subgroup_lane_base = lane_id - sublane;
    const device_mask_t subgroup_mask =
        (((device_mask_t)1 << kFixedShiftCandidateLeafSubgroupSize) - 1u)
        << subgroup_lane_base;

    while (true)
    {
        int logical_item = task_capacity;
        if (sublane == 0)
        {
            logical_item = atomicAdd(task_work_cursor, 1);
        }
        logical_item =
            deviceShfl(subgroup_mask, logical_item, subgroup_lane_base,
                       kClusteredBuilderWarpSize);
        const int task_count =
            *task_counter < task_capacity ? *task_counter : task_capacity;
        if (logical_item >= task_count)
        {
            break;
        }

        const int sci = root_child_task_sci_ids[logical_item];
        if (sci < 0 || sci >= candidate_sci_numbers)
        {
            continue;
        }

        const int sci_base =
            candidate_shift_ids == NULL ? sci / kClusteredShiftCount : sci;
        const int candidate_shift_id =
            candidate_shift_ids != NULL ? candidate_shift_ids[sci]
                                        : (sci % kClusteredShiftCount);
        const int super_i = sci_supercluster_ids[sci_base];
        const VECTOR target_center = super_cluster_centers[super_i];
        const VECTOR target_size = super_cluster_sizes[super_i];
        const int cluster_i_start = super_cluster_offsets[super_i];
        const int cluster_i_end = super_cluster_offsets[super_i + 1];
        const VECTOR shifted_target_center =
            target_center + Shift_Fractional_From_Id_Probe(candidate_shift_id);
        const VECTOR shift_vec =
            Shift_Vector_From_Id_Probe(candidate_shift_id, cell);
        const float cutoff_sq = cutoff * cutoff;
        const int cluster_i = cluster_i_start + sublane;
        const bool lane_i_valid =
            cluster_i < cluster_i_end && cluster_local_masks[cluster_i] != 0u;
        const VECTOR center_i = lane_i_valid ? cluster_centers[cluster_i]
                                             : VECTOR{0.0f, 0.0f, 0.0f};
        const VECTOR extent_i = lane_i_valid ? cluster_extents[cluster_i]
                                             : VECTOR{0.0f, 0.0f, 0.0f};
        int count = 0;
        int running_max_end = 0;

        auto overlaps = [&](int node)
        {
            if constexpr (kFastNodeOverlap)
            {
                return Cornerstone_Node_Overlaps_Preshifted_Box_Probe(
                    node_prefixes[node], shifted_target_center, target_size,
                    use_morton_sfc);
            }
            else
            {
                return Cornerstone_Node_Overlaps_Shifted_Box_Probe(
                    node_prefixes[node], target_center, target_size,
                    candidate_shift_id, use_morton_sfc);
            }
        };

        auto endpoint_leaf = [&](int leaf_j)
        {
            if (leaf_j < 0)
            {
                return;
            }
            if (central_halfshell_culling &&
                candidate_shift_id == kClusteredCentralShiftId &&
                leaf_all_local[leaf_j] != 0 &&
                leaf_cluster_ends[leaf_j] <= cluster_i_start)
            {
                return;
            }
            if (!Leaf_Has_Fixed_Shift_Candidate_Overlap_Subgroup_Probe(
                    cluster_i_start, leaf_cluster_starts[leaf_j],
                    leaf_cluster_ends[leaf_j], candidate_shift_id, cutoff_sq,
                    shift_vec, cluster_valid_masks, cluster_centers,
                    cluster_extents, cluster_i, lane_i_valid, center_i,
                    extent_i, subgroup_mask))
            {
                return;
            }
            if (sublane == 0)
            {
                count += 1;
                running_max_end =
                    IntMaxProbe(running_max_end, leaf_cluster_ends[leaf_j]);
            }
        };
        auto endpoint = [&](int node) { endpoint_leaf(internal_to_leaf[node]); };

        constexpr int init_node = 0;
        const int task_root_node = root_child_task_nodes[logical_item];
        if (task_root_node == init_node)
        {
            endpoint(init_node);
        }
        else
        {
            int node = task_root_node;
            bool backtrack = false;
            while (true)
            {
                const bool is_leaf = child_offsets[node] == 0;
                const bool descend = !backtrack && overlaps(node);
                if (is_leaf && descend)
                {
                    endpoint(node);
                }
                if (!is_leaf && descend)
                {
                    node = child_offsets[node];
                    backtrack = false;
                }
                else if (node == task_root_node)
                {
                    break;
                }
                else
                {
                    const int sibling_idx = (node - 1) % 8;
                    if (sibling_idx < 7)
                    {
                        node += 1;
                        backtrack = false;
                    }
                    else
                    {
                        node = parents[(node - 1) / 8];
                        backtrack = true;
                    }
                }
            }
        }
        if (sublane == 0 && task_leaf_counts != NULL)
        {
            task_leaf_counts[logical_item] = count;
        }
        if (sublane == 0 && probe_counts != NULL && count > 0)
        {
            atomicAdd(&probe_counts[sci], count);
        }
    }
}

template <bool kFastNodeOverlap>
static __global__ void Build_Candidate_Leaf_Root_Child_Tasks(
    const int candidate_sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    const uint64_t* node_prefixes, const int* child_offsets,
    const int* candidate_shift_ids, const bool use_morton_sfc,
    const int task_capacity, int* task_counter, int* task_overflow,
    int* task_sci_ids, int* task_nodes, const int task_split_depth)
{
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    const int sci = item / 8;
    const int root_child_slot = item % 8;
    if (sci >= candidate_sci_numbers)
    {
        return;
    }
    const int sci_base =
        candidate_shift_ids == NULL ? sci / kClusteredShiftCount : sci;
    const int candidate_shift_id =
        candidate_shift_ids != NULL ? candidate_shift_ids[sci]
                                    : (sci % kClusteredShiftCount);
    const int super_i = sci_supercluster_ids[sci_base];
    const VECTOR target_center = super_cluster_centers[super_i];
    const VECTOR target_size = super_cluster_sizes[super_i];
    const VECTOR shifted_target_center =
        target_center + Shift_Fractional_From_Id_Probe(candidate_shift_id);
    auto overlaps = [&](int node)
    {
        if constexpr (kFastNodeOverlap)
        {
            return Cornerstone_Node_Overlaps_Preshifted_Box_Probe(
                node_prefixes[node], shifted_target_center, target_size,
                use_morton_sfc);
        }
        else
        {
            return Cornerstone_Node_Overlaps_Shifted_Box_Probe(
                node_prefixes[node], target_center, target_size,
                candidate_shift_id, use_morton_sfc);
        }
    };

    constexpr int init_node = 0;
    if (!overlaps(init_node))
    {
        return;
    }
    const int root_child = child_offsets[init_node];
    auto emit_task = [&](int node)
    {
        const int write_idx = atomicAdd(task_counter, 1);
        if (write_idx < task_capacity)
        {
            task_sci_ids[write_idx] = sci;
            task_nodes[write_idx] = node;
        }
        else if (task_overflow != NULL)
        {
            atomicAdd(task_overflow, 1);
        }
    };
    if (root_child == 0)
    {
        if (root_child_slot == 0)
        {
            emit_task(init_node);
        }
    }
    else
    {
        const int node = root_child + root_child_slot;
        if (!overlaps(node))
        {
            return;
        }
        const int node_child = child_offsets[node];
        if (task_split_depth <= 1 || node_child == 0)
        {
            emit_task(node);
        }
        else
        {
            for (int child_slot = 0; child_slot < 8; child_slot += 1)
            {
                const int child_node = node_child + child_slot;
                if (overlaps(child_node))
                {
                    emit_task(child_node);
                }
            }
        }
    }
}

template <bool kFastNodeOverlap>
static __global__ void Emit_Candidate_Leaf_Root_Child_Device_Counter_Subgroup(
    const int candidate_sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    const int* super_cluster_offsets, const int* leaf_cluster_starts,
    const int* leaf_cluster_ends, const int* leaf_all_local,
    const LTMatrix3 cell, const float cutoff, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, const bool central_halfshell_culling,
    const bool use_morton_sfc, const int task_capacity,
    const int* task_counter, int* task_work_cursor, int* emit_counts,
    const int* candidate_leaf_offsets, int* candidate_leaf_ids,
    const int* task_leaf_offsets, int* emit_overflow,
    const int* root_child_task_sci_ids,
    const int* root_child_task_nodes)
{
    const int sublane = threadIdx.x % kFixedShiftCandidateLeafSubgroupSize;
    const int lane_id = threadIdx.x & (kClusteredBuilderWarpSize - 1);
    const int subgroup_lane_base = lane_id - sublane;
    const device_mask_t subgroup_mask =
        (((device_mask_t)1 << kFixedShiftCandidateLeafSubgroupSize) - 1u)
        << subgroup_lane_base;

    while (true)
    {
        int logical_item = task_capacity;
        if (sublane == 0)
        {
            logical_item = atomicAdd(task_work_cursor, 1);
        }
        logical_item =
            deviceShfl(subgroup_mask, logical_item, subgroup_lane_base,
                       kClusteredBuilderWarpSize);
        const int task_count =
            *task_counter < task_capacity ? *task_counter : task_capacity;
        if (logical_item >= task_count)
        {
            break;
        }

        const int sci = root_child_task_sci_ids[logical_item];
        if (sci < 0 || sci >= candidate_sci_numbers)
        {
            continue;
        }

        const int sci_base =
            candidate_shift_ids == NULL ? sci / kClusteredShiftCount : sci;
        const int candidate_shift_id =
            candidate_shift_ids != NULL ? candidate_shift_ids[sci]
                                        : (sci % kClusteredShiftCount);
        const int super_i = sci_supercluster_ids[sci_base];
        const VECTOR target_center = super_cluster_centers[super_i];
        const VECTOR target_size = super_cluster_sizes[super_i];
        const int cluster_i_start = super_cluster_offsets[super_i];
        const int cluster_i_end = super_cluster_offsets[super_i + 1];
        const VECTOR shifted_target_center =
            target_center + Shift_Fractional_From_Id_Probe(candidate_shift_id);
        const VECTOR shift_vec =
            Shift_Vector_From_Id_Probe(candidate_shift_id, cell);
        const float cutoff_sq = cutoff * cutoff;
        const int cluster_i = cluster_i_start + sublane;
        const bool lane_i_valid =
            cluster_i < cluster_i_end && cluster_local_masks[cluster_i] != 0u;
        const VECTOR center_i = lane_i_valid ? cluster_centers[cluster_i]
                                             : VECTOR{0.0f, 0.0f, 0.0f};
        const VECTOR extent_i = lane_i_valid ? cluster_extents[cluster_i]
                                             : VECTOR{0.0f, 0.0f, 0.0f};
        int local_leaf_rank = 0;

        auto overlaps = [&](int node)
        {
            if constexpr (kFastNodeOverlap)
            {
                return Cornerstone_Node_Overlaps_Preshifted_Box_Probe(
                    node_prefixes[node], shifted_target_center, target_size,
                    use_morton_sfc);
            }
            else
            {
                return Cornerstone_Node_Overlaps_Shifted_Box_Probe(
                    node_prefixes[node], target_center, target_size,
                    candidate_shift_id, use_morton_sfc);
            }
        };

        auto endpoint_leaf = [&](int leaf_j)
        {
            if (leaf_j < 0)
            {
                return;
            }
            if (central_halfshell_culling &&
                candidate_shift_id == kClusteredCentralShiftId &&
                leaf_all_local[leaf_j] != 0 &&
                leaf_cluster_ends[leaf_j] <= cluster_i_start)
            {
                return;
            }
            if (!Leaf_Has_Fixed_Shift_Candidate_Overlap_Subgroup_Probe(
                    cluster_i_start, leaf_cluster_starts[leaf_j],
                    leaf_cluster_ends[leaf_j], candidate_shift_id, cutoff_sq,
                    shift_vec, cluster_valid_masks, cluster_centers,
                    cluster_extents, cluster_i, lane_i_valid, center_i,
                    extent_i, subgroup_mask))
            {
                return;
            }
            if (sublane == 0)
            {
                const int rank =
                    task_leaf_offsets != NULL
                        ? local_leaf_rank++
                        : atomicAdd(&emit_counts[sci], 1);
                const int begin = candidate_leaf_offsets[sci];
                const int end = candidate_leaf_offsets[sci + 1];
                const int write_idx =
                    task_leaf_offsets != NULL
                        ? task_leaf_offsets[logical_item] + rank
                        : begin + rank;
                if (write_idx < end)
                {
                    candidate_leaf_ids[write_idx] = leaf_j;
                }
                else if (emit_overflow != NULL)
                {
                    atomicAdd(emit_overflow, 1);
                }
            }
        };
        auto endpoint = [&](int node) { endpoint_leaf(internal_to_leaf[node]); };

        constexpr int init_node = 0;
        const int task_root_node = root_child_task_nodes[logical_item];
        if (task_root_node == init_node)
        {
            endpoint(init_node);
        }
        else
        {
            int node = task_root_node;
            bool backtrack = false;
            while (true)
            {
                const bool is_leaf = child_offsets[node] == 0;
                const bool descend = !backtrack && overlaps(node);
                if (is_leaf && descend)
                {
                    endpoint(node);
                }
                if (!is_leaf && descend)
                {
                    node = child_offsets[node];
                    backtrack = false;
                }
                else if (node == task_root_node)
                {
                    break;
                }
                else
                {
                    const int sibling_idx = (node - 1) % 8;
                    if (sibling_idx < 7)
                    {
                        node += 1;
                        backtrack = false;
                    }
                    else
                    {
                        node = parents[(node - 1) / 8];
                        backtrack = true;
                    }
                }
            }
        }
    }
}

template <bool kDynamicWorkQueue, bool kSlimEmit, bool kCooperativePrune>
static __global__ void Dedicated_Count_Nbnxm_Payload_Fixed_Light(
    const int candidate_sci_numbers, const int cluster_size,
    const int local_atom_numbers, const float record_stream_cutoff,
    const LTMatrix3 cell, const LTMatrix3 rcell, const VECTOR* crd,
    const int* permutation, const int* cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* super_cluster_offsets, const int* cluster_to_supercluster,
    const int* sci_supercluster_ids, const int* candidate_leaf_offsets,
    const int* candidate_leaf_ids, const int candidate_leaf_cluster_stride,
    const int* candidate_leaf_prev_running_max_ends,
    const unsigned int* candidate_leaf_reach_masks,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const uint64_t* cluster_molecule_signatures,
    const int* cluster_molecule_ids, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers,
    const int max_leaf_cluster_span, int* sci_shift_flags,
    int* cjpacked_group_counts, int* exclusion_counts,
    int* record_stream_source_rows,
    int* record_stream_source_counts_by_candidate,
    int* dynamic_work_counter,
    const bool accumulate_record_stream_source_rows_by_candidate,
    LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT* count_light_source_fragments,
    LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT_SLIM*
        count_slim_source_fragments,
    const int count_source_fragment_capacity, int* count_source_fragment_cursor,
    int* count_source_fragment_overflow_rows)
{
    constexpr int kSubgroupSize = kClusteredClusterSize;
    constexpr int kCountSubgroups =
        kClusteredBuilderWarpSize / kSubgroupSize;
    constexpr int kWarpsPerBlock =
        kClusteredBuilderBlockSize / kClusteredBuilderWarpSize;
    const int lane_id = threadIdx.x & (warpSize - 1);
    const int warp_id = threadIdx.x / warpSize;
    const int warps_per_block = blockDim.x / warpSize;
    const int subgroup = lane_id / kSubgroupSize;
    const int sublane = lane_id % kSubgroupSize;
    const device_mask_t subgroup_mask =
        static_cast<device_mask_t>(0xFFu) << (subgroup * kSubgroupSize);
    const device_mask_t warp_mask = static_cast<device_mask_t>(0xFFFFFFFFu);
    int candidate_sci = blockIdx.x * warps_per_block + warp_id;

    __shared__ int shared_record_counts[kWarpsPerBlock];
    __shared__ int shared_exclusion_counts[kWarpsPerBlock];
    __shared__ int shared_source_counts[kWarpsPerBlock];
    __shared__ unsigned int shared_i_local_masks[kWarpsPerBlock]
                                                [kClusteredMaxSuperClusterClusters];
    __shared__ int shared_i_atom_ids[kWarpsPerBlock]
                                    [kClusteredMaxSuperClusterClusters]
                                    [kClusteredClusterSize];
    __shared__ int shared_i_molecule_ids[kWarpsPerBlock]
                                        [kClusteredMaxSuperClusterClusters]
                                        [kClusteredClusterSize];
    __shared__ uint64_t shared_i_signatures[kWarpsPerBlock]
                                           [kClusteredMaxSuperClusterClusters];
    __shared__ float shared_i_center_x[kWarpsPerBlock]
                                      [kClusteredMaxSuperClusterClusters];
    __shared__ float shared_i_center_y[kWarpsPerBlock]
                                      [kClusteredMaxSuperClusterClusters];
    __shared__ float shared_i_center_z[kWarpsPerBlock]
                                      [kClusteredMaxSuperClusterClusters];
    __shared__ int shared_j_atom_ids[kWarpsPerBlock][kCountSubgroups]
                                    [kClusteredClusterSize];
    __shared__ int shared_j_molecule_ids[kWarpsPerBlock][kCountSubgroups]
                                        [kClusteredClusterSize];
    __shared__ uint64_t shared_j_signature[kWarpsPerBlock][kCountSubgroups];
    __shared__ float shared_j_shifted_x[kWarpsPerBlock][kCountSubgroups]
                                      [kClusteredClusterSize];
    __shared__ float shared_j_shifted_y[kWarpsPerBlock][kCountSubgroups]
                                      [kClusteredClusterSize];
    __shared__ float shared_j_shifted_z[kWarpsPerBlock][kCountSubgroups]
                                      [kClusteredClusterSize];
    __shared__ unsigned long long
        shared_exclusion_masks[kWarpsPerBlock][kCountSubgroups]
                              [kClusteredMaxSuperClusterClusters];

    while (true)
    {
    if constexpr (kDynamicWorkQueue)
    {
        int dynamic_candidate_sci = candidate_sci_numbers;
        if (lane_id == 0)
        {
            dynamic_candidate_sci = atomicAdd(dynamic_work_counter, 1);
        }
        candidate_sci =
            deviceShfl(warp_mask, dynamic_candidate_sci, 0, warpSize);
    }
    if (candidate_sci >= candidate_sci_numbers)
    {
        break;
    }

    if (lane_id == 0)
    {
        shared_record_counts[warp_id] = 0;
        shared_exclusion_counts[warp_id] = 0;
        shared_source_counts[warp_id] = 0;
    }
    __syncwarp();

    const bool has_molecule_metadata =
        cluster_molecule_signatures != NULL && cluster_molecule_ids != NULL;
    const int fixed_shift_id = candidate_sci % kClusteredShiftCount;
    const VECTOR fixed_shift_vec =
        Shift_Vector_From_Id_Probe(fixed_shift_id, cell);
    const float record_stream_cutoff_sq =
        record_stream_cutoff * record_stream_cutoff;
    const int sci_base = candidate_sci / kClusteredShiftCount;
    const int super_i = sci_supercluster_ids[sci_base];
    const int cluster_i_start = super_cluster_offsets[super_i];
    const int cluster_i_end = super_cluster_offsets[super_i + 1];
    const int active_cluster_count = cluster_i_end - cluster_i_start;
    const unsigned int active_i_lane_mask =
        active_cluster_count > 0
            ? ((1u << static_cast<unsigned int>(active_cluster_count)) - 1u)
            : 0u;

    if (lane_id < kClusteredMaxSuperClusterClusters)
    {
        const int cluster_i = cluster_i_start + lane_id;
        if (cluster_i < cluster_i_end)
        {
            const VECTOR center_i = cluster_centers[cluster_i];
            shared_i_local_masks[warp_id][lane_id] =
                cluster_local_masks[cluster_i];
            shared_i_center_x[warp_id][lane_id] = center_i.x;
            shared_i_center_y[warp_id][lane_id] = center_i.y;
            shared_i_center_z[warp_id][lane_id] = center_i.z;
            shared_i_signatures[warp_id][lane_id] =
                has_molecule_metadata ? cluster_molecule_signatures[cluster_i]
                                      : 0ull;
        }
        else
        {
            shared_i_local_masks[warp_id][lane_id] = 0u;
            shared_i_center_x[warp_id][lane_id] = 0.0f;
            shared_i_center_y[warp_id][lane_id] = 0.0f;
            shared_i_center_z[warp_id][lane_id] = 0.0f;
            shared_i_signatures[warp_id][lane_id] = 0ull;
        }
    }
    for (int atom_slot = lane_id;
         atom_slot < kClusteredMaxSuperClusterClusters *
                         kClusteredClusterSize;
         atom_slot += warpSize)
    {
        const int i_local = atom_slot / kClusteredClusterSize;
        const int atom_lane = atom_slot % kClusteredClusterSize;
        const int cluster_i = cluster_i_start + i_local;
        if (cluster_i < cluster_i_end)
        {
            const int sorted_atom_i = cluster_offsets[cluster_i] + atom_lane;
            shared_i_atom_ids[warp_id][i_local][atom_lane] =
                permutation[sorted_atom_i];
            shared_i_molecule_ids[warp_id][i_local][atom_lane] =
                has_molecule_metadata
                    ? cluster_molecule_ids[cluster_i * kClusteredClusterSize +
                                           atom_lane]
                    : -1;
        }
        else
        {
            shared_i_atom_ids[warp_id][i_local][atom_lane] = -1;
            shared_i_molecule_ids[warp_id][i_local][atom_lane] = -1;
        }
    }
    __syncwarp();

    const int leaf_begin = candidate_leaf_offsets[candidate_sci];
    const int leaf_end = candidate_leaf_offsets[candidate_sci + 1];
    const int s_max = max_leaf_cluster_span;
    const bool use_candidate_leaf_metadata =
        candidate_leaf_prev_running_max_ends != NULL;
    for (int candidate_idx = leaf_begin + subgroup; candidate_idx < leaf_end;
         candidate_idx += kCountSubgroups)
    {
        const int leaf_j = candidate_leaf_ids[candidate_idx];
        const int cluster_j_start = leaf_cluster_starts[leaf_j];
        const int cluster_j_end = leaf_cluster_ends[leaf_j];
        int prev_running_max_end = 0;
        if (use_candidate_leaf_metadata)
        {
            prev_running_max_end =
                candidate_leaf_prev_running_max_ends[candidate_idx];
        }
        else
        {
            for (int b = candidate_idx - 1; b >= leaf_begin; b -= 1)
            {
                const int b_leaf = candidate_leaf_ids[b];
                const int b_start = leaf_cluster_starts[b_leaf];
                const int b_end = leaf_cluster_ends[b_leaf];
                if (b_end > prev_running_max_end)
                {
                    prev_running_max_end = b_end;
                }
                if (b_start + s_max <= prev_running_max_end)
                {
                    break;
                }
            }
        }
        const int leaf_mask_base =
            candidate_idx * candidate_leaf_cluster_stride;
        const int deduped_cluster_j_start =
            IntMaxProbe(cluster_j_start, prev_running_max_end);
        for (int cluster_j = deduped_cluster_j_start; cluster_j < cluster_j_end;
             cluster_j += 1)
        {
            unsigned int precomputed_i_mask = 0u;
            if (sublane == 0)
            {
                precomputed_i_mask =
                    candidate_leaf_reach_masks[leaf_mask_base +
                                               (cluster_j - cluster_j_start)];
            }
            precomputed_i_mask =
                deviceShfl(subgroup_mask, precomputed_i_mask,
                           subgroup * kSubgroupSize, warpSize) &
                active_i_lane_mask;
            if (precomputed_i_mask == 0u)
            {
                continue;
            }

            unsigned int valid_mask_j = 0u;
            unsigned int local_mask_j = 0u;
            int super_j = 0;
            uint64_t signature_j = 0ull;
            VECTOR center_j = {0.0f, 0.0f, 0.0f};
            if (sublane == 0)
            {
                valid_mask_j = cluster_valid_masks[cluster_j];
                local_mask_j = cluster_local_masks[cluster_j];
                if (valid_mask_j != 0u)
                {
                    super_j = cluster_to_supercluster[cluster_j];
                    center_j = cluster_centers[cluster_j];
                    if (has_molecule_metadata)
                    {
                        signature_j = cluster_molecule_signatures[cluster_j];
                    }
                }
            }
            const int subgroup_leader = subgroup * kSubgroupSize;
            valid_mask_j =
                deviceShfl(subgroup_mask, valid_mask_j, subgroup_leader,
                           warpSize);
            local_mask_j =
                deviceShfl(subgroup_mask, local_mask_j, subgroup_leader,
                           warpSize);
            super_j =
                deviceShfl(subgroup_mask, super_j, subgroup_leader, warpSize);
            center_j.x =
                deviceShfl(subgroup_mask, center_j.x, subgroup_leader, warpSize);
            center_j.y =
                deviceShfl(subgroup_mask, center_j.y, subgroup_leader, warpSize);
            center_j.z =
                deviceShfl(subgroup_mask, center_j.z, subgroup_leader, warpSize);
            unsigned int sig_lo = static_cast<unsigned int>(signature_j);
            unsigned int sig_hi =
                static_cast<unsigned int>(signature_j >> 32);
            sig_lo =
                deviceShfl(subgroup_mask, sig_lo, subgroup_leader, warpSize);
            sig_hi =
                deviceShfl(subgroup_mask, sig_hi, subgroup_leader, warpSize);
            signature_j = (static_cast<uint64_t>(sig_hi) << 32) |
                          static_cast<uint64_t>(sig_lo);
            if (valid_mask_j == 0u)
            {
                continue;
            }
            if (local_mask_j != 0u && super_j < super_i)
            {
                continue;
            }

            if (sublane < kClusteredClusterSize)
            {
                if ((valid_mask_j & (1u << sublane)) != 0u)
                {
                    const int sorted_atom_j = cluster_offsets[cluster_j] + sublane;
                    shared_j_atom_ids[warp_id][subgroup][sublane] =
                        permutation[sorted_atom_j];
                    shared_j_molecule_ids[warp_id][subgroup][sublane] =
                        has_molecule_metadata
                            ? cluster_molecule_ids[cluster_j *
                                                       kClusteredClusterSize +
                                                   sublane]
                            : -1;
                }
                else
                {
                    shared_j_atom_ids[warp_id][subgroup][sublane] = -1;
                    shared_j_molecule_ids[warp_id][subgroup][sublane] = -1;
                }
            }
            if (sublane == 0)
            {
                shared_j_signature[warp_id][subgroup] = signature_j;
            }
            deviceSyncWarp(subgroup_mask);
            if constexpr (kCooperativePrune)
            {
                VECTOR shifted_j = {0.0f, 0.0f, 0.0f};
                const int atom_j =
                    shared_j_atom_ids[warp_id][subgroup][sublane];
                if (atom_j >= 0)
                {
                    shifted_j =
                        Shift_Clustered_Atom_Into_Sorted_XQ_Frame_Probe(
                            crd[atom_j], center_j, cell, rcell);
                }
                shared_j_shifted_x[warp_id][subgroup][sublane] = shifted_j.x;
                shared_j_shifted_y[warp_id][subgroup][sublane] = shifted_j.y;
                shared_j_shifted_z[warp_id][subgroup][sublane] = shifted_j.z;
                deviceSyncWarp(subgroup_mask);
            }

            unsigned long long lane_exclusion_mask = 0ull;
            bool has_lane_exclusion = false;
            if (sublane < active_cluster_count &&
                (precomputed_i_mask &
                 (1u << static_cast<unsigned int>(sublane))) != 0u)
            {
                const bool exclusion_candidate =
                    !has_molecule_metadata ||
                    (shared_i_signatures[warp_id][sublane] & signature_j) != 0ull;
                if (exclusion_candidate)
                {
                    const unsigned int local_mask_i =
                        shared_i_local_masks[warp_id][sublane];
                    lane_exclusion_mask =
                        Build_Exclusion_Mask_From_Cached_Atoms_Probe(
                            shared_i_atom_ids[warp_id][sublane],
                            shared_j_atom_ids[warp_id][subgroup],
                            shared_i_molecule_ids[warp_id][sublane],
                            shared_j_molecule_ids[warp_id][subgroup],
                            shared_i_signatures[warp_id][sublane],
                            shared_j_signature[warp_id][subgroup],
                            has_molecule_metadata, local_mask_i, valid_mask_j,
                            cluster_size, local_atom_numbers,
                            excluded_list_start, excluded_list,
                            excluded_numbers);
                    has_lane_exclusion = lane_exclusion_mask != 0ull;
                }
            }
            if (sublane < kClusteredMaxSuperClusterClusters)
            {
                shared_exclusion_masks[warp_id][subgroup][sublane] =
                    lane_exclusion_mask;
            }
            unsigned int split_local_imask = 0u;
            bool split_has_source_row = false;
            if constexpr (kCooperativePrune)
            {
                const unsigned int cooperative_imasks =
                    Prune_Source_Imask_With_Shift_Cooperative_Probe(
                        subgroup, sublane, subgroup_mask, precomputed_i_mask,
                        valid_mask_j, fixed_shift_vec, cell, rcell, crd,
                        shared_i_center_x[warp_id],
                        shared_i_center_y[warp_id],
                        shared_i_center_z[warp_id], shared_i_atom_ids[warp_id],
                        shared_j_shifted_x[warp_id][subgroup],
                        shared_j_shifted_y[warp_id][subgroup],
                        shared_j_shifted_z[warp_id][subgroup],
                        shared_i_local_masks[warp_id],
                        record_stream_cutoff_sq);
                if (sublane < kClusteredWarpSplitCount)
                {
                    split_local_imask =
                        (cooperative_imasks >>
                         (sublane * kClusteredClusterSize)) &
                        0xFFu;
                    split_has_source_row =
                        Clustered_Split_Has_Atoms_Probe(valid_mask_j,
                                                        sublane) &&
                        split_local_imask != 0u;
                }
            }
            else if (sublane < kClusteredWarpSplitCount)
            {
                split_local_imask =
                    Prune_Source_Imask_With_Shift_Probe(
                        sublane, precomputed_i_mask, valid_mask_j,
                        fixed_shift_vec, cell, rcell, crd,
                        shared_i_center_x[warp_id],
                        shared_i_center_y[warp_id],
                        shared_i_center_z[warp_id], center_j,
                        shared_i_atom_ids[warp_id],
                        shared_j_atom_ids[warp_id][subgroup],
                        shared_i_local_masks[warp_id],
                        record_stream_cutoff_sq);
                split_has_source_row =
                    Clustered_Split_Has_Atoms_Probe(valid_mask_j, sublane) &&
                    split_local_imask != 0u;
            }
            deviceSyncWarp(subgroup_mask);

            const unsigned int source_row_lane_mask =
                (deviceBallot(subgroup_mask, split_has_source_row) >>
                 (subgroup * kSubgroupSize)) &
                0xFFu;
            const int source_rows_for_group = __popc(source_row_lane_mask);
            const unsigned int exclusion_lane_mask =
                (deviceBallot(subgroup_mask, has_lane_exclusion) >>
                 (subgroup * kSubgroupSize)) &
                0xFFu;
            const int exclusion_count_for_group = __popc(exclusion_lane_mask);
            const int leader_sublane =
                __ffs(static_cast<int>(precomputed_i_mask)) - 1;
            int fragment_base = -1;
            if (sublane == leader_sublane && source_rows_for_group > 0)
            {
                fragment_base =
                    atomicAdd(count_source_fragment_cursor,
                              source_rows_for_group);
            }
            fragment_base =
                deviceShfl(subgroup_mask, fragment_base,
                           subgroup * kSubgroupSize + leader_sublane,
                           warpSize);
            if (split_has_source_row)
            {
                const unsigned int lower_source_rows =
                    source_row_lane_mask &
                    ((1u << static_cast<unsigned int>(sublane)) - 1u);
                const int fragment_idx =
                    fragment_base + __popc(lower_source_rows);
                if (fragment_idx < count_source_fragment_capacity)
                {
                    if constexpr (kSlimEmit)
                    {
                        LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT_SLIM*
                            fragment =
                                count_slim_source_fragments + fragment_idx;
                        fragment->sci_id = candidate_sci;
                        fragment->shift_id = fixed_shift_id;
                        fragment->supercluster_id = super_i;
                        fragment->cluster_j = cluster_j;
                        fragment->split_id = sublane;
                        fragment->imask = split_local_imask;
                        fragment->valid_mask_j = valid_mask_j;
                        fragment->local_mask_j = local_mask_j;
                        fragment->source_order = fragment_idx;
                    }
                    else
                    {
                        LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT*
                            fragment =
                                count_light_source_fragments + fragment_idx;
                        fragment->sci_id = candidate_sci;
                        fragment->shift_id = fixed_shift_id;
                        fragment->supercluster_id = super_i;
                        fragment->cluster_j = cluster_j;
                        fragment->split_id = sublane;
                        fragment->imask = split_local_imask;
                        fragment->valid_mask_j = valid_mask_j;
                        fragment->local_mask_j = local_mask_j;
                        fragment->source_order = fragment_idx;
#pragma unroll
                        for (int i_local = 0;
                             i_local < kClusteredMaxSuperClusterClusters;
                             i_local += 1)
                        {
                            fragment->exclusion_masks[i_local] =
                                shared_exclusion_masks[warp_id][subgroup]
                                                      [i_local];
                        }
                    }
                }
                else
                {
                    atomicAdd(count_source_fragment_overflow_rows, 1);
                }
            }
            if (sublane == leader_sublane)
            {
                atomicAdd(&shared_record_counts[warp_id], 1);
                if (source_rows_for_group > 0)
                {
                    atomicAdd(&shared_source_counts[warp_id],
                              source_rows_for_group);
                }
                if (exclusion_count_for_group > 0)
                {
                    atomicAdd(&shared_exclusion_counts[warp_id],
                              exclusion_count_for_group);
                }
            }
            deviceSyncWarp(subgroup_mask);
        }
    }
    __syncwarp();
    if (lane_id == 0)
    {
        const int record_count = shared_record_counts[warp_id];
        const int source_count = shared_source_counts[warp_id];
        sci_shift_flags[candidate_sci] = record_count > 0 ? 1 : 0;
        cjpacked_group_counts[candidate_sci] =
            (record_count + kClusteredMaxJGroupSize - 1) /
            kClusteredMaxJGroupSize;
        exclusion_counts[candidate_sci] = shared_exclusion_counts[warp_id];
        if (record_stream_source_counts_by_candidate != NULL)
        {
            record_stream_source_counts_by_candidate[candidate_sci] =
                source_count;
        }
        if (accumulate_record_stream_source_rows_by_candidate &&
            record_stream_source_rows != NULL && source_count > 0)
        {
            atomicAdd(record_stream_source_rows, source_count);
        }
    }
    if constexpr (!kDynamicWorkQueue)
    {
        break;
    }
    }
}

} // namespace

#endif

void Launch_Clustered_Gmxpacked_Candidate_Leaf_Probe(
    ClusteredGmxpackedCandidateLeafProbeMode mode,
    int candidate_sci_blocks, int builder_block_size, int candidate_sci_numbers,
    const int* sci_supercluster_ids, const VECTOR* super_cluster_centers,
    const VECTOR* super_cluster_sizes, const int* super_cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* leaf_all_local, LTMatrix3 cell, float cutoff,
    const VECTOR* cluster_centers, const VECTOR* cluster_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, bool central_halfshell_culling,
    bool use_morton_sfc, bool use_fast_node_overlap,
    bool use_cooperative_traversal, bool use_root_child_split,
    int onepass_capacity, int* probe_counts,
    LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_RECORD* probe_records,
    int* probe_cursor, int* probe_overflow,
    LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_STATS* probe_stats,
    const int* candidate_sci_index_map, const int* root_child_task_sci_ids,
    const int* root_child_task_nodes)
{
#ifndef USE_CPU
#define LAUNCH_CANDIDATE_LEAF_PROBE(SCREEN, EMIT)                         \
    do                                                                     \
    {                                                                      \
        if (use_root_child_split)                                          \
        {                                                                  \
            auto* kernel = use_fast_node_overlap                           \
                               ? Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup<SCREEN, false, true, false, false, true> \
                               : Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup<SCREEN, false, false, false, false, true>; \
            Launch_Device_Kernel(                                         \
                kernel, candidate_sci_blocks, builder_block_size, 0, NULL, \
                candidate_sci_numbers, sci_supercluster_ids,               \
                super_cluster_centers, super_cluster_sizes,                \
                super_cluster_offsets, leaf_cluster_starts,                \
                leaf_cluster_ends, leaf_all_local, cell, cutoff,           \
                cluster_centers, cluster_extents, cluster_valid_masks,     \
                cluster_local_masks, node_prefixes, child_offsets,         \
                parents, internal_to_leaf, candidate_shift_ids,            \
                central_halfshell_culling, use_morton_sfc,                 \
                onepass_capacity, probe_counts, NULL, NULL, NULL, NULL,    \
                candidate_sci_index_map, root_child_task_sci_ids,          \
                root_child_task_nodes);                                     \
            break;                                                         \
        }                                                                  \
        if (probe_stats != NULL)                                           \
        {                                                                  \
            auto* kernel = use_cooperative_traversal                       \
                               ? (use_fast_node_overlap                    \
                                      ? Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup<SCREEN, EMIT, true, true, true, false> \
                                      : Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup<SCREEN, EMIT, false, true, true, false>) \
                               : (use_fast_node_overlap                    \
                                      ? Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup<SCREEN, EMIT, true, false, true, false> \
                                      : Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup<SCREEN, EMIT, false, false, true, false>); \
            Launch_Device_Kernel(                                         \
                kernel, candidate_sci_blocks, builder_block_size, 0, NULL, \
                candidate_sci_numbers, sci_supercluster_ids,               \
                super_cluster_centers, super_cluster_sizes,                \
                super_cluster_offsets, leaf_cluster_starts,                \
                leaf_cluster_ends, leaf_all_local, cell, cutoff,           \
                cluster_centers, cluster_extents, cluster_valid_masks,     \
                cluster_local_masks, node_prefixes, child_offsets,         \
                parents, internal_to_leaf, candidate_shift_ids,            \
                central_halfshell_culling, use_morton_sfc,                 \
                onepass_capacity, probe_counts, probe_records,             \
                probe_cursor, probe_overflow, probe_stats,                 \
                candidate_sci_index_map, root_child_task_sci_ids,          \
                root_child_task_nodes);                                     \
        }                                                                  \
        else                                                               \
        {                                                                  \
            auto* kernel = use_cooperative_traversal                       \
                               ? (use_fast_node_overlap                    \
                                      ? Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup<SCREEN, EMIT, true, true, false, false> \
                                      : Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup<SCREEN, EMIT, false, true, false, false>) \
                               : (use_fast_node_overlap                    \
                                      ? Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup<SCREEN, EMIT, true, false, false, false> \
                                      : Probe_Candidate_Leaf_Collect_Fixed_Shift_Subgroup<SCREEN, EMIT, false, false, false, false>); \
            Launch_Device_Kernel(                                         \
                kernel, candidate_sci_blocks, builder_block_size, 0, NULL, \
                candidate_sci_numbers, sci_supercluster_ids,               \
                super_cluster_centers, super_cluster_sizes,                \
                super_cluster_offsets, leaf_cluster_starts,                \
                leaf_cluster_ends, leaf_all_local, cell, cutoff,           \
                cluster_centers, cluster_extents, cluster_valid_masks,     \
                cluster_local_masks, node_prefixes, child_offsets,         \
                parents, internal_to_leaf, candidate_shift_ids,            \
                central_halfshell_culling, use_morton_sfc,                 \
                onepass_capacity, probe_counts, probe_records,             \
                probe_cursor, probe_overflow, probe_stats,                 \
                candidate_sci_index_map, root_child_task_sci_ids,          \
                root_child_task_nodes);                                     \
        }                                                                  \
    } while (0)
    switch (mode)
    {
    case ClusteredGmxpackedCandidateLeafProbeMode::Traversal:
    {
        LAUNCH_CANDIDATE_LEAF_PROBE(false, false);
        break;
    }
    case ClusteredGmxpackedCandidateLeafProbeMode::Screen:
    {
        LAUNCH_CANDIDATE_LEAF_PROBE(true, false);
        break;
    }
    case ClusteredGmxpackedCandidateLeafProbeMode::Emit:
    {
        LAUNCH_CANDIDATE_LEAF_PROBE(true, true);
        break;
    }
    }
#undef LAUNCH_CANDIDATE_LEAF_PROBE
#else
    (void)mode;
    (void)candidate_sci_blocks;
    (void)builder_block_size;
    (void)candidate_sci_numbers;
    (void)sci_supercluster_ids;
    (void)super_cluster_centers;
    (void)super_cluster_sizes;
    (void)super_cluster_offsets;
    (void)leaf_cluster_starts;
    (void)leaf_cluster_ends;
    (void)leaf_all_local;
    (void)cell;
    (void)cutoff;
    (void)cluster_centers;
    (void)cluster_extents;
    (void)cluster_valid_masks;
    (void)cluster_local_masks;
    (void)node_prefixes;
    (void)child_offsets;
    (void)parents;
    (void)internal_to_leaf;
    (void)candidate_shift_ids;
    (void)central_halfshell_culling;
    (void)use_morton_sfc;
    (void)use_fast_node_overlap;
    (void)use_cooperative_traversal;
    (void)use_root_child_split;
    (void)onepass_capacity;
    (void)probe_counts;
    (void)probe_records;
    (void)probe_cursor;
    (void)probe_overflow;
    (void)probe_stats;
    (void)candidate_sci_index_map;
    (void)root_child_task_sci_ids;
    (void)root_child_task_nodes;
#endif
}

void Launch_Clustered_Gmxpacked_Candidate_Leaf_Root_Child_Device_Counter_Probe(
    int collect_blocks, int builder_block_size, int candidate_sci_numbers,
    const int* sci_supercluster_ids, const VECTOR* super_cluster_centers,
    const VECTOR* super_cluster_sizes, const int* super_cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* leaf_all_local, LTMatrix3 cell, float cutoff,
    const VECTOR* cluster_centers, const VECTOR* cluster_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, bool central_halfshell_culling,
    bool use_morton_sfc, bool use_fast_node_overlap, int task_capacity,
    const int* task_counter, int* task_work_cursor, int* probe_counts,
    int* task_leaf_counts, const int* root_child_task_sci_ids,
    const int* root_child_task_nodes)
{
#ifndef USE_CPU
    auto* kernel =
        use_fast_node_overlap
            ? Probe_Candidate_Leaf_Root_Child_Device_Counter_Subgroup<true>
            : Probe_Candidate_Leaf_Root_Child_Device_Counter_Subgroup<false>;
    Launch_Device_Kernel(
        kernel, collect_blocks, builder_block_size, 0, NULL,
        candidate_sci_numbers, sci_supercluster_ids, super_cluster_centers,
        super_cluster_sizes, super_cluster_offsets, leaf_cluster_starts,
        leaf_cluster_ends, leaf_all_local, cell, cutoff, cluster_centers,
        cluster_extents, cluster_valid_masks, cluster_local_masks,
        node_prefixes, child_offsets, parents, internal_to_leaf,
        candidate_shift_ids, central_halfshell_culling, use_morton_sfc,
        task_capacity, task_counter, task_work_cursor, probe_counts,
        task_leaf_counts, root_child_task_sci_ids, root_child_task_nodes);
#else
    (void)collect_blocks;
    (void)builder_block_size;
    (void)candidate_sci_numbers;
    (void)sci_supercluster_ids;
    (void)super_cluster_centers;
    (void)super_cluster_sizes;
    (void)super_cluster_offsets;
    (void)leaf_cluster_starts;
    (void)leaf_cluster_ends;
    (void)leaf_all_local;
    (void)cell;
    (void)cutoff;
    (void)cluster_centers;
    (void)cluster_extents;
    (void)cluster_valid_masks;
    (void)cluster_local_masks;
    (void)node_prefixes;
    (void)child_offsets;
    (void)parents;
    (void)internal_to_leaf;
    (void)candidate_shift_ids;
    (void)central_halfshell_culling;
    (void)use_morton_sfc;
    (void)use_fast_node_overlap;
    (void)task_capacity;
    (void)task_counter;
    (void)task_work_cursor;
    (void)probe_counts;
    (void)task_leaf_counts;
    (void)root_child_task_sci_ids;
    (void)root_child_task_nodes;
#endif
}

void Launch_Clustered_Gmxpacked_Candidate_Leaf_Root_Child_Device_Counter_Emit(
    int collect_blocks, int builder_block_size, int candidate_sci_numbers,
    const int* sci_supercluster_ids, const VECTOR* super_cluster_centers,
    const VECTOR* super_cluster_sizes, const int* super_cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* leaf_all_local, LTMatrix3 cell, float cutoff,
    const VECTOR* cluster_centers, const VECTOR* cluster_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, bool central_halfshell_culling,
    bool use_morton_sfc, bool use_fast_node_overlap, int task_capacity,
    const int* task_counter, int* task_work_cursor, int* emit_counts,
    const int* candidate_leaf_offsets, int* candidate_leaf_ids,
    const int* task_leaf_offsets, int* emit_overflow,
    const int* root_child_task_sci_ids,
    const int* root_child_task_nodes)
{
#ifndef USE_CPU
    auto* kernel =
        use_fast_node_overlap
            ? Emit_Candidate_Leaf_Root_Child_Device_Counter_Subgroup<true>
            : Emit_Candidate_Leaf_Root_Child_Device_Counter_Subgroup<false>;
    Launch_Device_Kernel(
        kernel, collect_blocks, builder_block_size, 0, NULL,
        candidate_sci_numbers, sci_supercluster_ids, super_cluster_centers,
        super_cluster_sizes, super_cluster_offsets, leaf_cluster_starts,
        leaf_cluster_ends, leaf_all_local, cell, cutoff, cluster_centers,
        cluster_extents, cluster_valid_masks, cluster_local_masks,
        node_prefixes, child_offsets, parents, internal_to_leaf,
        candidate_shift_ids, central_halfshell_culling, use_morton_sfc,
        task_capacity, task_counter, task_work_cursor, emit_counts,
        candidate_leaf_offsets, candidate_leaf_ids, task_leaf_offsets,
        emit_overflow, root_child_task_sci_ids, root_child_task_nodes);
#else
    (void)collect_blocks;
    (void)builder_block_size;
    (void)candidate_sci_numbers;
    (void)sci_supercluster_ids;
    (void)super_cluster_centers;
    (void)super_cluster_sizes;
    (void)super_cluster_offsets;
    (void)leaf_cluster_starts;
    (void)leaf_cluster_ends;
    (void)leaf_all_local;
    (void)cell;
    (void)cutoff;
    (void)cluster_centers;
    (void)cluster_extents;
    (void)cluster_valid_masks;
    (void)cluster_local_masks;
    (void)node_prefixes;
    (void)child_offsets;
    (void)parents;
    (void)internal_to_leaf;
    (void)candidate_shift_ids;
    (void)central_halfshell_culling;
    (void)use_morton_sfc;
    (void)use_fast_node_overlap;
    (void)task_capacity;
    (void)task_counter;
    (void)task_work_cursor;
    (void)emit_counts;
    (void)candidate_leaf_offsets;
    (void)candidate_leaf_ids;
    (void)task_leaf_offsets;
    (void)emit_overflow;
    (void)root_child_task_sci_ids;
    (void)root_child_task_nodes;
#endif
}

void Launch_Clustered_Gmxpacked_Candidate_Leaf_Root_Child_Task_Build(
    int task_build_blocks, int task_build_block_size, int candidate_sci_numbers,
    const int* sci_supercluster_ids, const VECTOR* super_cluster_centers,
    const VECTOR* super_cluster_sizes, const uint64_t* node_prefixes,
    const int* child_offsets, const int* candidate_shift_ids,
    bool use_morton_sfc, bool use_fast_node_overlap, int task_capacity,
    int* task_counter, int* task_overflow, int* task_sci_ids, int* task_nodes,
    int task_split_depth)
{
#ifndef USE_CPU
    auto* kernel = use_fast_node_overlap
                       ? Build_Candidate_Leaf_Root_Child_Tasks<true>
                       : Build_Candidate_Leaf_Root_Child_Tasks<false>;
    Launch_Device_Kernel(
        kernel, task_build_blocks, task_build_block_size, 0, NULL,
        candidate_sci_numbers, sci_supercluster_ids, super_cluster_centers,
        super_cluster_sizes, node_prefixes, child_offsets, candidate_shift_ids,
        use_morton_sfc, task_capacity, task_counter, task_overflow,
        task_sci_ids, task_nodes, task_split_depth);
#else
    (void)task_build_blocks;
    (void)task_build_block_size;
    (void)candidate_sci_numbers;
    (void)sci_supercluster_ids;
    (void)super_cluster_centers;
    (void)super_cluster_sizes;
    (void)node_prefixes;
    (void)child_offsets;
    (void)candidate_shift_ids;
    (void)use_morton_sfc;
    (void)use_fast_node_overlap;
    (void)task_capacity;
    (void)task_counter;
    (void)task_overflow;
    (void)task_sci_ids;
    (void)task_nodes;
    (void)task_split_depth;
#endif
}

void Launch_Clustered_Gmxpacked_Count_Fixed_Light_Probe(
    ClusteredGmxpackedCountFixedLightProbeMode mode,
    int candidate_sci_blocks, int builder_block_size,
    int candidate_sci_numbers, int cluster_size, int local_atom_numbers,
    float record_stream_cutoff, LTMatrix3 cell, LTMatrix3 rcell,
    const VECTOR* crd, const int* permutation, const int* cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* super_cluster_offsets, const int* cluster_to_supercluster,
    const int* sci_supercluster_ids, const int* candidate_leaf_offsets,
    const int* candidate_leaf_ids, int candidate_leaf_cluster_stride,
    const unsigned int* candidate_leaf_reach_masks,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const uint64_t* cluster_molecule_signatures,
    const int* cluster_molecule_ids, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers,
    int max_leaf_cluster_span, int* probe_counts,
    LJ_CLUSTERED_GMXPACKED_COUNT_EXPERIMENT_LIGHT_FRAGMENT* probe_fragments,
    int probe_fragment_capacity, int* probe_fragment_cursor,
    int* probe_fragment_overflow_rows)
{
#ifndef USE_CPU
    switch (mode)
    {
    case ClusteredGmxpackedCountFixedLightProbeMode::Traversal:
    {
        auto* kernel =
            Probe_Count_Nbnxm_Payload_Fixed_Light<false, false, false>;
        Launch_Device_Kernel(
            kernel,
            candidate_sci_blocks, builder_block_size, 0, NULL,
            candidate_sci_numbers, cluster_size, local_atom_numbers,
            record_stream_cutoff, cell, rcell, crd, permutation,
            cluster_offsets, leaf_cluster_starts, leaf_cluster_ends,
            super_cluster_offsets, cluster_to_supercluster,
            sci_supercluster_ids, candidate_leaf_offsets, candidate_leaf_ids,
            candidate_leaf_cluster_stride, candidate_leaf_reach_masks,
            cluster_valid_masks, cluster_local_masks, cluster_centers,
            cluster_molecule_signatures, cluster_molecule_ids,
            excluded_list_start, excluded_list, excluded_numbers,
            max_leaf_cluster_span, probe_counts, probe_fragments,
            probe_fragment_capacity, probe_fragment_cursor,
            probe_fragment_overflow_rows);
        break;
    }
    case ClusteredGmxpackedCountFixedLightProbeMode::SourcePrune:
    {
        auto* kernel =
            Probe_Count_Nbnxm_Payload_Fixed_Light<true, false, false>;
        Launch_Device_Kernel(
            kernel,
            candidate_sci_blocks, builder_block_size, 0, NULL,
            candidate_sci_numbers, cluster_size, local_atom_numbers,
            record_stream_cutoff, cell, rcell, crd, permutation,
            cluster_offsets, leaf_cluster_starts, leaf_cluster_ends,
            super_cluster_offsets, cluster_to_supercluster,
            sci_supercluster_ids, candidate_leaf_offsets, candidate_leaf_ids,
            candidate_leaf_cluster_stride, candidate_leaf_reach_masks,
            cluster_valid_masks, cluster_local_masks, cluster_centers,
            cluster_molecule_signatures, cluster_molecule_ids,
            excluded_list_start, excluded_list, excluded_numbers,
            max_leaf_cluster_span, probe_counts, probe_fragments,
            probe_fragment_capacity, probe_fragment_cursor,
            probe_fragment_overflow_rows);
        break;
    }
    case ClusteredGmxpackedCountFixedLightProbeMode::SourceEmit:
    {
        auto* kernel =
            Probe_Count_Nbnxm_Payload_Fixed_Light<true, true, false>;
        Launch_Device_Kernel(
            kernel,
            candidate_sci_blocks, builder_block_size, 0, NULL,
            candidate_sci_numbers, cluster_size, local_atom_numbers,
            record_stream_cutoff, cell, rcell, crd, permutation,
            cluster_offsets, leaf_cluster_starts, leaf_cluster_ends,
            super_cluster_offsets, cluster_to_supercluster,
            sci_supercluster_ids, candidate_leaf_offsets, candidate_leaf_ids,
            candidate_leaf_cluster_stride, candidate_leaf_reach_masks,
            cluster_valid_masks, cluster_local_masks, cluster_centers,
            cluster_molecule_signatures, cluster_molecule_ids,
            excluded_list_start, excluded_list, excluded_numbers,
            max_leaf_cluster_span, probe_counts, probe_fragments,
            probe_fragment_capacity, probe_fragment_cursor,
            probe_fragment_overflow_rows);
        break;
    }
    case ClusteredGmxpackedCountFixedLightProbeMode::ExclusionEmit:
    {
        auto* kernel =
            Probe_Count_Nbnxm_Payload_Fixed_Light<true, true, true>;
        Launch_Device_Kernel(
            kernel,
            candidate_sci_blocks, builder_block_size, 0, NULL,
            candidate_sci_numbers, cluster_size, local_atom_numbers,
            record_stream_cutoff, cell, rcell, crd, permutation,
            cluster_offsets, leaf_cluster_starts, leaf_cluster_ends,
            super_cluster_offsets, cluster_to_supercluster,
            sci_supercluster_ids, candidate_leaf_offsets, candidate_leaf_ids,
            candidate_leaf_cluster_stride, candidate_leaf_reach_masks,
            cluster_valid_masks, cluster_local_masks, cluster_centers,
            cluster_molecule_signatures, cluster_molecule_ids,
            excluded_list_start, excluded_list, excluded_numbers,
            max_leaf_cluster_span, probe_counts, probe_fragments,
            probe_fragment_capacity, probe_fragment_cursor,
            probe_fragment_overflow_rows);
        break;
    }
    }
#else
    (void)mode;
    (void)candidate_sci_blocks;
    (void)builder_block_size;
    (void)candidate_sci_numbers;
    (void)cluster_size;
    (void)local_atom_numbers;
    (void)record_stream_cutoff;
    (void)cell;
    (void)rcell;
    (void)crd;
    (void)permutation;
    (void)cluster_offsets;
    (void)leaf_cluster_starts;
    (void)leaf_cluster_ends;
    (void)super_cluster_offsets;
    (void)cluster_to_supercluster;
    (void)sci_supercluster_ids;
    (void)candidate_leaf_offsets;
    (void)candidate_leaf_ids;
    (void)candidate_leaf_cluster_stride;
    (void)candidate_leaf_reach_masks;
    (void)cluster_valid_masks;
    (void)cluster_local_masks;
    (void)cluster_centers;
    (void)cluster_molecule_signatures;
    (void)cluster_molecule_ids;
    (void)excluded_list_start;
    (void)excluded_list;
    (void)excluded_numbers;
    (void)max_leaf_cluster_span;
    (void)probe_counts;
    (void)probe_fragments;
    (void)probe_fragment_capacity;
    (void)probe_fragment_cursor;
    (void)probe_fragment_overflow_rows;
#endif
}

void Launch_Clustered_Gmxpacked_Count_Fixed_Light_Dedicated(
    int candidate_sci_blocks, int builder_block_size,
    int candidate_sci_numbers, int cluster_size, int local_atom_numbers,
    float record_stream_cutoff, LTMatrix3 cell, LTMatrix3 rcell,
    const VECTOR* crd, const int* permutation, const int* cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* super_cluster_offsets, const int* cluster_to_supercluster,
    const int* sci_supercluster_ids, const int* candidate_leaf_offsets,
    const int* candidate_leaf_ids, int candidate_leaf_cluster_stride,
    const int* candidate_leaf_prev_running_max_ends,
    const unsigned int* candidate_leaf_reach_masks,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const uint64_t* cluster_molecule_signatures,
    const int* cluster_molecule_ids, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers,
    int max_leaf_cluster_span, int* sci_shift_flags,
    int* cjpacked_group_counts, int* exclusion_counts,
    int* record_stream_source_rows,
    int* record_stream_source_counts_by_candidate,
    bool accumulate_record_stream_source_rows_by_candidate,
    LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT* count_light_source_fragments,
    int count_source_fragment_capacity, int* count_source_fragment_cursor,
    int* count_source_fragment_overflow_rows)
{
#ifndef USE_CPU
    auto* kernel = Dedicated_Count_Nbnxm_Payload_Fixed_Light<false, false, false>;
    Launch_Device_Kernel(
        kernel,
        candidate_sci_blocks, builder_block_size, 0, NULL,
        candidate_sci_numbers, cluster_size, local_atom_numbers,
        record_stream_cutoff, cell, rcell, crd, permutation, cluster_offsets,
        leaf_cluster_starts, leaf_cluster_ends, super_cluster_offsets,
        cluster_to_supercluster, sci_supercluster_ids, candidate_leaf_offsets,
        candidate_leaf_ids, candidate_leaf_cluster_stride,
        candidate_leaf_prev_running_max_ends,
        candidate_leaf_reach_masks, cluster_valid_masks, cluster_local_masks,
        cluster_centers, cluster_molecule_signatures, cluster_molecule_ids,
        excluded_list_start, excluded_list, excluded_numbers,
        max_leaf_cluster_span, sci_shift_flags, cjpacked_group_counts,
        exclusion_counts, record_stream_source_rows,
        record_stream_source_counts_by_candidate, NULL,
        accumulate_record_stream_source_rows_by_candidate,
        count_light_source_fragments, NULL, count_source_fragment_capacity,
        count_source_fragment_cursor, count_source_fragment_overflow_rows);
#else
    (void)candidate_sci_blocks;
    (void)builder_block_size;
    (void)candidate_sci_numbers;
    (void)cluster_size;
    (void)local_atom_numbers;
    (void)record_stream_cutoff;
    (void)cell;
    (void)rcell;
    (void)crd;
    (void)permutation;
    (void)cluster_offsets;
    (void)leaf_cluster_starts;
    (void)leaf_cluster_ends;
    (void)super_cluster_offsets;
    (void)cluster_to_supercluster;
    (void)sci_supercluster_ids;
    (void)candidate_leaf_offsets;
    (void)candidate_leaf_ids;
    (void)candidate_leaf_cluster_stride;
    (void)candidate_leaf_prev_running_max_ends;
    (void)candidate_leaf_reach_masks;
    (void)cluster_valid_masks;
    (void)cluster_local_masks;
    (void)cluster_centers;
    (void)cluster_molecule_signatures;
    (void)cluster_molecule_ids;
    (void)excluded_list_start;
    (void)excluded_list;
    (void)excluded_numbers;
    (void)max_leaf_cluster_span;
    (void)sci_shift_flags;
    (void)cjpacked_group_counts;
    (void)exclusion_counts;
    (void)record_stream_source_rows;
    (void)record_stream_source_counts_by_candidate;
    (void)accumulate_record_stream_source_rows_by_candidate;
    (void)count_light_source_fragments;
    (void)count_source_fragment_capacity;
    (void)count_source_fragment_cursor;
    (void)count_source_fragment_overflow_rows;
#endif
}

void Launch_Clustered_Gmxpacked_Count_Fixed_Light_Dedicated_Cooperative(
    int candidate_sci_blocks, int builder_block_size,
    int candidate_sci_numbers, int cluster_size, int local_atom_numbers,
    float record_stream_cutoff, LTMatrix3 cell, LTMatrix3 rcell,
    const VECTOR* crd, const int* permutation, const int* cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* super_cluster_offsets, const int* cluster_to_supercluster,
    const int* sci_supercluster_ids, const int* candidate_leaf_offsets,
    const int* candidate_leaf_ids, int candidate_leaf_cluster_stride,
    const int* candidate_leaf_prev_running_max_ends,
    const unsigned int* candidate_leaf_reach_masks,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const uint64_t* cluster_molecule_signatures,
    const int* cluster_molecule_ids, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers,
    int max_leaf_cluster_span, int* sci_shift_flags,
    int* cjpacked_group_counts, int* exclusion_counts,
    int* record_stream_source_rows,
    int* record_stream_source_counts_by_candidate,
    bool accumulate_record_stream_source_rows_by_candidate,
    LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT* count_light_source_fragments,
    int count_source_fragment_capacity, int* count_source_fragment_cursor,
    int* count_source_fragment_overflow_rows)
{
#ifndef USE_CPU
    auto* kernel =
        Dedicated_Count_Nbnxm_Payload_Fixed_Light<false, false, true>;
    Launch_Device_Kernel(
        kernel,
        candidate_sci_blocks, builder_block_size, 0, NULL,
        candidate_sci_numbers, cluster_size, local_atom_numbers,
        record_stream_cutoff, cell, rcell, crd, permutation, cluster_offsets,
        leaf_cluster_starts, leaf_cluster_ends, super_cluster_offsets,
        cluster_to_supercluster, sci_supercluster_ids, candidate_leaf_offsets,
        candidate_leaf_ids, candidate_leaf_cluster_stride,
        candidate_leaf_prev_running_max_ends,
        candidate_leaf_reach_masks, cluster_valid_masks, cluster_local_masks,
        cluster_centers, cluster_molecule_signatures, cluster_molecule_ids,
        excluded_list_start, excluded_list, excluded_numbers,
        max_leaf_cluster_span, sci_shift_flags, cjpacked_group_counts,
        exclusion_counts, record_stream_source_rows,
        record_stream_source_counts_by_candidate, NULL,
        accumulate_record_stream_source_rows_by_candidate,
        count_light_source_fragments, NULL, count_source_fragment_capacity,
        count_source_fragment_cursor, count_source_fragment_overflow_rows);
#else
    (void)candidate_sci_blocks;
    (void)builder_block_size;
    (void)candidate_sci_numbers;
    (void)cluster_size;
    (void)local_atom_numbers;
    (void)record_stream_cutoff;
    (void)cell;
    (void)rcell;
    (void)crd;
    (void)permutation;
    (void)cluster_offsets;
    (void)leaf_cluster_starts;
    (void)leaf_cluster_ends;
    (void)super_cluster_offsets;
    (void)cluster_to_supercluster;
    (void)sci_supercluster_ids;
    (void)candidate_leaf_offsets;
    (void)candidate_leaf_ids;
    (void)candidate_leaf_cluster_stride;
    (void)candidate_leaf_prev_running_max_ends;
    (void)candidate_leaf_reach_masks;
    (void)cluster_valid_masks;
    (void)cluster_local_masks;
    (void)cluster_centers;
    (void)cluster_molecule_signatures;
    (void)cluster_molecule_ids;
    (void)excluded_list_start;
    (void)excluded_list;
    (void)excluded_numbers;
    (void)max_leaf_cluster_span;
    (void)sci_shift_flags;
    (void)cjpacked_group_counts;
    (void)exclusion_counts;
    (void)record_stream_source_rows;
    (void)record_stream_source_counts_by_candidate;
    (void)accumulate_record_stream_source_rows_by_candidate;
    (void)count_light_source_fragments;
    (void)count_source_fragment_capacity;
    (void)count_source_fragment_cursor;
    (void)count_source_fragment_overflow_rows;
#endif
}

void Launch_Clustered_Gmxpacked_Count_Fixed_Light_Dedicated_Dynamic(
    int candidate_sci_blocks, int builder_block_size,
    int candidate_sci_numbers, int cluster_size, int local_atom_numbers,
    float record_stream_cutoff, LTMatrix3 cell, LTMatrix3 rcell,
    const VECTOR* crd, const int* permutation, const int* cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* super_cluster_offsets, const int* cluster_to_supercluster,
    const int* sci_supercluster_ids, const int* candidate_leaf_offsets,
    const int* candidate_leaf_ids, int candidate_leaf_cluster_stride,
    const int* candidate_leaf_prev_running_max_ends,
    const unsigned int* candidate_leaf_reach_masks,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const uint64_t* cluster_molecule_signatures,
    const int* cluster_molecule_ids, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers,
    int max_leaf_cluster_span, int* sci_shift_flags,
    int* cjpacked_group_counts, int* exclusion_counts,
    int* record_stream_source_rows,
    int* record_stream_source_counts_by_candidate,
    bool accumulate_record_stream_source_rows_by_candidate,
    LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT* count_light_source_fragments,
    int count_source_fragment_capacity, int* count_source_fragment_cursor,
    int* count_source_fragment_overflow_rows, int* dynamic_work_counter)
{
#ifndef USE_CPU
    auto* kernel = Dedicated_Count_Nbnxm_Payload_Fixed_Light<true, false, false>;
    Launch_Device_Kernel(
        kernel,
        candidate_sci_blocks, builder_block_size, 0, NULL,
        candidate_sci_numbers, cluster_size, local_atom_numbers,
        record_stream_cutoff, cell, rcell, crd, permutation, cluster_offsets,
        leaf_cluster_starts, leaf_cluster_ends, super_cluster_offsets,
        cluster_to_supercluster, sci_supercluster_ids, candidate_leaf_offsets,
        candidate_leaf_ids, candidate_leaf_cluster_stride,
        candidate_leaf_prev_running_max_ends,
        candidate_leaf_reach_masks, cluster_valid_masks, cluster_local_masks,
        cluster_centers, cluster_molecule_signatures, cluster_molecule_ids,
        excluded_list_start, excluded_list, excluded_numbers,
        max_leaf_cluster_span, sci_shift_flags, cjpacked_group_counts,
        exclusion_counts, record_stream_source_rows,
        record_stream_source_counts_by_candidate, dynamic_work_counter,
        accumulate_record_stream_source_rows_by_candidate,
        count_light_source_fragments, NULL, count_source_fragment_capacity,
        count_source_fragment_cursor, count_source_fragment_overflow_rows);
#else
    (void)candidate_sci_blocks;
    (void)builder_block_size;
    (void)candidate_sci_numbers;
    (void)cluster_size;
    (void)local_atom_numbers;
    (void)record_stream_cutoff;
    (void)cell;
    (void)rcell;
    (void)crd;
    (void)permutation;
    (void)cluster_offsets;
    (void)leaf_cluster_starts;
    (void)leaf_cluster_ends;
    (void)super_cluster_offsets;
    (void)cluster_to_supercluster;
    (void)sci_supercluster_ids;
    (void)candidate_leaf_offsets;
    (void)candidate_leaf_ids;
    (void)candidate_leaf_cluster_stride;
    (void)candidate_leaf_prev_running_max_ends;
    (void)candidate_leaf_reach_masks;
    (void)cluster_valid_masks;
    (void)cluster_local_masks;
    (void)cluster_centers;
    (void)cluster_molecule_signatures;
    (void)cluster_molecule_ids;
    (void)excluded_list_start;
    (void)excluded_list;
    (void)excluded_numbers;
    (void)max_leaf_cluster_span;
    (void)sci_shift_flags;
    (void)cjpacked_group_counts;
    (void)exclusion_counts;
    (void)record_stream_source_rows;
    (void)record_stream_source_counts_by_candidate;
    (void)accumulate_record_stream_source_rows_by_candidate;
    (void)count_light_source_fragments;
    (void)count_source_fragment_capacity;
    (void)count_source_fragment_cursor;
    (void)count_source_fragment_overflow_rows;
    (void)dynamic_work_counter;
#endif
}

void Launch_Clustered_Gmxpacked_Count_Fixed_Light_Dedicated_Slim(
    int candidate_sci_blocks, int builder_block_size,
    int candidate_sci_numbers, int cluster_size, int local_atom_numbers,
    float record_stream_cutoff, LTMatrix3 cell, LTMatrix3 rcell,
    const VECTOR* crd, const int* permutation, const int* cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* super_cluster_offsets, const int* cluster_to_supercluster,
    const int* sci_supercluster_ids, const int* candidate_leaf_offsets,
    const int* candidate_leaf_ids, int candidate_leaf_cluster_stride,
    const int* candidate_leaf_prev_running_max_ends,
    const unsigned int* candidate_leaf_reach_masks,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const uint64_t* cluster_molecule_signatures,
    const int* cluster_molecule_ids, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers,
    int max_leaf_cluster_span, int* sci_shift_flags,
    int* cjpacked_group_counts, int* exclusion_counts,
    int* record_stream_source_rows,
    int* record_stream_source_counts_by_candidate,
    bool accumulate_record_stream_source_rows_by_candidate,
    LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT_SLIM*
        count_slim_source_fragments,
    int count_source_fragment_capacity, int* count_source_fragment_cursor,
    int* count_source_fragment_overflow_rows)
{
#ifndef USE_CPU
    auto* kernel = Dedicated_Count_Nbnxm_Payload_Fixed_Light<false, true, false>;
    Launch_Device_Kernel(
        kernel,
        candidate_sci_blocks, builder_block_size, 0, NULL,
        candidate_sci_numbers, cluster_size, local_atom_numbers,
        record_stream_cutoff, cell, rcell, crd, permutation, cluster_offsets,
        leaf_cluster_starts, leaf_cluster_ends, super_cluster_offsets,
        cluster_to_supercluster, sci_supercluster_ids, candidate_leaf_offsets,
        candidate_leaf_ids, candidate_leaf_cluster_stride,
        candidate_leaf_prev_running_max_ends,
        candidate_leaf_reach_masks, cluster_valid_masks, cluster_local_masks,
        cluster_centers, cluster_molecule_signatures, cluster_molecule_ids,
        excluded_list_start, excluded_list, excluded_numbers,
        max_leaf_cluster_span, sci_shift_flags, cjpacked_group_counts,
        exclusion_counts, record_stream_source_rows,
        record_stream_source_counts_by_candidate, NULL,
        accumulate_record_stream_source_rows_by_candidate,
        NULL, count_slim_source_fragments, count_source_fragment_capacity,
        count_source_fragment_cursor, count_source_fragment_overflow_rows);
#else
    (void)candidate_sci_blocks;
    (void)builder_block_size;
    (void)candidate_sci_numbers;
    (void)cluster_size;
    (void)local_atom_numbers;
    (void)record_stream_cutoff;
    (void)cell;
    (void)rcell;
    (void)crd;
    (void)permutation;
    (void)cluster_offsets;
    (void)leaf_cluster_starts;
    (void)leaf_cluster_ends;
    (void)super_cluster_offsets;
    (void)cluster_to_supercluster;
    (void)sci_supercluster_ids;
    (void)candidate_leaf_offsets;
    (void)candidate_leaf_ids;
    (void)candidate_leaf_cluster_stride;
    (void)candidate_leaf_prev_running_max_ends;
    (void)candidate_leaf_reach_masks;
    (void)cluster_valid_masks;
    (void)cluster_local_masks;
    (void)cluster_centers;
    (void)cluster_molecule_signatures;
    (void)cluster_molecule_ids;
    (void)excluded_list_start;
    (void)excluded_list;
    (void)excluded_numbers;
    (void)max_leaf_cluster_span;
    (void)sci_shift_flags;
    (void)cjpacked_group_counts;
    (void)exclusion_counts;
    (void)record_stream_source_rows;
    (void)record_stream_source_counts_by_candidate;
    (void)accumulate_record_stream_source_rows_by_candidate;
    (void)count_slim_source_fragments;
    (void)count_source_fragment_capacity;
    (void)count_source_fragment_cursor;
    (void)count_source_fragment_overflow_rows;
#endif
}

void Launch_Clustered_Gmxpacked_Record_Stream_Source_Materialize_From_Gmxpacked(
    int sci_numbers, int builder_block_size,
    const LJ_CLUSTERED_GMXPACKED_SCI* gmxpacked_sci,
    const LJ_CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked,
    const LJ_CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    int source_capacity,
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    int* source_cursor, int* overflow_rows)
{
#ifndef USE_CPU
    Launch_Device_Kernel(Materialize_Record_Stream_Sources_From_Gmxpacked_Probe,
                         sci_numbers, builder_block_size, 0, NULL, sci_numbers,
                         gmxpacked_sci, gmxpacked_cjpacked, exclusion_entries,
                         source_capacity, sources, source_cursor,
                         overflow_rows);
#else
    (void)sci_numbers;
    (void)builder_block_size;
    (void)gmxpacked_sci;
    (void)gmxpacked_cjpacked;
    (void)exclusion_entries;
    (void)source_capacity;
    (void)sources;
    (void)source_cursor;
    (void)overflow_rows;
#endif
}

void Launch_Clustered_Gmxpacked_Record_Stream_Inner_Active_Count_Probe(
    int source_rows, int builder_block_size,
    const LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    const int* permutation, const int* cluster_offsets,
    const int* super_cluster_offsets, const unsigned int* cluster_local_masks,
    const VECTOR* cluster_centers, const VECTOR* crd, LTMatrix3 cell,
    LTMatrix3 rcell, float cutoff_sq, int* active_flags,
    unsigned int* active_imasks_by_source)
{
#ifndef USE_CPU
    const int blocks = (source_rows + builder_block_size - 1) /
                       builder_block_size;
    Launch_Device_Kernel(Count_Record_Stream_Inner_Active_Sources_Probe,
                         blocks, builder_block_size, 0, NULL, source_rows,
                         sources, permutation, cluster_offsets,
                         super_cluster_offsets, cluster_local_masks,
                         cluster_centers, crd, cell, rcell, cutoff_sq,
                         active_flags, active_imasks_by_source);
#else
    (void)source_rows;
    (void)builder_block_size;
    (void)sources;
    (void)permutation;
    (void)cluster_offsets;
    (void)super_cluster_offsets;
    (void)cluster_local_masks;
    (void)cluster_centers;
    (void)crd;
    (void)cell;
    (void)rcell;
    (void)cutoff_sq;
    (void)active_flags;
    (void)active_imasks_by_source;
#endif
}

void Launch_Clustered_Gmxpacked_Record_Stream_Inner_Active_Fill_Probe(
    int source_rows, int builder_block_size,
    const LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    const int* permutation, const int* cluster_offsets,
    const int* super_cluster_offsets, const unsigned int* cluster_local_masks,
    const VECTOR* cluster_centers, const VECTOR* crd, LTMatrix3 cell,
    LTMatrix3 rcell, float cutoff_sq, const int* active_offsets,
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* active_sources)
{
#ifndef USE_CPU
    const int blocks = (source_rows + builder_block_size - 1) /
                       builder_block_size;
    Launch_Device_Kernel(Fill_Record_Stream_Inner_Active_Sources_Probe,
                         blocks, builder_block_size, 0, NULL, source_rows,
                         sources, permutation, cluster_offsets,
                         super_cluster_offsets, cluster_local_masks,
                         cluster_centers, crd, cell, rcell, cutoff_sq,
                         active_offsets, active_sources);
#else
    (void)source_rows;
    (void)builder_block_size;
    (void)sources;
    (void)permutation;
    (void)cluster_offsets;
    (void)super_cluster_offsets;
    (void)cluster_local_masks;
    (void)cluster_centers;
    (void)crd;
    (void)cell;
    (void)rcell;
    (void)cutoff_sq;
    (void)active_offsets;
    (void)active_sources;
#endif
}

void Launch_Clustered_Gmxpacked_Record_Stream_Inner_Active_Fill_Cached_Probe(
    int source_rows, int builder_block_size,
    const LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    const unsigned int* active_imasks_by_source, const int* active_offsets,
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* active_sources)
{
#ifndef USE_CPU
    const int blocks = (source_rows + builder_block_size - 1) /
                       builder_block_size;
    Launch_Device_Kernel(Fill_Record_Stream_Inner_Active_Sources_Cached_Probe,
                         blocks, builder_block_size, 0, NULL, source_rows,
                         sources, active_imasks_by_source, active_offsets,
                         active_sources);
#else
    (void)source_rows;
    (void)builder_block_size;
    (void)sources;
    (void)active_imasks_by_source;
    (void)active_offsets;
    (void)active_sources;
#endif
}
