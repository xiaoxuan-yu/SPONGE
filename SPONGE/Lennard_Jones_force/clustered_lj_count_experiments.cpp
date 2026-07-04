#include "clustered_lj_count_experiments.h"

#include <cstdint>

#ifndef USE_CPU

namespace
{

constexpr int kClusteredBuilderBlockSize = 128;
constexpr int kClusteredBuilderWarpSize = 32;
constexpr int kClusteredMaxSuperClusterClusters =
    kClusteredSuperClusterClusters;
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

static __global__ void Dedicated_Count_Nbnxm_Payload_Fixed_Light(
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
    const int max_leaf_cluster_span, int* sci_shift_flags,
    int* cjpacked_group_counts, int* exclusion_counts,
    int* record_stream_source_rows,
    int* record_stream_source_counts_by_candidate,
    const bool accumulate_record_stream_source_rows_by_candidate,
    LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT* count_light_source_fragments,
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
    const int candidate_sci = blockIdx.x * warps_per_block + warp_id;
    if (candidate_sci >= candidate_sci_numbers)
    {
        return;
    }

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
    __shared__ unsigned long long
        shared_exclusion_masks[kWarpsPerBlock][kCountSubgroups]
                              [kClusteredMaxSuperClusterClusters];

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
            if (sublane < kClusteredWarpSplitCount)
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
                    LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT* fragment =
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
                            shared_exclusion_masks[warp_id][subgroup][i_local];
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
}

} // namespace

#endif

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
    Launch_Device_Kernel(
        Dedicated_Count_Nbnxm_Payload_Fixed_Light,
        candidate_sci_blocks, builder_block_size, 0, NULL,
        candidate_sci_numbers, cluster_size, local_atom_numbers,
        record_stream_cutoff, cell, rcell, crd, permutation, cluster_offsets,
        leaf_cluster_starts, leaf_cluster_ends, super_cluster_offsets,
        cluster_to_supercluster, sci_supercluster_ids, candidate_leaf_offsets,
        candidate_leaf_ids, candidate_leaf_cluster_stride,
        candidate_leaf_reach_masks, cluster_valid_masks, cluster_local_masks,
        cluster_centers, cluster_molecule_signatures, cluster_molecule_ids,
        excluded_list_start, excluded_list, excluded_numbers,
        max_leaf_cluster_span, sci_shift_flags, cjpacked_group_counts,
        exclusion_counts, record_stream_source_rows,
        record_stream_source_counts_by_candidate,
        accumulate_record_stream_source_rows_by_candidate,
        count_light_source_fragments, count_source_fragment_capacity,
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
