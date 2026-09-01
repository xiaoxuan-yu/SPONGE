#include "candidate_builder.h"

#include <cstdint>

#ifndef USE_CPU

#include "candidate_common.cuh"

static __host__ __device__ __forceinline__ int
Encode_Clustered_Pair_Shift_Id_Replay(int sx, int sy, int sz)
{
    sx = sx < -1 ? -1 : (sx > 1 ? 1 : sx);
    sy = sy < -1 ? -1 : (sy > 1 ? 1 : sy);
    sz = sz < -1 ? -1 : (sz > 1 ? 1 : sz);
    return (sx + 1) * 9 + (sy + 1) * 3 + (sz + 1);
}

static __host__ __device__ __forceinline__ int
Determine_Clustered_Pair_Shift_Component_Replay(float dfrac,
                                                float combined_fractional_extent,
                                                int preferred_component)
{
    const int nearest = static_cast<int>(floorf(dfrac + 0.5f));
    constexpr float image_boundary_tolerance = 1.0e-6f;
    if (fabsf(fabsf(dfrac) - 0.5f) >
        combined_fractional_extent + image_boundary_tolerance)
    {
        return nearest;
    }
    const int lower = static_cast<int>(floorf(dfrac));
    const int upper = lower + 1;
    return preferred_component == lower || preferred_component == upper
               ? preferred_component
               : nearest;
}

static __host__ __device__ __forceinline__ int
Determine_Clustered_Center_Pair_Shift_Id_From_Fractional_Replay(
    const VECTOR fractional_center_i, const VECTOR fractional_center_j)
{
    const VECTOR dfrac = fractional_center_j - fractional_center_i;
    return Encode_Clustered_Pair_Shift_Id_Replay(
        static_cast<int>(floorf(dfrac.x + 0.5f)),
        static_cast<int>(floorf(dfrac.y + 0.5f)),
        static_cast<int>(floorf(dfrac.z + 0.5f)));
}

static __host__ __device__ __forceinline__ int
Determine_Clustered_Pair_Shift_Id_From_Fractional_Replay(
    const VECTOR fractional_center_i, const VECTOR fractional_center_j,
    const VECTOR fractional_extent_i, const VECTOR fractional_extent_j,
    int preferred_shift_id)
{
    const VECTOR dfrac = fractional_center_j - fractional_center_i;
    const int preferred_x = preferred_shift_id / 9 - 1;
    const int preferred_y = (preferred_shift_id % 9) / 3 - 1;
    const int preferred_z = preferred_shift_id % 3 - 1;
    return Encode_Clustered_Pair_Shift_Id_Replay(
        Determine_Clustered_Pair_Shift_Component_Replay(
            dfrac.x, fractional_extent_i.x + fractional_extent_j.x,
            preferred_x),
        Determine_Clustered_Pair_Shift_Component_Replay(
            dfrac.y, fractional_extent_i.y + fractional_extent_j.y,
            preferred_y),
        Determine_Clustered_Pair_Shift_Component_Replay(
            dfrac.z, fractional_extent_i.z + fractional_extent_j.z,
            preferred_z));
}

static __device__ __noinline__ bool
Gmxpacked_Split_Bit_Is_Covered(
    const CLUSTERED_GMXPACKED_SPLIT& current_split,
    unsigned int current_bit,
    const CLUSTERED_GMXPACKED_SPLIT& owner_split,
    unsigned int owner_bit, int split, int cluster_i, int cluster_j,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries)
{
    if ((current_split.imask & current_bit) == 0u ||
        (owner_split.imask & owner_bit) == 0u)
    {
        return false;
    }
    bool current_has_pair = false;
    for (int split_j_lane = 0;
         split_j_lane < kClusteredSplitJClusterSize; split_j_lane += 1)
    {
        const int j_lane =
            split * kClusteredSplitJClusterSize + split_j_lane;
        if (cluster_valid_masks != NULL &&
            (cluster_valid_masks[cluster_j] &
             (1u << static_cast<unsigned int>(j_lane))) == 0u)
        {
            continue;
        }
        for (int i_lane = 0; i_lane < kClusteredClusterSize; i_lane += 1)
        {
            if ((cluster_valid_masks != NULL &&
                 (cluster_valid_masks[cluster_i] &
                  (1u << static_cast<unsigned int>(i_lane))) == 0u) ||
                (cluster_local_masks != NULL &&
                 (cluster_local_masks[cluster_i] &
                  (1u << static_cast<unsigned int>(i_lane))) == 0u))
            {
                continue;
            }
            unsigned int current_pair_bits = 0xffffffffu;
            if (current_split.exclusion_index != 0 &&
                exclusion_entries != NULL)
            {
                current_pair_bits =
                    exclusion_entries[current_split.exclusion_index]
                        .pair[split_j_lane * kClusteredClusterSize + i_lane];
            }
            if ((current_pair_bits & current_bit) == 0u)
            {
                continue;
            }
            current_has_pair = true;
            unsigned int owner_pair_bits = 0xffffffffu;
            if (owner_split.exclusion_index != 0 &&
                exclusion_entries != NULL)
            {
                owner_pair_bits =
                    exclusion_entries[owner_split.exclusion_index]
                        .pair[split_j_lane * kClusteredClusterSize + i_lane];
            }
            if ((owner_pair_bits & owner_bit) == 0u)
            {
                return false;
            }
        }
    }
    return current_has_pair;
}

static __device__ __noinline__ void
Deduplicate_Gmxpacked_Current_Shift(
    int sci_numbers, int sci, int cluster_i, int cluster_j, int jm,
    int i_local, int shift_id, VECTOR fractional_center_i,
    VECTOR fractional_center_j, VECTOR fractional_extent_i,
    VECTOR fractional_extent_j,
    const CLUSTERED_GMXPACKED_CJ& current_packed,
    const CLUSTERED_GMXPACKED_SCI* sci_entries,
    const CLUSTERED_GMXPACKED_CJ* cjpacked_entries,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    unsigned int* split_0_active_mask, unsigned int* split_1_active_mask)
{
    const CLUSTERED_GMXPACKED_SCI current_sci = sci_entries[sci];
    if (shift_id == current_sci.shift_id)
    {
        return;
    }
    const unsigned int current_bit =
        1u << static_cast<unsigned int>(
            Clustered_Jm_Imask_Shift(jm) + i_local);
    const unsigned int active_i_bit =
        1u << static_cast<unsigned int>(i_local);
    for (int owner_sci_index = 0; owner_sci_index < sci_numbers;
         owner_sci_index += 1)
    {
        if (owner_sci_index == sci)
        {
            continue;
        }
        const CLUSTERED_GMXPACKED_SCI owner_sci =
            sci_entries[owner_sci_index];
        if (owner_sci.supercluster_id != current_sci.supercluster_id)
        {
            continue;
        }
        int owner_shift_id =
            Determine_Clustered_Center_Pair_Shift_Id_From_Fractional_Replay(
                fractional_center_i, fractional_center_j);
        if (owner_shift_id != owner_sci.shift_id)
        {
            owner_shift_id =
                Determine_Clustered_Pair_Shift_Id_From_Fractional_Replay(
                    fractional_center_i, fractional_center_j,
                    fractional_extent_i, fractional_extent_j,
                    owner_sci.shift_id);
        }
        if (owner_shift_id != shift_id)
        {
            continue;
        }
        const bool owner_is_fixed = owner_shift_id == owner_sci.shift_id;
        if (!owner_is_fixed && owner_sci_index > sci)
        {
            continue;
        }
        for (int owner_packed_idx = owner_sci.cjpacked_begin;
             owner_packed_idx < owner_sci.cjpacked_end;
             owner_packed_idx += 1)
        {
            const CLUSTERED_GMXPACKED_CJ owner_packed =
                cjpacked_entries[owner_packed_idx];
            for (int owner_jm = 0; owner_jm < kClusteredJGroupSize;
                 owner_jm += 1)
            {
                if (owner_packed.cj[owner_jm] != cluster_j)
                {
                    continue;
                }
                const unsigned int owner_bit =
                    1u << static_cast<unsigned int>(
                        Clustered_Jm_Imask_Shift(owner_jm) + i_local);
                for (int split = 0; split < kClusteredWarpSplitCount;
                     split += 1)
                {
                    unsigned int* active_mask =
                        split == 0 ? split_0_active_mask
                                   : split_1_active_mask;
                    if ((*active_mask &
                         active_i_bit) == 0u)
                    {
                        continue;
                    }
                    if (Gmxpacked_Split_Bit_Is_Covered(
                            current_packed.split[split], current_bit,
                            owner_packed.split[split], owner_bit, split,
                            cluster_i, cluster_j, cluster_valid_masks,
                            cluster_local_masks, exclusion_entries))
                    {
                        *active_mask &= ~active_i_bit;
                    }
                }
                if (((*split_0_active_mask | *split_1_active_mask) &
                     active_i_bit) == 0u)
                {
                    return;
                }
            }
        }
    }
}

template<bool deduplicate_periodic_images>
static __device__ __forceinline__ void Refresh_Gmxpacked_Pair_Shift_Bits_Impl(
    const int sci_numbers, const int* super_cluster_offsets,
    const VECTOR* cluster_fractional_centers,
    const VECTOR* cluster_fractional_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const CLUSTERED_GMXPACKED_SCI* gmxpacked_sci,
    const CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked,
    const CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    uint64_t* pair_shift_bits, int* sci_shift_safe_flags)
{
    const int sci = blockIdx.x;
    if (sci >= sci_numbers)
    {
        return;
    }
    if (threadIdx.x == 0 && sci_shift_safe_flags != NULL)
    {
        sci_shift_safe_flags[sci] = 1;
    }
    __syncthreads();

    const CLUSTERED_GMXPACKED_SCI sci_entry = gmxpacked_sci[sci];
    const int cluster_i_start =
        super_cluster_offsets[sci_entry.supercluster_id];
    const int cluster_i_end =
        super_cluster_offsets[sci_entry.supercluster_id + 1];
    const int active_cluster_count = cluster_i_end - cluster_i_start;
    const int packed_count = sci_entry.cjpacked_end - sci_entry.cjpacked_begin;
    const int total_records = packed_count * kClusteredJGroupSize;

    __shared__ float4
        shared_i_fractional_centers[kClusteredSuperClusterClusters];
    __shared__ float4
        shared_i_fractional_extents[kClusteredSuperClusterClusters];
    if (threadIdx.x < active_cluster_count)
    {
        const int cluster_i = cluster_i_start + threadIdx.x;
        const VECTOR fractional_center_i =
            cluster_fractional_centers[cluster_i];
        const VECTOR fractional_extent_i =
            cluster_fractional_extents[cluster_i];
        shared_i_fractional_centers[threadIdx.x] =
            {fractional_center_i.x, fractional_center_i.y,
             fractional_center_i.z, 0.0f};
        shared_i_fractional_extents[threadIdx.x] =
            {fractional_extent_i.x, fractional_extent_i.y,
             fractional_extent_i.z, 0.0f};
    }
    __syncthreads();

    for (int record = threadIdx.x; record < total_records; record += blockDim.x)
    {
        const int local_packed = record / kClusteredJGroupSize;
        const int jm = record % kClusteredJGroupSize;
        const int packed_idx = sci_entry.cjpacked_begin + local_packed;
        const CLUSTERED_GMXPACKED_CJ packed = gmxpacked_cjpacked[packed_idx];
        const int cluster_j = packed.cj[jm];

        uint64_t shift_bits = 0ull;
        unsigned int split_0_active_mask = 0u;
        unsigned int split_1_active_mask = 0u;
        if (cluster_j >= 0)
        {
            split_0_active_mask =
                (packed.split[0].imask >>
                 Clustered_Jm_Imask_Shift(jm)) &
                ((1u << kClusteredSuperClusterClusters) - 1u);
            split_1_active_mask =
                (packed.split[1].imask >>
                 Clustered_Jm_Imask_Shift(jm)) &
                ((1u << kClusteredSuperClusterClusters) - 1u);
            const unsigned int combined_imask =
                split_0_active_mask | split_1_active_mask;
            const VECTOR fractional_center_j =
                cluster_fractional_centers[cluster_j];
            VECTOR fractional_extent_j = {0.0f, 0.0f, 0.0f};
            bool fractional_extent_j_ready = false;
            for (int i_local = 0; i_local < active_cluster_count; i_local += 1)
            {
                int shift_id = kClusteredCentralShiftId;
                if ((combined_imask &
                     (1u << static_cast<unsigned int>(i_local))) != 0u)
                {
                    const float4 cached_fractional_center_i =
                        shared_i_fractional_centers[i_local];
                    const float4 cached_fractional_extent_i =
                        shared_i_fractional_extents[i_local];
                    const VECTOR fractional_center_i = {
                        cached_fractional_center_i.x,
                        cached_fractional_center_i.y,
                        cached_fractional_center_i.z};
                    shift_id =
                        Determine_Clustered_Center_Pair_Shift_Id_From_Fractional_Replay(
                            fractional_center_i, fractional_center_j);
                    if (shift_id != sci_entry.shift_id)
                    {
                        if (!fractional_extent_j_ready)
                        {
                            fractional_extent_j =
                                cluster_fractional_extents[cluster_j];
                            fractional_extent_j_ready = true;
                        }
                        shift_id =
                            Determine_Clustered_Pair_Shift_Id_From_Fractional_Replay(
                                fractional_center_i, fractional_center_j,
                                {cached_fractional_extent_i.x,
                                 cached_fractional_extent_i.y,
                                 cached_fractional_extent_i.z},
                                fractional_extent_j, sci_entry.shift_id);
                    }
                    if (shift_id != sci_entry.shift_id)
                    {
                        atomicExch(sci_shift_safe_flags + sci, 0);
                    }
                    if constexpr (deduplicate_periodic_images)
                    {
                        if (shift_id != sci_entry.shift_id)
                        {
                            const int cluster_i = cluster_i_start + i_local;
                            Deduplicate_Gmxpacked_Current_Shift(
                                sci_numbers, sci, cluster_i, cluster_j, jm,
                                i_local, shift_id, fractional_center_i,
                                fractional_center_j,
                                {cached_fractional_extent_i.x,
                                 cached_fractional_extent_i.y,
                                 cached_fractional_extent_i.z},
                                fractional_extent_j, packed, gmxpacked_sci,
                                gmxpacked_cjpacked, cluster_valid_masks,
                                cluster_local_masks, exclusion_entries,
                                &split_0_active_mask,
                                &split_1_active_mask);
                        }
                    }
                }
                Clustered_Set_Pair_Shift_Id(&shift_bits, i_local, shift_id);
            }
        }
        Clustered_Set_Pair_Active_I_Masks(
            &shift_bits, split_0_active_mask, split_1_active_mask);
        pair_shift_bits[packed_idx * kClusteredJGroupSize + jm] = shift_bits;
    }
}

__global__ void Refresh_Gmxpacked_Pair_Shift_Bits(
    const int sci_numbers, const int* super_cluster_offsets,
    const VECTOR* cluster_fractional_centers,
    const VECTOR* cluster_fractional_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const CLUSTERED_GMXPACKED_SCI* gmxpacked_sci,
    const CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked,
    const CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    uint64_t* pair_shift_bits, int* sci_shift_safe_flags)
{
    Refresh_Gmxpacked_Pair_Shift_Bits_Impl<true>(
        sci_numbers, super_cluster_offsets, cluster_fractional_centers,
        cluster_fractional_extents, cluster_valid_masks, cluster_local_masks,
        gmxpacked_sci, gmxpacked_cjpacked, exclusion_entries, pair_shift_bits,
        sci_shift_safe_flags);
}

__global__ void Refresh_Gmxpacked_Pair_Shift_Bits_Unique_Image(
    const int sci_numbers, const int* super_cluster_offsets,
    const VECTOR* cluster_fractional_centers,
    const VECTOR* cluster_fractional_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const CLUSTERED_GMXPACKED_SCI* gmxpacked_sci,
    const CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked,
    const CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    uint64_t* pair_shift_bits, int* sci_shift_safe_flags)
{
    Refresh_Gmxpacked_Pair_Shift_Bits_Impl<false>(
        sci_numbers, super_cluster_offsets, cluster_fractional_centers,
        cluster_fractional_extents, cluster_valid_masks, cluster_local_masks,
        gmxpacked_sci, gmxpacked_cjpacked, exclusion_entries, pair_shift_bits,
        sci_shift_safe_flags);
}

namespace
{


template <bool kFastNodeOverlap>
static __global__ void Count_Candidate_Leaf_Root_Child_Device_Counter_Subgroup(
    const int candidate_sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    const int* super_cluster_offsets, const int* leaf_cluster_starts,
    const int* leaf_cluster_ends, const LTMatrix3 cell, const float cutoff,
    const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, const int task_capacity,
    const int* task_counter, int* task_work_cursor, int* counts,
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
            target_center + Candidate_Shift_Fractional_From_Id(candidate_shift_id);
        const VECTOR shift_vec =
            Candidate_Shift_Vector_From_Id(candidate_shift_id, cell);
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
                return Candidate_Node_Overlaps_Preshifted_Box(
                    node_prefixes[node], shifted_target_center, target_size);
            }
            else
            {
                return Candidate_Node_Overlaps_Shifted_Box(
                    node_prefixes[node], target_center, target_size,
                    candidate_shift_id);
            }
        };

        auto endpoint_leaf = [&](int leaf_j)
        {
            if (leaf_j < 0)
            {
                return;
            }
            if (!Candidate_Leaf_Has_Fixed_Shift_Overlap_Subgroup(
                    cluster_i_start, leaf_cluster_starts[leaf_j],
                    leaf_cluster_ends[leaf_j], candidate_shift_id, cutoff_sq,
                    shift_vec, cluster_valid_masks, cluster_local_masks,
                    cluster_centers,
                    cluster_extents, cluster_i, lane_i_valid, center_i,
                    extent_i, subgroup_mask))
            {
                return;
            }
            if (sublane == 0)
            {
                count += 1;
                running_max_end =
                    Candidate_Int_Max(running_max_end, leaf_cluster_ends[leaf_j]);
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
        if (sublane == 0 && counts != NULL && count > 0)
        {
            atomicAdd(&counts[sci], count);
        }
    }
}

template <bool kFastNodeOverlap>
static __global__ void Fused_Candidate_Leaf_Root_Child_Device_Counter_Subgroup(
    const int candidate_sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    const int* super_cluster_offsets, const int* leaf_cluster_starts,
    const int* leaf_cluster_ends, const LTMatrix3 cell, const float cutoff,
    const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, const int task_capacity,
    const int* task_counter, int* task_work_cursor, int* counts,
    int* task_leaf_counts, const int fused_record_capacity,
    int* fused_record_cursor, int* fused_record_overflow, int* fused_task_ids,
    int* fused_leaf_ranks, int* fused_leaf_ids,
    const int* root_child_task_sci_ids, const int* root_child_task_nodes)
{
    const int group_id = threadIdx.x / kFixedShiftCandidateLeafSubgroupSize;
    const int sublane = threadIdx.x % kFixedShiftCandidateLeafSubgroupSize;
    const int lane_id = threadIdx.x & (kClusteredBuilderWarpSize - 1);
    const int subgroup_lane_base = lane_id - sublane;
    const device_mask_t subgroup_mask =
        (((device_mask_t)1 << kFixedShiftCandidateLeafSubgroupSize) - 1u)
        << subgroup_lane_base;
    (void)group_id;

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
            target_center + Candidate_Shift_Fractional_From_Id(candidate_shift_id);
        const VECTOR shift_vec =
            Candidate_Shift_Vector_From_Id(candidate_shift_id, cell);
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
                return Candidate_Node_Overlaps_Preshifted_Box(
                    node_prefixes[node], shifted_target_center, target_size);
            }
            else
            {
                return Candidate_Node_Overlaps_Shifted_Box(
                    node_prefixes[node], target_center, target_size,
                    candidate_shift_id);
            }
        };

        auto endpoint_leaf = [&](int leaf_j)
        {
            if (leaf_j < 0)
            {
                return;
            }
            if (!Candidate_Leaf_Has_Fixed_Shift_Overlap_Subgroup(
                    cluster_i_start, leaf_cluster_starts[leaf_j],
                    leaf_cluster_ends[leaf_j], candidate_shift_id, cutoff_sq,
                    shift_vec, cluster_valid_masks, cluster_local_masks,
                    cluster_centers,
                    cluster_extents, cluster_i, lane_i_valid, center_i,
                    extent_i, subgroup_mask))
            {
                return;
            }
            if (sublane == 0)
            {
                const int rank = count;
                count += 1;
                running_max_end =
                    Candidate_Int_Max(running_max_end, leaf_cluster_ends[leaf_j]);
                const int record_idx = atomicAdd(fused_record_cursor, 1);
                if (record_idx < fused_record_capacity)
                {
                    fused_task_ids[record_idx] = logical_item;
                    fused_leaf_ranks[record_idx] = rank;
                    fused_leaf_ids[record_idx] = leaf_j;
                }
                else if (fused_record_overflow != NULL)
                {
                    atomicAdd(fused_record_overflow, 1);
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
        if (sublane == 0 && task_leaf_counts != NULL)
        {
            task_leaf_counts[logical_item] = count;
        }
        if (sublane == 0 && counts != NULL && count > 0)
        {
            atomicAdd(&counts[sci], count);
        }
        (void)running_max_end;
    }
}

template <bool kFastNodeOverlap>
static __global__ void Build_Candidate_Leaf_Root_Child_Tasks(
    const int candidate_sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    const uint64_t* node_prefixes, const int* child_offsets,
    const int* candidate_shift_ids, const int task_capacity,
    int* task_counter, int* task_overflow,
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
        target_center + Candidate_Shift_Fractional_From_Id(candidate_shift_id);
    auto overlaps = [&](int node)
    {
        if constexpr (kFastNodeOverlap)
        {
            return Candidate_Node_Overlaps_Preshifted_Box(
                node_prefixes[node], shifted_target_center, target_size);
        }
        else
        {
            return Candidate_Node_Overlaps_Shifted_Box(
                node_prefixes[node], target_center, target_size,
                candidate_shift_id);
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
    const int* leaf_cluster_ends, const LTMatrix3 cell, const float cutoff,
    const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, const int task_capacity,
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
            target_center + Candidate_Shift_Fractional_From_Id(candidate_shift_id);
        const VECTOR shift_vec =
            Candidate_Shift_Vector_From_Id(candidate_shift_id, cell);
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
                return Candidate_Node_Overlaps_Preshifted_Box(
                    node_prefixes[node], shifted_target_center, target_size);
            }
            else
            {
                return Candidate_Node_Overlaps_Shifted_Box(
                    node_prefixes[node], target_center, target_size,
                    candidate_shift_id);
            }
        };

        auto endpoint_leaf = [&](int leaf_j)
        {
            if (leaf_j < 0)
            {
                return;
            }
            if (!Candidate_Leaf_Has_Fixed_Shift_Overlap_Subgroup(
                    cluster_i_start, leaf_cluster_starts[leaf_j],
                    leaf_cluster_ends[leaf_j], candidate_shift_id, cutoff_sq,
                    shift_vec, cluster_valid_masks, cluster_local_masks,
                    cluster_centers,
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

template <bool kCooperativePrune>
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
    const bool accumulate_record_stream_source_rows_by_candidate,
    CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT* count_light_source_fragments,
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

    if (candidate_sci >= candidate_sci_numbers)
    {
        return;
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
        Candidate_Shift_Vector_From_Id(fixed_shift_id, cell);
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
            Candidate_Int_Max(cluster_j_start, prev_running_max_end);
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
            if (Candidate_Valid_Lanes_Are_All_Local(valid_mask_j, local_mask_j) &&
                super_j < super_i)
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
                        Candidate_Shift_Atom_Into_Sorted_XQ_Frame(
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
                        Candidate_Build_Exclusion_Mask_From_Cached_Atoms(
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
                    Candidate_Prune_Source_Imask_With_Shift_Cooperative(
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
                        Candidate_Split_Has_Atoms(valid_mask_j,
                                                        sublane) &&
                        split_local_imask != 0u;
                }
            }
            else if (sublane < kClusteredWarpSplitCount)
            {
                split_local_imask =
                    Candidate_Prune_Source_Imask_With_Shift(
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
                    Candidate_Split_Has_Atoms(valid_mask_j, sublane) &&
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
                    CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT* fragment =
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


void Launch_Clustered_Gmxpacked_Candidate_Leaf_Root_Child_Device_Counter_Count(
    int collect_blocks, int builder_block_size, int candidate_sci_numbers,
    const int* sci_supercluster_ids, const VECTOR* super_cluster_centers,
    const VECTOR* super_cluster_sizes, const int* super_cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    LTMatrix3 cell, float cutoff,
    const VECTOR* cluster_centers, const VECTOR* cluster_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, bool use_fast_node_overlap,
    int task_capacity,
    const int* task_counter, int* task_work_cursor, int* counts,
    int* task_leaf_counts, const int* root_child_task_sci_ids,
    const int* root_child_task_nodes)
{
#ifndef USE_CPU
    auto* kernel =
        use_fast_node_overlap
            ? Count_Candidate_Leaf_Root_Child_Device_Counter_Subgroup<true>
            : Count_Candidate_Leaf_Root_Child_Device_Counter_Subgroup<false>;
    Launch_Device_Kernel(
        kernel, collect_blocks, builder_block_size, 0, NULL,
        candidate_sci_numbers, sci_supercluster_ids, super_cluster_centers,
        super_cluster_sizes, super_cluster_offsets, leaf_cluster_starts,
        leaf_cluster_ends, cell, cutoff, cluster_centers,
        cluster_extents, cluster_valid_masks, cluster_local_masks,
        node_prefixes, child_offsets, parents, internal_to_leaf,
        candidate_shift_ids, task_capacity, task_counter, task_work_cursor,
        counts,
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
    (void)use_fast_node_overlap;
    (void)task_capacity;
    (void)task_counter;
    (void)task_work_cursor;
    (void)counts;
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
    LTMatrix3 cell, float cutoff,
    const VECTOR* cluster_centers, const VECTOR* cluster_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, bool use_fast_node_overlap,
    int task_capacity,
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
        leaf_cluster_ends, cell, cutoff, cluster_centers,
        cluster_extents, cluster_valid_masks, cluster_local_masks,
        node_prefixes, child_offsets, parents, internal_to_leaf,
        candidate_shift_ids, task_capacity, task_counter, task_work_cursor,
        emit_counts,
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
    bool use_fast_node_overlap, int task_capacity,
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
        task_capacity, task_counter, task_overflow,
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
    (void)use_fast_node_overlap;
    (void)task_capacity;
    (void)task_counter;
    (void)task_overflow;
    (void)task_sci_ids;
    (void)task_nodes;
    (void)task_split_depth;
#endif
}

void Launch_Clustered_Gmxpacked_Candidate_Leaf_Root_Child_Device_Counter_Fused(
    int collect_blocks, int builder_block_size, int candidate_sci_numbers,
    const int* sci_supercluster_ids, const VECTOR* super_cluster_centers,
    const VECTOR* super_cluster_sizes, const int* super_cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    LTMatrix3 cell, float cutoff,
    const VECTOR* cluster_centers, const VECTOR* cluster_extents,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const uint64_t* node_prefixes,
    const int* child_offsets, const int* parents, const int* internal_to_leaf,
    const int* candidate_shift_ids, bool use_fast_node_overlap,
    int task_capacity,
    const int* task_counter, int* task_work_cursor, int* counts,
    int* task_leaf_counts, int fused_record_capacity, int* fused_record_cursor,
    int* fused_record_overflow, int* fused_task_ids, int* fused_leaf_ranks,
    int* fused_leaf_ids, const int* root_child_task_sci_ids,
    const int* root_child_task_nodes)
{
#ifndef USE_CPU
    auto* kernel =
        use_fast_node_overlap
            ? Fused_Candidate_Leaf_Root_Child_Device_Counter_Subgroup<true>
            : Fused_Candidate_Leaf_Root_Child_Device_Counter_Subgroup<false>;
    Launch_Device_Kernel(
        kernel, collect_blocks, builder_block_size, 0, NULL,
        candidate_sci_numbers, sci_supercluster_ids, super_cluster_centers,
        super_cluster_sizes, super_cluster_offsets, leaf_cluster_starts,
        leaf_cluster_ends, cell, cutoff, cluster_centers,
        cluster_extents, cluster_valid_masks, cluster_local_masks,
        node_prefixes, child_offsets, parents, internal_to_leaf,
        candidate_shift_ids, task_capacity, task_counter, task_work_cursor,
        counts,
        task_leaf_counts, fused_record_capacity, fused_record_cursor,
        fused_record_overflow, fused_task_ids, fused_leaf_ranks, fused_leaf_ids,
        root_child_task_sci_ids, root_child_task_nodes);
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
    (void)use_fast_node_overlap;
    (void)task_capacity;
    (void)task_counter;
    (void)task_work_cursor;
    (void)counts;
    (void)task_leaf_counts;
    (void)fused_record_capacity;
    (void)fused_record_cursor;
    (void)fused_record_overflow;
    (void)fused_task_ids;
    (void)fused_leaf_ranks;
    (void)fused_leaf_ids;
    (void)root_child_task_sci_ids;
    (void)root_child_task_nodes;
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
    CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT* count_light_source_fragments,
    int count_source_fragment_capacity, int* count_source_fragment_cursor,
    int* count_source_fragment_overflow_rows)
{
#ifndef USE_CPU
    auto* kernel = Dedicated_Count_Nbnxm_Payload_Fixed_Light<true>;
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
