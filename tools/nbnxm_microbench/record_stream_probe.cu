#include "record_stream_probe.h"

namespace
{

static __host__ __device__ __forceinline__ VECTOR
Shift_Clustered_Atom_Into_Probe_Frame(const VECTOR atom_crd,
                                      const VECTOR cluster_center,
                                      const LTMatrix3 cell,
                                      const LTMatrix3 rcell)
{
    return cluster_center +
           Get_Periodic_Displacement(atom_crd, cluster_center, cell, rcell);
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
    const VECTOR pair_shift =
        Clustered_Shift_Vector_From_Id(source.shift_id, cell);
    const int j_lane_base = source.split_id * kClusteredSplitJClusterSize;
    unsigned int pruned_imask = 0u;
#pragma unroll
    for (int i_local = 0; i_local < kClusteredSuperClusterClusters;
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
            const VECTOR shifted_i = Shift_Clustered_Atom_Into_Probe_Frame(
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
                const VECTOR shifted_j = Shift_Clustered_Atom_Into_Probe_Frame(
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

} // namespace

void Launch_Clustered_Gmxpacked_Record_Stream_Source_Materialize_From_Gmxpacked(
    int sci_numbers, int builder_block_size,
    const LJ_CLUSTERED_GMXPACKED_SCI* gmxpacked_sci,
    const LJ_CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked,
    const LJ_CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    int source_capacity,
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    int* source_cursor, int* overflow_rows)
{
    Launch_Device_Kernel(Materialize_Record_Stream_Sources_From_Gmxpacked_Probe,
                         sci_numbers, builder_block_size, 0, NULL, sci_numbers,
                         gmxpacked_sci, gmxpacked_cjpacked, exclusion_entries,
                         source_capacity, sources, source_cursor,
                         overflow_rows);
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
    const int blocks = (source_rows + builder_block_size - 1) /
                       builder_block_size;
    Launch_Device_Kernel(Count_Record_Stream_Inner_Active_Sources_Probe,
                         blocks, builder_block_size, 0, NULL, source_rows,
                         sources, permutation, cluster_offsets,
                         super_cluster_offsets, cluster_local_masks,
                         cluster_centers, crd, cell, rcell, cutoff_sq,
                         active_flags, active_imasks_by_source);
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
    const int blocks = (source_rows + builder_block_size - 1) /
                       builder_block_size;
    Launch_Device_Kernel(Fill_Record_Stream_Inner_Active_Sources_Probe,
                         blocks, builder_block_size, 0, NULL, source_rows,
                         sources, permutation, cluster_offsets,
                         super_cluster_offsets, cluster_local_masks,
                         cluster_centers, crd, cell, rcell, cutoff_sq,
                         active_offsets, active_sources);
}

void Launch_Clustered_Gmxpacked_Record_Stream_Inner_Active_Fill_Cached_Probe(
    int source_rows, int builder_block_size,
    const LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    const unsigned int* active_imasks_by_source, const int* active_offsets,
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* active_sources)
{
    const int blocks = (source_rows + builder_block_size - 1) /
                       builder_block_size;
    Launch_Device_Kernel(Fill_Record_Stream_Inner_Active_Sources_Cached_Probe,
                         blocks, builder_block_size, 0, NULL, source_rows,
                         sources, active_imasks_by_source, active_offsets,
                         active_sources);
}
