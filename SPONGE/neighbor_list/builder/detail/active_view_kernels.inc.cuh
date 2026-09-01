#ifndef USE_CPU
static __device__ __forceinline__ unsigned int
Prune_Gmxpacked_Record_Stream_Source_Imask_From_Layout(
    const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE& source,
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

    const int cluster_i_start = super_cluster_offsets[source.supercluster_id];
    const int cluster_j = source.cluster_j;
    const VECTOR center_j = cluster_centers[cluster_j];
    const VECTOR pair_shift = Shift_Vector_From_Id(source.shift_id, cell);
    const int j_lane_base = source.split_id * kClusteredSplitJClusterSize;
    unsigned int pruned_imask = 0u;
#pragma unroll
    for (int i_local = 0; i_local < kClusteredMaxSuperClusterClusters;
         i_local += 1)
    {
        const unsigned int i_local_bit = 1u
                                         << static_cast<unsigned int>(i_local);
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
            const VECTOR shifted_i = Shift_Clustered_Atom_Into_Sorted_XQ_Frame(
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
                    Shift_Clustered_Atom_Into_Sorted_XQ_Frame(
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

static __device__ __forceinline__ float
Gmxpacked_Record_Stream_Source_Atom_Anchor_Displacement(
    const int atom_id, const VECTOR* crd, const VECTOR* anchor_crd,
    const LTMatrix3 cell, const LTMatrix3 rcell)
{
    if (atom_id < 0 || crd == NULL || anchor_crd == NULL)
    {
        return INFINITY;
    }
    const VECTOR dr = Get_Periodic_Displacement(
        crd[atom_id], anchor_crd[atom_id], cell, rcell);
    const float dr2 = dr * dr;
    return isfinite(dr2) && dr2 >= 0.0f ? sqrtf(dr2) : INFINITY;
}

static __device__ __forceinline__ bool
Gmxpacked_Record_Stream_Source_Active_Anchor_Dirty(
    const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE& source,
    const int* permutation, const int* cluster_offsets,
    const int* super_cluster_offsets, const unsigned int* cluster_local_masks,
    const VECTOR* crd, const VECTOR* active_anchor_crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float reuse_margin)
{
    if (active_anchor_crd == NULL)
    {
        return false;
    }
    if (source.imask == 0u || permutation == NULL || cluster_offsets == NULL ||
        super_cluster_offsets == NULL || cluster_local_masks == NULL ||
        crd == NULL || reuse_margin <= 0.0f)
    {
        return true;
    }

    const int cluster_i_start = super_cluster_offsets[source.supercluster_id];
    const int cluster_j = source.cluster_j;
    const int j_lane_base = source.split_id * kClusteredSplitJClusterSize;
    float max_i_displacement = 0.0f;
    float max_j_displacement = 0.0f;

#pragma unroll
    for (int i_local = 0; i_local < kClusteredMaxSuperClusterClusters;
         i_local += 1)
    {
        const unsigned int i_local_bit = 1u
                                         << static_cast<unsigned int>(i_local);
        if ((source.imask & i_local_bit) == 0u)
        {
            continue;
        }
        const int cluster_i = cluster_i_start + i_local;
        const unsigned int local_mask_i = cluster_local_masks[cluster_i];
#pragma unroll
        for (int i_lane = 0; i_lane < kClusteredClusterSize; i_lane += 1)
        {
            if ((local_mask_i & (1u << static_cast<unsigned int>(i_lane))) ==
                0u)
            {
                continue;
            }
            const int atom_i = permutation[cluster_offsets[cluster_i] + i_lane];
            max_i_displacement =
                fmaxf(max_i_displacement,
                      Gmxpacked_Record_Stream_Source_Atom_Anchor_Displacement(
                          atom_i, crd, active_anchor_crd, cell, rcell));
        }
    }

#pragma unroll
    for (int split_j_lane = 0; split_j_lane < kClusteredSplitJClusterSize;
         split_j_lane += 1)
    {
        const int j_lane = j_lane_base + split_j_lane;
        if ((source.valid_mask_j & (1u << static_cast<unsigned int>(j_lane))) ==
            0u)
        {
            continue;
        }
        const int atom_j = permutation[cluster_offsets[cluster_j] + j_lane];
        max_j_displacement =
            fmaxf(max_j_displacement,
                  Gmxpacked_Record_Stream_Source_Atom_Anchor_Displacement(
                      atom_j, crd, active_anchor_crd, cell, rcell));
    }

    return max_i_displacement + max_j_displacement > reuse_margin + 1e-6f;
}

static __global__ void Count_Gmxpacked_Record_Stream_Inner_Active_Sources(
    const int source_rows,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    const int* permutation, const int* cluster_offsets,
    const int* super_cluster_offsets, const unsigned int* cluster_local_masks,
    const VECTOR* cluster_centers, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float cutoff_sq, int* active_flags,
    unsigned int* active_imasks_by_source)
{
    SIMPLE_DEVICE_FOR(source_idx, source_rows)
    {
        const unsigned int pruned_imask =
            Prune_Gmxpacked_Record_Stream_Source_Imask_From_Layout(
                sources[source_idx], permutation, cluster_offsets,
                super_cluster_offsets, cluster_local_masks, cluster_centers,
                crd, cell, rcell, cutoff_sq);
        active_flags[source_idx] = pruned_imask != 0u ? 1 : 0;
        if (active_imasks_by_source != NULL)
        {
            active_imasks_by_source[source_idx] = pruned_imask;
        }
    }
}

static __global__ void Count_Gmxpacked_Dirty_Source_Rows_By_Candidate(
    const int candidate_sci_numbers, const int* dirty_candidate_sci,
    const int* source_offsets_by_candidate, int* dirty_source_counts)
{
    SIMPLE_DEVICE_FOR(candidate_sci, candidate_sci_numbers)
    {
        int source_rows = 0;
        if (dirty_candidate_sci != NULL &&
            dirty_candidate_sci[candidate_sci] != 0)
        {
            source_rows = source_offsets_by_candidate[candidate_sci + 1] -
                          source_offsets_by_candidate[candidate_sci];
            source_rows = source_rows > 0 ? source_rows : 0;
        }
        dirty_source_counts[candidate_sci] = source_rows;
    }
}

static __global__ void Fill_Gmxpacked_Dirty_Source_Indices_By_Candidate(
    const int candidate_sci_numbers, const int* dirty_candidate_sci,
    const int* source_offsets_by_candidate, const int* dirty_source_offsets,
    int* dirty_source_indices)
{
    SIMPLE_DEVICE_FOR(candidate_sci, candidate_sci_numbers)
    {
        if (dirty_candidate_sci == NULL ||
            dirty_candidate_sci[candidate_sci] == 0)
        {
            return;
        }
        const int src_begin = source_offsets_by_candidate[candidate_sci];
        const int src_end = source_offsets_by_candidate[candidate_sci + 1];
        const int dst_begin = dirty_source_offsets[candidate_sci];
        for (int src_idx = src_begin; src_idx < src_end; src_idx += 1)
        {
            dirty_source_indices[dst_begin + (src_idx - src_begin)] = src_idx;
        }
    }
}

static __global__ void Fill_Gmxpacked_Record_Stream_Inner_Active_Sources(
    const int source_rows,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    const int* permutation, const int* cluster_offsets,
    const int* super_cluster_offsets, const unsigned int* cluster_local_masks,
    const VECTOR* cluster_centers, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float cutoff_sq, const int* active_offsets,
    CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* active_sources)
{
    SIMPLE_DEVICE_FOR(source_idx, source_rows)
    {
        const unsigned int pruned_imask =
            Prune_Gmxpacked_Record_Stream_Source_Imask_From_Layout(
                sources[source_idx], permutation, cluster_offsets,
                super_cluster_offsets, cluster_local_masks, cluster_centers,
                crd, cell, rcell, cutoff_sq);
        if (pruned_imask == 0u)
        {
            return;
        }
        const int write_idx = active_offsets[source_idx];
        CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE source =
            sources[source_idx];
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
}

static __global__ void Refresh_Gmxpacked_Record_Stream_Active_View_Flags(
    const int source_rows,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* outer_sources,
    const int candidate_sci_numbers, const int* dirty_candidate_sci,
    const int* permutation, const int* cluster_offsets,
    const int* super_cluster_offsets, const unsigned int* cluster_local_masks,
    const VECTOR* cluster_centers, const VECTOR* crd,
    const VECTOR* active_anchor_crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float cutoff_sq, const float reuse_margin,
    unsigned int* active_imasks_by_source, int* active_flags,
    int* fallback_flag, int* changed_active_source_rows, int union_active_imask)
{
    SIMPLE_DEVICE_FOR(source_idx, source_rows)
    {
        const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE& source =
            outer_sources[source_idx];
        const int candidate_sci = source.sci_id;
        if (candidate_sci < 0 || candidate_sci >= candidate_sci_numbers)
        {
            if (fallback_flag != NULL)
            {
                atomicExch(fallback_flag, 1);
            }
            active_flags[source_idx] = 0;
            return;
        }
        const bool dirty = dirty_candidate_sci == NULL ||
                           dirty_candidate_sci[candidate_sci] != 0 ||
                           Gmxpacked_Record_Stream_Source_Active_Anchor_Dirty(
                               source, permutation, cluster_offsets,
                               super_cluster_offsets, cluster_local_masks, crd,
                               active_anchor_crd, cell, rcell, reuse_margin);
        unsigned int active_imask = active_imasks_by_source != NULL
                                        ? active_imasks_by_source[source_idx]
                                        : 0u;
        const unsigned int old_active_imask = active_imask;
        if (dirty)
        {
            const unsigned int refreshed_imask =
                Prune_Gmxpacked_Record_Stream_Source_Imask_From_Layout(
                    source, permutation, cluster_offsets, super_cluster_offsets,
                    cluster_local_masks, cluster_centers, crd, cell, rcell,
                    cutoff_sq);
            active_imask = union_active_imask != 0
                               ? (old_active_imask | refreshed_imask)
                               : refreshed_imask;
            if (active_imasks_by_source != NULL)
            {
                active_imasks_by_source[source_idx] = active_imask;
            }
        }
        if (dirty && active_imask != old_active_imask &&
            changed_active_source_rows != NULL)
        {
            atomicAdd(changed_active_source_rows, 1);
        }
        active_flags[source_idx] = active_imask != 0u ? 1 : 0;
    }
}

static __global__ void Fill_Gmxpacked_Record_Stream_Active_View_Cached_Flags(
    const int source_rows, const unsigned int* active_imasks_by_source,
    int* active_flags)
{
    SIMPLE_DEVICE_FOR(source_idx, source_rows)
    {
        const unsigned int active_imask =
            active_imasks_by_source != NULL
                ? active_imasks_by_source[source_idx]
                : 0u;
        active_flags[source_idx] = active_imask != 0u ? 1 : 0;
    }
}

static __global__ void
Refresh_Gmxpacked_Record_Stream_Active_View_Flags_From_Source_Indices(
    const int dirty_source_rows, const int* dirty_source_indices,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* outer_sources,
    const int candidate_sci_numbers, const int* dirty_candidate_sci,
    const int* permutation, const int* cluster_offsets,
    const int* super_cluster_offsets, const unsigned int* cluster_local_masks,
    const VECTOR* cluster_centers, const VECTOR* crd,
    const VECTOR* active_anchor_crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float cutoff_sq, const float reuse_margin,
    unsigned int* active_imasks_by_source, int* active_flags,
    int* fallback_flag, int* changed_active_source_rows, int union_active_imask)
{
    SIMPLE_DEVICE_FOR(dirty_idx, dirty_source_rows)
    {
        const int source_idx = dirty_source_indices[dirty_idx];
        const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE& source =
            outer_sources[source_idx];
        const int candidate_sci = source.sci_id;
        if (candidate_sci < 0 || candidate_sci >= candidate_sci_numbers)
        {
            if (fallback_flag != NULL)
            {
                atomicExch(fallback_flag, 1);
            }
            active_flags[source_idx] = 0;
            return;
        }
        const bool dirty = dirty_candidate_sci == NULL ||
                           dirty_candidate_sci[candidate_sci] != 0 ||
                           Gmxpacked_Record_Stream_Source_Active_Anchor_Dirty(
                               source, permutation, cluster_offsets,
                               super_cluster_offsets, cluster_local_masks, crd,
                               active_anchor_crd, cell, rcell, reuse_margin);
        unsigned int active_imask = active_imasks_by_source != NULL
                                        ? active_imasks_by_source[source_idx]
                                        : 0u;
        const unsigned int old_active_imask = active_imask;
        if (dirty)
        {
            const unsigned int refreshed_imask =
                Prune_Gmxpacked_Record_Stream_Source_Imask_From_Layout(
                    source, permutation, cluster_offsets, super_cluster_offsets,
                    cluster_local_masks, cluster_centers, crd, cell, rcell,
                    cutoff_sq);
            active_imask = union_active_imask != 0
                               ? (old_active_imask | refreshed_imask)
                               : refreshed_imask;
            if (active_imasks_by_source != NULL)
            {
                active_imasks_by_source[source_idx] = active_imask;
            }
        }
        if (dirty && active_imask != old_active_imask &&
            changed_active_source_rows != NULL)
        {
            atomicAdd(changed_active_source_rows, 1);
        }
        active_flags[source_idx] = active_imask != 0u ? 1 : 0;
    }
}

static __global__ void Fill_Gmxpacked_Record_Stream_Active_View_Sources(
    const int source_rows,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* outer_sources,
    const unsigned int* active_imasks_by_source, const int* active_offsets,
    CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* active_sources)
{
    SIMPLE_DEVICE_FOR(source_idx, source_rows)
    {
        const unsigned int active_imask =
            active_imasks_by_source != NULL
                ? active_imasks_by_source[source_idx]
                : 0u;
        if (active_imask == 0u)
        {
            return;
        }
        const int write_idx = active_offsets[source_idx];
        CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE source =
            outer_sources[source_idx];
        source.imask = active_imask;
        source.source_order = write_idx;
#pragma unroll
        for (int pair_idx = 0; pair_idx < kClusteredGmxpackedExclusionPairCount;
             pair_idx += 1)
        {
            source.pair_exclusion_words[pair_idx] &= active_imask;
        }
        active_sources[write_idx] = source;
    }
}

static __global__ void Count_Gmxpacked_Active_View_Dirty_Source_Rows(
    const int source_rows,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* outer_sources,
    const int candidate_sci_numbers, const int* dirty_candidate_sci,
    const int* permutation, const int* cluster_offsets,
    const int* super_cluster_offsets, const unsigned int* cluster_local_masks,
    const VECTOR* crd, const VECTOR* active_anchor_crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float reuse_margin, int* dirty_source_rows)
{
    SIMPLE_DEVICE_FOR(source_idx, source_rows)
    {
        const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE& source =
            outer_sources[source_idx];
        const int candidate_sci = source.sci_id;
        bool dirty =
            candidate_sci < 0 || candidate_sci >= candidate_sci_numbers;
        if (!dirty)
        {
            dirty = dirty_candidate_sci == NULL ||
                    dirty_candidate_sci[candidate_sci] != 0 ||
                    Gmxpacked_Record_Stream_Source_Active_Anchor_Dirty(
                        source, permutation, cluster_offsets,
                        super_cluster_offsets, cluster_local_masks, crd,
                        active_anchor_crd, cell, rcell, reuse_margin);
        }
        if (dirty)
        {
            atomicAdd(dirty_source_rows, 1);
        }
    }
}

#endif
