static void Commit_Clustered_Build_Cache(ClusteredNeighborProvider* layout,
                                         const VECTOR* crd, float cutoff)
{
    if (ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers > 0)
    {
        Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers,
                              &ClusteredNeighborProviderInternal::Spatial(layout).ordering.cached_crd);
        deviceMemcpy(ClusteredNeighborProviderInternal::Spatial(layout).ordering.cached_crd.data, crd,
                     sizeof(VECTOR) * ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers,
                     deviceMemcpyDeviceToDevice);
    }
    ClusteredNeighborProviderInternal::CachedCutoff(layout) = cutoff;
    ClusteredNeighborProviderInternal::CachedBuildStep(layout) = md_info.sys.steps;
    ClusteredNeighborProviderInternal::Spatial(layout).ready = ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers > 0;
    ClusteredNeighborProviderInternal::RebuildDirty(layout) = false;
}

static int Exclusive_Scan_Counts(ClusteredNeighborProvider* layout, int count_numbers,
                                 int* d_counts, int* d_starts);

static bool Clustered_Build_Is_Needed(ClusteredNeighborProvider* layout,
                                      const VECTOR* crd, const LTMatrix3 cell,
                                      const LTMatrix3 rcell, const float cutoff)
{
    if (ClusteredNeighborProviderInternal::RebuildDirty(layout) || !ClusteredNeighborProviderInternal::Spatial(layout).ready ||
        ClusteredNeighborProviderInternal::Spatial(layout).ordering.cached_crd.data == NULL)
    {
        return true;
    }
    if (fabsf(ClusteredNeighborProviderInternal::CachedCutoff(layout) - cutoff) > 1e-6f)
    {
        return true;
    }
    if (ClusteredNeighborProviderInternal::Config(layout).refresh_interval > 0)
    {
        const int steps_since_build =
            md_info.sys.steps - ClusteredNeighborProviderInternal::CachedBuildStep(layout);
        if (steps_since_build < ClusteredNeighborProviderInternal::Config(layout).refresh_interval)
        {
            return false;
        }
        return true;
    }
    if (ClusteredNeighborProviderInternal::EffectiveRebuildSkin(layout) <= 0.0f ||
        ClusteredNeighborProviderInternal::Config(layout).skin_permit <= 0.0f ||
        ClusteredNeighborProviderInternal::Domain(layout).local_atom_count <= 0)
    {
        return true;
    }

    Reserve_Device_Buffer(1, &ClusteredNeighborProviderInternal::Workspace(layout).need_rebuild);
    int h_need_rebuild = 0;
    deviceMemcpy(ClusteredNeighborProviderInternal::Workspace(layout).need_rebuild.data, &h_need_rebuild,
                 sizeof(int), deviceMemcpyHostToDevice);
    const float permit =
        ClusteredNeighborProviderInternal::EffectiveRebuildSkin(layout) * ClusteredNeighborProviderInternal::Config(layout).skin_permit;
    Launch_Device_Kernel(
        Check_Clustered_Rebuild,
        (ClusteredNeighborProviderInternal::Domain(layout).local_atom_count + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, ClusteredNeighborProviderInternal::Domain(layout).local_atom_count,
        crd, ClusteredNeighborProviderInternal::Spatial(layout).ordering.cached_crd.data, cell, rcell,
        ClusteredNeighborProviderInternal::Workspace(layout).need_rebuild.data, permit * permit);
    deviceMemcpy(&h_need_rebuild, ClusteredNeighborProviderInternal::Workspace(layout).need_rebuild.data,
                 sizeof(int), deviceMemcpyDeviceToHost);
    return h_need_rebuild != 0;
}

static float Clustered_Max_Gmxpacked_Inner_Active_Anchor_Displacement(
    ClusteredNeighborProvider* layout, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell)
{
    if (layout == NULL || crd == NULL ||
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_anchor_crd.data == NULL ||
        ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers <= 0)
    {
        return 0.0f;
    }
#ifdef USE_CPU
    (void)cell;
    (void)rcell;
    return 0.0f;
#else
    Bind_Clustered_Working_Device(&ClusteredNeighborProviderInternal::WorkingDevice(layout));
    Reserve_Device_Buffer(1, &ClusteredNeighborProviderInternal::Workspace(layout).need_rebuild);
    int max_displacement_sq_bits = 0;
    deviceMemcpy(ClusteredNeighborProviderInternal::Workspace(layout).need_rebuild.data, &max_displacement_sq_bits,
                 sizeof(int), deviceMemcpyHostToDevice);
    Launch_Device_Kernel(
        Atomic_Max_Clustered_Anchor_Displacement_Sq,
        (ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers + CONTROLLER::device_max_thread -
         1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL,
        ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers, crd,
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_anchor_crd.data, cell, rcell,
        ClusteredNeighborProviderInternal::Workspace(layout).need_rebuild.data);
    const cudaError_t sync_err = cudaDeviceSynchronize();
    Clustered_Check_Cuda_Status(sync_err,
                                "clustered-inner-active-displacement-max-sync");
    deviceMemcpy(&max_displacement_sq_bits, ClusteredNeighborProviderInternal::Workspace(layout).need_rebuild.data,
                 sizeof(int), deviceMemcpyDeviceToHost);
    union
    {
        int bits;
        float value;
    } max_displacement_square = {max_displacement_sq_bits};
    return sqrtf(fmaxf(max_displacement_square.value, 0.0f));
#endif
}

static int Exclusive_Scan_Counts(ClusteredNeighborProvider* layout, int count_numbers,
                                 int* d_counts, int* d_starts)
{
    if (count_numbers <= 0)
    {
        return 0;
    }
#ifndef USE_CPU
    if (layout == NULL)
    {
        return 0;
    }
    Bind_Clustered_Working_Device(&ClusteredNeighborProviderInternal::WorkingDevice(layout));

    size_t reduce_storage_bytes = 0;
    size_t scan_storage_bytes = 0;

    Reserve_Device_Buffer(1, &ClusteredNeighborProviderInternal::Workspace(layout).scan_total, "scan-total");
    cub::DeviceReduce::Sum(NULL, reduce_storage_bytes, d_counts,
                           ClusteredNeighborProviderInternal::Workspace(layout).scan_total.data, count_numbers);
    cub::DeviceScan::ExclusiveSum(NULL, scan_storage_bytes, d_counts, d_starts,
                                  count_numbers);

    clustered_neighbor_runtime::Reserve_Raw_Device_Workspace(
        reduce_storage_bytes, &ClusteredNeighborProviderInternal::Workspace(layout).reduce);
    clustered_neighbor_runtime::Reserve_Raw_Device_Workspace(scan_storage_bytes,
                                                       &ClusteredNeighborProviderInternal::Workspace(layout).scan);

    cub::DeviceReduce::Sum(ClusteredNeighborProviderInternal::Workspace(layout).reduce.data, reduce_storage_bytes,
                           d_counts, ClusteredNeighborProviderInternal::Workspace(layout).scan_total.data,
                           count_numbers);
    cub::DeviceScan::ExclusiveSum(ClusteredNeighborProviderInternal::Workspace(layout).scan.data,
                                  scan_storage_bytes, d_counts, d_starts,
                                  count_numbers);
    deviceMemcpy(d_starts + count_numbers, ClusteredNeighborProviderInternal::Workspace(layout).scan_total.data,
                 sizeof(int), deviceMemcpyDeviceToDevice);

    int total = 0;
    deviceMemcpy(&total, ClusteredNeighborProviderInternal::Workspace(layout).scan_total.data, sizeof(int),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(d_starts + count_numbers, &total, sizeof(int),
                 deviceMemcpyHostToDevice);
    return total;
#else
    std::vector<int> h_counts(static_cast<size_t>(count_numbers));
    std::vector<int> h_starts(static_cast<size_t>(count_numbers) + 1, 0);
    deviceMemcpy(h_counts.data(), d_counts, sizeof(int) * count_numbers,
                 deviceMemcpyDeviceToHost);
    int total = 0;
    for (int i = 0; i < count_numbers; i += 1)
    {
        h_starts[static_cast<size_t>(i)] = total;
        total += h_counts[static_cast<size_t>(i)];
    }
    h_starts[static_cast<size_t>(count_numbers)] = total;
    deviceMemcpy(d_starts, h_starts.data(), sizeof(int) * (count_numbers + 1),
                 deviceMemcpyHostToDevice);
    return total;
#endif
}

#ifndef USE_CPU
static bool Build_Gmxpacked_Incremental_Dirty_Device_Mask(
    ClusteredNeighborProvider* layout, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const int* candidate_sci_supercluster_ids,
    bool* j_leaf_scope_ready)
{
    if (j_leaf_scope_ready != NULL)
    {
        *j_leaf_scope_ready = false;
    }
    if (layout == NULL || crd == NULL ||
        ClusteredNeighborProviderInternal::Spatial(layout).ordering.cached_crd.data == NULL ||
        ClusteredNeighborProviderInternal::Spatial(layout).ordering.sort_permutation.data == NULL ||
        ClusteredNeighborProviderInternal::Spatial(layout).superclusters.cluster_to_supercluster.data == NULL ||
        candidate_sci_supercluster_ids == NULL ||
        ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers <= 0 ||
        ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers <= 0 ||
        ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers <= 0 ||
        ClusteredNeighborProviderInternal::Spatial(layout).candidates.candidate_sci_numbers <= 0)
    {
        return false;
    }
    Bind_Clustered_Working_Device(&ClusteredNeighborProviderInternal::WorkingDevice(layout));
    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers,
                          &ClusteredNeighborProviderInternal::Workspace(layout).incremental_dirty_atoms);
    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers,
                          &ClusteredNeighborProviderInternal::Workspace(layout).incremental_dirty_clusters);
    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers,
                          &ClusteredNeighborProviderInternal::Workspace(layout).incremental_dirty_superclusters);
    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).candidates.candidate_sci_numbers,
                          &ClusteredNeighborProviderInternal::Workspace(layout).incremental_dirty_i_candidate_sci);
    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).candidates.candidate_sci_numbers,
                          &ClusteredNeighborProviderInternal::Workspace(layout).incremental_dirty_candidate_sci);

    deviceMemset(ClusteredNeighborProviderInternal::Workspace(layout).incremental_dirty_atoms.data, 0,
                 sizeof(int) * ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers);
    deviceMemset(ClusteredNeighborProviderInternal::Workspace(layout).incremental_dirty_clusters.data, 0,
                 sizeof(int) * ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers);
    deviceMemset(ClusteredNeighborProviderInternal::Workspace(layout).incremental_dirty_superclusters.data, 0,
                 sizeof(int) * ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers);
    deviceMemset(
        ClusteredNeighborProviderInternal::Workspace(layout).incremental_dirty_i_candidate_sci.data, 0,
        sizeof(int) * ClusteredNeighborProviderInternal::Spatial(layout).candidates.candidate_sci_numbers);
    deviceMemset(
        ClusteredNeighborProviderInternal::Workspace(layout).incremental_dirty_candidate_sci.data, 0,
        sizeof(int) * ClusteredNeighborProviderInternal::Spatial(layout).candidates.candidate_sci_numbers);

    const float permit =
        ClusteredNeighborProviderInternal::EffectiveRebuildSkin(layout) * ClusteredNeighborProviderInternal::Config(layout).skin_permit;
    Launch_Device_Kernel(
        Mark_Clustered_Incremental_Dirty_Clusters,
        (ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers + CONTROLLER::device_max_thread -
         1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL,
        ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers, ClusteredNeighborProviderInternal::Config(layout).cluster_size,
        ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers, ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers,
        crd, ClusteredNeighborProviderInternal::Spatial(layout).ordering.cached_crd.data, cell, rcell,
        ClusteredNeighborProviderInternal::Spatial(layout).ordering.sort_permutation.data,
        ClusteredNeighborProviderInternal::Spatial(layout).superclusters.cluster_to_supercluster.data,
        permit * permit, ClusteredNeighborProviderInternal::Workspace(layout).incremental_dirty_atoms.data,
        ClusteredNeighborProviderInternal::Workspace(layout).incremental_dirty_clusters.data,
        ClusteredNeighborProviderInternal::Workspace(layout).incremental_dirty_superclusters.data);

    const int candidate_super_id_numbers =
        (ClusteredNeighborProviderInternal::Spatial(layout).candidates.candidate_sci_numbers +
         kClusteredShiftCount - 1) /
        kClusteredShiftCount;
    Launch_Device_Kernel(
        Mark_Gmxpacked_Incremental_Dirty_I_Candidates,
        (ClusteredNeighborProviderInternal::Spatial(layout).candidates.candidate_sci_numbers +
         CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL,
        ClusteredNeighborProviderInternal::Spatial(layout).candidates.candidate_sci_numbers,
        candidate_super_id_numbers, ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers,
        candidate_sci_supercluster_ids,
        ClusteredNeighborProviderInternal::Workspace(layout).incremental_dirty_superclusters.data,
        ClusteredNeighborProviderInternal::Workspace(layout).incremental_dirty_i_candidate_sci.data,
        ClusteredNeighborProviderInternal::Workspace(layout).incremental_dirty_candidate_sci.data);

    const bool has_j_leaf_scope =
        ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_numbers > 0 &&
        ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_offsets.data != NULL &&
        ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_ids.data != NULL &&
        ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_starts.data != NULL &&
        ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_ends.data != NULL;
    if (has_j_leaf_scope)
    {
        Launch_Device_Kernel(
            Mark_Gmxpacked_Incremental_Dirty_J_Candidates_Parallel,
            ClusteredNeighborProviderInternal::Spatial(layout).candidates.candidate_sci_numbers, 64, 0, NULL,
            ClusteredNeighborProviderInternal::Spatial(layout).candidates.candidate_sci_numbers,
            ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_numbers,
            ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers,
            ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_offsets.data,
            ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_ids.data,
            ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_starts.data,
            ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_ends.data,
            ClusteredNeighborProviderInternal::Workspace(layout).incremental_dirty_clusters.data,
            ClusteredNeighborProviderInternal::Workspace(layout).incremental_dirty_candidate_sci.data);
    }
    if (j_leaf_scope_ready != NULL)
    {
        *j_leaf_scope_ready = has_j_leaf_scope;
    }
    return true;
}

#else
static bool Build_Gmxpacked_Incremental_Dirty_Device_Mask(
    ClusteredNeighborProvider*, const VECTOR*, LTMatrix3, LTMatrix3, const int*,
    bool* j_leaf_scope_ready)
{
    if (j_leaf_scope_ready != NULL)
    {
        *j_leaf_scope_ready = false;
    }
    return false;
}

#endif

static int Count_Gmxpacked_Record_Stream_Inner_Active_Source_Rows(
    ClusteredNeighborProvider* layout, int source_rows,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    const VECTOR* crd, LTMatrix3 cell, LTMatrix3 rcell, float cutoff,
    bool record_active_imasks)
{
    if (layout == NULL || source_rows <= 0 || sources == NULL || crd == NULL)
    {
        return 0;
    }
#ifdef USE_CPU
    (void)cell;
    (void)rcell;
    (void)cutoff;
    return source_rows;
#else
    Reserve_Device_Buffer(source_rows + 1,
                          &ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts);
    Reserve_Device_Buffer(source_rows + 1,
                          &ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_offsets);
    unsigned int* active_imasks_by_source = NULL;
    if (record_active_imasks)
    {
        Reserve_Device_Buffer(
            source_rows,
            &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_imasks);
        active_imasks_by_source =
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_imasks.data;
    }
    Launch_Device_Kernel(
        Count_Gmxpacked_Record_Stream_Inner_Active_Sources,
        (source_rows + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, source_rows, sources,
        ClusteredNeighborProviderInternal::Spatial(layout).ordering.sort_permutation.data,
        ClusteredNeighborProviderInternal::Spatial(layout).clusters.offsets.data,
        ClusteredNeighborProviderInternal::Spatial(layout).superclusters.offsets.data,
        ClusteredNeighborProviderInternal::Spatial(layout).clusters.local_masks.data,
        ClusteredNeighborProviderInternal::Spatial(layout).clusters.centers.data, crd, cell, rcell,
        cutoff * cutoff, ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts.data,
        active_imasks_by_source);
    if (record_active_imasks)
    {
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_imasks_ready = true;
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_imask_numbers =
            source_rows;
    }
    return Exclusive_Scan_Counts(layout, source_rows,
                                 ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts.data,
                                 ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_offsets.data);
#endif
}

static int Prune_Gmxpacked_Record_Stream_To_Inner_Active_Sources(
    ClusteredNeighborProvider* layout, const VECTOR* crd, LTMatrix3 cell,
    LTMatrix3 rcell, float cutoff, bool record_active_imasks = false)
{
    if (layout == NULL ||
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_source_numbers <= 0 ||
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_sources.data == NULL ||
        crd == NULL)
    {
        return layout != NULL
                   ? ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_source_numbers
                   : 0;
    }
#ifdef USE_CPU
    (void)cell;
    (void)rcell;
    (void)cutoff;
    return ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_source_numbers;
#else
    const int source_rows =
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_source_numbers;
    const float cutoff_sq = cutoff * cutoff;
    const int active_source_rows =
        Count_Gmxpacked_Record_Stream_Inner_Active_Source_Rows(
            layout, source_rows,
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_sources.data, crd, cell,
            rcell, cutoff, record_active_imasks);
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_source_numbers =
        active_source_rows;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregate_numbers = 0;
    if (record_active_imasks)
    {
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_rows_baseline =
            active_source_rows;
    }
    if (active_source_rows <= 0)
    {
        return 0;
    }

    {
        clustered_neighbor_runtime::Reserve_Raw_Device_Workspace(
            sizeof(CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE) *
                static_cast<size_t>(active_source_rows),
            &ClusteredNeighborProviderInternal::Workspace(layout).sort_values);
        CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* active_sources =
            reinterpret_cast<CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE*>(
                ClusteredNeighborProviderInternal::Workspace(layout).sort_values.data);
        const bool use_cached_fill =
            record_active_imasks &&
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_imasks_ready &&
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_imask_numbers ==
                source_rows &&
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_imasks.data != NULL;
        if (use_cached_fill)
        {
            Launch_Device_Kernel(
                Fill_Gmxpacked_Record_Stream_Active_View_Sources,
                (source_rows + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, source_rows,
                ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_sources.data,
                ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_imasks.data,
                ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_offsets.data, active_sources);
        }
        else
        {
            Launch_Device_Kernel(
                Fill_Gmxpacked_Record_Stream_Inner_Active_Sources,
                (source_rows + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, source_rows,
                ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_sources.data,
                ClusteredNeighborProviderInternal::Spatial(layout).ordering.sort_permutation.data,
                ClusteredNeighborProviderInternal::Spatial(layout).clusters.offsets.data,
                ClusteredNeighborProviderInternal::Spatial(layout).superclusters.offsets.data,
                ClusteredNeighborProviderInternal::Spatial(layout).clusters.local_masks.data,
                ClusteredNeighborProviderInternal::Spatial(layout).clusters.centers.data, crd, cell, rcell,
                cutoff_sq, ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_offsets.data,
                active_sources);
        }
        deviceMemcpy(ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_sources.data,
                     active_sources,
                     sizeof(CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE) *
                         static_cast<size_t>(active_source_rows),
                     deviceMemcpyDeviceToDevice);
    }
    return active_source_rows;
#endif
}

static int Refresh_Gmxpacked_Record_Stream_Active_View_Sources(
    ClusteredNeighborProvider* layout, int outer_source_rows,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* outer_sources,
    int candidate_sci_numbers, const int* dirty_candidate_sci,
    const int* source_offsets_by_candidate, const VECTOR* crd, LTMatrix3 cell,
    LTMatrix3 rcell, float cutoff, const VECTOR* active_anchor_crd,
    float active_reuse_margin, int* dirty_source_rows_out,
    int* changed_source_rows_out, int zero_dirty_reuse_source_rows = 0,
    bool allow_full_dirty_refresh = false)
{
    if (dirty_source_rows_out != NULL)
    {
        *dirty_source_rows_out = -1;
    }
    if (changed_source_rows_out != NULL)
    {
        *changed_source_rows_out = -1;
    }
    if (layout == NULL || outer_source_rows <= 0 || outer_sources == NULL ||
        candidate_sci_numbers <= 0 ||
        (dirty_candidate_sci == NULL && !allow_full_dirty_refresh) ||
        source_offsets_by_candidate == NULL || crd == NULL ||
        !ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_imasks_ready ||
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_imask_numbers !=
            outer_source_rows ||
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_imasks.data == NULL)
    {
        return -1;
    }
#ifdef USE_CPU
    (void)cell;
    (void)rcell;
    (void)cutoff;
    (void)active_anchor_crd;
    (void)active_reuse_margin;
    return -1;
#else
    int dirty_source_rows = 0;
    const bool use_dirty_index_refresh = kActiveViewDirtyIndexRefresh &&
                                         dirty_candidate_sci != NULL &&
                                         !allow_full_dirty_refresh;
    {
        if (use_dirty_index_refresh)
        {
            Reserve_Device_Buffer(candidate_sci_numbers + 1,
                                  &ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts);
            Reserve_Device_Buffer(
                candidate_sci_numbers + 1,
                &ClusteredNeighborProviderInternal::Workspace(layout)
                     .incremental_replacement_source_offsets_by_candidate);
            Launch_Device_Kernel(
                Count_Gmxpacked_Dirty_Source_Rows_By_Candidate,
                (candidate_sci_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, candidate_sci_numbers,
                dirty_candidate_sci, source_offsets_by_candidate,
                ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts.data);
            dirty_source_rows = Exclusive_Scan_Counts(
                layout, candidate_sci_numbers,
                ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts.data,
                ClusteredNeighborProviderInternal::Workspace(layout)
                    .incremental_replacement_source_offsets_by_candidate.data);
        }
        else
        {
            Reserve_Device_Buffer(1, &ClusteredNeighborProviderInternal::Workspace(layout).need_rebuild);
            deviceMemset(ClusteredNeighborProviderInternal::Workspace(layout).need_rebuild.data, 0, sizeof(int));
            Launch_Device_Kernel(
                Count_Gmxpacked_Active_View_Dirty_Source_Rows,
                (outer_source_rows + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, outer_source_rows,
                outer_sources, candidate_sci_numbers, dirty_candidate_sci,
                ClusteredNeighborProviderInternal::Spatial(layout).ordering.sort_permutation.data,
                ClusteredNeighborProviderInternal::Spatial(layout).clusters.offsets.data,
                ClusteredNeighborProviderInternal::Spatial(layout).superclusters.offsets.data,
                ClusteredNeighborProviderInternal::Spatial(layout).clusters.local_masks.data, crd,
                active_anchor_crd, cell, rcell, active_reuse_margin,
                ClusteredNeighborProviderInternal::Workspace(layout).need_rebuild.data);
            deviceMemcpy(&dirty_source_rows,
                         ClusteredNeighborProviderInternal::Workspace(layout).need_rebuild.data, sizeof(int),
                         deviceMemcpyDeviceToHost);
        }
    }
    if (dirty_source_rows_out != NULL)
    {
        *dirty_source_rows_out = dirty_source_rows;
    }
    int changed_source_rows = -1;
    const float dirty_ratio = outer_source_rows > 0
                                  ? static_cast<float>(dirty_source_rows) /
                                        static_cast<float>(outer_source_rows)
                                  : 1.0f;
    const bool full_dirty_refresh =
        allow_full_dirty_refresh && dirty_candidate_sci == NULL;
    const float dirty_ratio_limit = kActiveViewDirtySourceRatioLimit;
    if (dirty_source_rows < 0 || dirty_source_rows > outer_source_rows ||
        (!full_dirty_refresh && dirty_ratio > dirty_ratio_limit))
    {
        return -1;
    }
    if (dirty_source_rows == 0 && zero_dirty_reuse_source_rows > 0)
    {
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_source_numbers =
            zero_dirty_reuse_source_rows;
        changed_source_rows = 0;
        if (changed_source_rows_out != NULL)
        {
            *changed_source_rows_out = changed_source_rows;
        }
        return zero_dirty_reuse_source_rows;
    }

    int active_view_fallback = 0;
    int active_source_rows = 0;
    {
        Reserve_Device_Buffer(outer_source_rows + 1,
                              &ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts);
        Reserve_Device_Buffer(outer_source_rows + 1,
                              &ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_offsets);
        Reserve_Device_Buffer(1, &ClusteredNeighborProviderInternal::Workspace(layout).need_rebuild);
        Reserve_Device_Buffer(
            1, &ClusteredNeighborProviderInternal::Workspace(layout)
                    .incremental_replacement_source_counts_by_candidate);
        deviceMemset(ClusteredNeighborProviderInternal::Workspace(layout).need_rebuild.data, 0, sizeof(int));
        deviceMemset(
            ClusteredNeighborProviderInternal::Workspace(layout).incremental_replacement_source_counts_by_candidate
                .data,
            0, sizeof(int));
        if (use_dirty_index_refresh)
        {
            Launch_Device_Kernel(
                Fill_Gmxpacked_Record_Stream_Active_View_Cached_Flags,
                (outer_source_rows + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, outer_source_rows,
                ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_imasks.data,
                ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts.data);
            if (dirty_source_rows > 0)
            {
                Reserve_Device_Buffer(
                    dirty_source_rows,
                    &ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices);
                Launch_Device_Kernel(
                    Fill_Gmxpacked_Dirty_Source_Indices_By_Candidate,
                    (candidate_sci_numbers + CONTROLLER::device_max_thread -
                     1) /
                        CONTROLLER::device_max_thread,
                    CONTROLLER::device_max_thread, 0, NULL,
                    candidate_sci_numbers, dirty_candidate_sci,
                    source_offsets_by_candidate,
                    ClusteredNeighborProviderInternal::Workspace(layout)
                        .incremental_replacement_source_offsets_by_candidate
                        .data,
                    ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data);
                Launch_Device_Kernel(
                    Refresh_Gmxpacked_Record_Stream_Active_View_Flags_From_Source_Indices,
                    (dirty_source_rows + CONTROLLER::device_max_thread - 1) /
                        CONTROLLER::device_max_thread,
                    CONTROLLER::device_max_thread, 0, NULL, dirty_source_rows,
                    ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data,
                    outer_sources, candidate_sci_numbers, dirty_candidate_sci,
                    ClusteredNeighborProviderInternal::Spatial(layout).ordering.sort_permutation.data,
                    ClusteredNeighborProviderInternal::Spatial(layout).clusters.offsets.data,
                    ClusteredNeighborProviderInternal::Spatial(layout).superclusters.offsets.data,
                    ClusteredNeighborProviderInternal::Spatial(layout).clusters.local_masks.data,
                    ClusteredNeighborProviderInternal::Spatial(layout).clusters.centers.data, crd,
                    active_anchor_crd, cell, rcell, cutoff * cutoff,
                    active_reuse_margin,
                    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_imasks.data,
                    ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts.data,
                    ClusteredNeighborProviderInternal::Workspace(layout).need_rebuild.data,
                    ClusteredNeighborProviderInternal::Workspace(layout)
                        .incremental_replacement_source_counts_by_candidate
                        .data,
                    0);
            }
        }
        else
        {
            Launch_Device_Kernel(
                Refresh_Gmxpacked_Record_Stream_Active_View_Flags,
                (outer_source_rows + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, outer_source_rows,
                outer_sources, candidate_sci_numbers, dirty_candidate_sci,
                ClusteredNeighborProviderInternal::Spatial(layout).ordering.sort_permutation.data,
                ClusteredNeighborProviderInternal::Spatial(layout).clusters.offsets.data,
                ClusteredNeighborProviderInternal::Spatial(layout).superclusters.offsets.data,
                ClusteredNeighborProviderInternal::Spatial(layout).clusters.local_masks.data,
                ClusteredNeighborProviderInternal::Spatial(layout).clusters.centers.data, crd, active_anchor_crd,
                cell, rcell, cutoff * cutoff, active_reuse_margin,
                ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_imasks.data,
                ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts.data,
                ClusteredNeighborProviderInternal::Workspace(layout).need_rebuild.data,
                ClusteredNeighborProviderInternal::Workspace(layout)
                    .incremental_replacement_source_counts_by_candidate.data,
                0);
        }
        deviceMemcpy(&active_view_fallback, ClusteredNeighborProviderInternal::Workspace(layout).need_rebuild.data,
                     sizeof(int), deviceMemcpyDeviceToHost);
        deviceMemcpy(
            &changed_source_rows,
            ClusteredNeighborProviderInternal::Workspace(layout).incremental_replacement_source_counts_by_candidate
                .data,
            sizeof(int), deviceMemcpyDeviceToHost);
        active_source_rows = Exclusive_Scan_Counts(
            layout, outer_source_rows,
            ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts.data,
            ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_offsets.data);
    }
    if (changed_source_rows_out != NULL)
    {
        *changed_source_rows_out = changed_source_rows;
    }
    if (active_view_fallback != 0)
    {
        return -1;
    }

    const int previous_active_source_rows =
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_source_numbers;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_source_numbers =
        active_source_rows;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregate_numbers = 0;
    if (active_source_rows <= 0)
    {
        return 0;
    }
    {
        clustered_neighbor_runtime::Reserve_Raw_Device_Workspace(
            sizeof(CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE) *
                static_cast<size_t>(active_source_rows),
            &ClusteredNeighborProviderInternal::Workspace(layout).sort_values);
        CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* active_sources =
            reinterpret_cast<CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE*>(
                ClusteredNeighborProviderInternal::Workspace(layout).sort_values.data);
        Launch_Device_Kernel(
            Fill_Gmxpacked_Record_Stream_Active_View_Sources,
            (outer_source_rows + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, outer_source_rows,
            outer_sources,
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_imasks.data,
            ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_offsets.data, active_sources);
        Reserve_Device_Buffer(
            active_source_rows,
            &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_sources);
        deviceMemcpy(ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_sources.data,
                     active_sources,
                     sizeof(CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE) *
                         static_cast<size_t>(active_source_rows),
                     deviceMemcpyDeviceToDevice);
    }
    return active_source_rows;
#endif
}

using ClusteredGmxpackedRecordStreamCompactSummary =
    clustered_neighbor_builder_internal::CompactPayloadSummary;
