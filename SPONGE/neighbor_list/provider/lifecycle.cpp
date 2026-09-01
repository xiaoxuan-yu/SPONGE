#include "lifecycle.h"

#include <algorithm>

#include "endpoint_incidence.h"
#include "internal.h"
#include "runtime.h"

void ClusteredNeighborProvider::Initialize(CONTROLLER* controller,
                                           const ClusteredBuildConfig& config)
{
    if (initialized_)
    {
        return;
    }
    config_ = config;
    effective_rebuild_skin_ = config_.rebuild_skin;
    rebuild_dirty_ = true;
    spatial_.ready = false;
    cached_cutoff_ = -1.0f;
    Invalidate_Gmxpacked_Incremental_Source_Cache_State(this);
    controller_ = controller;
    working_device_ = controller_->working_device;
    cached_build_step_ = -1;
    initialized_ = true;
}

void ClusteredNeighborProvider::BindDomain(const ClusteredDomainBinding& domain)
{
    if (!initialized_)
    {
        return;
    }
    ClusteredDomainBinding normalized = domain;
    normalized.direct_local_atom_count =
        std::min(normalized.local_atom_count,
                 std::max(0, normalized.direct_local_atom_count));
    const bool local_metadata_changed =
        domain_.local_atom_count != normalized.local_atom_count ||
        domain_.direct_local_atom_count != normalized.direct_local_atom_count ||
        domain_.ghost_atom_count != normalized.ghost_atom_count ||
        domain_.atom_local != normalized.atom_local ||
        domain_.excluded_list_start != normalized.excluded_list_start ||
        domain_.excluded_list != normalized.excluded_list ||
        domain_.excluded_numbers != normalized.excluded_numbers;
    domain_ = normalized;
    spatial_.total_atom_numbers =
        domain_.local_atom_count + domain_.ghost_atom_count;
    const bool conservative_dd_refresh = CONTROLLER::PP_MPI_size > 1;
    spatial_.padded_total_atom_numbers = spatial_.total_atom_numbers;
    if (local_metadata_changed || conservative_dd_refresh)
    {
        Invalidate_Gmxpacked_Incremental_Source_Cache_State(this);
        rebuild_dirty_ = true;
        spatial_.ready = false;
        Publish_Gathered_Cluster_Geometry(this);
    }
    Ensure_Cornerstone_State(this);
}

void ClusteredNeighborProvider::Build(const ClusteredBuildRequest& request)
{
    if (!initialized_)
    {
        return;
    }
    BuildInternal(request.coordinates, request.cell, request.reciprocal_cell,
                  request.cutoff, request.need_endpoint_incidence);
    if (request.need_endpoint_incidence)
    {
        Build_Gmxpacked_Endpoint_Incidence_Metadata(this);
    }
}

void Invalidate_Gmxpacked_Incremental_Source_Cache_State(
    ClusteredNeighborProvider* layout)
{
    if (layout == NULL)
    {
        return;
    }
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_incremental_source_offsets_ready = false;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_incremental_source_cache_ready = false;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_incremental_candidate_sci_numbers = 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_incremental_source_numbers = 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_incremental_source_cutoff = -1.0f;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_incremental_source_anchor_generation = -1;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_incremental_source_generation = -1;
}

void Invalidate_Gmxpacked_Pair_Shift_Metadata(ClusteredNeighborProvider* layout)
{
    if (layout == NULL)
    {
        return;
    }
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_pair_shift_metadata_ready = false;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_pair_shift_metadata_sci_numbers = 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_pair_shift_metadata_cjpacked_numbers = 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_pair_shift_metadata_exclusion_numbers = 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_pair_shift_metadata_payload_generation = -1;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_pair_shift_metadata_geometry_generation = -1;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_pair_shift_metadata_rcell = {};
}

void Publish_Gathered_Cluster_Geometry(ClusteredNeighborProvider* layout)
{
    if (layout == NULL)
    {
        return;
    }
    ClusteredNeighborProviderInternal::LeaseEpoch(layout) += 1;
    ClusteredNeighborProviderInternal::Spatial(layout).geometry_generation += 1;
    Invalidate_Gmxpacked_Pair_Shift_Metadata(layout);
}

void Publish_Gmxpacked_Compact_Payload(ClusteredNeighborProvider* layout)
{
    if (layout == NULL)
    {
        return;
    }
    ClusteredNeighborProviderInternal::LeaseEpoch(layout) += 1;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_compact_payload_generation += 1;
    Invalidate_Gmxpacked_Endpoint_Incidence(layout);
    Invalidate_Gmxpacked_Pair_Shift_Metadata(layout);
}

void Clear_Gmxpacked_Compact_Payload(ClusteredNeighborProvider* layout)
{
    if (layout == NULL)
    {
        return;
    }
    const bool had_primary_payload =
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers > 0 ||
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked_numbers > 0 ||
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_exclusion_numbers > 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers = 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked_numbers = 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_exclusion_numbers = 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_split_exclusion_numbers = 0;
    Invalidate_Gmxpacked_Endpoint_Incidence(layout);
    Invalidate_Gmxpacked_Pair_Shift_Metadata(layout);
    if (had_primary_payload)
    {
        ClusteredNeighborProviderInternal::LeaseEpoch(layout) += 1;
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_compact_payload_generation += 1;
    }
}

void Reset_Gmxpacked_Payload(ClusteredNeighborProvider* layout)
{
    if (layout == NULL)
    {
        return;
    }
    Clear_Gmxpacked_Compact_Payload(layout);
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_source_numbers = 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregate_numbers = 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_guard_cutoff = -1.0f;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_anchor_ready = false;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_imasks_ready = false;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_imask_numbers = 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_rows_baseline = 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_incremental_source_anchor_generation = -1;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_incremental_source_generation = -1;
}

namespace
{
void FreeGmxpackedPayload(ClusteredNeighborProvider* layout)
{
    if (layout == NULL)
    {
        return;
    }
    clustered_neighbor_runtime::Release_Device_Buffer(
        &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_exclusions);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_sources);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregates);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_pair_shift_sci_safe_flags);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_anchor_crd);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_imasks);
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_anchor_ready = false;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_imasks_ready = false;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_imask_numbers = 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_inner_active_source_rows_baseline = 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_current_source_anchor_generation = 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_current_source_generation = 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_incremental_source_anchor_generation = -1;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_incremental_source_generation = -1;
    Reset_Gmxpacked_Payload(layout);
}

void ReleaseBuildWorkspace(ClusteredBuildWorkspace* workspace)
{
    clustered_neighbor_runtime::Release_Raw_Device_Workspace(&workspace->sort_keys);
    clustered_neighbor_runtime::Release_Raw_Device_Workspace(&workspace->sort_values);
    clustered_neighbor_runtime::Release_Raw_Device_Workspace(&workspace->sort);
    clustered_neighbor_runtime::Release_Raw_Device_Workspace(&workspace->reduce);
    clustered_neighbor_runtime::Release_Raw_Device_Workspace(&workspace->scan);
    clustered_neighbor_runtime::Release_Device_Buffer(&workspace->scan_total);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->leaf_cluster_span_max);
    clustered_neighbor_runtime::Release_Device_Buffer(&workspace->stable_sort_keys);
    clustered_neighbor_runtime::Release_Device_Buffer(&workspace->need_rebuild);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->record_scratch_counts);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->record_scratch_offsets);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->record_scratch_indices);

    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->candidate_leaf_onepass_sci_ids);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->candidate_leaf_onepass_ranks);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->candidate_leaf_onepass_leaf_ids);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->candidate_leaf_onepass_prev_running_max_ends);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->candidate_leaf_onepass_cursor);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->candidate_leaf_queue2_task_counter);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->candidate_leaf_queue2_task_overflow);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->candidate_leaf_queue2_task_work_cursor);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->candidate_leaf_queue2_task_sci_ids);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->candidate_leaf_queue2_task_nodes);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->candidate_leaf_queue2_task_sort_keys);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->candidate_leaf_queue2_task_pairs);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->candidate_leaf_queue2_task_leaf_counts);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->candidate_leaf_queue2_task_leaf_offsets);

    clustered_neighbor_runtime::Release_Device_Buffer(&workspace->sci_shift_flags);
    clustered_neighbor_runtime::Release_Device_Buffer(&workspace->sci_shift_offsets);
    clustered_neighbor_runtime::Release_Device_Buffer(&workspace->cjpacked_counts);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->cjpacked_group_offsets);
    clustered_neighbor_runtime::Release_Device_Buffer(&workspace->exclusion_counts);
    clustered_neighbor_runtime::Release_Device_Buffer(&workspace->exclusion_offsets);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->count_light_source_fragments);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->count_source_fragment_cursor);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->count_source_fragment_overflow_rows);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->count_source_materialize_cursors);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->source_counts_by_candidate);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->source_offsets_by_candidate);
    clustered_neighbor_runtime::Release_Device_Buffer(&workspace->source_fill_cursor);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->source_overflow_rows);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->endpoint_incidence_keys);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->endpoint_incidence_error);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->endpoint_incidence_counts);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->incremental_replacement_source_counts_by_candidate);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->incremental_replacement_source_offsets_by_candidate);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->incremental_dirty_atoms);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->incremental_dirty_clusters);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->incremental_dirty_superclusters);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->incremental_dirty_i_candidate_sci);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &workspace->incremental_dirty_candidate_sci);

    workspace->candidate_leaf_onepass_record_capacity = 0;
    workspace->candidate_leaf_onepass_high_water = 0;
    workspace->candidate_leaf_onepass_overflow_count = 0;
    workspace->candidate_leaf_queue2_task_capacity = 0;
}
}  // namespace

void Ensure_Cornerstone_State(ClusteredNeighborProvider* layout)
{
    if (ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state == NULL)
    {
        ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state = new ClusteredTreeState();
    }
}

void ClusteredNeighborProvider::Clear()
{
    if (!initialized_)
    {
        return;
    }
#ifndef USE_CPU
    if (controller_ != NULL)
    {
        working_device_ = controller_->working_device;
    }
    clustered_neighbor_runtime::Bind_Clustered_Working_Device(&working_device_);
#endif
    RetireSpatialProviderLifetime();
    Invalidate_Gmxpacked_Pair_Shift_Metadata(this);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &spatial_.ordering.sort_permutation);
    clustered_neighbor_runtime::Release_Device_Buffer(&spatial_.ordering.cached_crd);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &pair_list_.gmxpacked_inner_active_anchor_crd);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &spatial_.molecules.global_atom_to_molecule);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &spatial_.molecules.local_atom_to_molecule);
    clustered_neighbor_runtime::Release_Device_Buffer(&spatial_.clusters.offsets);
    clustered_neighbor_runtime::Release_Device_Buffer(&spatial_.clusters.valid_masks);
    clustered_neighbor_runtime::Release_Device_Buffer(&spatial_.clusters.local_masks);
    clustered_neighbor_runtime::Release_Device_Buffer(&spatial_.clusters.centers);
    clustered_neighbor_runtime::Release_Device_Buffer(&spatial_.clusters.extents);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &spatial_.clusters.fractional_centers);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &spatial_.clusters.fractional_extents);
    clustered_neighbor_runtime::Release_Device_Buffer(&spatial_.clusters.radii);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &spatial_.molecules.cluster_signatures);
    clustered_neighbor_runtime::Release_Device_Buffer(&spatial_.molecules.cluster_ids);
    clustered_neighbor_runtime::Release_Device_Buffer(&spatial_.leaves.atom_offsets);
    clustered_neighbor_runtime::Release_Device_Buffer(&spatial_.leaves.cluster_starts);
    clustered_neighbor_runtime::Release_Device_Buffer(&spatial_.leaves.cluster_ends);
    clustered_neighbor_runtime::Release_Device_Buffer(&spatial_.superclusters.offsets);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &spatial_.superclusters.cluster_to_supercluster);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &spatial_.superclusters.has_local);
    clustered_neighbor_runtime::Release_Device_Buffer(&spatial_.superclusters.centers);
    clustered_neighbor_runtime::Release_Device_Buffer(&spatial_.superclusters.sizes);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &spatial_.candidates.sci_supercluster_ids);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &spatial_.candidates.leaf_counts);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &spatial_.candidates.leaf_offsets);
    clustered_neighbor_runtime::Release_Device_Buffer(&spatial_.candidates.leaf_ids);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &spatial_.candidates.leaf_reach_masks);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &pair_list_.gmxpacked_endpoint_incidence_offsets);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &pair_list_.gmxpacked_endpoint_incidence_references);
    FreeGmxpackedPayload(this);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &pair_list_.gmxpacked_incremental_source_offsets_by_candidate);
    clustered_neighbor_runtime::Release_Device_Buffer(
        &pair_list_.gmxpacked_incremental_record_stream_sources);
    clustered_neighbor_runtime::Release_Device_Buffer(&pair_list_.pair_shift_bits);
    ReleaseBuildWorkspace(&workspace_);

    delete spatial_.tree.cornerstone_state;
    spatial_.tree.cornerstone_state = NULL;

    domain_ = {};
    spatial_.total_atom_numbers = 0;
    spatial_.padded_total_atom_numbers = 0;
    spatial_.cluster_numbers = 0;
    spatial_.super_cluster_numbers = 0;
    spatial_.local_cluster_numbers = 0;
    spatial_.leaf_numbers = 0;
    spatial_.candidates.candidate_sci_numbers = 0;
    spatial_.candidates.sci_numbers = 0;
    pair_list_.gmxpacked_sci_numbers = 0;
    pair_list_.gmxpacked_cjpacked_numbers = 0;
    pair_list_.gmxpacked_exclusion_numbers = 0;
    pair_list_.gmxpacked_split_exclusion_numbers = 0;
    pair_list_.gmxpacked_record_stream_source_numbers = 0;
    pair_list_.gmxpacked_record_stream_aggregate_numbers = 0;
    pair_list_.gmxpacked_inner_active_guard_cutoff = -1.0f;
    pair_list_.gmxpacked_incremental_source_cutoff = -1.0f;
    pair_list_.gmxpacked_incremental_source_numbers = 0;
    pair_list_.gmxpacked_incremental_candidate_sci_numbers = 0;
    pair_list_.gmxpacked_current_source_anchor_generation = 0;
    pair_list_.gmxpacked_current_source_generation = 0;
    pair_list_.gmxpacked_incremental_source_anchor_generation = -1;
    pair_list_.gmxpacked_incremental_source_generation = -1;
    pair_list_.gmxpacked_pair_shift_metadata_ready = false;
    pair_list_.gmxpacked_pair_shift_metadata_sci_numbers = 0;
    pair_list_.gmxpacked_pair_shift_metadata_cjpacked_numbers = 0;
    pair_list_.gmxpacked_pair_shift_metadata_exclusion_numbers = 0;
    pair_list_.gmxpacked_pair_shift_metadata_payload_generation = -1;
    pair_list_.gmxpacked_pair_shift_metadata_geometry_generation = -1;
    pair_list_.gmxpacked_pair_shift_metadata_rcell = {};
    spatial_.candidates.leaf_numbers = 0;
    spatial_.candidates.leaf_cluster_stride = 0;
    spatial_.leaves.max_cluster_span = 0;
    spatial_.leaves.periodic_image_max_fractional_extent_bound = 0.0f;
    Invalidate_Gmxpacked_Endpoint_Incidence(this);
    pair_list_.gmxpacked_incremental_source_offsets_ready = false;
    pair_list_.gmxpacked_incremental_source_cache_ready = false;
    pair_list_.gmxpacked_inner_active_anchor_ready = false;
    pair_list_.gmxpacked_inner_active_source_imasks_ready = false;
    pair_list_.gmxpacked_inner_active_source_imask_numbers = 0;
    pair_list_.gmxpacked_inner_active_source_rows_baseline = 0;
    rebuild_dirty_ = true;
    spatial_.ready = false;
    cached_cutoff_ = -1.0f;
    config_ = {};
    effective_rebuild_skin_ = config_.rebuild_skin;
    initialized_ = false;
}
