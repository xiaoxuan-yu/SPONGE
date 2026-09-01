#include "active_refresh.h"
#include "internal.h"

#include "../provider/internal.h"
#include "../provider/lifecycle.h"
#include "../provider/pair_shift.h"
#include "../provider/runtime.h"

#include <cmath>
#include <cstddef>

namespace clustered_neighbor_builder_active_refresh
{
namespace
{
constexpr float kRecordBuilderInnerActiveGuardMargin = 2.0f;
constexpr float kActiveViewRefreshFraction = 0.4f;
constexpr float kActiveViewPayloadSafetyMargin = 0.25f;

float ActiveViewPayloadGuardMargin(float guard_margin)
{
    if (guard_margin <= 0.0f)
    {
        return 0.0f;
    }
    return fmaxf(0.0f, fminf(guard_margin,
                             2.0f * guard_margin *
                                     kActiveViewRefreshFraction +
                                 kActiveViewPayloadSafetyMargin));
}

void RefreshInnerActiveAnchor(ClusteredNeighborProvider* provider,
                              const VECTOR* coordinates)
{
    if (provider == nullptr || coordinates == nullptr ||
        ClusteredNeighborProviderInternal::Spatial(provider)
                .total_atom_numbers <= 0)
    {
        return;
    }
#ifndef USE_CPU
    auto& pair_list = ClusteredNeighborProviderInternal::PairList(provider);
    const int atom_count =
        ClusteredNeighborProviderInternal::Spatial(provider).total_atom_numbers;
    clustered_neighbor_runtime::Reserve_Device_Buffer(
        atom_count, &pair_list.gmxpacked_inner_active_anchor_crd);
    deviceMemcpy(pair_list.gmxpacked_inner_active_anchor_crd.data, coordinates,
                 sizeof(VECTOR) * atom_count, deviceMemcpyDeviceToDevice);
#endif
    ClusteredNeighborProviderInternal::PairList(provider)
        .gmxpacked_inner_active_anchor_ready = true;
}
}

Snapshot CaptureSnapshot(const ClusteredNeighborProvider* provider)
{
    const auto& pair_list = ClusteredNeighborProviderInternal::PairList(provider);
    Snapshot snapshot;
    snapshot.incremental_record_source_cache_ready =
        pair_list.gmxpacked_incremental_source_cache_ready &&
        pair_list.gmxpacked_incremental_record_stream_sources.data != nullptr;
    snapshot.incremental_source_cache_ready =
        pair_list.gmxpacked_incremental_source_offsets_ready &&
        snapshot.incremental_record_source_cache_ready &&
        pair_list.gmxpacked_incremental_source_offsets_by_candidate.data != nullptr;
    snapshot.incremental_candidate_sci_numbers =
        pair_list.gmxpacked_incremental_candidate_sci_numbers;
    snapshot.incremental_source_numbers =
        pair_list.gmxpacked_incremental_source_numbers;
    snapshot.compact_sci_numbers = pair_list.gmxpacked_sci_numbers;
    snapshot.compact_cjpacked_numbers = pair_list.gmxpacked_cjpacked_numbers;
    snapshot.compact_exclusion_numbers = pair_list.gmxpacked_exclusion_numbers;
    snapshot.compact_split_exclusion_numbers =
        pair_list.gmxpacked_split_exclusion_numbers;
    snapshot.compact_payload_ready =
        snapshot.compact_sci_numbers > 0 &&
        snapshot.compact_cjpacked_numbers > 0 &&
        snapshot.compact_exclusion_numbers > 0 &&
        pair_list.gmxpacked_sci.data != nullptr &&
        pair_list.gmxpacked_cjpacked.data != nullptr &&
        pair_list.gmxpacked_exclusions.data != nullptr;
    return snapshot;
}

bool TryRefreshInnerPayload(
    ClusteredNeighborProvider* provider, const Snapshot& snapshot,
    const VECTOR* coordinates, LTMatrix3 cell, LTMatrix3 reciprocal_cell,
    float cutoff, bool* source_cache_coverage_miss)
{
    auto& spatial = ClusteredNeighborProviderInternal::Spatial(provider);
    auto& pair_list = ClusteredNeighborProviderInternal::PairList(provider);
    auto& workspace = ClusteredNeighborProviderInternal::Workspace(provider);
    const auto& config = ClusteredNeighborProviderInternal::Config(provider);
    const float rebuild_skin =
        ClusteredNeighborProviderInternal::EffectiveRebuildSkin(provider);

    const float anchor_max_displacement = 0.0f;
    const float anchor_current_permit =
        (rebuild_skin > 0.0f && config.skin_permit > 0.0f)
            ? rebuild_skin * config.skin_permit
            : -1.0f;
    const bool anchor_current =
        anchor_current_permit >= 0.0f &&
        anchor_max_displacement <= anchor_current_permit + 1e-5f;
    const float outer_source_cutoff = cutoff + rebuild_skin;
    const float source_coverage_cutoff =
        fmaxf(cutoff, cutoff + 2.0f * anchor_max_displacement);
    const float requested_active_cutoff =
        fmaxf(cutoff, cutoff + 2.0f * anchor_max_displacement);
    const float required_guard_cutoff =
        fminf(outer_source_cutoff, requested_active_cutoff + 1e-4f);
    const bool source_covers_requested_active_cutoff =
        source_coverage_cutoff <= outer_source_cutoff + 1e-5f;
    if (!anchor_current && !source_covers_requested_active_cutoff)
    {
        if (source_cache_coverage_miss != nullptr)
        {
            *source_cache_coverage_miss = true;
        }
        return false;
    }
    const float target_guard_cutoff =
        fminf(outer_source_cutoff,
              required_guard_cutoff + kRecordBuilderInnerActiveGuardMargin);
    if (snapshot.compact_payload_ready &&
        pair_list.gmxpacked_inner_active_guard_cutoff + 1e-6f >=
            required_guard_cutoff)
    {
        Refresh_Gmxpacked_Pair_Shift_Metadata_If_Needed(provider,
                                                        reciprocal_cell);
        return true;
    }

    const VECTOR* active_prune_coordinates = nullptr;
    bool used_active_view_refresh = false;
    int active_view_dirty_source_rows = -1;
    int active_view_changed_source_rows = -1;
    int active_source_rows = -1;
    const int previous_active_source_rows =
        pair_list.gmxpacked_record_stream_source_numbers;
    const int previous_active_aggregate_rows =
        pair_list.gmxpacked_record_stream_aggregate_numbers;
    const bool can_reuse_previous_compact_payload =
        snapshot.compact_payload_ready && previous_active_source_rows > 0 &&
        previous_active_aggregate_rows > 0;
    const bool active_view_cutoff_exact_matches =
        pair_list.gmxpacked_inner_active_guard_cutoff > 0.0f &&
        fabsf(pair_list.gmxpacked_inner_active_guard_cutoff -
              target_guard_cutoff) <= 1e-6f;
    if (active_view_cutoff_exact_matches &&
        pair_list.gmxpacked_incremental_source_offsets_by_candidate.data !=
            nullptr &&
        pair_list.gmxpacked_inner_active_source_imasks_ready &&
        pair_list.gmxpacked_inner_active_source_imask_numbers ==
            snapshot.incremental_source_numbers &&
        pair_list.gmxpacked_inner_active_source_imasks.data != nullptr)
    {
        const int* cached_candidate_sci_supercluster_ids =
            spatial.candidates.sci_supercluster_ids.data;
        bool active_view_j_leaf_scope_ready = false;
        const bool dirty_mask_ready =
            clustered_neighbor_builder_internal::BuildIncrementalDirtyMask(
                provider, coordinates, cell, reciprocal_cell,
                cached_candidate_sci_supercluster_ids,
                &active_view_j_leaf_scope_ready);
        if (dirty_mask_ready &&
            snapshot.incremental_candidate_sci_numbers ==
                spatial.candidates.candidate_sci_numbers &&
            workspace.incremental_dirty_candidate_sci.data != nullptr)
        {
            active_source_rows =
                clustered_neighbor_builder_internal::RefreshActiveViewSources(
                    provider, snapshot.incremental_source_numbers,
                    pair_list.gmxpacked_incremental_record_stream_sources.data,
                    snapshot.incremental_candidate_sci_numbers,
                    workspace.incremental_dirty_candidate_sci.data,
                    pair_list.gmxpacked_incremental_source_offsets_by_candidate
                        .data,
                    active_prune_coordinates, cell, reciprocal_cell,
                    target_guard_cutoff, nullptr,
                    fmaxf(0.0f, target_guard_cutoff - cutoff),
                    &active_view_dirty_source_rows,
                    &active_view_changed_source_rows,
                    pair_list.gmxpacked_record_stream_source_numbers, false);
            used_active_view_refresh = active_source_rows >= 0;
        }
    }
    if (!used_active_view_refresh)
    {
        return false;
    }

    int filled_aggregate_rows = 0;
    const bool reused_compact_payload =
        active_view_changed_source_rows == 0 &&
        active_source_rows == previous_active_source_rows &&
        can_reuse_previous_compact_payload &&
        pair_list.gmxpacked_sci_numbers == snapshot.compact_sci_numbers &&
        pair_list.gmxpacked_cjpacked_numbers ==
            snapshot.compact_cjpacked_numbers &&
        pair_list.gmxpacked_exclusion_numbers ==
            snapshot.compact_exclusion_numbers &&
        pair_list.gmxpacked_split_exclusion_numbers ==
            snapshot.compact_split_exclusion_numbers;
    if (!reused_compact_payload)
    {
        Clear_Gmxpacked_Compact_Payload(provider);
        pair_list.gmxpacked_record_stream_aggregate_numbers = 0;
    }
    if (reused_compact_payload)
    {
        filled_aggregate_rows = previous_active_aggregate_rows;
        pair_list.gmxpacked_record_stream_aggregate_numbers =
            previous_active_aggregate_rows;
    }
    else if (active_source_rows > 0)
    {
        filled_aggregate_rows =
            clustered_neighbor_builder_internal::BuildRecordStreamAggregates(
                provider);
    }
    clustered_neighbor_builder_internal::CompactPayloadSummary compact_summary;
    bool rebuilt_full_compact_payload = false;
    if (reused_compact_payload)
    {
        compact_summary.source_rows = active_source_rows;
        compact_summary.aggregate_rows = previous_active_aggregate_rows;
        compact_summary.compact_sci = pair_list.gmxpacked_sci_numbers;
        compact_summary.compact_cj = pair_list.gmxpacked_cjpacked_numbers;
        compact_summary.compact_excl = pair_list.gmxpacked_exclusion_numbers;
        compact_summary.split_excl =
            pair_list.gmxpacked_split_exclusion_numbers;
    }
    else if (active_source_rows > 0 &&
             pair_list.gmxpacked_record_stream_aggregate_numbers > 0 &&
             filled_aggregate_rows ==
                 pair_list.gmxpacked_record_stream_aggregate_numbers)
    {
        compact_summary =
            clustered_neighbor_builder_internal::BuildCompactPayload(provider);
        rebuilt_full_compact_payload = compact_summary.compact_sci > 0 &&
                                       compact_summary.compact_cj > 0 &&
                                       compact_summary.compact_excl > 0;
    }
    const bool compact_ready =
        pair_list.gmxpacked_sci_numbers > 0 &&
        pair_list.gmxpacked_cjpacked_numbers > 0 &&
        pair_list.gmxpacked_exclusion_numbers > 0 &&
        pair_list.gmxpacked_sci.data != nullptr &&
        pair_list.gmxpacked_cjpacked.data != nullptr &&
        pair_list.gmxpacked_exclusions.data != nullptr;
    if (!compact_ready)
    {
        return false;
    }
    if (rebuilt_full_compact_payload)
    {
        pair_list.gmxpacked_inner_active_source_rows_baseline =
            active_source_rows;
    }
    pair_list.gmxpacked_inner_active_guard_cutoff = target_guard_cutoff;
    Refresh_Gmxpacked_Pair_Shift_Metadata_If_Needed(provider, reciprocal_cell);
    return true;
}

bool RebuildPayloadFromOuterSource(
    ClusteredNeighborProvider* provider, const Snapshot& snapshot,
    const VECTOR* coordinates, LTMatrix3 cell, LTMatrix3 reciprocal_cell,
    float cutoff)
{
    auto& pair_list = ClusteredNeighborProviderInternal::PairList(provider);
    if (!snapshot.incremental_record_source_cache_ready ||
        snapshot.incremental_source_numbers <= 0 ||
        pair_list.gmxpacked_incremental_record_stream_sources.data == nullptr)
    {
        return false;
    }
#ifdef USE_CPU
    (void)coordinates;
    (void)cell;
    (void)reciprocal_cell;
    (void)cutoff;
    return false;
#else
    const float rebuild_skin =
        ClusteredNeighborProviderInternal::EffectiveRebuildSkin(provider);
    const int outer_source_rows = snapshot.incremental_source_numbers;
    const float outer_source_cutoff = cutoff + rebuild_skin;
    const float active_payload_guard_margin =
        ActiveViewPayloadGuardMargin(kRecordBuilderInnerActiveGuardMargin);
    const float active_cutoff =
        fminf(outer_source_cutoff,
              cutoff + fmaxf(0.0f, active_payload_guard_margin));
    clustered_neighbor_runtime::Reserve_Device_Buffer(
        outer_source_rows, &pair_list.gmxpacked_record_stream_sources);
    deviceMemcpy(
        pair_list.gmxpacked_record_stream_sources.data,
        pair_list.gmxpacked_incremental_record_stream_sources.data,
        sizeof(CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE) *
            static_cast<size_t>(outer_source_rows),
        deviceMemcpyDeviceToDevice);
    pair_list.gmxpacked_record_stream_source_numbers = outer_source_rows;
    pair_list.gmxpacked_record_stream_aggregate_numbers = 0;
    const int active_source_rows =
        clustered_neighbor_builder_internal::PruneInnerActiveSources(
            provider, coordinates, cell, reciprocal_cell, active_cutoff, true);
    if (active_source_rows <= 0)
    {
        return false;
    }
    pair_list.gmxpacked_record_stream_source_numbers = active_source_rows;
    const int filled_aggregate_rows =
        clustered_neighbor_builder_internal::BuildRecordStreamAggregates(
            provider);
    if (pair_list.gmxpacked_record_stream_aggregate_numbers <= 0 ||
        filled_aggregate_rows !=
            pair_list.gmxpacked_record_stream_aggregate_numbers)
    {
        return false;
    }
    const auto compact_summary =
        clustered_neighbor_builder_internal::BuildCompactPayload(provider);
    if (compact_summary.compact_sci <= 0 || compact_summary.compact_cj <= 0 ||
        compact_summary.compact_excl <= 0)
    {
        return false;
    }
    RefreshInnerActiveAnchor(provider, coordinates);
    pair_list.gmxpacked_inner_active_guard_cutoff = active_cutoff;
    Refresh_Gmxpacked_Pair_Shift_Metadata_If_Needed(provider, reciprocal_cell);
    return true;
#endif
}
}  // namespace clustered_neighbor_builder_active_refresh
