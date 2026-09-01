#pragma once

#include "../contract/types.h"

class ClusteredNeighborProvider;

namespace clustered_neighbor_builder_active_refresh
{
struct Snapshot
{
    bool incremental_record_source_cache_ready = false;
    bool incremental_source_cache_ready = false;
    int incremental_candidate_sci_numbers = 0;
    int incremental_source_numbers = 0;
    bool compact_payload_ready = false;
    int compact_sci_numbers = 0;
    int compact_cjpacked_numbers = 0;
    int compact_exclusion_numbers = 0;
    int compact_split_exclusion_numbers = 0;
};

Snapshot CaptureSnapshot(const ClusteredNeighborProvider* provider);

bool TryRefreshInnerPayload(
    ClusteredNeighborProvider* provider, const Snapshot& snapshot,
    const VECTOR* coordinates, LTMatrix3 cell, LTMatrix3 reciprocal_cell,
    float cutoff, bool* source_cache_coverage_miss);

bool RebuildPayloadFromOuterSource(
    ClusteredNeighborProvider* provider, const Snapshot& snapshot,
    const VECTOR* coordinates, LTMatrix3 cell, LTMatrix3 reciprocal_cell,
    float cutoff);
}  // namespace clustered_neighbor_builder_active_refresh
