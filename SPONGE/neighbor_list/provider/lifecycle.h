#pragma once

#include "provider.h"

// Internal publication and invalidation transitions for clustered spatial
// view state. Device allocation, copies, kernels and launch policy remain in
// the owning builder/gather modules.
void Invalidate_Gmxpacked_Incremental_Source_Cache_State(
    ClusteredNeighborProvider* layout);
void Invalidate_Gmxpacked_Pair_Shift_Metadata(ClusteredNeighborProvider* layout);
void Publish_Gathered_Cluster_Geometry(ClusteredNeighborProvider* layout);
void Publish_Gmxpacked_Compact_Payload(ClusteredNeighborProvider* layout);
void Clear_Gmxpacked_Compact_Payload(ClusteredNeighborProvider* layout);
void Reset_Gmxpacked_Payload(ClusteredNeighborProvider* layout);
void Ensure_Cornerstone_State(ClusteredNeighborProvider* layout);
