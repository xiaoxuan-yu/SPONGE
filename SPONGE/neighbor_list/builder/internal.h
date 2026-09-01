#pragma once

#include <cstdint>

#include "../contract/types.h"

class ClusteredNeighborProvider;

static __host__ __device__ __forceinline__ int IntMin(int a, int b)
{
    return a < b ? a : b;
}

static __host__ __device__ __forceinline__ int IntMax(int a, int b)
{
    return a > b ? a : b;
}

// Narrow host-side services shared by clustered builder translation units.
// These are internal implementation details, not part of the public clustered
// neighbor-list API.
namespace clustered_neighbor_builder_internal
{
enum class BuildPreparationResult
{
    ready,
    empty
};

struct SpatialGeometryBuildResult
{
    BuildPreparationResult status = BuildPreparationResult::empty;
};

struct CandidateBuildResult
{
    BuildPreparationResult status = BuildPreparationResult::empty;
    bool candidate_leaf_queue2_payload_ready = false;
};

struct CompactPayloadSummary
{
    int source_rows = 0;
    int aggregate_rows = 0;
    int compact_entries = 0;
    int compact_sci = 0;
    int compact_cj = 0;
    int split_excl = 0;
    int compact_excl = 0;
};

BuildPreparationResult PrepareBuildState(ClusteredNeighborProvider* provider,
                                         LTMatrix3 reciprocal_cell,
                                         float cutoff);

SpatialGeometryBuildResult BuildSpatialGeometry(
    ClusteredNeighborProvider* provider, const VECTOR* coordinates,
    const VECTOR* geometry_coordinates, LTMatrix3 cell,
    LTMatrix3 reciprocal_cell, float build_cutoff);

CandidateBuildResult BuildCandidateStage(
    ClusteredNeighborProvider* provider, const VECTOR* commit_cache_coordinates,
    LTMatrix3 cell, float cutoff, float build_cutoff);

int ExclusiveScanCounts(ClusteredNeighborProvider* provider, int count,
                        int* counts, int* offsets);

void StableSortU64Pair(ClusteredNeighborProvider* provider, int count,
                       uint64_t* keys, uint64_t* values);

void CommitBuildCache(ClusteredNeighborProvider* provider,
                      const VECTOR* coordinates, float cutoff);

bool BuildIncrementalDirtyMask(
    ClusteredNeighborProvider* provider, const VECTOR* coordinates,
    LTMatrix3 cell, LTMatrix3 reciprocal_cell,
    const int* candidate_sci_supercluster_ids, bool* j_leaf_scope_ready);

int RefreshActiveViewSources(
    ClusteredNeighborProvider* provider, int outer_source_rows,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* outer_sources,
    int candidate_sci_numbers, const int* dirty_candidate_sci,
    const int* source_offsets_by_candidate, const VECTOR* coordinates,
    LTMatrix3 cell, LTMatrix3 reciprocal_cell, float cutoff,
    const VECTOR* active_anchor_coordinates, float active_reuse_margin,
    int* dirty_source_rows, int* changed_source_rows,
    int zero_dirty_reuse_source_rows, bool allow_full_dirty_refresh);

int PruneInnerActiveSources(ClusteredNeighborProvider* provider,
                            const VECTOR* coordinates, LTMatrix3 cell,
                            LTMatrix3 reciprocal_cell, float cutoff,
                            bool record_active_imasks);

int BuildRecordStreamAggregates(ClusteredNeighborProvider* provider);
CompactPayloadSummary BuildCompactPayload(ClusteredNeighborProvider* provider);

#ifndef USE_CPU
void StableSortU64Int(ClusteredNeighborProvider* provider, int count,
                      uint64_t* keys, int* values);

void CheckCudaStatus(cudaError_t error, const char* tag);

void StableSortU64EndpointReferences(
    ClusteredNeighborProvider* layout, int count, uint64_t* keys,
    CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE* values);
#endif
} // namespace clustered_neighbor_builder_internal
