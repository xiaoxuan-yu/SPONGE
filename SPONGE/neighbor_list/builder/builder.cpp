#include "active_refresh.h"
#include "internal.h"
#include "payload_builder.h"
#include "../provider/endpoint_incidence.h"
#include "../provider/internal.h"
#include "../provider/lifecycle.h"
#include "../provider/pair_shift.h"
#include "../provider/runtime.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../MD_core/MD_core.h"
#include "../../third_party/cornerstone_octree/include/cstone/sfc/box.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/sfc/common.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/sfc/sfc.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/traversal/boxoverlap.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/traversal/traversal.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/tree/csarray.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/tree/octree.hpp"

#ifndef USE_CPU
#include <cub/cub.cuh>

#include "../../third_party/cornerstone_octree/include/cstone/cuda/device_vector.h"
#include "../../third_party/cornerstone_octree/include/cstone/execution.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/primitives/primitives_gpu.h"
#include "../../third_party/cornerstone_octree/include/cstone/tree/octree_gpu.h"
#include "../../third_party/cornerstone_octree/include/cstone/tree/update_gpu.cuh"
#endif

extern MD_INFORMATION md_info;

namespace
{

using CornerstoneKey = uint64_t;
using CornerstoneNodeIndex = cstone::TreeNodeIndex;
using clustered_neighbor_runtime::Clustered_Device_Malloc_Safely;
using clustered_neighbor_runtime::Clustered_Minimum_Box_Face_Height;
using clustered_neighbor_runtime::Note_Clustered_Step_Counter;
using clustered_neighbor_runtime::RecorderScope;
using clustered_neighbor_runtime::Reserve_Device_Buffer;
using cstone::rawPtr;
#ifndef USE_CPU
using clustered_neighbor_runtime::Bind_Clustered_Working_Device;
#endif
constexpr int kClusteredMaxSuperClusterClusters =
    kClusteredSuperClusterClusters;
constexpr float kActiveViewDirtySourceRatioLimit = 0.25f;
constexpr bool kActiveViewDirtyIndexRefresh = true;
constexpr float kRecordBuilderInnerActiveGuardMargin = 2.0f;

#ifndef USE_CPU
static void Clustered_Check_Cuda_Status(cudaError_t error, const char* tag)
{
    if (error == cudaSuccess)
    {
        return;
    }
    int current_device = -1;
    cudaGetDevice(&current_device);
    fprintf(stderr, "clustered cuda failure: tag=%s device=%d error=%s\n", tag,
            current_device, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
}

template <typename Key, typename Value>
static void Stable_Sort_Device_By_Key(ClusteredNeighborProvider* layout, int count,
                                      Key* d_keys, Value* d_values)
{
    if (count <= 1 || d_keys == NULL || d_values == NULL)
    {
        return;
    }
    static_assert(std::is_trivially_copyable_v<Key>,
                  "Clustered device sort requires trivially copyable keys.");
    static_assert(std::is_trivially_copyable_v<Value>,
                  "Clustered device sort requires trivially copyable values.");
    if (layout == NULL)
    {
        return;
    }
    Bind_Clustered_Working_Device(&ClusteredNeighborProviderInternal::WorkingDevice(layout));

    (void)cudaGetLastError();
    size_t temp_storage_bytes = 0;

    cub::DoubleBuffer<Key> key_buffers(d_keys, NULL);
    cub::DoubleBuffer<Value> value_buffers(d_values, NULL);
    const cudaError_t query_err = cub::DeviceRadixSort::SortPairs(
        NULL, temp_storage_bytes, key_buffers, value_buffers, count);
    Clustered_Check_Cuda_Status(query_err, "clustered-radix-sort-query");

    clustered_neighbor_runtime::Reserve_Raw_Device_Workspace(
        sizeof(Key) * static_cast<size_t>(count), &ClusteredNeighborProviderInternal::Workspace(layout).sort_keys);
    clustered_neighbor_runtime::Reserve_Raw_Device_Workspace(
        sizeof(Value) * static_cast<size_t>(count),
        &ClusteredNeighborProviderInternal::Workspace(layout).sort_values);
    clustered_neighbor_runtime::Reserve_Raw_Device_Workspace(temp_storage_bytes,
                                                       &ClusteredNeighborProviderInternal::Workspace(layout).sort);

    key_buffers = cub::DoubleBuffer<Key>(
        d_keys, reinterpret_cast<Key*>(ClusteredNeighborProviderInternal::Workspace(layout).sort_keys.data));
    value_buffers = cub::DoubleBuffer<Value>(
        d_values, reinterpret_cast<Value*>(ClusteredNeighborProviderInternal::Workspace(layout).sort_values.data));
    const cudaError_t sort_err = cub::DeviceRadixSort::SortPairs(
        ClusteredNeighborProviderInternal::Workspace(layout).sort.data, temp_storage_bytes, key_buffers,
        value_buffers, count);
    Clustered_Check_Cuda_Status(sort_err, "clustered-radix-sort-run");
    const cudaError_t sync_err = cudaDeviceSynchronize();
    Clustered_Check_Cuda_Status(sync_err, "clustered-radix-sort-sync");

    if (key_buffers.Current() != d_keys)
    {
        deviceMemcpy(d_keys, key_buffers.Current(), sizeof(Key) * count,
                     deviceMemcpyDeviceToDevice);
    }
    if (value_buffers.Current() != d_values)
    {
        deviceMemcpy(d_values, value_buffers.Current(), sizeof(Value) * count,
                     deviceMemcpyDeviceToDevice);
    }
}

#else
template <typename Key, typename Value>
static void Stable_Sort_Device_By_Key(ClusteredNeighborProvider*, int count, Key* keys,
                                      Value* values)
{
    if (count <= 1 || keys == NULL || values == NULL)
    {
        return;
    }
    std::vector<std::pair<Key, Value> > entries(static_cast<size_t>(count));
    for (int i = 0; i < count; i += 1)
    {
        entries[static_cast<size_t>(i)] = {keys[i], values[i]};
    }
    std::stable_sort(entries.begin(), entries.end(),
                     [](const auto& lhs, const auto& rhs)
                     { return lhs.first < rhs.first; });
    for (int i = 0; i < count; i += 1)
    {
        keys[i] = entries[static_cast<size_t>(i)].first;
        values[i] = entries[static_cast<size_t>(i)].second;
    }
}
#endif

#include "primitives/geometry.cuh"
#include "detail/pair_mask_primitives.inc.cuh"
#include "detail/incremental_kernels.inc.cuh"
#include "detail/active_view_kernels.inc.cuh"

#include "detail/storage.inc"

}  // namespace

void ClusteredNeighborProvider::BuildInternal(const VECTOR* crd, LTMatrix3 cell,
                              LTMatrix3 rcell, float cutoff,
                              bool need_endpoint_incidence)
{
    auto& spatial = spatial_;
    auto& pair_list = pair_list_;
    const uint64_t provider_incarnation = provider_incarnation_;
    const float cached_cutoff = cached_cutoff_;
    const float rebuild_skin = effective_rebuild_skin_;
    if (clustered_neighbor_builder_internal::PrepareBuildState(
            this, rcell, cutoff) ==
        clustered_neighbor_builder_internal::BuildPreparationResult::empty)
    {
        return;
    }

    const auto active_refresh_snapshot =
        clustered_neighbor_builder_active_refresh::CaptureSnapshot(this);
    const bool clustered_build_needed =
        Clustered_Build_Is_Needed(this, crd, cell, rcell, cutoff);
#ifdef USE_CPU
    const bool requested_endpoint_incidence_ready =
        !need_endpoint_incidence ||
        (pair_list.gmxpacked_endpoint_incidence_ready &&
         pair_list.gmxpacked_endpoint_incidence_provider_incarnation ==
             provider_incarnation &&
         pair_list.gmxpacked_endpoint_incidence_payload_generation ==
             pair_list.gmxpacked_compact_payload_generation);
    if (!clustered_build_needed && active_refresh_snapshot.compact_payload_ready &&
        requested_endpoint_incidence_ready)
    {
        return;
    }
#endif
    bool active_view_source_cache_coverage_miss = false;
    if (!clustered_build_needed)
    {
        const bool can_attempt_inner_active_from_cached_outer_source =
            active_refresh_snapshot.incremental_source_numbers > 0 &&
            active_refresh_snapshot.incremental_source_cache_ready &&
            fabsf(cached_cutoff - cutoff) <= 1e-6f;
        if (can_attempt_inner_active_from_cached_outer_source &&
            clustered_neighbor_builder_active_refresh::TryRefreshInnerPayload(
                this, active_refresh_snapshot, crd, cell, rcell, cutoff,
                &active_view_source_cache_coverage_miss))
        {
            return;
        }
        if (!active_view_source_cache_coverage_miss)
        {
            bool can_reuse_active_payload =
                active_refresh_snapshot.compact_payload_ready &&
                pair_list.gmxpacked_inner_active_anchor_ready &&
                pair_list.gmxpacked_inner_active_anchor_crd.data != NULL &&
                pair_list.gmxpacked_inner_active_source_imasks_ready &&
                pair_list.gmxpacked_inner_active_source_imask_numbers ==
                    active_refresh_snapshot.incremental_source_numbers;
            float active_anchor_max_displacement = 0.0f;
            const float guard_margin = kRecordBuilderInnerActiveGuardMargin;
            const float reuse_limit = fmaxf(0.0f, 0.5f * guard_margin);
            if (can_reuse_active_payload)
            {
                active_anchor_max_displacement =
                    Clustered_Max_Gmxpacked_Inner_Active_Anchor_Displacement(
                        this, crd, cell, rcell);
                can_reuse_active_payload =
                    active_anchor_max_displacement <= reuse_limit + 1.0e-5f;
            }
            if (can_reuse_active_payload)
            {
                Refresh_Gmxpacked_Pair_Shift_Metadata_If_Needed(this, rcell);
                return;
            }
            if (clustered_neighbor_builder_active_refresh::
                    RebuildPayloadFromOuterSource(
                        this, active_refresh_snapshot, crd, cell, rcell,
                        cutoff))
            {
                return;
            }
        }
        return;
    }
    Invalidate_Gmxpacked_Incremental_Source_Cache_State(this);
    Ensure_Cornerstone_State(this);
    Reset_Gmxpacked_Payload(this);
    spatial.local_cluster_numbers = 0;
    spatial.candidates.candidate_sci_numbers = 0;
    spatial.candidates.leaf_numbers = 0;
    float build_cutoff = rebuild_skin > 0.0f ? cutoff + rebuild_skin : cutoff;
    const VECTOR* builder_geometry_crd = crd;
    const VECTOR* commit_cache_crd = crd;
    const auto geometry_result =
        clustered_neighbor_builder_internal::BuildSpatialGeometry(
            this, crd, builder_geometry_crd, cell, rcell, build_cutoff);
    if (geometry_result.status ==
        clustered_neighbor_builder_internal::BuildPreparationResult::empty)
    {
        Commit_Clustered_Build_Cache(this, commit_cache_crd, cutoff);
        return;
    }
    const auto candidate_result =
        clustered_neighbor_builder_internal::BuildCandidateStage(
            this, commit_cache_crd, cell, cutoff, build_cutoff);
    if (candidate_result.status ==
        clustered_neighbor_builder_internal::BuildPreparationResult::empty)
    {
        return;
    }
#ifndef USE_CPU
    (void)Build_Gmxpacked_Incremental_Dirty_Device_Mask(
        this, crd, cell, rcell, spatial.candidates.sci_supercluster_ids.data,
        NULL);
#endif
    BuildPayloadInput payload_input;
    payload_input.crd = crd;
    payload_input.commit_cache_crd = commit_cache_crd;
    payload_input.cell = cell;
    payload_input.rcell = rcell;
    payload_input.cutoff = cutoff;
    payload_input.build_cutoff = build_cutoff;
    payload_input.candidate_leaf_queue2_payload_ready =
        candidate_result.candidate_leaf_queue2_payload_ready;
    BuildPayload(payload_input);
}

void ClusteredNeighborProvider::BuildPayload(const BuildPayloadInput& input)
{
#ifdef USE_CPU
    const bool payload_ready = BuildPayloadCpu(input);
    if (!payload_ready)
    {
        Commit_Clustered_Build_Cache(this, input.commit_cache_crd,
                                     input.cutoff);
        return;
    }
#else
    const bool payload_ready = BuildPayloadGpu(input);
    if (!payload_ready)
    {
        return;
    }
#endif
    Refresh_Gmxpacked_Pair_Shift_Metadata_If_Needed(this, input.rcell);
    Commit_Clustered_Build_Cache(this, input.commit_cache_crd, input.cutoff);
}


namespace clustered_neighbor_builder_internal
{

#ifndef USE_CPU
void StableSortU64Int(ClusteredNeighborProvider* layout, int count, uint64_t* keys,
                      int* values)
{
    Stable_Sort_Device_By_Key(layout, count, keys, values);
}

void StableSortU64EndpointReferences(
    ClusteredNeighborProvider* layout, int count, uint64_t* keys,
    CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE* values)
{
    Stable_Sort_Device_By_Key(layout, count, keys, values);
}

void CheckCudaStatus(cudaError_t error, const char* tag)
{
    Clustered_Check_Cuda_Status(error, tag);
}

#endif

void StableSortU64Pair(ClusteredNeighborProvider* layout, int count,
                       uint64_t* keys, uint64_t* values)
{
    Stable_Sort_Device_By_Key(layout, count, keys, values);
}

int ExclusiveScanCounts(ClusteredNeighborProvider* layout, int count_numbers,
                        int* counts, int* starts)
{
    return Exclusive_Scan_Counts(layout, count_numbers, counts, starts);
}

void CommitBuildCache(ClusteredNeighborProvider* layout, const VECTOR* crd,
                      float cutoff)
{
    Commit_Clustered_Build_Cache(layout, crd, cutoff);
}

bool BuildIncrementalDirtyMask(
    ClusteredNeighborProvider* layout, const VECTOR* crd, LTMatrix3 cell,
    LTMatrix3 rcell, const int* candidate_sci_supercluster_ids,
    bool* j_leaf_scope_ready)
{
    return Build_Gmxpacked_Incremental_Dirty_Device_Mask(
        layout, crd, cell, rcell, candidate_sci_supercluster_ids,
        j_leaf_scope_ready);
}

int RefreshActiveViewSources(
    ClusteredNeighborProvider* layout, int outer_source_rows,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* outer_sources,
    int candidate_sci_numbers, const int* dirty_candidate_sci,
    const int* source_offsets_by_candidate, const VECTOR* crd, LTMatrix3 cell,
    LTMatrix3 rcell, float cutoff, const VECTOR* active_anchor_crd,
    float active_reuse_margin, int* dirty_source_rows_out,
    int* changed_source_rows_out, int zero_dirty_reuse_source_rows,
    bool allow_full_dirty_refresh)
{
    return Refresh_Gmxpacked_Record_Stream_Active_View_Sources(
        layout, outer_source_rows, outer_sources, candidate_sci_numbers,
        dirty_candidate_sci, source_offsets_by_candidate, crd, cell, rcell,
        cutoff, active_anchor_crd, active_reuse_margin, dirty_source_rows_out,
        changed_source_rows_out, zero_dirty_reuse_source_rows,
        allow_full_dirty_refresh);
}

int PruneInnerActiveSources(ClusteredNeighborProvider* layout,
                            const VECTOR* crd, LTMatrix3 cell, LTMatrix3 rcell,
                            float cutoff, bool record_active_imasks)
{
    return Prune_Gmxpacked_Record_Stream_To_Inner_Active_Sources(
        layout, crd, cell, rcell, cutoff, record_active_imasks);
}

}  // namespace clustered_neighbor_builder_internal
