#include "payload_builder.h"

#ifdef USE_CPU

#include "cpu_builder.h"
#include "internal.h"

#include "../provider/lifecycle.h"
#include "../provider/runtime.h"

#include <algorithm>
#include <cstddef>

[[gnu::noinline]] bool ClusteredNeighborProvider::BuildPayloadCpu(
    const BuildPayloadInput& input)
{
    auto& spatial = spatial_;
    auto& pair_list = pair_list_;
    const auto& config = config_;
    const auto& domain = domain_;
    const auto& ordering = spatial.ordering;
    const auto& clusters = spatial.clusters;
    const auto& leaves = spatial.leaves;
    const auto& superclusters = spatial.superclusters;
    const auto& candidates = spatial.candidates;
    const int total_atom_numbers = spatial.total_atom_numbers;
    const int cluster_numbers = spatial.cluster_numbers;
    const int super_cluster_numbers = spatial.super_cluster_numbers;
    const int candidate_leaf_numbers = candidates.leaf_numbers;
    const auto& sort_permutation = ordering.sort_permutation;
    const auto& cluster_offsets = clusters.offsets;
    const auto& cluster_valid_masks = clusters.valid_masks;
    const auto& cluster_local_masks = clusters.local_masks;
    const auto& cluster_centers = clusters.centers;
    const auto& cluster_extents = clusters.extents;
    const auto& leaf_cluster_starts = leaves.cluster_starts;
    const auto& leaf_cluster_ends = leaves.cluster_ends;
    const auto& super_cluster_offsets = superclusters.offsets;
    const auto& sci_candidate_leaf_offsets = candidates.leaf_offsets;
    const auto& sci_candidate_leaf_ids = candidates.leaf_ids;
    const int cluster_size = config.cluster_size;
    const int local_atom_numbers = domain.local_atom_count;
    const int* d_excluded_list_start = domain.excluded_list_start;
    const int* d_excluded_list = domain.excluded_list;
    const int* d_excluded_numbers = domain.excluded_numbers;
    const LTMatrix3 cell = input.cell;
    const float build_cutoff = input.build_cutoff;
    const int leaf_numbers = spatial.leaf_numbers;
    const int super_sci_numbers = candidates.sci_numbers;
    const int candidate_sci_numbers = candidates.candidate_sci_numbers;
    const int* candidate_sci_supercluster_ids =
        candidates.sci_supercluster_ids.data;
    clustered_neighbor_cpu_builder::BuildInput host_input = {};
    host_input.local_atom_numbers = local_atom_numbers;
    host_input.cluster_size = cluster_size;
    host_input.candidate_sci_numbers = candidate_sci_numbers;
    host_input.dense_shift_partitioned_candidates = true;
    host_input.cutoff = build_cutoff;
    host_input.cell = cell;
    host_input.permutation.resize(static_cast<size_t>(total_atom_numbers));
    host_input.cluster_offsets.resize(static_cast<size_t>(cluster_numbers) + 1);
    host_input.cluster_valid_masks.resize(static_cast<size_t>(cluster_numbers));
    host_input.cluster_local_masks.resize(static_cast<size_t>(cluster_numbers));
    host_input.cluster_centers.resize(static_cast<size_t>(cluster_numbers));
    host_input.cluster_extents.resize(static_cast<size_t>(cluster_numbers));
    host_input.leaf_cluster_starts.resize(static_cast<size_t>(leaf_numbers));
    host_input.leaf_cluster_ends.resize(static_cast<size_t>(leaf_numbers));
    host_input.super_cluster_offsets.resize(
        static_cast<size_t>(super_cluster_numbers) + 1);
    host_input.cluster_to_supercluster.resize(
        static_cast<size_t>(cluster_numbers), -1);
    const int candidate_super_id_numbers = super_sci_numbers;
    host_input.sci_supercluster_ids.resize(
        static_cast<size_t>(candidate_super_id_numbers));
    host_input.candidate_leaf_offsets.resize(
        static_cast<size_t>(candidate_sci_numbers) + 1);
    host_input.candidate_leaf_ids.resize(
        static_cast<size_t>(candidate_leaf_numbers));
    host_input.excluded_list_start.resize(
        static_cast<size_t>(local_atom_numbers), 0);
    host_input.excluded_numbers.resize(static_cast<size_t>(local_atom_numbers),
                                       0);

    deviceMemcpy(host_input.permutation.data(), sort_permutation.data,
                 sizeof(int) * total_atom_numbers, deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.cluster_offsets.data(), cluster_offsets.data,
                 sizeof(int) * (cluster_numbers + 1), deviceMemcpyDeviceToHost);
    deviceMemcpy(
        host_input.cluster_valid_masks.data(), cluster_valid_masks.data,
        sizeof(unsigned int) * cluster_numbers, deviceMemcpyDeviceToHost);
    deviceMemcpy(
        host_input.cluster_local_masks.data(), cluster_local_masks.data,
        sizeof(unsigned int) * cluster_numbers, deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.cluster_centers.data(), cluster_centers.data,
                 sizeof(VECTOR) * cluster_numbers, deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.cluster_extents.data(), cluster_extents.data,
                 sizeof(VECTOR) * cluster_numbers, deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.leaf_cluster_starts.data(),
                 leaf_cluster_starts.data, sizeof(int) * leaf_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.leaf_cluster_ends.data(), leaf_cluster_ends.data,
                 sizeof(int) * leaf_numbers, deviceMemcpyDeviceToHost);
    deviceMemcpy(
        host_input.super_cluster_offsets.data(), super_cluster_offsets.data,
        sizeof(int) * (super_cluster_numbers + 1), deviceMemcpyDeviceToHost);
    for (int super_i = 0; super_i < super_cluster_numbers; super_i += 1)
    {
        const int cluster_start =
            host_input.super_cluster_offsets[static_cast<size_t>(super_i)];
        const int cluster_end =
            host_input.super_cluster_offsets[static_cast<size_t>(super_i) + 1];
        for (int cluster_i = cluster_start; cluster_i < cluster_end;
             cluster_i += 1)
        {
            host_input.cluster_to_supercluster[static_cast<size_t>(cluster_i)] =
                super_i;
        }
    }
    deviceMemcpy(
        host_input.sci_supercluster_ids.data(), candidate_sci_supercluster_ids,
        sizeof(int) * candidate_super_id_numbers, deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.candidate_leaf_offsets.data(),
                 sci_candidate_leaf_offsets.data,
                 sizeof(int) * (candidate_sci_numbers + 1),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.candidate_leaf_ids.data(),
                 sci_candidate_leaf_ids.data,
                 sizeof(int) * candidate_leaf_numbers,
                 deviceMemcpyDeviceToHost);
    if (local_atom_numbers > 0 && d_excluded_list_start != NULL &&
        d_excluded_numbers != NULL)
    {
        deviceMemcpy(host_input.excluded_list_start.data(),
                     d_excluded_list_start, sizeof(int) * local_atom_numbers,
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(host_input.excluded_numbers.data(), d_excluded_numbers,
                     sizeof(int) * local_atom_numbers,
                     deviceMemcpyDeviceToHost);
        const int total_excluded = host_input.excluded_list_start.back() +
                                   host_input.excluded_numbers.back();
        host_input.excluded_list.resize(
            static_cast<size_t>(IntMax(total_excluded, 0)));
        if (total_excluded > 0 && d_excluded_list != NULL)
        {
            deviceMemcpy(host_input.excluded_list.data(), d_excluded_list,
                         sizeof(int) * total_excluded,
                         deviceMemcpyDeviceToHost);
        }
    }

    clustered_neighbor_cpu_builder::BuildOutput host_payload;
    clustered_neighbor_cpu_builder::BuildPayload(host_input, &host_payload);
    std::stable_sort(host_payload.gmxpacked_scis.begin(),
                     host_payload.gmxpacked_scis.end(),
                     [](const CLUSTERED_GMXPACKED_SCI& lhs,
                        const CLUSTERED_GMXPACKED_SCI& rhs)
                     {
                         return (lhs.cjpacked_end - lhs.cjpacked_begin) >
                                (rhs.cjpacked_end - rhs.cjpacked_begin);
                     });

    pair_list.gmxpacked_sci_numbers =
        static_cast<int>(host_payload.gmxpacked_scis.size());
    pair_list.gmxpacked_cjpacked_numbers =
        static_cast<int>(host_payload.gmxpacked_cjpacked.size());
    pair_list.gmxpacked_exclusion_numbers =
        static_cast<int>(host_payload.gmxpacked_exclusions.size());
    pair_list.gmxpacked_split_exclusion_numbers =
        host_payload.gmxpacked_split_exclusion_numbers;
    if (pair_list.gmxpacked_sci_numbers <= 0 ||
        pair_list.gmxpacked_cjpacked_numbers <= 0 ||
        pair_list.gmxpacked_exclusion_numbers <= 0)
    {
        return false;
    }

    clustered_neighbor_runtime::Reserve_Device_Buffer(
        pair_list.gmxpacked_sci_numbers, &pair_list.gmxpacked_sci);
    clustered_neighbor_runtime::Reserve_Device_Buffer(
        pair_list.gmxpacked_cjpacked_numbers, &pair_list.gmxpacked_cjpacked);
    clustered_neighbor_runtime::Reserve_Device_Buffer(
        pair_list.gmxpacked_exclusion_numbers,
        &pair_list.gmxpacked_exclusions);
    deviceMemcpy(pair_list.gmxpacked_sci.data,
                 host_payload.gmxpacked_scis.data(),
                 sizeof(CLUSTERED_GMXPACKED_SCI) *
                     pair_list.gmxpacked_sci_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(pair_list.gmxpacked_cjpacked.data,
                 host_payload.gmxpacked_cjpacked.data(),
                 sizeof(CLUSTERED_GMXPACKED_CJ) *
                     pair_list.gmxpacked_cjpacked_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(pair_list.gmxpacked_exclusions.data,
                 host_payload.gmxpacked_exclusions.data(),
                 sizeof(CLUSTERED_GMXPACKED_EXCLUSION) *
                     pair_list.gmxpacked_exclusion_numbers,
                 deviceMemcpyHostToDevice);
    Publish_Gmxpacked_Compact_Payload(this);
    return true;
}

#endif

#ifndef USE_CPU

#include "candidate_builder.h"
#include "internal.h"

#include "../provider/internal.h"
#include "../provider/lifecycle.h"
#include "../provider/runtime.h"

#include "../../MD_core/MD_core.h"
#include "../../third_party/cornerstone_octree/include/cstone/traversal/boxoverlap.hpp"

#include <cstdio>
#include <stdexcept>

extern MD_INFORMATION md_info;

namespace
{

using clustered_neighbor_runtime::Reserve_Device_Buffer;

constexpr int kClusteredBuilderBlockSize = 128;
constexpr int kClusteredBuilderWarpSize = 32;
constexpr int kClusteredMaxSuperClusterClusters =
    kClusteredSuperClusterClusters;
constexpr int kClusteredMaxJGroupSize = kClusteredJGroupSize;
constexpr float kRecordBuilderInnerActiveGuardMargin = 2.0f;

static int Estimate_Gmxpacked_Primary_Record_Stream_Source_Capacity(
    int candidate_sci_numbers, int candidate_leaf_numbers)
{
    if (candidate_sci_numbers <= 0 || candidate_leaf_numbers <= 0)
    {
        return 0;
    }
    const long long estimate =
        static_cast<long long>(candidate_leaf_numbers) / 8ll +
        static_cast<long long>(candidate_sci_numbers) * 16ll + 65536ll;
    const long long max_int = 0x7fffffffll;
    return static_cast<int>(estimate > max_int ? max_int : estimate);
}

static int Estimate_Gmxpacked_Primary_Fill_Prune_Reuse_Source_Capacity(
    int candidate_sci_numbers, int candidate_leaf_numbers)
{
    if (candidate_sci_numbers <= 0 || candidate_leaf_numbers <= 0)
    {
        return 0;
    }
    const long long primary_estimate = static_cast<long long>(
        Estimate_Gmxpacked_Primary_Record_Stream_Source_Capacity(
            candidate_sci_numbers, candidate_leaf_numbers));
    const long long reuse_estimate =
        static_cast<long long>(candidate_leaf_numbers) / 2ll +
        static_cast<long long>(candidate_sci_numbers) * 32ll + 65536ll;
    const long long estimate =
        primary_estimate > reuse_estimate ? primary_estimate : reuse_estimate;
    const long long max_int = 0x7fffffffll;
    return static_cast<int>(estimate > max_int ? max_int : estimate);
}

static void Mark_Gmxpacked_Current_Source_Generation(
    ClusteredNeighborProvider* provider)
{
    if (provider == NULL)
    {
        return;
    }
    ClusteredNeighborProviderInternal::PairList(provider)
        .gmxpacked_current_source_generation += 1;
}

static void Mark_Gmxpacked_Incremental_Source_Cache_Key(
    ClusteredNeighborProvider* provider)
{
    if (provider == NULL)
    {
        return;
    }
    auto& pair_list = ClusteredNeighborProviderInternal::PairList(provider);
    pair_list.gmxpacked_incremental_source_anchor_generation =
        pair_list.gmxpacked_current_source_anchor_generation;
    pair_list.gmxpacked_incremental_source_generation =
        pair_list.gmxpacked_current_source_generation;
}

using ClusteredGmxpackedRecordStreamCompactSummary =
    clustered_neighbor_builder_internal::CompactPayloadSummary;

#include "primitives/geometry.cuh"
#include "detail/pair_mask_primitives.inc.cuh"
#include "detail/payload_count_kernels.inc.cuh"
#include "detail/payload_source_kernels.inc.cuh"

}  // namespace

[[gnu::noinline]] bool ClusteredNeighborProvider::BuildPayloadGpu(
    const BuildPayloadInput& input)
{
    auto& spatial = spatial_;
    auto& pair_list = pair_list_;
    auto& workspace = workspace_;
    const auto& config = config_;
    const auto& domain = domain_;
    const auto& ordering = spatial.ordering;
    const auto& clusters = spatial.clusters;
    const auto& leaves = spatial.leaves;
    const auto& superclusters = spatial.superclusters;
    const auto& candidates = spatial.candidates;
    const int candidate_leaf_numbers = candidates.leaf_numbers;
    const int candidate_leaf_cluster_stride = candidates.leaf_cluster_stride;
    const int max_leaf_cluster_span = leaves.max_cluster_span;
    const auto& sort_permutation = ordering.sort_permutation;
    const auto& cluster_offsets = clusters.offsets;
    const auto& cluster_valid_masks = clusters.valid_masks;
    const auto& cluster_local_masks = clusters.local_masks;
    const auto& cluster_centers = clusters.centers;
    const auto& cluster_extents = clusters.extents;
    const auto& cluster_radii = clusters.radii;
    const auto& leaf_cluster_starts = leaves.cluster_starts;
    const auto& leaf_cluster_ends = leaves.cluster_ends;
    const auto& super_cluster_offsets = superclusters.offsets;
    const auto& cluster_to_supercluster =
        superclusters.cluster_to_supercluster;
    const auto& super_cluster_centers = superclusters.centers;
    const auto& sci_candidate_leaf_offsets = candidates.leaf_offsets;
    const auto& sci_candidate_leaf_ids = candidates.leaf_ids;
    const auto& candidate_leaf_reach_masks = candidates.leaf_reach_masks;
    const int cluster_size = config.cluster_size;
    const int super_cluster_clusters = config.clusters_per_supercluster;
    const int local_atom_numbers = domain.local_atom_count;
    const int* d_excluded_list_start = domain.excluded_list_start;
    const int* d_excluded_list = domain.excluded_list;
    const int* d_excluded_numbers = domain.excluded_numbers;
    const VECTOR* crd = input.crd;
    const VECTOR* commit_cache_crd = input.commit_cache_crd;
    const LTMatrix3 cell = input.cell;
    const LTMatrix3 rcell = input.rcell;
    const float cutoff = input.cutoff;
    const float build_cutoff = input.build_cutoff;
    const uint64_t* cluster_molecule_signatures =
        spatial.molecules.cluster_signatures.data;
    const int* cluster_molecule_ids = spatial.molecules.cluster_ids.data;
    const int candidate_sci_numbers = candidates.candidate_sci_numbers;
    const int* candidate_sci_supercluster_ids =
        candidates.sci_supercluster_ids.data;
    const bool candidate_leaf_queue2_payload_ready =
        input.candidate_leaf_queue2_payload_ready;
    auto& sci_shift_flags = workspace.sci_shift_flags;
    auto& sci_shift_offsets = workspace.sci_shift_offsets;
    auto& cjpacked_counts = workspace.cjpacked_counts;
    auto& cjpacked_group_offsets = workspace.cjpacked_group_offsets;
    auto& exclusion_counts = workspace.exclusion_counts;
    auto& exclusion_offsets = workspace.exclusion_offsets;
    auto& gmxpacked_count_light_source_fragments =
        workspace.count_light_source_fragments;
    auto& gmxpacked_count_source_fragment_cursor =
        workspace.count_source_fragment_cursor;
    auto& gmxpacked_count_source_fragment_overflow_rows_buffer =
        workspace.count_source_fragment_overflow_rows;
    auto& gmxpacked_count_source_materialize_cursors =
        workspace.count_source_materialize_cursors;
    auto& gmxpacked_record_stream_source_counts_by_candidate =
        workspace.source_counts_by_candidate;
    auto& gmxpacked_record_stream_source_offsets_by_candidate =
        workspace.source_offsets_by_candidate;
    auto& gmxpacked_record_stream_source_fill_cursor =
        workspace.source_fill_cursor;
    auto& gmxpacked_record_stream_source_overflow_rows_buffer =
        workspace.source_overflow_rows;

    const int sci_shift_numbers = candidate_sci_numbers;
    Reserve_Device_Buffer(sci_shift_numbers, &sci_shift_flags);
    Reserve_Device_Buffer(sci_shift_numbers, &cjpacked_counts);
    Reserve_Device_Buffer(sci_shift_numbers, &exclusion_counts);
    Reserve_Device_Buffer(sci_shift_numbers + 1, &sci_shift_offsets);
    Reserve_Device_Buffer(sci_shift_numbers + 1, &cjpacked_group_offsets);
    Reserve_Device_Buffer(sci_shift_numbers + 1, &exclusion_offsets);
    const int builder_warps_per_block =
        kClusteredBuilderBlockSize / kClusteredBuilderWarpSize;
    const int candidate_sci_blocks =
        (candidate_sci_numbers + builder_warps_per_block - 1) /
        builder_warps_per_block;
    const float gmxpacked_record_stream_cutoff = build_cutoff;
    const VECTOR* gmxpacked_record_stream_prune_crd = crd;
    int gmxpacked_count_source_fragment_capacity_request = 0;
    int gmxpacked_count_source_fragment_rows = 0;
    int gmxpacked_count_source_fragment_overflow_rows = 0;
    int gmxpacked_record_stream_filled_rows = 0;
    int gmxpacked_record_stream_overflow_rows = 0;
    int gmxpacked_record_stream_filled_aggregate_rows = 0;
    bool built_gmxpacked_record_stream_compact = false;
    bool route_gmxpacked_record_stream_compact = false;
    ClusteredGmxpackedRecordStreamCompactSummary
        gmxpacked_record_stream_compact_summary = {};
    Reserve_Device_Buffer(candidate_sci_numbers,
                          &gmxpacked_record_stream_source_counts_by_candidate);
    Reserve_Device_Buffer(candidate_sci_numbers + 1,
                          &gmxpacked_record_stream_source_offsets_by_candidate);
    deviceMemset(gmxpacked_record_stream_source_counts_by_candidate.data, 0,
                 sizeof(int) * candidate_sci_numbers);

    {
        const bool capture_fill_prune_reuse_sources =
            gmxpacked_record_stream_source_counts_by_candidate.data != NULL;
        if (capture_fill_prune_reuse_sources)
        {
            gmxpacked_count_source_fragment_capacity_request =
                Estimate_Gmxpacked_Primary_Fill_Prune_Reuse_Source_Capacity(
                    candidate_sci_numbers, candidate_leaf_numbers);
            Reserve_Device_Buffer(
                gmxpacked_count_source_fragment_capacity_request,
                &gmxpacked_count_light_source_fragments);
            Reserve_Device_Buffer(1, &gmxpacked_count_source_fragment_cursor);
            Reserve_Device_Buffer(
                1, &gmxpacked_count_source_fragment_overflow_rows_buffer);
            deviceMemset(gmxpacked_count_source_fragment_cursor.data, 0,
                         sizeof(int));
            deviceMemset(
                gmxpacked_count_source_fragment_overflow_rows_buffer.data, 0,
                sizeof(int));
        }
        const bool use_gmxpacked_count_fragment_parallel_emit =
            capture_fill_prune_reuse_sources;
        const bool use_gmxpacked_fixed_shift_builder_specialized =
            candidate_leaf_queue2_payload_ready &&
            use_gmxpacked_count_fragment_parallel_emit &&
            candidate_leaf_reach_masks.data != NULL;
        const bool use_gmxpacked_count_fixed_light_dedicated =
            use_gmxpacked_fixed_shift_builder_specialized &&
            capture_fill_prune_reuse_sources &&
            candidate_leaf_reach_masks.data != NULL &&
            gmxpacked_count_light_source_fragments.data != NULL &&
            gmxpacked_count_source_fragment_cursor.data != NULL &&
            gmxpacked_count_source_fragment_overflow_rows_buffer.data != NULL;
        auto* subgroup_count_kernel =
            use_gmxpacked_fixed_shift_builder_specialized
                ? Count_Nbnxm_Payload_From_Candidate_Leaves_Subgroup<true, true,
                                                                     true, true>
                : (use_gmxpacked_count_fragment_parallel_emit
                       ? Count_Nbnxm_Payload_From_Candidate_Leaves_Subgroup<
                             true, true, true>
                       : Count_Nbnxm_Payload_From_Candidate_Leaves_Subgroup<
                             true, false, true>);
        auto* count_kernel = subgroup_count_kernel;
        if (use_gmxpacked_count_fixed_light_dedicated)
        {
            Launch_Clustered_Gmxpacked_Count_Fixed_Light_Dedicated_Cooperative(
                candidate_sci_blocks, kClusteredBuilderBlockSize,
                candidate_sci_numbers, cluster_size, local_atom_numbers,
                gmxpacked_record_stream_cutoff, cell, rcell,
                gmxpacked_record_stream_prune_crd, sort_permutation.data,
                cluster_offsets.data, leaf_cluster_starts.data,
                leaf_cluster_ends.data, super_cluster_offsets.data,
                cluster_to_supercluster.data, candidate_sci_supercluster_ids,
                sci_candidate_leaf_offsets.data, sci_candidate_leaf_ids.data,
                candidate_leaf_cluster_stride, NULL,
                candidate_leaf_reach_masks.data, cluster_valid_masks.data,
                cluster_local_masks.data, cluster_centers.data,
                cluster_molecule_signatures, cluster_molecule_ids,
                d_excluded_list_start, d_excluded_list, d_excluded_numbers,
                max_leaf_cluster_span, sci_shift_flags.data,
                cjpacked_counts.data, exclusion_counts.data, NULL,
                gmxpacked_record_stream_source_counts_by_candidate.data, false,
                gmxpacked_count_light_source_fragments.data,
                gmxpacked_count_source_fragment_capacity_request,
                gmxpacked_count_source_fragment_cursor.data,
                gmxpacked_count_source_fragment_overflow_rows_buffer.data);
        }
        else
        {
            Launch_Device_Kernel(
                count_kernel, candidate_sci_blocks, kClusteredBuilderBlockSize,
                0, NULL, candidate_sci_numbers, sci_shift_numbers, cluster_size,
                super_cluster_clusters, local_atom_numbers, build_cutoff, cell,
                rcell, sort_permutation.data, cluster_offsets.data,
                leaf_cluster_starts.data, leaf_cluster_ends.data,
                super_cluster_offsets.data, cluster_to_supercluster.data,
                candidate_sci_supercluster_ids, super_cluster_centers.data,
                sci_candidate_leaf_offsets.data, sci_candidate_leaf_ids.data,
                NULL, candidate_leaf_cluster_stride,
                candidate_leaf_reach_masks.data,
                cluster_valid_masks.data, cluster_local_masks.data,
                cluster_centers.data, cluster_extents.data, cluster_radii.data,
                cluster_molecule_signatures, cluster_molecule_ids,
                d_excluded_list_start, d_excluded_list, d_excluded_numbers,
                gmxpacked_record_stream_cutoff,
                gmxpacked_record_stream_prune_crd, sci_shift_flags.data,
                cjpacked_counts.data, exclusion_counts.data, NULL,
                gmxpacked_record_stream_source_counts_by_candidate.data, false,
                NULL, max_leaf_cluster_span,
                capture_fill_prune_reuse_sources
                    ? gmxpacked_count_light_source_fragments.data
                    : NULL,
                capture_fill_prune_reuse_sources
                    ? gmxpacked_count_source_fragment_capacity_request
                    : 0,
                capture_fill_prune_reuse_sources
                    ? gmxpacked_count_source_fragment_cursor.data
                    : NULL,
                capture_fill_prune_reuse_sources
                    ? gmxpacked_count_source_fragment_overflow_rows_buffer.data
                    : NULL);
        }
        if (capture_fill_prune_reuse_sources)
        {
            deviceMemcpy(&gmxpacked_count_source_fragment_rows,
                         gmxpacked_count_source_fragment_cursor.data,
                         sizeof(int), deviceMemcpyDeviceToHost);
            deviceMemcpy(
                &gmxpacked_count_source_fragment_overflow_rows,
                gmxpacked_count_source_fragment_overflow_rows_buffer.data,
                sizeof(int), deviceMemcpyDeviceToHost);
        }

        (void)clustered_neighbor_builder_internal::ExclusiveScanCounts(this, sci_shift_numbers,
                                    sci_shift_flags.data,
                                    sci_shift_offsets.data);
        (void)clustered_neighbor_builder_internal::ExclusiveScanCounts(this, sci_shift_numbers,
                                    cjpacked_counts.data,
                                    cjpacked_group_offsets.data);
        (void)clustered_neighbor_builder_internal::ExclusiveScanCounts(this, sci_shift_numbers,
                                    exclusion_counts.data,
                                    exclusion_offsets.data);
        pair_list.gmxpacked_record_stream_source_numbers = clustered_neighbor_builder_internal::ExclusiveScanCounts(
            this, candidate_sci_numbers,
            gmxpacked_record_stream_source_counts_by_candidate.data,
            gmxpacked_record_stream_source_offsets_by_candidate.data);
        pair_list.gmxpacked_record_stream_aggregate_numbers = 0;
    }

    int gmxpacked_record_stream_source_capacity_request =
        pair_list.gmxpacked_record_stream_source_numbers;
    if (gmxpacked_record_stream_source_capacity_request > 0)
    {
        Reserve_Device_Buffer(1, &gmxpacked_record_stream_source_fill_cursor);
        Reserve_Device_Buffer(
            1, &gmxpacked_record_stream_source_overflow_rows_buffer);
        bool fill_prune_reuse_materialize_failed = false;
        for (int source_fill_attempt = 0; source_fill_attempt < 2;
             source_fill_attempt += 1)
        {
            bool materialized_count_fragments = false;
            Reserve_Device_Buffer(
                gmxpacked_record_stream_source_capacity_request,
                &pair_list.gmxpacked_record_stream_sources);
            deviceMemset(gmxpacked_record_stream_source_fill_cursor.data, 0,
                         sizeof(int));
            deviceMemset(
                gmxpacked_record_stream_source_overflow_rows_buffer.data, 0,
                sizeof(int));
            const bool can_materialize_count_fragments =
                !fill_prune_reuse_materialize_failed &&
                gmxpacked_count_light_source_fragments.data != NULL &&
                gmxpacked_record_stream_source_offsets_by_candidate.data !=
                    NULL &&
                gmxpacked_count_source_fragment_overflow_rows == 0 &&
                gmxpacked_count_source_fragment_rows ==
                    pair_list.gmxpacked_record_stream_source_numbers &&
                gmxpacked_count_source_fragment_rows ==
                    gmxpacked_record_stream_source_capacity_request;
            if (can_materialize_count_fragments)
            {
                Reserve_Device_Buffer(
                    candidate_sci_numbers,
                    &gmxpacked_count_source_materialize_cursors);
                deviceMemcpy(
                    gmxpacked_count_source_materialize_cursors.data,
                    gmxpacked_record_stream_source_offsets_by_candidate.data,
                    sizeof(int) * candidate_sci_numbers,
                    deviceMemcpyDeviceToDevice);
                constexpr int kLightMaterializeWarpsPerBlock =
                    kClusteredBuilderBlockSize / kClusteredBuilderWarpSize;
                Launch_Device_Kernel(
                    Materialize_Gmxpacked_Record_Stream_Sources_From_Light_Count_Fragments,
                    (gmxpacked_count_source_fragment_rows +
                     kLightMaterializeWarpsPerBlock - 1) /
                        kLightMaterializeWarpsPerBlock,
                    kClusteredBuilderBlockSize, 0, NULL,
                    gmxpacked_count_source_fragment_rows, candidate_sci_numbers,
                    gmxpacked_record_stream_source_capacity_request,
                    gmxpacked_count_light_source_fragments.data,
                    gmxpacked_record_stream_source_offsets_by_candidate.data,
                    gmxpacked_count_source_materialize_cursors.data,
                    super_cluster_offsets.data, cluster_local_masks.data,
                    pair_list.gmxpacked_record_stream_sources.data,
                    gmxpacked_record_stream_source_fill_cursor.data,
                    gmxpacked_record_stream_source_overflow_rows_buffer.data);
                materialized_count_fragments = true;
            }
            else
            {
                auto* fill_kernel =
                    Fill_Gmxpacked_Record_Stream_Sources_From_Candidate_Leaves_Subgroup;
                Launch_Device_Kernel(
                    fill_kernel, candidate_sci_blocks,
                    kClusteredBuilderBlockSize, 0, NULL, candidate_sci_numbers,
                    cluster_size, local_atom_numbers, build_cutoff, cell, rcell,
                    sort_permutation.data, cluster_offsets.data,
                    leaf_cluster_starts.data, leaf_cluster_ends.data,
                    super_cluster_offsets.data, cluster_to_supercluster.data,
                    candidate_sci_supercluster_ids,
                    sci_candidate_leaf_offsets.data,
                    sci_candidate_leaf_ids.data, candidate_leaf_cluster_stride,
                    candidate_leaf_reach_masks.data,
                    cluster_valid_masks.data, cluster_local_masks.data,
                    cluster_centers.data, cluster_extents.data,
                    cluster_molecule_signatures, cluster_molecule_ids,
                    d_excluded_list_start, d_excluded_list, d_excluded_numbers,
                    gmxpacked_record_stream_cutoff,
                    gmxpacked_record_stream_prune_crd,
                    gmxpacked_record_stream_source_capacity_request,
                    gmxpacked_record_stream_source_offsets_by_candidate.data,
                    pair_list.gmxpacked_record_stream_sources.data,
                    gmxpacked_record_stream_source_fill_cursor.data,
                    gmxpacked_record_stream_source_overflow_rows_buffer.data,
                    NULL, max_leaf_cluster_span);
            }

            deviceMemcpy(&gmxpacked_record_stream_filled_rows,
                         gmxpacked_record_stream_source_fill_cursor.data,
                         sizeof(int), deviceMemcpyDeviceToHost);
            deviceMemcpy(
                &gmxpacked_record_stream_overflow_rows,
                gmxpacked_record_stream_source_overflow_rows_buffer.data,
                sizeof(int), deviceMemcpyDeviceToHost);
            if (materialized_count_fragments &&
                (gmxpacked_record_stream_filled_rows !=
                     pair_list.gmxpacked_record_stream_source_numbers ||
                 gmxpacked_record_stream_overflow_rows != 0) &&
                source_fill_attempt == 0)
            {
                fprintf(stderr,
                        "[clustered gmxpacked fill prune reuse fallback] "
                        "step=%d source_rows=%d fragment_rows=%d "
                        "filled_rows=%d overflow_rows=%d\n",
                        md_info.sys.steps,
                        pair_list.gmxpacked_record_stream_source_numbers,
                        gmxpacked_count_source_fragment_rows,
                        gmxpacked_record_stream_filled_rows,
                        gmxpacked_record_stream_overflow_rows);
                fflush(stderr);
                fill_prune_reuse_materialize_failed = true;
                continue;
            }
            break;
        }
        {
            const bool record_stream_source_rows_match =
                gmxpacked_record_stream_filled_rows ==
                pair_list.gmxpacked_record_stream_source_numbers;
            const bool record_stream_source_overflow_free =
                gmxpacked_record_stream_overflow_rows == 0;
            const bool record_stream_source_offsets_available =
                gmxpacked_record_stream_source_offsets_by_candidate.data !=
                NULL;
            const bool source_cache_ready =
                record_stream_source_rows_match &&
                record_stream_source_overflow_free &&
                pair_list.gmxpacked_record_stream_source_numbers > 0 &&
                pair_list.gmxpacked_record_stream_sources.data != NULL &&
                record_stream_source_offsets_available;
            if (source_cache_ready)
            {
                if (record_stream_source_offsets_available)
                {
                    Reserve_Device_Buffer(
                        candidate_sci_numbers + 1,
                        &pair_list.gmxpacked_incremental_source_offsets_by_candidate);
                    deviceMemcpy(
                        pair_list.gmxpacked_incremental_source_offsets_by_candidate.data,
                        gmxpacked_record_stream_source_offsets_by_candidate
                            .data,
                        sizeof(int) * (candidate_sci_numbers + 1),
                        deviceMemcpyDeviceToDevice);
                    pair_list.gmxpacked_incremental_source_offsets_ready = true;
                }
                else
                {
                    pair_list.gmxpacked_incremental_source_offsets_ready = false;
                }
                Reserve_Device_Buffer(
                    pair_list.gmxpacked_record_stream_source_numbers,
                    &pair_list.gmxpacked_incremental_record_stream_sources);
                deviceMemcpy(
                    pair_list.gmxpacked_incremental_record_stream_sources.data,
                    pair_list.gmxpacked_record_stream_sources.data,
                    sizeof(CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE) *
                        static_cast<size_t>(
                            pair_list.gmxpacked_record_stream_source_numbers),
                    deviceMemcpyDeviceToDevice);
                pair_list.gmxpacked_incremental_candidate_sci_numbers =
                    candidate_sci_numbers;
                pair_list.gmxpacked_incremental_source_numbers =
                    pair_list.gmxpacked_record_stream_source_numbers;
                pair_list.gmxpacked_incremental_source_cutoff =
                    gmxpacked_record_stream_cutoff;
                pair_list.gmxpacked_incremental_source_cache_ready = true;
                Mark_Gmxpacked_Current_Source_Generation(this);
                Mark_Gmxpacked_Incremental_Source_Cache_Key(this);
            }
            else
            {
                Invalidate_Gmxpacked_Incremental_Source_Cache_State(this);
            }
        }
    }
    if (pair_list.gmxpacked_record_stream_source_numbers > 0 &&
        gmxpacked_record_stream_filled_rows ==
            pair_list.gmxpacked_record_stream_source_numbers &&
        gmxpacked_record_stream_overflow_rows == 0 &&
        pair_list.gmxpacked_record_stream_sources.data != NULL)
    {
        Reserve_Device_Buffer(pair_list.gmxpacked_record_stream_source_numbers,
                              &pair_list.gmxpacked_incremental_record_stream_sources);
        deviceMemcpy(
            pair_list.gmxpacked_incremental_record_stream_sources.data,
            pair_list.gmxpacked_record_stream_sources.data,
            sizeof(CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE) *
                static_cast<size_t>(pair_list.gmxpacked_record_stream_source_numbers),
            deviceMemcpyDeviceToDevice);
        pair_list.gmxpacked_incremental_candidate_sci_numbers = candidate_sci_numbers;
        pair_list.gmxpacked_incremental_source_numbers =
            pair_list.gmxpacked_record_stream_source_numbers;
        pair_list.gmxpacked_incremental_source_cutoff = gmxpacked_record_stream_cutoff;
        pair_list.gmxpacked_incremental_source_cache_ready = true;
        Mark_Gmxpacked_Current_Source_Generation(this);
        Mark_Gmxpacked_Incremental_Source_Cache_Key(this);
        pair_list.gmxpacked_incremental_source_offsets_ready =
            pair_list.gmxpacked_incremental_source_offsets_ready &&
            pair_list.gmxpacked_incremental_source_offsets_by_candidate.data != NULL;
    }
    if (pair_list.gmxpacked_record_stream_source_numbers > 0 &&
        gmxpacked_record_stream_filled_rows ==
            pair_list.gmxpacked_record_stream_source_numbers &&
        gmxpacked_record_stream_overflow_rows == 0 &&
        pair_list.gmxpacked_record_stream_sources.data != NULL)
    {
        const float configured_guard_margin =
            kRecordBuilderInnerActiveGuardMargin;
        const float guard_margin = configured_guard_margin;
        const float active_payload_guard_margin = guard_margin;
        float initial_active_cutoff =
            fminf(gmxpacked_record_stream_cutoff,
                  cutoff + active_payload_guard_margin);
        const VECTOR* initial_active_prune_crd = crd;
        const int active_source_rows =
            clustered_neighbor_builder_internal::PruneInnerActiveSources(
                this, initial_active_prune_crd, cell, rcell,
                initial_active_cutoff, true);
        pair_list.gmxpacked_inner_active_guard_cutoff = initial_active_cutoff;
        pair_list.gmxpacked_record_stream_source_numbers = active_source_rows;
        gmxpacked_record_stream_filled_rows = active_source_rows;
        gmxpacked_record_stream_overflow_rows = 0;
        gmxpacked_record_stream_filled_aggregate_rows = 0;
    }
    {
        const bool record_stream_source_rows_match =
            gmxpacked_record_stream_filled_rows ==
            pair_list.gmxpacked_record_stream_source_numbers;
        const bool record_stream_source_overflow_free =
            gmxpacked_record_stream_overflow_rows == 0;
        const bool build_record_stream_aggregates =
            record_stream_source_rows_match &&
            record_stream_source_overflow_free &&
            pair_list.gmxpacked_record_stream_source_numbers > 0;
        if (!record_stream_source_rows_match ||
            !record_stream_source_overflow_free)
        {
            fprintf(stderr,
                    "[clustered gmxpacked record builder mismatch] step=%d "
                    "source_rows=%d filled_rows=%d overflow_rows=%d\n",
                    md_info.sys.steps, pair_list.gmxpacked_record_stream_source_numbers,
                    gmxpacked_record_stream_filled_rows,
                    gmxpacked_record_stream_overflow_rows);
            fflush(stderr);
        }
        if (build_record_stream_aggregates)
        {
            gmxpacked_record_stream_filled_aggregate_rows =
                clustered_neighbor_builder_internal::BuildRecordStreamAggregates(this);
        }
        if (build_record_stream_aggregates &&
            (pair_list.gmxpacked_record_stream_aggregate_numbers <= 0 ||
             pair_list.gmxpacked_record_stream_aggregate_numbers >
                 pair_list.gmxpacked_record_stream_source_numbers ||
             gmxpacked_record_stream_filled_aggregate_rows !=
                 pair_list.gmxpacked_record_stream_aggregate_numbers))
        {
            fprintf(stderr,
                    "[clustered gmxpacked record aggregate mismatch] "
                    "step=%d source_rows=%d aggregate_rows=%d "
                    "filled_aggregate_rows=%d\n",
                    md_info.sys.steps, pair_list.gmxpacked_record_stream_source_numbers,
                    pair_list.gmxpacked_record_stream_aggregate_numbers,
                    gmxpacked_record_stream_filled_aggregate_rows);
            fflush(stderr);
        }
        if (build_record_stream_aggregates &&
            pair_list.gmxpacked_record_stream_aggregate_numbers > 0 &&
            gmxpacked_record_stream_filled_aggregate_rows ==
                pair_list.gmxpacked_record_stream_aggregate_numbers)
        {
            gmxpacked_record_stream_compact_summary =
                clustered_neighbor_builder_internal::BuildCompactPayload(this);
            built_gmxpacked_record_stream_compact = true;
        }
        if (built_gmxpacked_record_stream_compact)
        {
            route_gmxpacked_record_stream_compact =
                gmxpacked_record_stream_compact_summary.compact_sci > 0 &&
                gmxpacked_record_stream_compact_summary.compact_cj > 0 &&
                gmxpacked_record_stream_compact_summary.compact_excl > 0 &&
                pair_list.gmxpacked_cjpacked_numbers > 0 && pair_list.gmxpacked_sci.data != NULL &&
                pair_list.gmxpacked_cjpacked.data != NULL &&
                pair_list.gmxpacked_exclusions.data != NULL;
        }
    }
    if (!route_gmxpacked_record_stream_compact)
    {
        const bool empty_record_stream =
            pair_list.gmxpacked_record_stream_source_numbers == 0 &&
            gmxpacked_record_stream_filled_rows == 0 &&
            gmxpacked_record_stream_overflow_rows == 0 &&
            pair_list.gmxpacked_record_stream_aggregate_numbers == 0 &&
            gmxpacked_record_stream_filled_aggregate_rows == 0 &&
            gmxpacked_record_stream_compact_summary.compact_sci == 0 &&
            gmxpacked_record_stream_compact_summary.compact_cj == 0 &&
            gmxpacked_record_stream_compact_summary.compact_excl == 0;
        if (empty_record_stream)
        {
            clustered_neighbor_builder_internal::CommitBuildCache(this, commit_cache_crd, cutoff);
            return false;
        }
        fprintf(stderr,
                "[clustered gmxpacked record builder invariant failure] "
                "step=%d source_rows=%d filled_rows=%d overflow_rows=%d "
                "aggregate_rows=%d filled_aggregate_rows=%d compact_sci=%d "
                "compact_cj=%d compact_excl=%d\n",
                md_info.sys.steps, pair_list.gmxpacked_record_stream_source_numbers,
                gmxpacked_record_stream_filled_rows,
                gmxpacked_record_stream_overflow_rows,
                pair_list.gmxpacked_record_stream_aggregate_numbers,
                gmxpacked_record_stream_filled_aggregate_rows,
                gmxpacked_record_stream_compact_summary.compact_sci,
                gmxpacked_record_stream_compact_summary.compact_cj,
                gmxpacked_record_stream_compact_summary.compact_excl);
        fflush(stderr);
        throw std::runtime_error(
            "clustered gmxpacked record builder invariant failure");
    }

    return true;
}
#endif
