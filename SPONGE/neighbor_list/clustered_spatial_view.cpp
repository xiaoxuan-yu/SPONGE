#include "clustered_spatial_view.h"

#include <limits>

#include "../Lennard_Jones_force/clustered_lj.h"

namespace
{
bool Clustered_Spatial_View_Fail(const char* reason,
                                 const char** failure_reason)
{
    if (failure_reason != nullptr)
    {
        *failure_reason = reason;
    }
    return false;
}

bool Clustered_Rcell_Matches(LTMatrix3 lhs, LTMatrix3 rhs)
{
    return lhs.a11 == rhs.a11 && lhs.a21 == rhs.a21 &&
           lhs.a22 == rhs.a22 && lhs.a31 == rhs.a31 &&
           lhs.a32 == rhs.a32 && lhs.a33 == rhs.a33;
}

int Clustered_Find_Supercluster_For_Cluster(
    int cluster_id, int super_cluster_numbers,
    const int* super_cluster_offsets)
{
    int low = 0;
    int high = super_cluster_numbers;
    while (low < high)
    {
        const int middle = low + (high - low) / 2;
        if (super_cluster_offsets[middle + 1] <= cluster_id)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    return low < super_cluster_numbers &&
                   super_cluster_offsets[low] <= cluster_id &&
                   cluster_id < super_cluster_offsets[low + 1]
               ? low
               : -1;
}
}  // namespace

bool Clustered_Build_Gmxpacked_Endpoint_Incidence_Host(
    long long provider_incarnation, long long gmxpacked_payload_generation,
    int cluster_numbers, int super_cluster_numbers,
    const int* super_cluster_offsets, int sci_numbers,
    const CLUSTERED_GMXPACKED_SCI* sci_entries, int cjpacked_numbers,
    const CLUSTERED_GMXPACKED_CJ* cjpacked_entries,
    CLUSTERED_GMXPACKED_ENDPOINT_INCIDENCE_HOST* incidence,
    const char** failure_reason)
{
    if (incidence == nullptr)
    {
        return Clustered_Spatial_View_Fail(
            "clustered endpoint-incidence output is null", failure_reason);
    }
    incidence->Clear();
    if (provider_incarnation <= 0 ||
        gmxpacked_payload_generation < 0)
    {
        return Clustered_Spatial_View_Fail(
            "clustered endpoint-incidence lifecycle identity is invalid",
            failure_reason);
    }
    if (cluster_numbers <= 0 || super_cluster_numbers <= 0 ||
        super_cluster_offsets == nullptr || sci_numbers <= 0 ||
        sci_entries == nullptr || cjpacked_numbers <= 0 ||
        cjpacked_entries == nullptr)
    {
        return Clustered_Spatial_View_Fail(
            "clustered endpoint-incidence input payload is incomplete",
            failure_reason);
    }
    if (cjpacked_numbers >
        std::numeric_limits<int>::max() /
            (2 * kClusteredJGroupSize))
    {
        return Clustered_Spatial_View_Fail(
            "clustered endpoint-incidence reference bound overflows",
            failure_reason);
    }
    if (super_cluster_offsets[0] != 0 ||
        super_cluster_offsets[super_cluster_numbers] != cluster_numbers)
    {
        return Clustered_Spatial_View_Fail(
            "clustered endpoint-incidence supercluster offsets are invalid",
            failure_reason);
    }
    for (int supercluster_id = 0;
         supercluster_id < super_cluster_numbers; supercluster_id += 1)
    {
        if (super_cluster_offsets[supercluster_id] >
            super_cluster_offsets[supercluster_id + 1])
        {
            return Clustered_Spatial_View_Fail(
                "clustered endpoint-incidence supercluster offsets are invalid",
                failure_reason);
        }
    }

    std::vector<std::vector<CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE>>
        grouped_references(super_cluster_numbers);
    for (int sci_id = 0; sci_id < sci_numbers; sci_id += 1)
    {
        const CLUSTERED_GMXPACKED_SCI& sci = sci_entries[sci_id];
        if (sci.supercluster_id < 0 ||
            sci.supercluster_id >= super_cluster_numbers ||
            sci.cjpacked_begin < 0 ||
            sci.cjpacked_begin > sci.cjpacked_end ||
            sci.cjpacked_end > cjpacked_numbers)
        {
            return Clustered_Spatial_View_Fail(
                "clustered endpoint-incidence SCI record is invalid",
                failure_reason);
        }
        const int i_cluster_count =
            super_cluster_offsets[sci.supercluster_id + 1] -
            super_cluster_offsets[sci.supercluster_id];
        if (i_cluster_count <= 0 ||
            i_cluster_count > kClusteredSuperClusterClusters)
        {
            return Clustered_Spatial_View_Fail(
                "clustered endpoint-incidence SCI supercluster span is invalid",
                failure_reason);
        }
        const unsigned int valid_i_cluster_mask =
            (1u << static_cast<unsigned int>(i_cluster_count)) - 1u;
        for (int cjpacked_id = sci.cjpacked_begin;
             cjpacked_id < sci.cjpacked_end; cjpacked_id += 1)
        {
            const CLUSTERED_GMXPACKED_CJ& packed =
                cjpacked_entries[cjpacked_id];
            const unsigned int combined_imask =
                packed.split[0].imask | packed.split[1].imask;
            for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
            {
                const unsigned int i_cluster_mask =
                    (combined_imask >>
                     static_cast<unsigned int>(
                         jm * kClusteredSuperClusterClusters)) &
                    ((1u << kClusteredSuperClusterClusters) - 1u);
                if (i_cluster_mask == 0u)
                {
                    continue;
                }
                if ((i_cluster_mask & ~valid_i_cluster_mask) != 0u)
                {
                    return Clustered_Spatial_View_Fail(
                        "clustered endpoint-incidence SCI mask exceeds its supercluster span",
                        failure_reason);
                }
                const int cluster_j = packed.cj[jm];
                if (cluster_j < 0 || cluster_j >= cluster_numbers)
                {
                    return Clustered_Spatial_View_Fail(
                        "clustered endpoint-incidence CJ record is invalid",
                        failure_reason);
                }
                const int j_supercluster =
                    Clustered_Find_Supercluster_For_Cluster(
                        cluster_j, super_cluster_numbers,
                        super_cluster_offsets);
                if (j_supercluster < 0)
                {
                    return Clustered_Spatial_View_Fail(
                        "clustered endpoint-incidence CJ cluster has no supercluster",
                        failure_reason);
                }
                CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE native = {};
                native.sci_id = sci_id;
                native.cjpacked_id = cjpacked_id;
                native.i_cluster_mask = i_cluster_mask;
                native.jm = static_cast<unsigned char>(jm);
                native.orientation =
                    CLUSTERED_ENDPOINT_ORIENTATION::NATIVE_I;
                grouped_references[sci.supercluster_id].push_back(native);

                CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE transposed = native;
                transposed.orientation =
                    CLUSTERED_ENDPOINT_ORIENTATION::TRANSPOSED_J;
                grouped_references[j_supercluster].push_back(transposed);
            }
        }
    }

    incidence->provider_incarnation = provider_incarnation;
    incidence->gmxpacked_payload_generation =
        gmxpacked_payload_generation;
    incidence->super_cluster_numbers = super_cluster_numbers;
    incidence->offsets.resize(super_cluster_numbers + 1, 0);
    for (int supercluster_id = 0;
         supercluster_id < super_cluster_numbers; supercluster_id += 1)
    {
        incidence->offsets[supercluster_id + 1] =
            incidence->offsets[supercluster_id] +
            static_cast<int>(
                grouped_references[supercluster_id].size());
    }
    incidence->references.reserve(incidence->offsets.back());
    for (int supercluster_id = 0;
         supercluster_id < super_cluster_numbers; supercluster_id += 1)
    {
        incidence->references.insert(
            incidence->references.end(),
            grouped_references[supercluster_id].begin(),
            grouped_references[supercluster_id].end());
    }
    incidence->ready = true;
    if (failure_reason != nullptr)
    {
        *failure_reason = nullptr;
    }
    return true;
}

bool Clustered_Validate_Spatial_View(
    const CLUSTERED_SPATIAL_VIEW& view,
    const CLUSTERED_SPATIAL_VIEW_REQUIREMENTS& requirements,
    const char** failure_reason)
{
    if (!view.ready)
    {
        return Clustered_Spatial_View_Fail("clustered spatial view is not ready",
                                           failure_reason);
    }
    if (view.provider_incarnation <= 0 || view.lease_epoch < 0 ||
        view.native_payload_generation < 0 ||
        view.gmxpacked_payload_generation < 0 ||
        view.geometry_generation < 0)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view lifecycle identity is invalid",
            failure_reason);
    }
    if (view.cluster_size != kClusteredClusterSize ||
        view.super_cluster_clusters != kClusteredSuperClusterClusters)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view layout constants do not match",
            failure_reason);
    }
    if (requirements.require_backend &&
        view.backend != requirements.backend)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view backend does not match",
            failure_reason);
    }
    if (requirements.require_same_producer_stream &&
        (view.readiness_scope !=
             CLUSTERED_SPATIAL_READINESS_SCOPE::PRODUCER_STREAM_ORDERED ||
         view.producer_stream != requirements.consumer_stream))
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view producer stream does not match",
            failure_reason);
    }
    if (requirements.provider_incarnation >= 0 &&
        view.provider_incarnation != requirements.provider_incarnation)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view provider incarnation is stale",
            failure_reason);
    }
    if (requirements.lease_epoch >= 0 &&
        view.lease_epoch != requirements.lease_epoch)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view lease epoch is stale", failure_reason);
    }
    if (requirements.native_payload_generation >= 0 &&
        view.native_payload_generation !=
            requirements.native_payload_generation)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view native payload generation is stale",
            failure_reason);
    }
    if (requirements.gmxpacked_payload_generation >= 0 &&
        view.gmxpacked_payload_generation !=
            requirements.gmxpacked_payload_generation)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view gmxpacked payload generation is stale",
            failure_reason);
    }
    if (requirements.source_generation >= 0 &&
        view.source_generation != requirements.source_generation)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view source generation is stale",
            failure_reason);
    }
    if (requirements.geometry_generation >= 0 &&
        view.geometry_generation != requirements.geometry_generation)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view geometry generation is stale",
            failure_reason);
    }
    if (view.local_atom_numbers < 0 || view.ghost_numbers < 0 ||
        static_cast<long long>(view.total_atom_numbers) !=
            static_cast<long long>(view.local_atom_numbers) +
                static_cast<long long>(view.ghost_numbers) ||
        view.padded_total_atom_numbers < view.total_atom_numbers ||
        view.cluster_numbers <= 0 || view.super_cluster_numbers <= 0)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view local/ghost domain is inconsistent",
            failure_reason);
    }
    if (requirements.local_atom_numbers >= 0 &&
        requirements.local_atom_numbers != view.local_atom_numbers)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view local atom count does not match",
            failure_reason);
    }
    if (requirements.ghost_numbers >= 0 &&
        requirements.ghost_numbers != view.ghost_numbers)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view ghost atom count does not match",
            failure_reason);
    }
    if (requirements.require_all_local_atoms &&
        view.direct_local_atom_numbers != view.local_atom_numbers)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view covers only a local atom subset",
            failure_reason);
    }
    if (requirements.cutoff >= 0.0f &&
        (view.cached_cutoff < 0.0f ||
         view.cached_cutoff + 1.0e-4f < requirements.cutoff))
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view cutoff is not a semantic superset",
            failure_reason);
    }
#define REQUIRE_CLUSTERED_VIEW_FIELD(field)                                  \
    if (view.field == nullptr)                                               \
    {                                                                        \
        return Clustered_Spatial_View_Fail(                                  \
            "clustered spatial view is missing " #field, failure_reason);    \
    }
    REQUIRE_CLUSTERED_VIEW_FIELD(sort_permutation);
    REQUIRE_CLUSTERED_VIEW_FIELD(cluster_offsets);
    REQUIRE_CLUSTERED_VIEW_FIELD(cluster_valid_masks);
    REQUIRE_CLUSTERED_VIEW_FIELD(cluster_local_masks);
    REQUIRE_CLUSTERED_VIEW_FIELD(cluster_centers);
    REQUIRE_CLUSTERED_VIEW_FIELD(cluster_extents);
    REQUIRE_CLUSTERED_VIEW_FIELD(super_cluster_offsets);
#undef REQUIRE_CLUSTERED_VIEW_FIELD
    const bool native_payload_ready =
        view.sci_numbers > 0 && view.cjpacked_numbers > 0 &&
        view.exclusion_pool_numbers >= 0 && view.sci != nullptr &&
        view.cjpacked != nullptr &&
        (view.exclusion_pool_numbers == 0 ||
         view.exclusion_mask_pool != nullptr);
    const bool gmxpacked_payload_ready =
        view.gmxpacked_sci_numbers > 0 &&
        view.gmxpacked_cjpacked_numbers > 0 &&
        view.gmxpacked_sci != nullptr && view.gmxpacked_cjpacked != nullptr &&
        view.gmxpacked_exclusion_numbers > 0 &&
        view.gmxpacked_exclusions != nullptr;
    if (!native_payload_ready && !gmxpacked_payload_ready)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view has no usable pair payload",
            failure_reason);
    }
    if (requirements.require_native_payload && !native_payload_ready)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view has no native pair payload",
            failure_reason);
    }
    if ((requirements.require_gmxpacked_payload ||
         requirements.require_gmxpacked_endpoint_incidence) &&
        !gmxpacked_payload_ready)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view has no gmxpacked pair payload",
            failure_reason);
    }
    if (requirements.require_native_grouped_sci &&
        (!view.native_grouped_sci_ready ||
         view.native_grouped_sci_offsets == nullptr ||
         view.native_grouped_sci_ids == nullptr))
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view has no native grouped SCI range",
            failure_reason);
    }
    if (requirements.require_gmxpacked_grouped_sci &&
        (!view.gmxpacked_grouped_sci_ready ||
         view.gmxpacked_grouped_sci_offsets == nullptr ||
         view.gmxpacked_grouped_sci_ids == nullptr))
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view has no gmxpacked grouped SCI range",
            failure_reason);
    }
    if (requirements.require_gmxpacked_endpoint_incidence &&
        (!view.gmxpacked_endpoint_incidence_ready ||
         view.endpoint_incidence_reference_numbers <= 0 ||
         view.gmxpacked_endpoint_incidence_offsets == nullptr ||
         view.gmxpacked_endpoint_incidence_references == nullptr))
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view has no gmxpacked endpoint incidence",
            failure_reason);
    }
    if (requirements.require_gmxpacked_endpoint_incidence &&
        view.endpoint_incidence_provider_incarnation !=
            view.provider_incarnation)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view endpoint-incidence provider is stale",
            failure_reason);
    }
    if (requirements.require_gmxpacked_endpoint_incidence &&
        view.endpoint_incidence_payload_generation !=
            view.gmxpacked_payload_generation)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view endpoint-incidence payload is stale",
            failure_reason);
    }
    if (requirements.require_gmxpacked_endpoint_incidence &&
        (view.endpoint_incidence_sci_numbers !=
             view.gmxpacked_sci_numbers ||
         view.endpoint_incidence_cjpacked_numbers !=
             view.gmxpacked_cjpacked_numbers ||
         view.endpoint_incidence_super_cluster_numbers !=
             view.super_cluster_numbers))
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view endpoint-incidence counts are stale",
            failure_reason);
    }
    if (requirements.require_gmxpacked_endpoint_incidence &&
        static_cast<long long>(view.endpoint_incidence_reference_numbers) >
            static_cast<long long>(2 * kClusteredJGroupSize) *
                view.gmxpacked_cjpacked_numbers)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view endpoint-incidence exceeds structural bound",
            failure_reason);
    }
    if (requirements.require_gmxpacked_endpoint_incidence &&
        view.endpoint_incidence_offset_tail !=
            view.endpoint_incidence_reference_numbers)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view endpoint-incidence offset tail is stale",
            failure_reason);
    }
    if (requirements.require_pair_shift_metadata &&
        (!view.pair_shift_metadata_ready || view.pair_shift_bits == nullptr))
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view pair-shift metadata is unavailable",
            failure_reason);
    }
    if (requirements.require_pair_shift_metadata &&
        view.pair_shift_payload_generation !=
            view.gmxpacked_payload_generation)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view pair-shift payload generation is stale",
            failure_reason);
    }
    if (requirements.require_pair_shift_metadata &&
        view.pair_shift_geometry_generation != view.geometry_generation)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view pair-shift geometry generation is stale",
            failure_reason);
    }
    if (requirements.require_pair_shift_metadata &&
        (view.pair_shift_sci_numbers != view.gmxpacked_sci_numbers ||
         view.pair_shift_cjpacked_numbers !=
             view.gmxpacked_cjpacked_numbers ||
         view.pair_shift_exclusion_numbers !=
             view.gmxpacked_exclusion_numbers))
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view pair-shift counts are stale",
            failure_reason);
    }
    if (requirements.require_pair_shift_rcell &&
        (!requirements.require_pair_shift_metadata ||
         !Clustered_Rcell_Matches(view.pair_shift_rcell,
                                  requirements.pair_shift_rcell)))
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view pair-shift rcell does not match",
            failure_reason);
    }
    if (failure_reason != nullptr)
    {
        *failure_reason = nullptr;
    }
    return true;
}

bool Make_Clustered_Spatial_View_From_LJ_Cache(
    const LJ_CLUSTERED_DIRECT_CACHE* cache, CLUSTERED_SPATIAL_VIEW* view,
    const char** failure_reason)
{
    if (view == nullptr)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view output is null", failure_reason);
    }
    *view = {};
    if (cache == nullptr || !cache->initialized)
    {
        return Clustered_Spatial_View_Fail(
            "clustered payload cache is absent or uninitialized",
            failure_reason);
    }

    const LJ_CLUSTER_LAYOUT& layout = cache->layout;
    if (!layout.Use_Clustered_Direct() || !layout.cache_ready)
    {
        return Clustered_Spatial_View_Fail(
            "clustered payload cache is disabled or stale", failure_reason);
    }

#if defined(USE_CUDA)
    view->backend = CLUSTERED_SPATIAL_BACKEND::CUDA;
    view->readiness_scope =
        CLUSTERED_SPATIAL_READINESS_SCOPE::PRODUCER_STREAM_ORDERED;
#elif defined(USE_HIP)
    view->backend = CLUSTERED_SPATIAL_BACKEND::HIP;
    view->readiness_scope =
        CLUSTERED_SPATIAL_READINESS_SCOPE::PRODUCER_STREAM_ORDERED;
#else
    view->backend = CLUSTERED_SPATIAL_BACKEND::CPU;
    view->readiness_scope =
        CLUSTERED_SPATIAL_READINESS_SCOPE::HOST_COMPLETE;
#endif
    view->producer_stream = nullptr;
    view->provider_incarnation = layout.provider_incarnation;
    view->lease_epoch = layout.spatial_view_lease_epoch;
    view->native_payload_generation = layout.native_payload_generation;
    view->gmxpacked_payload_generation =
        layout.gmxpacked_compact_payload_generation;
    view->source_generation = layout.gmxpacked_current_source_generation;
    view->geometry_generation = layout.geometry_generation;
    view->cluster_size = layout.cluster_size;
    view->super_cluster_clusters = layout.super_cluster_clusters;
    view->local_atom_numbers = layout.local_atom_numbers;
    view->direct_local_atom_numbers = layout.direct_local_atom_numbers;
    view->ghost_numbers = layout.ghost_numbers;
    view->total_atom_numbers = layout.total_atom_numbers;
    view->padded_total_atom_numbers = layout.padded_total_atom_numbers;
    view->cluster_numbers = layout.cluster_numbers;
    view->super_cluster_numbers = layout.super_cluster_numbers;
    view->sci_numbers = layout.sci_numbers;
    view->cjpacked_numbers = layout.cjpacked_numbers;
    view->exclusion_pool_numbers = layout.exclusion_pool_numbers;
    view->gmxpacked_sci_numbers = layout.gmxpacked_sci_numbers;
    view->gmxpacked_cjpacked_numbers = layout.gmxpacked_cjpacked_numbers;
    view->gmxpacked_exclusion_numbers = layout.gmxpacked_exclusion_numbers;
    view->cached_cutoff = layout.cached_cutoff;
    view->rebuild_skin = layout.rebuild_skin;
    view->native_grouped_sci_ready = layout.grouped_sci_ready;
    view->gmxpacked_grouped_sci_ready =
        layout.gmxpacked_grouped_sci_ready;
    view->endpoint_incidence_provider_incarnation =
        layout.gmxpacked_endpoint_incidence_provider_incarnation;
    view->endpoint_incidence_payload_generation =
        layout.gmxpacked_endpoint_incidence_payload_generation;
    view->endpoint_incidence_sci_numbers =
        layout.gmxpacked_endpoint_incidence_sci_numbers;
    view->endpoint_incidence_cjpacked_numbers =
        layout.gmxpacked_endpoint_incidence_cjpacked_numbers;
    view->endpoint_incidence_super_cluster_numbers =
        layout.gmxpacked_endpoint_incidence_super_cluster_numbers;
    view->endpoint_incidence_reference_numbers =
        layout.gmxpacked_endpoint_incidence_reference_numbers;
    view->endpoint_incidence_offset_tail =
        layout.gmxpacked_endpoint_incidence_offset_tail;
    view->gmxpacked_endpoint_incidence_ready =
        layout.gmxpacked_endpoint_incidence_ready &&
        view->endpoint_incidence_provider_incarnation ==
            view->provider_incarnation &&
        view->endpoint_incidence_payload_generation ==
            view->gmxpacked_payload_generation &&
        view->endpoint_incidence_sci_numbers ==
            view->gmxpacked_sci_numbers &&
        view->endpoint_incidence_cjpacked_numbers ==
            view->gmxpacked_cjpacked_numbers &&
        view->endpoint_incidence_super_cluster_numbers ==
            view->super_cluster_numbers &&
        view->endpoint_incidence_reference_numbers > 0 &&
        view->endpoint_incidence_offset_tail ==
            view->endpoint_incidence_reference_numbers;
    view->pair_shift_payload_generation =
        layout.gmxpacked_pair_shift_metadata_payload_generation;
    view->pair_shift_geometry_generation =
        layout.gmxpacked_pair_shift_metadata_geometry_generation;
    view->pair_shift_sci_numbers =
        layout.gmxpacked_pair_shift_metadata_sci_numbers;
    view->pair_shift_cjpacked_numbers =
        layout.gmxpacked_pair_shift_metadata_cjpacked_numbers;
    view->pair_shift_exclusion_numbers =
        layout.gmxpacked_pair_shift_metadata_exclusion_numbers;
    view->pair_shift_rcell = layout.gmxpacked_pair_shift_metadata_rcell;
    view->pair_shift_metadata_ready =
        layout.gmxpacked_pair_shift_metadata_ready &&
        view->pair_shift_payload_generation ==
            view->gmxpacked_payload_generation &&
        view->pair_shift_geometry_generation == view->geometry_generation &&
        view->pair_shift_sci_numbers == view->gmxpacked_sci_numbers &&
        view->pair_shift_cjpacked_numbers ==
            view->gmxpacked_cjpacked_numbers &&
        view->pair_shift_exclusion_numbers ==
            view->gmxpacked_exclusion_numbers;

    view->sort_permutation = layout.d_sort_permutation;
    view->cluster_offsets = layout.d_cluster_offsets;
    view->cluster_valid_masks = layout.d_cluster_valid_masks;
    view->cluster_local_masks = layout.d_cluster_local_masks;
    view->cluster_centers = layout.d_cluster_centers;
    view->cluster_extents = layout.d_cluster_extents;
    view->super_cluster_offsets = layout.d_super_cluster_offsets;
    view->native_grouped_sci_offsets = layout.d_grouped_sci_offsets;
    view->native_grouped_sci_ids = layout.d_grouped_sci_ids;
    view->gmxpacked_grouped_sci_offsets =
        layout.d_gmxpacked_grouped_sci_offsets;
    view->gmxpacked_grouped_sci_ids = layout.d_gmxpacked_grouped_sci_ids;
    view->gmxpacked_endpoint_incidence_offsets =
        layout.d_gmxpacked_endpoint_incidence_offsets;
    view->gmxpacked_endpoint_incidence_references =
        layout.d_gmxpacked_endpoint_incidence_references;
    view->sci = layout.d_nbnxm_sci;
    view->cjpacked = layout.d_nbnxm_cjpacked;
    view->exclusion_mask_pool = layout.d_exclusion_mask_pool;
    view->gmxpacked_sci = layout.d_gmxpacked_sci;
    view->gmxpacked_cjpacked = layout.d_gmxpacked_cjpacked;
    view->gmxpacked_exclusions = layout.d_gmxpacked_exclusions;
    view->pair_shift_bits = layout.d_pair_shift_bits;
    view->gmxpacked_sci_shift_safe_flags =
        layout.d_gmxpacked_pair_shift_sci_safe_flags;
    view->ready = true;

    CLUSTERED_SPATIAL_VIEW_REQUIREMENTS requirements;
    requirements.local_atom_numbers = layout.local_atom_numbers;
    requirements.ghost_numbers = layout.ghost_numbers;
    requirements.cutoff = layout.cached_cutoff;
    requirements.provider_incarnation = layout.provider_incarnation;
    requirements.lease_epoch = layout.spatial_view_lease_epoch;
    requirements.native_payload_generation =
        layout.native_payload_generation;
    requirements.gmxpacked_payload_generation =
        layout.gmxpacked_compact_payload_generation;
    requirements.source_generation =
        layout.gmxpacked_current_source_generation;
    requirements.geometry_generation = layout.geometry_generation;
#if defined(USE_CUDA) || defined(USE_HIP)
    requirements.require_same_producer_stream = true;
    requirements.consumer_stream = nullptr;
#endif
    requirements.require_all_local_atoms = false;
    return Clustered_Validate_Spatial_View(*view, requirements,
                                           failure_reason);
}
