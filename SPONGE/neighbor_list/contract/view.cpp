#include "view.h"

#include <limits>

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
    const bool gmxpacked_payload_empty =
        view.gmxpacked_sci_numbers == 0 &&
        view.gmxpacked_cjpacked_numbers == 0;
    const bool gmxpacked_payload_ready =
        gmxpacked_payload_empty ||
        (view.gmxpacked_sci_numbers > 0 &&
         view.gmxpacked_cjpacked_numbers > 0 &&
         view.gmxpacked_sci != nullptr &&
         view.gmxpacked_cjpacked != nullptr &&
         view.gmxpacked_exclusion_numbers > 0 &&
         view.gmxpacked_exclusions != nullptr);
    if ((requirements.require_gmxpacked_payload ||
         requirements.require_gmxpacked_endpoint_incidence) &&
        !gmxpacked_payload_ready)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view has no gmxpacked pair payload",
            failure_reason);
    }
    if (requirements.require_gmxpacked_endpoint_incidence &&
        !gmxpacked_payload_empty &&
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
        !gmxpacked_payload_empty &&
        view.endpoint_incidence_provider_incarnation !=
            view.provider_incarnation)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view endpoint-incidence provider is stale",
            failure_reason);
    }
    if (requirements.require_gmxpacked_endpoint_incidence &&
        !gmxpacked_payload_empty &&
        view.endpoint_incidence_payload_generation !=
            view.gmxpacked_payload_generation)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view endpoint-incidence payload is stale",
            failure_reason);
    }
    if (requirements.require_gmxpacked_endpoint_incidence &&
        !gmxpacked_payload_empty &&
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
        !gmxpacked_payload_empty &&
        static_cast<long long>(view.endpoint_incidence_reference_numbers) >
            static_cast<long long>(2 * kClusteredJGroupSize) *
                view.gmxpacked_cjpacked_numbers)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view endpoint-incidence exceeds structural bound",
            failure_reason);
    }
    if (requirements.require_gmxpacked_endpoint_incidence &&
        !gmxpacked_payload_empty &&
        view.endpoint_incidence_offset_tail !=
            view.endpoint_incidence_reference_numbers)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view endpoint-incidence offset tail is stale",
            failure_reason);
    }
    if (requirements.require_pair_shift_metadata &&
        !gmxpacked_payload_empty &&
        (!view.pair_shift_metadata_ready || view.pair_shift_bits == nullptr))
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view pair-shift metadata is unavailable",
            failure_reason);
    }
    if (requirements.require_pair_shift_metadata &&
        !gmxpacked_payload_empty &&
        view.pair_shift_payload_generation !=
            view.gmxpacked_payload_generation)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view pair-shift payload generation is stale",
            failure_reason);
    }
    if (requirements.require_pair_shift_metadata &&
        !gmxpacked_payload_empty &&
        view.pair_shift_geometry_generation != view.geometry_generation)
    {
        return Clustered_Spatial_View_Fail(
            "clustered spatial view pair-shift geometry generation is stale",
            failure_reason);
    }
    if (requirements.require_pair_shift_metadata &&
        !gmxpacked_payload_empty &&
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
        !gmxpacked_payload_empty &&
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
