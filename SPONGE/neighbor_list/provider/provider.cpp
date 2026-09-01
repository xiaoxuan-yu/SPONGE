#include "provider.h"

#include "lifecycle.h"
#include "runtime.h"

namespace
{
bool Clustered_Provider_Fail(const char* reason, const char** failure_reason)
{
    if (failure_reason != nullptr)
    {
        *failure_reason = reason;
    }
    return false;
}
}  // namespace

bool ClusteredNeighborProvider::AcquireGatherBinding(
    ClusteredGatherBinding* binding)
{
    if (binding == nullptr)
    {
        return false;
    }
    *binding = {};
    if (!initialized_ || !spatial_.ready ||
        spatial_.total_atom_numbers <= 0 || spatial_.cluster_numbers <= 0)
    {
        return false;
    }
    binding->provider_incarnation = provider_incarnation_;
    binding->geometry_generation = spatial_.geometry_generation;
    binding->total_atom_numbers = spatial_.total_atom_numbers;
    binding->cluster_numbers = spatial_.cluster_numbers;
    binding->sort_permutation = spatial_.ordering.sort_permutation.data;
    binding->cluster_offsets = spatial_.clusters.offsets.data;
    binding->cluster_centers = spatial_.clusters.centers.data;
    binding->cluster_fractional_centers =
        spatial_.clusters.fractional_centers.data;
    binding->cluster_fractional_extents =
        spatial_.clusters.fractional_extents.data;
    return binding->sort_permutation != nullptr &&
           binding->cluster_offsets != nullptr &&
           binding->cluster_centers != nullptr &&
           binding->cluster_fractional_centers != nullptr &&
           binding->cluster_fractional_extents != nullptr;
}

bool ClusteredNeighborProvider::IsGatheredGeometryCurrent(
    uint64_t provider_incarnation, uint64_t geometry_generation) const
{
    return initialized_ && spatial_.ready &&
           provider_incarnation_ == provider_incarnation &&
           spatial_.geometry_generation == geometry_generation;
}

void ClusteredNeighborProvider::BindWorkingDevice()
{
#ifndef USE_CPU
    if (controller_ != nullptr)
    {
        working_device_ = controller_->working_device;
    }
    clustered_neighbor_runtime::Bind_Clustered_Working_Device(
        &working_device_);
#endif
}

void ClusteredNeighborProvider::PublishGatheredGeometry(
    ClusteredGatherBinding* binding)
{
    Publish_Gathered_Cluster_Geometry(this);
    if (binding != nullptr)
    {
        binding->provider_incarnation = provider_incarnation_;
        binding->geometry_generation = spatial_.geometry_generation;
    }
}

bool ClusteredNeighborProvider::AcquireView(
    const CLUSTERED_SPATIAL_VIEW_REQUIREMENTS& requirements,
    CLUSTERED_SPATIAL_VIEW* view, const char** failure_reason) const
{
    if (view == nullptr)
    {
        return Clustered_Provider_Fail("clustered spatial view output is null",
                                       failure_reason);
    }
    *view = {};
    if (!initialized_)
    {
        return Clustered_Provider_Fail(
            "clustered provider is absent or uninitialized", failure_reason);
    }
    if (!spatial_.ready)
    {
        return Clustered_Provider_Fail("clustered payload cache is stale",
                                       failure_reason);
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
    view->readiness_scope = CLUSTERED_SPATIAL_READINESS_SCOPE::HOST_COMPLETE;
#endif
    view->producer_stream = nullptr;
    view->provider_incarnation = provider_incarnation_;
    view->lease_epoch = lease_epoch_;
    view->gmxpacked_payload_generation =
        pair_list_.gmxpacked_compact_payload_generation;
    view->source_generation =
        pair_list_.gmxpacked_current_source_generation;
    view->geometry_generation = spatial_.geometry_generation;
    view->cluster_size = config_.cluster_size;
    view->super_cluster_clusters = config_.clusters_per_supercluster;
    view->local_atom_numbers = domain_.local_atom_count;
    view->direct_local_atom_numbers = domain_.direct_local_atom_count;
    view->ghost_numbers = domain_.ghost_atom_count;
    view->total_atom_numbers = spatial_.total_atom_numbers;
    view->padded_total_atom_numbers = spatial_.padded_total_atom_numbers;
    view->cluster_numbers = spatial_.cluster_numbers;
    view->super_cluster_numbers = spatial_.super_cluster_numbers;
    view->gmxpacked_sci_numbers = pair_list_.gmxpacked_sci_numbers;
    view->gmxpacked_cjpacked_numbers =
        pair_list_.gmxpacked_cjpacked_numbers;
    view->gmxpacked_exclusion_numbers =
        pair_list_.gmxpacked_exclusion_numbers;
    view->cached_cutoff = cached_cutoff_;
    view->rebuild_skin = effective_rebuild_skin_;
    view->endpoint_incidence_provider_incarnation =
        pair_list_.gmxpacked_endpoint_incidence_provider_incarnation;
    view->endpoint_incidence_payload_generation =
        pair_list_.gmxpacked_endpoint_incidence_payload_generation;
    view->endpoint_incidence_sci_numbers =
        pair_list_.gmxpacked_endpoint_incidence_sci_numbers;
    view->endpoint_incidence_cjpacked_numbers =
        pair_list_.gmxpacked_endpoint_incidence_cjpacked_numbers;
    view->endpoint_incidence_super_cluster_numbers =
        pair_list_.gmxpacked_endpoint_incidence_super_cluster_numbers;
    view->endpoint_incidence_reference_numbers =
        pair_list_.gmxpacked_endpoint_incidence_reference_numbers;
    view->endpoint_incidence_offset_tail =
        pair_list_.gmxpacked_endpoint_incidence_offset_tail;
    view->gmxpacked_endpoint_incidence_ready =
        pair_list_.gmxpacked_endpoint_incidence_ready &&
        view->endpoint_incidence_provider_incarnation ==
            view->provider_incarnation &&
        view->endpoint_incidence_payload_generation ==
            view->gmxpacked_payload_generation &&
        view->endpoint_incidence_sci_numbers == view->gmxpacked_sci_numbers &&
        view->endpoint_incidence_cjpacked_numbers ==
            view->gmxpacked_cjpacked_numbers &&
        view->endpoint_incidence_super_cluster_numbers ==
            view->super_cluster_numbers &&
        view->endpoint_incidence_reference_numbers > 0 &&
        view->endpoint_incidence_offset_tail ==
            view->endpoint_incidence_reference_numbers;
    view->pair_shift_payload_generation =
        pair_list_.gmxpacked_pair_shift_metadata_payload_generation;
    view->pair_shift_geometry_generation =
        pair_list_.gmxpacked_pair_shift_metadata_geometry_generation;
    view->pair_shift_sci_numbers =
        pair_list_.gmxpacked_pair_shift_metadata_sci_numbers;
    view->pair_shift_cjpacked_numbers =
        pair_list_.gmxpacked_pair_shift_metadata_cjpacked_numbers;
    view->pair_shift_exclusion_numbers =
        pair_list_.gmxpacked_pair_shift_metadata_exclusion_numbers;
    view->pair_shift_rcell =
        pair_list_.gmxpacked_pair_shift_metadata_rcell;
    view->pair_shift_metadata_ready =
        pair_list_.gmxpacked_pair_shift_metadata_ready &&
        view->pair_shift_payload_generation ==
            view->gmxpacked_payload_generation &&
        view->pair_shift_geometry_generation == view->geometry_generation &&
        view->pair_shift_sci_numbers == view->gmxpacked_sci_numbers &&
        view->pair_shift_cjpacked_numbers == view->gmxpacked_cjpacked_numbers &&
        view->pair_shift_exclusion_numbers == view->gmxpacked_exclusion_numbers;

    view->atom_local = domain_.atom_local;
    view->sort_permutation = spatial_.ordering.sort_permutation.data;
    view->cluster_offsets = spatial_.clusters.offsets.data;
    view->cluster_valid_masks = spatial_.clusters.valid_masks.data;
    view->cluster_local_masks = spatial_.clusters.local_masks.data;
    view->cluster_centers = spatial_.clusters.centers.data;
    view->cluster_extents = spatial_.clusters.extents.data;
    view->super_cluster_offsets = spatial_.superclusters.offsets.data;
    view->gmxpacked_endpoint_incidence_offsets =
        pair_list_.gmxpacked_endpoint_incidence_offsets.data;
    view->gmxpacked_endpoint_incidence_references =
        pair_list_.gmxpacked_endpoint_incidence_references.data;
    view->gmxpacked_sci = pair_list_.gmxpacked_sci.data;
    view->gmxpacked_cjpacked = pair_list_.gmxpacked_cjpacked.data;
    view->gmxpacked_exclusions = pair_list_.gmxpacked_exclusions.data;
    view->pair_shift_bits = pair_list_.pair_shift_bits.data;
    view->gmxpacked_sci_shift_safe_flags =
        pair_list_.gmxpacked_pair_shift_sci_safe_flags.data;
    view->ready = true;

    CLUSTERED_SPATIAL_VIEW_REQUIREMENTS effective = requirements;
    if (effective.local_atom_numbers < 0)
    {
        effective.local_atom_numbers = domain_.local_atom_count;
    }
    if (effective.ghost_numbers < 0)
    {
        effective.ghost_numbers = domain_.ghost_atom_count;
    }
    if (effective.cutoff < 0.0f)
    {
        effective.cutoff = cached_cutoff_;
    }
    if (effective.provider_incarnation < 0)
    {
        effective.provider_incarnation = provider_incarnation_;
    }
    if (effective.lease_epoch < 0)
    {
        effective.lease_epoch = lease_epoch_;
    }
    if (effective.gmxpacked_payload_generation < 0)
    {
        effective.gmxpacked_payload_generation =
            pair_list_.gmxpacked_compact_payload_generation;
    }
    if (effective.source_generation < 0)
    {
        effective.source_generation =
            pair_list_.gmxpacked_current_source_generation;
    }
    if (effective.geometry_generation < 0)
    {
        effective.geometry_generation = spatial_.geometry_generation;
    }
    if (!effective.require_backend)
    {
        effective.require_backend = true;
        effective.backend = view->backend;
    }
#if defined(USE_CUDA) || defined(USE_HIP)
    if (!effective.require_same_producer_stream)
    {
        effective.require_same_producer_stream = true;
        effective.consumer_stream = view->producer_stream;
    }
#endif
    return Clustered_Validate_Spatial_View(*view, effective, failure_reason);
}
