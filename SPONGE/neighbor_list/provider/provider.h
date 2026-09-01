#pragma once

#include "state.h"

struct ClusteredNeighborProviderInternal;

struct ClusteredGatherBinding
{
    uint64_t provider_incarnation = 0;
    uint64_t geometry_generation = 0;
    int total_atom_numbers = 0;
    int cluster_numbers = 0;
    const int* sort_permutation = nullptr;
    const int* cluster_offsets = nullptr;
    VECTOR* cluster_centers = nullptr;
    VECTOR* cluster_fractional_centers = nullptr;
    VECTOR* cluster_fractional_extents = nullptr;
};

class ClusteredNeighborProvider
{
   public:
    ClusteredNeighborProvider() = default;
    ClusteredNeighborProvider(const ClusteredNeighborProvider&) = delete;
    ClusteredNeighborProvider& operator=(const ClusteredNeighborProvider&) =
        delete;

    void Initialize(CONTROLLER* controller, const ClusteredBuildConfig& config);
    void BindDomain(const ClusteredDomainBinding& domain);
    void Build(const ClusteredBuildRequest& request);
    bool AcquireView(
        const CLUSTERED_SPATIAL_VIEW_REQUIREMENTS& requirements,
        CLUSTERED_SPATIAL_VIEW* view,
        const char** failure_reason = nullptr) const;
    bool AcquireGatherBinding(ClusteredGatherBinding* binding);
    bool IsGatheredGeometryCurrent(uint64_t provider_incarnation,
                                   uint64_t geometry_generation) const;
    void BindWorkingDevice();
    void PublishGatheredGeometry(ClusteredGatherBinding* binding);
    void Clear();

    bool IsInitialized() const { return initialized_; }
    int TotalAtomNumbers() const { return spatial_.total_atom_numbers; }

private:
    friend struct ClusteredNeighborProviderInternal;

    struct BuildPayloadInput;

    void BuildInternal(const VECTOR* coordinates, LTMatrix3 cell,
                       LTMatrix3 reciprocal_cell, float cutoff,
                       bool need_endpoint_incidence);
    void BuildPayload(const BuildPayloadInput& input);
#ifdef USE_CPU
    bool BuildPayloadCpu(const BuildPayloadInput& input);
#else
    bool BuildPayloadGpu(const BuildPayloadInput& input);
#endif
    void RetireSpatialProviderLifetime()
    {
        pair_list_.gmxpacked_endpoint_incidence_ready = false;
        pair_list_.gmxpacked_endpoint_incidence_provider_incarnation = -1;
        pair_list_.gmxpacked_endpoint_incidence_payload_generation = -1;
        pair_list_.gmxpacked_endpoint_incidence_sci_numbers = 0;
        pair_list_.gmxpacked_endpoint_incidence_cjpacked_numbers = 0;
        pair_list_.gmxpacked_endpoint_incidence_super_cluster_numbers = 0;
        pair_list_.gmxpacked_endpoint_incidence_reference_numbers = 0;
        pair_list_.gmxpacked_endpoint_incidence_offset_tail = 0;
        provider_incarnation_ += 1;
        lease_epoch_ += 1;
        pair_list_.gmxpacked_compact_payload_generation += 1;
        spatial_.geometry_generation += 1;
    }

    bool initialized_ = false;
    CONTROLLER* controller_ = nullptr;
    int working_device_ = 0;
    ClusteredBuildConfig config_ = {};
    ClusteredDomainBinding domain_ = {};

    uint64_t provider_incarnation_ = 1;
    uint64_t lease_epoch_ = 0;
    bool rebuild_dirty_ = true;
    int cached_build_step_ = -1;
    float effective_rebuild_skin_ = 10.0f;
    float cached_cutoff_ = -1.0f;

    ClusteredSpatialLayout spatial_ = {};
    ClusteredPairList pair_list_ = {};
    ClusteredBuildWorkspace workspace_ = {};
};
