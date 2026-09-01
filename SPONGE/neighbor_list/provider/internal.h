#pragma once

#include "provider.h"

struct ClusteredNeighborProviderInternal
{
    static CONTROLLER*& Controller(ClusteredNeighborProvider* provider)
    {
        return provider->controller_;
    }
    static CONTROLLER* Controller(const ClusteredNeighborProvider* provider)
    {
        return provider->controller_;
    }

    static int& WorkingDevice(ClusteredNeighborProvider* provider)
    {
        return provider->working_device_;
    }
    static int WorkingDevice(const ClusteredNeighborProvider* provider)
    {
        return provider->working_device_;
    }

    static ClusteredBuildConfig& Config(ClusteredNeighborProvider* provider)
    {
        return provider->config_;
    }
    static const ClusteredBuildConfig& Config(
        const ClusteredNeighborProvider* provider)
    {
        return provider->config_;
    }

    static ClusteredDomainBinding& Domain(ClusteredNeighborProvider* provider)
    {
        return provider->domain_;
    }
    static const ClusteredDomainBinding& Domain(
        const ClusteredNeighborProvider* provider)
    {
        return provider->domain_;
    }

    static bool& RebuildDirty(ClusteredNeighborProvider* provider)
    {
        return provider->rebuild_dirty_;
    }
    static bool RebuildDirty(const ClusteredNeighborProvider* provider)
    {
        return provider->rebuild_dirty_;
    }

    static int& CachedBuildStep(ClusteredNeighborProvider* provider)
    {
        return provider->cached_build_step_;
    }
    static int CachedBuildStep(const ClusteredNeighborProvider* provider)
    {
        return provider->cached_build_step_;
    }

    static float& EffectiveRebuildSkin(ClusteredNeighborProvider* provider)
    {
        return provider->effective_rebuild_skin_;
    }
    static float EffectiveRebuildSkin(const ClusteredNeighborProvider* provider)
    {
        return provider->effective_rebuild_skin_;
    }

    static float& CachedCutoff(ClusteredNeighborProvider* provider)
    {
        return provider->cached_cutoff_;
    }
    static float CachedCutoff(const ClusteredNeighborProvider* provider)
    {
        return provider->cached_cutoff_;
    }

    static uint64_t& ProviderIncarnation(
        ClusteredNeighborProvider* provider)
    {
        return provider->provider_incarnation_;
    }
    static uint64_t ProviderIncarnation(
        const ClusteredNeighborProvider* provider)
    {
        return provider->provider_incarnation_;
    }

    static uint64_t& LeaseEpoch(ClusteredNeighborProvider* provider)
    {
        return provider->lease_epoch_;
    }
    static uint64_t LeaseEpoch(const ClusteredNeighborProvider* provider)
    {
        return provider->lease_epoch_;
    }

    static ClusteredSpatialLayout& Spatial(ClusteredNeighborProvider* provider)
    {
        return provider->spatial_;
    }
    static const ClusteredSpatialLayout& Spatial(
        const ClusteredNeighborProvider* provider)
    {
        return provider->spatial_;
    }

    static ClusteredPairList& PairList(ClusteredNeighborProvider* provider)
    {
        return provider->pair_list_;
    }
    static const ClusteredPairList& PairList(
        const ClusteredNeighborProvider* provider)
    {
        return provider->pair_list_;
    }

    static ClusteredBuildWorkspace& Workspace(
        ClusteredNeighborProvider* provider)
    {
        return provider->workspace_;
    }
    static const ClusteredBuildWorkspace& Workspace(
        const ClusteredNeighborProvider* provider)
    {
        return provider->workspace_;
    }
};
