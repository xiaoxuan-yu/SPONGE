#pragma once

#include <vector>

#include "types.h"
#include "traversal.cuh"

struct CLUSTERED_GMXPACKED_PAIR_SHIFT_CACHE_KEY
{
    long long gmxpacked_payload_generation = -1;
    long long geometry_generation = -1;
    int sci_numbers = 0;
    int cjpacked_numbers = 0;
    int exclusion_numbers = 0;
    LTMatrix3 rcell = {};
};

inline bool Clustered_Gmxpacked_Pair_Shift_Cache_Key_Matches(
    const CLUSTERED_GMXPACKED_PAIR_SHIFT_CACHE_KEY& cached,
    const CLUSTERED_GMXPACKED_PAIR_SHIFT_CACHE_KEY& current)
{
    return cached.gmxpacked_payload_generation >= 0 &&
           cached.gmxpacked_payload_generation ==
               current.gmxpacked_payload_generation &&
           cached.geometry_generation >= 0 &&
           cached.geometry_generation == current.geometry_generation &&
           cached.sci_numbers == current.sci_numbers &&
           cached.cjpacked_numbers == current.cjpacked_numbers &&
           cached.exclusion_numbers == current.exclusion_numbers &&
           cached.rcell.a11 == current.rcell.a11 &&
           cached.rcell.a21 == current.rcell.a21 &&
           cached.rcell.a22 == current.rcell.a22 &&
           cached.rcell.a31 == current.rcell.a31 &&
           cached.rcell.a32 == current.rcell.a32 &&
           cached.rcell.a33 == current.rcell.a33;
}

inline bool Clustered_Gmxpacked_Pair_Shift_Metadata_Should_Refresh(
    bool cache_enabled, bool pair_shift_storage_ready, bool metadata_ready,
    const CLUSTERED_GMXPACKED_PAIR_SHIFT_CACHE_KEY& cached,
    const CLUSTERED_GMXPACKED_PAIR_SHIFT_CACHE_KEY& current)
{
    return !cache_enabled || !pair_shift_storage_ready || !metadata_ready ||
           current.sci_numbers <= 0 || current.cjpacked_numbers <= 0 ||
           current.exclusion_numbers <= 0 ||
           !Clustered_Gmxpacked_Pair_Shift_Cache_Key_Matches(cached, current);
}

struct CLUSTERED_GMXPACKED_ENDPOINT_INCIDENCE_HOST
{
    bool ready = false;
    long long provider_incarnation = -1;
    long long gmxpacked_payload_generation = -1;
    int super_cluster_numbers = 0;
    std::vector<int> offsets;
    std::vector<CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE> references;

    void Clear()
    {
        ready = false;
        provider_incarnation = -1;
        gmxpacked_payload_generation = -1;
        super_cluster_numbers = 0;
        offsets.clear();
        references.clear();
    }
};

bool Clustered_Build_Gmxpacked_Endpoint_Incidence_Host(
    long long provider_incarnation, long long gmxpacked_payload_generation,
    int cluster_numbers, int super_cluster_numbers,
    const int* super_cluster_offsets, int sci_numbers,
    const CLUSTERED_GMXPACKED_SCI* sci_entries, int cjpacked_numbers,
    const CLUSTERED_GMXPACKED_CJ* cjpacked_entries,
    CLUSTERED_GMXPACKED_ENDPOINT_INCIDENCE_HOST* incidence,
    const char** failure_reason = nullptr);

struct CLUSTERED_SPATIAL_VIEW_REQUIREMENTS
{
    int local_atom_numbers = -1;
    int ghost_numbers = -1;
    float cutoff = -1.0f;
    long long provider_incarnation = -1;
    long long lease_epoch = -1;
    long long gmxpacked_payload_generation = -1;
    long long source_generation = -1;
    long long geometry_generation = -1;
    bool require_backend = false;
    CLUSTERED_SPATIAL_BACKEND backend = CLUSTERED_SPATIAL_BACKEND::CPU;
    bool require_same_producer_stream = false;
    const void* consumer_stream = nullptr;
    bool require_all_local_atoms = true;
    bool require_gmxpacked_payload = false;
    bool require_gmxpacked_endpoint_incidence = false;
    bool require_pair_shift_metadata = false;
    bool require_pair_shift_rcell = false;
    LTMatrix3 pair_shift_rcell = {};
};

bool Clustered_Validate_Spatial_View(
    const CLUSTERED_SPATIAL_VIEW& view,
    const CLUSTERED_SPATIAL_VIEW_REQUIREMENTS& requirements,
    const char** failure_reason = nullptr);
