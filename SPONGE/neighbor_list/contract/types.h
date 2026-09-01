#pragma once

#include <cstdint>

#include "../../common.h"

constexpr int kClusteredClusterSize = 8;
constexpr int kClusteredSuperClusterClusters = 8;
constexpr int kClusteredWarpSplitCount = 2;
constexpr int kClusteredSplitJClusterSize =
    kClusteredClusterSize / kClusteredWarpSplitCount;
constexpr int kClusteredJGroupSize = kClusteredSplitJClusterSize;
constexpr int kClusteredGmxpackedExclusionPairCount =
    kClusteredSplitJClusterSize * kClusteredClusterSize;
constexpr int kClusteredShiftCount = 27;
constexpr int kClusteredCentralShiftId = 13;
constexpr int kClusteredPairShiftBits = 5;
constexpr uint64_t kClusteredPairShiftMask =
    (1ull << kClusteredPairShiftBits) - 1ull;
constexpr int kClusteredPairActiveMaskBits = kClusteredSuperClusterClusters;
constexpr int kClusteredPairActiveMaskOffset =
    kClusteredSuperClusterClusters * kClusteredPairShiftBits;
constexpr int kClusteredPairActiveMarkerOffset =
    kClusteredPairActiveMaskOffset +
    kClusteredWarpSplitCount * kClusteredPairActiveMaskBits;
constexpr uint64_t kClusteredPairActiveMarker =
    1ull << kClusteredPairActiveMarkerOffset;
static_assert(kClusteredPairActiveMarkerOffset < 64,
              "Clustered pair shift bit packing exceeds 64 bits.");

struct CLUSTERED_GMXPACKED_SCI
{
    int supercluster_id = 0;
    int shift_id = kClusteredCentralShiftId;
    int cjpacked_begin = 0;
    int cjpacked_end = 0;
};

struct CLUSTERED_GMXPACKED_SPLIT
{
    unsigned int imask = 0u;
    int exclusion_index = 0;
};

struct CLUSTERED_GMXPACKED_CJ
{
    int cj[kClusteredJGroupSize] = { -1, -1, -1, -1 };
    CLUSTERED_GMXPACKED_SPLIT split[kClusteredWarpSplitCount] = {};
};
static_assert(sizeof(CLUSTERED_GMXPACKED_CJ) == 32,
              "Unexpected clustered gmxpacked cj size.");

struct CLUSTERED_GMXPACKED_EXCLUSION
{
    unsigned int pair[kClusteredGmxpackedExclusionPairCount] = {};
};
static_assert(sizeof(CLUSTERED_GMXPACKED_EXCLUSION) ==
                  sizeof(unsigned int) * kClusteredGmxpackedExclusionPairCount,
              "Unexpected clustered gmxpacked exclusion size.");

enum class CLUSTERED_ENDPOINT_ORIENTATION : unsigned char
{
    NATIVE_I = 0,
    TRANSPOSED_J = 1
};

// One structural reference addresses an authoritative CJ/jm tile. The
// i_cluster_mask retains the fixed-width cluster mask; it is never expanded
// into accepted atom-pair rows.
struct CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE
{
    int sci_id = -1;
    int cjpacked_id = -1;
    unsigned int i_cluster_mask = 0u;
    unsigned char jm = 0u;
    CLUSTERED_ENDPOINT_ORIENTATION orientation =
        CLUSTERED_ENDPOINT_ORIENTATION::NATIVE_I;
    unsigned short reserved = 0u;
};
static_assert(sizeof(CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE) == 16,
              "Unexpected clustered endpoint-reference size.");

struct CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE
{
    int sci_id = -1;
    int shift_id = kClusteredCentralShiftId;
    int supercluster_id = -1;
    int cluster_j = -1;
    int split_id = 0;
    unsigned int imask = 0u;
    unsigned int valid_mask_j = 0u;
    unsigned int local_mask_j = 0u;
    unsigned int pair_exclusion_words[kClusteredGmxpackedExclusionPairCount] =
        {};
    int source_order = 0;
};

struct CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE
{
    int sci_id = -1;
    int shift_id = kClusteredCentralShiftId;
    int supercluster_id = -1;
    int cluster_j = -1;
    unsigned int split_imask[kClusteredWarpSplitCount] = {};
    unsigned int valid_mask_j = 0u;
    unsigned int local_mask_j = 0u;
    unsigned int pair_exclusion_words[kClusteredWarpSplitCount]
                                      [kClusteredGmxpackedExclusionPairCount] =
                                          {};
    int source_order_begin = 0;
    int source_order_end = 0;
};

struct CLUSTERED_J_ENTRY
{
    int supercluster_id = 0;
    int shift_id = kClusteredCentralShiftId;
    int cluster_j = -1;
    unsigned int imask[kClusteredWarpSplitCount] = {};
    int excl_ind[kClusteredWarpSplitCount * kClusteredSuperClusterClusters] =
        {};
};

enum class CLUSTERED_SPATIAL_BACKEND : int
{
    CPU = 0,
    CUDA = 1,
    HIP = 2
};

enum class CLUSTERED_SPATIAL_READINESS_SCOPE : int
{
    HOST_COMPLETE = 0,
    PRODUCER_STREAM_ORDERED = 1
};

struct CLUSTERED_SPATIAL_VIEW
{
    bool ready = false;
    CLUSTERED_SPATIAL_BACKEND backend = CLUSTERED_SPATIAL_BACKEND::CPU;
    CLUSTERED_SPATIAL_READINESS_SCOPE readiness_scope =
        CLUSTERED_SPATIAL_READINESS_SCOPE::HOST_COMPLETE;
    const void* producer_stream = nullptr;
    long long provider_incarnation = -1;
    long long lease_epoch = -1;
    long long gmxpacked_payload_generation = -1;
    long long source_generation = -1;
    long long geometry_generation = -1;

    int cluster_size = 0;
    int super_cluster_clusters = 0;
    int local_atom_numbers = 0;
    int direct_local_atom_numbers = 0;
    int ghost_numbers = 0;
    int total_atom_numbers = 0;
    int padded_total_atom_numbers = 0;
    int cluster_numbers = 0;
    int super_cluster_numbers = 0;
    int gmxpacked_sci_numbers = 0;
    int gmxpacked_cjpacked_numbers = 0;
    int gmxpacked_exclusion_numbers = 0;

    float cached_cutoff = -1.0f;
    float rebuild_skin = 0.0f;
    bool gmxpacked_endpoint_incidence_ready = false;
    long long endpoint_incidence_provider_incarnation = -1;
    long long endpoint_incidence_payload_generation = -1;
    int endpoint_incidence_sci_numbers = 0;
    int endpoint_incidence_cjpacked_numbers = 0;
    int endpoint_incidence_super_cluster_numbers = 0;
    int endpoint_incidence_reference_numbers = 0;
    int endpoint_incidence_offset_tail = 0;
    bool pair_shift_metadata_ready = false;
    long long pair_shift_payload_generation = -1;
    long long pair_shift_geometry_generation = -1;
    int pair_shift_sci_numbers = 0;
    int pair_shift_cjpacked_numbers = 0;
    int pair_shift_exclusion_numbers = 0;
    LTMatrix3 pair_shift_rcell = {};

    const int* atom_local = nullptr;
    const int* sort_permutation = nullptr;
    const int* cluster_offsets = nullptr;
    const unsigned int* cluster_valid_masks = nullptr;
    const unsigned int* cluster_local_masks = nullptr;
    const VECTOR* cluster_centers = nullptr;
    const VECTOR* cluster_extents = nullptr;
    const int* super_cluster_offsets = nullptr;
    const int* gmxpacked_endpoint_incidence_offsets = nullptr;
    const CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE*
        gmxpacked_endpoint_incidence_references = nullptr;

    const CLUSTERED_GMXPACKED_SCI* gmxpacked_sci = nullptr;
    const CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked = nullptr;
    const CLUSTERED_GMXPACKED_EXCLUSION* gmxpacked_exclusions = nullptr;
    const uint64_t* pair_shift_bits = nullptr;
    const int* gmxpacked_sci_shift_safe_flags = nullptr;
};

struct CLUSTERED_ENDPOINT_INCIDENCE_RANGE
{
    int begin = 0;
    int end = 0;
};

struct CLUSTERED_GMXPACKED_CENTER_CURSOR
{
    int center_cluster = -1;
    int center_supercluster = -1;
    int center_i_local = -1;
    int next_reference = 0;
    int end_reference = 0;
};

struct CLUSTERED_GMXPACKED_CENTER_TILE
{
    int sci_id = -1;
    int cjpacked_id = -1;
    int center_cluster = -1;
    int neighbor_cluster_base = -1;
    unsigned int neighbor_cluster_mask = 0u;
    unsigned char jm = 0u;
    unsigned char original_i_local = 0u;
    CLUSTERED_ENDPOINT_ORIENTATION orientation =
        CLUSTERED_ENDPOINT_ORIENTATION::NATIVE_I;
    unsigned char reserved = 0u;
};

// Compatibility aliases keep the current LJ kernels byte-for-byte equivalent
// while ownership of the structural contract moves out of the LJ module.
using LJ_CLUSTERED_GMXPACKED_SCI = CLUSTERED_GMXPACKED_SCI;
using LJ_CLUSTERED_GMXPACKED_SPLIT = CLUSTERED_GMXPACKED_SPLIT;
using LJ_CLUSTERED_GMXPACKED_CJ = CLUSTERED_GMXPACKED_CJ;
using LJ_CLUSTERED_GMXPACKED_EXCLUSION = CLUSTERED_GMXPACKED_EXCLUSION;
using LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE =
    CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE;
using LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE =
    CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE;
using LJ_CLUSTERED_J_ENTRY = CLUSTERED_J_ENTRY;
