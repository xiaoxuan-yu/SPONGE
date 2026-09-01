#include "pair_shift.h"

#include "internal.h"
#include "lifecycle.h"
#include "runtime.h"

namespace
{

using clustered_neighbor_runtime::Clustered_Minimum_Box_Face_Height;
using clustered_neighbor_runtime::Reserve_Device_Buffer;

constexpr int kClusteredGmxpackedPairShiftRefreshBlockSize = 128;

#ifndef USE_CPU
static bool Clustered_Gmxpacked_Should_Refresh_Pair_Shift_Metadata(
    const ClusteredNeighborProvider* provider, LTMatrix3 reciprocal_cell)
{
    if (provider == NULL)
    {
        return true;
    }
    const auto& spatial =
        ClusteredNeighborProviderInternal::Spatial(provider);
    const auto& pair_list =
        ClusteredNeighborProviderInternal::PairList(provider);
    const CLUSTERED_GMXPACKED_PAIR_SHIFT_CACHE_KEY cached_key = {
        pair_list.gmxpacked_pair_shift_metadata_payload_generation,
        pair_list.gmxpacked_pair_shift_metadata_geometry_generation,
        pair_list.gmxpacked_pair_shift_metadata_sci_numbers,
        pair_list.gmxpacked_pair_shift_metadata_cjpacked_numbers,
        pair_list.gmxpacked_pair_shift_metadata_exclusion_numbers,
        pair_list.gmxpacked_pair_shift_metadata_rcell};
    const CLUSTERED_GMXPACKED_PAIR_SHIFT_CACHE_KEY current_key = {
        pair_list.gmxpacked_compact_payload_generation,
        spatial.geometry_generation,
        pair_list.gmxpacked_sci_numbers,
        pair_list.gmxpacked_cjpacked_numbers,
        pair_list.gmxpacked_exclusion_numbers,
        reciprocal_cell};
    if (Clustered_Gmxpacked_Pair_Shift_Metadata_Should_Refresh(
            true, pair_list.pair_shift_bits.data != NULL,
            pair_list.gmxpacked_pair_shift_metadata_ready, cached_key,
            current_key))
    {
        return true;
    }
    return pair_list.gmxpacked_pair_shift_sci_safe_flags.data == NULL;
}

static void Refresh_Gmxpacked_Pair_Shift_Metadata(
    ClusteredNeighborProvider* provider, LTMatrix3 reciprocal_cell)
{
    Invalidate_Gmxpacked_Pair_Shift_Metadata(provider);
    if (provider == NULL)
    {
        return;
    }
    auto& spatial = ClusteredNeighborProviderInternal::Spatial(provider);
    auto& pair_list = ClusteredNeighborProviderInternal::PairList(provider);
    if (pair_list.gmxpacked_sci_numbers <= 0 ||
        pair_list.gmxpacked_cjpacked_numbers <= 0 ||
        pair_list.pair_shift_bits.data == NULL ||
        pair_list.gmxpacked_sci.data == NULL ||
        pair_list.gmxpacked_cjpacked.data == NULL ||
        spatial.clusters.fractional_centers.data == NULL ||
        spatial.clusters.fractional_extents.data == NULL)
    {
        return;
    }
    Reserve_Device_Buffer(pair_list.gmxpacked_sci_numbers,
                          &pair_list.gmxpacked_pair_shift_sci_safe_flags);
    int* d_sci_shift_safe_flags =
        pair_list.gmxpacked_pair_shift_sci_safe_flags.data;
    const int refresh_block_size =
        kClusteredGmxpackedPairShiftRefreshBlockSize;
    const float minimum_box_face_height =
        Clustered_Minimum_Box_Face_Height(reciprocal_cell);
    const bool periodic_image_dedup_required =
        minimum_box_face_height <= 0.0f ||
        ClusteredNeighborProviderInternal::CachedCutoff(provider) +
                    ClusteredNeighborProviderInternal::EffectiveRebuildSkin(
                        provider) +
                    2.0f * spatial.leaves
                               .periodic_image_max_fractional_extent_bound *
                        minimum_box_face_height >=
                0.5f * minimum_box_face_height;
    if (periodic_image_dedup_required)
    {
        Launch_Device_Kernel(
            Refresh_Gmxpacked_Pair_Shift_Bits,
            pair_list.gmxpacked_sci_numbers, refresh_block_size, 0, NULL,
            pair_list.gmxpacked_sci_numbers,
            spatial.superclusters.offsets.data,
            spatial.clusters.fractional_centers.data,
            spatial.clusters.fractional_extents.data,
            spatial.clusters.valid_masks.data,
            spatial.clusters.local_masks.data, pair_list.gmxpacked_sci.data,
            pair_list.gmxpacked_cjpacked.data,
            pair_list.gmxpacked_exclusions.data, pair_list.pair_shift_bits.data,
            d_sci_shift_safe_flags);
    }
    else
    {
        Launch_Device_Kernel(
            Refresh_Gmxpacked_Pair_Shift_Bits_Unique_Image,
            pair_list.gmxpacked_sci_numbers, refresh_block_size, 0, NULL,
            pair_list.gmxpacked_sci_numbers,
            spatial.superclusters.offsets.data,
            spatial.clusters.fractional_centers.data,
            spatial.clusters.fractional_extents.data,
            spatial.clusters.valid_masks.data,
            spatial.clusters.local_masks.data, pair_list.gmxpacked_sci.data,
            pair_list.gmxpacked_cjpacked.data,
            pair_list.gmxpacked_exclusions.data, pair_list.pair_shift_bits.data,
            d_sci_shift_safe_flags);
    }
    ClusteredNeighborProviderInternal::LeaseEpoch(provider) += 1;
    pair_list.gmxpacked_pair_shift_metadata_ready = true;
    pair_list.gmxpacked_pair_shift_metadata_sci_numbers =
        pair_list.gmxpacked_sci_numbers;
    pair_list.gmxpacked_pair_shift_metadata_cjpacked_numbers =
        pair_list.gmxpacked_cjpacked_numbers;
    pair_list.gmxpacked_pair_shift_metadata_exclusion_numbers =
        pair_list.gmxpacked_exclusion_numbers;
    pair_list.gmxpacked_pair_shift_metadata_payload_generation =
        pair_list.gmxpacked_compact_payload_generation;
    pair_list.gmxpacked_pair_shift_metadata_geometry_generation =
        spatial.geometry_generation;
    pair_list.gmxpacked_pair_shift_metadata_rcell = reciprocal_cell;
}
#endif

}  // namespace

void Refresh_Gmxpacked_Pair_Shift_Metadata_If_Needed(
    ClusteredNeighborProvider* provider, LTMatrix3 reciprocal_cell)
{
    if (provider == NULL)
    {
        return;
    }
#ifdef USE_CPU
    (void)provider;
    (void)reciprocal_cell;
#else
    if (!Clustered_Gmxpacked_Should_Refresh_Pair_Shift_Metadata(
            provider, reciprocal_cell))
    {
        return;
    }
    auto& pair_list = ClusteredNeighborProviderInternal::PairList(provider);
    Reserve_Device_Buffer(
        pair_list.gmxpacked_cjpacked_numbers * kClusteredJGroupSize,
        &pair_list.pair_shift_bits);
    Refresh_Gmxpacked_Pair_Shift_Metadata(provider, reciprocal_cell);
#endif
}
