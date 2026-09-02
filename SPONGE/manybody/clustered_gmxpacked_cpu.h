#pragma once

#include <cstdint>

#include "../neighbor_list/contract/view.h"

inline uint64_t Clustered_Gmxpacked_CPU_Pair_Mask(
    const CLUSTERED_GMXPACKED_CJ& packed, const unsigned int packed_bit,
    const CLUSTERED_GMXPACKED_EXCLUSION* exclusions)
{
    constexpr uint64_t split_pair_masks[kClusteredWarpSplitCount] = {
        0x0f0f0f0f0f0f0f0full,
        0xf0f0f0f0f0f0f0f0ull,
    };
    uint64_t pair_mask = 0ull;
    for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
    {
        const CLUSTERED_GMXPACKED_SPLIT& split_entry = packed.split[split];
        if ((split_entry.imask & packed_bit) == 0u)
        {
            continue;
        }
        if (split_entry.exclusion_index == 0)
        {
            pair_mask |= split_pair_masks[split];
            continue;
        }
        const CLUSTERED_GMXPACKED_EXCLUSION& exclusion =
            exclusions[split_entry.exclusion_index];
        for (int split_j_lane = 0;
             split_j_lane < kClusteredSplitJClusterSize;
             split_j_lane += 1)
        {
            const int j_lane =
                split * kClusteredSplitJClusterSize + split_j_lane;
            for (int i_lane = 0; i_lane < kClusteredClusterSize;
                 i_lane += 1)
            {
                if ((exclusion.pair[split_j_lane * kClusteredClusterSize +
                                    i_lane] &
                     packed_bit) != 0u)
                {
                    pair_mask |= 1ull <<
                                 (i_lane * kClusteredClusterSize + j_lane);
                }
            }
        }
    }
    return pair_mask;
}
