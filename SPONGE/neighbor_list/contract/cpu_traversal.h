#pragma once

#include <cstdint>

#include "view.h"

inline uint64_t Clustered_Gmxpacked_CPU_Pair_Mask(
    const CLUSTERED_GMXPACKED_CJ& packed, const unsigned int packed_bit,
    const CLUSTERED_GMXPACKED_EXCLUSION* exclusions,
    const int i_cluster_size = kClusteredClusterSize)
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
        for (int split_j_lane = 0; split_j_lane < kClusteredSplitJClusterSize;
             split_j_lane += 1)
        {
            const int j_lane =
                split * kClusteredSplitJClusterSize + split_j_lane;
            for (int i_lane = 0; i_lane < i_cluster_size; i_lane += 1)
            {
                if ((exclusion.pair[split_j_lane * i_cluster_size + i_lane] &
                     packed_bit) != 0u)
                {
                    pair_mask |= 1ull
                                 << (i_lane * kClusteredClusterSize + j_lane);
                }
            }
        }
    }
    return pair_mask;
}

struct CLUSTERED_GMXPACKED_CPU_J_CANDIDATE
{
    int sci_id = -1;
    int packed_id = -1;
    int jm = -1;
    int cluster_j = -1;
    int j_lane = -1;
    int sorted_j = -1;
    int shift_id = kClusteredCentralShiftId;
    bool j_is_local = false;
};

struct CLUSTERED_GMXPACKED_CPU_PAIR_CANDIDATE
{
    CLUSTERED_GMXPACKED_CPU_J_CANDIDATE j = {};
    int cluster_i = -1;
    int i_local = -1;
    int i_lane = -1;
    int sorted_i = -1;
    unsigned int packed_bit = 0u;
    bool i_is_local = false;
};

struct CLUSTERED_GMXPACKED_CPU_I_TILE_CANDIDATE
{
    int sci_id = -1;
    int packed_id = -1;
    int jm = -1;
    int cluster_i = -1;
    int cluster_j = -1;
    int i_local = -1;
    int shift_id = kClusteredCentralShiftId;
    unsigned int packed_bit = 0u;
    uint64_t pair_mask = 0ull;
};

struct CLUSTERED_GMXPACKED_CPU_LOCAL_I_CANDIDATE
{
    int sci_id = -1;
    int cluster_i = -1;
    int i_local = -1;
    int i_lane = -1;
    int sorted_i = -1;
    int shift_id = kClusteredCentralShiftId;
};

struct CLUSTERED_GMXPACKED_CPU_J_TILE_CANDIDATE
{
    int sci_id = -1;
    int packed_id = -1;
    int jm = -1;
    int cluster_i = -1;
    int cluster_j = -1;
    int i_local = -1;
    int i_lane = -1;
    int sorted_i = -1;
    int shift_id = kClusteredCentralShiftId;
    unsigned int packed_bit = 0u;
    unsigned int active_j_mask = 0u;
};

inline int Clustered_Gmxpacked_CPU_J_Tile_Pair_Shift_Id(
    const CLUSTERED_SPATIAL_VIEW& view,
    const CLUSTERED_GMXPACKED_CPU_J_TILE_CANDIDATE& tile)
{
    const CLUSTERED_GMXPACKED_SCI& sci_entry = view.gmxpacked_sci[tile.sci_id];
    return Clustered_Gmxpacked_Pair_Shift_Id(view, sci_entry, tile.packed_id,
                                             tile.jm, tile.i_local);
}

inline unsigned int Clustered_Gmxpacked_CPU_Active_I_Cluster_Mask(
    const CLUSTERED_GMXPACKED_CJ& packed, const int jm, const int j_lane,
    const int i_lane, const CLUSTERED_GMXPACKED_EXCLUSION* exclusions,
    const int i_cluster_size = kClusteredClusterSize)
{
    const int split = j_lane / kClusteredSplitJClusterSize;
    const int split_j_lane = j_lane - split * kClusteredSplitJClusterSize;
    const unsigned int effective_imask = Clustered_Gmxpacked_Effective_Imask(
        packed, exclusions, split, split_j_lane, i_lane, i_cluster_size);
    return effective_imask >>
           static_cast<unsigned int>(jm * kClusteredSuperClusterClusters);
}

// Enumerates active, structurally valid pair candidates for one SCI while
// preserving the existing CPU consumer order:
// packed -> jm -> j_lane -> i_lane -> i_local. Ownership, self-pair removal,
// displacement, cutoff and model math remain consumer responsibilities.
// begin_j may return false to reject a complete J candidate before I lanes
// are expanded. The caller owns SCI scheduling (including any OpenMP policy).
template <class BeginJ, class ConsumePair, class EndJ>
inline void Clustered_Gmxpacked_CPU_For_Each_Pair_In_SCI(
    const CLUSTERED_SPATIAL_VIEW& view, const int sci_id, BeginJ&& begin_j,
    ConsumePair&& consume_pair, EndJ&& end_j)
{
    const CLUSTERED_GMXPACKED_SCI sci_entry = view.gmxpacked_sci[sci_id];
    const int cluster_i_begin =
        view.super_cluster_offsets[sci_entry.supercluster_id];
    const int cluster_i_end =
        view.super_cluster_offsets[sci_entry.supercluster_id + 1];
    const int cluster_i_numbers = cluster_i_end - cluster_i_begin;

    for (int packed_id = sci_entry.cjpacked_begin;
         packed_id < sci_entry.cjpacked_end; packed_id += 1)
    {
        const CLUSTERED_GMXPACKED_CJ& packed =
            view.gmxpacked_cjpacked[packed_id];
        for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
        {
            const int cluster_j = packed.cj[jm];
            if (cluster_j < 0)
            {
                continue;
            }
            for (int j_lane = 0; j_lane < view.cluster_size; j_lane += 1)
            {
                if (!Clustered_Lane_Is_Valid(
                        view.cluster_valid_masks[cluster_j], j_lane))
                {
                    continue;
                }
                CLUSTERED_GMXPACKED_CPU_J_CANDIDATE j_candidate;
                j_candidate.sci_id = sci_id;
                j_candidate.packed_id = packed_id;
                j_candidate.jm = jm;
                j_candidate.cluster_j = cluster_j;
                j_candidate.j_lane = j_lane;
                j_candidate.sorted_j = view.cluster_offsets[cluster_j] + j_lane;
                j_candidate.shift_id = sci_entry.shift_id;
                j_candidate.j_is_local = Clustered_Lane_Is_Local(
                    view.cluster_local_masks[cluster_j], j_lane);
                if (!begin_j(j_candidate))
                {
                    continue;
                }

                for (int i_lane = 0; i_lane < view.cluster_size; i_lane += 1)
                {
                    const unsigned int active_i_mask =
                        Clustered_Gmxpacked_CPU_Active_I_Cluster_Mask(
                            packed, jm, j_lane, i_lane,
                            view.gmxpacked_exclusions, view.cluster_size);
                    if (active_i_mask == 0u)
                    {
                        continue;
                    }
                    for (int i_local = 0; i_local < cluster_i_numbers;
                         i_local += 1)
                    {
                        const unsigned int packed_bit =
                            1u << static_cast<unsigned int>(
                                jm * kClusteredSuperClusterClusters + i_local);
                        if ((active_i_mask &
                             (1u << static_cast<unsigned int>(i_local))) == 0u)
                        {
                            continue;
                        }
                        const int cluster_i = cluster_i_begin + i_local;
                        if (!Clustered_Lane_Is_Valid(
                                view.cluster_valid_masks[cluster_i], i_lane))
                        {
                            continue;
                        }
                        CLUSTERED_GMXPACKED_CPU_PAIR_CANDIDATE pair_candidate;
                        pair_candidate.j = j_candidate;
                        pair_candidate.cluster_i = cluster_i;
                        pair_candidate.i_local = i_local;
                        pair_candidate.i_lane = i_lane;
                        pair_candidate.sorted_i =
                            view.cluster_offsets[cluster_i] + i_lane;
                        pair_candidate.packed_bit = packed_bit;
                        pair_candidate.i_is_local = Clustered_Lane_Is_Local(
                            view.cluster_local_masks[cluster_i], i_lane);
                        consume_pair(pair_candidate);
                    }
                }
                end_j(j_candidate);
            }
        }
    }
}

// Row-oriented CSR builders require i_local to remain outside the packed/J
// loops so their per-center insertion order is stable. This adapter therefore
// emits active I/J tiles without expanding atom lanes.
template <class ConsumeTile>
inline void Clustered_Gmxpacked_CPU_For_Each_I_Tile_In_SCI(
    const CLUSTERED_SPATIAL_VIEW& view, const int sci_id,
    ConsumeTile&& consume_tile)
{
    const CLUSTERED_GMXPACKED_SCI sci_entry = view.gmxpacked_sci[sci_id];
    const int cluster_i_begin =
        view.super_cluster_offsets[sci_entry.supercluster_id];
    const int cluster_i_end =
        view.super_cluster_offsets[sci_entry.supercluster_id + 1];
    const int cluster_i_numbers = cluster_i_end - cluster_i_begin;

    for (int i_local = 0; i_local < cluster_i_numbers; i_local += 1)
    {
        const int cluster_i = cluster_i_begin + i_local;
        for (int packed_id = sci_entry.cjpacked_begin;
             packed_id < sci_entry.cjpacked_end; packed_id += 1)
        {
            const CLUSTERED_GMXPACKED_CJ& packed =
                view.gmxpacked_cjpacked[packed_id];
            for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
            {
                const int cluster_j = packed.cj[jm];
                if (cluster_j < 0)
                {
                    continue;
                }
                const unsigned int packed_bit =
                    1u << static_cast<unsigned int>(
                        jm * kClusteredSuperClusterClusters + i_local);
                const uint64_t pair_mask = Clustered_Gmxpacked_CPU_Pair_Mask(
                    packed, packed_bit, view.gmxpacked_exclusions,
                    view.cluster_size);
                if (pair_mask == 0ull)
                {
                    continue;
                }
                CLUSTERED_GMXPACKED_CPU_I_TILE_CANDIDATE tile;
                tile.sci_id = sci_id;
                tile.packed_id = packed_id;
                tile.jm = jm;
                tile.cluster_i = cluster_i;
                tile.cluster_j = cluster_j;
                tile.i_local = i_local;
                tile.shift_id = sci_entry.shift_id;
                tile.packed_bit = packed_bit;
                tile.pair_mask = pair_mask;
                consume_tile(tile);
            }
        }
    }
}

// Direct LJ-style CPU consumers aggregate over one local I atom while walking
// packed J tiles. These two adapters preserve that I-major order and expose a
// per-I-cluster pair shift without imposing force or ownership semantics.
template <class ConsumeI>
inline void Clustered_Gmxpacked_CPU_For_Each_Local_I_In_SCI(
    const CLUSTERED_SPATIAL_VIEW& view, const int sci_id, ConsumeI&& consume_i)
{
    const CLUSTERED_GMXPACKED_SCI sci_entry = view.gmxpacked_sci[sci_id];
    const int cluster_i_begin =
        view.super_cluster_offsets[sci_entry.supercluster_id];
    const int cluster_i_end =
        view.super_cluster_offsets[sci_entry.supercluster_id + 1];
    for (int cluster_i = cluster_i_begin; cluster_i < cluster_i_end;
         cluster_i += 1)
    {
        const int i_local = cluster_i - cluster_i_begin;
        for (int i_lane = 0; i_lane < view.cluster_size; i_lane += 1)
        {
            if (!Clustered_Lane_Is_Valid(view.cluster_valid_masks[cluster_i],
                                         i_lane) ||
                !Clustered_Lane_Is_Local(view.cluster_local_masks[cluster_i],
                                         i_lane))
            {
                continue;
            }
            CLUSTERED_GMXPACKED_CPU_LOCAL_I_CANDIDATE candidate;
            candidate.sci_id = sci_id;
            candidate.cluster_i = cluster_i;
            candidate.i_local = i_local;
            candidate.i_lane = i_lane;
            candidate.sorted_i = view.cluster_offsets[cluster_i] + i_lane;
            candidate.shift_id = sci_entry.shift_id;
            consume_i(candidate);
        }
    }
}

template <class ConsumeTile>
inline void Clustered_Gmxpacked_CPU_For_Each_J_Tile_For_Local_I(
    const CLUSTERED_SPATIAL_VIEW& view,
    const CLUSTERED_GMXPACKED_CPU_LOCAL_I_CANDIDATE& pair_i,
    ConsumeTile&& consume_tile)
{
    const CLUSTERED_GMXPACKED_SCI sci_entry = view.gmxpacked_sci[pair_i.sci_id];
    for (int packed_id = sci_entry.cjpacked_begin;
         packed_id < sci_entry.cjpacked_end; packed_id += 1)
    {
        const CLUSTERED_GMXPACKED_CJ& packed =
            view.gmxpacked_cjpacked[packed_id];
        for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
        {
            const int cluster_j = packed.cj[jm];
            if (cluster_j < 0)
            {
                continue;
            }
            const unsigned int packed_bit =
                1u << static_cast<unsigned int>(
                    jm * kClusteredSuperClusterClusters + pair_i.i_local);
            const uint64_t pair_mask = Clustered_Gmxpacked_CPU_Pair_Mask(
                packed, packed_bit, view.gmxpacked_exclusions,
                view.cluster_size);
            const unsigned int active_j_mask =
                static_cast<unsigned int>(
                    pair_mask >> (pair_i.i_lane * kClusteredClusterSize)) &
                view.cluster_valid_masks[cluster_j];
            if (active_j_mask == 0u)
            {
                continue;
            }
            CLUSTERED_GMXPACKED_CPU_J_TILE_CANDIDATE tile;
            tile.sci_id = pair_i.sci_id;
            tile.packed_id = packed_id;
            tile.jm = jm;
            tile.cluster_i = pair_i.cluster_i;
            tile.cluster_j = cluster_j;
            tile.i_local = pair_i.i_local;
            tile.i_lane = pair_i.i_lane;
            tile.sorted_i = pair_i.sorted_i;
            tile.shift_id = sci_entry.shift_id;
            tile.packed_bit = packed_bit;
            tile.active_j_mask = active_j_mask;
            consume_tile(tile);
        }
    }
}
