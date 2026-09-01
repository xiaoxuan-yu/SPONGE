#include "cpu_builder.h"

#ifdef USE_CPU

#include <algorithm>
#include <array>

namespace
{

constexpr int kMaxSuperClusterClusters = kClusteredSuperClusterClusters;
constexpr int kMaxJGroupSize = kClusteredJGroupSize;

#include "primitives/geometry.cuh"

struct HostClusteredJRecord
{
    int cluster_j = -1;
    unsigned int imask = 0u;
    std::array<unsigned long long, kMaxSuperClusterClusters> exclusion_masks =
        {};
};

static inline bool ExclusionListContains(int atom_i, int atom_j,
                                         const int* excluded_list_start,
                                         const int* excluded_list,
                                         const int* excluded_numbers)
{
    const int exclude_start = excluded_list_start[atom_i];
    const int exclude_count = excluded_numbers[atom_i];
    for (int k = 0; k < exclude_count; k += 1)
    {
        if (excluded_list[exclude_start + k] == atom_j)
        {
            return true;
        }
    }
    return false;
}

static inline unsigned long long BuildExclusionMask(
    const int* permutation, const int* cluster_offsets, int cluster_i,
    int cluster_j, unsigned int local_mask_i, unsigned int valid_mask_j,
    int cluster_size, int local_atom_numbers, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers)
{
    unsigned long long mask = 0ull;
    const int start_i = cluster_offsets[cluster_i];
    const int start_j = cluster_offsets[cluster_j];
    for (int lane_i = 0; lane_i < cluster_size; lane_i += 1)
    {
        if ((local_mask_i & (1u << lane_i)) == 0u)
        {
            continue;
        }
        const int atom_i = permutation[start_i + lane_i];
        for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
        {
            if ((valid_mask_j & (1u << lane_j)) == 0u)
            {
                continue;
            }
            const int atom_j = permutation[start_j + lane_j];
            bool excluded =
                ExclusionListContains(atom_i, atom_j, excluded_list_start,
                                      excluded_list, excluded_numbers);
            if (!excluded && atom_j < local_atom_numbers)
            {
                excluded =
                    ExclusionListContains(atom_j, atom_i, excluded_list_start,
                                          excluded_list, excluded_numbers);
            }
            if (excluded)
            {
                mask |= (1ull << (lane_i * cluster_size + lane_j));
            }
        }
    }
    return mask;
}

static inline CLUSTERED_GMXPACKED_EXCLUSION MakeNoExclusion()
{
    CLUSTERED_GMXPACKED_EXCLUSION exclusion = {};
    for (unsigned int& pair_word : exclusion.pair)
    {
        pair_word = 0xffffffffu;
    }
    return exclusion;
}

static inline bool SplitHasAtoms(unsigned int valid_mask_j, int split)
{
    return (valid_mask_j & Clustered_Split_Valid_Mask(split)) != 0u;
}

static void FillGmxpackedPairWords(
    const clustered_neighbor_cpu_builder::BuildInput& input,
    const HostClusteredJRecord& record, const int cluster_i_start,
    const int valid_mask_j, const int local_mask_j, const int split,
    const int jm, CLUSTERED_GMXPACKED_EXCLUSION* exclusion)
{
    const unsigned int packed_bit =
        1u << static_cast<unsigned int>(Clustered_Jm_Imask_Shift(jm));
    for (int i_local = 0; i_local < kMaxSuperClusterClusters; i_local += 1)
    {
        if ((record.imask & (1u << static_cast<unsigned int>(i_local))) == 0u)
        {
            continue;
        }
        const int cluster_i = cluster_i_start + i_local;
        const unsigned int local_mask_i =
            input.cluster_local_masks[(size_t)cluster_i];
        const unsigned long long exclusion_mask =
            record.exclusion_masks[(size_t)i_local];
        for (int split_j_lane = 0;
             split_j_lane < kClusteredSplitJClusterSize; split_j_lane += 1)
        {
            const int lane_j = split * kClusteredSplitJClusterSize +
                               split_j_lane;
            const bool valid_j =
                (valid_mask_j & (1u << static_cast<unsigned int>(lane_j))) !=
                0u;
            const bool local_j =
                (local_mask_j & (1u << static_cast<unsigned int>(lane_j))) !=
                0u;
            for (int lane_i = 0; lane_i < input.cluster_size; lane_i += 1)
            {
                const bool local_i =
                    (local_mask_i &
                     (1u << static_cast<unsigned int>(lane_i))) != 0u;
                bool allow_pair = valid_j && local_i;
                if (allow_pair &&
                    !Clustered_Local_I_Owns_Pair(
                        cluster_i, lane_i, record.cluster_j, lane_j, local_j))
                {
                    allow_pair = false;
                }
                if (allow_pair &&
                    (exclusion_mask &
                     (1ull << static_cast<unsigned int>(
                          lane_i * input.cluster_size + lane_j))) != 0ull)
                {
                    allow_pair = false;
                }
                if (allow_pair)
                {
                    exclusion
                        ->pair[split_j_lane * input.cluster_size + lane_i] |=
                        packed_bit << static_cast<unsigned int>(i_local);
                }
            }
        }
    }
}

static bool GmxpackedExclusionIsNeeded(
    const CLUSTERED_GMXPACKED_EXCLUSION& exclusion,
    const unsigned int split_imask)
{
    for (const unsigned int pair_word : exclusion.pair)
    {
        if (pair_word != split_imask)
        {
            return true;
        }
    }
    return false;
}

static void BuildCjPackedClusterMetadataShifted(
    const clustered_neighbor_cpu_builder::BuildInput& input,
    const int cluster_i_start, const int cluster_i_end, const int cluster_j,
    const int shift_id, const unsigned int valid_mask_j, unsigned int* imask,
    std::array<unsigned long long, kMaxSuperClusterClusters>* exclusion_masks)
{
    *imask = 0u;
    exclusion_masks->fill(0ull);
    const VECTOR shift_vec = Shift_Vector_From_Id(shift_id, input.cell);

    for (int cluster_i = cluster_i_start; cluster_i < cluster_i_end;
         cluster_i += 1)
    {
        const unsigned int local_mask_i =
            input.cluster_local_masks[(size_t)cluster_i];
        if (local_mask_i == 0u)
        {
            continue;
        }
        if (cluster_j >= cluster_i_start && cluster_j < cluster_i_end &&
            cluster_i > cluster_j &&
            Clustered_Valid_Lanes_Are_All_Local(
                valid_mask_j, input.cluster_local_masks[(size_t)cluster_j]))
        {
            continue;
        }
        if (!Cluster_Aabb_Overlaps_Shifted(
                input.cluster_centers[(size_t)cluster_i],
                input.cluster_extents[(size_t)cluster_i],
                input.cluster_centers[(size_t)cluster_j],
                input.cluster_extents[(size_t)cluster_j], input.cutoff,
                shift_vec))
        {
            continue;
        }

        const int i_local = cluster_i - cluster_i_start;
        *imask |= (1u << static_cast<unsigned int>(i_local));
        const unsigned long long exclusion_mask = BuildExclusionMask(
            input.permutation.data(), input.cluster_offsets.data(), cluster_i,
            cluster_j, local_mask_i, valid_mask_j, input.cluster_size,
            input.local_atom_numbers, input.excluded_list_start.data(),
            input.excluded_list.data(), input.excluded_numbers.data());
        (*exclusion_masks)[(size_t)i_local] = exclusion_mask;
    }
}

}  // namespace

void clustered_neighbor_cpu_builder::BuildPayload(const BuildInput& input,
                                            BuildOutput* output)
{
    *output = {};
    if (input.candidate_sci_numbers <= 0)
    {
        return;
    }
    output->gmxpacked_exclusions.push_back(MakeNoExclusion());
    std::array<std::vector<HostClusteredJRecord>, kClusteredShiftCount>
        shift_buckets;

    for (int candidate_sci = 0; candidate_sci < input.candidate_sci_numbers;
         candidate_sci += 1)
    {
        for (auto& bucket : shift_buckets)
        {
            bucket.clear();
        }

        const int sci_base = input.dense_shift_partitioned_candidates
                                 ? candidate_sci / kClusteredShiftCount
                                 : candidate_sci;
        const int super_i = input.sci_supercluster_ids[(size_t)sci_base];
        const int cluster_i_start =
            input.super_cluster_offsets[(size_t)super_i];
        const int cluster_i_end =
            input.super_cluster_offsets[(size_t)super_i + 1];
        int shift_begin = 0;
        int shift_end = kClusteredShiftCount;
        if (input.dense_shift_partitioned_candidates)
        {
            shift_begin = candidate_sci % kClusteredShiftCount;
            shift_end = shift_begin + 1;
        }
        else if (!input.candidate_shift_ids.empty())
        {
            shift_begin = input.candidate_shift_ids[(size_t)candidate_sci];
            shift_end = shift_begin + 1;
        }
        int processed_cluster_end = 0;
        for (int candidate_idx =
                 input.candidate_leaf_offsets[(size_t)candidate_sci];
             candidate_idx <
             input.candidate_leaf_offsets[(size_t)candidate_sci + 1];
             candidate_idx += 1)
        {
            const int leaf_j = input.candidate_leaf_ids[(size_t)candidate_idx];
            const int cluster_j_start =
                input.leaf_cluster_starts[(size_t)leaf_j];
            const int cluster_j_end = input.leaf_cluster_ends[(size_t)leaf_j];
            const int deduped_cluster_j_start =
                std::max(cluster_j_start, processed_cluster_end);
            for (int cluster_j = deduped_cluster_j_start;
                 cluster_j < cluster_j_end; cluster_j += 1)
            {
                const unsigned int valid_mask_j =
                    input.cluster_valid_masks[(size_t)cluster_j];
                if (valid_mask_j == 0u)
                {
                    continue;
                }
                const int super_j =
                    input.cluster_to_supercluster[(size_t)cluster_j];
                if (Clustered_Valid_Lanes_Are_All_Local(
                        valid_mask_j,
                        input.cluster_local_masks[(size_t)cluster_j]) &&
                    super_j < super_i)
                {
                    continue;
                }
                HostClusteredJRecord record = {};
                record.cluster_j = cluster_j;
                for (int shift_id = shift_begin; shift_id < shift_end;
                     shift_id += 1)
                {
                    BuildCjPackedClusterMetadataShifted(
                        input, cluster_i_start, cluster_i_end, cluster_j,
                        shift_id, valid_mask_j, &record.imask,
                        &record.exclusion_masks);
                    if (record.imask != 0u)
                    {
                        shift_buckets[(size_t)shift_id].push_back(record);
                    }
                }
            }
            processed_cluster_end =
                std::max(processed_cluster_end, cluster_j_end);
        }

        for (int shift_id = shift_begin; shift_id < shift_end; shift_id += 1)
        {
            auto& bucket = shift_buckets[(size_t)shift_id];
            if (bucket.empty())
            {
                continue;
            }
            const int gmxpacked_cj_begin =
                static_cast<int>(output->gmxpacked_cjpacked.size());
            for (size_t bucket_begin = 0; bucket_begin < bucket.size();
                 bucket_begin += kMaxJGroupSize)
            {
                CLUSTERED_GMXPACKED_CJ gmxpacked = {};
                CLUSTERED_GMXPACKED_EXCLUSION
                    split_exclusions[kClusteredWarpSplitCount] = {};
                for (int jm = 0; jm < kMaxJGroupSize; jm += 1)
                {
                    const size_t record_index = bucket_begin + (size_t)jm;
                    if (record_index >= bucket.size())
                    {
                        break;
                    }
                    const auto& record = bucket[record_index];
                    gmxpacked.cj[jm] = record.cluster_j;
                    const unsigned int valid_mask_j =
                        input.cluster_valid_masks[(size_t)record.cluster_j];
                    const unsigned int local_mask_j =
                        input.cluster_local_masks[(size_t)record.cluster_j];
                    for (int split = 0; split < kClusteredWarpSplitCount;
                         split += 1)
                    {
                        if (!SplitHasAtoms(valid_mask_j, split))
                        {
                            continue;
                        }
                        gmxpacked.split[split].imask |=
                            record.imask << Clustered_Jm_Imask_Shift(jm);
                        FillGmxpackedPairWords(
                            input, record, cluster_i_start, valid_mask_j,
                            local_mask_j, split, jm,
                            &split_exclusions[split]);
                    }
                }
                for (int split = 0; split < kClusteredWarpSplitCount;
                     split += 1)
                {
                    const unsigned int split_imask =
                        gmxpacked.split[split].imask;
                    if (split_imask == 0u ||
                        !GmxpackedExclusionIsNeeded(split_exclusions[split],
                                                   split_imask))
                    {
                        continue;
                    }
                    gmxpacked.split[split].exclusion_index =
                        static_cast<int>(
                            output->gmxpacked_exclusions.size());
                    output->gmxpacked_exclusions.push_back(
                        split_exclusions[split]);
                    output->gmxpacked_split_exclusion_numbers += 1;
                }
                output->gmxpacked_cjpacked.push_back(gmxpacked);
            }
            output->gmxpacked_scis.push_back(
                {super_i, shift_id, gmxpacked_cj_begin,
                 static_cast<int>(output->gmxpacked_cjpacked.size())});
        }
    }
    if (output->gmxpacked_cjpacked.empty())
    {
        output->gmxpacked_exclusions.clear();
    }
}

#endif
