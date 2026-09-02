static ClusteredGmxpackedRecordStreamCompactSummary
Build_Gmxpacked_Record_Stream_Compact_Payload(ClusteredNeighborProvider* layout)
{
    ClusteredGmxpackedRecordStreamCompactSummary summary = {};
    if (layout == NULL)
    {
        return summary;
    }

    summary.source_rows =
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_source_numbers;
    summary.aggregate_rows =
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregate_numbers;
    Clear_Gmxpacked_Compact_Payload(layout);
    if (summary.aggregate_rows <= 0 ||
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregates.data == NULL)
    {
        return summary;
    }

#ifndef USE_CPU
    bool reset_payload_on_failure = false;
    if (Build_Gmxpacked_Record_Stream_Compact_Payload_On_Device(
            layout, &summary, &reset_payload_on_failure))
    {
        Publish_Gmxpacked_Compact_Payload(layout);
        return summary;
    }
    if (reset_payload_on_failure)
    {
        const int compact_entry_numbers = summary.compact_entries;
        Reset_Gmxpacked_Payload(layout);
        summary.compact_entries = compact_entry_numbers;
    }
    return summary;
#else

    const std::vector<CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE>
        host_aggregates = Copy_Device_Buffer_To_Host(
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregates.data,
            static_cast<size_t>(summary.aggregate_rows));
    if (host_aggregates.empty())
    {
        return summary;
    }

    struct RecordStreamCompactEntry
    {
        int supercluster_id = -1;
        int shift_id = kClusteredCentralShiftId;
        int cluster_j = -1;
        unsigned int split_imask[kClusteredWarpSplitCount] = {};
        unsigned int valid_mask_j = 0u;
        unsigned int local_mask_j = 0u;
        unsigned int
            pair_exclusion_words[kClusteredWarpSplitCount]
                                [kClusteredGmxpackedExclusionPairCount] = {};
        int source_order_begin = 0;
        int source_order_end = 0;
    };

    std::vector<CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE>
        sorted_aggregates = host_aggregates;
    std::sort(sorted_aggregates.begin(), sorted_aggregates.end(),
              [](const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE& lhs,
                 const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE& rhs)
              {
                  if (lhs.supercluster_id != rhs.supercluster_id)
                  {
                      return lhs.supercluster_id < rhs.supercluster_id;
                  }
                  if (lhs.cluster_j != rhs.cluster_j)
                  {
                      return lhs.cluster_j < rhs.cluster_j;
                  }
                  if (lhs.shift_id != rhs.shift_id)
                  {
                      return lhs.shift_id < rhs.shift_id;
                  }
                  if (lhs.source_order_begin != rhs.source_order_begin)
                  {
                      return lhs.source_order_begin < rhs.source_order_begin;
                  }
                  if (lhs.source_order_end != rhs.source_order_end)
                  {
                      return lhs.source_order_end < rhs.source_order_end;
                  }
                  return lhs.sci_id < rhs.sci_id;
              });
    std::vector<RecordStreamCompactEntry> compact_entries;
    compact_entries.reserve(sorted_aggregates.size());
    for (const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE& aggregate :
         sorted_aggregates)
    {
        const bool same_compact_key =
            !compact_entries.empty() &&
            compact_entries.back().supercluster_id ==
                aggregate.supercluster_id &&
            compact_entries.back().shift_id == aggregate.shift_id &&
            compact_entries.back().cluster_j == aggregate.cluster_j;
        if (!same_compact_key)
        {
            RecordStreamCompactEntry entry = {};
            entry.supercluster_id = aggregate.supercluster_id;
            entry.shift_id = aggregate.shift_id;
            entry.cluster_j = aggregate.cluster_j;
            entry.valid_mask_j = aggregate.valid_mask_j;
            entry.local_mask_j = aggregate.local_mask_j;
            entry.source_order_begin = aggregate.source_order_begin;
            entry.source_order_end = aggregate.source_order_end;
            for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
            {
                entry.split_imask[split] = aggregate.split_imask[split];
                for (int pair_idx = 0;
                     pair_idx < kClusteredGmxpackedExclusionPairCount;
                     pair_idx += 1)
                {
                    entry.pair_exclusion_words[split][pair_idx] =
                        aggregate.pair_exclusion_words[split][pair_idx];
                }
            }
            compact_entries.push_back(entry);
            continue;
        }

        RecordStreamCompactEntry& entry = compact_entries.back();
        entry.valid_mask_j |= aggregate.valid_mask_j;
        entry.local_mask_j |= aggregate.local_mask_j;
        entry.source_order_begin =
            IntMin(entry.source_order_begin, aggregate.source_order_begin);
        entry.source_order_end =
            IntMax(entry.source_order_end, aggregate.source_order_end);
        for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
        {
            entry.split_imask[split] |= aggregate.split_imask[split];
            for (int pair_idx = 0;
                 pair_idx < kClusteredGmxpackedExclusionPairCount;
                 pair_idx += 1)
            {
                entry.pair_exclusion_words[split][pair_idx] |=
                    aggregate.pair_exclusion_words[split][pair_idx];
            }
        }
    }

    summary.compact_entries = static_cast<int>(compact_entries.size());
    if (compact_entries.empty())
    {
        return summary;
    }

    std::stable_sort(compact_entries.begin(), compact_entries.end(),
                     [](const RecordStreamCompactEntry& lhs,
                        const RecordStreamCompactEntry& rhs)
                     {
                         if (lhs.supercluster_id != rhs.supercluster_id)
                         {
                             return lhs.supercluster_id < rhs.supercluster_id;
                         }
                         if (lhs.shift_id != rhs.shift_id)
                         {
                             return lhs.shift_id < rhs.shift_id;
                         }
                         if (lhs.source_order_begin != rhs.source_order_begin)
                         {
                             return lhs.source_order_begin <
                                    rhs.source_order_begin;
                         }
                         if (lhs.source_order_end != rhs.source_order_end)
                         {
                             return lhs.source_order_end < rhs.source_order_end;
                         }
                         return lhs.cluster_j < rhs.cluster_j;
                     });

    const auto split_exclusion_needed =
        [](const CLUSTERED_GMXPACKED_EXCLUSION& compact_exclusion,
           const unsigned int split_imask) -> bool
    {
        for (int pair_idx = 0; pair_idx < kClusteredGmxpackedExclusionPairCount;
             pair_idx += 1)
        {
            if (compact_exclusion.pair[pair_idx] != split_imask)
            {
                return true;
            }
        }
        return false;
    };

    std::vector<CLUSTERED_GMXPACKED_SCI> gmxpacked_sci;
    std::vector<CLUSTERED_GMXPACKED_CJ> gmxpacked_cjpacked;
    std::vector<CLUSTERED_GMXPACKED_EXCLUSION> gmxpacked_exclusions;
    gmxpacked_sci.reserve(compact_entries.size());
    gmxpacked_cjpacked.reserve((compact_entries.size() +
                                static_cast<size_t>(kClusteredJGroupSize) -
                                1u) /
                               static_cast<size_t>(kClusteredJGroupSize));
    gmxpacked_exclusions.reserve(
        1u +
        compact_entries.size() * static_cast<size_t>(kClusteredWarpSplitCount));
    gmxpacked_exclusions.push_back(Make_Empty_Gmxpacked_No_Exclusion());
    int split_exclusion_numbers = 0;

    for (size_t sci_entry_begin = 0; sci_entry_begin < compact_entries.size();)
    {
        size_t sci_entry_end = sci_entry_begin + 1;
        while (sci_entry_end < compact_entries.size() &&
               compact_entries[sci_entry_end].supercluster_id ==
                   compact_entries[sci_entry_begin].supercluster_id &&
               compact_entries[sci_entry_end].shift_id ==
                   compact_entries[sci_entry_begin].shift_id)
        {
            sci_entry_end += 1;
        }

        const int compact_begin = static_cast<int>(gmxpacked_cjpacked.size());
        for (size_t packed_entry_begin = sci_entry_begin;
             packed_entry_begin < sci_entry_end;
             packed_entry_begin += static_cast<size_t>(kClusteredJGroupSize))
        {
            const size_t packed_entry_end = std::min(
                sci_entry_end,
                packed_entry_begin + static_cast<size_t>(kClusteredJGroupSize));
            CLUSTERED_GMXPACKED_CJ compact_packed = {};
            CLUSTERED_GMXPACKED_EXCLUSION
            split_exclusions[kClusteredWarpSplitCount] = {};
            for (size_t entry_idx = packed_entry_begin;
                 entry_idx < packed_entry_end; entry_idx += 1)
            {
                const int jm = static_cast<int>(entry_idx - packed_entry_begin);
                const unsigned int jm_imask_shift =
                    static_cast<unsigned int>(Clustered_Jm_Imask_Shift(jm));
                const RecordStreamCompactEntry& entry =
                    compact_entries[entry_idx];
                compact_packed.cj[jm] = entry.cluster_j;
                for (int split = 0; split < kClusteredWarpSplitCount;
                     split += 1)
                {
                    compact_packed.split[split].imask |=
                        entry.split_imask[split] << jm_imask_shift;
                    for (int pair_idx = 0;
                         pair_idx < kClusteredGmxpackedExclusionPairCount;
                         pair_idx += 1)
                    {
                        split_exclusions[split].pair[pair_idx] |=
                            entry.pair_exclusion_words[split][pair_idx]
                            << jm_imask_shift;
                    }
                }
            }

            for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
            {
                const unsigned int split_imask =
                    compact_packed.split[split].imask;
                compact_packed.split[split].exclusion_index = 0;
                if (split_imask == 0u ||
                    !split_exclusion_needed(split_exclusions[split],
                                            split_imask))
                {
                    continue;
                }
                compact_packed.split[split].exclusion_index =
                    static_cast<int>(gmxpacked_exclusions.size());
                gmxpacked_exclusions.push_back(split_exclusions[split]);
                split_exclusion_numbers += 1;
            }

            gmxpacked_cjpacked.push_back(compact_packed);
        }

        gmxpacked_sci.push_back(
            {compact_entries[sci_entry_begin].supercluster_id,
             compact_entries[sci_entry_begin].shift_id, compact_begin,
             static_cast<int>(gmxpacked_cjpacked.size())});
        sci_entry_begin = sci_entry_end;
    }

    if (gmxpacked_cjpacked.empty())
    {
        gmxpacked_sci.clear();
        gmxpacked_exclusions.clear();
        split_exclusion_numbers = 0;
    }

    summary.compact_sci = static_cast<int>(gmxpacked_sci.size());
    summary.compact_cj = static_cast<int>(gmxpacked_cjpacked.size());
    summary.split_excl = split_exclusion_numbers;
    summary.compact_excl = static_cast<int>(gmxpacked_exclusions.size());
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers = summary.compact_sci;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked_numbers = summary.compact_cj;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_split_exclusion_numbers = summary.split_excl;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_exclusion_numbers = summary.compact_excl;

    if (ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers > 0)
    {
        Reserve_Device_Buffer(ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers,
                              &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci);
        deviceMemcpy(ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci.data, gmxpacked_sci.data(),
                     sizeof(CLUSTERED_GMXPACKED_SCI) * gmxpacked_sci.size(),
                     deviceMemcpyHostToDevice);
    }
    if (ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked_numbers > 0)
    {
        Reserve_Device_Buffer(ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked_numbers,
                              &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked);
        deviceMemcpy(
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked.data,
            gmxpacked_cjpacked.data(),
            sizeof(CLUSTERED_GMXPACKED_CJ) * gmxpacked_cjpacked.size(),
            deviceMemcpyHostToDevice);
    }
    if (ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_exclusion_numbers > 0)
    {
        Reserve_Device_Buffer(ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_exclusion_numbers,
                              &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_exclusions);
        deviceMemcpy(ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_exclusions.data,
                     gmxpacked_exclusions.data(),
                     sizeof(CLUSTERED_GMXPACKED_EXCLUSION) *
                         gmxpacked_exclusions.size(),
                     deviceMemcpyHostToDevice);
    }

    if (summary.compact_sci > 0 && summary.compact_cj > 0 &&
        summary.compact_excl > 0)
    {
        Publish_Gmxpacked_Compact_Payload(layout);
    }
    return summary;
#endif
}

#ifndef USE_CPU
static __device__ __forceinline__ bool Gmxpacked_Exclusion_Row_Is_Needed(
    const CLUSTERED_GMXPACKED_EXCLUSION& compact_exclusion,
    const unsigned int split_imask)
{
#pragma unroll
    for (int pair_idx = 0; pair_idx < kClusteredGmxpackedExclusionPairCount;
         pair_idx += 1)
    {
        if (compact_exclusion.pair[pair_idx] != split_imask)
        {
            return true;
        }
    }
    return false;
}

static __device__ __forceinline__ CLUSTERED_GMXPACKED_CJ
Make_Empty_Gmxpacked_CjPacked()
{
    CLUSTERED_GMXPACKED_CJ compact_packed = {};
#pragma unroll
    for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
    {
        compact_packed.cj[jm] = -1;
    }
    return compact_packed;
}
#endif
