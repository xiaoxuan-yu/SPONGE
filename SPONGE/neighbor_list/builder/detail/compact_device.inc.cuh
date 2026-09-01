namespace
{

static __global__ void Build_Gmxpacked_Record_Stream_Source_Low_Sort_Keys(
    const int source_rows, const int* source_indices,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    uint64_t* sort_keys)
{
    SIMPLE_DEVICE_FOR(source_idx, source_rows)
    {
        const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE& source =
            sources[source_indices != NULL ? source_indices[source_idx]
                                           : source_idx];
        sort_keys[source_idx] =
            (static_cast<uint64_t>(
                 static_cast<unsigned int>(source.supercluster_id))
             << 32) |
            static_cast<uint64_t>(static_cast<unsigned int>(source.cluster_j));
    }
}

static __global__ void Build_Gmxpacked_Record_Stream_Source_High_Sort_Keys(
    const int source_rows, const int* source_indices,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    uint64_t* sort_keys)
{
    SIMPLE_DEVICE_FOR(source_idx, source_rows)
    {
        const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE& source =
            sources[source_indices != NULL ? source_indices[source_idx]
                                           : source_idx];
        sort_keys[source_idx] =
            (static_cast<uint64_t>(static_cast<unsigned int>(source.sci_id))
             << 32) |
            static_cast<uint64_t>(static_cast<unsigned int>(source.shift_id));
    }
}

static __global__ void Gather_Gmxpacked_Record_Stream_Sources_By_Index(
    const int source_rows, const int* source_indices,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* src_sources,
    CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* dst_sources)
{
    SIMPLE_DEVICE_FOR(source_idx, source_rows)
    {
        dst_sources[source_idx] = src_sources[source_indices[source_idx]];
    }
}

static __host__ __device__ __forceinline__ bool
Gmxpacked_Record_Stream_Source_Same_Aggregate_Key(
    const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE& lhs,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE& rhs)
{
    return lhs.sci_id == rhs.sci_id && lhs.shift_id == rhs.shift_id &&
           lhs.supercluster_id == rhs.supercluster_id &&
           lhs.cluster_j == rhs.cluster_j;
}

static __global__ void Build_Gmxpacked_Record_Stream_Aggregate_Flags(
    const int source_rows,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sorted_sources,
    int* aggregate_flags)
{
    SIMPLE_DEVICE_FOR(source_idx, source_rows)
    {
        int aggregate_start = 0;
        if (source_idx == 0)
        {
            aggregate_start = 1;
        }
        else
        {
            aggregate_start =
                Gmxpacked_Record_Stream_Source_Same_Aggregate_Key(
                    sorted_sources[source_idx], sorted_sources[source_idx - 1])
                    ? 0
                    : 1;
        }
        aggregate_flags[source_idx] = aggregate_start;
    }
}

static __global__ void Fill_Gmxpacked_Record_Stream_Aggregates_From_Sources(
    const int aggregate_rows, const int* aggregate_starts,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sorted_sources,
    CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE* aggregates,
    int* aggregate_fill_cursor)
{
    SIMPLE_DEVICE_FOR(aggregate_idx, aggregate_rows)
    {
        const int source_begin = aggregate_starts[aggregate_idx];
        const int source_end = aggregate_starts[aggregate_idx + 1];
        if (source_begin >= source_end)
        {
#ifdef USE_CPU
            continue;
#else
            return;
#endif
        }

        const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE& first_source =
            sorted_sources[source_begin];
        CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE aggregate = {};
        aggregate.sci_id = first_source.sci_id;
        aggregate.shift_id = first_source.shift_id;
        aggregate.supercluster_id = first_source.supercluster_id;
        aggregate.cluster_j = first_source.cluster_j;
        aggregate.valid_mask_j = first_source.valid_mask_j;
        aggregate.local_mask_j = first_source.local_mask_j;
        aggregate.source_order_begin = first_source.source_order;
        aggregate.source_order_end = first_source.source_order + 1;

        for (int source_idx = source_begin; source_idx < source_end;
             source_idx += 1)
        {
            const CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE& source =
                sorted_sources[source_idx];
            const int split = source.split_id;
            aggregate.valid_mask_j |= source.valid_mask_j;
            aggregate.local_mask_j |= source.local_mask_j;
            aggregate.source_order_begin =
                IntMin(aggregate.source_order_begin, source.source_order);
            aggregate.source_order_end =
                IntMax(aggregate.source_order_end, source.source_order + 1);
            if (split < 0 || split >= kClusteredWarpSplitCount)
            {
                continue;
            }
            aggregate.split_imask[split] |= source.imask;
#pragma unroll
            for (int pair_idx = 0;
                 pair_idx < kClusteredGmxpackedExclusionPairCount;
                 pair_idx += 1)
            {
                aggregate.pair_exclusion_words[split][pair_idx] |=
                    source.pair_exclusion_words[pair_idx];
            }
        }

        aggregates[aggregate_idx] = aggregate;
        if (aggregate_fill_cursor != NULL)
        {
            atomicAdd(aggregate_fill_cursor, 1);
        }
    }
}

}  // namespace

#ifndef USE_CPU

namespace
{

static __host__ __device__ __forceinline__ bool
Gmxpacked_Record_Stream_Aggregate_Same_Compact_Key(
    const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE& lhs,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE& rhs)
{
    return lhs.supercluster_id == rhs.supercluster_id &&
           lhs.shift_id == rhs.shift_id && lhs.cluster_j == rhs.cluster_j;
}

}  // namespace

__global__ void Gather_Gmxpacked_Record_Stream_Aggregates_By_Index(
    const int aggregate_rows, const int* aggregate_indices,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE* src_aggregates,
    CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE* dst_aggregates)
{
    SIMPLE_DEVICE_FOR(aggregate_idx, aggregate_rows)
    {
        dst_aggregates[aggregate_idx] =
            src_aggregates[aggregate_indices[aggregate_idx]];
    }
}

__global__ void Build_Gmxpacked_Record_Stream_Compact_Entry_Flags(
    const int aggregate_rows,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE* sorted_aggregates,
    int* compact_entry_flags)
{
    SIMPLE_DEVICE_FOR(aggregate_idx, aggregate_rows)
    {
        int compact_start = 0;
        if (aggregate_idx == 0)
        {
            compact_start = 1;
        }
        else
        {
            compact_start = Gmxpacked_Record_Stream_Aggregate_Same_Compact_Key(
                                sorted_aggregates[aggregate_idx],
                                sorted_aggregates[aggregate_idx - 1])
                                ? 0
                                : 1;
        }
        compact_entry_flags[aggregate_idx] = compact_start;
    }
}

__global__ void Fill_Gmxpacked_Record_Stream_Compact_Entries(
    const int compact_entry_numbers, const int* compact_entry_starts,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE* sorted_aggregates,
    CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE* compact_entries)
{
    SIMPLE_DEVICE_FOR(compact_idx, compact_entry_numbers)
    {
        const int aggregate_begin = compact_entry_starts[compact_idx];
        const int aggregate_end = compact_entry_starts[compact_idx + 1];
        if (aggregate_begin >= aggregate_end)
        {
            return;
        }

        const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE& first =
            sorted_aggregates[aggregate_begin];
        compact_entries[compact_idx].sci_id = first.sci_id;
        compact_entries[compact_idx].shift_id = first.shift_id;
        compact_entries[compact_idx].supercluster_id = first.supercluster_id;
        compact_entries[compact_idx].cluster_j = first.cluster_j;
        compact_entries[compact_idx].valid_mask_j = first.valid_mask_j;
        compact_entries[compact_idx].local_mask_j = first.local_mask_j;
        compact_entries[compact_idx].source_order_begin =
            first.source_order_begin;
        compact_entries[compact_idx].source_order_end = first.source_order_end;
        for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
        {
            compact_entries[compact_idx].split_imask[split] =
                first.split_imask[split];
            for (int pair_idx = 0;
                 pair_idx < kClusteredGmxpackedExclusionPairCount;
                 pair_idx += 1)
            {
                compact_entries[compact_idx]
                    .pair_exclusion_words[split][pair_idx] =
                    first.pair_exclusion_words[split][pair_idx];
            }
        }
        if (aggregate_begin + 1 >= aggregate_end)
        {
            return;
        }

        for (int aggregate_idx = aggregate_begin + 1;
             aggregate_idx < aggregate_end; aggregate_idx += 1)
        {
            const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE& aggregate =
                sorted_aggregates[aggregate_idx];
            compact_entries[compact_idx].sci_id =
                IntMin(compact_entries[compact_idx].sci_id, aggregate.sci_id);
            compact_entries[compact_idx].valid_mask_j |= aggregate.valid_mask_j;
            compact_entries[compact_idx].local_mask_j |= aggregate.local_mask_j;
            compact_entries[compact_idx].source_order_begin =
                IntMin(compact_entries[compact_idx].source_order_begin,
                       aggregate.source_order_begin);
            compact_entries[compact_idx].source_order_end =
                IntMax(compact_entries[compact_idx].source_order_end,
                       aggregate.source_order_end);
            for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
            {
                compact_entries[compact_idx].split_imask[split] |=
                    aggregate.split_imask[split];
                for (int pair_idx = 0;
                     pair_idx < kClusteredGmxpackedExclusionPairCount;
                     pair_idx += 1)
                {
                    compact_entries[compact_idx]
                        .pair_exclusion_words[split][pair_idx] |=
                        aggregate.pair_exclusion_words[split][pair_idx];
                }
            }
        }
    }
}

__global__ void Build_Gmxpacked_Record_Stream_Compact_Cluster_J_Sort_Keys(
    const int compact_entry_numbers, const int* compact_entry_indices,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE* compact_entries,
    uint64_t* sort_keys)
{
    SIMPLE_DEVICE_FOR(entry_idx, compact_entry_numbers)
    {
        const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE& entry =
            compact_entries[compact_entry_indices != NULL
                                ? compact_entry_indices[entry_idx]
                                : entry_idx];
        sort_keys[entry_idx] =
            static_cast<uint64_t>(static_cast<unsigned int>(entry.cluster_j));
    }
}

__global__ void Build_Gmxpacked_Record_Stream_Compact_Source_End_Sort_Keys(
    const int compact_entry_numbers, const int* compact_entry_indices,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE* compact_entries,
    uint64_t* sort_keys)
{
    SIMPLE_DEVICE_FOR(entry_idx, compact_entry_numbers)
    {
        const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE& entry =
            compact_entries[compact_entry_indices != NULL
                                ? compact_entry_indices[entry_idx]
                                : entry_idx];
        sort_keys[entry_idx] = static_cast<uint64_t>(
            static_cast<unsigned int>(entry.source_order_end));
    }
}

__global__ void Build_Gmxpacked_Record_Stream_Compact_Source_Begin_Sort_Keys(
    const int compact_entry_numbers, const int* compact_entry_indices,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE* compact_entries,
    uint64_t* sort_keys)
{
    SIMPLE_DEVICE_FOR(entry_idx, compact_entry_numbers)
    {
        const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE& entry =
            compact_entries[compact_entry_indices != NULL
                                ? compact_entry_indices[entry_idx]
                                : entry_idx];
        sort_keys[entry_idx] = static_cast<uint64_t>(
            static_cast<unsigned int>(entry.source_order_begin));
    }
}

__global__ void Build_Gmxpacked_Record_Stream_Compact_Shift_Sort_Keys(
    const int compact_entry_numbers, const int* compact_entry_indices,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE* compact_entries,
    uint64_t* sort_keys)
{
    SIMPLE_DEVICE_FOR(entry_idx, compact_entry_numbers)
    {
        const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE& entry =
            compact_entries[compact_entry_indices != NULL
                                ? compact_entry_indices[entry_idx]
                                : entry_idx];
        sort_keys[entry_idx] =
            static_cast<uint64_t>(static_cast<unsigned int>(entry.shift_id));
    }
}

__global__ void Build_Gmxpacked_Record_Stream_Compact_Supercluster_Sort_Keys(
    const int compact_entry_numbers, const int* compact_entry_indices,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE* compact_entries,
    uint64_t* sort_keys)
{
    SIMPLE_DEVICE_FOR(entry_idx, compact_entry_numbers)
    {
        const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE& entry =
            compact_entries[compact_entry_indices != NULL
                                ? compact_entry_indices[entry_idx]
                                : entry_idx];
        sort_keys[entry_idx] = static_cast<uint64_t>(
            static_cast<unsigned int>(entry.supercluster_id));
    }
}

__global__ void Build_Gmxpacked_Record_Stream_Compact_Sci_Flags(
    const int compact_entry_numbers,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE* compact_entries,
    int* sci_flags)
{
    SIMPLE_DEVICE_FOR(entry_idx, compact_entry_numbers)
    {
        int sci_start = 0;
        if (entry_idx == 0)
        {
            sci_start = 1;
        }
        else
        {
            const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE& current =
                compact_entries[entry_idx];
            const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE& previous =
                compact_entries[entry_idx - 1];
            sci_start = (current.supercluster_id != previous.supercluster_id ||
                         current.shift_id != previous.shift_id)
                            ? 1
                            : 0;
        }
        sci_flags[entry_idx] = sci_start;
    }
}

__global__ void Count_Gmxpacked_CjPacked_Per_Sci(const int sci_numbers,
                                                 const int* compact_sci_starts,
                                                 int* cjpacked_counts)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const int entry_count =
            compact_sci_starts[sci + 1] - compact_sci_starts[sci];
        cjpacked_counts[sci] = entry_count > 0
                                   ? (entry_count + kClusteredJGroupSize - 1) /
                                         kClusteredJGroupSize
                                   : 0;
    }
}

__global__ void Initialize_Gmxpacked_Exclusion_Rows(
    const int exclusion_numbers, CLUSTERED_GMXPACKED_EXCLUSION* exclusions)
{
    SIMPLE_DEVICE_FOR(exclusion_idx, exclusion_numbers)
    {
        CLUSTERED_GMXPACKED_EXCLUSION exclusion = {};
        const unsigned int fill_value = exclusion_idx == 0 ? 0xffffffffu : 0u;
#pragma unroll
        for (int pair_idx = 0; pair_idx < kClusteredGmxpackedExclusionPairCount;
             pair_idx += 1)
        {
            exclusion.pair[pair_idx] = fill_value;
        }
        exclusions[exclusion_idx] = exclusion;
    }
}

#endif

#ifndef USE_CPU
namespace
{

constexpr int kCompactExclusionBlockSize = 64;

static __global__ void Fill_Gmxpacked_Record_Stream_Sci_And_Cj(
    const int sci_numbers, const int* compact_sci_starts,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE* compact_entries,
    const int* compact_cjpacked_offsets,
    CLUSTERED_GMXPACKED_SCI* gmxpacked_sci,
    CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const int entry_begin = compact_sci_starts[sci];
        const int entry_end = compact_sci_starts[sci + 1];
        if (entry_begin >= entry_end)
        {
            return;
        }

        const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE first_entry =
            compact_entries[entry_begin];
        const int compact_begin = compact_cjpacked_offsets[sci];
        const int compact_end = compact_cjpacked_offsets[sci + 1];
        gmxpacked_sci[sci] = {first_entry.supercluster_id, first_entry.shift_id,
                              compact_begin, compact_end};

        const int packed_count = compact_end - compact_begin;
        for (int local_packed = 0; local_packed < packed_count;
             local_packed += 1)
        {
            const int packed_entry_begin =
                entry_begin + local_packed * kClusteredJGroupSize;
            const int packed_entry_end =
                IntMin(entry_end, packed_entry_begin + kClusteredJGroupSize);
            CLUSTERED_GMXPACKED_CJ compact_packed =
                Make_Empty_Gmxpacked_CjPacked();
            for (int entry_idx = packed_entry_begin;
                 entry_idx < packed_entry_end; entry_idx += 1)
            {
                const int jm = entry_idx - packed_entry_begin;
                const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE& entry =
                    compact_entries[entry_idx];
                compact_packed.cj[jm] = entry.cluster_j;
                for (int split = 0; split < kClusteredWarpSplitCount;
                     split += 1)
                {
                    const unsigned int imask = entry.split_imask[split];
                    if (imask == 0u)
                    {
                        continue;
                    }
                    compact_packed.split[split].imask |=
                        imask << Clustered_Jm_Imask_Shift(jm);
                }
            }
            gmxpacked_cjpacked[compact_begin + local_packed] = compact_packed;
        }
    }
}

static __device__ __forceinline__ CLUSTERED_GMXPACKED_EXCLUSION
Build_Gmxpacked_Record_Stream_Split_Exclusion_Row(
    const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE* compact_entries,
    const int entry_begin, const int entry_end, const int split_idx)
{
    CLUSTERED_GMXPACKED_EXCLUSION compact_exclusion = {};
    for (int entry_idx = entry_begin; entry_idx < entry_end; entry_idx += 1)
    {
        const int jm = entry_idx - entry_begin;
        const unsigned int jm_imask_shift = Clustered_Jm_Imask_Shift(jm);
        const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE& entry =
            compact_entries[entry_idx];
        for (int pair_idx = 0; pair_idx < kClusteredGmxpackedExclusionPairCount;
             pair_idx += 1)
        {
            compact_exclusion.pair[pair_idx] |=
                entry.pair_exclusion_words[split_idx][pair_idx]
                << jm_imask_shift;
        }
    }
    return compact_exclusion;
}

static __global__ void Count_Gmxpacked_Record_Stream_Split_Exclusions(
    const int sci_numbers, const CLUSTERED_GMXPACKED_SCI* gmxpacked_sci,
    const CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE* compact_entries,
    const int* compact_sci_starts, int* split_exclusion_counts)
{
    const int sci = blockIdx.x;
    if (sci >= sci_numbers)
    {
        return;
    }

    const CLUSTERED_GMXPACKED_SCI compact_sci_entry = gmxpacked_sci[sci];
    const int compact_begin = compact_sci_entry.cjpacked_begin;
    const int packed_count = compact_sci_entry.cjpacked_end - compact_begin;
    const int sci_entry_begin = compact_sci_starts[sci];
    const int sci_entry_end = compact_sci_starts[sci + 1];
    const int split_entry_count = packed_count * kClusteredWarpSplitCount;
    for (int local_split_entry = threadIdx.x;
         local_split_entry < split_entry_count; local_split_entry += blockDim.x)
    {
        const int local_packed = local_split_entry / kClusteredWarpSplitCount;
        const int split_idx = local_split_entry % kClusteredWarpSplitCount;
        const int compact_packed_idx = compact_begin + local_packed;
        const int compact_split_idx =
            compact_packed_idx * kClusteredWarpSplitCount + split_idx;
        const int packed_entry_begin =
            sci_entry_begin + local_packed * kClusteredJGroupSize;
        const int packed_entry_end =
            IntMin(sci_entry_end, packed_entry_begin + kClusteredJGroupSize);
        const unsigned int split_imask =
            gmxpacked_cjpacked[compact_packed_idx].split[split_idx].imask;
        int needs_exclusion = 0;
        if (split_imask != 0u)
        {
            const CLUSTERED_GMXPACKED_EXCLUSION compact_exclusion =
                Build_Gmxpacked_Record_Stream_Split_Exclusion_Row(
                    compact_entries, packed_entry_begin, packed_entry_end,
                    split_idx);
            needs_exclusion = Gmxpacked_Exclusion_Row_Is_Needed(
                                  compact_exclusion, split_imask)
                                  ? 1
                                  : 0;
        }
        split_exclusion_counts[compact_split_idx] = needs_exclusion;
    }
}

static __global__ void Fill_Gmxpacked_Record_Stream_Split_Exclusions(
    const int sci_numbers, const CLUSTERED_GMXPACKED_SCI* gmxpacked_sci,
    const CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked_read,
    const CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE* compact_entries,
    const int* compact_sci_starts, const int* split_exclusion_offsets,
    CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked,
    CLUSTERED_GMXPACKED_EXCLUSION* gmxpacked_exclusions)
{
    const int sci = blockIdx.x;
    if (sci >= sci_numbers)
    {
        return;
    }

    const CLUSTERED_GMXPACKED_SCI compact_sci_entry = gmxpacked_sci[sci];
    const int compact_begin = compact_sci_entry.cjpacked_begin;
    const int packed_count = compact_sci_entry.cjpacked_end - compact_begin;
    const int sci_entry_begin = compact_sci_starts[sci];
    const int sci_entry_end = compact_sci_starts[sci + 1];
    const int split_entry_count = packed_count * kClusteredWarpSplitCount;
    for (int local_split_entry = threadIdx.x;
         local_split_entry < split_entry_count; local_split_entry += blockDim.x)
    {
        const int local_packed = local_split_entry / kClusteredWarpSplitCount;
        const int split_idx = local_split_entry % kClusteredWarpSplitCount;
        const int compact_packed_idx = compact_begin + local_packed;
        const int compact_split_idx =
            compact_packed_idx * kClusteredWarpSplitCount + split_idx;
        const int packed_entry_begin =
            sci_entry_begin + local_packed * kClusteredJGroupSize;
        const int packed_entry_end =
            IntMin(sci_entry_end, packed_entry_begin + kClusteredJGroupSize);
        const unsigned int split_imask =
            gmxpacked_cjpacked_read[compact_packed_idx].split[split_idx].imask;
        if (split_imask == 0u)
        {
            continue;
        }
        const CLUSTERED_GMXPACKED_EXCLUSION compact_exclusion =
            Build_Gmxpacked_Record_Stream_Split_Exclusion_Row(
                compact_entries, packed_entry_begin, packed_entry_end,
                split_idx);
        if (!Gmxpacked_Exclusion_Row_Is_Needed(compact_exclusion, split_imask))
        {
            continue;
        }

        const int compact_exclusion_idx =
            1 + split_exclusion_offsets[compact_split_idx];
        gmxpacked_cjpacked[compact_packed_idx]
            .split[split_idx]
            .exclusion_index = compact_exclusion_idx;
        gmxpacked_exclusions[compact_exclusion_idx] = compact_exclusion;
    }
}

}  // namespace
#endif

__attribute__((noinline, noclone)) int Build_Gmxpacked_Record_Stream_Aggregates(
    ClusteredNeighborProvider* layout)
{
    if (layout == NULL ||
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_source_numbers <= 0 ||
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_sources.data == NULL)
    {
        if (layout != NULL)
        {
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregate_numbers = 0;
        }
        return 0;
    }
#ifdef USE_CPU
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregate_numbers = 0;
    return 0;
#else
    using namespace clustered_neighbor_builder_internal;

    const int source_rows =
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_source_numbers;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregate_numbers = 0;

    clustered_neighbor_runtime::Reserve_Device_Buffer(
        source_rows, &ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys);
    if (source_rows > 1)
    {
        clustered_neighbor_runtime::Reserve_Device_Buffer(
            source_rows, &ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices);
        BuildLinearIndices(source_rows,
                           ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data);
        Launch_Device_Kernel(
            Build_Gmxpacked_Record_Stream_Source_Low_Sort_Keys,
            (source_rows + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, source_rows,
            ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data,
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_sources.data,
            ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys.data);
        StableSortU64Int(layout, source_rows,
                         ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys.data,
                         ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data);
        Launch_Device_Kernel(
            Build_Gmxpacked_Record_Stream_Source_High_Sort_Keys,
            (source_rows + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, source_rows,
            ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data,
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_sources.data,
            ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys.data);
        StableSortU64Int(layout, source_rows,
                         ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys.data,
                         ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data);

        clustered_neighbor_runtime::Reserve_Raw_Device_Workspace(
            sizeof(CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE) *
                static_cast<std::size_t>(source_rows),
            &ClusteredNeighborProviderInternal::Workspace(layout).sort_values);
        auto* d_sorted_sources =
            reinterpret_cast<CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE*>(
                ClusteredNeighborProviderInternal::Workspace(layout).sort_values.data);
        Launch_Device_Kernel(
            Gather_Gmxpacked_Record_Stream_Sources_By_Index,
            (source_rows + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, source_rows,
            ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data,
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_sources.data,
            d_sorted_sources);
        deviceMemcpy(ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_sources.data,
                     d_sorted_sources,
                     sizeof(CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE) *
                         static_cast<std::size_t>(source_rows),
                     deviceMemcpyDeviceToDevice);
    }

    clustered_neighbor_runtime::Reserve_Device_Buffer(
        source_rows + 1, &ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts);
    clustered_neighbor_runtime::Reserve_Device_Buffer(
        source_rows + 1, &ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_offsets);
    Launch_Device_Kernel(Build_Gmxpacked_Record_Stream_Aggregate_Flags,
                         (source_rows + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, source_rows,
                         ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_sources.data,
                         ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts.data);
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregate_numbers =
        ExclusiveScanCounts(layout, source_rows,
                            ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts.data,
                            ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_offsets.data);
    if (ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregate_numbers <= 0)
    {
        return 0;
    }

    clustered_neighbor_runtime::Reserve_Device_Buffer(
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregate_numbers + 1,
        &ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices);
    ScatterEntryStarts(source_rows,
                       ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts.data,
                       ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_offsets.data,
                       ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data);
    deviceMemcpy(
        ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data +
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregate_numbers,
        &source_rows, sizeof(int), deviceMemcpyHostToDevice);

    clustered_neighbor_runtime::Reserve_Device_Buffer(
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregate_numbers,
        &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregates);
    DeviceBuffer<int> fill_cursor;
    clustered_neighbor_runtime::Reserve_Device_Buffer(1, &fill_cursor);
    deviceMemset(fill_cursor.data, 0, sizeof(int));
    Launch_Device_Kernel(
        Fill_Gmxpacked_Record_Stream_Aggregates_From_Sources,
        (ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregate_numbers +
         CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL,
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregate_numbers,
        ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data,
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_sources.data,
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregates.data,
        fill_cursor.data);

    int filled_aggregate_rows = 0;
    deviceMemcpy(&filled_aggregate_rows, fill_cursor.data, sizeof(int),
                 deviceMemcpyDeviceToHost);
    clustered_neighbor_runtime::Release_Device_Buffer(&fill_cursor);
    return filled_aggregate_rows;
#endif
}

#ifndef USE_CPU
namespace
{

static void FreeCompactTemps(
    CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE** sorted_aggregates,
    CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE** compact_entries)
{
    Free_Single_Device_Pointer(reinterpret_cast<void**>(sorted_aggregates));
    Free_Single_Device_Pointer(reinterpret_cast<void**>(compact_entries));
}

static int MaterializeCompactEntries(
    ClusteredNeighborProvider* layout, const int aggregate_rows,
    CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE** sorted_aggregates,
    CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE** compact_entries)
{
    using namespace clustered_neighbor_builder_internal;
    AllocateAggregateBuffer(aggregate_rows, sorted_aggregates,
                            "record-stream-sorted-aggregates");
    deviceMemcpy(*sorted_aggregates,
                 ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregates.data,
                 sizeof(CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE) *
                     static_cast<std::size_t>(aggregate_rows),
                 deviceMemcpyDeviceToDevice);

    clustered_neighbor_runtime::Reserve_Device_Buffer(
        aggregate_rows, &ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys);
    if (aggregate_rows > 1)
    {
        clustered_neighbor_runtime::Reserve_Device_Buffer(
            aggregate_rows, &ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices);
        BuildLinearIndices(aggregate_rows,
                           ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data);
        Launch_Device_Kernel(
            Build_Gmxpacked_Record_Stream_Compact_Shift_Sort_Keys,
            (aggregate_rows + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, aggregate_rows,
            ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data,
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregates.data,
            ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys.data);
        StableSortU64Int(layout, aggregate_rows,
                         ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys.data,
                         ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data);
        Launch_Device_Kernel(
            Build_Gmxpacked_Record_Stream_Compact_Cluster_J_Sort_Keys,
            (aggregate_rows + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, aggregate_rows,
            ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data,
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregates.data,
            ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys.data);
        StableSortU64Int(layout, aggregate_rows,
                         ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys.data,
                         ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data);
        Launch_Device_Kernel(
            Build_Gmxpacked_Record_Stream_Compact_Supercluster_Sort_Keys,
            (aggregate_rows + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, aggregate_rows,
            ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data,
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregates.data,
            ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys.data);
        StableSortU64Int(layout, aggregate_rows,
                         ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys.data,
                         ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data);
        Launch_Device_Kernel(
            Gather_Gmxpacked_Record_Stream_Aggregates_By_Index,
            (aggregate_rows + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, aggregate_rows,
            ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data,
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregates.data,
            *sorted_aggregates);
    }

    clustered_neighbor_runtime::Reserve_Device_Buffer(
        aggregate_rows + 1, &ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts);
    clustered_neighbor_runtime::Reserve_Device_Buffer(
        aggregate_rows + 1, &ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_offsets);
    Launch_Device_Kernel(Build_Gmxpacked_Record_Stream_Compact_Entry_Flags,
                         (aggregate_rows + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, aggregate_rows,
                         *sorted_aggregates,
                         ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts.data);
    const int compact_entry_numbers = ExclusiveScanCounts(
        layout, aggregate_rows, ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts.data,
        ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_offsets.data);
    if (compact_entry_numbers <= 0)
    {
        return compact_entry_numbers;
    }

    clustered_neighbor_runtime::Reserve_Device_Buffer(
        compact_entry_numbers + 1, &ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices);
    ScatterEntryStarts(aggregate_rows,
                       ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts.data,
                       ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_offsets.data,
                       ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data);
    deviceMemcpy(
        ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data + compact_entry_numbers,
        &aggregate_rows, sizeof(int), deviceMemcpyHostToDevice);

    AllocateAggregateBuffer(compact_entry_numbers, compact_entries,
                            "record-stream-compact-entries");
    (void)cudaGetLastError();
    Launch_Device_Kernel(
        Fill_Gmxpacked_Record_Stream_Compact_Entries,
        (compact_entry_numbers + kCompactExclusionBlockSize - 1) /
            kCompactExclusionBlockSize,
        kCompactExclusionBlockSize, 0, NULL, compact_entry_numbers,
        ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data, *sorted_aggregates,
        *compact_entries);
    CheckCudaStatus(cudaGetLastError(),
                    "record-stream-compact-entry-fill-launch");
    return compact_entry_numbers;
}

static inline __attribute__((always_inline)) bool MaterializeCompactSciAndCj(
    ClusteredNeighborProvider* layout, const int compact_entry_numbers,
    CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE* compact_entries,
    CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE* sorted_aggregates)
{
    using namespace clustered_neighbor_builder_internal;
    clustered_neighbor_runtime::Reserve_Device_Buffer(
        compact_entry_numbers, &ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys);
    if (compact_entry_numbers > 1)
    {
        clustered_neighbor_runtime::Reserve_Device_Buffer(
            compact_entry_numbers, &ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices);
        BuildLinearIndices(compact_entry_numbers,
                           ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data);
#define SORT_COMPACT_ENTRIES_BY(kernel)                                 \
    Launch_Device_Kernel(                                               \
        kernel,                                                         \
        (compact_entry_numbers + CONTROLLER::device_max_thread - 1) /   \
            CONTROLLER::device_max_thread,                              \
        CONTROLLER::device_max_thread, 0, NULL, compact_entry_numbers,  \
        ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data, compact_entries, \
        ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys.data);                       \
    StableSortU64Int(layout, compact_entry_numbers,                     \
                     ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys.data,           \
                     ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data)
        SORT_COMPACT_ENTRIES_BY(
            Build_Gmxpacked_Record_Stream_Compact_Cluster_J_Sort_Keys);
        SORT_COMPACT_ENTRIES_BY(
            Build_Gmxpacked_Record_Stream_Compact_Source_End_Sort_Keys);
        SORT_COMPACT_ENTRIES_BY(
            Build_Gmxpacked_Record_Stream_Compact_Source_Begin_Sort_Keys);
        SORT_COMPACT_ENTRIES_BY(
            Build_Gmxpacked_Record_Stream_Compact_Shift_Sort_Keys);
        SORT_COMPACT_ENTRIES_BY(
            Build_Gmxpacked_Record_Stream_Compact_Supercluster_Sort_Keys);
#undef SORT_COMPACT_ENTRIES_BY
        Launch_Device_Kernel(
            Gather_Gmxpacked_Record_Stream_Aggregates_By_Index,
            (compact_entry_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, compact_entry_numbers,
            ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data, compact_entries,
            sorted_aggregates);
        deviceMemcpy(compact_entries, sorted_aggregates,
                     sizeof(CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE) *
                         static_cast<std::size_t>(compact_entry_numbers),
                     deviceMemcpyDeviceToDevice);
    }

    clustered_neighbor_runtime::Reserve_Device_Buffer(
        compact_entry_numbers + 1, &ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts);
    clustered_neighbor_runtime::Reserve_Device_Buffer(
        compact_entry_numbers + 1, &ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_offsets);
    Launch_Device_Kernel(
        Build_Gmxpacked_Record_Stream_Compact_Sci_Flags,
        (compact_entry_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, compact_entry_numbers,
        compact_entries, ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts.data);
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers =
        ExclusiveScanCounts(layout, compact_entry_numbers,
                            ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts.data,
                            ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_offsets.data);
    if (ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers <= 0)
    {
        return false;
    }

    clustered_neighbor_runtime::Reserve_Device_Buffer(
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers + 1,
        &ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices);
    ScatterEntryStarts(compact_entry_numbers,
                       ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_counts.data,
                       ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_offsets.data,
                       ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data);
    deviceMemcpy(ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data +
                     ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers,
                 &compact_entry_numbers, sizeof(int), deviceMemcpyHostToDevice);

    clustered_neighbor_runtime::Reserve_Device_Buffer(
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers,
        &ClusteredNeighborProviderInternal::Workspace(layout).cjpacked_counts);
    clustered_neighbor_runtime::Reserve_Device_Buffer(
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers + 1,
        &ClusteredNeighborProviderInternal::Workspace(layout).cjpacked_group_offsets);
    Launch_Device_Kernel(Count_Gmxpacked_CjPacked_Per_Sci,
                         (ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers,
                         ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data,
                         ClusteredNeighborProviderInternal::Workspace(layout).cjpacked_counts.data);
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked_numbers =
        ExclusiveScanCounts(layout, ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers,
                            ClusteredNeighborProviderInternal::Workspace(layout).cjpacked_counts.data,
                            ClusteredNeighborProviderInternal::Workspace(layout).cjpacked_group_offsets.data);
    if (ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked_numbers <= 0)
    {
        return false;
    }

    clustered_neighbor_runtime::Reserve_Device_Buffer(
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers,
        &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci);
    clustered_neighbor_runtime::Reserve_Device_Buffer(
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked_numbers,
        &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked);
    Launch_Device_Kernel(Fill_Gmxpacked_Record_Stream_Sci_And_Cj,
                         (ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers,
                         ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data,
                         compact_entries,
                         ClusteredNeighborProviderInternal::Workspace(layout).cjpacked_group_offsets.data,
                         ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci.data,
                         ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked.data);
    return true;
}

}  // namespace

__attribute__((noinline, noclone)) bool
Build_Gmxpacked_Record_Stream_Compact_Payload_On_Device(
    ClusteredNeighborProvider* layout,
    clustered_neighbor_builder_internal::CompactPayloadSummary* summary,
    bool* reset_payload_on_failure)
{
    using namespace clustered_neighbor_builder_internal;
    if (reset_payload_on_failure != NULL)
    {
        *reset_payload_on_failure = false;
    }
    if (layout == NULL || summary == NULL || summary->aggregate_rows <= 0 ||
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_record_stream_aggregates.data == NULL)
    {
        return false;
    }

    Bind_Clustered_Working_Device(&ClusteredNeighborProviderInternal::WorkingDevice(layout));
    CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE* sorted_aggregates = NULL;
    CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE* compact_entries = NULL;
    const int compact_entry_numbers = MaterializeCompactEntries(
        layout, summary->aggregate_rows, &sorted_aggregates, &compact_entries);
    summary->compact_entries = compact_entry_numbers;
    if (compact_entry_numbers <= 0)
    {
        FreeCompactTemps(&sorted_aggregates, &compact_entries);
        return false;
    }
    if (!MaterializeCompactSciAndCj(layout, compact_entry_numbers,
                                    compact_entries, sorted_aggregates))
    {
        FreeCompactTemps(&sorted_aggregates, &compact_entries);
        if (reset_payload_on_failure != NULL)
        {
            *reset_payload_on_failure = true;
        }
        return false;
    }

    const int split_entry_numbers =
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked_numbers * kClusteredWarpSplitCount;
    clustered_neighbor_runtime::Reserve_Device_Buffer(
        split_entry_numbers, &ClusteredNeighborProviderInternal::Workspace(layout).exclusion_counts);
    clustered_neighbor_runtime::Reserve_Device_Buffer(
        split_entry_numbers + 1, &ClusteredNeighborProviderInternal::Workspace(layout).exclusion_offsets);
    Launch_Device_Kernel(
        Count_Gmxpacked_Record_Stream_Split_Exclusions,
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers, kCompactExclusionBlockSize, 0,
        NULL, ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers,
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci.data,
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked.data, compact_entries,
        ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data,
        ClusteredNeighborProviderInternal::Workspace(layout).exclusion_counts.data);
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_split_exclusion_numbers = ExclusiveScanCounts(
        layout, split_entry_numbers, ClusteredNeighborProviderInternal::Workspace(layout).exclusion_counts.data,
        ClusteredNeighborProviderInternal::Workspace(layout).exclusion_offsets.data);
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_exclusion_numbers =
        1 + ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_split_exclusion_numbers;
    clustered_neighbor_runtime::Reserve_Device_Buffer(
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_exclusion_numbers,
        &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_exclusions);
    Launch_Device_Kernel(Initialize_Gmxpacked_Exclusion_Rows,
                         (ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_exclusion_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_exclusion_numbers,
                         ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_exclusions.data);
    if (ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_split_exclusion_numbers > 0)
    {
        Launch_Device_Kernel(
            Fill_Gmxpacked_Record_Stream_Split_Exclusions,
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers, kCompactExclusionBlockSize,
            0, NULL, ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers,
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci.data,
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked.data, compact_entries,
            ClusteredNeighborProviderInternal::Workspace(layout).record_scratch_indices.data,
            ClusteredNeighborProviderInternal::Workspace(layout).exclusion_offsets.data,
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked.data,
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_exclusions.data);
    }

    summary->compact_sci = ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers;
    summary->compact_cj = ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked_numbers;
    summary->split_excl = ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_split_exclusion_numbers;
    summary->compact_excl = ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_exclusion_numbers;
    FreeCompactTemps(&sorted_aggregates, &compact_entries);
    return summary->compact_sci > 0 && summary->compact_cj > 0 &&
           summary->compact_excl > 0;
}
#endif
