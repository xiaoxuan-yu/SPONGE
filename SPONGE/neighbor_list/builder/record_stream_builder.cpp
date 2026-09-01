#include "internal.h"

#include "../provider/internal.h"
#include "../provider/lifecycle.h"
#include "../provider/runtime.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../MD_core/MD_core.h"

#ifndef USE_CPU
__global__ void Count_Gmxpacked_CjPacked_Per_Sci(
    int sci_numbers, const int* compact_sci_starts, int* cjpacked_counts);
__global__ void Initialize_Gmxpacked_Exclusion_Rows(
    int exclusion_numbers, CLUSTERED_GMXPACKED_EXCLUSION* exclusions);
bool Build_Gmxpacked_Record_Stream_Compact_Payload_On_Device(
    ClusteredNeighborProvider* layout,
    clustered_neighbor_builder_internal::CompactPayloadSummary* summary,
    bool* reset_payload_on_failure);
#endif

namespace
{

using clustered_neighbor_runtime::Reserve_Device_Buffer;
#ifndef USE_CPU
using clustered_neighbor_runtime::Bind_Clustered_Working_Device;
using clustered_neighbor_runtime::Clustered_Device_Malloc_Safely;
#endif

template <typename T>
static std::vector<T> Copy_Device_Buffer_To_Host(const T* device_ptr,
                                                 size_t count)
{
    std::vector<T> host(count);
    if (device_ptr != NULL && count > 0)
    {
        deviceMemcpy(host.data(), device_ptr, sizeof(T) * count,
                     deviceMemcpyDeviceToHost);
    }
    return host;
}

static CLUSTERED_GMXPACKED_EXCLUSION Make_Empty_Gmxpacked_No_Exclusion()
{
    CLUSTERED_GMXPACKED_EXCLUSION exclusion = {};
    for (unsigned int& pair_word : exclusion.pair)
    {
        pair_word = 0xffffffffu;
    }
    return exclusion;
}

#ifndef USE_CPU
static __global__ void Scatter_J_Entry_Sci_Starts(const int entry_numbers,
                                                  const int* sci_flags,
                                                  const int* sci_ids,
                                                  int* sci_starts)
{
    SIMPLE_DEVICE_FOR(entry_idx, entry_numbers)
    {
        if (sci_flags[entry_idx] != 0)
        {
            sci_starts[sci_ids[entry_idx]] = entry_idx;
        }
    }
}

static __global__ void Build_Linear_Indices(const int count, int* indices)
{
    SIMPLE_DEVICE_FOR(i, count) { indices[i] = i; }
}
#endif

using ClusteredGmxpackedRecordStreamCompactSummary =
    clustered_neighbor_builder_internal::CompactPayloadSummary;

#include "detail/compact_dispatch.inc.cuh"

}  // namespace

#ifndef USE_CPU
namespace clustered_neighbor_builder_internal
{

void AllocateAggregateBuffer(
    int capacity, CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE** pointer,
    const char* tag)
{
    Clustered_Device_Malloc_Safely(
        reinterpret_cast<void**>(pointer),
        sizeof(CLUSTERED_GMXPACKED_RECORD_STREAM_AGGREGATE) *
            static_cast<std::size_t>(capacity),
        tag);
}

void BuildLinearIndices(int count, int* indices)
{
    Launch_Device_Kernel(Build_Linear_Indices,
                         (count + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, count,
                         indices);
}

void ScatterEntryStarts(int entry_numbers, const int* entry_flags,
                        const int* entry_ids, int* entry_starts)
{
    Launch_Device_Kernel(Scatter_J_Entry_Sci_Starts,
                         (entry_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, entry_numbers,
                         entry_flags, entry_ids, entry_starts);
}

}  // namespace clustered_neighbor_builder_internal
#endif

#include "detail/compact_device.inc.cuh"

namespace clustered_neighbor_builder_internal
{

int BuildRecordStreamAggregates(ClusteredNeighborProvider* layout)
{
    return Build_Gmxpacked_Record_Stream_Aggregates(layout);
}

CompactPayloadSummary BuildCompactPayload(ClusteredNeighborProvider* layout)
{
    return Build_Gmxpacked_Record_Stream_Compact_Payload(layout);
}

}  // namespace clustered_neighbor_builder_internal
