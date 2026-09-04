#pragma once

#include "../common.h"
#include "../neighbor_list/contract/view.h"

#ifdef USE_GPU
#ifdef USE_HIP
#include <hipcub/hipcub.hpp>
#else
#include <cub/cub.cuh>
#endif
#endif

#ifdef USE_GPU
struct ClusteredCSRScanStats
{
    long long item_count = 0;
    int min_count = 0;
};
#endif

struct ClusteredPayloadStamp
{
    long long provider_incarnation = -1;
    long long payload_generation = -1;

    bool Matches(const CLUSTERED_SPATIAL_VIEW& view) const
    {
        return provider_incarnation == view.provider_incarnation &&
               payload_generation == view.gmxpacked_payload_generation;
    }

    void Capture(const CLUSTERED_SPATIAL_VIEW& view)
    {
        provider_incarnation = view.provider_incarnation;
        payload_generation = view.gmxpacked_payload_generation;
    }

    void Reset()
    {
        provider_incarnation = -1;
        payload_generation = -1;
    }
};

struct ClusteredCSRStorage
{
    int item_count = 0;
    int* counts = nullptr;
    int counts_capacity = 0;
    int* offsets = nullptr;
    int offsets_capacity = 0;
    int* items = nullptr;
    int items_capacity = 0;
#ifdef USE_GPU
    void* scan_workspace = nullptr;
    size_t scan_workspace_capacity = 0;
    ClusteredCSRScanStats* scan_stats = nullptr;
#endif

    void ReserveCounts(const int required)
    {
        Reserve(&counts, &counts_capacity, required);
    }

    void ReserveOffsets(const int required)
    {
        Reserve(&offsets, &offsets_capacity, required);
    }

    void ReserveItems(const int required)
    {
        Reserve(&items, &items_capacity, required);
    }

    void Clear()
    {
        Free_Single_Device_Pointer(reinterpret_cast<void**>(&counts));
        Free_Single_Device_Pointer(reinterpret_cast<void**>(&offsets));
        Free_Single_Device_Pointer(reinterpret_cast<void**>(&items));
        counts_capacity = 0;
        offsets_capacity = 0;
        items_capacity = 0;
        item_count = 0;
#ifdef USE_GPU
        Free_Single_Device_Pointer(&scan_workspace);
        Free_Single_Device_Pointer(reinterpret_cast<void**>(&scan_stats));
        scan_workspace_capacity = 0;
#endif
    }

#ifdef USE_GPU
    void ReserveScanWorkspace(const size_t required)
    {
        if (required <= scan_workspace_capacity && scan_workspace != nullptr)
        {
            return;
        }
        Free_Single_Device_Pointer(&scan_workspace);
        scan_workspace_capacity = 0;
        if (required > 0)
        {
            Device_Malloc_Safely(&scan_workspace, required);
            scan_workspace_capacity = required;
        }
    }

    void ReserveScanStats()
    {
        if (scan_stats == nullptr)
        {
            Device_Malloc_Safely(reinterpret_cast<void**>(&scan_stats),
                                 sizeof(ClusteredCSRScanStats));
        }
    }
#endif

   private:
    static void Reserve(int** pointer, int* capacity, const int required)
    {
        if (required <= *capacity && *pointer != nullptr)
        {
            return;
        }
        Free_Single_Device_Pointer(reinterpret_cast<void**>(pointer));
        *capacity = 0;
        if (required > 0)
        {
            Device_Malloc_Safely(reinterpret_cast<void**>(pointer),
                                 sizeof(int) * static_cast<size_t>(required));
            *capacity = required;
        }
    }
};

#ifdef USE_GPU
namespace clustered_csr_detail
{
#ifdef USE_HIP
namespace cub_backend = hipcub;
#else
namespace cub_backend = cub;
#endif

static __global__ void ValidateCountsAndSum(const int row_count,
                                            const int* counts,
                                            ClusteredCSRScanStats* stats)
{
    __shared__ long long block_sums[256];
    __shared__ int block_minima[256];
    const int lane = static_cast<int>(threadIdx.x);
    long long sum = 0;
    int minimum = 0x7fffffff;
    for (int row = lane; row < row_count; row += static_cast<int>(blockDim.x))
    {
        const int count = counts[row];
        sum += static_cast<long long>(count);
        minimum = count < minimum ? count : minimum;
    }
    block_sums[lane] = sum;
    block_minima[lane] = minimum;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride /= 2)
    {
        if (lane < stride)
        {
            block_sums[lane] += block_sums[lane + stride];
            const int other_minimum = block_minima[lane + stride];
            block_minima[lane] = other_minimum < block_minima[lane]
                                     ? other_minimum
                                     : block_minima[lane];
        }
        __syncthreads();
    }
    if (lane == 0)
    {
        stats->item_count = block_sums[0];
        stats->min_count = block_minima[0];
    }
}

static __global__ void FinalizeExclusiveScan(const int row_count, int* offsets,
                                             const int item_count)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        offsets[row_count] = item_count;
    }
}
}  // namespace clustered_csr_detail

inline bool Clustered_CSR_Device_Exclusive_Scan(ClusteredCSRStorage* storage,
                                                const int row_count)
{
    if (storage != nullptr) storage->item_count = 0;
    if (storage == nullptr || row_count < 0 ||
        (row_count > 0 &&
         (storage->counts == nullptr || storage->offsets == nullptr)))
    {
        return false;
    }
    if (row_count == 0)
    {
        storage->item_count = 0;
        if (storage->offsets != nullptr)
        {
            deviceMemset(storage->offsets, 0, sizeof(int));
        }
        return true;
    }

    storage->ReserveScanStats();
    Launch_Device_Kernel(clustered_csr_detail::ValidateCountsAndSum, dim3(1),
                         dim3(256), 0, nullptr, row_count, storage->counts,
                         storage->scan_stats);
    deviceError_t status = deviceGetLastError();
    if (status != DEVICE_MALLOC_SUCCESS)
    {
        return false;
    }
    ClusteredCSRScanStats host_stats;
    status =
        deviceMemcpy(&host_stats, storage->scan_stats,
                     sizeof(ClusteredCSRScanStats), deviceMemcpyDeviceToHost);
    if (status != DEVICE_MALLOC_SUCCESS || host_stats.min_count < 0 ||
        host_stats.item_count < 0 || host_stats.item_count > 0x7fffffffLL)
    {
        return false;
    }

    size_t scan_workspace_bytes = 0;
    status = clustered_csr_detail::cub_backend::DeviceScan::ExclusiveSum(
        nullptr, scan_workspace_bytes, storage->counts, storage->offsets,
        row_count);
    if (status != DEVICE_MALLOC_SUCCESS)
    {
        return false;
    }
    storage->ReserveScanWorkspace(scan_workspace_bytes);
    status = clustered_csr_detail::cub_backend::DeviceScan::ExclusiveSum(
        storage->scan_workspace, scan_workspace_bytes, storage->counts,
        storage->offsets, row_count);
    if (status != DEVICE_MALLOC_SUCCESS)
    {
        return false;
    }
    Launch_Device_Kernel(clustered_csr_detail::FinalizeExclusiveScan, dim3(1),
                         dim3(1), 0, nullptr, row_count, storage->offsets,
                         static_cast<int>(host_stats.item_count));
    status = deviceGetLastError();
    if (status != DEVICE_MALLOC_SUCCESS)
    {
        return false;
    }
    storage->item_count = static_cast<int>(host_stats.item_count);
    return true;
}
#endif
