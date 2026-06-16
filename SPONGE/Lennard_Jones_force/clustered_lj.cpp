#include "clustered_lj.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <numeric>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "LJ_soft_core.h"
#include "Lennard_Jones_force.h"
#include "../MD_core/MD_core.h"
#include "../third_party/cornerstone_octree/include/cstone/sfc/box.hpp"
#include "../third_party/cornerstone_octree/include/cstone/sfc/common.hpp"
#include "../third_party/cornerstone_octree/include/cstone/sfc/sfc.hpp"
#include "../third_party/cornerstone_octree/include/cstone/tree/csarray.hpp"
#include "../third_party/cornerstone_octree/include/cstone/tree/octree.hpp"
#include "../third_party/cornerstone_octree/include/cstone/traversal/boxoverlap.hpp"
#include "../third_party/cornerstone_octree/include/cstone/traversal/traversal.hpp"

#ifndef USE_CPU
#include <cub/cub.cuh>
#include "../third_party/cornerstone_octree/include/cstone/tree/octree_gpu.h"
#include "../third_party/cornerstone_octree/include/cstone/tree/update_gpu.cuh"
#endif

struct LJ_CORNERSTONE_STATE
{
#ifndef USE_CPU
    cstone::DeviceVector<uint64_t> leaves;
    cstone::DeviceVector<uint64_t> tmp_leaves;
    cstone::DeviceVector<unsigned> leaf_counts;
    cstone::DeviceVector<cstone::TreeNodeIndex> work_array;
    cstone::OctreeData<uint64_t, cstone::GpuTag> octree;
#else
    std::vector<uint64_t> leaves;
    std::vector<unsigned> leaf_counts;
    cstone::OctreeData<uint64_t, cstone::CpuTag> octree;
#endif
};

extern MD_INFORMATION md_info;

namespace
{

using CornerstoneKey = uint64_t;
using CornerstoneNodeIndex = cstone::TreeNodeIndex;
constexpr int kClusteredBuilderBlockSize = 128;
constexpr int kClusteredBuilderWarpSize = 32;
constexpr int kClusteredPruneBlockSize = 64;
constexpr int kClusteredGmxpackedExclusionBlockSize = 64;
constexpr int kClusteredMaxSuperClusterClusters = kClusteredSuperClusterClusters;
constexpr int kClusteredMaxJGroupSize = kClusteredJGroupSize;
constexpr int kClusteredForceonlyLocalSortCapacity = 2048;
constexpr unsigned int kClusteredForceonlyInvalidSortKey = 0xffffffffu;

#ifdef USE_CPU
struct HostClusteredJRecord
{
    int cluster_j = -1;
    unsigned int imask = 0u;
    std::array<unsigned long long, kClusteredMaxSuperClusterClusters>
        exclusion_masks = {};

};

struct HostClusteredBuildInput
{
    int total_atom_numbers = 0;
    int leaf_numbers = 0;
    int super_cluster_clusters = kClusteredSuperClusterClusters;
    int local_atom_numbers = 0;
    int cluster_size = kClusteredClusterSize;
    int candidate_sci_numbers = 0;
    float cutoff = 0.0f;
    LTMatrix3 cell = {};
    LTMatrix3 rcell = {};

    std::vector<int> permutation;
    std::vector<int> cluster_offsets;
    std::vector<unsigned int> cluster_valid_masks;
    std::vector<unsigned int> cluster_local_masks;
    std::vector<VECTOR> cluster_centers;
    std::vector<VECTOR> cluster_extents;
    std::vector<float> cluster_radii;
    std::vector<int> leaf_cluster_starts;
    std::vector<int> leaf_cluster_ends;
    std::vector<int> super_cluster_offsets;
    std::vector<int> cluster_to_supercluster;
    std::vector<VECTOR> super_cluster_centers;
    std::vector<int> sci_supercluster_ids;
    std::vector<int> candidate_leaf_offsets;
    std::vector<int> candidate_leaf_ids;
    std::vector<int> excluded_list_start;
    std::vector<int> excluded_list;
    std::vector<int> excluded_numbers;
};
#endif

static __device__ __forceinline__ VECTOR Clustered_Warp_Broadcast_Vector(
    const VECTOR value, const int src_lane)
{
    VECTOR result = value;
    result.x = deviceShfl(FULL_MASK, result.x, src_lane, warpSize);
    result.y = deviceShfl(FULL_MASK, result.y, src_lane, warpSize);
    result.z = deviceShfl(FULL_MASK, result.z, src_lane, warpSize);
    return result;
}

static __device__ __forceinline__ uint64_t Broadcast_Clustered_Warp_U64(
    uint64_t value, int src_lane)
{
    unsigned int lo = static_cast<unsigned int>(value);
    unsigned int hi = static_cast<unsigned int>(value >> 32);
    lo = deviceShfl(FULL_MASK, lo, src_lane, warpSize);
    hi = deviceShfl(FULL_MASK, hi, src_lane, warpSize);
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

struct ClusteredRecorderScope
{
    TIME_RECORDER* time_recorder;

    explicit ClusteredRecorderScope(TIME_RECORDER* recorder)
        : time_recorder(recorder)
    {
        if (time_recorder != NULL)
        {
            time_recorder->Start();
        }
    }

    ~ClusteredRecorderScope()
    {
        if (time_recorder != NULL)
        {
            time_recorder->Stop();
        }
    }
};

static int Note_Clustered_Step_Counter(int current_step, int* counter_step,
                                       int* count_this_step,
                                       long long* count_total)
{
    if (*counter_step != current_step)
    {
        *counter_step = current_step;
        *count_this_step = 0;
    }
    *count_this_step += 1;
    *count_total += 1;
    return *count_this_step;
}

struct ClusteredEarlyRecordAnalyzeEntry
{
    int supercluster_id = 0;
    int cluster_j = -1;
    uint32_t semantic_bits = 0u;
    unsigned long long exclusion_masks[kClusteredMaxSuperClusterClusters] = {};
};
static_assert(sizeof(ClusteredEarlyRecordAnalyzeEntry) == 80,
              "Unexpected early-record analysis entry size.");

static __host__ __device__ __forceinline__ uint32_t
Pack_Clustered_Early_Record_Semantic_Bits(int output_shift_id,
                                          unsigned int record_imask,
                                          unsigned int valid_mask_j,
                                          unsigned int local_mask_j)
{
    return (static_cast<uint32_t>(output_shift_id) & 0xffu) |
           ((record_imask & 0xffu) << 8) |
           ((valid_mask_j & 0xffu) << 16) |
           ((local_mask_j & 0xffu) << 24);
}

static __host__ __device__ __forceinline__ int
Clustered_Early_Record_Output_Shift_Id(uint32_t semantic_bits)
{
    return static_cast<int>(semantic_bits & 0xffu);
}

static __host__ __device__ __forceinline__ unsigned int
Clustered_Early_Record_Record_Imask(uint32_t semantic_bits)
{
    return (semantic_bits >> 8) & 0xffu;
}

static __host__ __device__ __forceinline__ unsigned int
Clustered_Early_Record_Valid_Mask(uint32_t semantic_bits)
{
    return (semantic_bits >> 16) & 0xffu;
}

static __host__ __device__ __forceinline__ unsigned int
Clustered_Early_Record_Local_Mask(uint32_t semantic_bits)
{
    return (semantic_bits >> 24) & 0xffu;
}

#ifndef USE_CPU
static __device__ __forceinline__ void Record_Clustered_Early_Record_Analysis(
    int supercluster_id, int output_shift_id, int cluster_j,
    unsigned int record_imask, unsigned int valid_mask_j,
    unsigned int local_mask_j, const unsigned long long* exclusion_masks,
    int analysis_record_capacity, int* analysis_raw_record_count,
    ClusteredEarlyRecordAnalyzeEntry* analysis_entries)
{
    if (analysis_raw_record_count == NULL || analysis_entries == NULL ||
        analysis_record_capacity <= 0)
    {
        return;
    }

    const int record_index = atomicAdd(analysis_raw_record_count, 1);
    if (record_index >= analysis_record_capacity)
    {
        return;
    }

    ClusteredEarlyRecordAnalyzeEntry entry = {};
    entry.supercluster_id = supercluster_id;
    entry.cluster_j = cluster_j;
    entry.semantic_bits = Pack_Clustered_Early_Record_Semantic_Bits(
        output_shift_id, record_imask, valid_mask_j, local_mask_j);
#pragma unroll
    for (int i_local = 0; i_local < kClusteredMaxSuperClusterClusters;
         i_local += 1)
    {
        entry.exclusion_masks[i_local] = exclusion_masks[i_local];
    }
    analysis_entries[record_index] = entry;
}
#endif

static __host__ __device__ __forceinline__ bool Clusters_May_Share_Molecule(
    int cluster_i, int cluster_j, unsigned int local_mask_i,
    unsigned int valid_mask_j, int cluster_size,
    const uint64_t* cluster_molecule_signatures,
    const int* cluster_molecule_ids)
{
    if (cluster_molecule_signatures == NULL || cluster_molecule_ids == NULL)
    {
        return true;
    }

    const uint64_t signature_i = cluster_molecule_signatures[cluster_i];
    const uint64_t signature_j = cluster_molecule_signatures[cluster_j];
    if ((signature_i & signature_j) == 0ull)
    {
        return false;
    }

    const int* molecule_ids_i = cluster_molecule_ids + cluster_i * cluster_size;
    const int* molecule_ids_j = cluster_molecule_ids + cluster_j * cluster_size;
    for (int lane_i = 0; lane_i < cluster_size; lane_i += 1)
    {
        if ((local_mask_i & (1u << lane_i)) == 0u)
        {
            continue;
        }
        const int mol_i = molecule_ids_i[lane_i];
        if (mol_i < 0)
        {
            return true;
        }
        for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
        {
            if ((valid_mask_j & (1u << lane_j)) == 0u)
            {
                continue;
            }
            const int mol_j = molecule_ids_j[lane_j];
            if (mol_j < 0)
            {
                return true;
            }
            if (mol_i == mol_j)
            {
                return true;
            }
        }
    }
    return false;
}

#ifndef USE_CPU
static void Clustered_Device_Malloc_Safely(void** pointer, size_t bytes,
                                           const char* tag)
{
    if (bytes == 0)
    {
        *pointer = NULL;
        return;
    }
    const deviceError_t error = deviceMalloc(pointer, bytes);
    if (error != DEVICE_MALLOC_SUCCESS)
    {
        int current_device = -1;
        cudaGetDevice(&current_device);
        fprintf(stderr,
                "clustered alloc failed: tag=%s bytes=%zu device=%d error=%s\n",
                tag, bytes, current_device, deviceGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#endif

template <typename T>
static void Reserve_Device_Buffer(int capacity, T** pointer,
                                  int* current_capacity)
{
    if (capacity <= *current_capacity && *pointer != NULL)
    {
        return;
    }
    if (*pointer != NULL)
    {
        Free_Single_Device_Pointer((void**)pointer);
    }
#ifndef USE_CPU
    Clustered_Device_Malloc_Safely((void**)pointer, sizeof(T) * capacity,
                                   "reserve-device-buffer");
#else
    Device_Malloc_Safely((void**)pointer, sizeof(T) * capacity);
#endif
    *current_capacity = capacity;
}

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

#ifndef USE_CPU
static void Reserve_Device_Byte_Buffer(size_t capacity_bytes,
                                       unsigned char** pointer,
                                       size_t* current_capacity_bytes)
{
    if (capacity_bytes == 0)
    {
        return;
    }
    if (capacity_bytes <= *current_capacity_bytes && *pointer != NULL)
    {
        return;
    }
    if (*pointer != NULL)
    {
        Free_Single_Device_Pointer((void**)pointer);
    }
    Clustered_Device_Malloc_Safely((void**)pointer, capacity_bytes,
                                   "reserve-byte-buffer");
    *current_capacity_bytes = capacity_bytes;
}

static void Reserve_Device_Opaque_Buffer(size_t capacity_bytes, void** pointer,
                                         size_t* current_capacity_bytes)
{
    if (capacity_bytes == 0)
    {
        return;
    }
    if (capacity_bytes <= *current_capacity_bytes && *pointer != NULL)
    {
        return;
    }
    if (*pointer != NULL)
    {
        Free_Single_Device_Pointer(pointer);
    }
    Clustered_Device_Malloc_Safely(pointer, capacity_bytes,
                                   "reserve-opaque-buffer");
    *current_capacity_bytes = capacity_bytes;
}
#endif

#ifndef USE_CPU
static inline int Normalize_Clustered_Working_Device(int working_device)
{
    int device_count = 0;
    deviceGetDeviceCount(&device_count);
    if (device_count <= 0)
    {
        return 0;
    }
    return working_device >= 0 && working_device < device_count ? working_device
                                                                 : 0;
}

static inline void Bind_Clustered_Working_Device(int* working_device)
{
    const int target_device = Normalize_Clustered_Working_Device(
        working_device != NULL ? *working_device : 0);
    if (working_device != NULL)
    {
        *working_device = target_device;
    }
    setWorkingDevice(target_device);
}

static bool Clustered_Trace_Warp_Records_Enabled();

static void Clustered_Check_Cuda_Status(cudaError_t error, const char* tag)
{
    if (error == cudaSuccess)
    {
        return;
    }
    int current_device = -1;
    cudaGetDevice(&current_device);
    fprintf(stderr,
            "clustered cuda failure: tag=%s device=%d error=%s\n", tag,
            current_device, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
}

static void Reset_Clustered_Sort_Scratch(LJ_CLUSTER_LAYOUT* layout)
{
    if (layout == NULL)
    {
        return;
    }
    Free_Single_Device_Pointer((void**)&layout->d_sort_key_buffer);
    Free_Single_Device_Pointer((void**)&layout->d_sort_value_buffer);
    Free_Single_Device_Pointer((void**)&layout->d_sort_temp_storage);
    layout->sort_key_buffer_bytes = 0;
    layout->sort_value_buffer_bytes = 0;
    layout->sort_temp_storage_bytes = 0;
}

static void Sort_Cornerstone_Keys_On_Device(LJ_CLUSTER_LAYOUT* layout,
                                            int count,
                                            uint64_t* d_keys,
                                            int* d_values)
{
    if (count <= 1 || d_keys == NULL || d_values == NULL || layout == NULL)
    {
        return;
    }
    Bind_Clustered_Working_Device(&layout->working_device);
    if (Clustered_Trace_Warp_Records_Enabled())
    {
        uint64_t first_key = 0;
        uint64_t mid_key = 0;
        uint64_t last_key = 0;
        const int mid_index = count / 2;
        deviceMemcpy(&first_key, d_keys, sizeof(uint64_t),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&mid_key, d_keys + mid_index, sizeof(uint64_t),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&last_key, d_keys + count - 1, sizeof(uint64_t),
                     deviceMemcpyDeviceToHost);
        fprintf(stderr,
                "[clustered sort pre] step=%d first=%llu mid=%llu last=%llu\n",
                md_info.sys.steps,
                static_cast<unsigned long long>(first_key),
                static_cast<unsigned long long>(mid_key),
                static_cast<unsigned long long>(last_key));
    }

    const cudaError_t preexisting_err = cudaGetLastError();
    (void)preexisting_err;
    const uint64_t temp_storage_bytes =
        cstone::sortByKeyTempStorage<uint64_t, int>(
            static_cast<uint64_t>(count));
    Reserve_Device_Byte_Buffer(sizeof(uint64_t) * static_cast<size_t>(count),
                               &layout->d_sort_key_buffer,
                               &layout->sort_key_buffer_bytes);
    Reserve_Device_Byte_Buffer(sizeof(int) * static_cast<size_t>(count),
                               &layout->d_sort_value_buffer,
                               &layout->sort_value_buffer_bytes);
    Reserve_Device_Opaque_Buffer(temp_storage_bytes,
                                 &layout->d_sort_temp_storage,
                                 &layout->sort_temp_storage_bytes);

    cstone::sortByKeyGpu<uint64_t, int>(
        d_keys, d_keys + count, d_values,
        reinterpret_cast<uint64_t*>(layout->d_sort_key_buffer),
        reinterpret_cast<int*>(layout->d_sort_value_buffer),
        layout->d_sort_temp_storage, temp_storage_bytes);

    const cudaError_t post_sort_err = cudaGetLastError();
    const cudaError_t sync_err = cudaDeviceSynchronize();
    Clustered_Check_Cuda_Status(post_sort_err,
                                "clustered-cstone-sort-post-launch");
    Clustered_Check_Cuda_Status(sync_err, "clustered-cstone-sort-sync");

    if (Clustered_Trace_Warp_Records_Enabled())
    {
        uint64_t first_key = 0;
        uint64_t mid_key = 0;
        uint64_t last_key = 0;
        const int mid_index = count / 2;
        deviceMemcpy(&first_key, d_keys, sizeof(uint64_t),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&mid_key, d_keys + mid_index, sizeof(uint64_t),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&last_key, d_keys + count - 1, sizeof(uint64_t),
                     deviceMemcpyDeviceToHost);
        fprintf(stderr,
                "[clustered sort post] step=%d first=%llu mid=%llu last=%llu\n",
                md_info.sys.steps,
                static_cast<unsigned long long>(first_key),
                static_cast<unsigned long long>(mid_key),
                static_cast<unsigned long long>(last_key));
    }
}

template <typename Key, typename Value>
static void Stable_Sort_Device_By_Key(LJ_CLUSTER_LAYOUT* layout, int count,
                                      Key* d_keys, Value* d_values)
{
    if (count <= 1 || d_keys == NULL || d_values == NULL)
    {
        return;
    }
    static_assert(std::is_trivially_copyable_v<Key>,
                  "Clustered device sort requires trivially copyable keys.");
    static_assert(std::is_trivially_copyable_v<Value>,
                  "Clustered device sort requires trivially copyable values.");
    if (layout == NULL)
    {
        return;
    }
    Bind_Clustered_Working_Device(&layout->working_device);

    if constexpr (std::is_same_v<Key, uint64_t>)
    {
        if (Clustered_Trace_Warp_Records_Enabled() &&
            count == layout->total_atom_numbers && count > 0)
        {
            uint64_t first_key = 0;
            uint64_t mid_key = 0;
            uint64_t last_key = 0;
            const int mid_index = count / 2;
            deviceMemcpy(&first_key, d_keys, sizeof(uint64_t),
                         deviceMemcpyDeviceToHost);
            deviceMemcpy(&mid_key, d_keys + mid_index, sizeof(uint64_t),
                         deviceMemcpyDeviceToHost);
            deviceMemcpy(&last_key, d_keys + count - 1, sizeof(uint64_t),
                         deviceMemcpyDeviceToHost);
            fprintf(stderr,
                    "[clustered sort pre] step=%d first=%llu mid=%llu last=%llu\n",
                    md_info.sys.steps,
                    static_cast<unsigned long long>(first_key),
                    static_cast<unsigned long long>(mid_key),
                    static_cast<unsigned long long>(last_key));
        }
    }

    const cudaError_t preexisting_err = cudaGetLastError();
    size_t temp_storage_bytes = 0;

    cub::DoubleBuffer<Key> key_buffers(d_keys, NULL);
    cub::DoubleBuffer<Value> value_buffers(d_values, NULL);
    const cudaError_t query_err = cub::DeviceRadixSort::SortPairs(
        NULL, temp_storage_bytes, key_buffers, value_buffers, count);
    Clustered_Check_Cuda_Status(query_err, "clustered-radix-sort-query");

    Reserve_Device_Byte_Buffer(sizeof(Key) * static_cast<size_t>(count),
                               &layout->d_sort_key_buffer,
                               &layout->sort_key_buffer_bytes);
    Reserve_Device_Byte_Buffer(sizeof(Value) * static_cast<size_t>(count),
                               &layout->d_sort_value_buffer,
                               &layout->sort_value_buffer_bytes);
    Reserve_Device_Opaque_Buffer(temp_storage_bytes, &layout->d_sort_temp_storage,
                                 &layout->sort_temp_storage_bytes);

    key_buffers = cub::DoubleBuffer<Key>(
        d_keys, reinterpret_cast<Key*>(layout->d_sort_key_buffer));
    value_buffers = cub::DoubleBuffer<Value>(
        d_values, reinterpret_cast<Value*>(layout->d_sort_value_buffer));
    const cudaError_t sort_err = cub::DeviceRadixSort::SortPairs(
        layout->d_sort_temp_storage, temp_storage_bytes, key_buffers,
        value_buffers, count);
    Clustered_Check_Cuda_Status(sort_err, "clustered-radix-sort-run");
    const cudaError_t sync_err = cudaDeviceSynchronize();
    Clustered_Check_Cuda_Status(sync_err, "clustered-radix-sort-sync");

    if (Clustered_Trace_Warp_Records_Enabled())
    {
        fprintf(stderr,
                "[clustered sort status] step=%d count=%d pre_err=%s query_err=%s sort_err=%s sync_err=%s temp_bytes=%zu key_current=%p value_current=%p key_base=%p value_base=%p\n",
                md_info.sys.steps, count,
                cudaGetErrorString(preexisting_err),
                cudaGetErrorString(query_err), cudaGetErrorString(sort_err),
                cudaGetErrorString(sync_err), temp_storage_bytes,
                (void*)key_buffers.Current(), (void*)value_buffers.Current(),
                (void*)d_keys, (void*)d_values);
    }

    if (key_buffers.Current() != d_keys)
    {
        deviceMemcpy(d_keys, key_buffers.Current(), sizeof(Key) * count,
                     deviceMemcpyDeviceToDevice);
    }
    if (value_buffers.Current() != d_values)
    {
        deviceMemcpy(d_values, value_buffers.Current(), sizeof(Value) * count,
                     deviceMemcpyDeviceToDevice);
    }

    if constexpr (std::is_same_v<Key, uint64_t>)
    {
        if (Clustered_Trace_Warp_Records_Enabled() &&
            count == layout->total_atom_numbers && count > 0)
        {
            uint64_t first_key = 0;
            uint64_t mid_key = 0;
            uint64_t last_key = 0;
            const int mid_index = count / 2;
            deviceMemcpy(&first_key, d_keys, sizeof(uint64_t),
                         deviceMemcpyDeviceToHost);
            deviceMemcpy(&mid_key, d_keys + mid_index, sizeof(uint64_t),
                         deviceMemcpyDeviceToHost);
            deviceMemcpy(&last_key, d_keys + count - 1, sizeof(uint64_t),
                         deviceMemcpyDeviceToHost);
            fprintf(stderr,
                    "[clustered sort post] step=%d first=%llu mid=%llu last=%llu\n",
                    md_info.sys.steps,
                    static_cast<unsigned long long>(first_key),
                    static_cast<unsigned long long>(mid_key),
                    static_cast<unsigned long long>(last_key));
        }
    }
}

#endif

static __host__ __device__ __forceinline__ int Clamp_Shift_Component(int shift)
{
    return shift < -1 ? -1 : (shift > 1 ? 1 : shift);
}

static __host__ __device__ __forceinline__ int Encode_Shift_Id(int sx, int sy,
                                                               int sz)
{
    return (Clamp_Shift_Component(sx) + 1) * 9 +
           (Clamp_Shift_Component(sy) + 1) * 3 +
           (Clamp_Shift_Component(sz) + 1);
}

static __host__ __device__ __forceinline__ VECTOR Shift_Fractional_From_Id(
    int shift_id)
{
    return {
        static_cast<float>(shift_id / 9 - 1),
        static_cast<float>((shift_id % 9) / 3 - 1),
        static_cast<float>(shift_id % 3 - 1)};
}

static __host__ __device__ __forceinline__ VECTOR Shift_Vector_From_Id(
    int shift_id, LTMatrix3 cell)
{
    const VECTOR frac_shift = Shift_Fractional_From_Id(shift_id);
    return frac_shift * cell;
}

static __host__ __device__ __forceinline__ int Determine_Shift_Id(
    VECTOR frac_i, VECTOR frac_j)
{
    const VECTOR dfrac = frac_j - frac_i;
    const int sx = static_cast<int>(floorf(dfrac.x + 0.5f));
    const int sy = static_cast<int>(floorf(dfrac.y + 0.5f));
    const int sz = static_cast<int>(floorf(dfrac.z + 0.5f));
    return Encode_Shift_Id(sx, sy, sz);
}

static __host__ __device__ __forceinline__ int Determine_Cluster_Shift_Id(
    VECTOR super_cluster_frac_center, VECTOR cluster_center, LTMatrix3 rcell)
{
    VECTOR cluster_frac = cluster_center * rcell;
    cluster_frac.x -= floorf(cluster_frac.x);
    cluster_frac.y -= floorf(cluster_frac.y);
    cluster_frac.z -= floorf(cluster_frac.z);
    return Determine_Shift_Id(super_cluster_frac_center, cluster_frac);
}

static __host__ __device__ __forceinline__ int Determine_Cluster_Pair_Shift_Id(
    VECTOR center_i, VECTOR center_j, LTMatrix3 rcell)
{
    VECTOR frac_i = center_i * rcell;
    frac_i.x -= floorf(frac_i.x);
    frac_i.y -= floorf(frac_i.y);
    frac_i.z -= floorf(frac_i.z);
    VECTOR frac_j = center_j * rcell;
    frac_j.x -= floorf(frac_j.x);
    frac_j.y -= floorf(frac_j.y);
    frac_j.z -= floorf(frac_j.z);
    return Determine_Shift_Id(frac_i, frac_j);
}

static __host__ __device__ __forceinline__ bool Cluster_Reach_Overlaps_Shifted(
    VECTOR center_i, VECTOR center_j, float radius_i, float radius_j,
    float cutoff, VECTOR shift_vec)
{
    const VECTOR dr = center_j - (center_i + shift_vec);
    const float reach = cutoff + radius_i + radius_j;
    return dr * dr <= reach * reach;
}

static __host__ __device__ __forceinline__ VECTOR Fractional_Extent_Pad(
    VECTOR extent, LTMatrix3 rcell)
{
    return {
        fabsf(extent.x * rcell.a11) + fabsf(extent.y * rcell.a21) +
            fabsf(extent.z * rcell.a31),
        fabsf(extent.y * rcell.a22) + fabsf(extent.z * rcell.a32),
        fabsf(extent.z * rcell.a33)};
}

static __host__ __device__ __forceinline__ bool Cluster_Aabb_Overlaps_Shifted(
    VECTOR center_i, VECTOR extent_i, VECTOR center_j, VECTOR extent_j,
    float cutoff, VECTOR shift_vec)
{
    const VECTOR dr = center_j - (center_i + shift_vec);
    const float gap_x =
        fmaxf(fabsf(dr.x) - (extent_i.x + extent_j.x), 0.0f);
    const float gap_y =
        fmaxf(fabsf(dr.y) - (extent_i.y + extent_j.y), 0.0f);
    const float gap_z =
        fmaxf(fabsf(dr.z) - (extent_i.z + extent_j.z), 0.0f);
    return gap_x * gap_x + gap_y * gap_y + gap_z * gap_z <= cutoff * cutoff;
}

static __host__ __device__ __forceinline__ bool Cluster_Aabb_Overlaps(
    VECTOR center_i, VECTOR extent_i, VECTOR center_j, VECTOR extent_j,
    float cutoff, LTMatrix3 cell, LTMatrix3 rcell)
{
    const VECTOR dr = Get_Periodic_Displacement(center_j, center_i, cell, rcell);
    const float gap_x =
        fmaxf(fabsf(dr.x) - (extent_i.x + extent_j.x), 0.0f);
    const float gap_y =
        fmaxf(fabsf(dr.y) - (extent_i.y + extent_j.y), 0.0f);
    const float gap_z =
        fmaxf(fabsf(dr.z) - (extent_i.z + extent_j.z), 0.0f);
    return gap_x * gap_x + gap_y * gap_y + gap_z * gap_z <= cutoff * cutoff;
}

static __host__ __device__ __forceinline__ unsigned int
Build_Fixed_Shift_Cluster_I_Mask(
    int cluster_i_start, int cluster_i_end, int cluster_j, int fixed_shift_id,
    float cutoff, VECTOR shift_vec, const unsigned int* cluster_local_masks,
    const VECTOR* cluster_centers, const VECTOR* cluster_extents)
{
    const VECTOR center_j = cluster_centers[cluster_j];
    const VECTOR extent_j = cluster_extents[cluster_j];
    unsigned int i_mask = 0u;
    for (int cluster_i = cluster_i_start; cluster_i < cluster_i_end;
         cluster_i += 1)
    {
        const int i_local = cluster_i - cluster_i_start;
        if (cluster_local_masks[cluster_i] == 0u)
        {
            continue;
        }
        if (fixed_shift_id == kClusteredCentralShiftId &&
            cluster_j >= cluster_i_start && cluster_j < cluster_i_end &&
            cluster_i > cluster_j)
        {
            continue;
        }
        if (Cluster_Aabb_Overlaps_Shifted(
                cluster_centers[cluster_i], cluster_extents[cluster_i],
                center_j, extent_j, cutoff, shift_vec))
        {
            i_mask |= (1u << static_cast<unsigned int>(i_local));
        }
    }
    return i_mask;
}

static __host__ __device__ __forceinline__ bool
Leaf_Has_Fixed_Shift_Candidate_Overlap(
    int cluster_i_start, int cluster_i_end, int leaf_cluster_start,
    int leaf_cluster_end, int fixed_shift_id, float cutoff, VECTOR shift_vec,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents)
{
    for (int cluster_j = leaf_cluster_start; cluster_j < leaf_cluster_end;
         cluster_j += 1)
    {
        if (cluster_valid_masks[cluster_j] == 0u)
        {
            continue;
        }
        if (Build_Fixed_Shift_Cluster_I_Mask(
                cluster_i_start, cluster_i_end, cluster_j, fixed_shift_id,
                cutoff, shift_vec, cluster_local_masks, cluster_centers,
                cluster_extents) != 0u)
        {
            return true;
        }
    }
    return false;
}

static __host__ __device__ __forceinline__ LJ_CLUSTERED_IMEI
Make_Empty_Clustered_Imei()
{
    LJ_CLUSTERED_IMEI imei;
    imei.imask = 0u;
#pragma unroll
    for (int index = 0;
         index < kClusteredMaxJGroupSize * kClusteredMaxSuperClusterClusters;
         index += 1)
    {
        imei.excl_ind[index] = -1;
    }
    return imei;
}

static __host__ __device__ __forceinline__ LJ_CLUSTERED_CJ_PACKED
Make_Empty_Clustered_CjPacked()
{
    LJ_CLUSTERED_CJ_PACKED packed;
#pragma unroll
    for (int jm = 0; jm < kClusteredMaxJGroupSize; jm += 1)
    {
        packed.cj[jm] = -1;
    }
#pragma unroll
    for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
    {
        packed.imei[split] = Make_Empty_Clustered_Imei();
    }
    return packed;
}

static LJ_CLUSTERED_GMXPACKED_EXCLUSION Make_Empty_Gmxpacked_No_Exclusion()
{
    LJ_CLUSTERED_GMXPACKED_EXCLUSION exclusion = {};
    for (unsigned int& pair_word : exclusion.pair)
    {
        pair_word = 0xffffffffu;
    }
    return exclusion;
}

static __host__ __device__ __forceinline__ bool Clustered_Split_Has_Atoms(
    unsigned int valid_mask_j, int split)
{
    return (valid_mask_j & Clustered_Split_Valid_Mask(split)) != 0u;
}

static __host__ __device__ __forceinline__ void
Append_Record_To_Clustered_CjPacked(
    LJ_CLUSTERED_CJ_PACKED* packed, const int jm, const int cluster_j,
    const unsigned int valid_mask_j, const unsigned int record_imask,
    const unsigned long long* exclusion_masks, int* write_exclusion,
    unsigned long long* exclusion_mask_pool,
    const bool dedup_exclusion_masks = false)
{
    packed->cj[jm] = cluster_j;

    int exclusion_indices[kClusteredMaxSuperClusterClusters];
#pragma unroll
    for (int i_local = 0; i_local < kClusteredMaxSuperClusterClusters;
         i_local += 1)
    {
        exclusion_indices[i_local] = -1;
        if (exclusion_masks[i_local] != 0ull)
        {
            int reused_index = -1;
            if (dedup_exclusion_masks)
            {
#pragma unroll
                for (int prev_local = 0; prev_local < i_local; prev_local += 1)
                {
                    if (exclusion_indices[prev_local] >= 0 &&
                        exclusion_masks[prev_local] == exclusion_masks[i_local])
                    {
                        reused_index = exclusion_indices[prev_local];
                        break;
                    }
                }
            }
            if (reused_index >= 0)
            {
                exclusion_indices[i_local] = reused_index;
            }
            else
            {
                exclusion_indices[i_local] = *write_exclusion;
                exclusion_mask_pool[*write_exclusion] = exclusion_masks[i_local];
                *write_exclusion += 1;
            }
        }
    }

#pragma unroll
    for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
    {
        if (!Clustered_Split_Has_Atoms(valid_mask_j, split))
        {
            continue;
        }
        packed->imei[split].imask |=
            record_imask << Clustered_Jm_Imask_Shift(jm);
#pragma unroll
        for (int i_local = 0; i_local < kClusteredMaxSuperClusterClusters;
             i_local += 1)
        {
            if (exclusion_indices[i_local] >= 0)
            {
                Clustered_Exclusion_Index_Ref(packed->imei[split], jm, i_local) =
                    exclusion_indices[i_local];
            }
        }
    }
}

static __host__ __device__ __forceinline__ void
Append_Record_To_Gmxpacked_CjPacked(
    LJ_CLUSTERED_GMXPACKED_CJ* packed, const int packed_idx, const int jm,
    const int cluster_i_start, const int cluster_numbers, const int cluster_j,
    const int shift_id, const unsigned int valid_mask_j,
    const unsigned int local_mask_j, const unsigned int record_imask,
    const unsigned int* shared_i_local_masks,
    const unsigned long long* exclusion_masks,
    LJ_CLUSTERED_GMXPACKED_EXCLUSION* gmxpacked_exclusions)
{
    packed->cj[jm] = cluster_j;

#pragma unroll
    for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
    {
        if (!Clustered_Split_Has_Atoms(valid_mask_j, split))
        {
            continue;
        }
        packed->split[split].imask |=
            record_imask << Clustered_Jm_Imask_Shift(jm);

        const int exclusion_index =
            1 + packed_idx * kClusteredWarpSplitCount + split;
        packed->split[split].exclusion_index = exclusion_index;
        LJ_CLUSTERED_GMXPACKED_EXCLUSION* row =
            gmxpacked_exclusions + exclusion_index;

#pragma unroll
        for (int i_local = 0; i_local < kClusteredMaxSuperClusterClusters;
             i_local += 1)
        {
            const unsigned int imask_bit = 1u << static_cast<unsigned int>(i_local);
            if ((record_imask & imask_bit) == 0u)
            {
                continue;
            }
            const int cluster_i = cluster_i_start + i_local;
            const unsigned int local_mask_i =
                cluster_i >= 0 && cluster_i < cluster_numbers
                    ? shared_i_local_masks[i_local]
                    : 0u;
            const unsigned long long exclusion_mask = exclusion_masks[i_local];
            const unsigned int packed_bit =
                1u << static_cast<unsigned int>(jm * kClusteredSuperClusterClusters +
                                                i_local);

#pragma unroll
            for (int split_j_lane = 0;
                 split_j_lane < kClusteredSplitJClusterSize; split_j_lane += 1)
            {
                const int j_lane =
                    split * kClusteredSplitJClusterSize + split_j_lane;
                const bool valid_j =
                    (valid_mask_j & (1u << static_cast<unsigned int>(j_lane))) != 0u;
                const bool local_j =
                    (local_mask_j & (1u << static_cast<unsigned int>(j_lane))) != 0u;
#pragma unroll
                for (int i_lane = 0; i_lane < kClusteredClusterSize; i_lane += 1)
                {
                    const bool local_i =
                        (local_mask_i &
                         (1u << static_cast<unsigned int>(i_lane))) != 0u;
                    bool allow_pair = valid_j && local_i;
                    if (allow_pair && shift_id == kClusteredCentralShiftId &&
                        cluster_i == cluster_j && local_j && j_lane <= i_lane)
                    {
                        allow_pair = false;
                    }
                    if (allow_pair &&
                        (exclusion_mask &
                         (1ull << static_cast<unsigned int>(
                              i_lane * kClusteredClusterSize + j_lane))) != 0ull)
                    {
                        allow_pair = false;
                    }
                    if (allow_pair)
                    {
                        row->pair[split_j_lane * kClusteredClusterSize + i_lane] |=
                            packed_bit;
                    }
                }
            }
        }
    }
}

static __global__ void Gather_Sorted_LJ_Direct_Scratch(
    const int atom_numbers, const int cluster_numbers, const int* permutation,
    const int* cluster_offsets, const VECTOR* cluster_centers,
    const LTMatrix3 cell, const LTMatrix3 rcell, const VECTOR_LJ* src,
    int* sorted_atom_ids, float4* sorted_xq, int* sorted_lj_type)
{
    SIMPLE_DEVICE_FOR(sorted_i, atom_numbers)
    {
        int cluster_lo = 0;
        int cluster_hi = cluster_numbers;
        while (cluster_lo + 1 < cluster_hi)
        {
            const int cluster_mid = (cluster_lo + cluster_hi) >> 1;
            if (cluster_offsets[cluster_mid] <= sorted_i)
            {
                cluster_lo = cluster_mid;
            }
            else
            {
                cluster_hi = cluster_mid;
            }
        }
        const VECTOR center = cluster_centers[cluster_lo];
        const int atom_i = permutation[sorted_i];
        const VECTOR_LJ atom = src[atom_i];
        const VECTOR shifted_crd =
            center + Get_Periodic_Displacement(atom.crd, center, cell, rcell);
        sorted_atom_ids[sorted_i] = atom_i;
        sorted_xq[sorted_i] = {shifted_crd.x, shifted_crd.y, shifted_crd.z,
                               atom.charge};
        sorted_lj_type[sorted_i] = atom.LJ_type;
    }
}

static __global__ void Refresh_Current_Cluster_Centers_From_Crd(
    const int cluster_numbers, const int* permutation, const int* cluster_offsets,
    const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell,
    VECTOR* cluster_centers)
{
    SIMPLE_DEVICE_FOR(cluster_i, cluster_numbers)
    {
        const int start = cluster_offsets[cluster_i];
        const int end = cluster_offsets[cluster_i + 1];
        const int count = end > start ? end - start : 0;
        VECTOR center = {0.0f, 0.0f, 0.0f};
        if (count > 0)
        {
            const VECTOR anchor = crd[permutation[start]];
            for (int atom_offset = start; atom_offset < end; atom_offset += 1)
            {
                const VECTOR pos = crd[permutation[atom_offset]];
                center = center + (anchor + Get_Periodic_Displacement(
                                                pos, anchor, cell, rcell));
            }
            center = (1.0f / static_cast<float>(count)) * center;
            center = Get_Periodic_Coordinate(center, cell, rcell);
        }
        cluster_centers[cluster_i] = center;
    }
}

static __global__ void Gather_Sorted_LJ_Direct_Scratch_From_Plain(
    const int atom_numbers, const int cluster_numbers, const int* permutation,
    const int* cluster_offsets, const VECTOR* cluster_centers,
    const LTMatrix3 cell, const LTMatrix3 rcell, const VECTOR* crd,
    const float* charge, const VECTOR_LJ* lj_type_src, int* sorted_atom_ids,
    float4* sorted_xq, int* sorted_lj_type)
{
    SIMPLE_DEVICE_FOR(sorted_i, atom_numbers)
    {
        int cluster_lo = 0;
        int cluster_hi = cluster_numbers;
        while (cluster_lo + 1 < cluster_hi)
        {
            const int cluster_mid = (cluster_lo + cluster_hi) >> 1;
            if (cluster_offsets[cluster_mid] <= sorted_i)
            {
                cluster_lo = cluster_mid;
            }
            else
            {
                cluster_hi = cluster_mid;
            }
        }
        const VECTOR center = cluster_centers[cluster_lo];
        const int atom_i = permutation[sorted_i];
        const VECTOR atom_crd = crd[atom_i];
        const VECTOR shifted_crd =
            center + Get_Periodic_Displacement(atom_crd, center, cell, rcell);
        sorted_atom_ids[sorted_i] = atom_i;
        sorted_xq[sorted_i] = {shifted_crd.x, shifted_crd.y, shifted_crd.z,
                               charge[atom_i]};
        sorted_lj_type[sorted_i] = lj_type_src[atom_i].LJ_type;
    }
}

static __host__ __device__ __forceinline__ VECTOR
Wrap_Clustered_Center_Fractional(const VECTOR center, const LTMatrix3 rcell)
{
    VECTOR frac = center * rcell;
    frac.x -= floorf(frac.x);
    frac.y -= floorf(frac.y);
    frac.z -= floorf(frac.z);
    return frac;
}

static __host__ __device__ __forceinline__ int Encode_Clustered_Pair_Shift_Id(
    int sx, int sy, int sz)
{
    sx = sx < -1 ? -1 : (sx > 1 ? 1 : sx);
    sy = sy < -1 ? -1 : (sy > 1 ? 1 : sy);
    sz = sz < -1 ? -1 : (sz > 1 ? 1 : sz);
    return (sx + 1) * 9 + (sy + 1) * 3 + (sz + 1);
}

static __host__ __device__ __forceinline__ int
Determine_Clustered_Pair_Shift_Id(const VECTOR center_i, const VECTOR center_j,
                                  const LTMatrix3 rcell)
{
    const VECTOR frac_i = Wrap_Clustered_Center_Fractional(center_i, rcell);
    const VECTOR frac_j = Wrap_Clustered_Center_Fractional(center_j, rcell);
    const VECTOR dfrac = frac_j - frac_i;
    return Encode_Clustered_Pair_Shift_Id(
        static_cast<int>(floorf(dfrac.x + 0.5f)),
        static_cast<int>(floorf(dfrac.y + 0.5f)),
        static_cast<int>(floorf(dfrac.z + 0.5f)));
}

static __global__ void Refresh_Nbnxm_Pair_Shift_Bits(
    const int sci_numbers, const int* super_cluster_offsets,
    const VECTOR* cluster_centers, const LJ_CLUSTERED_SCI* nbnxm_sci,
    const LJ_CLUSTERED_CJ_PACKED* nbnxm_cjpacked, const LTMatrix3 rcell,
    uint64_t* pair_shift_bits)
{
    const int sci = blockIdx.x;
    if (sci >= sci_numbers)
    {
        return;
    }

    const LJ_CLUSTERED_SCI sci_entry = nbnxm_sci[sci];
    const int cluster_i_start =
        super_cluster_offsets[sci_entry.supercluster_id];
    const int cluster_i_end =
        super_cluster_offsets[sci_entry.supercluster_id + 1];
    const int active_cluster_count = cluster_i_end - cluster_i_start;
    const int packed_count = sci_entry.cjpacked_end - sci_entry.cjpacked_begin;
    const int total_records = packed_count * kClusteredJGroupSize;

    for (int record = threadIdx.x; record < total_records; record += blockDim.x)
    {
        const int local_packed = record / kClusteredJGroupSize;
        const int jm = record % kClusteredJGroupSize;
        const int packed_idx = sci_entry.cjpacked_begin + local_packed;
        const LJ_CLUSTERED_CJ_PACKED packed = nbnxm_cjpacked[packed_idx];
        const int cluster_j = packed.cj[jm];

        uint64_t shift_bits = 0ull;
        if (cluster_j >= 0)
        {
            const unsigned int combined_imask =
                Clustered_Jm_Imask(packed.imei[0], jm) |
                Clustered_Jm_Imask(packed.imei[1], jm);
            const VECTOR center_j = cluster_centers[cluster_j];
            for (int i_local = 0; i_local < active_cluster_count; i_local += 1)
            {
                int shift_id = kClusteredCentralShiftId;
                if ((combined_imask & (1u << static_cast<unsigned int>(i_local))) !=
                    0u)
                {
                    shift_id = Determine_Clustered_Pair_Shift_Id(
                        cluster_centers[cluster_i_start + i_local], center_j,
                        rcell);
                }
                Clustered_Set_Pair_Shift_Id(&shift_bits, i_local, shift_id);
            }
        }
        pair_shift_bits[packed_idx * kClusteredJGroupSize + jm] = shift_bits;
    }
}

static __global__ void Refresh_Gmxpacked_Pair_Shift_Bits(
    const int sci_numbers, const int* super_cluster_offsets,
    const VECTOR* cluster_centers, const LJ_CLUSTERED_GMXPACKED_SCI* gmxpacked_sci,
    const LJ_CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked,
    const LTMatrix3 rcell, uint64_t* pair_shift_bits)
{
    const int sci = blockIdx.x;
    if (sci >= sci_numbers)
    {
        return;
    }

    const LJ_CLUSTERED_GMXPACKED_SCI sci_entry = gmxpacked_sci[sci];
    const int cluster_i_start =
        super_cluster_offsets[sci_entry.supercluster_id];
    const int cluster_i_end =
        super_cluster_offsets[sci_entry.supercluster_id + 1];
    const int active_cluster_count = cluster_i_end - cluster_i_start;
    const int packed_count = sci_entry.cjpacked_end - sci_entry.cjpacked_begin;
    const int total_records = packed_count * kClusteredJGroupSize;

    for (int record = threadIdx.x; record < total_records; record += blockDim.x)
    {
        const int local_packed = record / kClusteredJGroupSize;
        const int jm = record % kClusteredJGroupSize;
        const int packed_idx = sci_entry.cjpacked_begin + local_packed;
        const LJ_CLUSTERED_GMXPACKED_CJ packed = gmxpacked_cjpacked[packed_idx];
        const int cluster_j = packed.cj[jm];

        uint64_t shift_bits = 0ull;
        if (cluster_j >= 0)
        {
            const unsigned int combined_imask =
                ((packed.split[0].imask | packed.split[1].imask) >>
                 Clustered_Jm_Imask_Shift(jm)) &
                ((1u << kClusteredSuperClusterClusters) - 1u);
            const VECTOR center_j = cluster_centers[cluster_j];
            for (int i_local = 0; i_local < active_cluster_count; i_local += 1)
            {
                int shift_id = kClusteredCentralShiftId;
                if ((combined_imask &
                     (1u << static_cast<unsigned int>(i_local))) != 0u)
                {
                    shift_id = Determine_Clustered_Pair_Shift_Id(
                        cluster_centers[cluster_i_start + i_local], center_j,
                        rcell);
                }
                Clustered_Set_Pair_Shift_Id(&shift_bits, i_local, shift_id);
            }
        }
        pair_shift_bits[packed_idx * kClusteredJGroupSize + jm] = shift_bits;
    }
}

static __global__ void Snapshot_Clustered_Outer_Imask(
    const int cjpacked_numbers, const LJ_CLUSTERED_CJ_PACKED* nbnxm_cjpacked,
    unsigned int* outer_imask)
{
    SIMPLE_DEVICE_FOR(split_idx, cjpacked_numbers * kClusteredWarpSplitCount)
    {
        const int packed_idx = split_idx / kClusteredWarpSplitCount;
        const int split = split_idx % kClusteredWarpSplitCount;
        outer_imask[split_idx] = nbnxm_cjpacked[packed_idx].imei[split].imask;
    }
}

static __global__ void Prune_Clustered_Inner_Imask(
    const int sci_numbers, const float cutoff_sq, const LTMatrix3 cell,
    const LTMatrix3 rcell, const int* super_cluster_offsets,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const LJ_CLUSTERED_SCI* nbnxm_sci, const unsigned int* outer_imask,
    const float4* sorted_xq, LJ_CLUSTERED_CJ_PACKED* nbnxm_cjpacked)
{
    const int sci = blockIdx.x;
    if (sci >= sci_numbers)
    {
        return;
    }

    const LJ_CLUSTERED_SCI sci_entry = nbnxm_sci[sci];
    const int cluster_i_start =
        super_cluster_offsets[sci_entry.supercluster_id];
    const int cluster_i_end =
        super_cluster_offsets[sci_entry.supercluster_id + 1];
    const int active_cluster_count = cluster_i_end - cluster_i_start;
    const int packed_count = sci_entry.cjpacked_end - sci_entry.cjpacked_begin;
    const int total_split_entries = packed_count * kClusteredWarpSplitCount;

    __shared__ unsigned int
        shared_i_local_masks[kClusteredMaxSuperClusterClusters];
    __shared__ float shared_i_center_x[kClusteredMaxSuperClusterClusters];
    __shared__ float shared_i_center_y[kClusteredMaxSuperClusterClusters];
    __shared__ float shared_i_center_z[kClusteredMaxSuperClusterClusters];
    __shared__ float4
        shared_i_xq[kClusteredMaxSuperClusterClusters * kClusteredClusterSize];

    if (threadIdx.x < active_cluster_count)
    {
        const int cluster_i = cluster_i_start + threadIdx.x;
        const VECTOR center_i = cluster_centers[cluster_i];
        shared_i_local_masks[threadIdx.x] = cluster_local_masks[cluster_i];
        shared_i_center_x[threadIdx.x] = center_i.x;
        shared_i_center_y[threadIdx.x] = center_i.y;
        shared_i_center_z[threadIdx.x] = center_i.z;
    }
    __syncthreads();
    for (int atom_slot = threadIdx.x;
         atom_slot < active_cluster_count * kClusteredClusterSize;
         atom_slot += blockDim.x)
    {
        const int i_local = atom_slot / kClusteredClusterSize;
        const int i_lane = atom_slot % kClusteredClusterSize;
        const int cluster_i = cluster_i_start + i_local;
        if ((shared_i_local_masks[i_local] &
             (1u << static_cast<unsigned int>(i_lane))) != 0u)
        {
            shared_i_xq[atom_slot] =
                sorted_xq[cluster_offsets[cluster_i] + i_lane];
        }
        else
        {
            shared_i_xq[atom_slot] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
    __syncthreads();

    for (int split_entry = threadIdx.x; split_entry < total_split_entries;
         split_entry += blockDim.x)
    {
        const int local_packed = split_entry / kClusteredWarpSplitCount;
        const int split = split_entry % kClusteredWarpSplitCount;
        const int packed_idx = sci_entry.cjpacked_begin + local_packed;
        const LJ_CLUSTERED_CJ_PACKED packed = nbnxm_cjpacked[packed_idx];
        const unsigned int outer_packed_imask =
            outer_imask[packed_idx * kClusteredWarpSplitCount + split];
        const int warp_j_base = split * kClusteredSplitJClusterSize;
        unsigned int inner_packed_imask = 0u;

        for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
        {
            const int cluster_j = packed.cj[jm];
            const unsigned int outer_jm_imask =
                (outer_packed_imask >> Clustered_Jm_Imask_Shift(jm)) &
                ((1u << kClusteredSuperClusterClusters) - 1u);
            if (cluster_j < 0 || outer_jm_imask == 0u)
            {
                continue;
            }

            const unsigned int valid_mask_j =
                (cluster_valid_masks[cluster_j] >> warp_j_base) &
                ((1u << kClusteredSplitJClusterSize) - 1u);
            if (valid_mask_j == 0u)
            {
                continue;
            }

            const int sorted_j_base = cluster_offsets[cluster_j] + warp_j_base;
            const VECTOR center_j = cluster_centers[cluster_j];
            float4 cached_j_xq[kClusteredSplitJClusterSize];
#pragma unroll
            for (int warp_j_local = 0; warp_j_local < kClusteredSplitJClusterSize;
                 warp_j_local += 1)
            {
                if ((valid_mask_j &
                     (1u << static_cast<unsigned int>(warp_j_local))) != 0u)
                {
                    cached_j_xq[warp_j_local] =
                        sorted_xq[sorted_j_base + warp_j_local];
                }
                else
                {
                    cached_j_xq[warp_j_local] = {0.0f, 0.0f, 0.0f, 0.0f};
                }
            }

            unsigned int inner_jm_imask = 0u;
            for (int i_local = 0; i_local < active_cluster_count; i_local += 1)
            {
                if ((outer_jm_imask & (1u << static_cast<unsigned int>(i_local))) ==
                    0u)
                {
                    continue;
                }

                const unsigned int local_mask_i = shared_i_local_masks[i_local];
                if (local_mask_i == 0u)
                {
                    continue;
                }

                const VECTOR center_i = {shared_i_center_x[i_local],
                                         shared_i_center_y[i_local],
                                         shared_i_center_z[i_local]};
                const VECTOR pair_shift = Clustered_Shift_Vector_From_Id(
                    Determine_Clustered_Pair_Shift_Id(center_i, center_j, rcell),
                    cell);

                bool any_in_range = false;
                for (int i_lane = 0; i_lane < kClusteredClusterSize; i_lane += 1)
                {
                    if ((local_mask_i & (1u << i_lane)) == 0u)
                    {
                        continue;
                    }
                    const float4 r1_xq =
                        shared_i_xq[i_local * kClusteredClusterSize + i_lane];
                    for (int warp_j_local = 0;
                         warp_j_local < kClusteredSplitJClusterSize;
                         warp_j_local += 1)
                    {
                        if ((valid_mask_j &
                             (1u << static_cast<unsigned int>(warp_j_local))) == 0u)
                        {
                            continue;
                        }
                        const float4 r2_xq = cached_j_xq[warp_j_local];
                        const float dr_x = r2_xq.x - r1_xq.x - pair_shift.x;
                        const float dr_y = r2_xq.y - r1_xq.y - pair_shift.y;
                        const float dr_z = r2_xq.z - r1_xq.z - pair_shift.z;
                        const float dr2 = dr_x * dr_x + dr_y * dr_y +
                                          dr_z * dr_z;
                        if (dr2 < cutoff_sq && dr2 != 0.0f)
                        {
                            any_in_range = true;
                            break;
                        }
                    }
                    if (any_in_range)
                    {
                        break;
                    }
                }

                if (any_in_range)
                {
                    inner_jm_imask |=
                        (1u << static_cast<unsigned int>(i_local));
                }
            }

            inner_packed_imask |=
                inner_jm_imask << Clustered_Jm_Imask_Shift(jm);
        }

        nbnxm_cjpacked[packed_idx].imei[split].imask = inner_packed_imask;
    }
}

static __host__ __device__ __forceinline__ int
Clustered_Warp_Record_Global_Index(int packed_idx, int warp_id, int jm)
{
    return (packed_idx * kClusteredWarpSplitCount + warp_id) *
               kClusteredJGroupSize +
           jm;
}

static __global__ void Build_Nbnxm_Warp_J_Records(
    const int sci_numbers, const int* super_cluster_offsets,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const LJ_CLUSTERED_SCI* nbnxm_sci,
    const LJ_CLUSTERED_CJ_PACKED* nbnxm_cjpacked,
    const unsigned long long* exclusion_mask_pool,
    LJ_CLUSTERED_WARP_J_RECORD* warp_records)
{
    const int sci = blockIdx.x;
    if (sci >= sci_numbers)
    {
        return;
    }

    const LJ_CLUSTERED_SCI sci_entry = nbnxm_sci[sci];
    const int cluster_i_start =
        super_cluster_offsets[sci_entry.supercluster_id];
    const int cluster_i_end =
        super_cluster_offsets[sci_entry.supercluster_id + 1];
    const int active_cluster_count = cluster_i_end - cluster_i_start;
    const int packed_count = sci_entry.cjpacked_end - sci_entry.cjpacked_begin;
    const int records_per_packed =
        kClusteredWarpSplitCount * kClusteredJGroupSize;
    const int total_records = packed_count * records_per_packed;

    for (int record = threadIdx.x; record < total_records; record += blockDim.x)
    {
        const int local_packed = record / records_per_packed;
        const int record_in_packed = record % records_per_packed;
        const int warp_id = record_in_packed / kClusteredJGroupSize;
        const int jm = record_in_packed % kClusteredJGroupSize;
        const int packed_idx = sci_entry.cjpacked_begin + local_packed;
        const LJ_CLUSTERED_CJ_PACKED packed = nbnxm_cjpacked[packed_idx];
        const LJ_CLUSTERED_IMEI imei = packed.imei[warp_id];
        const int cluster_j = packed.cj[jm];
        LJ_CLUSTERED_WARP_J_RECORD record_out = {};
        record_out.cluster_j = cluster_j;
        if (cluster_j >= 0)
        {
            const int warp_j_base = warp_id * kClusteredSplitJClusterSize;
            const unsigned int valid_mask =
                (cluster_valid_masks[cluster_j] >> warp_j_base) &
                ((1u << kClusteredSplitJClusterSize) - 1u);
            const unsigned int local_mask =
                (cluster_local_masks[cluster_j] >> warp_j_base) &
                ((1u << kClusteredSplitJClusterSize) - 1u);
            const unsigned int imask = Clustered_Jm_Imask(imei, jm);
            record_out.sorted_j_base = cluster_offsets[cluster_j] + warp_j_base;
            record_out.pair_shift_index =
                packed_idx * kClusteredJGroupSize + jm;
            record_out.valid_mask = static_cast<unsigned char>(valid_mask);
            record_out.imask = static_cast<unsigned char>(imask);
            record_out.local_mask = static_cast<unsigned char>(local_mask);
            record_out.j_lane_base = static_cast<unsigned char>(warp_j_base);
            for (int i_local = 0; i_local < active_cluster_count; i_local += 1)
            {
                if ((imask & (1u << static_cast<unsigned int>(i_local))) == 0u)
                {
                    continue;
                }
                const int exclusion_index =
                    Clustered_Exclusion_Index(imei, jm, i_local);
                if (exclusion_index < 0)
                {
                    continue;
                }
                const unsigned long long exclusion_mask =
                    exclusion_mask_pool[exclusion_index];
                for (int warp_j_local = 0;
                     warp_j_local < kClusteredSplitJClusterSize;
                     warp_j_local += 1)
                {
                    const int absolute_j_lane = warp_j_base + warp_j_local;
                    for (int i_lane = 0; i_lane < kClusteredClusterSize;
                         i_lane += 1)
                        {
                            if ((exclusion_mask &
                                 (1ull << (i_lane * kClusteredClusterSize +
                                           absolute_j_lane))) != 0ull)
                            {
                                record_out
                                    .pair_excl[warp_j_local *
                                                   kClusteredClusterSize +
                                               i_lane] |=
                                    static_cast<unsigned char>(1u << i_local);
                            }
                        }
                    }
                }
        }
        warp_records[Clustered_Warp_Record_Global_Index(packed_idx, warp_id,
                                                        jm)] = record_out;
    }
}

static __global__ void Count_Active_Nbnxm_Warp_J_Records(
    const int sci_numbers, const LJ_CLUSTERED_SCI* nbnxm_sci,
    const LJ_CLUSTERED_CJ_PACKED* nbnxm_cjpacked, int* record_counts)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const LJ_CLUSTERED_SCI sci_entry = nbnxm_sci[sci];
        int active_records = 0;
        constexpr unsigned int jm_imask_mask =
            (1u << kClusteredSuperClusterClusters) - 1u;
        for (int packed_idx = sci_entry.cjpacked_begin;
             packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
        {
            const LJ_CLUSTERED_CJ_PACKED* packed = nbnxm_cjpacked + packed_idx;
            for (int warp_id = 0; warp_id < kClusteredWarpSplitCount;
                 warp_id += 1)
            {
                const unsigned int packed_imask = packed->imei[warp_id].imask;
                for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
                {
                    if (packed->cj[jm] >= 0 &&
                        ((packed_imask >> Clustered_Jm_Imask_Shift(jm)) &
                         jm_imask_mask) != 0u)
                    {
                        active_records += 1;
                    }
                }
            }
        }
        record_counts[sci] = active_records;
    }
}

static __device__ __forceinline__ bool
Clustered_Forceonly_Record_Sort_Greater(unsigned int lhs_key, int lhs_source,
                                        unsigned int rhs_key, int rhs_source)
{
    return lhs_key > rhs_key ||
           (lhs_key == rhs_key && lhs_source > rhs_source);
}

static __device__ __forceinline__ LJ_CLUSTERED_WARP_J_RECORD
Build_Active_Nbnxm_Warp_J_Record(
    const int packed_idx, const int warp_id, const int jm,
    const int active_cluster_count, const int* cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const LJ_CLUSTERED_CJ_PACKED* nbnxm_cjpacked,
    const unsigned long long* exclusion_mask_pool)
{
    const LJ_CLUSTERED_CJ_PACKED* packed = nbnxm_cjpacked + packed_idx;
    const LJ_CLUSTERED_IMEI* imei = packed->imei + warp_id;
    const int cluster_j = packed->cj[jm];
    LJ_CLUSTERED_WARP_J_RECORD record_out = {};
    record_out.cluster_j = cluster_j;
    if (cluster_j < 0)
    {
        return record_out;
    }

    const int warp_j_base = warp_id * kClusteredSplitJClusterSize;
    const unsigned int imask = Clustered_Jm_Imask(*imei, jm);
    record_out.imask = static_cast<unsigned char>(imask);
    if (imask == 0u)
    {
        return record_out;
    }

    const unsigned int valid_mask =
        (cluster_valid_masks[cluster_j] >> warp_j_base) &
        ((1u << kClusteredSplitJClusterSize) - 1u);
    const unsigned int local_mask =
        (cluster_local_masks[cluster_j] >> warp_j_base) &
        ((1u << kClusteredSplitJClusterSize) - 1u);
    record_out.sorted_j_base = cluster_offsets[cluster_j] + warp_j_base;
    record_out.pair_shift_index = packed_idx * kClusteredJGroupSize + jm;
    record_out.valid_mask = static_cast<unsigned char>(valid_mask);
    record_out.local_mask = static_cast<unsigned char>(local_mask);
    record_out.j_lane_base = static_cast<unsigned char>(warp_j_base);
    for (int i_local = 0; i_local < active_cluster_count; i_local += 1)
    {
        if ((imask & (1u << static_cast<unsigned int>(i_local))) == 0u)
        {
            continue;
        }
        const int exclusion_index = Clustered_Exclusion_Index(*imei, jm, i_local);
        if (exclusion_index < 0)
        {
            continue;
        }
        const unsigned long long exclusion_mask =
            exclusion_mask_pool[exclusion_index];
        for (int warp_j_local = 0;
             warp_j_local < kClusteredSplitJClusterSize; warp_j_local += 1)
        {
            const int absolute_j_lane = warp_j_base + warp_j_local;
            for (int i_lane = 0; i_lane < kClusteredClusterSize; i_lane += 1)
            {
                if ((exclusion_mask &
                     (1ull << (i_lane * kClusteredClusterSize +
                               absolute_j_lane))) != 0ull)
                {
                    record_out
                        .pair_excl[warp_j_local * kClusteredClusterSize +
                                   i_lane] |=
                        static_cast<unsigned char>(1u << i_local);
                }
            }
        }
    }
    return record_out;
}

static __global__ void Fill_Active_Nbnxm_Warp_J_Records(
    const int sci_numbers, const int* super_cluster_offsets,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const LJ_CLUSTERED_SCI* nbnxm_sci,
    const LJ_CLUSTERED_CJ_PACKED* nbnxm_cjpacked,
    const unsigned long long* exclusion_mask_pool,
    const int* record_offsets,
    LJ_CLUSTERED_WARP_J_RECORD* compact_records)
{
    const int sci = blockIdx.x;
    if (sci >= sci_numbers)
    {
        return;
    }

    const LJ_CLUSTERED_SCI sci_entry = nbnxm_sci[sci];
    const int cluster_i_start =
        super_cluster_offsets[sci_entry.supercluster_id];
    const int cluster_i_end =
        super_cluster_offsets[sci_entry.supercluster_id + 1];
    const int active_cluster_count = cluster_i_end - cluster_i_start;
    const int packed_count = sci_entry.cjpacked_end - sci_entry.cjpacked_begin;
    const int records_per_packed =
        kClusteredWarpSplitCount * kClusteredJGroupSize;
    const int total_records = packed_count * records_per_packed;
    constexpr int subgroup_records = kClusteredWarpSplitCount * kClusteredJGroupSize;
    static_assert(kClusteredBuilderBlockSize % subgroup_records == 0,
                  "Clustered builder block size must be a multiple of packed record span.");
    constexpr int subgroup_count = kClusteredBuilderBlockSize / subgroup_records;
    __shared__ LJ_CLUSTERED_CJ_PACKED shared_packed[subgroup_count];
    __shared__ int shared_output_base;
    if (threadIdx.x == 0)
    {
        shared_output_base = record_offsets[sci];
    }
    __syncthreads();

    const int subgroup_id = threadIdx.x / subgroup_records;
    const int subgroup_lane = threadIdx.x % subgroup_records;
    for (int record_base = 0; record_base < total_records;
         record_base += blockDim.x)
    {
        const int subgroup_record = record_base + subgroup_id * subgroup_records;
        if (subgroup_lane == 0 && subgroup_record < total_records)
        {
            const int subgroup_local_packed = subgroup_record / records_per_packed;
            shared_packed[subgroup_id] =
                nbnxm_cjpacked[sci_entry.cjpacked_begin + subgroup_local_packed];
        }
        __syncwarp();

        const int record = record_base + threadIdx.x;
        if (record >= total_records)
        {
            continue;
        }

        const int record_in_packed = subgroup_lane;
        const int warp_id = record_in_packed / kClusteredJGroupSize;
        const int jm = record_in_packed % kClusteredJGroupSize;
        const int packed_idx = sci_entry.cjpacked_begin + record / records_per_packed;
        const LJ_CLUSTERED_CJ_PACKED& packed = shared_packed[subgroup_id];
        const LJ_CLUSTERED_IMEI* imei = packed.imei + warp_id;
        const int cluster_j = packed.cj[jm];
        LJ_CLUSTERED_WARP_J_RECORD record_out = {};
        record_out.cluster_j = cluster_j;
        if (cluster_j >= 0)
        {
            const int warp_j_base = warp_id * kClusteredSplitJClusterSize;
            const unsigned int imask = Clustered_Jm_Imask(*imei, jm);
            record_out.imask = static_cast<unsigned char>(imask);
            if (imask != 0u)
            {
                const unsigned int valid_mask =
                    (cluster_valid_masks[cluster_j] >> warp_j_base) &
                    ((1u << kClusteredSplitJClusterSize) - 1u);
                const unsigned int local_mask =
                    (cluster_local_masks[cluster_j] >> warp_j_base) &
                    ((1u << kClusteredSplitJClusterSize) - 1u);
                record_out.sorted_j_base =
                    cluster_offsets[cluster_j] + warp_j_base;
                record_out.pair_shift_index =
                    packed_idx * kClusteredJGroupSize + jm;
                record_out.valid_mask = static_cast<unsigned char>(valid_mask);
                record_out.local_mask = static_cast<unsigned char>(local_mask);
                record_out.j_lane_base =
                    static_cast<unsigned char>(warp_j_base);
                for (int i_local = 0; i_local < active_cluster_count;
                     i_local += 1)
                {
                    if ((imask & (1u << static_cast<unsigned int>(i_local))) ==
                        0u)
                    {
                        continue;
                    }
                    const int exclusion_index =
                        Clustered_Exclusion_Index(*imei, jm, i_local);
                    if (exclusion_index < 0)
                    {
                        continue;
                    }
                    const unsigned long long exclusion_mask =
                        exclusion_mask_pool[exclusion_index];
                    for (int warp_j_local = 0;
                         warp_j_local < kClusteredSplitJClusterSize;
                         warp_j_local += 1)
                    {
                        const int absolute_j_lane = warp_j_base + warp_j_local;
                        for (int i_lane = 0; i_lane < kClusteredClusterSize;
                             i_lane += 1)
                        {
                            if ((exclusion_mask &
                                 (1ull << (i_lane * kClusteredClusterSize +
                                           absolute_j_lane))) != 0ull)
                            {
                                record_out
                                    .pair_excl[warp_j_local *
                                                   kClusteredClusterSize +
                                               i_lane] |=
                                    static_cast<unsigned char>(1u << i_local);
                            }
                        }
                    }
                }
            }
        }
        const int active_record =
            (record_out.cluster_j >= 0 && record_out.imask != 0u) ? 1 : 0;
        const unsigned int lane =
            static_cast<unsigned int>(threadIdx.x) &
            static_cast<unsigned int>(warpSize - 1);
        const unsigned int active_mask =
            __ballot_sync(FULL_MASK, active_record != 0);
        if (active_mask != 0u)
        {
            int warp_write_base = 0;
            if (lane == 0)
            {
                warp_write_base =
                    atomicAdd(&shared_output_base, __popc(active_mask));
            }
            warp_write_base = deviceShfl(FULL_MASK, warp_write_base, 0,
                                         warpSize);
            if (active_record != 0)
            {
                const unsigned int lane_prefix_mask =
                    lane == 0 ? 0u : ((1u << lane) - 1u);
                const int write_index_in_warp =
                    __popc(active_mask & lane_prefix_mask);
                compact_records[warp_write_base + write_index_in_warp] =
                    record_out;
            }
        }
    }
}

static __global__ void Fill_Sorted_Active_Nbnxm_Warp_J_Records(
    const int sci_numbers, const int* super_cluster_offsets,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const LJ_CLUSTERED_SCI* nbnxm_sci,
    const LJ_CLUSTERED_CJ_PACKED* nbnxm_cjpacked,
    const unsigned long long* exclusion_mask_pool,
    const int* record_offsets,
    LJ_CLUSTERED_WARP_J_RECORD* compact_records)
{
    const int sci = blockIdx.x;
    if (sci >= sci_numbers)
    {
        return;
    }

    const LJ_CLUSTERED_SCI sci_entry = nbnxm_sci[sci];
    const int cluster_i_start =
        super_cluster_offsets[sci_entry.supercluster_id];
    const int cluster_i_end =
        super_cluster_offsets[sci_entry.supercluster_id + 1];
    const int active_cluster_count = cluster_i_end - cluster_i_start;
    const int packed_count = sci_entry.cjpacked_end - sci_entry.cjpacked_begin;
    const int records_per_packed =
        kClusteredWarpSplitCount * kClusteredJGroupSize;
    const int total_records = packed_count * records_per_packed;
    const int output_begin = record_offsets[sci];
    const int active_record_count = record_offsets[sci + 1] - output_begin;
    if (active_record_count <= 0)
    {
        return;
    }

    __shared__ unsigned int
        shared_sort_keys[kClusteredForceonlyLocalSortCapacity];
    __shared__ int shared_sort_sources[kClusteredForceonlyLocalSortCapacity];
    __shared__ int shared_active_count;
    __shared__ int shared_output_base;

    if (active_record_count > kClusteredForceonlyLocalSortCapacity)
    {
        if (threadIdx.x == 0)
        {
            shared_output_base = output_begin;
        }
        __syncthreads();
        for (int record = threadIdx.x; record < total_records;
             record += blockDim.x)
        {
            const int packed_idx =
                sci_entry.cjpacked_begin + record / records_per_packed;
            const int record_in_packed = record % records_per_packed;
            const int warp_id = record_in_packed / kClusteredJGroupSize;
            const int jm = record_in_packed % kClusteredJGroupSize;
            const LJ_CLUSTERED_WARP_J_RECORD record_out =
                Build_Active_Nbnxm_Warp_J_Record(
                    packed_idx, warp_id, jm, active_cluster_count,
                    cluster_offsets, cluster_valid_masks, cluster_local_masks,
                    nbnxm_cjpacked, exclusion_mask_pool);
            const int active_record =
                (record_out.cluster_j >= 0 && record_out.imask != 0u) ? 1 : 0;
            const unsigned int lane =
                static_cast<unsigned int>(threadIdx.x) &
                static_cast<unsigned int>(warpSize - 1);
            const unsigned int active_mask =
                __ballot_sync(FULL_MASK, active_record != 0);
            if (active_mask == 0u)
            {
                continue;
            }
            int warp_write_base = 0;
            if (lane == 0)
            {
                warp_write_base =
                    atomicAdd(&shared_output_base, __popc(active_mask));
            }
            warp_write_base =
                deviceShfl(FULL_MASK, warp_write_base, 0, warpSize);
            if (active_record != 0)
            {
                const unsigned int lane_prefix_mask =
                    lane == 0 ? 0u : ((1u << lane) - 1u);
                const int write_index_in_warp =
                    __popc(active_mask & lane_prefix_mask);
                compact_records[warp_write_base + write_index_in_warp] =
                    record_out;
            }
        }
        return;
    }

    if (threadIdx.x == 0)
    {
        shared_active_count = 0;
    }
    __syncthreads();

    const bool sort_by_cluster_j =
        sci_entry.shift_id == kClusteredCentralShiftId;
    for (int record = threadIdx.x; record < total_records; record += blockDim.x)
    {
        const int packed_idx =
            sci_entry.cjpacked_begin + record / records_per_packed;
        const int record_in_packed = record % records_per_packed;
        const int warp_id = record_in_packed / kClusteredJGroupSize;
        const int jm = record_in_packed % kClusteredJGroupSize;
        const LJ_CLUSTERED_CJ_PACKED* packed = nbnxm_cjpacked + packed_idx;
        const LJ_CLUSTERED_IMEI* imei = packed->imei + warp_id;
        const int cluster_j = packed->cj[jm];
        if (cluster_j < 0)
        {
            continue;
        }
        const unsigned int imask = Clustered_Jm_Imask(*imei, jm);
        if (imask == 0u)
        {
            continue;
        }
        const int write_idx = atomicAdd(&shared_active_count, 1);
        if (write_idx < kClusteredForceonlyLocalSortCapacity)
        {
            shared_sort_keys[write_idx] =
                sort_by_cluster_j
                    ? ((static_cast<unsigned int>(cluster_j) << 1) |
                       static_cast<unsigned int>(warp_id))
                    : static_cast<unsigned int>(record);
            shared_sort_sources[write_idx] = record;
        }
    }
    __syncthreads();

    const int active_count = shared_active_count;
    int sort_count = 1;
    while (sort_count < active_count)
    {
        sort_count <<= 1;
    }
    for (int idx = active_count + threadIdx.x; idx < sort_count;
         idx += blockDim.x)
    {
        shared_sort_keys[idx] = kClusteredForceonlyInvalidSortKey;
        shared_sort_sources[idx] = 0x7fffffff;
    }
    __syncthreads();

    for (int width = 2; width <= sort_count; width <<= 1)
    {
        for (int stride = width >> 1; stride > 0; stride >>= 1)
        {
            for (int idx = threadIdx.x; idx < sort_count; idx += blockDim.x)
            {
                const int other = idx ^ stride;
                if (other <= idx)
                {
                    continue;
                }
                const bool ascending = (idx & width) == 0;
                const unsigned int key_a = shared_sort_keys[idx];
                const unsigned int key_b = shared_sort_keys[other];
                const int source_a = shared_sort_sources[idx];
                const int source_b = shared_sort_sources[other];
                const bool should_swap =
                    ascending
                        ? Clustered_Forceonly_Record_Sort_Greater(
                              key_a, source_a, key_b, source_b)
                        : Clustered_Forceonly_Record_Sort_Greater(
                              key_b, source_b, key_a, source_a);
                if (should_swap)
                {
                    shared_sort_keys[idx] = key_b;
                    shared_sort_keys[other] = key_a;
                    shared_sort_sources[idx] = source_b;
                    shared_sort_sources[other] = source_a;
                }
            }
            __syncthreads();
        }
    }

    for (int sorted_idx = threadIdx.x; sorted_idx < active_count;
         sorted_idx += blockDim.x)
    {
        const int record = shared_sort_sources[sorted_idx];
        const int packed_idx =
            sci_entry.cjpacked_begin + record / records_per_packed;
        const int record_in_packed = record % records_per_packed;
        const int warp_id = record_in_packed / kClusteredJGroupSize;
        const int jm = record_in_packed % kClusteredJGroupSize;
        compact_records[output_begin + sorted_idx] =
            Build_Active_Nbnxm_Warp_J_Record(
                packed_idx, warp_id, jm, active_cluster_count, cluster_offsets,
                cluster_valid_masks, cluster_local_masks, nbnxm_cjpacked,
                exclusion_mask_pool);
    }
}

template <typename SrcType>
static __global__ void Refresh_Current_Cluster_Centers(
    const int cluster_numbers, const int* permutation, const int* cluster_offsets,
    const SrcType* src, const LTMatrix3 cell, const LTMatrix3 rcell,
    VECTOR* cluster_centers)
{
    SIMPLE_DEVICE_FOR(cluster_i, cluster_numbers)
    {
        const int start = cluster_offsets[cluster_i];
        const int end = cluster_offsets[cluster_i + 1];
        const int count = end > start ? end - start : 0;
        VECTOR center = {0.0f, 0.0f, 0.0f};
        if (count > 0)
        {
            const VECTOR anchor = src[permutation[start]].crd;
            for (int atom_offset = start; atom_offset < end; atom_offset += 1)
            {
                const VECTOR pos = src[permutation[atom_offset]].crd;
                center = center + (anchor + Get_Periodic_Displacement(
                                                pos, anchor, cell, rcell));
            }
            center = (1.0f / static_cast<float>(count)) * center;
            center = Get_Periodic_Coordinate(center, cell, rcell);
        }
        cluster_centers[cluster_i] = center;
    }
}

static __global__ void Gather_Sorted_Soft_Core_Scratch(
    const int atom_numbers, const int* permutation,
    const VECTOR_LJ_SOFT_TYPE* src, VECTOR_LJ_SOFT_TYPE* dest)
{
    SIMPLE_DEVICE_FOR(sorted_i, atom_numbers)
    {
        dest[sorted_i] = src[permutation[sorted_i]];
    }
}

static __global__ void Check_Clustered_Finite_Coordinates(
    const int atom_numbers, const VECTOR* crd, int* bad_flag)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        const VECTOR value = crd[atom_i];
        if (!isfinite(value.x) || !isfinite(value.y) || !isfinite(value.z))
        {
            *bad_flag = 1;
        }
    }
}

static __host__ __device__ __forceinline__ int IntMin(int a, int b)
{
    return a < b ? a : b;
}

static __host__ __device__ __forceinline__ int IntMax(int a, int b)
{
    return a > b ? a : b;
}

static __host__ __device__ __forceinline__ VECTOR Wrap_To_Box_Fractional(
    VECTOR crd, LTMatrix3 rcell)
{
    VECTOR frac = crd * rcell;
    frac.x -= floorf(frac.x);
    frac.y -= floorf(frac.y);
    frac.z -= floorf(frac.z);
    return frac;
}

static __host__ __device__ __forceinline__ uint32_t Quantize_Unit_Coordinate(
    float value, int bits)
{
    if (bits <= 0)
    {
        return 0;
    }
    const uint32_t grid = 1u << bits;
    float clamped = fmaxf(0.0f, fminf(0.99999994f, value));
    uint32_t coord = static_cast<uint32_t>(clamped * static_cast<float>(grid));
    if (coord >= grid)
    {
        coord = grid - 1;
    }
    return coord;
}

static __host__ __device__ __forceinline__ VECTOR Wrap_Unit_Coordinate(
    VECTOR frac)
{
    frac.x -= floorf(frac.x);
    frac.y -= floorf(frac.y);
    frac.z -= floorf(frac.z);
    return frac;
}

static __host__ __device__ __forceinline__ VECTOR Periodic_Unit_Displacement(
    VECTOR a, VECTOR b)
{
    VECTOR dr = a - b;
    dr.x -= floorf(dr.x + 0.5f);
    dr.y -= floorf(dr.y + 0.5f);
    dr.z -= floorf(dr.z + 0.5f);
    return dr;
}

static __host__ __device__ __forceinline__ cstone::Vec3<float> To_Cstone_Vec(
    VECTOR v)
{
    return {v.x, v.y, v.z};
}

static __host__ __device__ __forceinline__ VECTOR Fractional_Cutoff_Pad(
    float cutoff, LTMatrix3 rcell)
{
    return {
        cutoff *
            sqrtf(rcell.a11 * rcell.a11 + rcell.a21 * rcell.a21 +
                  rcell.a31 * rcell.a31),
        cutoff * sqrtf(rcell.a22 * rcell.a22 + rcell.a32 * rcell.a32),
        cutoff * fabsf(rcell.a33)};
}

template <typename KeyType>
static __host__ __device__ __forceinline__ bool Cornerstone_Node_Overlaps_Box(
    KeyType prefix, VECTOR target_center, VECTOR target_size,
    bool use_morton_sfc)
{
    const auto unit_box =
        cstone::Box<float>(0.0f, 1.0f, cstone::BoundaryType::periodic);
    const KeyType start_key = cstone::decodePlaceholderBit(prefix);
    const unsigned level = cstone::decodePrefixLength(prefix) / 3;
    const auto node_ibox =
        use_morton_sfc ? cstone::mortonIBox<KeyType>(start_key, level)
                       : cstone::hilbertIBox<KeyType>(start_key, level);
    const auto [node_center, node_size] =
        cstone::centerAndSize<KeyType>(node_ibox, unit_box);
    return cstone::overlap(node_center, node_size, To_Cstone_Vec(target_center),
                           To_Cstone_Vec(target_size), unit_box);
}

template <typename KeyType>
static __host__ __device__ __forceinline__ bool
Cornerstone_Node_Overlaps_Shifted_Box(KeyType prefix, VECTOR target_center,
                                      VECTOR target_size, int shift_id,
                                      bool use_morton_sfc)
{
    const auto unit_box =
        cstone::Box<float>(0.0f, 1.0f, cstone::BoundaryType::open);
    const KeyType start_key = cstone::decodePlaceholderBit(prefix);
    const unsigned level = cstone::decodePrefixLength(prefix) / 3;
    const auto node_ibox =
        use_morton_sfc ? cstone::mortonIBox<KeyType>(start_key, level)
                       : cstone::hilbertIBox<KeyType>(start_key, level);
    const auto [node_center, node_size] =
        cstone::centerAndSize<KeyType>(node_ibox, unit_box);
    const VECTOR shifted_center =
        target_center + Shift_Fractional_From_Id(shift_id);
    return cstone::overlap(node_center, node_size, To_Cstone_Vec(shifted_center),
                           To_Cstone_Vec(target_size), unit_box);
}

static __host__ __device__ __forceinline__ bool Accept_Leaf_Pair(
    int leaf_i, int leaf_j, const int* leaf_has_local)
{
    return !(leaf_j < leaf_i && leaf_has_local[leaf_j] != 0);
}

static __host__ __device__ __forceinline__ bool Exclusion_List_Contains(
    int atom_i, int atom_j, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers)
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

static __host__ __device__ __forceinline__ unsigned long long
Build_Exclusion_Mask(const int* permutation, const int* cluster_offsets,
                     int cluster_i, int cluster_j, unsigned int local_mask_i,
                     unsigned int valid_mask_j, int cluster_size,
                     int local_atom_numbers, const int* excluded_list_start,
                     const int* excluded_list, const int* excluded_numbers,
                     const uint64_t* cluster_molecule_signatures,
                     const int* cluster_molecule_ids)
{
    if (!Clusters_May_Share_Molecule(cluster_i, cluster_j, local_mask_i,
                                     valid_mask_j, cluster_size,
                                     cluster_molecule_signatures,
                                     cluster_molecule_ids))
    {
        return 0ull;
    }

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
            bool excluded = Exclusion_List_Contains(
                atom_i, atom_j, excluded_list_start, excluded_list,
                excluded_numbers);
            if (!excluded && atom_j < local_atom_numbers)
            {
                excluded = Exclusion_List_Contains(
                    atom_j, atom_i, excluded_list_start, excluded_list,
                    excluded_numbers);
            }
            if (excluded)
            {
                mask |= (1ull << (lane_i * cluster_size + lane_j));
            }
        }
    }
    return mask;
}

static __device__ __forceinline__ bool Cached_Clusters_May_Share_Molecule(
    const int* molecule_ids_i, const int* molecule_ids_j,
    uint64_t signature_i, uint64_t signature_j, unsigned int local_mask_i,
    unsigned int valid_mask_j, int cluster_size,
    bool has_molecule_metadata)
{
    if (!has_molecule_metadata)
    {
        return true;
    }
    if ((signature_i & signature_j) == 0ull)
    {
        return false;
    }
    for (int lane_i = 0; lane_i < cluster_size; lane_i += 1)
    {
        if ((local_mask_i & (1u << lane_i)) == 0u)
        {
            continue;
        }
        const int mol_i = molecule_ids_i[lane_i];
        if (mol_i < 0)
        {
            return true;
        }
        for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
        {
            if ((valid_mask_j & (1u << lane_j)) == 0u)
            {
                continue;
            }
            const int mol_j = molecule_ids_j[lane_j];
            if (mol_j < 0)
            {
                return true;
            }
            if (mol_i == mol_j)
            {
                return true;
            }
        }
    }
    return false;
}

static __device__ __forceinline__ unsigned long long
Build_Exclusion_Mask_From_Cached_Atoms(
    const int* atom_ids_i, const int* atom_ids_j, const int* molecule_ids_i,
    const int* molecule_ids_j, uint64_t signature_i, uint64_t signature_j,
    bool has_molecule_metadata, unsigned int local_mask_i,
    unsigned int valid_mask_j, int cluster_size, int local_atom_numbers,
    const int* excluded_list_start, const int* excluded_list,
    const int* excluded_numbers)
{
    if (!Cached_Clusters_May_Share_Molecule(
            molecule_ids_i, molecule_ids_j, signature_i, signature_j,
            local_mask_i, valid_mask_j, cluster_size, has_molecule_metadata))
    {
        return 0ull;
    }

    unsigned long long mask = 0ull;
    for (int lane_i = 0; lane_i < cluster_size; lane_i += 1)
    {
        if ((local_mask_i & (1u << lane_i)) == 0u)
        {
            continue;
        }
        const int atom_i = atom_ids_i[lane_i];
        for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
        {
            if ((valid_mask_j & (1u << lane_j)) == 0u)
            {
                continue;
            }
            const int atom_j = atom_ids_j[lane_j];
            bool excluded = Exclusion_List_Contains(
                atom_i, atom_j, excluded_list_start, excluded_list,
                excluded_numbers);
            if (!excluded && atom_j < local_atom_numbers)
            {
                excluded = Exclusion_List_Contains(
                    atom_j, atom_i, excluded_list_start, excluded_list,
                    excluded_numbers);
            }
            if (excluded)
            {
                mask |= (1ull << (lane_i * cluster_size + lane_j));
            }
        }
    }
    return mask;
}

static __host__ __device__ __forceinline__ bool Cluster_Reach_Overlaps(
    VECTOR center_i, float radius_i, VECTOR center_j, float radius_j,
    float cutoff, LTMatrix3 cell, LTMatrix3 rcell)
{
    const VECTOR dr = Get_Periodic_Displacement(center_j, center_i, cell, rcell);
    const float reach = cutoff + radius_i + radius_j;
    return dr * dr <= reach * reach;
}

static __host__ __device__ __forceinline__ int
Build_CjPacked_Cluster_Metadata(
    const int cluster_i_start, const int cluster_i_end, const int cluster_j,
    const int cluster_size, const int local_atom_numbers, const float cutoff,
    const LTMatrix3 cell, const LTMatrix3 rcell, const int* permutation,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers,
    const unsigned int valid_mask_j, unsigned int* imask,
    unsigned long long* exclusion_masks)
{
    *imask = 0u;
#pragma unroll
    for (int i_local = 0; i_local < kClusteredMaxSuperClusterClusters;
         i_local += 1)
    {
        exclusion_masks[i_local] = 0ull;
    }

    int exclusion_count = 0;
    for (int cluster_i = cluster_i_start; cluster_i < cluster_i_end;
         cluster_i += 1)
    {
        const unsigned int local_mask_i = cluster_local_masks[cluster_i];
        if (local_mask_i == 0u)
        {
            continue;
        }
        if (cluster_j >= cluster_i_start && cluster_j < cluster_i_end &&
            cluster_i > cluster_j)
        {
            continue;
        }
        if (!Cluster_Aabb_Overlaps(cluster_centers[cluster_i],
                                   cluster_extents[cluster_i],
                                   cluster_centers[cluster_j],
                                   cluster_extents[cluster_j], cutoff, cell,
                                   rcell))
        {
            continue;
        }

        const int i_local = cluster_i - cluster_i_start;
        *imask |= (1u << static_cast<unsigned int>(i_local));
        const unsigned long long exclusion_mask = Build_Exclusion_Mask(
            permutation, cluster_offsets, cluster_i, cluster_j, local_mask_i,
            valid_mask_j, cluster_size, local_atom_numbers,
            excluded_list_start, excluded_list, excluded_numbers, NULL, NULL);
        exclusion_masks[i_local] = exclusion_mask;
        exclusion_count += exclusion_mask != 0ull ? 1 : 0;
    }
    return exclusion_count;
}

#ifdef USE_CPU
static int Build_CjPacked_Cluster_Metadata_Shifted(
    const HostClusteredBuildInput& input, const int cluster_i_start,
    const int cluster_i_end, const int cluster_j, const int shift_id,
    const unsigned int valid_mask_j, unsigned int* imask,
    std::array<unsigned long long, kClusteredMaxSuperClusterClusters>*
        exclusion_masks)
{
    *imask = 0u;
    exclusion_masks->fill(0ull);
    const VECTOR shift_vec = Shift_Vector_From_Id(shift_id, input.cell);
    int exclusion_count = 0;

    for (int cluster_i = cluster_i_start; cluster_i < cluster_i_end;
         cluster_i += 1)
    {
        const unsigned int local_mask_i =
            input.cluster_local_masks[(size_t)cluster_i];
        if (local_mask_i == 0u)
        {
            continue;
        }
        if (shift_id == kClusteredCentralShiftId &&
            cluster_j >= cluster_i_start && cluster_j < cluster_i_end &&
            cluster_i > cluster_j)
        {
            continue;
        }
        if (Determine_Cluster_Pair_Shift_Id(
                input.cluster_centers[(size_t)cluster_i],
                input.cluster_centers[(size_t)cluster_j], input.rcell) != shift_id)
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
        const unsigned long long exclusion_mask = Build_Exclusion_Mask(
            input.permutation.data(), input.cluster_offsets.data(), cluster_i,
            cluster_j, local_mask_i, valid_mask_j, input.cluster_size,
            input.local_atom_numbers, input.excluded_list_start.data(),
            input.excluded_list.data(), input.excluded_numbers.data(), NULL,
            NULL);
        (*exclusion_masks)[(size_t)i_local] = exclusion_mask;
        exclusion_count += exclusion_mask != 0ull ? 1 : 0;
    }

    return exclusion_count;
}
#endif

static __host__ __device__ __forceinline__ int
Build_CjPacked_Cluster_Metadata_Shifted_Raw(
    const int cluster_i_start, const int cluster_i_end, const int cluster_j,
    const int shift_id, const int cluster_size, const int local_atom_numbers,
    const float cutoff, const LTMatrix3 cell, const LTMatrix3 rcell,
    const int* permutation, const int* cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_numbers,
    const unsigned int valid_mask_j, unsigned int* imask,
    unsigned long long* exclusion_masks)
{
    *imask = 0u;
#pragma unroll
    for (int i_local = 0; i_local < kClusteredMaxSuperClusterClusters;
         i_local += 1)
    {
        exclusion_masks[i_local] = 0ull;
    }

    const VECTOR shift_vec = Shift_Vector_From_Id(shift_id, cell);
    int exclusion_count = 0;

    for (int cluster_i = cluster_i_start; cluster_i < cluster_i_end;
         cluster_i += 1)
    {
        const unsigned int local_mask_i = cluster_local_masks[cluster_i];
        if (local_mask_i == 0u)
        {
            continue;
        }
        if (shift_id == kClusteredCentralShiftId &&
            cluster_j >= cluster_i_start && cluster_j < cluster_i_end &&
            cluster_i > cluster_j)
        {
            continue;
        }
        if (Determine_Cluster_Pair_Shift_Id(cluster_centers[cluster_i],
                                            cluster_centers[cluster_j],
                                            rcell) != shift_id)
        {
            continue;
        }
        if (!Cluster_Aabb_Overlaps_Shifted(
                cluster_centers[cluster_i], cluster_extents[cluster_i],
                cluster_centers[cluster_j], cluster_extents[cluster_j], cutoff,
                shift_vec))
        {
            continue;
        }

        const int i_local = cluster_i - cluster_i_start;
        *imask |= (1u << static_cast<unsigned int>(i_local));
        const unsigned long long exclusion_mask = Build_Exclusion_Mask(
            permutation, cluster_offsets, cluster_i, cluster_j, local_mask_i,
            valid_mask_j, cluster_size, local_atom_numbers,
            excluded_list_start, excluded_list, excluded_numbers, NULL, NULL);
        exclusion_masks[i_local] = exclusion_mask;
        exclusion_count += exclusion_mask != 0ull ? 1 : 0;
    }

    return exclusion_count;
}

#ifdef USE_CPU
static void Build_Nbnxm_Payload_On_Host(
    const HostClusteredBuildInput& input, std::vector<LJ_CLUSTERED_SCI>* scis,
    std::vector<LJ_CLUSTERED_CJ_PACKED>* cjpacked,
    std::vector<unsigned long long>* exclusion_pool)
{
    scis->clear();
    cjpacked->clear();
    exclusion_pool->clear();
    if (input.candidate_sci_numbers <= 0)
    {
        return;
    }

    std::array<std::vector<HostClusteredJRecord>, kClusteredShiftCount>
        shift_buckets;

    for (int candidate_sci = 0; candidate_sci < input.candidate_sci_numbers;
         candidate_sci += 1)
    {
        for (auto& bucket : shift_buckets)
        {
            bucket.clear();
        }

        const int super_i =
            input.sci_supercluster_ids[(size_t)candidate_sci];
        const VECTOR super_center_frac =
            input.super_cluster_centers[(size_t)super_i];
        const int cluster_i_start =
            input.super_cluster_offsets[(size_t)super_i];
        const int cluster_i_end =
            input.super_cluster_offsets[(size_t)super_i + 1];
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
            const int cluster_j_end =
                input.leaf_cluster_ends[(size_t)leaf_j];
            const int deduped_cluster_j_start =
                std::max(cluster_j_start, processed_cluster_end);
            for (int cluster_j = deduped_cluster_j_start;
                 cluster_j < cluster_j_end;
                 cluster_j += 1)
            {
                const unsigned int valid_mask_j =
                    input.cluster_valid_masks[(size_t)cluster_j];
                if (valid_mask_j == 0u)
                {
                    continue;
                }
                const int super_j =
                    input.cluster_to_supercluster[(size_t)cluster_j];
                if (input.cluster_local_masks[(size_t)cluster_j] != 0u &&
                    super_j < super_i)
                {
                    continue;
                }
                HostClusteredJRecord record = {};
                record.cluster_j = cluster_j;
                for (int shift_id = 0; shift_id < kClusteredShiftCount;
                     shift_id += 1)
                {
                    Build_CjPacked_Cluster_Metadata_Shifted(
                        input, cluster_i_start, cluster_i_end, cluster_j, shift_id,
                        valid_mask_j, &record.imask, &record.exclusion_masks);
                    if (record.imask != 0u)
                    {
                        shift_buckets[(size_t)shift_id].push_back(record);
                    }
                }
            }
            processed_cluster_end = std::max(processed_cluster_end,
                                             cluster_j_end);
        }

        for (int shift_id = 0; shift_id < kClusteredShiftCount; shift_id += 1)
        {
            auto& bucket = shift_buckets[(size_t)shift_id];
            if (bucket.empty())
            {
                continue;
            }
            const int cj_begin = static_cast<int>(cjpacked->size());
            for (size_t bucket_begin = 0; bucket_begin < bucket.size();
                 bucket_begin += kClusteredMaxJGroupSize)
            {
                LJ_CLUSTERED_CJ_PACKED packed = Make_Empty_Clustered_CjPacked();
                for (int jm = 0; jm < kClusteredMaxJGroupSize; jm += 1)
                {
                    const size_t record_index = bucket_begin + (size_t)jm;
                    if (record_index >= bucket.size())
                    {
                        break;
                    }
                    const auto& record = bucket[record_index];
                    packed.cj[jm] = record.cluster_j;
                    const unsigned int valid_mask_j =
                        input.cluster_valid_masks[(size_t)record.cluster_j];
                    int exclusion_indices[kClusteredMaxSuperClusterClusters];
                    for (int i_local = 0;
                         i_local < kClusteredMaxSuperClusterClusters;
                         i_local += 1)
                    {
                        exclusion_indices[i_local] = -1;
                        const unsigned long long exclusion_mask =
                            record.exclusion_masks[(size_t)i_local];
                        if (exclusion_mask == 0ull)
                        {
                            continue;
                        }
                        exclusion_indices[i_local] =
                            static_cast<int>(exclusion_pool->size());
                        exclusion_pool->push_back(exclusion_mask);
                    }
#pragma unroll
                    for (int split = 0; split < kClusteredWarpSplitCount;
                         split += 1)
                    {
                        if (!Clustered_Split_Has_Atoms(valid_mask_j, split))
                        {
                            continue;
                        }
                        packed.imei[split].imask |=
                            record.imask << Clustered_Jm_Imask_Shift(jm);
                        for (int i_local = 0;
                             i_local < kClusteredMaxSuperClusterClusters;
                             i_local += 1)
                        {
                            if (exclusion_indices[i_local] >= 0)
                            {
                                Clustered_Exclusion_Index_Ref(
                                    packed.imei[split], jm, i_local) =
                                    exclusion_indices[i_local];
                            }
                        }
                    }
                }
                cjpacked->push_back(packed);
            }
            scis->push_back(
                {super_i, shift_id, cj_begin, static_cast<int>(cjpacked->size())});
        }
    }
}
#endif

static __global__ void Check_Clustered_Rebuild(
    const int atom_numbers, const VECTOR* crd, const VECTOR* cached_crd,
    const LTMatrix3 cell, const LTMatrix3 rcell, int* need_rebuild,
    const float permit_square)
{
    SIMPLE_DEVICE_FOR(tid, atom_numbers)
    {
        const VECTOR dr = Get_Periodic_Displacement(crd[tid], cached_crd[tid],
                                                    cell, rcell);
        if (dr * dr > permit_square)
        {
            need_rebuild[0] = 1;
        }
    }
}

static __global__ void Build_Cornerstone_Sort_Keys(const int atom_numbers,
                                                   const VECTOR* crd,
                                                   const LTMatrix3 rcell,
                                                   const bool use_morton_sfc,
                                                   uint64_t* keys,
                                                   int* permutation)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        const VECTOR frac = Wrap_To_Box_Fractional(crd[i], rcell);
        constexpr int bits = cstone::maxTreeLevel<CornerstoneKey>{};
        const uint32_t x = Quantize_Unit_Coordinate(frac.x, bits);
        const uint32_t y = Quantize_Unit_Coordinate(frac.y, bits);
        const uint32_t z = Quantize_Unit_Coordinate(frac.z, bits);
        keys[i] = use_morton_sfc ? cstone::iMorton<CornerstoneKey>(x, y, z)
                                 : cstone::iHilbert<CornerstoneKey>(x, y, z);
        permutation[i] = i;
    }
}

static __global__ void Copy_UInt_To_Int(const int count, const unsigned* src,
                                        int* dest)
{
    SIMPLE_DEVICE_FOR(i, count) { dest[i] = static_cast<int>(src[i]); }
}

static __global__ void Build_Local_Atom_To_Molecule_Map(
    const int atom_numbers, const int global_atom_numbers,
    const int* atom_local, const int* global_atom_to_molecule,
    int* local_atom_to_molecule)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        const int global_atom = atom_local != NULL ? atom_local[i] : i;
        int molecule_id = -1;
        if (global_atom >= 0 && global_atom < global_atom_numbers &&
            global_atom_to_molecule != NULL)
        {
            molecule_id = global_atom_to_molecule[global_atom];
        }
        local_atom_to_molecule[i] = molecule_id;
    }
}

static __global__ void Build_Cluster_Molecule_Metadata(
    const int cluster_numbers, const int cluster_size, const int* permutation,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const int* atom_to_molecule, uint64_t* cluster_molecule_signatures,
    int* cluster_molecule_ids)
{
    SIMPLE_DEVICE_FOR(cluster_i, cluster_numbers)
    {
        const int atom_start = cluster_offsets[cluster_i];
        const unsigned int valid_mask = cluster_valid_masks[cluster_i];
        uint64_t signature = 0ull;
        for (int lane = 0; lane < cluster_size; lane += 1)
        {
            int molecule_id = -1;
            if ((valid_mask & (1u << lane)) != 0u && atom_to_molecule != NULL)
            {
                const int atom = permutation[atom_start + lane];
                molecule_id = atom_to_molecule[atom];
                if (molecule_id >= 0)
                {
                    signature |=
                        1ull << (static_cast<unsigned int>(molecule_id) & 63u);
                }
            }
            cluster_molecule_ids[cluster_i * cluster_size + lane] = molecule_id;
        }
        cluster_molecule_signatures[cluster_i] = signature;
    }
}

static __global__ void Build_Leaf_Molecule_Sort_Keys(
    const int leaf_numbers, const int* leaf_atom_offsets,
    const int* permutation, const int* atom_to_molecule,
    const int* atom_local, uint64_t* keys)
{
    SIMPLE_DEVICE_FOR(leaf_i, leaf_numbers)
    {
        const int atom_start = leaf_atom_offsets[leaf_i];
        const int atom_end = leaf_atom_offsets[leaf_i + 1];
        const uint64_t leaf_key =
            (static_cast<uint64_t>(leaf_i) & 0x00ffffffull) << 40;
        for (int sorted_pos = atom_start; sorted_pos < atom_end; sorted_pos += 1)
        {
            const int atom = permutation[sorted_pos];
            const int molecule_id =
                atom_to_molecule != NULL ? atom_to_molecule[atom] : -1;
            const int global_atom = atom_local != NULL ? atom_local[atom] : atom;
            const uint64_t molecule_key =
                static_cast<uint64_t>(
                    molecule_id >= 0 ? molecule_id : global_atom) &
                0xffffffffull;
            const uint64_t atom_key =
                static_cast<uint64_t>(global_atom) & 0xffull;
            keys[sorted_pos] = leaf_key | (molecule_key << 8) | atom_key;
        }
    }
}

static __global__ void Build_Leaf_Oxygen_Key_Packing_Sort_Keys(
    const int leaf_numbers, const int* leaf_atom_offsets,
    const int* permutation, const int* atom_to_molecule,
    const int* atom_local, uint64_t* keys)
{
    SIMPLE_DEVICE_FOR(leaf_i, leaf_numbers)
    {
        const int atom_start = leaf_atom_offsets[leaf_i];
        const int atom_end = leaf_atom_offsets[leaf_i + 1];
        const uint64_t leaf_key =
            (static_cast<uint64_t>(leaf_i) & 0x00ffffffull) << 40;

        for (int sorted_pos = atom_start; sorted_pos < atom_end; sorted_pos += 1)
        {
            const int atom = permutation[sorted_pos];
            const int molecule_id =
                atom_to_molecule != NULL ? atom_to_molecule[atom] : -1;
            const int global_atom = atom_local != NULL ? atom_local[atom] : atom;

            int representative_pos = sorted_pos;
            int representative_global = global_atom;
            if (molecule_id >= 0 && atom_to_molecule != NULL)
            {
                for (int scan_pos = atom_start; scan_pos < atom_end;
                     scan_pos += 1)
                {
                    const int scan_atom = permutation[scan_pos];
                    if (atom_to_molecule[scan_atom] != molecule_id)
                    {
                        continue;
                    }
                    const int scan_global =
                        atom_local != NULL ? atom_local[scan_atom] : scan_atom;
                    if (scan_global < representative_global)
                    {
                        representative_global = scan_global;
                        representative_pos = scan_pos;
                    }
                }
            }

            const int representative_offset = representative_pos - atom_start;
            int atom_order = global_atom - representative_global;
            if (atom_order < 0 || atom_order > 255)
            {
                atom_order = global_atom & 0xff;
            }
            const uint64_t representative_key =
                (static_cast<uint64_t>(representative_offset) & 0xffffull)
                << 24;
            const uint64_t molecule_key =
                (static_cast<uint64_t>(
                     molecule_id >= 0 ? molecule_id : global_atom) &
                 0xffffull)
                << 8;
            const uint64_t atom_key =
                static_cast<uint64_t>(atom_order) & 0xffull;
            keys[sorted_pos] = leaf_key | representative_key | molecule_key |
                               atom_key;
        }
    }
}

static __global__ void Build_Leaf_Geometry_Repack_Sort_Keys(
    const int leaf_numbers, const int* leaf_atom_offsets,
    const int* permutation, const VECTOR* crd, const LTMatrix3 rcell,
    uint64_t* keys)
{
    SIMPLE_DEVICE_FOR(leaf_i, leaf_numbers)
    {
        const int atom_start = leaf_atom_offsets[leaf_i];
        const int atom_end = leaf_atom_offsets[leaf_i + 1];
        const uint64_t leaf_key =
            (static_cast<uint64_t>(leaf_i) & 0x00ffffffull) << 40;
        constexpr int micro_bits = 13;
        for (int sorted_pos = atom_start; sorted_pos < atom_end; sorted_pos += 1)
        {
            const int atom = permutation[sorted_pos];
            const VECTOR frac = Wrap_To_Box_Fractional(crd[atom], rcell);
            const uint32_t x = Quantize_Unit_Coordinate(frac.x, micro_bits);
            const uint32_t y = Quantize_Unit_Coordinate(frac.y, micro_bits);
            const uint32_t z = Quantize_Unit_Coordinate(frac.z, micro_bits);
            const uint64_t local_key =
                cstone::iMorton<CornerstoneKey>(x, y, z) & 0xffffffffffull;
            keys[sorted_pos] = leaf_key | local_key;
        }
    }
}

static __global__ void Build_Global_Cluster_Metadata(
    const int cluster_numbers, const int total_atom_numbers,
    const int direct_local_atom_numbers, const int cluster_size,
    const int* permutation, const VECTOR* crd,
    const LTMatrix3 cell, const LTMatrix3 rcell, int* cluster_offsets,
    unsigned int* cluster_valid_masks, unsigned int* cluster_local_masks,
    VECTOR* cluster_centers, VECTOR* cluster_extents, float* cluster_radii)
{
    SIMPLE_DEVICE_FOR(cluster_i, cluster_numbers)
    {
        const int start = cluster_i * cluster_size;
        const int end = IntMin(total_atom_numbers, start + cluster_size);
        const int count = IntMax(0, end - start);
        unsigned int valid_mask = 0u;
        unsigned int local_mask = 0u;
        VECTOR center = {0.0f, 0.0f, 0.0f};
        VECTOR extent = {0.0f, 0.0f, 0.0f};
        float radius = 0.0f;

        cluster_offsets[cluster_i] = start;
        if (count > 0)
        {
            const VECTOR anchor = crd[permutation[start]];
            for (int lane = 0; lane < count; lane += 1)
            {
                valid_mask |= (1u << lane);
                const int atom_index = permutation[start + lane];
                if (atom_index < direct_local_atom_numbers)
                {
                    local_mask |= (1u << lane);
                }
                const VECTOR pos = crd[atom_index];
                center = center +
                         (anchor + Get_Periodic_Displacement(pos, anchor,
                                                             cell, rcell));
            }
            center = (1.0f / static_cast<float>(count)) * center;
            center = Get_Periodic_Coordinate(center, cell, rcell);
            for (int lane = 0; lane < count; lane += 1)
            {
                const int atom_index = permutation[start + lane];
                const VECTOR pos = crd[atom_index];
                const VECTOR dr =
                    Get_Periodic_Displacement(pos, center, cell, rcell);
                extent.x = fmaxf(extent.x, fabsf(dr.x));
                extent.y = fmaxf(extent.y, fabsf(dr.y));
                extent.z = fmaxf(extent.z, fabsf(dr.z));
                radius = fmaxf(radius, norm3df(dr.x, dr.y, dr.z));
            }
        }

        cluster_valid_masks[cluster_i] = valid_mask;
        cluster_local_masks[cluster_i] = local_mask;
        cluster_centers[cluster_i] = center;
        cluster_extents[cluster_i] = extent;
        cluster_radii[cluster_i] = radius;
        if (cluster_i == cluster_numbers - 1)
        {
            cluster_offsets[cluster_numbers] = end;
        }
    }
}

static __global__ void Build_Leaf_Cluster_Ranges(const int leaf_numbers,
                                                 const int cluster_size,
                                                 const int* leaf_atom_offsets,
                                                 int* leaf_cluster_starts,
                                                 int* leaf_cluster_ends)
{
    SIMPLE_DEVICE_FOR(leaf_i, leaf_numbers)
    {
        const int atom_start = leaf_atom_offsets[leaf_i];
        const int atom_end = leaf_atom_offsets[leaf_i + 1];
        const int cluster_start = atom_start / cluster_size;
        const int cluster_end =
            atom_end > atom_start ? (atom_end + cluster_size - 1) / cluster_size
                                  : cluster_start;
        leaf_cluster_starts[leaf_i] = cluster_start;
        leaf_cluster_ends[leaf_i] = cluster_end;
    }
}

static __global__ void Build_Leaf_All_Local_Flags(
    const int leaf_numbers, const int* leaf_cluster_starts,
    const int* leaf_cluster_ends, const unsigned int* cluster_local_masks,
    int* leaf_all_local)
{
    SIMPLE_DEVICE_FOR(leaf_i, leaf_numbers)
    {
        const int cluster_start = leaf_cluster_starts[leaf_i];
        const int cluster_end = leaf_cluster_ends[leaf_i];
        int all_local = cluster_end > cluster_start ? 1 : 0;
        for (int cluster_j = cluster_start; cluster_j < cluster_end; cluster_j += 1)
        {
            if (cluster_local_masks[cluster_j] == 0u)
            {
                all_local = 0;
                break;
            }
        }
        leaf_all_local[leaf_i] = all_local;
    }
}

static __global__ void Build_Fixed_Group_Offsets(const int offset_numbers,
                                                 const int group_size,
                                                 const int total_numbers,
                                                 int* offsets)
{
    SIMPLE_DEVICE_FOR(group_i, offset_numbers)
    {
        offsets[group_i] =
            IntMin(group_i * group_size, total_numbers);
    }
}

static __global__ void Build_Spatial_Group_Start_Flags(
    const int cluster_numbers, const int max_group_size,
    const float link_cutoff, const LTMatrix3 cell, const LTMatrix3 rcell,
    const unsigned int* cluster_valid_masks, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, int* group_start_flags)
{
    SIMPLE_DEVICE_FOR(cluster_i, cluster_numbers)
    {
        int group_start = 0;
        if (cluster_i == 0)
        {
            group_start = 1;
        }
        else if ((cluster_i % max_group_size) == 0)
        {
            group_start = 1;
        }
        else if (cluster_valid_masks[cluster_i - 1] == 0u ||
                 cluster_valid_masks[cluster_i] == 0u)
        {
            group_start = 1;
        }
        else
        {
            const bool linked = Cluster_Aabb_Overlaps(
                cluster_centers[cluster_i - 1], cluster_extents[cluster_i - 1],
                cluster_centers[cluster_i], cluster_extents[cluster_i],
                link_cutoff, cell, rcell);
            group_start = linked ? 0 : 1;
        }
        group_start_flags[cluster_i] = group_start;
    }
}

static __global__ void Build_Group_Offsets_From_Start_Flags(
    const int cluster_numbers, const int super_cluster_numbers,
    const int* group_start_flags, const int* group_ids, int* offsets)
{
    SIMPLE_DEVICE_FOR(cluster_i, cluster_numbers + 1)
    {
        if (cluster_i == cluster_numbers)
        {
            offsets[super_cluster_numbers] = cluster_numbers;
        }
        else if (group_start_flags[cluster_i] != 0)
        {
            offsets[group_ids[cluster_i]] = cluster_i;
        }
    }
}

static __global__ void Build_Cluster_To_Supercluster(
    const int super_cluster_numbers, const int* super_cluster_offsets,
    int* cluster_to_supercluster)
{
    SIMPLE_DEVICE_FOR(super_i, super_cluster_numbers)
    {
        const int cluster_start = super_cluster_offsets[super_i];
        const int cluster_end = super_cluster_offsets[super_i + 1];
        for (int cluster_i = cluster_start; cluster_i < cluster_end; cluster_i += 1)
        {
            cluster_to_supercluster[cluster_i] = super_i;
        }
    }
}

static __global__ void Build_Supercluster_Metadata(
    const int super_cluster_numbers, const int super_cluster_clusters,
    const float cutoff, const LTMatrix3 rcell, const int* super_cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, int* super_cluster_has_local,
    VECTOR* super_cluster_centers, VECTOR* super_cluster_sizes)
{
    SIMPLE_DEVICE_FOR(super_i, super_cluster_numbers)
    {
        const int cluster_start = super_cluster_offsets[super_i];
        const int cluster_end = super_cluster_offsets[super_i + 1];
        int cluster_count = 0;
        int has_local = 0;
        VECTOR anchor_frac = {0.0f, 0.0f, 0.0f};
        VECTOR frac_sum = {0.0f, 0.0f, 0.0f};
        VECTOR frac_center = {0.0f, 0.0f, 0.0f};
        VECTOR frac_size = {0.0f, 0.0f, 0.0f};

        for (int cluster_i = cluster_start; cluster_i < cluster_end; cluster_i += 1)
        {
            if (cluster_valid_masks[cluster_i] == 0u)
            {
                continue;
            }
            const VECTOR frac =
                Wrap_To_Box_Fractional(cluster_centers[cluster_i], rcell);
            if (cluster_count == 0)
            {
                anchor_frac = frac;
            }
            frac_sum = frac_sum +
                       (anchor_frac +
                        Periodic_Unit_Displacement(frac, anchor_frac));
            has_local |= cluster_local_masks[cluster_i] != 0u;
            cluster_count += 1;
        }

        if (cluster_count > 0)
        {
            frac_center = (1.0f / static_cast<float>(cluster_count)) * frac_sum;
            frac_center = Wrap_Unit_Coordinate(frac_center);
            const VECTOR cutoff_pad = Fractional_Cutoff_Pad(cutoff, rcell);
            for (int cluster_i = cluster_start; cluster_i < cluster_end;
                 cluster_i += 1)
            {
                if (cluster_valid_masks[cluster_i] == 0u)
                {
                    continue;
                }
                const VECTOR frac =
                    Wrap_To_Box_Fractional(cluster_centers[cluster_i], rcell);
                const VECTOR dfrac =
                    Periodic_Unit_Displacement(frac, frac_center);
                const VECTOR extent_pad =
                    Fractional_Extent_Pad(cluster_extents[cluster_i], rcell);
                frac_size.x =
                    fmaxf(frac_size.x, fabsf(dfrac.x) + extent_pad.x);
                frac_size.y =
                    fmaxf(frac_size.y, fabsf(dfrac.y) + extent_pad.y);
                frac_size.z =
                    fmaxf(frac_size.z, fabsf(dfrac.z) + extent_pad.z);
            }
            frac_size.x = fminf(0.5f, frac_size.x + cutoff_pad.x);
            frac_size.y = fminf(0.5f, frac_size.y + cutoff_pad.y);
            frac_size.z = fminf(0.5f, frac_size.z + cutoff_pad.z);
        }

        (void)super_cluster_clusters;
        super_cluster_has_local[super_i] = has_local;
        super_cluster_centers[super_i] = frac_center;
        super_cluster_sizes[super_i] = frac_size;
    }
}

static __global__ void Build_Local_Supercluster_Flags(
    const int super_cluster_numbers, const int* super_cluster_has_local,
    int* sci_flags)
{
    SIMPLE_DEVICE_FOR(super_i, super_cluster_numbers)
    {
        sci_flags[super_i] = super_cluster_has_local[super_i] != 0 ? 1 : 0;
    }
}

static __host__ __device__ __forceinline__ bool
Clustered_Shifted_Box_Overlaps_Unit_Interval(float center, float half_size,
                                             int shift_component)
{
    const float shifted_center =
        center + static_cast<float>(Clamp_Shift_Component(shift_component));
    const float lower = shifted_center - half_size;
    const float upper = shifted_center + half_size;
    return upper > 0.0f && lower < 1.0f;
}

static __global__ void Count_Supercluster_Active_Shifts(
    const int sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    int* candidate_shift_counts)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const int super_i = sci_supercluster_ids[sci];
        const VECTOR center = super_cluster_centers[super_i];
        const VECTOR size = super_cluster_sizes[super_i];
        int count = 0;
        for (int sx = -1; sx <= 1; sx += 1)
        {
            if (!Clustered_Shifted_Box_Overlaps_Unit_Interval(center.x, size.x,
                                                              sx))
            {
                continue;
            }
            for (int sy = -1; sy <= 1; sy += 1)
            {
                if (!Clustered_Shifted_Box_Overlaps_Unit_Interval(center.y,
                                                                  size.y, sy))
                {
                    continue;
                }
                for (int sz = -1; sz <= 1; sz += 1)
                {
                    if (!Clustered_Shifted_Box_Overlaps_Unit_Interval(center.z,
                                                                      size.z,
                                                                      sz))
                    {
                        continue;
                    }
                    count += 1;
                }
            }
        }
        candidate_shift_counts[sci] = count;
    }
}

static __global__ void Fill_Supercluster_Active_Shifts(
    const int sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    const int* candidate_shift_offsets, int* candidate_supercluster_ids,
    int* candidate_shift_ids)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const int super_i = sci_supercluster_ids[sci];
        const VECTOR center = super_cluster_centers[super_i];
        const VECTOR size = super_cluster_sizes[super_i];
        int write_offset = candidate_shift_offsets[sci];
        for (int sx = -1; sx <= 1; sx += 1)
        {
            if (!Clustered_Shifted_Box_Overlaps_Unit_Interval(center.x, size.x,
                                                              sx))
            {
                continue;
            }
            for (int sy = -1; sy <= 1; sy += 1)
            {
                if (!Clustered_Shifted_Box_Overlaps_Unit_Interval(center.y,
                                                                  size.y, sy))
                {
                    continue;
                }
                for (int sz = -1; sz <= 1; sz += 1)
                {
                    if (!Clustered_Shifted_Box_Overlaps_Unit_Interval(center.z,
                                                                      size.z,
                                                                      sz))
                    {
                        continue;
                    }
                    candidate_supercluster_ids[write_offset] = super_i;
                    candidate_shift_ids[write_offset] =
                        Encode_Shift_Id(sx, sy, sz);
                    write_offset += 1;
                }
            }
        }
    }
}

static __global__ void Fill_Local_Supercluster_Ids(
    const int super_cluster_numbers, const int* super_cluster_has_local,
    const int* sci_offsets, int* sci_supercluster_ids)
{
    SIMPLE_DEVICE_FOR(super_i, super_cluster_numbers)
    {
        if (super_cluster_has_local[super_i] != 0)
        {
            const int sci = sci_offsets[super_i];
            sci_supercluster_ids[sci] = super_i;
        }
    }
}

static __global__ void Count_Supercluster_Candidate_Leaves(
    const int sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    const CornerstoneKey* node_prefixes,
    const CornerstoneNodeIndex* child_offsets,
    const CornerstoneNodeIndex* parents,
    const CornerstoneNodeIndex* internal_to_leaf,
    const bool use_morton_sfc, int* candidate_leaf_counts)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const int super_i = sci_supercluster_ids[sci];
        const VECTOR target_center = super_cluster_centers[super_i];
        const VECTOR target_size = super_cluster_sizes[super_i];
        int count = 0;

        auto overlaps = [=](CornerstoneNodeIndex node)
        {
            return Cornerstone_Node_Overlaps_Box(node_prefixes[node],
                                                 target_center, target_size,
                                                 use_morton_sfc);
        };
        auto endpoint = [&](CornerstoneNodeIndex node)
        {
            const int leaf_j = internal_to_leaf[node];
            if (leaf_j >= 0)
            {
                count += 1;
            }
        };
        cstone::singleTraversal(child_offsets, parents, overlaps, endpoint);
        candidate_leaf_counts[sci] = count;
    }
}

static __global__ void Count_Supercluster_Candidate_Leaves_Fixed_Shift(
    const int sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    const int* super_cluster_offsets, const int* leaf_cluster_starts,
    const int* leaf_cluster_ends,
    const int* leaf_all_local,
    const LTMatrix3 cell,
    const float cutoff, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const CornerstoneKey* node_prefixes,
    const CornerstoneNodeIndex* child_offsets,
    const CornerstoneNodeIndex* parents,
    const CornerstoneNodeIndex* internal_to_leaf,
    const int* candidate_shift_ids, const bool central_halfshell_culling,
    const bool leaf_screening, const bool use_morton_sfc,
    int* candidate_leaf_counts)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const int sci_base =
            candidate_shift_ids == NULL ? sci / kClusteredShiftCount : sci;
        const int candidate_shift_id =
            candidate_shift_ids != NULL ? candidate_shift_ids[sci]
                                        : (sci % kClusteredShiftCount);
        const int super_i = sci_supercluster_ids[sci_base];
        const VECTOR target_center = super_cluster_centers[super_i];
        const VECTOR target_size = super_cluster_sizes[super_i];
        const int cluster_i_start = super_cluster_offsets[super_i];
        const int cluster_i_end = super_cluster_offsets[super_i + 1];
        const VECTOR shift_vec = Shift_Vector_From_Id(candidate_shift_id, cell);
        int count = 0;

        auto overlaps = [=](CornerstoneNodeIndex node)
        {
            return Cornerstone_Node_Overlaps_Shifted_Box(
                node_prefixes[node], target_center, target_size,
                candidate_shift_id, use_morton_sfc);
        };
        auto endpoint = [&](CornerstoneNodeIndex node)
        {
            const int leaf_j = internal_to_leaf[node];
            if (leaf_j >= 0)
            {
                if (central_halfshell_culling &&
                    candidate_shift_id == kClusteredCentralShiftId &&
                    leaf_all_local[leaf_j] != 0 &&
                    leaf_cluster_ends[leaf_j] <= cluster_i_start)
                {
                    return;
                }
                if (leaf_screening &&
                    !Leaf_Has_Fixed_Shift_Candidate_Overlap(
                        cluster_i_start, cluster_i_end,
                        leaf_cluster_starts[leaf_j], leaf_cluster_ends[leaf_j],
                        candidate_shift_id, cutoff, shift_vec,
                        cluster_valid_masks,
                        cluster_local_masks, cluster_centers, cluster_extents))
                {
                    return;
                }
                count += 1;
            }
        };
        cstone::singleTraversal(child_offsets, parents, overlaps, endpoint);
        candidate_leaf_counts[sci] = count;
    }
}

static __global__ void Fill_Supercluster_Candidate_Leaves(
    const int sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    const CornerstoneKey* node_prefixes,
    const CornerstoneNodeIndex* child_offsets,
    const CornerstoneNodeIndex* parents,
    const CornerstoneNodeIndex* internal_to_leaf,
    const int* candidate_leaf_offsets, const bool use_morton_sfc,
    int* candidate_leaf_ids)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const int super_i = sci_supercluster_ids[sci];
        const VECTOR target_center = super_cluster_centers[super_i];
        const VECTOR target_size = super_cluster_sizes[super_i];
        int write_offset = candidate_leaf_offsets[sci];

        auto overlaps = [=](CornerstoneNodeIndex node)
        {
            return Cornerstone_Node_Overlaps_Box(node_prefixes[node],
                                                 target_center, target_size,
                                                 use_morton_sfc);
        };
        auto endpoint = [&](CornerstoneNodeIndex node)
        {
            const int leaf_j = internal_to_leaf[node];
            if (leaf_j >= 0)
            {
                candidate_leaf_ids[write_offset] = leaf_j;
                write_offset += 1;
            }
        };
        cstone::singleTraversal(child_offsets, parents, overlaps, endpoint);
    }
}

static __global__ void Fill_Supercluster_Candidate_Leaves_Fixed_Shift(
    const int sci_numbers, const int* sci_supercluster_ids,
    const VECTOR* super_cluster_centers, const VECTOR* super_cluster_sizes,
    const int* super_cluster_offsets, const int* leaf_cluster_starts,
    const int* leaf_cluster_ends,
    const int* leaf_all_local,
    const LTMatrix3 cell,
    const float cutoff, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const CornerstoneKey* node_prefixes,
    const CornerstoneNodeIndex* child_offsets,
    const CornerstoneNodeIndex* parents,
    const CornerstoneNodeIndex* internal_to_leaf,
    const int* candidate_shift_ids, const int* candidate_leaf_offsets,
    const bool central_halfshell_culling, const bool leaf_screening,
    const bool use_morton_sfc, int* candidate_leaf_ids)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const int sci_base =
            candidate_shift_ids == NULL ? sci / kClusteredShiftCount : sci;
        const int candidate_shift_id =
            candidate_shift_ids != NULL ? candidate_shift_ids[sci]
                                        : (sci % kClusteredShiftCount);
        const int super_i = sci_supercluster_ids[sci_base];
        const VECTOR target_center = super_cluster_centers[super_i];
        const VECTOR target_size = super_cluster_sizes[super_i];
        const int cluster_i_start = super_cluster_offsets[super_i];
        const int cluster_i_end = super_cluster_offsets[super_i + 1];
        const VECTOR shift_vec = Shift_Vector_From_Id(candidate_shift_id, cell);
        int write_offset = candidate_leaf_offsets[sci];

        auto overlaps = [=](CornerstoneNodeIndex node)
        {
            return Cornerstone_Node_Overlaps_Shifted_Box(
                node_prefixes[node], target_center, target_size,
                candidate_shift_id, use_morton_sfc);
        };
        auto endpoint = [&](CornerstoneNodeIndex node)
        {
            const int leaf_j = internal_to_leaf[node];
            if (leaf_j >= 0)
            {
                if (central_halfshell_culling &&
                    candidate_shift_id == kClusteredCentralShiftId &&
                    leaf_all_local[leaf_j] != 0 &&
                    leaf_cluster_ends[leaf_j] <= cluster_i_start)
                {
                    return;
                }
                if (leaf_screening &&
                    !Leaf_Has_Fixed_Shift_Candidate_Overlap(
                        cluster_i_start, cluster_i_end,
                        leaf_cluster_starts[leaf_j], leaf_cluster_ends[leaf_j],
                        candidate_shift_id, cutoff, shift_vec,
                        cluster_valid_masks,
                        cluster_local_masks, cluster_centers, cluster_extents))
                {
                    return;
                }
                candidate_leaf_ids[write_offset] = leaf_j;
                write_offset += 1;
            }
        };
        cstone::singleTraversal(child_offsets, parents, overlaps, endpoint);
    }
}

static __global__ void Build_Fixed_Shift_Candidate_Leaf_Masks(
    const int candidate_sci_numbers, const int super_cluster_clusters,
    const float cutoff, const LTMatrix3 cell,
    const int* sci_supercluster_ids, const int* super_cluster_offsets,
    const int* cluster_to_supercluster,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* candidate_shift_ids, const int* candidate_leaf_offsets,
    const int* candidate_leaf_ids, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, const int candidate_leaf_cluster_stride,
    unsigned int* candidate_leaf_reach_masks)
{
    const int lane_id = threadIdx.x & (warpSize - 1);
    const int warp_id = threadIdx.x / warpSize;
    const int warps_per_block = blockDim.x / warpSize;
    const int candidate_sci = blockIdx.x * warps_per_block + warp_id;
    if (candidate_sci >= candidate_sci_numbers)
    {
        return;
    }

    const int sci_base =
        candidate_shift_ids == NULL ? candidate_sci / kClusteredShiftCount
                                    : candidate_sci;
    const int fixed_shift_id =
        candidate_shift_ids != NULL ? candidate_shift_ids[candidate_sci]
                                    : (candidate_sci % kClusteredShiftCount);
    const VECTOR shift_vec = Shift_Vector_From_Id(fixed_shift_id, cell);
    const int super_i = sci_supercluster_ids[sci_base];
    const int cluster_i_start = super_cluster_offsets[super_i];
    const int cluster_i_end = super_cluster_offsets[super_i + 1];

    for (int candidate_idx = candidate_leaf_offsets[candidate_sci];
         candidate_idx < candidate_leaf_offsets[candidate_sci + 1];
         candidate_idx += 1)
    {
        int leaf_j = 0;
        int cluster_j_start = 0;
        int cluster_j_end = 0;
        if (lane_id == 0)
        {
            leaf_j = candidate_leaf_ids[candidate_idx];
            cluster_j_start = leaf_cluster_starts[leaf_j];
            cluster_j_end = leaf_cluster_ends[leaf_j];
        }
        leaf_j = deviceShfl(FULL_MASK, leaf_j, 0, warpSize);
        cluster_j_start = deviceShfl(FULL_MASK, cluster_j_start, 0, warpSize);
        cluster_j_end = deviceShfl(FULL_MASK, cluster_j_end, 0, warpSize);

        const int mask_base = candidate_idx * candidate_leaf_cluster_stride;
        for (int slot = lane_id; slot < candidate_leaf_cluster_stride;
             slot += warpSize)
        {
            unsigned int i_mask = 0u;
            const int cluster_j = cluster_j_start + slot;
            if (cluster_j < cluster_j_end)
            {
                const unsigned int valid_mask_j = cluster_valid_masks[cluster_j];
                if (valid_mask_j != 0u)
                {
                    const int super_j = cluster_to_supercluster[cluster_j];
                    const unsigned int local_mask_j =
                        cluster_local_masks[cluster_j];
                    if (!(local_mask_j != 0u && super_j < super_i))
                    {
                        i_mask = Build_Fixed_Shift_Cluster_I_Mask(
                            cluster_i_start, cluster_i_end, cluster_j,
                            fixed_shift_id, cutoff, shift_vec,
                            cluster_local_masks, cluster_centers,
                            cluster_extents);
                    }
                }
            }
            candidate_leaf_reach_masks[mask_base + slot] = i_mask;
        }
    }
}

static __global__ void Count_Nbnxm_Payload_From_Candidate_Leaves(
    const int candidate_sci_numbers, const int sci_shift_numbers,
    const int cluster_size, const int super_cluster_clusters,
    const int local_atom_numbers, const float cutoff, const LTMatrix3 cell,
    const LTMatrix3 rcell, const int* permutation, const int* cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* super_cluster_offsets,
    const int* cluster_to_supercluster,
    const int* sci_supercluster_ids, const VECTOR* super_cluster_centers,
    const int* candidate_shift_ids,
    const int* candidate_leaf_offsets, const int* candidate_leaf_ids,
    const int candidate_leaf_cluster_stride,
    const unsigned int* candidate_leaf_reach_masks,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, const float* cluster_radii,
    const uint64_t* cluster_molecule_signatures,
    const int* cluster_molecule_ids,
    const int* excluded_list_start, const int* excluded_list,
    const int* excluded_numbers,
    const bool fixed_shift_candidates,
    int* sci_shift_flags, int* cjpacked_group_counts, int* exclusion_counts)
{
    const int lane_id = threadIdx.x & (warpSize - 1);
    const int warp_id = threadIdx.x / warpSize;
    const int warps_per_block = blockDim.x / warpSize;
    const int candidate_sci = blockIdx.x * warps_per_block + warp_id;
    if (candidate_sci >= candidate_sci_numbers)
    {
        return;
    }

    (void)sci_shift_numbers;
    (void)super_cluster_centers;
    (void)cluster_radii;

    const int sci_base =
        (fixed_shift_candidates && candidate_shift_ids == NULL)
            ? candidate_sci / kClusteredShiftCount
            : candidate_sci;
    const int fixed_shift_id =
        !fixed_shift_candidates
            ? -1
            : (candidate_shift_ids != NULL ? candidate_shift_ids[candidate_sci]
                                           : (candidate_sci % kClusteredShiftCount));
    const VECTOR fixed_shift_vec =
        fixed_shift_candidates
            ? Shift_Vector_From_Id(fixed_shift_id, cell)
            : VECTOR{0.0f, 0.0f, 0.0f};
    const int super_i = sci_supercluster_ids[sci_base];
    const int cluster_i_start = super_cluster_offsets[super_i];
    const int cluster_i_end = super_cluster_offsets[super_i + 1];
    __shared__ unsigned int
        shared_i_local_masks[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                            [kClusteredMaxSuperClusterClusters];
    __shared__ int
        shared_i_atom_ids[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                         [kClusteredMaxSuperClusterClusters][kClusteredClusterSize];
    __shared__ uint64_t
        shared_i_signatures[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                           [kClusteredMaxSuperClusterClusters];
    __shared__ int
        shared_i_molecule_ids[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                             [kClusteredMaxSuperClusterClusters]
                             [kClusteredClusterSize];
    __shared__ float
        shared_i_center_x[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                         [kClusteredMaxSuperClusterClusters];
    __shared__ float
        shared_i_center_y[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                         [kClusteredMaxSuperClusterClusters];
    __shared__ float
        shared_i_center_z[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                         [kClusteredMaxSuperClusterClusters];
    __shared__ float
        shared_i_extent_x[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                         [kClusteredMaxSuperClusterClusters];
    __shared__ float
        shared_i_extent_y[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                         [kClusteredMaxSuperClusterClusters];
    __shared__ float
        shared_i_extent_z[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                         [kClusteredMaxSuperClusterClusters];
    __shared__ int
        shared_j_atom_ids[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                         [kClusteredClusterSize];
    __shared__ uint64_t
        shared_j_signature[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize];
    __shared__ int
        shared_j_molecule_ids[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                             [kClusteredClusterSize];
    __shared__ int
        shared_shift_record_counts[kClusteredBuilderBlockSize /
                                   kClusteredBuilderWarpSize]
                                  [kClusteredShiftCount];
    __shared__ int
        shared_shift_exclusion_counts[kClusteredBuilderBlockSize /
                                      kClusteredBuilderWarpSize]
                                     [kClusteredShiftCount];
    const bool has_molecule_metadata =
        cluster_molecule_signatures != NULL && cluster_molecule_ids != NULL;
    if (lane_id < kClusteredMaxSuperClusterClusters)
    {
        const int cluster_i = cluster_i_start + lane_id;
        if (cluster_i < cluster_i_end)
        {
            const VECTOR center_i = cluster_centers[cluster_i];
            const VECTOR extent_i = cluster_extents[cluster_i];
            shared_i_local_masks[warp_id][lane_id] = cluster_local_masks[cluster_i];
            shared_i_center_x[warp_id][lane_id] = center_i.x;
            shared_i_center_y[warp_id][lane_id] = center_i.y;
            shared_i_center_z[warp_id][lane_id] = center_i.z;
            shared_i_extent_x[warp_id][lane_id] = extent_i.x;
            shared_i_extent_y[warp_id][lane_id] = extent_i.y;
            shared_i_extent_z[warp_id][lane_id] = extent_i.z;
            shared_i_signatures[warp_id][lane_id] =
                has_molecule_metadata ? cluster_molecule_signatures[cluster_i]
                                      : 0ull;
        }
        else
        {
            shared_i_local_masks[warp_id][lane_id] = 0u;
            shared_i_center_x[warp_id][lane_id] = 0.0f;
            shared_i_center_y[warp_id][lane_id] = 0.0f;
            shared_i_center_z[warp_id][lane_id] = 0.0f;
            shared_i_extent_x[warp_id][lane_id] = 0.0f;
            shared_i_extent_y[warp_id][lane_id] = 0.0f;
            shared_i_extent_z[warp_id][lane_id] = 0.0f;
            shared_i_signatures[warp_id][lane_id] = 0ull;
        }
    }
    for (int atom_slot = lane_id;
         atom_slot < kClusteredMaxSuperClusterClusters * kClusteredClusterSize;
         atom_slot += warpSize)
    {
        const int i_local = atom_slot / kClusteredClusterSize;
        const int atom_lane = atom_slot % kClusteredClusterSize;
        const int cluster_i = cluster_i_start + i_local;
        if (cluster_i < cluster_i_end)
        {
            const int sorted_atom_i = cluster_offsets[cluster_i] + atom_lane;
            shared_i_atom_ids[warp_id][i_local][atom_lane] =
                permutation[sorted_atom_i];
            shared_i_molecule_ids[warp_id][i_local][atom_lane] =
                has_molecule_metadata
                    ? cluster_molecule_ids[cluster_i * kClusteredClusterSize +
                                           atom_lane]
                    : -1;
        }
        else
        {
            shared_i_atom_ids[warp_id][i_local][atom_lane] = -1;
            shared_i_molecule_ids[warp_id][i_local][atom_lane] = -1;
        }
    }
    __syncwarp();
    const int sci_shift_base = fixed_shift_candidates
                                   ? candidate_sci
                                   : candidate_sci * kClusteredShiftCount;
    if (lane_id < kClusteredShiftCount)
    {
        shared_shift_record_counts[warp_id][lane_id] = 0;
        shared_shift_exclusion_counts[warp_id][lane_id] = 0;
    }
    __syncwarp();
    // Candidate leaves arrive in leaf/SFC order, so a monotonic end marker
    // removes duplicate cluster visits when a globally packed cluster straddles
    // multiple leaves.
    int processed_cluster_end = 0;
    const int active_cluster_count = cluster_i_end - cluster_i_start;
    const unsigned int active_i_lane_mask =
        active_cluster_count > 0
            ? ((1u << static_cast<unsigned int>(active_cluster_count)) - 1u)
            : 0u;

    for (int candidate_idx = candidate_leaf_offsets[candidate_sci];
         candidate_idx < candidate_leaf_offsets[candidate_sci + 1];
         candidate_idx += 1)
    {
        int leaf_j = 0;
        int cluster_j_start = 0;
        int cluster_j_end = 0;
        if (lane_id == 0)
        {
            leaf_j = candidate_leaf_ids[candidate_idx];
            cluster_j_start = leaf_cluster_starts[leaf_j];
            cluster_j_end = leaf_cluster_ends[leaf_j];
        }
        leaf_j = deviceShfl(FULL_MASK, leaf_j, 0, warpSize);
        cluster_j_start = deviceShfl(FULL_MASK, cluster_j_start, 0, warpSize);
        cluster_j_end = deviceShfl(FULL_MASK, cluster_j_end, 0, warpSize);
        const int leaf_mask_base =
            candidate_leaf_reach_masks != NULL
                ? candidate_idx * candidate_leaf_cluster_stride
                : 0;

        const int deduped_cluster_j_start =
            IntMax(cluster_j_start, processed_cluster_end);
        for (int cluster_j = deduped_cluster_j_start; cluster_j < cluster_j_end;
             cluster_j += 1)
        {
            unsigned int precomputed_i_mask = 0u;
            if (candidate_leaf_reach_masks != NULL)
            {
                if (lane_id == 0)
                {
                    precomputed_i_mask =
                        candidate_leaf_reach_masks[leaf_mask_base +
                                                   (cluster_j -
                                                    cluster_j_start)];
                }
                precomputed_i_mask =
                    deviceShfl(FULL_MASK, precomputed_i_mask, 0, warpSize) &
                    active_i_lane_mask;
                if (precomputed_i_mask == 0u)
                {
                    continue;
                }
            }
            unsigned int valid_mask_j = 0u;
            unsigned int local_mask_j = 0u;
            VECTOR center_j = {0.0f, 0.0f, 0.0f};
            VECTOR extent_j = {0.0f, 0.0f, 0.0f};
            int super_j = 0;
            uint64_t signature_j = 0ull;
            if (lane_id == 0)
            {
                valid_mask_j = cluster_valid_masks[cluster_j];
                local_mask_j = cluster_local_masks[cluster_j];
                if (valid_mask_j != 0u)
                {
                    super_j = cluster_to_supercluster[cluster_j];
                    if (!(local_mask_j != 0u && super_j < super_i))
                    {
                        center_j = cluster_centers[cluster_j];
                        extent_j = cluster_extents[cluster_j];
                    }
                    if (has_molecule_metadata)
                    {
                        signature_j = cluster_molecule_signatures[cluster_j];
                    }
                }
            }
            valid_mask_j = deviceShfl(FULL_MASK, valid_mask_j, 0, warpSize);
            local_mask_j = deviceShfl(FULL_MASK, local_mask_j, 0, warpSize);
            super_j = deviceShfl(FULL_MASK, super_j, 0, warpSize);
            center_j = Clustered_Warp_Broadcast_Vector(center_j, 0);
            extent_j = Clustered_Warp_Broadcast_Vector(extent_j, 0);
            signature_j = Broadcast_Clustered_Warp_U64(signature_j, 0);

            if (valid_mask_j == 0u)
            {
                continue;
            }
            if (local_mask_j != 0u && super_j < super_i)
            {
                continue;
            }
            if (lane_id == 0)
            {
                shared_j_signature[warp_id] = signature_j;
            }
            int pair_shift_id = -1;
            bool exclusion_candidate = false;
            if (lane_id < active_cluster_count)
            {
                const int i_local = lane_id;
                if (candidate_leaf_reach_masks != NULL)
                {
                    if ((precomputed_i_mask &
                         (1u << static_cast<unsigned int>(i_local))) != 0u)
                    {
                        pair_shift_id = fixed_shift_id;
                        exclusion_candidate =
                            !has_molecule_metadata ||
                            (shared_i_signatures[warp_id][i_local] &
                             signature_j) != 0ull;
                    }
                }
                else
                {
                    const unsigned int local_mask_i =
                        shared_i_local_masks[warp_id][i_local];
                    if (local_mask_i != 0u)
                    {
                        const VECTOR center_i = {
                            shared_i_center_x[warp_id][i_local],
                            shared_i_center_y[warp_id][i_local],
                            shared_i_center_z[warp_id][i_local]};
                        const VECTOR extent_i = {
                            shared_i_extent_x[warp_id][i_local],
                            shared_i_extent_y[warp_id][i_local],
                            shared_i_extent_z[warp_id][i_local]};
                        if (fixed_shift_candidates)
                        {
                            pair_shift_id = fixed_shift_id;
                        }
                        else
                        {
                            pair_shift_id = Determine_Cluster_Pair_Shift_Id(
                                center_i, center_j, rcell);
                        }
                        if (pair_shift_id == kClusteredCentralShiftId &&
                            cluster_j >= cluster_i_start &&
                            cluster_j < cluster_i_end &&
                            (cluster_i_start + i_local) > cluster_j)
                        {
                            pair_shift_id = -1;
                        }
                        else if (pair_shift_id >= 0 &&
                                 !Cluster_Aabb_Overlaps_Shifted(
                                     center_i, extent_i, center_j, extent_j,
                                     cutoff,
                                     fixed_shift_candidates
                                         ? fixed_shift_vec
                                         : Shift_Vector_From_Id(pair_shift_id,
                                                                cell)))
                        {
                            pair_shift_id = -1;
                        }
                        exclusion_candidate =
                            pair_shift_id >= 0 &&
                            (!has_molecule_metadata ||
                             (shared_i_signatures[warp_id][i_local] &
                              signature_j) != 0ull);
                    }
                }
            }

            const unsigned int active_pair_lane_mask =
                candidate_leaf_reach_masks != NULL
                    ? precomputed_i_mask
                    : __ballot_sync(FULL_MASK,
                                    lane_id < active_cluster_count &&
                                        pair_shift_id >= 0);
            if (active_pair_lane_mask == 0u)
            {
                continue;
            }
            const unsigned int exclusion_candidate_lane_mask =
                __ballot_sync(FULL_MASK,
                              lane_id < active_cluster_count &&
                                  exclusion_candidate);

            const bool need_j_cached_atoms =
                exclusion_candidate_lane_mask != 0u;
            if (need_j_cached_atoms)
            {
                if (lane_id < kClusteredClusterSize)
                {
                    if ((valid_mask_j & (1u << lane_id)) != 0u)
                    {
                        const int sorted_atom_j =
                            cluster_offsets[cluster_j] + lane_id;
                        shared_j_atom_ids[warp_id][lane_id] =
                            permutation[sorted_atom_j];
                        shared_j_molecule_ids[warp_id][lane_id] =
                            has_molecule_metadata
                                ? cluster_molecule_ids[cluster_j *
                                                           kClusteredClusterSize +
                                                       lane_id]
                                : -1;
                    }
                    else
                    {
                        shared_j_atom_ids[warp_id][lane_id] = -1;
                        shared_j_molecule_ids[warp_id][lane_id] = -1;
                    }
                }
                __syncwarp();
            }

            unsigned int remaining_lane_mask = active_pair_lane_mask;
            while (remaining_lane_mask != 0u)
            {
                const int leader_lane =
                    __ffs(static_cast<int>(remaining_lane_mask)) - 1;
                const int group_shift_id =
                    deviceShfl(FULL_MASK, pair_shift_id, leader_lane, warpSize);
                const unsigned int group_lane_mask =
                    __ballot_sync(FULL_MASK,
                                  lane_id < active_cluster_count &&
                                      pair_shift_id == group_shift_id);
                remaining_lane_mask &= ~group_lane_mask;
                if (lane_id == leader_lane)
                {
                    const int output_shift_idx =
                        fixed_shift_candidates ? 0 : group_shift_id;
                    shared_shift_record_counts[warp_id][output_shift_idx] += 1;
                    if (need_j_cached_atoms)
                    {
                        const unsigned int group_exclusion_imask =
                            exclusion_candidate_lane_mask &
                            (group_lane_mask & active_i_lane_mask);
                        int exclusion_count_for_group = 0;
                        unsigned int remaining_i = group_exclusion_imask;
                        while (remaining_i != 0u)
                        {
                            const int i_local =
                                __ffs(static_cast<int>(remaining_i)) - 1;
                            remaining_i &= (remaining_i - 1u);
                            const unsigned int local_mask_i =
                                shared_i_local_masks[warp_id][i_local];
                            const unsigned long long exclusion_mask =
                                Build_Exclusion_Mask_From_Cached_Atoms(
                                    shared_i_atom_ids[warp_id][i_local],
                                    shared_j_atom_ids[warp_id],
                                    shared_i_molecule_ids[warp_id][i_local],
                                    shared_j_molecule_ids[warp_id],
                                    shared_i_signatures[warp_id][i_local],
                                    shared_j_signature[warp_id],
                                    has_molecule_metadata, local_mask_i,
                                    valid_mask_j, cluster_size,
                                    local_atom_numbers, excluded_list_start,
                                    excluded_list, excluded_numbers);
                            exclusion_count_for_group +=
                                exclusion_mask != 0ull ? 1 : 0;
                        }
                        shared_shift_exclusion_counts[warp_id][output_shift_idx] +=
                            exclusion_count_for_group;
                    }
                }
                __syncwarp();
            }
        }
        processed_cluster_end = IntMax(processed_cluster_end, cluster_j_end);
    }

    __syncwarp();
    if (fixed_shift_candidates)
    {
        if (lane_id == 0)
        {
            const int record_count = shared_shift_record_counts[warp_id][0];
            sci_shift_flags[sci_shift_base] = record_count > 0 ? 1 : 0;
            cjpacked_group_counts[sci_shift_base] =
                (record_count + kClusteredMaxJGroupSize - 1) /
                kClusteredMaxJGroupSize;
            exclusion_counts[sci_shift_base] =
                shared_shift_exclusion_counts[warp_id][0];
        }
    }
    else if (lane_id < kClusteredShiftCount)
    {
        const int sci_shift = sci_shift_base + lane_id;
        const int record_count = shared_shift_record_counts[warp_id][lane_id];
        sci_shift_flags[sci_shift] = record_count > 0 ? 1 : 0;
        cjpacked_group_counts[sci_shift] =
            (record_count + kClusteredMaxJGroupSize - 1) /
            kClusteredMaxJGroupSize;
        exclusion_counts[sci_shift] =
            shared_shift_exclusion_counts[warp_id][lane_id];
    }
}

static __device__ __forceinline__ LJ_CLUSTERED_GMXPACKED_CJ
Make_Empty_Gmxpacked_CjPacked();

static __global__ void Fill_Nbnxm_Payload_From_Candidate_Leaves(
    const int candidate_sci_numbers, const int sci_shift_numbers,
    const int cluster_size, const int super_cluster_clusters,
    const int cluster_numbers, const int local_atom_numbers,
    const float cutoff, const LTMatrix3 cell,
    const LTMatrix3 rcell, const int* permutation, const int* cluster_offsets,
    const int* leaf_cluster_starts, const int* leaf_cluster_ends,
    const int* super_cluster_offsets,
    const int* cluster_to_supercluster,
    const int* sci_supercluster_ids, const VECTOR* super_cluster_centers,
    const int* candidate_shift_ids,
    const int* candidate_leaf_offsets, const int* candidate_leaf_ids,
    const int candidate_leaf_cluster_stride,
    const unsigned int* candidate_leaf_reach_masks,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const VECTOR* cluster_extents, const float* cluster_radii,
    const uint64_t* cluster_molecule_signatures,
    const int* cluster_molecule_ids,
    const int* excluded_list_start, const int* excluded_list,
    const int* excluded_numbers,
    const bool fixed_shift_candidates,
    const bool dedup_exclusion_masks,
    const int* sci_shift_flags, const int* sci_shift_offsets,
    const int* cjpacked_group_counts, const int* cjpacked_group_offsets,
    const int* exclusion_offsets, const bool build_gmxpacked_direct,
    LJ_CLUSTERED_SCI* nbnxm_sci, LJ_CLUSTERED_CJ_PACKED* nbnxm_cjpacked,
    unsigned long long* exclusion_mask_pool,
    LJ_CLUSTERED_GMXPACKED_SCI* gmxpacked_sci,
    LJ_CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked,
    LJ_CLUSTERED_GMXPACKED_EXCLUSION* gmxpacked_exclusions,
    const int early_record_analysis_capacity,
    int* early_record_analysis_raw_count,
    ClusteredEarlyRecordAnalyzeEntry* early_record_analysis_entries)
{
    const int lane_id = threadIdx.x & (warpSize - 1);
    const int warp_id = threadIdx.x / warpSize;
    const int warps_per_block = blockDim.x / warpSize;
    const int candidate_sci = blockIdx.x * warps_per_block + warp_id;
    if (candidate_sci >= candidate_sci_numbers)
    {
        return;
    }

    (void)sci_shift_numbers;
    (void)super_cluster_centers;
    (void)cluster_radii;

    const int sci_base =
        (fixed_shift_candidates && candidate_shift_ids == NULL)
            ? candidate_sci / kClusteredShiftCount
            : candidate_sci;
    const int fixed_shift_id =
        !fixed_shift_candidates
            ? -1
            : (candidate_shift_ids != NULL ? candidate_shift_ids[candidate_sci]
                                           : (candidate_sci % kClusteredShiftCount));
    const VECTOR fixed_shift_vec =
        fixed_shift_candidates
            ? Shift_Vector_From_Id(fixed_shift_id, cell)
            : VECTOR{0.0f, 0.0f, 0.0f};
    const int sci_shift_base = fixed_shift_candidates
                                   ? candidate_sci
                                   : candidate_sci * kClusteredShiftCount;
    const int super_i = sci_supercluster_ids[sci_base];
    const int cluster_i_start = super_cluster_offsets[super_i];
    const int cluster_i_end = super_cluster_offsets[super_i + 1];
    __shared__ unsigned int
        shared_i_local_masks[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                            [kClusteredMaxSuperClusterClusters];
    __shared__ int
        shared_i_atom_ids[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                         [kClusteredMaxSuperClusterClusters][kClusteredClusterSize];
    __shared__ uint64_t
        shared_i_signatures[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                           [kClusteredMaxSuperClusterClusters];
    __shared__ int
        shared_i_molecule_ids[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                             [kClusteredMaxSuperClusterClusters]
                             [kClusteredClusterSize];
    __shared__ float
        shared_i_center_x[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                         [kClusteredMaxSuperClusterClusters];
    __shared__ float
        shared_i_center_y[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                         [kClusteredMaxSuperClusterClusters];
    __shared__ float
        shared_i_center_z[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                         [kClusteredMaxSuperClusterClusters];
    __shared__ float
        shared_i_extent_x[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                         [kClusteredMaxSuperClusterClusters];
    __shared__ float
        shared_i_extent_y[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                         [kClusteredMaxSuperClusterClusters];
    __shared__ float
        shared_i_extent_z[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                         [kClusteredMaxSuperClusterClusters];
    __shared__ int
        shared_j_atom_ids[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                         [kClusteredClusterSize];
    __shared__ uint64_t
        shared_j_signature[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize];
    __shared__ int
        shared_j_molecule_ids[kClusteredBuilderBlockSize / kClusteredBuilderWarpSize]
                             [kClusteredClusterSize];
    __shared__ int
        shared_shift_write_packed[kClusteredBuilderBlockSize /
                                  kClusteredBuilderWarpSize]
                                 [kClusteredShiftCount];
    __shared__ int
        shared_shift_write_exclusion[kClusteredBuilderBlockSize /
                                     kClusteredBuilderWarpSize]
                                    [kClusteredShiftCount];
    __shared__ unsigned char
        shared_shift_jm_in_group[kClusteredBuilderBlockSize /
                                 kClusteredBuilderWarpSize]
                                [kClusteredShiftCount];
    const bool has_molecule_metadata =
        cluster_molecule_signatures != NULL && cluster_molecule_ids != NULL;
    if (lane_id < kClusteredMaxSuperClusterClusters)
    {
        const int cluster_i = cluster_i_start + lane_id;
        if (cluster_i < cluster_i_end)
        {
            const VECTOR center_i = cluster_centers[cluster_i];
            const VECTOR extent_i = cluster_extents[cluster_i];
            shared_i_local_masks[warp_id][lane_id] = cluster_local_masks[cluster_i];
            shared_i_center_x[warp_id][lane_id] = center_i.x;
            shared_i_center_y[warp_id][lane_id] = center_i.y;
            shared_i_center_z[warp_id][lane_id] = center_i.z;
            shared_i_extent_x[warp_id][lane_id] = extent_i.x;
            shared_i_extent_y[warp_id][lane_id] = extent_i.y;
            shared_i_extent_z[warp_id][lane_id] = extent_i.z;
            shared_i_signatures[warp_id][lane_id] =
                has_molecule_metadata ? cluster_molecule_signatures[cluster_i]
                                      : 0ull;
        }
        else
        {
            shared_i_local_masks[warp_id][lane_id] = 0u;
            shared_i_center_x[warp_id][lane_id] = 0.0f;
            shared_i_center_y[warp_id][lane_id] = 0.0f;
            shared_i_center_z[warp_id][lane_id] = 0.0f;
            shared_i_extent_x[warp_id][lane_id] = 0.0f;
            shared_i_extent_y[warp_id][lane_id] = 0.0f;
            shared_i_extent_z[warp_id][lane_id] = 0.0f;
            shared_i_signatures[warp_id][lane_id] = 0ull;
        }
    }
    for (int atom_slot = lane_id;
         atom_slot < kClusteredMaxSuperClusterClusters * kClusteredClusterSize;
         atom_slot += warpSize)
    {
        const int i_local = atom_slot / kClusteredClusterSize;
        const int atom_lane = atom_slot % kClusteredClusterSize;
        const int cluster_i = cluster_i_start + i_local;
        if (cluster_i < cluster_i_end)
        {
            const int sorted_atom_i = cluster_offsets[cluster_i] + atom_lane;
            shared_i_atom_ids[warp_id][i_local][atom_lane] =
                permutation[sorted_atom_i];
            shared_i_molecule_ids[warp_id][i_local][atom_lane] =
                has_molecule_metadata
                    ? cluster_molecule_ids[cluster_i * kClusteredClusterSize +
                                           atom_lane]
                    : -1;
        }
        else
        {
            shared_i_atom_ids[warp_id][i_local][atom_lane] = -1;
            shared_i_molecule_ids[warp_id][i_local][atom_lane] = -1;
        }
    }
    __syncwarp();
    if (fixed_shift_candidates)
    {
        if (lane_id == 0)
        {
            shared_shift_jm_in_group[warp_id][0] = 0u;
            if (sci_shift_flags[sci_shift_base] != 0)
            {
                const int write_sci = sci_shift_offsets[sci_shift_base];
                const int cjpacked_begin = cjpacked_group_offsets[sci_shift_base];
                const int cjpacked_end =
                    cjpacked_begin + cjpacked_group_counts[sci_shift_base];
                nbnxm_sci[write_sci] = {
                    super_i, fixed_shift_id, cjpacked_begin, cjpacked_end};
                if (build_gmxpacked_direct && gmxpacked_sci != NULL)
                {
                    gmxpacked_sci[write_sci] = {
                        super_i, fixed_shift_id, cjpacked_begin, cjpacked_end};
                }
                shared_shift_write_packed[warp_id][0] = cjpacked_begin;
                shared_shift_write_exclusion[warp_id][0] =
                    exclusion_offsets[sci_shift_base];
            }
            else
            {
                shared_shift_write_packed[warp_id][0] = -1;
                shared_shift_write_exclusion[warp_id][0] = 0;
            }
        }
    }
    else if (lane_id < kClusteredShiftCount)
    {
        const int sci_shift = sci_shift_base + lane_id;
        shared_shift_jm_in_group[warp_id][lane_id] = 0u;
        if (sci_shift_flags[sci_shift] != 0)
        {
            const int write_sci = sci_shift_offsets[sci_shift];
            const int cjpacked_begin = cjpacked_group_offsets[sci_shift];
            const int cjpacked_end =
                cjpacked_begin + cjpacked_group_counts[sci_shift];
            nbnxm_sci[write_sci] = {
                super_i, lane_id, cjpacked_begin, cjpacked_end};
            if (build_gmxpacked_direct && gmxpacked_sci != NULL)
            {
                gmxpacked_sci[write_sci] = {
                    super_i, lane_id, cjpacked_begin, cjpacked_end};
            }
            shared_shift_write_packed[warp_id][lane_id] = cjpacked_begin;
            shared_shift_write_exclusion[warp_id][lane_id] =
                exclusion_offsets[sci_shift];
        }
        else
        {
            shared_shift_write_packed[warp_id][lane_id] = -1;
            shared_shift_write_exclusion[warp_id][lane_id] = 0;
        }
    }
    __syncwarp();

    // Candidate leaves arrive in leaf/SFC order, so a monotonic end marker
    // removes duplicate cluster visits when a globally packed cluster straddles
    // multiple leaves.
    int processed_cluster_end = 0;
    const int active_cluster_count = cluster_i_end - cluster_i_start;
    const unsigned int active_i_lane_mask =
        active_cluster_count > 0
            ? ((1u << static_cast<unsigned int>(active_cluster_count)) - 1u)
            : 0u;

    for (int candidate_idx = candidate_leaf_offsets[candidate_sci];
         candidate_idx < candidate_leaf_offsets[candidate_sci + 1];
         candidate_idx += 1)
    {
        int leaf_j = 0;
        int cluster_j_start = 0;
        int cluster_j_end = 0;
        if (lane_id == 0)
        {
            leaf_j = candidate_leaf_ids[candidate_idx];
            cluster_j_start = leaf_cluster_starts[leaf_j];
            cluster_j_end = leaf_cluster_ends[leaf_j];
        }
        leaf_j = deviceShfl(FULL_MASK, leaf_j, 0, warpSize);
        cluster_j_start = deviceShfl(FULL_MASK, cluster_j_start, 0, warpSize);
        cluster_j_end = deviceShfl(FULL_MASK, cluster_j_end, 0, warpSize);
        const int leaf_mask_base =
            candidate_leaf_reach_masks != NULL
                ? candidate_idx * candidate_leaf_cluster_stride
                : 0;

        const int deduped_cluster_j_start =
            IntMax(cluster_j_start, processed_cluster_end);
        for (int cluster_j = deduped_cluster_j_start; cluster_j < cluster_j_end;
             cluster_j += 1)
        {
            unsigned int precomputed_i_mask = 0u;
            if (candidate_leaf_reach_masks != NULL)
            {
                if (lane_id == 0)
                {
                    precomputed_i_mask =
                        candidate_leaf_reach_masks[leaf_mask_base +
                                                   (cluster_j -
                                                    cluster_j_start)];
                }
                precomputed_i_mask =
                    deviceShfl(FULL_MASK, precomputed_i_mask, 0, warpSize) &
                    active_i_lane_mask;
                if (precomputed_i_mask == 0u)
                {
                    continue;
                }
            }
            unsigned int valid_mask_j = 0u;
            unsigned int local_mask_j = 0u;
            VECTOR center_j = {0.0f, 0.0f, 0.0f};
            VECTOR extent_j = {0.0f, 0.0f, 0.0f};
            int super_j = 0;
            uint64_t signature_j = 0ull;
            if (lane_id == 0)
            {
                valid_mask_j = cluster_valid_masks[cluster_j];
                local_mask_j = cluster_local_masks[cluster_j];
                if (valid_mask_j != 0u)
                {
                    super_j = cluster_to_supercluster[cluster_j];
                    if (!(local_mask_j != 0u && super_j < super_i))
                    {
                        center_j = cluster_centers[cluster_j];
                        extent_j = cluster_extents[cluster_j];
                    }
                    if (has_molecule_metadata)
                    {
                        signature_j = cluster_molecule_signatures[cluster_j];
                    }
                }
            }
            valid_mask_j = deviceShfl(FULL_MASK, valid_mask_j, 0, warpSize);
            local_mask_j = deviceShfl(FULL_MASK, local_mask_j, 0, warpSize);
            super_j = deviceShfl(FULL_MASK, super_j, 0, warpSize);
            center_j = Clustered_Warp_Broadcast_Vector(center_j, 0);
            extent_j = Clustered_Warp_Broadcast_Vector(extent_j, 0);
            signature_j = Broadcast_Clustered_Warp_U64(signature_j, 0);

            if (valid_mask_j == 0u)
            {
                continue;
            }
            if (local_mask_j != 0u && super_j < super_i)
            {
                continue;
            }
            if (lane_id == 0)
            {
                shared_j_signature[warp_id] = signature_j;
            }
            int pair_shift_id = -1;
            bool exclusion_candidate = false;
            if (lane_id < active_cluster_count)
            {
                const int i_local = lane_id;
                if (candidate_leaf_reach_masks != NULL)
                {
                    if ((precomputed_i_mask &
                         (1u << static_cast<unsigned int>(i_local))) != 0u)
                    {
                        pair_shift_id = fixed_shift_id;
                        exclusion_candidate =
                            !has_molecule_metadata ||
                            (shared_i_signatures[warp_id][i_local] &
                             signature_j) != 0ull;
                    }
                }
                else
                {
                    const unsigned int local_mask_i =
                        shared_i_local_masks[warp_id][i_local];
                    if (local_mask_i != 0u)
                    {
                        const VECTOR center_i = {
                            shared_i_center_x[warp_id][i_local],
                            shared_i_center_y[warp_id][i_local],
                            shared_i_center_z[warp_id][i_local]};
                        const VECTOR extent_i = {
                            shared_i_extent_x[warp_id][i_local],
                            shared_i_extent_y[warp_id][i_local],
                            shared_i_extent_z[warp_id][i_local]};
                        if (fixed_shift_candidates)
                        {
                            pair_shift_id = fixed_shift_id;
                        }
                        else
                        {
                            pair_shift_id = Determine_Cluster_Pair_Shift_Id(
                                center_i, center_j, rcell);
                        }
                        if (pair_shift_id == kClusteredCentralShiftId &&
                            cluster_j >= cluster_i_start &&
                            cluster_j < cluster_i_end &&
                            (cluster_i_start + i_local) > cluster_j)
                        {
                            pair_shift_id = -1;
                        }
                        else if (pair_shift_id >= 0 &&
                                 !Cluster_Aabb_Overlaps_Shifted(
                                     center_i, extent_i, center_j, extent_j,
                                     cutoff,
                                     fixed_shift_candidates
                                         ? fixed_shift_vec
                                         : Shift_Vector_From_Id(pair_shift_id,
                                                                cell)))
                        {
                            pair_shift_id = -1;
                        }
                        exclusion_candidate =
                            pair_shift_id >= 0 &&
                            (!has_molecule_metadata ||
                             (shared_i_signatures[warp_id][i_local] &
                              signature_j) != 0ull);
                    }
                }
            }

            const unsigned int active_pair_lane_mask =
                candidate_leaf_reach_masks != NULL
                    ? precomputed_i_mask
                    : __ballot_sync(FULL_MASK,
                                    lane_id < active_cluster_count &&
                                        pair_shift_id >= 0);
            if (active_pair_lane_mask == 0u)
            {
                continue;
            }
            const unsigned int exclusion_candidate_lane_mask =
                __ballot_sync(FULL_MASK,
                              lane_id < active_cluster_count &&
                                  exclusion_candidate);

            const bool need_j_cached_atoms =
                exclusion_candidate_lane_mask != 0u;
            if (need_j_cached_atoms)
            {
                if (lane_id < kClusteredClusterSize)
                {
                    if ((valid_mask_j & (1u << lane_id)) != 0u)
                    {
                        const int sorted_atom_j =
                            cluster_offsets[cluster_j] + lane_id;
                        shared_j_atom_ids[warp_id][lane_id] =
                            permutation[sorted_atom_j];
                        shared_j_molecule_ids[warp_id][lane_id] =
                            has_molecule_metadata
                                ? cluster_molecule_ids[cluster_j *
                                                           kClusteredClusterSize +
                                                       lane_id]
                                : -1;
                    }
                    else
                    {
                        shared_j_atom_ids[warp_id][lane_id] = -1;
                        shared_j_molecule_ids[warp_id][lane_id] = -1;
                    }
                }
                __syncwarp();
            }

            unsigned int remaining_lane_mask = active_pair_lane_mask;
            while (remaining_lane_mask != 0u)
            {
                const int leader_lane =
                    __ffs(static_cast<int>(remaining_lane_mask)) - 1;
                const int group_shift_id =
                    deviceShfl(FULL_MASK, pair_shift_id, leader_lane, warpSize);
                const unsigned int group_lane_mask =
                    __ballot_sync(FULL_MASK,
                                  lane_id < active_cluster_count &&
                                      pair_shift_id == group_shift_id);
                remaining_lane_mask &= ~group_lane_mask;
                if (lane_id == leader_lane)
                {
                    const int output_shift_idx =
                        fixed_shift_candidates ? 0 : group_shift_id;
                    const int write_packed =
                        shared_shift_write_packed[warp_id][output_shift_idx];
                    if (write_packed >= 0)
                    {
                        unsigned long long
                            exclusion_masks[kClusteredMaxSuperClusterClusters] = {};
                        unsigned int remaining_i =
                            exclusion_candidate_lane_mask &
                            (group_lane_mask & active_i_lane_mask);
                        while (remaining_i != 0u)
                        {
                            const int i_local =
                                __ffs(static_cast<int>(remaining_i)) - 1;
                            remaining_i &= (remaining_i - 1u);
                            const unsigned int local_mask_i =
                                shared_i_local_masks[warp_id][i_local];
                            exclusion_masks[i_local] =
                                Build_Exclusion_Mask_From_Cached_Atoms(
                                    shared_i_atom_ids[warp_id][i_local],
                                    shared_j_atom_ids[warp_id],
                                    shared_i_molecule_ids[warp_id][i_local],
                                    shared_j_molecule_ids[warp_id],
                                    shared_i_signatures[warp_id][i_local],
                                    shared_j_signature[warp_id],
                                    has_molecule_metadata, local_mask_i,
                                    valid_mask_j, cluster_size,
                                    local_atom_numbers, excluded_list_start,
                                    excluded_list, excluded_numbers);
                        }
                        int jm_in_group =
                            static_cast<int>(
                                shared_shift_jm_in_group[warp_id][output_shift_idx]);
                        if (jm_in_group == 0)
                        {
                            nbnxm_cjpacked[write_packed] =
                                Make_Empty_Clustered_CjPacked();
                            if (build_gmxpacked_direct &&
                                gmxpacked_cjpacked != NULL)
                            {
                                gmxpacked_cjpacked[write_packed] =
                                    Make_Empty_Gmxpacked_CjPacked();
                            }
                        }
                        Record_Clustered_Early_Record_Analysis(
                            super_i,
                            fixed_shift_candidates ? fixed_shift_id
                                                   : group_shift_id,
                            cluster_j,
                            group_lane_mask & active_i_lane_mask, valid_mask_j,
                            local_mask_j, exclusion_masks,
                            early_record_analysis_capacity,
                            early_record_analysis_raw_count,
                            early_record_analysis_entries);
                        Append_Record_To_Clustered_CjPacked(
                            nbnxm_cjpacked + write_packed, jm_in_group, cluster_j,
                            valid_mask_j, group_lane_mask & active_i_lane_mask,
                            exclusion_masks,
                            &shared_shift_write_exclusion[warp_id][output_shift_idx],
                            exclusion_mask_pool, dedup_exclusion_masks);
                        if (build_gmxpacked_direct &&
                            gmxpacked_cjpacked != NULL &&
                            gmxpacked_exclusions != NULL)
                        {
                            Append_Record_To_Gmxpacked_CjPacked(
                                gmxpacked_cjpacked + write_packed, write_packed,
                                jm_in_group, cluster_i_start, cluster_numbers,
                                cluster_j, group_shift_id, valid_mask_j,
                                local_mask_j, group_lane_mask & active_i_lane_mask,
                                shared_i_local_masks[warp_id], exclusion_masks,
                                gmxpacked_exclusions);
                        }
                        jm_in_group += 1;
                        if (jm_in_group == kClusteredMaxJGroupSize)
                        {
                            shared_shift_write_packed[warp_id][output_shift_idx] =
                                write_packed + 1;
                            jm_in_group = 0;
                        }
                        shared_shift_jm_in_group[warp_id][output_shift_idx] =
                            static_cast<unsigned char>(jm_in_group);
                    }
                }
                __syncwarp();
            }
        }
        processed_cluster_end = IntMax(processed_cluster_end, cluster_j_end);
    }
}

static __global__ void Build_Sci_Workload_Sort_Keys(
    const int sci_numbers, const LJ_CLUSTERED_SCI* sci_entries, int* sci_sort_keys)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        sci_sort_keys[sci] =
            -(sci_entries[sci].cjpacked_end - sci_entries[sci].cjpacked_begin);
    }
}

static __device__ __forceinline__ void Clustered_Copy_CjPacked_Jm(
    const LJ_CLUSTERED_CJ_PACKED& src, const int src_jm,
    LJ_CLUSTERED_CJ_PACKED* dst, const int dst_jm)
{
    dst->cj[dst_jm] = src.cj[src_jm];
#pragma unroll
    for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
    {
        const unsigned int jm_imask = Clustered_Jm_Imask(src.imei[split], src_jm);
        if (jm_imask != 0u)
        {
            dst->imei[split].imask |=
                jm_imask << Clustered_Jm_Imask_Shift(dst_jm);
        }
#pragma unroll
        for (int i_local = 0; i_local < kClusteredMaxSuperClusterClusters;
             i_local += 1)
        {
            Clustered_Exclusion_Index_Ref(dst->imei[split], dst_jm, i_local) =
                Clustered_Exclusion_Index(src.imei[split], src_jm, i_local);
        }
    }
}

static __host__ __device__ __forceinline__ int&
Clustered_J_Entry_Exclusion_Index_Ref(LJ_CLUSTERED_J_ENTRY& entry,
                                      int split, int i_local)
{
    return entry.excl_ind[split * kClusteredSuperClusterClusters + i_local];
}

static __host__ __device__ __forceinline__ int
Clustered_J_Entry_Exclusion_Index(const LJ_CLUSTERED_J_ENTRY& entry, int split,
                                  int i_local)
{
    return entry.excl_ind[split * kClusteredSuperClusterClusters + i_local];
}

static __global__ void Count_Cj_Entries_Per_Sci(
    const int sci_numbers, const LJ_CLUSTERED_SCI* sci_entries,
    const LJ_CLUSTERED_CJ_PACKED* nbnxm_cjpacked, int* entry_counts)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const LJ_CLUSTERED_SCI sci_entry = sci_entries[sci];
        int entry_count = 0;
        for (int packed_idx = sci_entry.cjpacked_begin;
             packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
        {
            const LJ_CLUSTERED_CJ_PACKED& packed = nbnxm_cjpacked[packed_idx];
            for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
            {
                const int cluster_j = packed.cj[jm];
                const unsigned int combined_imask =
                    Clustered_Jm_Imask(packed.imei[0], jm) |
                    Clustered_Jm_Imask(packed.imei[1], jm);
                if (cluster_j >= 0 && combined_imask != 0u)
                {
                    entry_count += 1;
                }
            }
        }
        entry_counts[sci] = entry_count;
    }
}

static __global__ void Fill_J_Entries_From_Payload(
    const int sci_numbers, const LJ_CLUSTERED_SCI* sci_entries,
    const LJ_CLUSTERED_CJ_PACKED* nbnxm_cjpacked, const int* entry_offsets,
    LJ_CLUSTERED_J_ENTRY* j_entries)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const LJ_CLUSTERED_SCI sci_entry = sci_entries[sci];
        const int write_base = entry_offsets[sci];
        int write_count = 0;
        for (int packed_idx = sci_entry.cjpacked_begin;
             packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
        {
            const LJ_CLUSTERED_CJ_PACKED& packed = nbnxm_cjpacked[packed_idx];
            for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
            {
                const int cluster_j = packed.cj[jm];
                if (cluster_j < 0)
                {
                    continue;
                }
                LJ_CLUSTERED_J_ENTRY entry = {};
                entry.supercluster_id = sci_entry.supercluster_id;
                entry.shift_id = sci_entry.shift_id;
                entry.cluster_j = cluster_j;
                unsigned int combined_imask = 0u;
                for (int split = 0; split < kClusteredWarpSplitCount;
                     split += 1)
                {
                    entry.imask[split] =
                        Clustered_Jm_Imask(packed.imei[split], jm);
                    combined_imask |= entry.imask[split];
                    for (int i_local = 0;
                         i_local < kClusteredSuperClusterClusters;
                         i_local += 1)
                    {
                        Clustered_J_Entry_Exclusion_Index_Ref(entry, split,
                                                              i_local) =
                            Clustered_Exclusion_Index(packed.imei[split], jm,
                                                      i_local);
                    }
                }
                if (combined_imask == 0u)
                {
                    continue;
                }
                j_entries[write_base + write_count] = entry;
                write_count += 1;
            }
        }
    }
}

static __global__ void Build_J_Entry_Sort_Keys(const int entry_numbers,
                                               const LJ_CLUSTERED_J_ENTRY* j_entries,
                                               uint64_t* sort_keys)
{
    SIMPLE_DEVICE_FOR(entry_idx, entry_numbers)
    {
        const LJ_CLUSTERED_J_ENTRY& entry = j_entries[entry_idx];
        sort_keys[entry_idx] =
            (static_cast<uint64_t>(
                 static_cast<unsigned int>(entry.supercluster_id))
             << 37) |
            (static_cast<uint64_t>(static_cast<unsigned int>(entry.shift_id))
             << 32) |
            static_cast<uint64_t>(static_cast<unsigned int>(entry.cluster_j));
    }
}

static __global__ void Build_J_Entry_Block_Sort_Keys(
    const int entry_numbers, const LJ_CLUSTERED_J_ENTRY* j_entries,
    const bool central_only, uint64_t* sort_keys)
{
    SIMPLE_DEVICE_FOR(entry_idx, entry_numbers)
    {
        const LJ_CLUSTERED_J_ENTRY& entry = j_entries[entry_idx];
        const bool sort_by_j_block =
            (!central_only || entry.shift_id == kClusteredCentralShiftId)
                ? true
                : false;
        const unsigned int block_key =
            sort_by_j_block
                ? static_cast<unsigned int>(
                      entry.cluster_j / kClusteredSuperClusterClusters)
                : 0u;
        const unsigned int cluster_key =
            sort_by_j_block ? static_cast<unsigned int>(entry.cluster_j) : 0u;
        sort_keys[entry_idx] =
            (static_cast<uint64_t>(
                 static_cast<unsigned int>(entry.supercluster_id))
             << 42) |
            (static_cast<uint64_t>(static_cast<unsigned int>(entry.shift_id))
             << 37) |
            (static_cast<uint64_t>(block_key) << 20) |
            static_cast<uint64_t>(cluster_key & 0xfffffu);
    }
}

static __global__ void Gather_J_Entries_By_Index(
    const int entry_numbers, const int* entry_indices,
    const LJ_CLUSTERED_J_ENTRY* src_entries,
    LJ_CLUSTERED_J_ENTRY* dst_entries)
{
    SIMPLE_DEVICE_FOR(entry_idx, entry_numbers)
    {
        dst_entries[entry_idx] = src_entries[entry_indices[entry_idx]];
    }
}

static __global__ void Build_J_Entry_Sci_Flags(
    const int entry_numbers, const LJ_CLUSTERED_J_ENTRY* j_entries, int* sci_flags)
{
    SIMPLE_DEVICE_FOR(entry_idx, entry_numbers)
    {
        int sci_start = 0;
        if (entry_idx == 0)
        {
            sci_start = 1;
        }
        else
        {
            const LJ_CLUSTERED_J_ENTRY& current = j_entries[entry_idx];
            const LJ_CLUSTERED_J_ENTRY& previous = j_entries[entry_idx - 1];
            sci_start = (current.supercluster_id != previous.supercluster_id ||
                         current.shift_id != previous.shift_id)
                            ? 1
                            : 0;
        }
        sci_flags[entry_idx] = sci_start;
    }
}

static __global__ void Build_Reduced_J_Entry_Sort_Keys(
    const int entry_numbers, const LJ_CLUSTERED_J_ENTRY* j_entries,
    uint64_t* sort_keys)
{
    SIMPLE_DEVICE_FOR(entry_idx, entry_numbers)
    {
        const LJ_CLUSTERED_J_ENTRY& entry = j_entries[entry_idx];
        sort_keys[entry_idx] =
            (static_cast<uint64_t>(
                 static_cast<unsigned int>(entry.supercluster_id))
             << 32) |
            static_cast<uint64_t>(static_cast<unsigned int>(entry.cluster_j));
    }
}

static __global__ void Build_Reduced_J_Entry_Flags(
    const int entry_numbers, const LJ_CLUSTERED_J_ENTRY* j_entries,
    int* reduced_entry_flags)
{
    SIMPLE_DEVICE_FOR(entry_idx, entry_numbers)
    {
        int entry_start = 0;
        if (entry_idx == 0)
        {
            entry_start = 1;
        }
        else
        {
            const LJ_CLUSTERED_J_ENTRY& current = j_entries[entry_idx];
            const LJ_CLUSTERED_J_ENTRY& previous = j_entries[entry_idx - 1];
            entry_start =
                (current.supercluster_id != previous.supercluster_id ||
                 current.cluster_j != previous.cluster_j)
                    ? 1
                    : 0;
        }
        reduced_entry_flags[entry_idx] = entry_start;
    }
}

static __global__ void Build_Reduced_Sci_Flags_From_J_Entry_Starts(
    const int reduced_entry_numbers, const int* reduced_entry_starts,
    const LJ_CLUSTERED_J_ENTRY* sorted_j_entries, int* sci_flags)
{
    SIMPLE_DEVICE_FOR(entry_idx, reduced_entry_numbers)
    {
        int sci_start = 0;
        if (entry_idx == 0)
        {
            sci_start = 1;
        }
        else
        {
            const LJ_CLUSTERED_J_ENTRY& current =
                sorted_j_entries[reduced_entry_starts[entry_idx]];
            const LJ_CLUSTERED_J_ENTRY& previous =
                sorted_j_entries[reduced_entry_starts[entry_idx - 1]];
            sci_start = current.supercluster_id != previous.supercluster_id ? 1
                                                                           : 0;
        }
        sci_flags[entry_idx] = sci_start;
    }
}

static __global__ void Scatter_J_Entry_Sci_Starts(
    const int entry_numbers, const int* sci_flags, const int* sci_ids,
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

static __global__ void Build_J_Entry_Packed_Counts(
    const int sci_numbers, const int* sci_starts, int* packed_counts)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const int entry_count = sci_starts[sci + 1] - sci_starts[sci];
        packed_counts[sci] =
            (entry_count + kClusteredJGroupSize - 1) / kClusteredJGroupSize;
    }
}

static __global__ void Write_Repacked_Sci_From_J_Entries(
    const int sci_numbers, const int* sci_starts, const int* packed_offsets,
    const LJ_CLUSTERED_J_ENTRY* j_entries, LJ_CLUSTERED_SCI* sci_entries)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const int begin = sci_starts[sci];
        const int end = sci_starts[sci + 1];
        if (begin >= end)
        {
            return;
        }
        const LJ_CLUSTERED_J_ENTRY& first = j_entries[begin];
        sci_entries[sci] = {
            first.supercluster_id, first.shift_id, packed_offsets[sci],
            packed_offsets[sci + 1]};
    }
}

static __global__ void Initialize_CjPacked_Array(
    const int cjpacked_numbers, LJ_CLUSTERED_CJ_PACKED* nbnxm_cjpacked)
{
    SIMPLE_DEVICE_FOR(packed_idx, cjpacked_numbers)
    {
        nbnxm_cjpacked[packed_idx] = Make_Empty_Clustered_CjPacked();
    }
}

static __global__ void Fill_Repacked_CjPacked_From_J_Entries(
    const int sci_numbers, const int* sci_starts, const int* packed_offsets,
    const LJ_CLUSTERED_J_ENTRY* j_entries,
    LJ_CLUSTERED_CJ_PACKED* nbnxm_cjpacked)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const int begin = sci_starts[sci];
        const int end = sci_starts[sci + 1];
        const int packed_begin = packed_offsets[sci];
        for (int entry_idx = begin; entry_idx < end; entry_idx += 1)
        {
            const LJ_CLUSTERED_J_ENTRY& entry = j_entries[entry_idx];
            const int local_entry = entry_idx - begin;
            const int packed_idx =
                packed_begin + local_entry / kClusteredJGroupSize;
            const int jm = local_entry % kClusteredJGroupSize;
            LJ_CLUSTERED_CJ_PACKED& packed = nbnxm_cjpacked[packed_idx];
            packed.cj[jm] = entry.cluster_j;
            for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
            {
                const unsigned int imask = entry.imask[split];
                if (imask == 0u)
                {
                    continue;
                }
                packed.imei[split].imask |=
                    imask << Clustered_Jm_Imask_Shift(jm);
                for (int i_local = 0; i_local < kClusteredSuperClusterClusters;
                     i_local += 1)
                {
                    Clustered_Exclusion_Index_Ref(packed.imei[split], jm,
                                                  i_local) =
                        Clustered_J_Entry_Exclusion_Index(entry, split,
                                                          i_local);
                }
            }
        }
    }
}

#ifndef USE_CPU
struct ClusteredPayloadTraceStats
{
    long long live_j_entries = 0;
    long long split_records = 0;
    long long imask_bits = 0;
    long long exclusion_refs = 0;
    long long central_entries = 0;
    long long shifted_entries = 0;
    long long cjpacked_span_sum = 0;
    int overlapping_sci = 0;
    int nonmonotonic_sci = 0;
    int max_cjpacked_end = 0;
    int first_overlap_sci = -1;
    int first_overlap_begin = 0;
    int first_overlap_end = 0;
    int first_overlap_prev_end = 0;
};

static ClusteredPayloadTraceStats Analyze_J_Entries_On_Host(
    const std::vector<LJ_CLUSTERED_J_ENTRY>& entries)
{
    ClusteredPayloadTraceStats stats = {};
    for (const LJ_CLUSTERED_J_ENTRY& entry : entries)
    {
        bool entry_live = false;
        for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
        {
            const unsigned int imask = entry.imask[split];
            if (imask == 0u)
            {
                continue;
            }
            entry_live = true;
            stats.split_records += 1;
            stats.imask_bits += __builtin_popcount(imask);
            for (int i_local = 0; i_local < kClusteredSuperClusterClusters;
                 i_local += 1)
            {
                if ((imask & (1u << static_cast<unsigned int>(i_local))) == 0u)
                {
                    continue;
                }
                if (Clustered_J_Entry_Exclusion_Index(entry, split, i_local) >=
                    0)
                {
                    stats.exclusion_refs += 1;
                }
            }
        }
        if (entry_live)
        {
            stats.live_j_entries += 1;
            if (entry.shift_id == kClusteredCentralShiftId)
            {
                stats.central_entries += 1;
            }
            else
            {
                stats.shifted_entries += 1;
            }
        }
    }
    return stats;
}

static ClusteredPayloadTraceStats Analyze_CjPacked_On_Host(
    const std::vector<LJ_CLUSTERED_SCI>& sci_entries,
    const std::vector<LJ_CLUSTERED_CJ_PACKED>& packed_entries)
{
    ClusteredPayloadTraceStats stats = {};
    int previous_end = 0;
    bool first_sci = true;
    for (size_t sci_idx = 0; sci_idx < sci_entries.size(); sci_idx += 1)
    {
        const LJ_CLUSTERED_SCI& sci_entry = sci_entries[sci_idx];
        stats.cjpacked_span_sum +=
            static_cast<long long>(sci_entry.cjpacked_end - sci_entry.cjpacked_begin);
        stats.max_cjpacked_end =
            IntMax(stats.max_cjpacked_end, sci_entry.cjpacked_end);
        if (!first_sci)
        {
            if (sci_entry.cjpacked_begin < previous_end)
            {
                stats.overlapping_sci += 1;
                if (stats.first_overlap_sci < 0)
                {
                    stats.first_overlap_sci = static_cast<int>(sci_idx);
                    stats.first_overlap_begin = sci_entry.cjpacked_begin;
                    stats.first_overlap_end = sci_entry.cjpacked_end;
                    stats.first_overlap_prev_end = previous_end;
                }
            }
            if (sci_entry.cjpacked_begin < 0 ||
                sci_entry.cjpacked_end < sci_entry.cjpacked_begin)
            {
                stats.nonmonotonic_sci += 1;
            }
        }
        else
        {
            if (sci_entry.cjpacked_begin < 0 ||
                sci_entry.cjpacked_end < sci_entry.cjpacked_begin)
            {
                stats.nonmonotonic_sci += 1;
            }
            first_sci = false;
        }
        previous_end = sci_entry.cjpacked_end;
        for (int packed_idx = sci_entry.cjpacked_begin;
             packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
        {
            const LJ_CLUSTERED_CJ_PACKED& packed =
                packed_entries[static_cast<size_t>(packed_idx)];
            for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
            {
                if (packed.cj[jm] < 0)
                {
                    continue;
                }
                bool jm_live = false;
                for (int split = 0; split < kClusteredWarpSplitCount;
                     split += 1)
                {
                    const unsigned int imask =
                        Clustered_Jm_Imask(packed.imei[split], jm);
                    if (imask == 0u)
                    {
                        continue;
                    }
                    jm_live = true;
                    stats.split_records += 1;
                    stats.imask_bits += __builtin_popcount(imask);
                    for (int i_local = 0;
                         i_local < kClusteredSuperClusterClusters;
                         i_local += 1)
                    {
                        if ((imask &
                             (1u << static_cast<unsigned int>(i_local))) == 0u)
                        {
                            continue;
                        }
                        if (Clustered_Exclusion_Index(packed.imei[split], jm,
                                                      i_local) >= 0)
                        {
                            stats.exclusion_refs += 1;
                        }
                    }
                }
                if (jm_live)
                {
                    stats.live_j_entries += 1;
                    if (sci_entry.shift_id == kClusteredCentralShiftId)
                    {
                        stats.central_entries += 1;
                    }
                    else
                    {
                        stats.shifted_entries += 1;
                    }
                }
            }
        }
    }
    return stats;
}

static std::vector<LJ_CLUSTERED_J_ENTRY> Build_J_Entries_On_Host(
    const std::vector<LJ_CLUSTERED_SCI>& sci_entries,
    const std::vector<LJ_CLUSTERED_CJ_PACKED>& packed_entries)
{
    std::vector<LJ_CLUSTERED_J_ENTRY> entries;
    for (const LJ_CLUSTERED_SCI& sci_entry : sci_entries)
    {
        for (int packed_idx = sci_entry.cjpacked_begin;
             packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
        {
            const LJ_CLUSTERED_CJ_PACKED& packed =
                packed_entries[static_cast<size_t>(packed_idx)];
            for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
            {
                const int cluster_j = packed.cj[jm];
                if (cluster_j < 0)
                {
                    continue;
                }
                LJ_CLUSTERED_J_ENTRY entry = {};
                entry.supercluster_id = sci_entry.supercluster_id;
                entry.shift_id = sci_entry.shift_id;
                entry.cluster_j = cluster_j;
                unsigned int combined_imask = 0u;
                for (int split = 0; split < kClusteredWarpSplitCount;
                     split += 1)
                {
                    entry.imask[split] =
                        Clustered_Jm_Imask(packed.imei[split], jm);
                    combined_imask |= entry.imask[split];
                    for (int i_local = 0;
                         i_local < kClusteredSuperClusterClusters;
                         i_local += 1)
                    {
                        Clustered_J_Entry_Exclusion_Index_Ref(entry, split,
                                                              i_local) =
                            Clustered_Exclusion_Index(packed.imei[split], jm,
                                                      i_local);
                    }
                }
                if (combined_imask != 0u)
                {
                    entries.push_back(entry);
                }
            }
        }
    }
    return entries;
}

static void Rebuild_Payload_From_J_Entries_On_Host(
    const std::vector<LJ_CLUSTERED_J_ENTRY>& input_entries,
    const bool block_sort_central_only,
    std::vector<LJ_CLUSTERED_SCI>* output_sci,
    std::vector<LJ_CLUSTERED_CJ_PACKED>* output_cjpacked)
{
    output_sci->clear();
    output_cjpacked->clear();
    if (input_entries.empty())
    {
        return;
    }
    std::vector<LJ_CLUSTERED_J_ENTRY> entries = input_entries;
    std::stable_sort(
        entries.begin(), entries.end(),
        [block_sort_central_only](const LJ_CLUSTERED_J_ENTRY& lhs,
                                  const LJ_CLUSTERED_J_ENTRY& rhs)
        {
            if (lhs.supercluster_id != rhs.supercluster_id)
            {
                return lhs.supercluster_id < rhs.supercluster_id;
            }
            if (lhs.shift_id != rhs.shift_id)
            {
                return lhs.shift_id < rhs.shift_id;
            }
            const unsigned int lhs_block =
                (!block_sort_central_only ||
                 lhs.shift_id == kClusteredCentralShiftId)
                    ? static_cast<unsigned int>(
                          lhs.cluster_j / kClusteredSuperClusterClusters)
                    : 0u;
            const unsigned int rhs_block =
                (!block_sort_central_only ||
                 rhs.shift_id == kClusteredCentralShiftId)
                    ? static_cast<unsigned int>(
                          rhs.cluster_j / kClusteredSuperClusterClusters)
                    : 0u;
            return lhs_block < rhs_block;
        });
    size_t begin = 0;
    while (begin < entries.size())
    {
        size_t end = begin + 1;
        while (end < entries.size() &&
               entries[end].supercluster_id == entries[begin].supercluster_id &&
               entries[end].shift_id == entries[begin].shift_id)
        {
            end += 1;
        }
        const int packed_begin = static_cast<int>(output_cjpacked->size());
        const int packed_count = static_cast<int>(
            (end - begin + kClusteredJGroupSize - 1) / kClusteredJGroupSize);
        output_cjpacked->resize(output_cjpacked->size() +
                                static_cast<size_t>(packed_count));
        for (int packed_local = 0; packed_local < packed_count; packed_local += 1)
        {
            (*output_cjpacked)[static_cast<size_t>(packed_begin + packed_local)] =
                Make_Empty_Clustered_CjPacked();
        }
        for (size_t entry_idx = begin; entry_idx < end; entry_idx += 1)
        {
            const LJ_CLUSTERED_J_ENTRY& entry = entries[entry_idx];
            const int local_entry = static_cast<int>(entry_idx - begin);
            const int packed_idx = packed_begin + local_entry / kClusteredJGroupSize;
            const int jm = local_entry % kClusteredJGroupSize;
            LJ_CLUSTERED_CJ_PACKED& packed =
                (*output_cjpacked)[static_cast<size_t>(packed_idx)];
            packed.cj[jm] = entry.cluster_j;
            for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
            {
                const unsigned int imask = entry.imask[split];
                if (imask == 0u)
                {
                    continue;
                }
                packed.imei[split].imask |=
                    imask << Clustered_Jm_Imask_Shift(jm);
                for (int i_local = 0; i_local < kClusteredSuperClusterClusters;
                     i_local += 1)
                {
                    Clustered_Exclusion_Index_Ref(packed.imei[split], jm,
                                                  i_local) =
                        Clustered_J_Entry_Exclusion_Index(entry, split, i_local);
                }
            }
        }
        output_sci->push_back({entries[begin].supercluster_id,
                               entries[begin].shift_id, packed_begin,
                               packed_begin + packed_count});
        begin = end;
    }
}

static void Trace_J_Entry_Rebuild_Payload(
    const char* stage, int sci_numbers, const LJ_CLUSTERED_SCI* d_sci_entries,
    int cjpacked_numbers, const LJ_CLUSTERED_CJ_PACKED* d_cjpacked,
    int jentry_numbers = 0, const LJ_CLUSTERED_J_ENTRY* d_j_entries = NULL)
{
    ClusteredPayloadTraceStats stats = {};
    if (jentry_numbers > 0 && d_j_entries != NULL)
    {
        std::vector<LJ_CLUSTERED_J_ENTRY> j_entries(
            static_cast<size_t>(jentry_numbers));
        deviceMemcpy(j_entries.data(), d_j_entries,
                     sizeof(LJ_CLUSTERED_J_ENTRY) * jentry_numbers,
                     deviceMemcpyDeviceToHost);
        stats = Analyze_J_Entries_On_Host(j_entries);
        fprintf(stderr,
                "[clustered jentry rebuild stats] stage=%s kind=jentry "
                "entries=%d live=%lld split_records=%lld imask_bits=%lld "
                "excl_refs=%lld central=%lld shifted=%lld\n",
                stage, jentry_numbers, stats.live_j_entries, stats.split_records,
                stats.imask_bits, stats.exclusion_refs, stats.central_entries,
                stats.shifted_entries);
    }
    else if (sci_numbers > 0 && cjpacked_numbers > 0 && d_sci_entries != NULL &&
             d_cjpacked != NULL)
    {
        std::vector<LJ_CLUSTERED_SCI> sci_entries(
            static_cast<size_t>(sci_numbers));
        std::vector<LJ_CLUSTERED_CJ_PACKED> cjpacked(
            static_cast<size_t>(cjpacked_numbers));
        deviceMemcpy(sci_entries.data(), d_sci_entries,
                     sizeof(LJ_CLUSTERED_SCI) * sci_numbers,
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(cjpacked.data(), d_cjpacked,
                     sizeof(LJ_CLUSTERED_CJ_PACKED) * cjpacked_numbers,
                     deviceMemcpyDeviceToHost);
        stats = Analyze_CjPacked_On_Host(sci_entries, cjpacked);
        fprintf(stderr,
                "[clustered jentry rebuild stats] stage=%s kind=payload "
                "sci=%d cjpacked=%d live=%lld split_records=%lld "
                "imask_bits=%lld excl_refs=%lld central=%lld shifted=%lld "
                "span=%lld overlap=%d nonmono=%d max_end=%d "
                "first_overlap=(%d,%d,%d,%d)\n",
                stage, sci_numbers, cjpacked_numbers, stats.live_j_entries,
                stats.split_records, stats.imask_bits, stats.exclusion_refs,
                stats.central_entries, stats.shifted_entries,
                stats.cjpacked_span_sum, stats.overlapping_sci,
                stats.nonmonotonic_sci, stats.max_cjpacked_end,
                stats.first_overlap_sci, stats.first_overlap_begin,
                stats.first_overlap_end, stats.first_overlap_prev_end);
    }
    fflush(stderr);
}

static void Trace_J_Entry_Host_Repack(
    const char* stage, const std::vector<LJ_CLUSTERED_SCI>& sci_entries,
    const std::vector<LJ_CLUSTERED_CJ_PACKED>& cjpacked_entries,
    const bool block_sort_central_only)
{
    const std::vector<LJ_CLUSTERED_J_ENTRY> j_entries =
        Build_J_Entries_On_Host(sci_entries, cjpacked_entries);
    std::vector<LJ_CLUSTERED_SCI> rebuilt_sci;
    std::vector<LJ_CLUSTERED_CJ_PACKED> rebuilt_cjpacked;
    Rebuild_Payload_From_J_Entries_On_Host(j_entries, block_sort_central_only,
                                           &rebuilt_sci, &rebuilt_cjpacked);
    const ClusteredPayloadTraceStats stats =
        Analyze_CjPacked_On_Host(rebuilt_sci, rebuilt_cjpacked);
    fprintf(stderr,
            "[clustered jentry rebuild stats] stage=%s kind=host-repack "
            "sci=%zu cjpacked=%zu live=%lld split_records=%lld "
            "imask_bits=%lld excl_refs=%lld central=%lld shifted=%lld "
            "span=%lld overlap=%d nonmono=%d max_end=%d "
            "first_overlap=(%d,%d,%d,%d) block_sort=%d\n",
            stage, rebuilt_sci.size(), rebuilt_cjpacked.size(),
            stats.live_j_entries, stats.split_records, stats.imask_bits,
            stats.exclusion_refs, stats.central_entries, stats.shifted_entries,
            stats.cjpacked_span_sum, stats.overlapping_sci,
            stats.nonmonotonic_sci, stats.max_cjpacked_end,
            stats.first_overlap_sci, stats.first_overlap_begin,
            stats.first_overlap_end, stats.first_overlap_prev_end,
            block_sort_central_only ? 1 : 0);
    fflush(stderr);
}

static void Trace_Sci_Window_On_Host(
    const char* stage, const std::vector<LJ_CLUSTERED_SCI>& sci_entries,
    int focus_sci, const std::vector<int>* sci_starts = NULL,
    const std::vector<int>* packed_offsets = NULL)
{
    if (focus_sci < 0 || sci_entries.empty())
    {
        return;
    }
    const int begin =
        IntMax(0, focus_sci - 2);
    const int end =
        IntMin(static_cast<int>(sci_entries.size()), focus_sci + 3);
    fprintf(stderr,
            "[clustered jentry rebuild window] stage=%s focus=%d range=[%d,%d)\n",
            stage, focus_sci, begin, end);
    for (int sci = begin; sci < end; sci += 1)
    {
        const LJ_CLUSTERED_SCI& entry = sci_entries[static_cast<size_t>(sci)];
        fprintf(stderr,
                "  sci=%d super=%d shift=%d cj=[%d,%d)",
                sci, entry.supercluster_id, entry.shift_id, entry.cjpacked_begin,
                entry.cjpacked_end);
        if (sci_starts != NULL && sci < static_cast<int>(sci_starts->size()))
        {
            fprintf(stderr, " start=%d",
                    (*sci_starts)[static_cast<size_t>(sci)]);
        }
        if (packed_offsets != NULL &&
            sci + 1 < static_cast<int>(packed_offsets->size()))
        {
            fprintf(stderr, " packed_off=[%d,%d)",
                    (*packed_offsets)[static_cast<size_t>(sci)],
                    (*packed_offsets)[static_cast<size_t>(sci + 1)]);
        }
        fprintf(stderr, "\n");
    }
    fflush(stderr);
}
#endif

static __global__ void Normalize_CjPacked_J_Order(
    const int cjpacked_numbers, LJ_CLUSTERED_CJ_PACKED* nbnxm_cjpacked)
{
    SIMPLE_DEVICE_FOR(packed_idx, cjpacked_numbers)
    {
        const LJ_CLUSTERED_CJ_PACKED src = nbnxm_cjpacked[packed_idx];
        int order[kClusteredJGroupSize];
#pragma unroll
        for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
        {
            order[jm] = jm;
        }
#pragma unroll
        for (int i = 1; i < kClusteredJGroupSize; i += 1)
        {
            const int current = order[i];
            const int current_key =
                src.cj[current] >= 0 ? src.cj[current] : 0x7fffffff;
            int j = i - 1;
            while (j >= 0)
            {
                const int prev_key =
                    src.cj[order[j]] >= 0 ? src.cj[order[j]] : 0x7fffffff;
                if (prev_key <= current_key)
                {
                    break;
                }
                order[j + 1] = order[j];
                j -= 1;
            }
            order[j + 1] = current;
        }

        LJ_CLUSTERED_CJ_PACKED sorted = Make_Empty_Clustered_CjPacked();
#pragma unroll
        for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
        {
            const int src_jm = order[jm];
            if (src.cj[src_jm] < 0)
            {
                continue;
            }
            Clustered_Copy_CjPacked_Jm(src, src_jm, &sorted, jm);
        }
        nbnxm_cjpacked[packed_idx] = sorted;
    }
}

static __global__ void Build_CjPacked_Sort_Keys(
    const int sci_numbers, const LJ_CLUSTERED_SCI* sci_entries,
    const LJ_CLUSTERED_CJ_PACKED* nbnxm_cjpacked, uint64_t* packed_sort_keys)
{
    SIMPLE_DEVICE_FOR(sci_idx, sci_numbers)
    {
        const LJ_CLUSTERED_SCI sci = sci_entries[sci_idx];
        const uint64_t sci_key =
            static_cast<uint64_t>(static_cast<unsigned int>(sci_idx)) << 32;
        for (int packed_idx = sci.cjpacked_begin; packed_idx < sci.cjpacked_end;
             packed_idx += 1)
        {
            const int cluster_j = nbnxm_cjpacked[packed_idx].cj[0];
            const unsigned int local_key =
                cluster_j >= 0 ? static_cast<unsigned int>(cluster_j)
                               : 0xffffffffu;
            packed_sort_keys[packed_idx] = sci_key | local_key;
        }
    }
}

static __global__ void Build_Linear_Indices(const int count, int* indices)
{
    SIMPLE_DEVICE_FOR(i, count) { indices[i] = i; }
}

static __global__ void Gather_CjPacked_By_Index(
    const int cjpacked_numbers, const int* packed_indices,
    const LJ_CLUSTERED_CJ_PACKED* src_cjpacked,
    LJ_CLUSTERED_CJ_PACKED* dst_cjpacked)
{
    SIMPLE_DEVICE_FOR(packed_idx, cjpacked_numbers)
    {
        dst_cjpacked[packed_idx] =
            src_cjpacked[packed_indices[packed_idx]];
    }
}

static __global__ void Build_Forceonly_Record_Sort_Keys(
    const int sci_numbers, const LJ_CLUSTERED_SCI* sci_entries,
    const int* record_offsets, const LJ_CLUSTERED_WARP_J_RECORD* records,
    uint64_t* sort_keys)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const LJ_CLUSTERED_SCI sci_entry = sci_entries[sci];
        const int begin = record_offsets[sci];
        const int end = record_offsets[sci + 1];
        for (int record_idx = begin; record_idx < end; record_idx += 1)
        {
            const LJ_CLUSTERED_WARP_J_RECORD& record = records[record_idx];
            const unsigned int local_key =
                sci_entry.shift_id == kClusteredCentralShiftId
                    ? (static_cast<unsigned int>(record.cluster_j) << 1) |
                          static_cast<unsigned int>(
                              record.j_lane_base /
                              kClusteredSplitJClusterSize)
                    : static_cast<unsigned int>(record_idx - begin);
            sort_keys[record_idx] =
                (static_cast<uint64_t>(static_cast<unsigned int>(sci)) << 32) |
                static_cast<uint64_t>(local_key);
        }
    }
}

static __global__ void Gather_Forceonly_Records_By_Index(
    const int record_numbers, const int* record_indices,
    const LJ_CLUSTERED_WARP_J_RECORD* src_records,
    LJ_CLUSTERED_WARP_J_RECORD* dst_records)
{
    SIMPLE_DEVICE_FOR(record_idx, record_numbers)
    {
        dst_records[record_idx] = src_records[record_indices[record_idx]];
    }
}

static void Reserve_Device_U64_Buffer(int capacity, uint64_t** pointer,
                                      int* current_capacity);
static void Reserve_Device_Int_Buffer(int capacity, int** pointer,
                                      int* current_capacity);
static bool Clustered_Trace_Warp_Records_Enabled();
static int Exclusive_Scan_Counts(LJ_CLUSTER_LAYOUT* layout, int count_numbers,
                                 int* d_counts, int* d_starts);

static void Reorder_CjPacked_By_Cluster_J(LJ_CLUSTER_LAYOUT* layout)
{
#ifdef USE_CPU
    (void)layout;
#else
    if (layout == NULL || layout->sci_numbers <= 0 || layout->cjpacked_numbers <= 1 ||
        layout->d_nbnxm_sci == NULL || layout->d_nbnxm_cjpacked == NULL)
    {
        return;
    }
    Reserve_Device_U64_Buffer(layout->cjpacked_numbers, &layout->d_sort_keys,
                              &layout->sort_key_capacity);
    Reserve_Device_Int_Buffer(layout->cjpacked_numbers,
                              &layout->d_cjpacked_sort_indices,
                              &layout->cjpacked_sort_index_capacity);
    Reserve_Device_Buffer(layout->cjpacked_numbers,
                          &layout->d_cjpacked_sort_buffer,
                          &layout->cjpacked_sort_buffer_capacity);
    Launch_Device_Kernel(
        Normalize_CjPacked_J_Order,
        (layout->cjpacked_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout->cjpacked_numbers,
        layout->d_nbnxm_cjpacked);
    Launch_Device_Kernel(
        Build_CjPacked_Sort_Keys,
        (layout->sci_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout->sci_numbers,
        layout->d_nbnxm_sci, layout->d_nbnxm_cjpacked, layout->d_sort_keys);
    Launch_Device_Kernel(
        Build_Linear_Indices,
        (layout->cjpacked_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout->cjpacked_numbers,
        layout->d_cjpacked_sort_indices);
    Stable_Sort_Device_By_Key(layout, layout->cjpacked_numbers,
                              layout->d_sort_keys,
                              layout->d_cjpacked_sort_indices);
    Launch_Device_Kernel(
        Gather_CjPacked_By_Index,
        (layout->cjpacked_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout->cjpacked_numbers,
        layout->d_cjpacked_sort_indices, layout->d_nbnxm_cjpacked,
        layout->d_cjpacked_sort_buffer);
    deviceMemcpy(layout->d_nbnxm_cjpacked, layout->d_cjpacked_sort_buffer,
                 sizeof(LJ_CLUSTERED_CJ_PACKED) *
                     static_cast<size_t>(layout->cjpacked_numbers),
                 deviceMemcpyDeviceToDevice);
#endif
}

static void Reorder_Forceonly_Warp_Records_By_Cluster_J(
    LJ_CLUSTER_LAYOUT* layout)
{
#ifdef USE_CPU
    (void)layout;
#else
    if (layout == NULL || layout->sci_numbers <= 0 ||
        layout->forceonly_warp_record_numbers <= 1 ||
        layout->d_nbnxm_sci == NULL ||
        layout->d_forceonly_warp_record_offsets == NULL ||
        layout->d_forceonly_warp_j_records == NULL)
    {
        return;
    }
    const int record_numbers = layout->forceonly_warp_record_numbers;
    Reserve_Device_U64_Buffer(record_numbers, &layout->d_sort_keys,
                              &layout->sort_key_capacity);
    Reserve_Device_Int_Buffer(record_numbers, &layout->d_jentry_indices,
                              &layout->jentry_index_capacity);
    Reserve_Device_Buffer(record_numbers, &layout->d_nbnxm_warp_j_records,
                          &layout->nbnxm_warp_record_capacity);
    Launch_Device_Kernel(
        Build_Forceonly_Record_Sort_Keys,
        (layout->sci_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout->sci_numbers,
        layout->d_nbnxm_sci, layout->d_forceonly_warp_record_offsets,
        layout->d_forceonly_warp_j_records, layout->d_sort_keys);
    Launch_Device_Kernel(
        Build_Linear_Indices,
        (record_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, record_numbers,
        layout->d_jentry_indices);
    Stable_Sort_Device_By_Key(layout, record_numbers, layout->d_sort_keys,
                              layout->d_jentry_indices);
    Launch_Device_Kernel(
        Gather_Forceonly_Records_By_Index,
        (record_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, record_numbers,
        layout->d_jentry_indices, layout->d_forceonly_warp_j_records,
        layout->d_nbnxm_warp_j_records);
    deviceMemcpy(layout->d_forceonly_warp_j_records,
                 layout->d_nbnxm_warp_j_records,
                 sizeof(LJ_CLUSTERED_WARP_J_RECORD) *
                     static_cast<size_t>(record_numbers),
                 deviceMemcpyDeviceToDevice);
#endif
}

static void Rebuild_Payload_From_Sorted_J_Entries(LJ_CLUSTER_LAYOUT* layout,
                                                  bool block_sort_central_only)
{
#ifdef USE_CPU
    (void)layout;
    (void)block_sort_central_only;
#else
    if (layout == NULL || layout->sci_numbers <= 0 || layout->cjpacked_numbers <= 0 ||
        layout->d_nbnxm_sci == NULL || layout->d_nbnxm_cjpacked == NULL)
    {
        return;
    }
    const bool trace = Clustered_Trace_Warp_Records_Enabled();

    const int sci_numbers = layout->sci_numbers;
    const int original_cjpacked_numbers = layout->cjpacked_numbers;
    const int scan_capacity = IntMax(layout->cjpacked_numbers + 1,
                                     layout->sci_numbers + 1);
    Reserve_Device_Int_Buffer(scan_capacity, &layout->d_jentry_counts,
                              &layout->jentry_count_capacity);
    Reserve_Device_Int_Buffer(scan_capacity, &layout->d_jentry_offsets,
                              &layout->jentry_offset_capacity);
    Reserve_Device_Int_Buffer(scan_capacity, &layout->d_jentry_indices,
                              &layout->jentry_index_capacity);

    Launch_Device_Kernel(
        Count_Cj_Entries_Per_Sci,
        (sci_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, sci_numbers,
        layout->d_nbnxm_sci, layout->d_nbnxm_cjpacked, layout->d_jentry_counts);
    const int jentry_numbers = Exclusive_Scan_Counts(
        layout, sci_numbers, layout->d_jentry_counts, layout->d_jentry_offsets);
    if (trace)
    {
        fprintf(stderr,
                "[clustered jentry rebuild] stage=count sci=%d cjpacked=%d "
                "jentries=%d\n",
                sci_numbers, original_cjpacked_numbers, jentry_numbers);
        fflush(stderr);
    }
    if (jentry_numbers <= 0)
    {
        return;
    }

    Reserve_Device_Buffer(jentry_numbers, &layout->d_j_entries,
                          &layout->jentry_capacity);
    Reserve_Device_Buffer(jentry_numbers, &layout->d_j_entry_buffer,
                          &layout->jentry_buffer_capacity);
    Reserve_Device_U64_Buffer(jentry_numbers, &layout->d_sort_keys,
                              &layout->sort_key_capacity);

    Launch_Device_Kernel(
        Fill_J_Entries_From_Payload,
        (sci_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, sci_numbers,
        layout->d_nbnxm_sci, layout->d_nbnxm_cjpacked, layout->d_jentry_offsets,
        layout->d_j_entries);
    if (trace)
    {
        Trace_J_Entry_Rebuild_Payload("pre", sci_numbers, layout->d_nbnxm_sci,
                                      original_cjpacked_numbers,
                                      layout->d_nbnxm_cjpacked, jentry_numbers,
                                      layout->d_j_entries);
        std::vector<LJ_CLUSTERED_SCI> host_sci(
            static_cast<size_t>(sci_numbers));
        std::vector<LJ_CLUSTERED_CJ_PACKED> host_cjpacked(
            static_cast<size_t>(original_cjpacked_numbers));
        deviceMemcpy(host_sci.data(), layout->d_nbnxm_sci,
                     sizeof(LJ_CLUSTERED_SCI) * sci_numbers,
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(host_cjpacked.data(), layout->d_nbnxm_cjpacked,
                     sizeof(LJ_CLUSTERED_CJ_PACKED) *
                         static_cast<size_t>(original_cjpacked_numbers),
                     deviceMemcpyDeviceToHost);
        Trace_J_Entry_Host_Repack("pre", host_sci, host_cjpacked,
                                  block_sort_central_only);
    }

    const int entry_scan_capacity = jentry_numbers + 1;
    Reserve_Device_Int_Buffer(entry_scan_capacity, &layout->d_jentry_counts,
                              &layout->jentry_count_capacity);
    Reserve_Device_Int_Buffer(entry_scan_capacity, &layout->d_jentry_offsets,
                              &layout->jentry_offset_capacity);
    Reserve_Device_Int_Buffer(entry_scan_capacity, &layout->d_jentry_indices,
                              &layout->jentry_index_capacity);
    if (block_sort_central_only)
    {
        Launch_Device_Kernel(
            Build_J_Entry_Block_Sort_Keys,
            (jentry_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, jentry_numbers,
            layout->d_j_entries, true, layout->d_sort_keys);
    }
    else
    {
        Launch_Device_Kernel(
            Build_J_Entry_Sort_Keys,
            (jentry_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, jentry_numbers,
            layout->d_j_entries, layout->d_sort_keys);
    }
    Launch_Device_Kernel(
        Build_Linear_Indices,
        (jentry_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, jentry_numbers,
        layout->d_jentry_indices);
    Stable_Sort_Device_By_Key(layout, jentry_numbers, layout->d_sort_keys,
                              layout->d_jentry_indices);
    Launch_Device_Kernel(
        Gather_J_Entries_By_Index,
        (jentry_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, jentry_numbers,
        layout->d_jentry_indices, layout->d_j_entries, layout->d_j_entry_buffer);
    deviceMemcpy(layout->d_j_entries, layout->d_j_entry_buffer,
                 sizeof(LJ_CLUSTERED_J_ENTRY) *
                     static_cast<size_t>(jentry_numbers),
                 deviceMemcpyDeviceToDevice);
    if (trace)
    {
        Trace_J_Entry_Rebuild_Payload("post-sort", 0, NULL, 0, NULL,
                                      jentry_numbers, layout->d_j_entries);
        fprintf(stderr,
                "[clustered jentry rebuild] stage=sort jentries=%d\n",
                jentry_numbers);
        fflush(stderr);
    }

    Launch_Device_Kernel(
        Build_J_Entry_Sci_Flags,
        (jentry_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, jentry_numbers,
        layout->d_j_entries, layout->d_jentry_counts);
    const int rebuilt_sci_numbers = Exclusive_Scan_Counts(
        layout, jentry_numbers, layout->d_jentry_counts, layout->d_jentry_offsets);
    if (trace)
    {
        fprintf(stderr,
                "[clustered jentry rebuild] stage=group jentries=%d sci=%d\n",
                jentry_numbers, rebuilt_sci_numbers);
        fflush(stderr);
    }
    if (rebuilt_sci_numbers <= 0)
    {
        return;
    }

    Launch_Device_Kernel(
        Scatter_J_Entry_Sci_Starts,
        (jentry_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, jentry_numbers,
        layout->d_jentry_counts, layout->d_jentry_offsets,
        layout->d_jentry_indices);
    deviceMemcpy(layout->d_jentry_indices + rebuilt_sci_numbers,
                 &jentry_numbers, sizeof(int), deviceMemcpyHostToDevice);

    Launch_Device_Kernel(
        Build_J_Entry_Packed_Counts,
        (rebuilt_sci_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, rebuilt_sci_numbers,
        layout->d_jentry_indices, layout->d_jentry_counts);
    const int rebuilt_cjpacked_numbers = Exclusive_Scan_Counts(
        layout, rebuilt_sci_numbers, layout->d_jentry_counts,
        layout->d_jentry_offsets);
    if (trace)
    {
        fprintf(stderr,
                "[clustered jentry rebuild] stage=pack sci=%d cjpacked=%d\n",
                rebuilt_sci_numbers, rebuilt_cjpacked_numbers);
        fflush(stderr);
    }
    if (rebuilt_cjpacked_numbers <= 0)
    {
        return;
    }

    Reserve_Device_Buffer(rebuilt_sci_numbers, &layout->d_nbnxm_sci,
                          &layout->nbnxm_sci_capacity);
    Reserve_Device_Buffer(rebuilt_cjpacked_numbers,
                          &layout->d_cjpacked_sort_buffer,
                          &layout->cjpacked_sort_buffer_capacity);
    Launch_Device_Kernel(
        Write_Repacked_Sci_From_J_Entries,
        (rebuilt_sci_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, rebuilt_sci_numbers,
        layout->d_jentry_indices, layout->d_jentry_offsets,
        layout->d_j_entries, layout->d_nbnxm_sci);
    Launch_Device_Kernel(
        Initialize_CjPacked_Array,
        (rebuilt_cjpacked_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, rebuilt_cjpacked_numbers,
        layout->d_cjpacked_sort_buffer);
    Launch_Device_Kernel(
        Fill_Repacked_CjPacked_From_J_Entries,
        (rebuilt_sci_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, rebuilt_sci_numbers,
        layout->d_jentry_indices, layout->d_jentry_offsets,
        layout->d_j_entries, layout->d_cjpacked_sort_buffer);

    Reserve_Device_Buffer(rebuilt_cjpacked_numbers, &layout->d_nbnxm_cjpacked,
                          &layout->nbnxm_cjpacked_capacity);
    deviceMemcpy(layout->d_nbnxm_cjpacked, layout->d_cjpacked_sort_buffer,
                 sizeof(LJ_CLUSTERED_CJ_PACKED) *
                     static_cast<size_t>(rebuilt_cjpacked_numbers),
                 deviceMemcpyDeviceToDevice);
    layout->sci_numbers = rebuilt_sci_numbers;
    layout->cjpacked_numbers = rebuilt_cjpacked_numbers;
    if (trace)
    {
        Trace_J_Entry_Rebuild_Payload("post-repack", rebuilt_sci_numbers,
                                      layout->d_nbnxm_sci,
                                      rebuilt_cjpacked_numbers,
                                      layout->d_nbnxm_cjpacked);
        std::vector<LJ_CLUSTERED_SCI> device_repacked_sci(
            static_cast<size_t>(rebuilt_sci_numbers));
        deviceMemcpy(device_repacked_sci.data(), layout->d_nbnxm_sci,
                     sizeof(LJ_CLUSTERED_SCI) * rebuilt_sci_numbers,
                     deviceMemcpyDeviceToHost);
        int first_overlap_sci = -1;
        int previous_end = 0;
        for (int sci = 0; sci < rebuilt_sci_numbers; sci += 1)
        {
            const LJ_CLUSTERED_SCI& entry =
                device_repacked_sci[static_cast<size_t>(sci)];
            if (sci > 0 && entry.cjpacked_begin < previous_end)
            {
                first_overlap_sci = sci;
                break;
            }
            previous_end = entry.cjpacked_end;
        }
        if (first_overlap_sci >= 0)
        {
            std::vector<int> sci_starts(static_cast<size_t>(rebuilt_sci_numbers) + 1);
            std::vector<int> packed_offsets(
                static_cast<size_t>(rebuilt_sci_numbers) + 1);
            deviceMemcpy(sci_starts.data(), layout->d_jentry_indices,
                         sizeof(int) * (rebuilt_sci_numbers + 1),
                         deviceMemcpyDeviceToHost);
            deviceMemcpy(packed_offsets.data(), layout->d_jentry_offsets,
                         sizeof(int) * (rebuilt_sci_numbers + 1),
                         deviceMemcpyDeviceToHost);
            Trace_Sci_Window_On_Host("device-post-repack", device_repacked_sci,
                                     first_overlap_sci,
                                     &sci_starts, &packed_offsets);
        }
        fprintf(stderr,
                "[clustered jentry rebuild] stage=done sci=%d->%d cjpacked=%d->%d "
                "jentries=%d block_sort=%d\n",
                sci_numbers, rebuilt_sci_numbers, original_cjpacked_numbers,
                rebuilt_cjpacked_numbers, jentry_numbers,
                block_sort_central_only ? 1 : 0);
        fflush(stderr);
    }
#endif
}

static __global__ void Count_Final_Sci_Per_Supercluster(
    const int sci_numbers, const LJ_CLUSTERED_SCI* sci_entries,
    int* grouped_sci_counts)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        atomicAdd(grouped_sci_counts + sci_entries[sci].supercluster_id, 1);
    }
}

static __global__ void Fill_Final_Sci_Groups_By_Supercluster(
    const int sci_numbers, const LJ_CLUSTERED_SCI* sci_entries,
    int* grouped_sci_write_offsets, int* grouped_sci_ids)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const int super_i = sci_entries[sci].supercluster_id;
        const int write_index =
            atomicAdd(grouped_sci_write_offsets + super_i, 1);
        grouped_sci_ids[write_index] = sci;
    }
}

static void Reserve_Device_Int_Buffer(int capacity, int** pointer,
                                      int* current_capacity)
{
    if (capacity <= *current_capacity && *pointer != NULL)
    {
        return;
    }
    if (*pointer != NULL)
    {
        Free_Single_Device_Pointer((void**)pointer);
    }
    Clustered_Device_Malloc_Safely((void**)pointer, sizeof(int) * capacity,
                                   "reserve-int-buffer");
    *current_capacity = capacity;
}

static void Reserve_Device_U64_Buffer(int capacity, uint64_t** pointer,
                                      int* current_capacity)
{
    if (capacity <= *current_capacity && *pointer != NULL)
    {
        return;
    }
    if (*pointer != NULL)
    {
        Free_Single_Device_Pointer((void**)pointer);
    }
    Clustered_Device_Malloc_Safely((void**)pointer,
                                   sizeof(uint64_t) * capacity,
                                   "reserve-u64-buffer");
    *current_capacity = capacity;
}

static void Reserve_Device_UInt_Buffer(int capacity, unsigned int** pointer,
                                       int* current_capacity)
{
    if (capacity <= *current_capacity && *pointer != NULL)
    {
        return;
    }
    if (*pointer != NULL)
    {
        Free_Single_Device_Pointer((void**)pointer);
    }
    Clustered_Device_Malloc_Safely((void**)pointer,
                                   sizeof(unsigned int) * capacity,
                                   "reserve-uint-buffer");
    *current_capacity = capacity;
}

static void Reserve_Device_PairMask_Buffer(int capacity,
                                           unsigned long long** pointer,
                                           int* current_capacity)
{
    if (capacity <= *current_capacity && *pointer != NULL)
    {
        return;
    }
    if (*pointer != NULL)
    {
        Free_Single_Device_Pointer((void**)pointer);
    }
    Clustered_Device_Malloc_Safely(
        (void**)pointer, sizeof(unsigned long long) * capacity,
        "reserve-pairmask-buffer");
    *current_capacity = capacity;
}

static void Reserve_Device_Vector_Buffer(int capacity, VECTOR** pointer,
                                         int* current_capacity)
{
    if (capacity <= *current_capacity && *pointer != NULL)
    {
        return;
    }
    if (*pointer != NULL)
    {
        Free_Single_Device_Pointer((void**)pointer);
    }
    Clustered_Device_Malloc_Safely((void**)pointer, sizeof(VECTOR) * capacity,
                                   "reserve-vector-buffer");
    *current_capacity = capacity;
}

static void Reserve_Device_Float_Buffer(int capacity, float** pointer,
                                        int* current_capacity)
{
    if (capacity <= *current_capacity && *pointer != NULL)
    {
        return;
    }
    if (*pointer != NULL)
    {
        Free_Single_Device_Pointer((void**)pointer);
    }
    Clustered_Device_Malloc_Safely((void**)pointer, sizeof(float) * capacity,
                                   "reserve-float-buffer");
    *current_capacity = capacity;
}

static void Commit_Clustered_Build_Cache(LJ_CLUSTER_LAYOUT* layout,
                                         const VECTOR* crd, float cutoff)
{
    if (layout->total_atom_numbers > 0)
    {
        Reserve_Device_Vector_Buffer(layout->total_atom_numbers,
                                     &layout->d_cached_crd,
                                     &layout->cached_crd_capacity);
        deviceMemcpy(layout->d_cached_crd, crd,
                     sizeof(VECTOR) * layout->total_atom_numbers,
                     deviceMemcpyDeviceToDevice);
    }
    layout->cached_cutoff = cutoff;
    layout->cached_build_step = md_info.sys.steps;
    layout->cache_ready = layout->total_atom_numbers > 0;
    layout->rebuild_dirty = false;
}

static bool Clustered_Build_Warp_Records_Enabled()
{
    const char* enabled = std::getenv("SPONGE_CLUSTERED_BUILD_WARP_RECORDS");
    return enabled == NULL || enabled[0] == '\0' || enabled[0] != '0';
}

static bool Clustered_Outer_Inner_Prune_Enabled(const LJ_CLUSTER_LAYOUT* layout)
{
    const char* enabled = std::getenv("SPONGE_CLUSTERED_USE_OUTER_INNER_PRUNE");
    if (enabled != NULL && enabled[0] != '\0')
    {
        return enabled[0] != '0';
    }
    return layout != NULL && layout->rebuild_skin > 0.0f;
}

static bool Clustered_Trace_Warp_Records_Enabled()
{
    const char* enabled = std::getenv("SPONGE_CLUSTERED_TRACE_WARP_RECORDS");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static void Invalidate_Clustered_Legacy_Neighbor_View(
    LJ_CLUSTER_LAYOUT* layout)
{
    if (layout == NULL)
    {
        return;
    }
    layout->legacy_neighbor_view_ready = false;
    layout->legacy_neighbor_view_payload_build_count = -1;
    layout->legacy_neighbor_view_step = -1;
    layout->legacy_neighbor_view_cutoff_skin = -1.0f;
}

#ifndef USE_CPU
static void Clustered_Debug_Device_Sync_If_Tracing(const char* tag)
{
    if (!Clustered_Trace_Warp_Records_Enabled())
    {
        return;
    }
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[clustered debug sync] %s: %s\n", tag,
                cudaGetErrorString(err));
    }
}
#endif

static bool Clustered_Use_Shift_Partitioned_Builder()
{
    const char* enabled =
        std::getenv("SPONGE_CLUSTERED_SHIFT_PARTITIONED_BUILDER");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static bool Clustered_Use_Sparse_Shift_Candidate_Builder()
{
    const char* enabled =
        std::getenv("SPONGE_CLUSTERED_SPARSE_SHIFT_CANDIDATES");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static bool Clustered_Use_Central_Candidate_Halfshell_Culling()
{
    const char* enabled =
        std::getenv("SPONGE_CLUSTERED_CENTRAL_CANDIDATE_HALFSHELL");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static bool Clustered_Use_Fixed_Shift_Leaf_Screening()
{
    const char* enabled =
        std::getenv("SPONGE_CLUSTERED_FIXED_SHIFT_LEAF_SCREENING");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static bool Clustered_Use_Morton_Sfc()
{
    const char* enabled = std::getenv("SPONGE_CLUSTERED_USE_MORTON_SFC");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static bool Clustered_Sort_Leaf_Atoms_By_Molecule_Enabled()
{
    const char* enabled =
        std::getenv("SPONGE_CLUSTERED_SORT_LEAF_ATOMS_BY_MOLECULE");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static bool Clustered_Oxygen_Key_Packing_Enabled()
{
    const char* enabled =
        std::getenv("SPONGE_CLUSTERED_OXYGEN_KEY_PACKING");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static bool Clustered_Leaf_Geometry_Repack_Enabled()
{
    const char* enabled =
        std::getenv("SPONGE_CLUSTERED_LEAF_GEOMETRY_REPACK");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static bool Clustered_Use_Spatial_Supercluster_Grouping()
{
    const char* enabled =
        std::getenv("SPONGE_CLUSTERED_SPATIAL_SUPERCLUSTER_GROUPING");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static float Clustered_Spatial_Supercluster_Link_Cutoff()
{
    const char* value =
        std::getenv("SPONGE_CLUSTERED_SPATIAL_SUPERCLUSTER_LINK_CUTOFF");
    return value != NULL && value[0] != '\0' ? atof(value) : 0.0f;
}

static bool Clustered_Sort_CjPacked_Enabled()
{
    const char* enabled = std::getenv("SPONGE_CLUSTERED_SORT_CJPACKED");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static bool Clustered_Sort_J_Entries_Enabled()
{
    const char* enabled = std::getenv("SPONGE_CLUSTERED_SORT_JENTRIES");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static bool Clustered_Block_Sort_J_Entries_Enabled()
{
    const char* enabled =
        std::getenv("SPONGE_CLUSTERED_BLOCK_SORT_JENTRIES");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static bool Clustered_Sort_Forceonly_Records_Enabled()
{
    const char* enabled =
        std::getenv("SPONGE_CLUSTERED_SORT_FORCEONLY_RECORDS");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static bool Clustered_Direct_Sort_Forceonly_Records_Enabled()
{
    const char* enabled =
        std::getenv("SPONGE_CLUSTERED_DIRECT_SORT_FORCEONLY_RECORDS");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static bool Clustered_Dedup_Exclusion_Masks_Enabled()
{
    const char* enabled =
        std::getenv("SPONGE_CLUSTERED_DEDUP_EXCLUSION_MASKS");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static bool Clustered_Gmxpacked_Direct_Candidate_Build_Enabled()
{
    const char* enabled =
        std::getenv("SPONGE_CLUSTERED_GMXPACKED_DIRECT_BUILD");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static bool Clustered_Gmxpacked_Reduced_Build_Enabled()
{
    const char* enabled =
        std::getenv("SPONGE_CLUSTERED_GMXPACKED_REDUCED_BUILD");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static bool Clustered_Gmxpacked_Early_Record_Analyze_Enabled()
{
    const char* enabled =
        std::getenv("SPONGE_CLUSTERED_GMXPACKED_EARLY_RECORD_ANALYZE");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static const char* Clustered_Gmxpacked_Reduced_Staged_Trace_Source()
{
    return "source=reduced-staged";
}

struct ClusteredReducedStagedCountSummary
{
    int source_entries = 0;
    int reduced_entries = 0;
    int reduced_sci_numbers = 0;
    int predicted_compact_cj_numbers = 0;
};

struct ClusteredEarlyRecordAnalyzeSummary
{
    int raw_records = 0;
    int stored_records = 0;
    int unique_semantic_buckets = 0;
    int predicted_compact_cj_numbers = 0;
    int duplicate_records = 0;
    int duplicate_buckets = 0;
    int max_bucket_multiplicity = 0;
    int record_capacity = 0;
    int overflow_records = 0;
    std::array<int, 5> duplicate_histogram = {};
    size_t raw_record_bytes = 0;
    size_t peak_temp_bytes_est = 0;
};

static bool Clustered_Early_Record_Same_Sci(
    const ClusteredEarlyRecordAnalyzeEntry& lhs,
    const ClusteredEarlyRecordAnalyzeEntry& rhs)
{
    return lhs.supercluster_id == rhs.supercluster_id &&
           Clustered_Early_Record_Output_Shift_Id(lhs.semantic_bits) ==
               Clustered_Early_Record_Output_Shift_Id(rhs.semantic_bits);
}

static bool Clustered_Early_Record_Key_Less(
    const ClusteredEarlyRecordAnalyzeEntry& lhs,
    const ClusteredEarlyRecordAnalyzeEntry& rhs)
{
    if (lhs.supercluster_id != rhs.supercluster_id)
    {
        return lhs.supercluster_id < rhs.supercluster_id;
    }
    const int lhs_shift =
        Clustered_Early_Record_Output_Shift_Id(lhs.semantic_bits);
    const int rhs_shift =
        Clustered_Early_Record_Output_Shift_Id(rhs.semantic_bits);
    if (lhs_shift != rhs_shift)
    {
        return lhs_shift < rhs_shift;
    }
    if (lhs.cluster_j != rhs.cluster_j)
    {
        return lhs.cluster_j < rhs.cluster_j;
    }
    const unsigned int lhs_record_imask =
        Clustered_Early_Record_Record_Imask(lhs.semantic_bits);
    const unsigned int rhs_record_imask =
        Clustered_Early_Record_Record_Imask(rhs.semantic_bits);
    if (lhs_record_imask != rhs_record_imask)
    {
        return lhs_record_imask < rhs_record_imask;
    }
    const unsigned int lhs_valid_mask =
        Clustered_Early_Record_Valid_Mask(lhs.semantic_bits);
    const unsigned int rhs_valid_mask =
        Clustered_Early_Record_Valid_Mask(rhs.semantic_bits);
    if (lhs_valid_mask != rhs_valid_mask)
    {
        return lhs_valid_mask < rhs_valid_mask;
    }
    const unsigned int lhs_local_mask =
        Clustered_Early_Record_Local_Mask(lhs.semantic_bits);
    const unsigned int rhs_local_mask =
        Clustered_Early_Record_Local_Mask(rhs.semantic_bits);
    if (lhs_local_mask != rhs_local_mask)
    {
        return lhs_local_mask < rhs_local_mask;
    }
    for (int i_local = 0; i_local < kClusteredMaxSuperClusterClusters;
         i_local += 1)
    {
        if (lhs.exclusion_masks[i_local] != rhs.exclusion_masks[i_local])
        {
            return lhs.exclusion_masks[i_local] < rhs.exclusion_masks[i_local];
        }
    }
    return false;
}

static bool Clustered_Early_Record_Key_Equal(
    const ClusteredEarlyRecordAnalyzeEntry& lhs,
    const ClusteredEarlyRecordAnalyzeEntry& rhs)
{
    if (lhs.supercluster_id != rhs.supercluster_id ||
        lhs.cluster_j != rhs.cluster_j || lhs.semantic_bits != rhs.semantic_bits)
    {
        return false;
    }
    for (int i_local = 0; i_local < kClusteredMaxSuperClusterClusters;
         i_local += 1)
    {
        if (lhs.exclusion_masks[i_local] != rhs.exclusion_masks[i_local])
        {
            return false;
        }
    }
    return true;
}

static ClusteredEarlyRecordAnalyzeSummary
Analyze_Clustered_Early_Record_Stream_On_Host(
    std::vector<ClusteredEarlyRecordAnalyzeEntry>& host_records,
    int raw_record_count, int record_capacity)
{
    ClusteredEarlyRecordAnalyzeSummary summary = {};
    summary.raw_records = std::max(raw_record_count, 0);
    summary.stored_records = static_cast<int>(host_records.size());
    summary.record_capacity = std::max(record_capacity, 0);
    summary.overflow_records =
        std::max(summary.raw_records - summary.stored_records, 0);
    summary.raw_record_bytes =
        static_cast<size_t>(summary.raw_records) *
        sizeof(ClusteredEarlyRecordAnalyzeEntry);
    summary.peak_temp_bytes_est =
        (static_cast<size_t>(summary.record_capacity) +
         static_cast<size_t>(summary.stored_records)) *
        sizeof(ClusteredEarlyRecordAnalyzeEntry);
    if (host_records.empty())
    {
        return summary;
    }

    std::sort(host_records.begin(), host_records.end(),
              Clustered_Early_Record_Key_Less);

    int unique_records_in_current_sci = 0;
    for (size_t begin = 0; begin < host_records.size();)
    {
        size_t end = begin + 1;
        while (end < host_records.size() &&
               Clustered_Early_Record_Key_Equal(host_records[begin],
                                                host_records[end]))
        {
            end += 1;
        }

        const int multiplicity = static_cast<int>(end - begin);
        summary.unique_semantic_buckets += 1;
        summary.duplicate_records += multiplicity - 1;
        summary.max_bucket_multiplicity =
            std::max(summary.max_bucket_multiplicity, multiplicity);
        unique_records_in_current_sci += 1;
        if (multiplicity > 1)
        {
            summary.duplicate_buckets += 1;
        }
        if (multiplicity <= 1)
        {
            summary.duplicate_histogram[0] += 1;
        }
        else if (multiplicity == 2)
        {
            summary.duplicate_histogram[1] += 1;
        }
        else if (multiplicity == 3)
        {
            summary.duplicate_histogram[2] += 1;
        }
        else if (multiplicity == 4)
        {
            summary.duplicate_histogram[3] += 1;
        }
        else
        {
            summary.duplicate_histogram[4] += 1;
        }

        if (end == host_records.size() ||
            !Clustered_Early_Record_Same_Sci(host_records[begin],
                                             host_records[end]))
        {
            summary.predicted_compact_cj_numbers +=
                (unique_records_in_current_sci + kClusteredJGroupSize - 1) /
                kClusteredJGroupSize;
            unique_records_in_current_sci = 0;
        }
        begin = end;
    }

    return summary;
}

static void Trace_Clustered_Early_Record_Analysis(
    const LJ_CLUSTER_LAYOUT* layout,
    const ClusteredEarlyRecordAnalyzeSummary& summary)
{
    if (layout == NULL ||
        (!Clustered_Trace_Warp_Records_Enabled() &&
         !Clustered_Gmxpacked_Early_Record_Analyze_Enabled()))
    {
        return;
    }

    fprintf(stderr,
            "[clustered early record analysis] step=%d raw_records=%d "
            "stored_records=%d unique_semantic_buckets=%d "
            "predicted_compact_cj=%d dup_records=%d dup_buckets=%d "
            "max_dup=%d dup_hist=1x:%d,2x:%d,3x:%d,4x:%d,5+x:%d "
            "raw_record_bytes=%zu peak_temp_bytes_est=%zu "
            "compat_native_cj=%d record_cap=%d overflow=%d\n",
            md_info.sys.steps, summary.raw_records, summary.stored_records,
            summary.unique_semantic_buckets,
            summary.predicted_compact_cj_numbers, summary.duplicate_records,
            summary.duplicate_buckets, summary.max_bucket_multiplicity,
            summary.duplicate_histogram[0], summary.duplicate_histogram[1],
            summary.duplicate_histogram[2], summary.duplicate_histogram[3],
            summary.duplicate_histogram[4], summary.raw_record_bytes,
            summary.peak_temp_bytes_est, layout->cjpacked_numbers,
            summary.record_capacity, summary.overflow_records);
    fflush(stderr);
}

#ifndef USE_CPU
static void Analyze_Clustered_Early_Record_Stream(
    LJ_CLUSTER_LAYOUT* layout, int record_capacity,
    int* d_early_record_analysis_raw_count,
    ClusteredEarlyRecordAnalyzeEntry* d_early_record_analysis_entries)
{
    if (layout == NULL || d_early_record_analysis_raw_count == NULL ||
        d_early_record_analysis_entries == NULL || record_capacity <= 0)
    {
        return;
    }

    int raw_record_count = 0;
    deviceMemcpy(&raw_record_count, d_early_record_analysis_raw_count,
                 sizeof(int), deviceMemcpyDeviceToHost);
    const int stored_records = std::min(std::max(raw_record_count, 0),
                                        std::max(record_capacity, 0));
    std::vector<ClusteredEarlyRecordAnalyzeEntry> host_records(
        static_cast<size_t>(stored_records));
    if (!host_records.empty())
    {
        deviceMemcpy(host_records.data(), d_early_record_analysis_entries,
                     sizeof(ClusteredEarlyRecordAnalyzeEntry) *
                         host_records.size(),
                     deviceMemcpyDeviceToHost);
    }

    Trace_Clustered_Early_Record_Analysis(
        layout, Analyze_Clustered_Early_Record_Stream_On_Host(
                    host_records, raw_record_count, record_capacity));
}
#endif

static void Trace_Clustered_Reduced_Staged_Count(
    const LJ_CLUSTER_LAYOUT* layout, int build_step_count,
    const ClusteredReducedStagedCountSummary& summary)
{
    if (layout == NULL || !Clustered_Trace_Warp_Records_Enabled())
    {
        return;
    }
    fprintf(stderr,
            "[clustered reduced staged count] step=%d step_builds=%d "
            "total_builds=%lld %s source_entries=%d reduced_entries=%d "
            "reduced_sci=%d predicted_compact_cj=%d compat_native_sci=%d "
            "compat_native_cj=%d\n",
            md_info.sys.steps, build_step_count,
            layout->primary_payload_build_count_total,
            Clustered_Gmxpacked_Reduced_Staged_Trace_Source(),
            summary.source_entries, summary.reduced_entries,
            summary.reduced_sci_numbers, summary.predicted_compact_cj_numbers,
            layout->sci_numbers, layout->cjpacked_numbers);
    fflush(stderr);
}

static int Exclusive_Scan_Counts(LJ_CLUSTER_LAYOUT* layout, int count_numbers,
                                 int* d_counts, int* d_starts);

static void Analyze_Grouped_Clustered_J_Reuse(const LJ_CLUSTER_LAYOUT& layout)
{
    if (std::getenv("SPONGE_CLUSTERED_ANALYZE_REUSE") == NULL ||
        layout.super_cluster_numbers <= 0 || layout.sci_numbers <= 0 ||
        layout.cjpacked_numbers <= 0 || layout.cluster_numbers <= 0 ||
        layout.d_grouped_sci_offsets == NULL || layout.d_grouped_sci_ids == NULL ||
        layout.d_nbnxm_sci == NULL || layout.d_nbnxm_cjpacked == NULL)
    {
        return;
    }

    std::vector<int> grouped_offsets((size_t)layout.super_cluster_numbers + 1);
    std::vector<int> grouped_ids((size_t)layout.sci_numbers);
    std::vector<LJ_CLUSTERED_SCI> scis((size_t)layout.sci_numbers);
    std::vector<LJ_CLUSTERED_CJ_PACKED> cjpacked((size_t)layout.cjpacked_numbers);
    deviceMemcpy(grouped_offsets.data(), layout.d_grouped_sci_offsets,
                 sizeof(int) * (layout.super_cluster_numbers + 1),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(grouped_ids.data(), layout.d_grouped_sci_ids,
                 sizeof(int) * layout.sci_numbers, deviceMemcpyDeviceToHost);
    deviceMemcpy(scis.data(), layout.d_nbnxm_sci,
                 sizeof(LJ_CLUSTERED_SCI) * layout.sci_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(cjpacked.data(), layout.d_nbnxm_cjpacked,
                 sizeof(LJ_CLUSTERED_CJ_PACKED) * layout.cjpacked_numbers,
                 deviceMemcpyDeviceToHost);

    std::vector<int> last_seen((size_t)layout.cluster_numbers, -1);
    std::vector<int> stamp((size_t)layout.cluster_numbers, -1);
    int sequence_id = 0;
    long long total_occurrences = 0;
    long long total_unique = 0;
    long long hits4 = 0;
    long long hits8 = 0;
    long long hits16 = 0;
    int active_superclusters = 0;
    int max_occurrences = 0;
    int max_unique = 0;

    for (int super_i = 0; super_i < layout.super_cluster_numbers; super_i += 1)
    {
        const int begin = grouped_offsets[(size_t)super_i];
        const int end = grouped_offsets[(size_t)super_i + 1];
        if (begin >= end)
        {
            continue;
        }
        active_superclusters += 1;
        sequence_id += 1;
        int position = 0;
        int unique = 0;
        for (int grouped_idx = begin; grouped_idx < end; grouped_idx += 1)
        {
            const LJ_CLUSTERED_SCI& sci = scis[(size_t)grouped_ids[(size_t)grouped_idx]];
            for (int packed_idx = sci.cjpacked_begin; packed_idx < sci.cjpacked_end;
                 packed_idx += 1)
            {
                const LJ_CLUSTERED_CJ_PACKED& packed = cjpacked[(size_t)packed_idx];
                for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
                {
                    const int cluster_j = packed.cj[jm];
                    if (cluster_j < 0)
                    {
                        continue;
                    }
                    total_occurrences += 1;
                    const bool seen_in_sequence =
                        stamp[(size_t)cluster_j] == sequence_id;
                    if (!seen_in_sequence)
                    {
                        stamp[(size_t)cluster_j] = sequence_id;
                        unique += 1;
                    }
                    const int previous = last_seen[(size_t)cluster_j];
                    if (seen_in_sequence && previous >= 0)
                    {
                        const int reuse_distance = position - previous;
                        if (reuse_distance <= 4)
                        {
                            hits4 += 1;
                        }
                        if (reuse_distance <= 8)
                        {
                            hits8 += 1;
                        }
                        if (reuse_distance <= 16)
                        {
                            hits16 += 1;
                        }
                    }
                    last_seen[(size_t)cluster_j] = position;
                    position += 1;
                }
            }
        }
        total_unique += unique;
        max_occurrences = std::max(max_occurrences, position);
        max_unique = std::max(max_unique, unique);
    }

    const double duplication =
        total_unique > 0 ? static_cast<double>(total_occurrences) / total_unique
                         : 0.0;
    const double reuse4 =
        total_occurrences > 0 ? static_cast<double>(hits4) / total_occurrences
                              : 0.0;
    const double reuse8 =
        total_occurrences > 0 ? static_cast<double>(hits8) / total_occurrences
                              : 0.0;
    const double reuse16 =
        total_occurrences > 0 ? static_cast<double>(hits16) / total_occurrences
                               : 0.0;
    fprintf(stderr,
            "[clustered grouped reuse] active_superclusters=%d total_occ=%lld "
            "total_unique=%lld dup=%.3f reuse<=4=%.3f reuse<=8=%.3f "
            "reuse<=16=%.3f max_occ=%d max_unique=%d\n",
            active_superclusters, total_occurrences, total_unique, duplication,
            reuse4, reuse8, reuse16, max_occurrences, max_unique);
}

#ifndef USE_CPU
static void Trace_Clustered_Builder_Stats(const LJ_CLUSTER_LAYOUT& layout,
                                          const int candidate_sci_numbers,
                                          const float cutoff,
                                          const LTMatrix3 cell,
                                          const LTMatrix3 rcell)
{
    const bool dense_shift_partitioned_candidates =
        Clustered_Use_Shift_Partitioned_Builder();
    const bool sparse_shift_candidates =
        !dense_shift_partitioned_candidates &&
        Clustered_Use_Sparse_Shift_Candidate_Builder();
    const bool fixed_shift_candidates =
        dense_shift_partitioned_candidates || sparse_shift_candidates;
    if (candidate_sci_numbers <= 0 || layout.cluster_numbers <= 0 ||
        layout.super_cluster_numbers <= 0 || layout.d_cluster_valid_masks == NULL ||
        layout.d_cluster_local_masks == NULL || layout.d_cluster_centers == NULL ||
        layout.d_cluster_extents == NULL || layout.d_cluster_radii == NULL ||
        layout.d_leaf_cluster_starts == NULL ||
        layout.d_leaf_cluster_ends == NULL ||
        layout.d_super_cluster_offsets == NULL ||
        layout.d_super_cluster_sizes == NULL ||
        layout.d_sci_supercluster_ids == NULL ||
        layout.d_sci_candidate_leaf_offsets == NULL ||
        layout.d_sci_candidate_leaf_ids == NULL)
    {
        return;
    }

    int candidate_leaf_numbers = 0;
    deviceMemcpy(&candidate_leaf_numbers,
                 layout.d_sci_candidate_leaf_offsets + candidate_sci_numbers,
                 sizeof(int), deviceMemcpyDeviceToHost);
    const int super_sci_numbers = dense_shift_partitioned_candidates
                                      ? candidate_sci_numbers /
                                            kClusteredShiftCount
                                      : candidate_sci_numbers;

    std::vector<unsigned int> cluster_valid_masks((size_t)layout.cluster_numbers);
    std::vector<unsigned int> cluster_local_masks((size_t)layout.cluster_numbers);
    std::vector<VECTOR> cluster_centers((size_t)layout.cluster_numbers);
    std::vector<VECTOR> cluster_extents((size_t)layout.cluster_numbers);
    std::vector<float> cluster_radii((size_t)layout.cluster_numbers);
    std::vector<int> leaf_cluster_starts((size_t)layout.cornerstone_state->octree.numLeafNodes);
    std::vector<int> leaf_cluster_ends((size_t)layout.cornerstone_state->octree.numLeafNodes);
    std::vector<int> super_cluster_offsets(
        (size_t)layout.super_cluster_numbers + 1);
    std::vector<int> cluster_to_supercluster((size_t)layout.cluster_numbers);
    std::vector<VECTOR> super_cluster_centers((size_t)layout.super_cluster_numbers);
    std::vector<VECTOR> super_cluster_sizes((size_t)layout.super_cluster_numbers);
    const int trace_sci_supercluster_numbers =
        sparse_shift_candidates ? candidate_sci_numbers : super_sci_numbers;
    std::vector<int> sci_supercluster_ids((size_t)trace_sci_supercluster_numbers);
    std::vector<int> candidate_shift_ids(
        sparse_shift_candidates ? (size_t)candidate_sci_numbers : 0u);
    std::vector<int> candidate_leaf_offsets((size_t)candidate_sci_numbers + 1);
    std::vector<int> candidate_leaf_ids((size_t)candidate_leaf_numbers);

    deviceMemcpy(cluster_valid_masks.data(), layout.d_cluster_valid_masks,
                 sizeof(unsigned int) * layout.cluster_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(cluster_local_masks.data(), layout.d_cluster_local_masks,
                 sizeof(unsigned int) * layout.cluster_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(cluster_centers.data(), layout.d_cluster_centers,
                 sizeof(VECTOR) * layout.cluster_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(cluster_extents.data(), layout.d_cluster_extents,
                 sizeof(VECTOR) * layout.cluster_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(cluster_radii.data(), layout.d_cluster_radii,
                 sizeof(float) * layout.cluster_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(leaf_cluster_starts.data(), layout.d_leaf_cluster_starts,
                 sizeof(int) * layout.cornerstone_state->octree.numLeafNodes,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(leaf_cluster_ends.data(), layout.d_leaf_cluster_ends,
                 sizeof(int) * layout.cornerstone_state->octree.numLeafNodes,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(super_cluster_offsets.data(), layout.d_super_cluster_offsets,
                 sizeof(int) * (layout.super_cluster_numbers + 1),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(cluster_to_supercluster.data(), layout.d_cluster_to_supercluster,
                 sizeof(int) * layout.cluster_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(super_cluster_centers.data(), layout.d_super_cluster_centers,
                 sizeof(VECTOR) * layout.super_cluster_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(super_cluster_sizes.data(), layout.d_super_cluster_sizes,
                 sizeof(VECTOR) * layout.super_cluster_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(sci_supercluster_ids.data(),
                 sparse_shift_candidates ? layout.d_candidate_sci_offsets
                                         : layout.d_sci_supercluster_ids,
                 sizeof(int) * trace_sci_supercluster_numbers,
                 deviceMemcpyDeviceToHost);
    if (sparse_shift_candidates)
    {
        deviceMemcpy(candidate_shift_ids.data(), layout.d_candidate_shift_ids,
                     sizeof(int) * candidate_sci_numbers,
                     deviceMemcpyDeviceToHost);
    }
    deviceMemcpy(candidate_leaf_offsets.data(), layout.d_sci_candidate_leaf_offsets,
                 sizeof(int) * (candidate_sci_numbers + 1),
                 deviceMemcpyDeviceToHost);
    if (candidate_leaf_numbers > 0)
    {
        deviceMemcpy(candidate_leaf_ids.data(), layout.d_sci_candidate_leaf_ids,
                     sizeof(int) * candidate_leaf_numbers,
                     deviceMemcpyDeviceToHost);
    }

    long long total_raw_leaf_clusters = 0;
    long long total_deduped_cluster_j = 0;
    long long total_valid_cluster_j = 0;
    long long total_accepted_cluster_j = 0;
    long long total_shift_candidates = 0;
    long long total_overlap_cluster_pairs = 0;
    long long total_accepted_records = 0;
    long long total_actual_central_cluster_j = 0;
    long long total_actual_shifted_cluster_j = 0;
    long long total_canonical_central_cluster_j = 0;
    long long total_canonical_shifted_cluster_j = 0;
    long long total_canonical_shift_matches = 0;
    long long total_canonical_shift_mismatches = 0;
    long long sum_active_local_i_clusters = 0;

    int max_raw_leaf_clusters_per_sci = 0;
    int max_deduped_cluster_j_per_sci = 0;
    int max_valid_cluster_j_per_sci = 0;
    int max_accepted_records_per_sci = 0;
    int max_active_local_i_clusters = 0;

    double total_super_span_x = 0.0;
    double total_super_span_y = 0.0;
    double total_super_span_z = 0.0;
    double total_cluster_radius = 0.0;
    float max_cluster_radius = 0.0f;
    for (int cluster_i = 0; cluster_i < layout.cluster_numbers; cluster_i += 1)
    {
        total_cluster_radius += cluster_radii[(size_t)cluster_i];
        max_cluster_radius =
            std::max(max_cluster_radius, cluster_radii[(size_t)cluster_i]);
    }

    VECTOR shift_vectors[kClusteredShiftCount];
    for (int shift_id = 0; shift_id < kClusteredShiftCount; shift_id += 1)
    {
        shift_vectors[shift_id] = Shift_Vector_From_Id(shift_id, cell);
    }

    for (int sci = 0; sci < candidate_sci_numbers; sci += 1)
    {
        const int sci_base = dense_shift_partitioned_candidates
                                 ? sci / kClusteredShiftCount
                                 : sci;
        const int fixed_shift_id =
            !fixed_shift_candidates
                ? -1
                : (sparse_shift_candidates ? candidate_shift_ids[(size_t)sci]
                                           : (sci % kClusteredShiftCount));
        const int super_i = sci_supercluster_ids[(size_t)sci_base];
        const int cluster_i_start = super_cluster_offsets[(size_t)super_i];
        const int cluster_i_end = super_cluster_offsets[(size_t)super_i + 1];

        int active_local_i_clusters = 0;
        for (int cluster_i = cluster_i_start; cluster_i < cluster_i_end;
             cluster_i += 1)
        {
            active_local_i_clusters +=
                cluster_local_masks[(size_t)cluster_i] != 0u ? 1 : 0;
        }
        sum_active_local_i_clusters += active_local_i_clusters;
        max_active_local_i_clusters =
            std::max(max_active_local_i_clusters, active_local_i_clusters);

        const VECTOR super_size = super_cluster_sizes[(size_t)super_i];
        total_super_span_x += 2.0 * static_cast<double>(super_size.x);
        total_super_span_y += 2.0 * static_cast<double>(super_size.y);
        total_super_span_z += 2.0 * static_cast<double>(super_size.z);

        int raw_leaf_clusters_per_sci = 0;
        int deduped_cluster_j_per_sci = 0;
        int valid_cluster_j_per_sci = 0;
        int accepted_records_per_sci = 0;
        int processed_cluster_end = 0;

        for (int candidate_idx = candidate_leaf_offsets[(size_t)sci];
             candidate_idx < candidate_leaf_offsets[(size_t)sci + 1];
             candidate_idx += 1)
        {
            const int leaf_j = candidate_leaf_ids[(size_t)candidate_idx];
            const int cluster_j_start = leaf_cluster_starts[(size_t)leaf_j];
            const int cluster_j_end = leaf_cluster_ends[(size_t)leaf_j];
            raw_leaf_clusters_per_sci +=
                IntMax(0, cluster_j_end - cluster_j_start);

            const int deduped_cluster_j_start =
                IntMax(cluster_j_start, processed_cluster_end);
            deduped_cluster_j_per_sci +=
                IntMax(0, cluster_j_end - deduped_cluster_j_start);

            for (int cluster_j = deduped_cluster_j_start; cluster_j < cluster_j_end;
                 cluster_j += 1)
            {
                const unsigned int valid_mask_j =
                    cluster_valid_masks[(size_t)cluster_j];
                if (valid_mask_j == 0u)
                {
                    continue;
                }
                const unsigned int local_mask_j =
                    cluster_local_masks[(size_t)cluster_j];
                const int super_j = cluster_to_supercluster[(size_t)cluster_j];
                if (local_mask_j != 0u && super_j < super_i)
                {
                    continue;
                }

                valid_cluster_j_per_sci += 1;
                const VECTOR center_j = cluster_centers[(size_t)cluster_j];
                const int canonical_shift_id = Determine_Cluster_Shift_Id(
                    super_cluster_centers[(size_t)super_i], center_j, rcell);
                unsigned int imasks[kClusteredShiftCount] = {};

                for (int cluster_i = cluster_i_start; cluster_i < cluster_i_end;
                     cluster_i += 1)
                {
                    const unsigned int local_mask_i =
                        cluster_local_masks[(size_t)cluster_i];
                    if (local_mask_i == 0u)
                    {
                        continue;
                    }
                    int pair_shift_id = fixed_shift_candidates
                                            ? fixed_shift_id
                                            : Determine_Cluster_Pair_Shift_Id(
                                                  cluster_centers[(size_t)cluster_i],
                                                  center_j, rcell);
                    if (pair_shift_id == kClusteredCentralShiftId &&
                        cluster_j >= cluster_i_start && cluster_j < cluster_i_end &&
                        cluster_i > cluster_j)
                    {
                        pair_shift_id = -1;
                    }
                    if (pair_shift_id < 0)
                    {
                        continue;
                    }
                    total_shift_candidates += 1;
                    if (!Cluster_Aabb_Overlaps_Shifted(
                            cluster_centers[(size_t)cluster_i],
                            cluster_extents[(size_t)cluster_i], center_j,
                            cluster_extents[(size_t)cluster_j], cutoff,
                            fixed_shift_candidates
                                ? shift_vectors[fixed_shift_id]
                                : shift_vectors[pair_shift_id]))
                    {
                        continue;
                    }
                    total_overlap_cluster_pairs += 1;
                    const int i_local = cluster_i - cluster_i_start;
                    imasks[pair_shift_id] |=
                        (1u << static_cast<unsigned int>(i_local));
                }

                int accepted_records_for_cluster_j = 0;
                int actual_unique_shift_id = -1;
                for (int shift_id = 0; shift_id < kClusteredShiftCount;
                     shift_id += 1)
                {
                    if (imasks[shift_id] == 0u)
                    {
                        continue;
                    }
                    if (actual_unique_shift_id < 0)
                    {
                        actual_unique_shift_id = shift_id;
                    }
                    accepted_records_for_cluster_j += 1;
                    total_accepted_records += 1;
                }
                if (accepted_records_for_cluster_j > 0)
                {
                    total_accepted_cluster_j += 1;
                    if (canonical_shift_id == kClusteredCentralShiftId)
                    {
                        total_canonical_central_cluster_j += 1;
                    }
                    else
                    {
                        total_canonical_shifted_cluster_j += 1;
                    }
                    if (actual_unique_shift_id == kClusteredCentralShiftId)
                    {
                        total_actual_central_cluster_j += 1;
                    }
                    else
                    {
                        total_actual_shifted_cluster_j += 1;
                    }
                    if (accepted_records_for_cluster_j == 1)
                    {
                        if (actual_unique_shift_id == canonical_shift_id)
                        {
                            total_canonical_shift_matches += 1;
                        }
                        else
                        {
                            total_canonical_shift_mismatches += 1;
                        }
                    }
                }
                accepted_records_per_sci += accepted_records_for_cluster_j;
            }
            processed_cluster_end = IntMax(processed_cluster_end, cluster_j_end);
        }

        total_raw_leaf_clusters += raw_leaf_clusters_per_sci;
        total_deduped_cluster_j += deduped_cluster_j_per_sci;
        total_valid_cluster_j += valid_cluster_j_per_sci;
        max_raw_leaf_clusters_per_sci =
            std::max(max_raw_leaf_clusters_per_sci, raw_leaf_clusters_per_sci);
        max_deduped_cluster_j_per_sci = std::max(max_deduped_cluster_j_per_sci,
                                                 deduped_cluster_j_per_sci);
        max_valid_cluster_j_per_sci =
            std::max(max_valid_cluster_j_per_sci, valid_cluster_j_per_sci);
        max_accepted_records_per_sci = std::max(max_accepted_records_per_sci,
                                                accepted_records_per_sci);
    }

    const double sci_scale = static_cast<double>(candidate_sci_numbers);
    const double dedup_ratio =
        total_deduped_cluster_j > 0
            ? static_cast<double>(total_raw_leaf_clusters) /
                  static_cast<double>(total_deduped_cluster_j)
            : 0.0;
    const double valid_ratio =
        total_deduped_cluster_j > 0
            ? static_cast<double>(total_valid_cluster_j) /
                  static_cast<double>(total_deduped_cluster_j)
            : 0.0;
    const double accepted_cluster_ratio =
        total_valid_cluster_j > 0
            ? static_cast<double>(total_accepted_cluster_j) /
                  static_cast<double>(total_valid_cluster_j)
            : 0.0;
    const double overlap_ratio =
        total_shift_candidates > 0
            ? static_cast<double>(total_overlap_cluster_pairs) /
                  static_cast<double>(total_shift_candidates)
            : 0.0;
    const double avg_records_per_accepted_cluster_j =
        total_accepted_cluster_j > 0
            ? static_cast<double>(total_accepted_records) /
                  static_cast<double>(total_accepted_cluster_j)
            : 0.0;
    const double avg_imask_bits_per_record =
        total_accepted_records > 0
            ? static_cast<double>(total_overlap_cluster_pairs) /
                  static_cast<double>(total_accepted_records)
            : 0.0;
    const double canonical_shift_match_ratio =
        (total_canonical_shift_matches + total_canonical_shift_mismatches) > 0
            ? static_cast<double>(total_canonical_shift_matches) /
                  static_cast<double>(total_canonical_shift_matches +
                                      total_canonical_shift_mismatches)
            : 0.0;

    fprintf(stderr,
            "[clustered builder trace] step=%d raw_leaf_clusters=%lld "
            "dedup_cluster_j=%lld valid_cluster_j=%lld accepted_cluster_j=%lld "
            "shift_candidates=%lld overlap_pairs=%lld accepted_records=%lld "
            "dup=%.3f valid=%.3f accepted=%.3f overlap=%.3f "
            "records_per_accept=%.3f bits_per_record=%.3f\n",
            md_info.sys.steps, total_raw_leaf_clusters, total_deduped_cluster_j,
            total_valid_cluster_j, total_accepted_cluster_j,
            total_shift_candidates, total_overlap_cluster_pairs,
            total_accepted_records, dedup_ratio, valid_ratio,
            accepted_cluster_ratio, overlap_ratio,
            avg_records_per_accepted_cluster_j, avg_imask_bits_per_record);
    fprintf(stderr,
            "[clustered builder trace] step=%d canonical_shift "
            "actual_central=%lld actual_shifted=%lld canonical_central=%lld "
            "canonical_shifted=%lld match=%lld mismatch=%lld "
            "match_ratio=%.6f\n",
            md_info.sys.steps, total_actual_central_cluster_j,
            total_actual_shifted_cluster_j, total_canonical_central_cluster_j,
            total_canonical_shifted_cluster_j, total_canonical_shift_matches,
            total_canonical_shift_mismatches, canonical_shift_match_ratio);
    fprintf(stderr,
            "[clustered builder trace] step=%d avg_raw=%.3f avg_dedup=%.3f "
            "avg_valid=%.3f avg_accept=%.3f avg_local_i=%.3f "
            "avg_super_span_frac=(%.5f,%.5f,%.5f) avg_cluster_radius=%.5f "
            "max_cluster_radius=%.5f max_raw=%d max_dedup=%d max_valid=%d "
            "max_records=%d max_local_i=%d\n",
            md_info.sys.steps,
            static_cast<double>(total_raw_leaf_clusters) / sci_scale,
            static_cast<double>(total_deduped_cluster_j) / sci_scale,
            static_cast<double>(total_valid_cluster_j) / sci_scale,
            static_cast<double>(total_accepted_cluster_j) / sci_scale,
            static_cast<double>(sum_active_local_i_clusters) / sci_scale,
            total_super_span_x / sci_scale, total_super_span_y / sci_scale,
            total_super_span_z / sci_scale,
            total_cluster_radius / static_cast<double>(layout.cluster_numbers),
            static_cast<double>(max_cluster_radius),
            max_raw_leaf_clusters_per_sci, max_deduped_cluster_j_per_sci,
            max_valid_cluster_j_per_sci, max_accepted_records_per_sci,
            max_active_local_i_clusters);
}
#endif

static void Refresh_Clustered_Pair_Shift_Metadata(LJ_CLUSTER_LAYOUT* layout,
                                                  LTMatrix3 rcell)
{
#ifdef USE_CPU
    (void)layout;
    (void)rcell;
#else
    if (layout == NULL || layout->sci_numbers <= 0 || layout->cjpacked_numbers <= 0 ||
        layout->d_pair_shift_bits == NULL || layout->d_nbnxm_sci == NULL ||
        layout->d_nbnxm_cjpacked == NULL)
    {
        return;
    }
    Launch_Device_Kernel(Refresh_Nbnxm_Pair_Shift_Bits, layout->sci_numbers,
                         CONTROLLER::device_max_thread, 0, NULL,
                         layout->sci_numbers, layout->d_super_cluster_offsets,
                         layout->d_cluster_centers, layout->d_nbnxm_sci,
                         layout->d_nbnxm_cjpacked, rcell,
                         layout->d_pair_shift_bits);
#endif
}

static void Refresh_Gmxpacked_Pair_Shift_Metadata(LJ_CLUSTER_LAYOUT* layout,
                                                  LTMatrix3 rcell)
{
#ifdef USE_CPU
    (void)layout;
    (void)rcell;
#else
    if (layout == NULL || layout->gmxpacked_sci_numbers <= 0 ||
        layout->gmxpacked_cjpacked_numbers <= 0 ||
        layout->d_pair_shift_bits == NULL || layout->d_gmxpacked_sci == NULL ||
        layout->d_gmxpacked_cjpacked == NULL)
    {
        return;
    }
    Launch_Device_Kernel(Refresh_Gmxpacked_Pair_Shift_Bits,
                         layout->gmxpacked_sci_numbers,
                         CONTROLLER::device_max_thread, 0, NULL,
                         layout->gmxpacked_sci_numbers,
                         layout->d_super_cluster_offsets,
                         layout->d_cluster_centers, layout->d_gmxpacked_sci,
                         layout->d_gmxpacked_cjpacked, rcell,
                         layout->d_pair_shift_bits);
#endif
}

static void Capture_Clustered_Outer_Imask(LJ_CLUSTER_LAYOUT* layout)
{
#ifdef USE_CPU
    (void)layout;
#else
    if (layout == NULL || layout->cjpacked_numbers <= 0 ||
        layout->d_nbnxm_cjpacked == NULL)
    {
        return;
    }
    Reserve_Device_UInt_Buffer(
        layout->cjpacked_numbers * kClusteredWarpSplitCount,
        &layout->d_outer_imask, &layout->outer_imask_capacity);
    Launch_Device_Kernel(
        Snapshot_Clustered_Outer_Imask,
        (layout->cjpacked_numbers * kClusteredWarpSplitCount +
         CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout->cjpacked_numbers,
        layout->d_nbnxm_cjpacked, layout->d_outer_imask);
#endif
}

static void Rebuild_Clustered_Forceonly_Warp_Records(LJ_CLUSTER_LAYOUT* layout)
{
#ifdef USE_CPU
    (void)layout;
#else
    if (layout == NULL)
    {
        return;
    }
    layout->forceonly_warp_record_numbers = 0;
    if (layout->sci_numbers <= 0 || layout->cjpacked_numbers <= 0 ||
        layout->d_nbnxm_sci == NULL || layout->d_nbnxm_cjpacked == NULL)
    {
        return;
    }

    Reserve_Device_Int_Buffer(layout->sci_numbers,
                              &layout->d_forceonly_warp_record_counts,
                              &layout->forceonly_warp_record_count_capacity);
    Reserve_Device_Int_Buffer(layout->sci_numbers + 1,
                              &layout->d_forceonly_warp_record_offsets,
                              &layout->forceonly_warp_record_offset_capacity);
    Launch_Device_Kernel(
        Count_Active_Nbnxm_Warp_J_Records,
        (layout->sci_numbers + kClusteredBuilderBlockSize - 1) /
            kClusteredBuilderBlockSize,
        kClusteredBuilderBlockSize, 0, NULL, layout->sci_numbers,
        layout->d_nbnxm_sci, layout->d_nbnxm_cjpacked,
        layout->d_forceonly_warp_record_counts);
    layout->forceonly_warp_record_numbers = Exclusive_Scan_Counts(
        layout, layout->sci_numbers, layout->d_forceonly_warp_record_counts,
        layout->d_forceonly_warp_record_offsets);
    if (layout->forceonly_warp_record_numbers > 0)
    {
        Reserve_Device_Buffer(layout->forceonly_warp_record_numbers,
                              &layout->d_forceonly_warp_j_records,
                              &layout->forceonly_warp_record_capacity);
        const bool direct_sort_forceonly =
            Clustered_Direct_Sort_Forceonly_Records_Enabled();
        if (direct_sort_forceonly)
        {
            Launch_Device_Kernel(
                Fill_Sorted_Active_Nbnxm_Warp_J_Records,
                layout->sci_numbers, kClusteredBuilderBlockSize, 0, NULL,
                layout->sci_numbers, layout->d_super_cluster_offsets,
                layout->d_cluster_offsets, layout->d_cluster_valid_masks,
                layout->d_cluster_local_masks, layout->d_nbnxm_sci,
                layout->d_nbnxm_cjpacked, layout->d_exclusion_mask_pool,
                layout->d_forceonly_warp_record_offsets,
                layout->d_forceonly_warp_j_records);
        }
        else
        {
            Launch_Device_Kernel(
                Fill_Active_Nbnxm_Warp_J_Records, layout->sci_numbers,
                kClusteredBuilderBlockSize, 0, NULL, layout->sci_numbers,
                layout->d_super_cluster_offsets, layout->d_cluster_offsets,
                layout->d_cluster_valid_masks, layout->d_cluster_local_masks,
                layout->d_nbnxm_sci, layout->d_nbnxm_cjpacked,
                layout->d_exclusion_mask_pool,
                layout->d_forceonly_warp_record_offsets,
                layout->d_forceonly_warp_j_records);
        }
        if (!direct_sort_forceonly && Clustered_Sort_Forceonly_Records_Enabled())
        {
            Reorder_Forceonly_Warp_Records_By_Cluster_J(layout);
        }
    }
    if (Clustered_Trace_Warp_Records_Enabled())
    {
        int tail_offset = 0;
        deviceMemcpy(&tail_offset,
                     layout->d_forceonly_warp_record_offsets +
                         layout->sci_numbers,
                     sizeof(int), deviceMemcpyDeviceToHost);
        fprintf(stderr,
                "[clustered warp-record build] step=%d sci=%d cjpacked=%d "
                "records=%d tail=%d\n",
                md_info.sys.steps, layout->sci_numbers, layout->cjpacked_numbers,
                layout->forceonly_warp_record_numbers, tail_offset);
    }
#endif
}

static void Build_Grouped_Sci_Metadata(LJ_CLUSTER_LAYOUT* layout)
{
#ifdef USE_CPU
    (void)layout;
#else
    if (layout == NULL || layout->grouped_sci_ready ||
        layout->super_cluster_numbers <= 0 ||
        layout->sci_numbers <= 0)
    {
        return;
    }
    Reserve_Device_Int_Buffer(layout->super_cluster_numbers + 1,
                              &layout->d_grouped_sci_offsets,
                              &layout->grouped_sci_offset_capacity);
    Reserve_Device_Int_Buffer(layout->sci_numbers, &layout->d_grouped_sci_ids,
                              &layout->grouped_sci_id_capacity);
    Reserve_Device_Int_Buffer(layout->super_cluster_numbers + 1,
                              &layout->d_candidate_sci_offsets,
                              &layout->candidate_offset_capacity);
    deviceMemset(layout->d_candidate_sci_offsets, 0,
                 sizeof(int) * (layout->super_cluster_numbers + 1));
    Launch_Device_Kernel(
        Count_Final_Sci_Per_Supercluster,
        (layout->sci_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout->sci_numbers,
        layout->d_nbnxm_sci, layout->d_candidate_sci_offsets);
    Exclusive_Scan_Counts(layout, layout->super_cluster_numbers,
                          layout->d_candidate_sci_offsets,
                          layout->d_grouped_sci_offsets);
    deviceMemcpy(layout->d_candidate_sci_offsets, layout->d_grouped_sci_offsets,
                 sizeof(int) * layout->super_cluster_numbers,
                 deviceMemcpyDeviceToDevice);
    Launch_Device_Kernel(
        Fill_Final_Sci_Groups_By_Supercluster,
        (layout->sci_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout->sci_numbers,
        layout->d_nbnxm_sci, layout->d_candidate_sci_offsets,
        layout->d_grouped_sci_ids);
    layout->grouped_sci_ready = true;
#endif
}

static bool Clustered_Build_Is_Needed(LJ_CLUSTER_LAYOUT* layout,
                                      const VECTOR* crd, const LTMatrix3 cell,
                                      const LTMatrix3 rcell, const float cutoff)
{
    if (layout->rebuild_dirty || !layout->cache_ready ||
        layout->d_cached_crd == NULL)
    {
        return true;
    }
    if (fabsf(layout->cached_cutoff - cutoff) > 1e-6f)
    {
        return true;
    }
    if (layout->rebuild_refresh_interval > 0)
    {
        const int steps_since_build =
            md_info.sys.steps - layout->cached_build_step;
        if (steps_since_build < layout->rebuild_refresh_interval)
        {
            return false;
        }
        return true;
    }
    if (layout->rebuild_skin <= 0.0f || layout->rebuild_skin_permit <= 0.0f ||
        layout->local_atom_numbers <= 0)
    {
        return true;
    }

    Reserve_Device_Int_Buffer(1, &layout->d_need_rebuild,
                              &layout->rebuild_flag_capacity);
    int h_need_rebuild = 0;
    deviceMemcpy(layout->d_need_rebuild, &h_need_rebuild, sizeof(int),
                 deviceMemcpyHostToDevice);
    const float permit = layout->rebuild_skin * layout->rebuild_skin_permit;
    Launch_Device_Kernel(Check_Clustered_Rebuild,
                         (layout->local_atom_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         layout->local_atom_numbers, crd, layout->d_cached_crd,
                         cell, rcell, layout->d_need_rebuild, permit * permit);
    deviceMemcpy(&h_need_rebuild, layout->d_need_rebuild, sizeof(int),
                 deviceMemcpyDeviceToHost);
    return h_need_rebuild != 0;
}

static void Initialize_Cornerstone_State(LJ_CLUSTER_LAYOUT* layout)
{
    if (layout->cornerstone_state == NULL)
    {
        layout->cornerstone_state = new LJ_CORNERSTONE_STATE();
    }
}

static bool Prepare_Atom_To_Molecule_Metadata(LJ_CLUSTER_LAYOUT* layout)
{
    if (layout == NULL || layout->total_atom_numbers <= 0 ||
        md_info.atom_numbers <= 0 || md_info.mol.molecule_numbers <= 0 ||
        md_info.mol.h_atom_start == NULL || md_info.mol.h_atom_end == NULL)
    {
        return false;
    }

    std::vector<int> host_global_atom_to_molecule((size_t)md_info.atom_numbers,
                                                  -1);
    for (int molecule = 0; molecule < md_info.mol.molecule_numbers;
         molecule += 1)
    {
        const int atom_start = md_info.mol.h_atom_start[molecule];
        const int atom_end = md_info.mol.h_atom_end[molecule];
        for (int atom = atom_start; atom < atom_end; atom += 1)
        {
            if (atom >= 0 && atom < md_info.atom_numbers)
            {
                host_global_atom_to_molecule[(size_t)atom] = molecule;
            }
        }
    }

    Reserve_Device_Int_Buffer(md_info.atom_numbers,
                              &layout->d_global_atom_to_molecule,
                              &layout->global_atom_to_molecule_capacity);
    deviceMemcpy(layout->d_global_atom_to_molecule,
                 host_global_atom_to_molecule.data(),
                 sizeof(int) * md_info.atom_numbers,
                 deviceMemcpyHostToDevice);

    Reserve_Device_Int_Buffer(layout->total_atom_numbers,
                              &layout->d_local_atom_to_molecule,
                              &layout->local_atom_to_molecule_capacity);
    Launch_Device_Kernel(
        Build_Local_Atom_To_Molecule_Map,
        (layout->total_atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout->total_atom_numbers,
        md_info.atom_numbers, layout->d_atom_local,
        layout->d_global_atom_to_molecule, layout->d_local_atom_to_molecule);
    return true;
}

static void Reset_Build_Buffers(LJ_CLUSTER_LAYOUT* layout)
{
    // Rebuilds are expected to be frequent. Keep device buffers alive and let
    // the reserve helpers grow them only when the topology actually demands it.
    // Full release still happens in LJ_CLUSTER_LAYOUT::Clear().
}

static void Reset_Cornerstone_Root(LJ_CORNERSTONE_STATE* state,
                                   int total_atom_numbers)
{
    std::vector<CornerstoneKey> root_leaves = {0ull,
                                               cstone::nodeRange<CornerstoneKey>(
                                                   0)};
    std::vector<unsigned> root_counts = {
        static_cast<unsigned>(total_atom_numbers)};
#ifndef USE_CPU
    state->leaves = root_leaves;
    state->leaf_counts = root_counts;
    state->tmp_leaves.resize(0);
    state->work_array.resize(0);
#else
    state->leaves = std::move(root_leaves);
    state->leaf_counts = std::move(root_counts);
#endif
}

static void Build_Cornerstone_Tree(LJ_CLUSTER_LAYOUT* layout)
{
    if (layout->total_atom_numbers <= 0)
    {
        return;
    }

    if (layout->cornerstone_state != NULL)
    {
        delete layout->cornerstone_state;
        layout->cornerstone_state = NULL;
    }
    layout->cornerstone_state = new LJ_CORNERSTONE_STATE();
    auto* state = layout->cornerstone_state;

    Reset_Cornerstone_Root(state, layout->total_atom_numbers);

#ifndef USE_CPU
    const auto* sorted_keys = layout->d_sort_keys;
    const std::span<const CornerstoneKey> key_span(
        sorted_keys, static_cast<size_t>(layout->total_atom_numbers));
    bool converged = false;
    for (int iter = 0; iter < 64 && !converged; iter += 1)
    {
        converged = cstone::updateOctreeGpu<CornerstoneKey>(
            key_span, static_cast<unsigned>(layout->cornerstone_leaf_size),
            state->leaves, state->leaf_counts, state->tmp_leaves,
            state->work_array);
    }
    state->octree.resize(static_cast<CornerstoneNodeIndex>(
        cstone::nNodes(state->leaves)));
    cstone::buildOctreeGpu(rawPtr(state->leaves), state->octree.data());
#else
    std::vector<CornerstoneKey> host_keys(
        static_cast<size_t>(layout->total_atom_numbers));
    deviceMemcpy(host_keys.data(), layout->d_sort_keys,
                 sizeof(CornerstoneKey) * layout->total_atom_numbers,
                 deviceMemcpyDeviceToHost);
    const std::span<const CornerstoneKey> key_span(host_keys.data(),
                                                   host_keys.size());
    bool converged = false;
    for (int iter = 0; iter < 64 && !converged; iter += 1)
    {
        converged = cstone::updateOctree<CornerstoneKey>(
            key_span, static_cast<unsigned>(layout->cornerstone_leaf_size),
            state->leaves, state->leaf_counts);
    }
    state->octree.resize(static_cast<CornerstoneNodeIndex>(
        cstone::nNodes(state->leaves)));
    cstone::updateInternalTree<CornerstoneKey>(
        std::span<const CornerstoneKey>(state->leaves.data(),
                                        state->leaves.size()),
        state->octree.data());
#endif
}

static int Exclusive_Scan_Counts(LJ_CLUSTER_LAYOUT* layout, int count_numbers,
                                 int* d_counts, int* d_starts)
{
    if (count_numbers <= 0)
    {
        return 0;
    }
#ifndef USE_CPU
    if (layout == NULL)
    {
        return 0;
    }
    Bind_Clustered_Working_Device(&layout->working_device);

    size_t reduce_storage_bytes = 0;
    size_t scan_storage_bytes = 0;

    if (layout->d_scan_total == NULL)
    {
        Clustered_Device_Malloc_Safely((void**)&layout->d_scan_total,
                                       sizeof(int), "scan-total");
    }
    cub::DeviceReduce::Sum(NULL, reduce_storage_bytes, d_counts,
                           layout->d_scan_total, count_numbers);
    cub::DeviceScan::ExclusiveSum(NULL, scan_storage_bytes, d_counts, d_starts,
                                  count_numbers);

    Reserve_Device_Opaque_Buffer(reduce_storage_bytes,
                                 &layout->d_reduce_temp_storage,
                                 &layout->reduce_temp_storage_bytes);
    Reserve_Device_Opaque_Buffer(scan_storage_bytes,
                                 &layout->d_scan_temp_storage,
                                 &layout->scan_temp_storage_bytes);

    cub::DeviceReduce::Sum(layout->d_reduce_temp_storage, reduce_storage_bytes,
                           d_counts, layout->d_scan_total, count_numbers);
    cub::DeviceScan::ExclusiveSum(layout->d_scan_temp_storage,
                                  scan_storage_bytes, d_counts, d_starts,
                                  count_numbers);
    deviceMemcpy(d_starts + count_numbers, layout->d_scan_total, sizeof(int),
                 deviceMemcpyDeviceToDevice);

    int total = 0;
    deviceMemcpy(&total, layout->d_scan_total, sizeof(int),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(d_starts + count_numbers, &total, sizeof(int),
                 deviceMemcpyHostToDevice);
    return total;
#else
    std::vector<int> h_counts(static_cast<size_t>(count_numbers));
    std::vector<int> h_starts(static_cast<size_t>(count_numbers) + 1, 0);
    deviceMemcpy(h_counts.data(), d_counts, sizeof(int) * count_numbers,
                 deviceMemcpyDeviceToHost);
    int total = 0;
    for (int i = 0; i < count_numbers; i += 1)
    {
        h_starts[static_cast<size_t>(i)] = total;
        total += h_counts[static_cast<size_t>(i)];
    }
    h_starts[static_cast<size_t>(count_numbers)] = total;
    deviceMemcpy(d_starts, h_starts.data(),
                 sizeof(int) * (count_numbers + 1), deviceMemcpyHostToDevice);
    return total;
#endif
}

#ifndef USE_CPU
static void Stable_Sort_Sci_By_Workload(int sci_numbers,
                                        LJ_CLUSTERED_SCI* d_sci_entries,
                                        int* d_sci_sort_keys,
                                        LJ_CLUSTER_LAYOUT* layout)
{
    if (sci_numbers <= 1 || d_sci_entries == NULL || d_sci_sort_keys == NULL)
    {
        return;
    }
    Launch_Device_Kernel(Build_Sci_Workload_Sort_Keys,
                         (sci_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, sci_numbers,
                         d_sci_entries, d_sci_sort_keys);
    Stable_Sort_Device_By_Key(layout, sci_numbers, d_sci_sort_keys,
                              d_sci_entries);
}
#endif

static void Reset_Gmxpacked_Payload(LJ_CLUSTER_LAYOUT* layout)
{
    if (layout == NULL)
    {
        return;
    }
    layout->gmxpacked_sci_numbers = 0;
    layout->gmxpacked_cjpacked_numbers = 0;
    layout->gmxpacked_exclusion_numbers = 0;
    layout->gmxpacked_split_exclusion_numbers = 0;
}

static void Free_Gmxpacked_Payload(LJ_CLUSTER_LAYOUT* layout)
{
    if (layout == NULL)
    {
        return;
    }
    Free_Single_Device_Pointer((void**)&layout->d_gmxpacked_sci);
    Free_Single_Device_Pointer((void**)&layout->d_gmxpacked_cjpacked);
    Free_Single_Device_Pointer((void**)&layout->d_gmxpacked_exclusions);
    layout->gmxpacked_sci_capacity = 0;
    layout->gmxpacked_cjpacked_capacity = 0;
    layout->gmxpacked_exclusion_capacity = 0;
    Reset_Gmxpacked_Payload(layout);
}

#ifndef USE_CPU
static __device__ __forceinline__ LJ_CLUSTERED_GMXPACKED_EXCLUSION
Build_Gmxpacked_Compact_Exclusion_Row(
    const LJ_CLUSTERED_SCI& native_sci_entry,
    const LJ_CLUSTERED_CJ_PACKED& native_packed, const int split_idx,
    const int cluster_numbers, const int* super_cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const unsigned long long* exclusion_mask_pool,
    const int exclusion_pool_numbers)
{
    LJ_CLUSTERED_GMXPACKED_EXCLUSION compact_exclusion = {};
#pragma unroll
    for (int pair_idx = 0; pair_idx < kClusteredGmxpackedExclusionPairCount;
         pair_idx += 1)
    {
        compact_exclusion.pair[pair_idx] = 0u;
    }

    const int cluster_i_start =
        super_cluster_offsets[native_sci_entry.supercluster_id];
#pragma unroll
    for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
    {
        const int cluster_j = native_packed.cj[jm];
        if (cluster_j < 0)
        {
            continue;
        }

        const unsigned int jm_imask =
            Clustered_Jm_Imask(native_packed.imei[split_idx], jm);
        if (jm_imask == 0u)
        {
            continue;
        }

        const unsigned int valid_mask_j = cluster_valid_masks[cluster_j];
        const unsigned int local_mask_j = cluster_local_masks[cluster_j];
#pragma unroll
        for (int i_local = 0; i_local < kClusteredSuperClusterClusters;
             i_local += 1)
        {
            const unsigned int imask_bit =
                1u << static_cast<unsigned int>(i_local);
            if ((jm_imask & imask_bit) == 0u)
            {
                continue;
            }

            const int cluster_i = cluster_i_start + i_local;
            const unsigned int local_mask_i =
                cluster_i >= 0 && cluster_i < cluster_numbers
                    ? cluster_local_masks[cluster_i]
                    : 0u;
            const int native_exclusion_index = Clustered_Exclusion_Index(
                native_packed.imei[split_idx], jm, i_local);
            const unsigned long long native_exclusion_mask =
                native_exclusion_index >= 0 &&
                        native_exclusion_index < exclusion_pool_numbers &&
                        exclusion_mask_pool != NULL
                    ? exclusion_mask_pool[native_exclusion_index]
                    : 0ull;
            const unsigned int packed_bit =
                1u << static_cast<unsigned int>(
                          jm * kClusteredSuperClusterClusters + i_local);

#pragma unroll
            for (int split_j_lane = 0;
                 split_j_lane < kClusteredSplitJClusterSize;
                 split_j_lane += 1)
            {
                const int j_lane =
                    split_idx * kClusteredSplitJClusterSize + split_j_lane;
                const bool valid_j =
                    (valid_mask_j &
                     (1u << static_cast<unsigned int>(j_lane))) != 0u;
                const bool local_j =
                    (local_mask_j &
                     (1u << static_cast<unsigned int>(j_lane))) != 0u;
#pragma unroll
                for (int i_lane = 0; i_lane < kClusteredClusterSize;
                     i_lane += 1)
                {
                    const bool local_i =
                        (local_mask_i &
                         (1u << static_cast<unsigned int>(i_lane))) != 0u;
                    bool allow_pair = valid_j && local_i;
                    if (allow_pair &&
                        native_sci_entry.shift_id == kClusteredCentralShiftId &&
                        cluster_i == cluster_j && local_j && j_lane <= i_lane)
                    {
                        allow_pair = false;
                    }
                    if (allow_pair &&
                        (native_exclusion_mask &
                         (1ull << static_cast<unsigned int>(
                              i_lane * kClusteredClusterSize + j_lane))) != 0ull)
                    {
                        allow_pair = false;
                    }
                    if (allow_pair)
                    {
                        compact_exclusion
                            .pair[split_j_lane * kClusteredClusterSize +
                                  i_lane] |= packed_bit;
                    }
                }
            }
        }
    }
    return compact_exclusion;
}

static __device__ __forceinline__ bool Gmxpacked_Exclusion_Row_Is_Needed(
    const LJ_CLUSTERED_GMXPACKED_EXCLUSION& compact_exclusion,
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

static __global__ void Count_Gmxpacked_CjPacked_Per_Sci(
    const int sci_numbers, const int* compact_sci_starts,
    int* cjpacked_counts)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const int entry_count = compact_sci_starts[sci + 1] - compact_sci_starts[sci];
        cjpacked_counts[sci] =
            entry_count > 0
                ? (entry_count + kClusteredJGroupSize - 1) /
                      kClusteredJGroupSize
                : 0;
    }
}

static __device__ __forceinline__ LJ_CLUSTERED_GMXPACKED_CJ
Make_Empty_Gmxpacked_CjPacked()
{
    LJ_CLUSTERED_GMXPACKED_CJ compact_packed = {};
#pragma unroll
    for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
    {
        compact_packed.cj[jm] = -1;
    }
    return compact_packed;
}

static __global__ void Fill_Gmxpacked_Sci_And_Cj(
    const int sci_numbers, const int* compact_sci_starts,
    const LJ_CLUSTERED_J_ENTRY* compact_j_entries,
    const int* compact_cjpacked_offsets,
    LJ_CLUSTERED_GMXPACKED_SCI* gmxpacked_sci,
    LJ_CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked)
{
    SIMPLE_DEVICE_FOR(sci, sci_numbers)
    {
        const int entry_begin = compact_sci_starts[sci];
        const int entry_end = compact_sci_starts[sci + 1];
        if (entry_begin >= entry_end)
        {
            return;
        }

        const LJ_CLUSTERED_J_ENTRY first_entry = compact_j_entries[entry_begin];
        const int compact_begin = compact_cjpacked_offsets[sci];
        const int compact_end = compact_cjpacked_offsets[sci + 1];
        gmxpacked_sci[sci] = {first_entry.supercluster_id,
                              first_entry.shift_id, compact_begin,
                              compact_end};

        const int packed_count = compact_end - compact_begin;
        for (int local_packed = 0; local_packed < packed_count;
             local_packed += 1)
        {
            const int packed_entry_begin =
                entry_begin + local_packed * kClusteredJGroupSize;
            const int packed_entry_end =
                IntMin(entry_end, packed_entry_begin + kClusteredJGroupSize);
            LJ_CLUSTERED_GMXPACKED_CJ compact_packed =
                Make_Empty_Gmxpacked_CjPacked();
            for (int entry_idx = packed_entry_begin; entry_idx < packed_entry_end;
                 entry_idx += 1)
            {
                const int jm = entry_idx - packed_entry_begin;
                const LJ_CLUSTERED_J_ENTRY& entry = compact_j_entries[entry_idx];
                compact_packed.cj[jm] = entry.cluster_j;
                for (int split_idx = 0; split_idx < kClusteredWarpSplitCount;
                     split_idx += 1)
                {
                    const unsigned int imask = entry.imask[split_idx];
                    if (imask == 0u)
                    {
                        continue;
                    }
                    compact_packed.split[split_idx].imask |=
                        imask << Clustered_Jm_Imask_Shift(jm);
                }
            }
            gmxpacked_cjpacked[compact_begin + local_packed] = compact_packed;
        }
    }
}

static __device__ __forceinline__ LJ_CLUSTERED_GMXPACKED_EXCLUSION
Build_Gmxpacked_Compact_Exclusion_Row_From_J_Entries(
    const LJ_CLUSTERED_GMXPACKED_SCI& compact_sci_entry,
    const LJ_CLUSTERED_J_ENTRY* compact_j_entries, const int entry_begin,
    const int entry_end, const int split_idx, const int cluster_numbers,
    const int* super_cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const unsigned long long* exclusion_mask_pool,
    const int exclusion_pool_numbers)
{
    LJ_CLUSTERED_GMXPACKED_EXCLUSION compact_exclusion = {};
#pragma unroll
    for (int pair_idx = 0; pair_idx < kClusteredGmxpackedExclusionPairCount;
         pair_idx += 1)
    {
        compact_exclusion.pair[pair_idx] = 0u;
    }

    const int cluster_i_start =
        super_cluster_offsets[compact_sci_entry.supercluster_id];
    for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
    {
        const int entry_idx = entry_begin + jm;
        if (entry_idx >= entry_end)
        {
            break;
        }

        const LJ_CLUSTERED_J_ENTRY& entry = compact_j_entries[entry_idx];
        const int cluster_j = entry.cluster_j;
        if (cluster_j < 0)
        {
            continue;
        }

        const unsigned int jm_imask = entry.imask[split_idx];
        if (jm_imask == 0u)
        {
            continue;
        }

        const unsigned int valid_mask_j = cluster_valid_masks[cluster_j];
        const unsigned int local_mask_j = cluster_local_masks[cluster_j];
        for (int i_local = 0; i_local < kClusteredSuperClusterClusters;
             i_local += 1)
        {
            const unsigned int imask_bit =
                1u << static_cast<unsigned int>(i_local);
            if ((jm_imask & imask_bit) == 0u)
            {
                continue;
            }

            const int cluster_i = cluster_i_start + i_local;
            const unsigned int local_mask_i =
                cluster_i >= 0 && cluster_i < cluster_numbers
                    ? cluster_local_masks[cluster_i]
                    : 0u;
            const int native_exclusion_index =
                Clustered_J_Entry_Exclusion_Index(entry, split_idx, i_local);
            const unsigned long long native_exclusion_mask =
                native_exclusion_index >= 0 &&
                        native_exclusion_index < exclusion_pool_numbers &&
                        exclusion_mask_pool != NULL
                    ? exclusion_mask_pool[native_exclusion_index]
                    : 0ull;
            const unsigned int packed_bit =
                1u << static_cast<unsigned int>(
                          jm * kClusteredSuperClusterClusters + i_local);

            for (int split_j_lane = 0;
                 split_j_lane < kClusteredSplitJClusterSize;
                 split_j_lane += 1)
            {
                const int j_lane =
                    split_idx * kClusteredSplitJClusterSize + split_j_lane;
                const bool valid_j =
                    (valid_mask_j &
                     (1u << static_cast<unsigned int>(j_lane))) != 0u;
                const bool local_j =
                    (local_mask_j &
                     (1u << static_cast<unsigned int>(j_lane))) != 0u;
                for (int i_lane = 0; i_lane < kClusteredClusterSize;
                     i_lane += 1)
                {
                    const bool local_i =
                        (local_mask_i &
                         (1u << static_cast<unsigned int>(i_lane))) != 0u;
                    bool allow_pair = valid_j && local_i;
                    if (allow_pair &&
                        compact_sci_entry.shift_id == kClusteredCentralShiftId &&
                        cluster_i == cluster_j && local_j && j_lane <= i_lane)
                    {
                        allow_pair = false;
                    }
                    if (allow_pair &&
                        (native_exclusion_mask &
                         (1ull << static_cast<unsigned int>(
                              i_lane * kClusteredClusterSize + j_lane))) != 0ull)
                    {
                        allow_pair = false;
                    }
                    if (allow_pair)
                    {
                        compact_exclusion
                            .pair[split_j_lane * kClusteredClusterSize +
                                  i_lane] |= packed_bit;
                    }
                }
            }
        }
    }
    return compact_exclusion;
}

static __global__ void Count_Gmxpacked_Split_Exclusions(
    const int sci_numbers, const int cluster_numbers,
    const LJ_CLUSTERED_GMXPACKED_SCI* gmxpacked_sci,
    const LJ_CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked,
    const LJ_CLUSTERED_J_ENTRY* compact_j_entries,
    const int* compact_sci_starts, const int* super_cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const unsigned long long* exclusion_mask_pool,
    const int exclusion_pool_numbers, int* split_exclusion_counts)
{
    const int sci = blockIdx.x;
    if (sci >= sci_numbers)
    {
        return;
    }

    const LJ_CLUSTERED_GMXPACKED_SCI compact_sci_entry = gmxpacked_sci[sci];
    const int compact_begin = compact_sci_entry.cjpacked_begin;
    const int packed_count = compact_sci_entry.cjpacked_end - compact_begin;
    const int sci_entry_begin = compact_sci_starts[sci];
    const int sci_entry_end = compact_sci_starts[sci + 1];
    const int split_entry_count = packed_count * kClusteredWarpSplitCount;
    for (int local_split_entry = threadIdx.x;
         local_split_entry < split_entry_count;
         local_split_entry += blockDim.x)
    {
        const int local_packed =
            local_split_entry / kClusteredWarpSplitCount;
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
            const LJ_CLUSTERED_GMXPACKED_EXCLUSION compact_exclusion =
                Build_Gmxpacked_Compact_Exclusion_Row_From_J_Entries(
                    compact_sci_entry, compact_j_entries, packed_entry_begin,
                    packed_entry_end, split_idx, cluster_numbers,
                    super_cluster_offsets, cluster_valid_masks,
                    cluster_local_masks, exclusion_mask_pool,
                    exclusion_pool_numbers);
            needs_exclusion =
                Gmxpacked_Exclusion_Row_Is_Needed(compact_exclusion,
                                                  split_imask)
                    ? 1
                    : 0;
        }
        split_exclusion_counts[compact_split_idx] = needs_exclusion;
    }
}

static __global__ void Initialize_Gmxpacked_Exclusion_Rows(
    const int exclusion_numbers, LJ_CLUSTERED_GMXPACKED_EXCLUSION* exclusions)
{
    SIMPLE_DEVICE_FOR(exclusion_idx, exclusion_numbers)
    {
        LJ_CLUSTERED_GMXPACKED_EXCLUSION exclusion = {};
        const unsigned int fill_value =
            exclusion_idx == 0 ? 0xffffffffu : 0u;
#pragma unroll
        for (int pair_idx = 0; pair_idx < kClusteredGmxpackedExclusionPairCount;
             pair_idx += 1)
        {
            exclusion.pair[pair_idx] = fill_value;
        }
        exclusions[exclusion_idx] = exclusion;
    }
}

static __global__ void Fill_Gmxpacked_Split_Exclusions(
    const int sci_numbers, const int cluster_numbers,
    const LJ_CLUSTERED_GMXPACKED_SCI* gmxpacked_sci,
    const LJ_CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked_read,
    const LJ_CLUSTERED_J_ENTRY* compact_j_entries,
    const int* compact_sci_starts, const int* split_exclusion_offsets,
    const int* super_cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const unsigned long long* exclusion_mask_pool,
    const int exclusion_pool_numbers, LJ_CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked,
    LJ_CLUSTERED_GMXPACKED_EXCLUSION* gmxpacked_exclusions)
{
    const int sci = blockIdx.x;
    if (sci >= sci_numbers)
    {
        return;
    }

    const LJ_CLUSTERED_GMXPACKED_SCI compact_sci_entry = gmxpacked_sci[sci];
    const int compact_begin = compact_sci_entry.cjpacked_begin;
    const int packed_count = compact_sci_entry.cjpacked_end - compact_begin;
    const int sci_entry_begin = compact_sci_starts[sci];
    const int sci_entry_end = compact_sci_starts[sci + 1];
    const int split_entry_count = packed_count * kClusteredWarpSplitCount;
    for (int local_split_entry = threadIdx.x;
         local_split_entry < split_entry_count;
         local_split_entry += blockDim.x)
    {
        const int local_packed =
            local_split_entry / kClusteredWarpSplitCount;
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
        const LJ_CLUSTERED_GMXPACKED_EXCLUSION compact_exclusion =
            Build_Gmxpacked_Compact_Exclusion_Row_From_J_Entries(
                compact_sci_entry, compact_j_entries, packed_entry_begin,
                packed_entry_end, split_idx, cluster_numbers,
                super_cluster_offsets, cluster_valid_masks, cluster_local_masks,
                exclusion_mask_pool, exclusion_pool_numbers);
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

static ClusteredReducedStagedCountSummary
Analyze_Reduced_Gmxpacked_Count_And_Scan_On_Host(
    const std::vector<LJ_CLUSTERED_SCI>& native_sci,
    const std::vector<LJ_CLUSTERED_CJ_PACKED>& native_cjpacked)
{
    ClusteredReducedStagedCountSummary summary = {};
    std::vector<std::pair<int, int>> reduced_keys;
    reduced_keys.reserve(native_cjpacked.size() *
                         static_cast<size_t>(kClusteredJGroupSize));

    for (const LJ_CLUSTERED_SCI& native_sci_entry : native_sci)
    {
        for (int packed_idx = native_sci_entry.cjpacked_begin;
             packed_idx < native_sci_entry.cjpacked_end; packed_idx += 1)
        {
            const LJ_CLUSTERED_CJ_PACKED& packed =
                native_cjpacked[static_cast<size_t>(packed_idx)];
            for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
            {
                const int cluster_j = packed.cj[jm];
                const unsigned int combined_imask =
                    Clustered_Jm_Imask(packed.imei[0], jm) |
                    Clustered_Jm_Imask(packed.imei[1], jm);
                if (cluster_j >= 0 && combined_imask != 0u)
                {
                    reduced_keys.emplace_back(native_sci_entry.supercluster_id,
                                              cluster_j);
                }
            }
        }
    }

    summary.source_entries = static_cast<int>(reduced_keys.size());
    if (reduced_keys.empty())
    {
        return summary;
    }

    std::sort(reduced_keys.begin(), reduced_keys.end());
    summary.reduced_entries = 1;
    summary.reduced_sci_numbers = 1;
    int current_supercluster = reduced_keys[0].first;
    int current_cluster_j = reduced_keys[0].second;
    int reduced_entries_in_sci = 1;

    for (size_t key_idx = 1; key_idx < reduced_keys.size(); key_idx += 1)
    {
        const std::pair<int, int>& key = reduced_keys[key_idx];
        if (key.first != current_supercluster)
        {
            summary.predicted_compact_cj_numbers +=
                (reduced_entries_in_sci + kClusteredJGroupSize - 1) /
                kClusteredJGroupSize;
            summary.reduced_entries += 1;
            summary.reduced_sci_numbers += 1;
            reduced_entries_in_sci = 1;
            current_supercluster = key.first;
            current_cluster_j = key.second;
            continue;
        }
        if (key.second != current_cluster_j)
        {
            summary.reduced_entries += 1;
            reduced_entries_in_sci += 1;
            current_cluster_j = key.second;
        }
    }
    summary.predicted_compact_cj_numbers +=
        (reduced_entries_in_sci + kClusteredJGroupSize - 1) /
        kClusteredJGroupSize;

    return summary;
}

#ifndef USE_CPU
static ClusteredReducedStagedCountSummary
Analyze_Reduced_Gmxpacked_Count_And_Scan_On_Device(LJ_CLUSTER_LAYOUT* layout)
{
    ClusteredReducedStagedCountSummary summary = {};
    if (layout == NULL || layout->sci_numbers <= 0 ||
        layout->cjpacked_numbers <= 0 || layout->d_nbnxm_sci == NULL ||
        layout->d_nbnxm_cjpacked == NULL)
    {
        return summary;
    }

    Reserve_Device_Int_Buffer(layout->sci_numbers + 1, &layout->d_jentry_counts,
                              &layout->jentry_count_capacity);
    Reserve_Device_Int_Buffer(layout->sci_numbers + 1,
                              &layout->d_jentry_offsets,
                              &layout->jentry_offset_capacity);
    Launch_Device_Kernel(
        Count_Cj_Entries_Per_Sci,
        (layout->sci_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout->sci_numbers,
        layout->d_nbnxm_sci, layout->d_nbnxm_cjpacked,
        layout->d_jentry_counts);
    summary.source_entries = Exclusive_Scan_Counts(
        layout, layout->sci_numbers, layout->d_jentry_counts,
        layout->d_jentry_offsets);
    if (summary.source_entries <= 0)
    {
        return summary;
    }

    Reserve_Device_Buffer(summary.source_entries, &layout->d_j_entries,
                          &layout->jentry_capacity);
    Reserve_Device_Buffer(summary.source_entries, &layout->d_j_entry_buffer,
                          &layout->jentry_buffer_capacity);
    Launch_Device_Kernel(
        Fill_J_Entries_From_Payload,
        (layout->sci_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout->sci_numbers,
        layout->d_nbnxm_sci, layout->d_nbnxm_cjpacked,
        layout->d_jentry_offsets, layout->d_j_entries);

    Reserve_Device_U64_Buffer(summary.source_entries, &layout->d_sort_keys,
                              &layout->sort_key_capacity);
    Reserve_Device_Int_Buffer(summary.source_entries + 1,
                              &layout->d_jentry_counts,
                              &layout->jentry_count_capacity);
    Reserve_Device_Int_Buffer(summary.source_entries + 1,
                              &layout->d_jentry_offsets,
                              &layout->jentry_offset_capacity);
    Reserve_Device_Int_Buffer(summary.source_entries + 1,
                              &layout->d_jentry_indices,
                              &layout->jentry_index_capacity);
    if (summary.source_entries > 1)
    {
        Launch_Device_Kernel(
            Build_Reduced_J_Entry_Sort_Keys,
            (summary.source_entries + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, summary.source_entries,
            layout->d_j_entries, layout->d_sort_keys);
        Launch_Device_Kernel(
            Build_Linear_Indices,
            (summary.source_entries + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, summary.source_entries,
            layout->d_jentry_indices);
        Stable_Sort_Device_By_Key(layout, summary.source_entries,
                                  layout->d_sort_keys,
                                  layout->d_jentry_indices);
        Launch_Device_Kernel(
            Gather_J_Entries_By_Index,
            (summary.source_entries + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, summary.source_entries,
            layout->d_jentry_indices, layout->d_j_entries,
            layout->d_j_entry_buffer);
        deviceMemcpy(layout->d_j_entries, layout->d_j_entry_buffer,
                     sizeof(LJ_CLUSTERED_J_ENTRY) *
                         static_cast<size_t>(summary.source_entries),
                     deviceMemcpyDeviceToDevice);
    }

    Launch_Device_Kernel(
        Build_Reduced_J_Entry_Flags,
        (summary.source_entries + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, summary.source_entries,
        layout->d_j_entries, layout->d_jentry_counts);
    summary.reduced_entries = Exclusive_Scan_Counts(
        layout, summary.source_entries, layout->d_jentry_counts,
        layout->d_jentry_offsets);
    if (summary.reduced_entries <= 0)
    {
        return summary;
    }

    Launch_Device_Kernel(
        Scatter_J_Entry_Sci_Starts,
        (summary.source_entries + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, summary.source_entries,
        layout->d_jentry_counts, layout->d_jentry_offsets,
        layout->d_jentry_indices);
    deviceMemcpy(layout->d_jentry_indices + summary.reduced_entries,
                 &summary.source_entries, sizeof(int),
                 deviceMemcpyHostToDevice);

    Reserve_Device_Int_Buffer(summary.reduced_entries + 1,
                              &layout->d_cjpacked_group_offsets,
                              &layout->cjpacked_group_offset_capacity);
    Launch_Device_Kernel(
        Build_Reduced_Sci_Flags_From_J_Entry_Starts,
        (summary.reduced_entries + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, summary.reduced_entries,
        layout->d_jentry_indices, layout->d_j_entries,
        layout->d_jentry_counts);
    summary.reduced_sci_numbers = Exclusive_Scan_Counts(
        layout, summary.reduced_entries, layout->d_jentry_counts,
        layout->d_jentry_offsets);
    if (summary.reduced_sci_numbers <= 0)
    {
        return summary;
    }

    Launch_Device_Kernel(
        Scatter_J_Entry_Sci_Starts,
        (summary.reduced_entries + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, summary.reduced_entries,
        layout->d_jentry_counts, layout->d_jentry_offsets,
        layout->d_cjpacked_group_offsets);
    deviceMemcpy(layout->d_cjpacked_group_offsets + summary.reduced_sci_numbers,
                 &summary.reduced_entries, sizeof(int),
                 deviceMemcpyHostToDevice);

    Reserve_Device_Int_Buffer(summary.reduced_sci_numbers,
                              &layout->d_cjpacked_counts,
                              &layout->cjpacked_count_capacity);
    Reserve_Device_Int_Buffer(summary.reduced_sci_numbers + 1,
                              &layout->d_exclusion_offsets,
                              &layout->exclusion_offset_capacity);
    Launch_Device_Kernel(
        Count_Gmxpacked_CjPacked_Per_Sci,
        (summary.reduced_sci_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL,
        summary.reduced_sci_numbers, layout->d_cjpacked_group_offsets,
        layout->d_cjpacked_counts);
    summary.predicted_compact_cj_numbers = Exclusive_Scan_Counts(
        layout, summary.reduced_sci_numbers, layout->d_cjpacked_counts,
        layout->d_exclusion_offsets);

    Clustered_Debug_Device_Sync_If_Tracing(
        "Analyze_Reduced_Gmxpacked_Count_And_Scan_On_Device");
    return summary;
}
#endif

static void Build_Gmxpacked_Payload_On_Device(LJ_CLUSTER_LAYOUT* layout,
                                              int build_step_count)
{
    Reserve_Device_Int_Buffer(layout->sci_numbers + 1, &layout->d_jentry_counts,
                              &layout->jentry_count_capacity);
    Reserve_Device_Int_Buffer(layout->sci_numbers + 1,
                              &layout->d_jentry_offsets,
                              &layout->jentry_offset_capacity);
    Launch_Device_Kernel(
        Count_Cj_Entries_Per_Sci,
        (layout->sci_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout->sci_numbers,
        layout->d_nbnxm_sci, layout->d_nbnxm_cjpacked,
        layout->d_jentry_counts);
    const int compact_jentry_numbers = Exclusive_Scan_Counts(
        layout, layout->sci_numbers, layout->d_jentry_counts,
        layout->d_jentry_offsets);
    if (compact_jentry_numbers <= 0)
    {
        Reset_Gmxpacked_Payload(layout);
        return;
    }

    Reserve_Device_Buffer(compact_jentry_numbers, &layout->d_j_entries,
                          &layout->jentry_capacity);
    Reserve_Device_Buffer(compact_jentry_numbers, &layout->d_j_entry_buffer,
                          &layout->jentry_buffer_capacity);
    Launch_Device_Kernel(
        Fill_J_Entries_From_Payload,
        (layout->sci_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout->sci_numbers,
        layout->d_nbnxm_sci, layout->d_nbnxm_cjpacked,
        layout->d_jentry_offsets, layout->d_j_entries);

    Reserve_Device_U64_Buffer(compact_jentry_numbers, &layout->d_sort_keys,
                              &layout->sort_key_capacity);
    Reserve_Device_Int_Buffer(compact_jentry_numbers + 1,
                              &layout->d_jentry_counts,
                              &layout->jentry_count_capacity);
    Reserve_Device_Int_Buffer(compact_jentry_numbers + 1,
                              &layout->d_jentry_offsets,
                              &layout->jentry_offset_capacity);
    Reserve_Device_Int_Buffer(compact_jentry_numbers + 1,
                              &layout->d_jentry_indices,
                              &layout->jentry_index_capacity);
    if (compact_jentry_numbers > 1)
    {
        Launch_Device_Kernel(
            Build_J_Entry_Block_Sort_Keys,
            (compact_jentry_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, compact_jentry_numbers,
            layout->d_j_entries, true, layout->d_sort_keys);
        Launch_Device_Kernel(
            Build_Linear_Indices,
            (compact_jentry_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, compact_jentry_numbers,
            layout->d_jentry_indices);
        Stable_Sort_Device_By_Key(layout, compact_jentry_numbers,
                                  layout->d_sort_keys,
                                  layout->d_jentry_indices);
        Launch_Device_Kernel(
            Gather_J_Entries_By_Index,
            (compact_jentry_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, compact_jentry_numbers,
            layout->d_jentry_indices, layout->d_j_entries,
            layout->d_j_entry_buffer);
        deviceMemcpy(layout->d_j_entries, layout->d_j_entry_buffer,
                     sizeof(LJ_CLUSTERED_J_ENTRY) *
                         static_cast<size_t>(compact_jentry_numbers),
                     deviceMemcpyDeviceToDevice);
    }

    Launch_Device_Kernel(
        Build_J_Entry_Sci_Flags,
        (compact_jentry_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, compact_jentry_numbers,
        layout->d_j_entries, layout->d_jentry_counts);
    layout->gmxpacked_sci_numbers = Exclusive_Scan_Counts(
        layout, compact_jentry_numbers, layout->d_jentry_counts,
        layout->d_jentry_offsets);
    if (layout->gmxpacked_sci_numbers <= 0)
    {
        Reset_Gmxpacked_Payload(layout);
        return;
    }

    Launch_Device_Kernel(
        Scatter_J_Entry_Sci_Starts,
        (compact_jentry_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, compact_jentry_numbers,
        layout->d_jentry_counts, layout->d_jentry_offsets,
        layout->d_jentry_indices);
    deviceMemcpy(layout->d_jentry_indices + layout->gmxpacked_sci_numbers,
                 &compact_jentry_numbers, sizeof(int),
                 deviceMemcpyHostToDevice);

    Reserve_Device_Int_Buffer(layout->gmxpacked_sci_numbers,
                              &layout->d_cjpacked_counts,
                              &layout->cjpacked_count_capacity);
    Reserve_Device_Int_Buffer(layout->gmxpacked_sci_numbers + 1,
                              &layout->d_cjpacked_group_offsets,
                              &layout->cjpacked_group_offset_capacity);
    Launch_Device_Kernel(
        Count_Gmxpacked_CjPacked_Per_Sci,
        (layout->gmxpacked_sci_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout->gmxpacked_sci_numbers,
        layout->d_jentry_indices, layout->d_cjpacked_counts);
    layout->gmxpacked_cjpacked_numbers = Exclusive_Scan_Counts(
        layout, layout->gmxpacked_sci_numbers, layout->d_cjpacked_counts,
        layout->d_cjpacked_group_offsets);
    if (layout->gmxpacked_cjpacked_numbers <= 0)
    {
        Reset_Gmxpacked_Payload(layout);
        return;
    }

    Reserve_Device_Buffer(layout->gmxpacked_sci_numbers,
                          &layout->d_gmxpacked_sci,
                          &layout->gmxpacked_sci_capacity);
    Reserve_Device_Buffer(layout->gmxpacked_cjpacked_numbers,
                          &layout->d_gmxpacked_cjpacked,
                          &layout->gmxpacked_cjpacked_capacity);
    Launch_Device_Kernel(
        Fill_Gmxpacked_Sci_And_Cj,
        (layout->gmxpacked_sci_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout->gmxpacked_sci_numbers,
        layout->d_jentry_indices, layout->d_j_entries,
        layout->d_cjpacked_group_offsets, layout->d_gmxpacked_sci,
        layout->d_gmxpacked_cjpacked);

    const int split_entry_numbers =
        layout->gmxpacked_cjpacked_numbers * kClusteredWarpSplitCount;
    Reserve_Device_Int_Buffer(split_entry_numbers, &layout->d_exclusion_counts,
                              &layout->exclusion_count_capacity);
    Reserve_Device_Int_Buffer(split_entry_numbers + 1,
                              &layout->d_exclusion_offsets,
                              &layout->exclusion_offset_capacity);
    Launch_Device_Kernel(
        Count_Gmxpacked_Split_Exclusions, layout->gmxpacked_sci_numbers,
        kClusteredGmxpackedExclusionBlockSize, 0, NULL,
        layout->gmxpacked_sci_numbers, layout->cluster_numbers,
        layout->d_gmxpacked_sci, layout->d_gmxpacked_cjpacked,
        layout->d_j_entries, layout->d_jentry_indices,
        layout->d_super_cluster_offsets, layout->d_cluster_valid_masks,
        layout->d_cluster_local_masks, layout->d_exclusion_mask_pool,
        layout->exclusion_pool_numbers, layout->d_exclusion_counts);
    layout->gmxpacked_split_exclusion_numbers = Exclusive_Scan_Counts(
        layout, split_entry_numbers, layout->d_exclusion_counts,
        layout->d_exclusion_offsets);

    layout->gmxpacked_exclusion_numbers =
        1 + layout->gmxpacked_split_exclusion_numbers;
    Reserve_Device_Buffer(layout->gmxpacked_exclusion_numbers,
                          &layout->d_gmxpacked_exclusions,
                          &layout->gmxpacked_exclusion_capacity);
    Launch_Device_Kernel(
        Initialize_Gmxpacked_Exclusion_Rows,
        (layout->gmxpacked_exclusion_numbers + CONTROLLER::device_max_thread -
         1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL,
        layout->gmxpacked_exclusion_numbers, layout->d_gmxpacked_exclusions);
    if (layout->gmxpacked_split_exclusion_numbers > 0)
    {
        Launch_Device_Kernel(
            Fill_Gmxpacked_Split_Exclusions, layout->gmxpacked_sci_numbers,
            kClusteredGmxpackedExclusionBlockSize, 0, NULL,
            layout->gmxpacked_sci_numbers, layout->cluster_numbers,
            layout->d_gmxpacked_sci, layout->d_gmxpacked_cjpacked,
            layout->d_j_entries, layout->d_jentry_indices,
            layout->d_exclusion_offsets, layout->d_super_cluster_offsets,
            layout->d_cluster_valid_masks, layout->d_cluster_local_masks,
            layout->d_exclusion_mask_pool, layout->exclusion_pool_numbers,
            layout->d_gmxpacked_cjpacked, layout->d_gmxpacked_exclusions);
    }

    Clustered_Debug_Device_Sync_If_Tracing("Build_Gmxpacked_Payload_On_Device");
    if (Clustered_Trace_Warp_Records_Enabled())
    {
        fprintf(stderr,
                "[clustered primary gmxpacked payload] step=%d "
                "step_builds=%d total_builds=%lld compact_sci=%d "
                "compact_cj=%d compact_excl=%d split_excl=%d "
                "compat_native_sci=%d compat_native_cj=%d "
                "compat_native_excl_pool=%d source=device-build\n",
                md_info.sys.steps, build_step_count,
                layout->primary_payload_build_count_total,
                layout->gmxpacked_sci_numbers,
                layout->gmxpacked_cjpacked_numbers,
                layout->gmxpacked_exclusion_numbers,
                layout->gmxpacked_split_exclusion_numbers, layout->sci_numbers,
                layout->cjpacked_numbers, layout->exclusion_pool_numbers);
        fflush(stderr);
    }
}
#endif

static void Build_Gmxpacked_Payload(LJ_CLUSTER_LAYOUT* layout)
{
    Reset_Gmxpacked_Payload(layout);
    if (layout == NULL || layout->sci_numbers <= 0 ||
        layout->cjpacked_numbers <= 0 || layout->cluster_numbers <= 0 ||
        layout->super_cluster_numbers <= 0 || layout->d_nbnxm_sci == NULL ||
        layout->d_nbnxm_cjpacked == NULL ||
        layout->d_super_cluster_offsets == NULL ||
        layout->d_cluster_valid_masks == NULL ||
        layout->d_cluster_local_masks == NULL)
    {
        return;
    }

    const int build_step_count = Note_Clustered_Step_Counter(
        md_info.sys.steps, &layout->primary_payload_build_step,
        &layout->primary_payload_build_count_this_step,
        &layout->primary_payload_build_count_total);
    ClusteredRecorderScope primary_payload_scope(
        layout->primary_payload_time_recorder);
    const bool run_reduced_staged_count =
        layout->runtime_gmxpacked_direct_requested &&
        Clustered_Gmxpacked_Reduced_Build_Enabled();

#ifndef USE_CPU
    if (run_reduced_staged_count)
    {
        Trace_Clustered_Reduced_Staged_Count(
            layout, build_step_count,
            Analyze_Reduced_Gmxpacked_Count_And_Scan_On_Device(layout));
    }
    Build_Gmxpacked_Payload_On_Device(layout, build_step_count);
    return;
#endif

    const std::vector<LJ_CLUSTERED_SCI> native_sci =
        Copy_Device_Buffer_To_Host(layout->d_nbnxm_sci,
                                   static_cast<size_t>(layout->sci_numbers));
    const std::vector<LJ_CLUSTERED_CJ_PACKED> native_cjpacked =
        Copy_Device_Buffer_To_Host(
            layout->d_nbnxm_cjpacked,
            static_cast<size_t>(layout->cjpacked_numbers));
    const std::vector<unsigned long long> native_exclusion_masks =
        Copy_Device_Buffer_To_Host(
            layout->d_exclusion_mask_pool,
            static_cast<size_t>(layout->exclusion_pool_numbers));
    const std::vector<int> super_cluster_offsets = Copy_Device_Buffer_To_Host(
        layout->d_super_cluster_offsets,
        static_cast<size_t>(layout->super_cluster_numbers + 1));
    const std::vector<unsigned int> cluster_valid_masks =
        Copy_Device_Buffer_To_Host(
            layout->d_cluster_valid_masks,
            static_cast<size_t>(layout->cluster_numbers));
    const std::vector<unsigned int> cluster_local_masks =
        Copy_Device_Buffer_To_Host(
            layout->d_cluster_local_masks,
            static_cast<size_t>(layout->cluster_numbers));

    if (run_reduced_staged_count)
    {
        Trace_Clustered_Reduced_Staged_Count(
            layout, build_step_count,
            Analyze_Reduced_Gmxpacked_Count_And_Scan_On_Host(native_sci,
                                                             native_cjpacked));
    }

    std::vector<LJ_CLUSTERED_GMXPACKED_SCI> gmxpacked_sci;
    std::vector<LJ_CLUSTERED_GMXPACKED_CJ> gmxpacked_cjpacked;
    std::vector<LJ_CLUSTERED_GMXPACKED_EXCLUSION> gmxpacked_exclusions;
    gmxpacked_sci.reserve(native_sci.size());
    gmxpacked_cjpacked.reserve(native_cjpacked.size());
    gmxpacked_exclusions.reserve(1u +
                                 native_cjpacked.size() *
                                     static_cast<size_t>(
                                         kClusteredWarpSplitCount));
    gmxpacked_exclusions.push_back(Make_Empty_Gmxpacked_No_Exclusion());
    int split_exclusion_numbers = 0;

    for (const LJ_CLUSTERED_SCI& native_sci_entry : native_sci)
    {
        const int compact_begin = static_cast<int>(gmxpacked_cjpacked.size());
        const int cluster_i_start =
            super_cluster_offsets[static_cast<size_t>(native_sci_entry.supercluster_id)];

        for (int packed_idx = native_sci_entry.cjpacked_begin;
             packed_idx < native_sci_entry.cjpacked_end; packed_idx += 1)
        {
            const LJ_CLUSTERED_CJ_PACKED& native_packed =
                native_cjpacked[static_cast<size_t>(packed_idx)];
            LJ_CLUSTERED_GMXPACKED_CJ compact_packed = {};
            for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
            {
                compact_packed.cj[jm] = native_packed.cj[jm];
            }

            for (int split_idx = 0; split_idx < kClusteredWarpSplitCount;
                 split_idx += 1)
            {
                const unsigned int split_imask = native_packed.imei[split_idx].imask;
                compact_packed.split[split_idx].imask = split_imask;
                compact_packed.split[split_idx].exclusion_index = 0;
                if (split_imask == 0u)
                {
                    continue;
                }

                LJ_CLUSTERED_GMXPACKED_EXCLUSION compact_exclusion = {};
                for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
                {
                    const int cluster_j = native_packed.cj[jm];
                    if (cluster_j < 0)
                    {
                        continue;
                    }

                    const unsigned int jm_imask =
                        Clustered_Jm_Imask(native_packed.imei[split_idx], jm);
                    if (jm_imask == 0u)
                    {
                        continue;
                    }

                    const unsigned int valid_mask_j =
                        cluster_valid_masks[static_cast<size_t>(cluster_j)];
                    const unsigned int local_mask_j =
                        cluster_local_masks[static_cast<size_t>(cluster_j)];
                    for (int i_local = 0;
                         i_local < kClusteredSuperClusterClusters; i_local += 1)
                    {
                        const unsigned int imask_bit =
                            1u << static_cast<unsigned int>(i_local);
                        if ((jm_imask & imask_bit) == 0u)
                        {
                            continue;
                        }

                        const int cluster_i = cluster_i_start + i_local;
                        const unsigned int local_mask_i =
                            cluster_i >= 0 && cluster_i < layout->cluster_numbers
                                ? cluster_local_masks[static_cast<size_t>(cluster_i)]
                                : 0u;
                        const int native_exclusion_index =
                            Clustered_Exclusion_Index(
                                native_packed.imei[split_idx], jm, i_local);
                        const unsigned long long native_exclusion_mask =
                            native_exclusion_index >= 0 &&
                                    native_exclusion_index <
                                        static_cast<int>(
                                            native_exclusion_masks.size())
                                ? native_exclusion_masks[static_cast<size_t>(
                                      native_exclusion_index)]
                                : 0ull;
                        const unsigned int packed_bit =
                            1u << static_cast<unsigned int>(
                                      jm * kClusteredSuperClusterClusters +
                                      i_local);

                        for (int split_j_lane = 0;
                             split_j_lane < kClusteredSplitJClusterSize;
                             split_j_lane += 1)
                        {
                            const int j_lane = split_idx *
                                                   kClusteredSplitJClusterSize +
                                               split_j_lane;
                            const bool valid_j =
                                (valid_mask_j &
                                 (1u << static_cast<unsigned int>(j_lane))) != 0u;
                            const bool local_j =
                                (local_mask_j &
                                 (1u << static_cast<unsigned int>(j_lane))) != 0u;
                            for (int i_lane = 0; i_lane < kClusteredClusterSize;
                                 i_lane += 1)
                            {
                                const bool local_i =
                                    (local_mask_i &
                                     (1u << static_cast<unsigned int>(i_lane))) !=
                                    0u;
                                bool allow_pair = valid_j && local_i;
                                if (allow_pair &&
                                    native_sci_entry.shift_id ==
                                        kClusteredCentralShiftId &&
                                    cluster_i == cluster_j && local_j &&
                                    j_lane <= i_lane)
                                {
                                    allow_pair = false;
                                }
                                if (allow_pair &&
                                    (native_exclusion_mask &
                                     (1ull
                                      << static_cast<unsigned int>(
                                             i_lane * kClusteredClusterSize +
                                             j_lane))) != 0ull)
                                {
                                    allow_pair = false;
                                }
                                if (allow_pair)
                                {
                                    compact_exclusion
                                        .pair[static_cast<size_t>(split_j_lane *
                                                                  kClusteredClusterSize +
                                                                  i_lane)] |= packed_bit;
                                }
                            }
                        }
                    }
                }

                bool needs_exclusion = false;
                for (const unsigned int pair_word : compact_exclusion.pair)
                {
                    if (pair_word != split_imask)
                    {
                        needs_exclusion = true;
                        break;
                    }
                }
                if (needs_exclusion)
                {
                    compact_packed.split[split_idx].exclusion_index =
                        static_cast<int>(gmxpacked_exclusions.size());
                    gmxpacked_exclusions.push_back(compact_exclusion);
                    split_exclusion_numbers += 1;
                }
            }

            gmxpacked_cjpacked.push_back(compact_packed);
        }

        gmxpacked_sci.push_back({native_sci_entry.supercluster_id,
                                 native_sci_entry.shift_id, compact_begin,
                                 static_cast<int>(gmxpacked_cjpacked.size())});
    }

    if (gmxpacked_cjpacked.empty())
    {
        gmxpacked_exclusions.clear();
    }

    layout->gmxpacked_sci_numbers = static_cast<int>(gmxpacked_sci.size());
    layout->gmxpacked_cjpacked_numbers =
        static_cast<int>(gmxpacked_cjpacked.size());
    layout->gmxpacked_exclusion_numbers =
        static_cast<int>(gmxpacked_exclusions.size());
    layout->gmxpacked_split_exclusion_numbers = split_exclusion_numbers;

    if (layout->gmxpacked_sci_numbers > 0)
    {
        Reserve_Device_Buffer(layout->gmxpacked_sci_numbers,
                              &layout->d_gmxpacked_sci,
                              &layout->gmxpacked_sci_capacity);
        deviceMemcpy(layout->d_gmxpacked_sci, gmxpacked_sci.data(),
                     sizeof(LJ_CLUSTERED_GMXPACKED_SCI) *
                         gmxpacked_sci.size(),
                     deviceMemcpyHostToDevice);
    }
    if (layout->gmxpacked_cjpacked_numbers > 0)
    {
        Reserve_Device_Buffer(layout->gmxpacked_cjpacked_numbers,
                              &layout->d_gmxpacked_cjpacked,
                              &layout->gmxpacked_cjpacked_capacity);
        deviceMemcpy(layout->d_gmxpacked_cjpacked, gmxpacked_cjpacked.data(),
                     sizeof(LJ_CLUSTERED_GMXPACKED_CJ) *
                         gmxpacked_cjpacked.size(),
                     deviceMemcpyHostToDevice);
    }
    if (layout->gmxpacked_exclusion_numbers > 0)
    {
        Reserve_Device_Buffer(layout->gmxpacked_exclusion_numbers,
                              &layout->d_gmxpacked_exclusions,
                              &layout->gmxpacked_exclusion_capacity);
        deviceMemcpy(layout->d_gmxpacked_exclusions,
                     gmxpacked_exclusions.data(),
                     sizeof(LJ_CLUSTERED_GMXPACKED_EXCLUSION) *
                         gmxpacked_exclusions.size(),
                     deviceMemcpyHostToDevice);
    }

#ifndef USE_CPU
    if (Clustered_Trace_Warp_Records_Enabled())
    {
        fprintf(stderr,
                "[clustered primary gmxpacked payload] step=%d "
                "step_builds=%d total_builds=%lld compact_sci=%d "
                "compact_cj=%d compact_excl=%d split_excl=%d "
                "compat_native_sci=%d compat_native_cj=%d "
                "compat_native_excl_pool=%d source=Build-cache\n",
                md_info.sys.steps, build_step_count,
                layout->primary_payload_build_count_total,
                layout->gmxpacked_sci_numbers,
                layout->gmxpacked_cjpacked_numbers,
                layout->gmxpacked_exclusion_numbers,
                layout->gmxpacked_split_exclusion_numbers,
                layout->sci_numbers, layout->cjpacked_numbers,
                layout->exclusion_pool_numbers);
        fflush(stderr);
    }
#endif
}

static void Reserve_Plain_Gather_Scratch(LJ_CLUSTERED_DIRECT_CACHE* cache)
{
    LJ_CLUSTER_LAYOUT& layout = cache->layout;
    Reserve_Device_Buffer(layout.total_atom_numbers, &cache->d_sorted_atom_ids,
                          &cache->scratch_capacity);
    Reserve_Device_Buffer(layout.total_atom_numbers, &cache->d_sorted_xq,
                          &cache->scratch_capacity);
    Reserve_Device_Buffer(layout.total_atom_numbers, &cache->d_sorted_lj_type,
                          &cache->scratch_capacity);
    Reserve_Device_Buffer(layout.total_atom_numbers, &cache->d_sorted_frc,
                          &cache->scratch_capacity);
    Reserve_Device_Float_Buffer(layout.total_atom_numbers,
                                &cache->d_sorted_frc_x,
                                &cache->scratch_capacity);
    Reserve_Device_Float_Buffer(layout.total_atom_numbers,
                                &cache->d_sorted_frc_y,
                                &cache->scratch_capacity);
    Reserve_Device_Float_Buffer(layout.total_atom_numbers,
                                &cache->d_sorted_frc_z,
                                &cache->scratch_capacity);
}

static void Refresh_Plain_Gather_Dependent_Metadata(
    LJ_CLUSTERED_DIRECT_CACHE* cache, LTMatrix3 cell, LTMatrix3 rcell)
{
    LJ_CLUSTER_LAYOUT& layout = cache->layout;
    if (layout.runtime_gmxpacked_direct_requested)
    {
        Refresh_Gmxpacked_Pair_Shift_Metadata(&layout, rcell);
        return;
    }
    const bool prune_enabled =
        Clustered_Outer_Inner_Prune_Enabled(&layout) &&
        layout.sci_numbers > 0 && layout.cjpacked_numbers > 0 &&
        layout.d_outer_imask != NULL && layout.d_nbnxm_sci != NULL &&
        layout.d_nbnxm_cjpacked != NULL && cache->d_sorted_xq != NULL;
    if (prune_enabled)
    {
        Launch_Device_Kernel(
            Prune_Clustered_Inner_Imask, layout.sci_numbers,
            kClusteredPruneBlockSize, 0, NULL, layout.sci_numbers,
            layout.cached_cutoff * layout.cached_cutoff, cell, rcell,
            layout.d_super_cluster_offsets, layout.d_cluster_offsets,
            layout.d_cluster_valid_masks, layout.d_cluster_local_masks,
            layout.d_cluster_centers, layout.d_nbnxm_sci, layout.d_outer_imask,
            cache->d_sorted_xq, layout.d_nbnxm_cjpacked);
    }
    Refresh_Clustered_Pair_Shift_Metadata(&layout, rcell);
    if (layout.runtime_aux_clustered_metadata_requested && prune_enabled &&
        Clustered_Build_Warp_Records_Enabled())
    {
        Rebuild_Clustered_Forceonly_Warp_Records(&layout);
    }
    if (layout.sci_numbers > 0 && layout.forceonly_warp_record_numbers > 0 &&
        layout.d_forceonly_warp_j_records != NULL &&
        layout.d_forceonly_warp_record_offsets != NULL)
    {
        if (Clustered_Trace_Warp_Records_Enabled())
        {
            int tail_offset = 0;
            deviceMemcpy(&tail_offset,
                         layout.d_forceonly_warp_record_offsets +
                             layout.sci_numbers,
                         sizeof(int), deviceMemcpyDeviceToHost);
            fprintf(stderr,
                    "[clustered diagnostic warp-record gather] step=%d "
                    "coord_gathers=%d total_coord_gathers=%lld sci=%d "
                    "records=%d tail=%d\n",
                    md_info.sys.steps, cache->coordinate_gather_count_this_step,
                    cache->coordinate_gather_count_total, layout.sci_numbers,
                    layout.forceonly_warp_record_numbers, tail_offset);
        }
    }
}

}  // namespace

void LJ_CLUSTER_LAYOUT::Initial(CONTROLLER* controller, const char* module_name,
                                bool ordered_layout_enabled)
{
    constexpr float kDefaultClusteredRebuildSkin = 10.0f;
    float halo_skin = 2.0f;

    enabled = false;
    warn_legacy_ordered_layout = ordered_layout_enabled;
    compression_requested = false;
    cluster_size = 8;
    super_cluster_clusters = 8;
    cornerstone_max_depth = 6;
    cornerstone_leaf_size = 32;
    rebuild_skin = 2.0f;
    rebuild_skin_permit = 0.5f;
    rebuild_dirty = true;
    cache_ready = false;
    cached_cutoff = -1.0f;
    primary_payload_build_step = -1;
    primary_payload_build_count_this_step = 0;
    primary_payload_build_count_total = 0;
    Invalidate_Clustered_Legacy_Neighbor_View(this);
    this->controller = controller;
    working_device = controller->working_device;
    rebuild_refresh_interval = 0;
    cached_build_step = -1;
    payload_build_time_recorder =
        controller->Get_Time_Recorder("clustered payload build");
    primary_payload_time_recorder =
        controller->Get_Time_Recorder("clustered primary payload");

    if (controller->Command_Choice("LJ", "direct_kernel", "clustered"))
    {
        enabled = true;
    }
    else if (controller->Command_Choice("LJ", "direct_kernel", "legacy"))
    {
        enabled = false;
    }
    if (enabled)
    {
        rebuild_skin = kDefaultClusteredRebuildSkin;
        halo_skin = kDefaultClusteredRebuildSkin;
    }

    if (controller->Command_Exist("LJ", "cluster_size"))
    {
        controller->Check_Int("LJ", "cluster_size", module_name);
        cluster_size = atoi(controller->Command("LJ", "cluster_size"));
    }
    if (controller->Command_Exist("LJ", "super_cluster_clusters"))
    {
        controller->Check_Int("LJ", "super_cluster_clusters", module_name);
        super_cluster_clusters =
            atoi(controller->Command("LJ", "super_cluster_clusters"));
    }
    if (controller->Command_Exist("LJ", "cornerstone_max_depth"))
    {
        controller->Check_Int("LJ", "cornerstone_max_depth", module_name);
        cornerstone_max_depth =
            atoi(controller->Command("LJ", "cornerstone_max_depth"));
    }
    if (controller->Command_Exist("LJ", "cornerstone_leaf_size"))
    {
        controller->Check_Int("LJ", "cornerstone_leaf_size", module_name);
        cornerstone_leaf_size =
            atoi(controller->Command("LJ", "cornerstone_leaf_size"));
    }
    if (controller->Command_Exist("LJ", "clustered_compression"))
    {
        compression_requested =
            controller->Get_Bool("LJ", "clustered_compression", module_name);
    }
    if (controller->Command_Exist("LJ", "cpu_simd"))
    {
#ifndef USE_CPU
        if (enabled)
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "LJ_CLUSTER_LAYOUT::Initial",
                "Reason:\n\t clustered direct LJ on GPU requires a "
                "device-native payload build; remove the deprecated "
                "LJ.cpu_simd setting.\n");
        }
#endif
    }
    if (controller->Command_Exist("skin"))
    {
        controller->Check_Float("skin", module_name);
        halo_skin = atof(controller->Command("skin"));
        if (!enabled)
        {
            rebuild_skin = halo_skin;
        }
    }
    if (controller->Command_Exist("LJ", "clustered_rebuild_skin"))
    {
        controller->Check_Float("LJ", "clustered_rebuild_skin", module_name);
        rebuild_skin = atof(controller->Command("LJ", "clustered_rebuild_skin"));
    }
    if (controller->Command_Exist("neighbor_list", "skin_permit"))
    {
        controller->Check_Float("neighbor_list", "skin_permit", module_name);
        rebuild_skin_permit =
            atof(controller->Command("neighbor_list", "skin_permit"));
    }
    if (controller->Command_Exist("neighbor_list", "refresh_interval"))
    {
        controller->Check_Int("neighbor_list", "refresh_interval", module_name);
        rebuild_refresh_interval =
            atoi(controller->Command("neighbor_list", "refresh_interval"));
    }

    cluster_size = std::max(1, cluster_size);
    super_cluster_clusters = std::max(1, super_cluster_clusters);
    cornerstone_max_depth = std::max(1, std::min(21, cornerstone_max_depth));
    cornerstone_leaf_size = std::max(1, cornerstone_leaf_size);
    rebuild_skin = fmaxf(0.0f, rebuild_skin);
    rebuild_skin_permit = fmaxf(0.0f, rebuild_skin_permit);

    if (enabled && cluster_size != 8)
    {
        controller->printf(
            "    Clustered LJ currently supports only cluster_size=8; "
            "override %d is ignored.\n",
            cluster_size);
        cluster_size = 8;
    }
    if (enabled && super_cluster_clusters != 8)
    {
        controller->printf(
            "    Clustered LJ currently supports only "
            "super_cluster_clusters=8; override %d is ignored.\n",
            super_cluster_clusters);
        super_cluster_clusters = 8;
    }
    if (enabled && compression_requested)
    {
        controller->printf(
            "    Clustered LJ compression is not implemented yet; "
            "continuing with uncompressed pair storage.\n");
    }
    if (enabled && ordered_layout_enabled)
    {
        controller->printf(
            "    Clustered direct LJ ignores ordered_layout and uses backend "
            "cluster builds instead.\n");
    }
    if (enabled)
    {
        controller->printf(
            "    direct_kernel: clustered (cluster_size=%d "
            "super_cluster_clusters=%d depth=%d leaf_size=%d reuse_skin=%.2f "
            "skin_permit=%.2f refresh_interval=%d)\n",
            cluster_size, super_cluster_clusters, cornerstone_max_depth,
            cornerstone_leaf_size, rebuild_skin, rebuild_skin_permit,
            rebuild_refresh_interval);
    }
}

void LJ_CLUSTER_LAYOUT::Refresh_Metadata(int input_local_atom_numbers,
                                         int input_direct_local_atom_numbers,
                                         int input_ghost_numbers,
                                         const int* d_input_atom_local,
                                         const int* d_input_excluded_list_start,
                                         const int* d_input_excluded_list,
                                         const int* d_input_excluded_numbers)
{
    local_atom_numbers = input_local_atom_numbers;
    direct_local_atom_numbers =
        IntMin(local_atom_numbers, IntMax(0, input_direct_local_atom_numbers));
    ghost_numbers = input_ghost_numbers;
    total_atom_numbers = local_atom_numbers + ghost_numbers;
    if (!enabled)
    {
        return;
    }
    d_atom_local = d_input_atom_local;
    d_excluded_list_start = d_input_excluded_list_start;
    d_excluded_list = d_input_excluded_list;
    d_excluded_numbers = d_input_excluded_numbers;
    rebuild_dirty = true;
    Invalidate_Clustered_Legacy_Neighbor_View(this);
    Initialize_Cornerstone_State(this);
}

void LJ_CLUSTER_LAYOUT::Build(const VECTOR* crd, LTMatrix3 cell,
                              LTMatrix3 rcell, float cutoff,
                              bool need_virial,
                              bool prefer_full_warp_record,
                              bool need_gmxpacked_payload,
                              bool need_aux_clustered_metadata,
                              bool runtime_gmxpacked_direct)
{
    if (!enabled)
    {
        return;
    }
#ifndef USE_CPU
    if (controller != NULL)
    {
        working_device = controller->working_device;
    }
    Bind_Clustered_Working_Device(&working_device);
#endif
    total_atom_numbers = local_atom_numbers + ghost_numbers;
    runtime_gmxpacked_direct_requested = runtime_gmxpacked_direct;
    runtime_aux_clustered_metadata_requested = need_aux_clustered_metadata;
    if (total_atom_numbers <= 0)
    {
        cluster_numbers = 0;
        super_cluster_numbers = 0;
        local_cluster_numbers = 0;
        candidate_sci_numbers = 0;
        sci_numbers = 0;
        cjpacked_numbers = 0;
        Reset_Gmxpacked_Payload(this);
        forceonly_warp_record_numbers = 0;
        candidate_leaf_numbers = 0;
        exclusion_pool_numbers = 0;
        grouped_sci_ready = false;
        cache_ready = false;
        cached_cutoff = -1.0f;
        runtime_aux_clustered_metadata_requested = false;
        Invalidate_Clustered_Legacy_Neighbor_View(this);
        return;
    }

    Initialize_Cornerstone_State(this);
    if (!Clustered_Build_Is_Needed(this, crd, cell, rcell, cutoff))
    {
        if (Clustered_Trace_Warp_Records_Enabled())
        {
            const int builds_this_step =
                primary_payload_build_step == md_info.sys.steps
                    ? primary_payload_build_count_this_step
                    : 0;
            fprintf(stderr,
                    "[clustered primary payload cache hit] step=%d "
                    "cached_step=%d step_builds=%d total_builds=%lld\n",
                    md_info.sys.steps, cached_build_step, builds_this_step,
                    primary_payload_build_count_total);
            fflush(stderr);
        }
        return;
    }
    if (Clustered_Trace_Warp_Records_Enabled())
    {
        Reserve_Device_Int_Buffer(1, &d_need_rebuild, &rebuild_flag_capacity);
        int h_bad_coordinates = 0;
        deviceMemcpy(d_need_rebuild, &h_bad_coordinates, sizeof(int),
                     deviceMemcpyHostToDevice);
        Launch_Device_Kernel(Check_Clustered_Finite_Coordinates,
                             (total_atom_numbers +
                              CONTROLLER::device_max_thread - 1) /
                                 CONTROLLER::device_max_thread,
                             CONTROLLER::device_max_thread, 0, NULL,
                             total_atom_numbers, crd, d_need_rebuild);
        deviceMemcpy(&h_bad_coordinates, d_need_rebuild, sizeof(int),
                     deviceMemcpyDeviceToHost);
        fprintf(stderr,
                "[clustered build begin] step=%d cached_step=%d refresh=%d "
                "atoms=%d bad_crd=%d\n",
                md_info.sys.steps, cached_build_step, rebuild_refresh_interval,
                total_atom_numbers, h_bad_coordinates);
    }
    ClusteredRecorderScope payload_build_scope(payload_build_time_recorder);

    Reset_Build_Buffers(this);
    Reset_Gmxpacked_Payload(this);
    Invalidate_Clustered_Legacy_Neighbor_View(this);
    grouped_sci_ready = false;
    local_cluster_numbers = 0;
    candidate_sci_numbers = 0;
    sci_numbers = 0;
    cjpacked_numbers = 0;
    forceonly_warp_record_numbers = 0;
    candidate_leaf_numbers = 0;
    exclusion_pool_numbers = 0;
    const bool build_warp_records = need_aux_clustered_metadata &&
                                    Clustered_Build_Warp_Records_Enabled();
    const bool prefer_full_record_builder =
        need_virial && prefer_full_warp_record && build_warp_records;
    const float build_cutoff =
        Clustered_Outer_Inner_Prune_Enabled(this) ? cutoff + rebuild_skin
                                                  : cutoff;
    const bool use_morton_sfc = Clustered_Use_Morton_Sfc();
    const bool has_atom_to_molecule = Prepare_Atom_To_Molecule_Metadata(this);
    const int* atom_to_molecule =
        has_atom_to_molecule ? d_local_atom_to_molecule : NULL;

    Reserve_Device_U64_Buffer(total_atom_numbers, &d_sort_keys,
                              &sort_key_capacity);
    Reserve_Device_Int_Buffer(total_atom_numbers, &d_sort_permutation,
                              &permutation_capacity);

    Launch_Device_Kernel(Build_Cornerstone_Sort_Keys,
                         (total_atom_numbers + CONTROLLER::device_max_thread -
                         1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         total_atom_numbers, crd, rcell, use_morton_sfc,
                         d_sort_keys, d_sort_permutation);

#ifndef USE_CPU
    Sort_Cornerstone_Keys_On_Device(this, total_atom_numbers, d_sort_keys,
                                    d_sort_permutation);
#else
    std::vector<uint64_t> h_keys(static_cast<size_t>(total_atom_numbers));
    std::vector<int> h_perm(static_cast<size_t>(total_atom_numbers));
    deviceMemcpy(h_keys.data(), d_sort_keys, sizeof(uint64_t) * total_atom_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_perm.data(), d_sort_permutation,
                 sizeof(int) * total_atom_numbers, deviceMemcpyDeviceToHost);
    std::vector<int> order(static_cast<size_t>(total_atom_numbers));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int lhs, int rhs)
                     {
                         if (h_keys[static_cast<size_t>(lhs)] !=
                             h_keys[static_cast<size_t>(rhs)])
                         {
                             return h_keys[static_cast<size_t>(lhs)] <
                                    h_keys[static_cast<size_t>(rhs)];
                         }
                         return h_perm[static_cast<size_t>(lhs)] <
                                h_perm[static_cast<size_t>(rhs)];
                     });
    std::vector<uint64_t> h_keys_sorted(static_cast<size_t>(total_atom_numbers));
    std::vector<int> h_perm_sorted(static_cast<size_t>(total_atom_numbers));
    for (int i = 0; i < total_atom_numbers; i += 1)
    {
        h_keys_sorted[static_cast<size_t>(i)] =
            h_keys[static_cast<size_t>(order[static_cast<size_t>(i)])];
        h_perm_sorted[static_cast<size_t>(i)] =
            h_perm[static_cast<size_t>(order[static_cast<size_t>(i)])];
    }
    deviceMemcpy(d_sort_keys, h_keys_sorted.data(),
                 sizeof(uint64_t) * total_atom_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_sort_permutation, h_perm_sorted.data(),
                 sizeof(int) * total_atom_numbers, deviceMemcpyHostToDevice);
#endif

    if (Clustered_Trace_Warp_Records_Enabled())
    {
        uint64_t first_key = 0;
        uint64_t mid_key = 0;
        uint64_t last_key = 0;
        const int mid_index = total_atom_numbers / 2;
        deviceMemcpy(&first_key, d_sort_keys, sizeof(uint64_t),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&mid_key, d_sort_keys + mid_index, sizeof(uint64_t),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&last_key, d_sort_keys + total_atom_numbers - 1,
                     sizeof(uint64_t), deviceMemcpyDeviceToHost);
        fprintf(stderr,
                "[clustered key span] step=%d first=%llu mid=%llu last=%llu\n",
                md_info.sys.steps,
                static_cast<unsigned long long>(first_key),
                static_cast<unsigned long long>(mid_key),
                static_cast<unsigned long long>(last_key));
    }

    Build_Cornerstone_Tree(this);
    const int leaf_numbers = cornerstone_state->octree.numLeafNodes;
    if (Clustered_Trace_Warp_Records_Enabled())
    {
        fprintf(stderr,
                "[clustered tree state] step=%d leaf_numbers=%d leaves_size=%zu leaf_counts_size=%zu\n",
                md_info.sys.steps, leaf_numbers,
                cornerstone_state->leaves.size(),
                cornerstone_state->leaf_counts.size());
    }
    if (leaf_numbers <= 0)
    {
        cluster_numbers = 0;
        super_cluster_numbers = 0;
        if (Clustered_Trace_Warp_Records_Enabled())
        {
            fprintf(stderr,
                    "[clustered build early-return] step=%d stage=no-leaves\n",
                    md_info.sys.steps);
        }
        Commit_Clustered_Build_Cache(this, crd, cutoff);
        return;
    }

    Reserve_Device_Int_Buffer(leaf_numbers + 1, &d_leaf_atom_offsets,
                              &leaf_capacity);
    Launch_Device_Kernel(Copy_UInt_To_Int,
                         (leaf_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         leaf_numbers, rawPtr(cornerstone_state->leaf_counts),
                         d_leaf_atom_offsets);
    Exclusive_Scan_Counts(this, leaf_numbers, d_leaf_atom_offsets,
                          d_leaf_atom_offsets);

    int leaf_atom_total = 0;
    deviceMemcpy(&leaf_atom_total, d_leaf_atom_offsets + leaf_numbers,
                 sizeof(int), deviceMemcpyDeviceToHost);
    cluster_numbers =
        IntMax(0, (leaf_atom_total + cluster_size - 1) / cluster_size);
    if (cluster_numbers <= 0)
    {
        super_cluster_numbers = 0;
        if (Clustered_Trace_Warp_Records_Enabled())
        {
            fprintf(stderr,
                    "[clustered build early-return] step=%d stage=no-clusters leaf_atom_total=%d\n",
                    md_info.sys.steps, leaf_atom_total);
        }
        Commit_Clustered_Build_Cache(this, crd, cutoff);
        return;
    }

#ifndef USE_CPU
    if (Clustered_Leaf_Geometry_Repack_Enabled() && leaf_atom_total > 1)
    {
        Launch_Device_Kernel(
            Build_Leaf_Geometry_Repack_Sort_Keys,
            (leaf_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, leaf_numbers,
            d_leaf_atom_offsets, d_sort_permutation, crd, rcell, d_sort_keys);
        Stable_Sort_Device_By_Key(this, leaf_atom_total, d_sort_keys,
                                  d_sort_permutation);
    }
    else if (Clustered_Oxygen_Key_Packing_Enabled() &&
        atom_to_molecule != NULL && leaf_atom_total > 1)
    {
        Launch_Device_Kernel(
            Build_Leaf_Oxygen_Key_Packing_Sort_Keys,
            (leaf_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, leaf_numbers,
            d_leaf_atom_offsets, d_sort_permutation, atom_to_molecule,
            d_atom_local, d_sort_keys);
        Stable_Sort_Device_By_Key(this, leaf_atom_total, d_sort_keys,
                                  d_sort_permutation);
    }
    else if (Clustered_Sort_Leaf_Atoms_By_Molecule_Enabled() &&
        atom_to_molecule != NULL && leaf_atom_total > 1)
    {
        Launch_Device_Kernel(
            Build_Leaf_Molecule_Sort_Keys,
            (leaf_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, leaf_numbers,
            d_leaf_atom_offsets, d_sort_permutation, atom_to_molecule,
            d_atom_local, d_sort_keys);
        Stable_Sort_Device_By_Key(this, leaf_atom_total, d_sort_keys,
                                  d_sort_permutation);
    }
#endif

    Reserve_Device_Int_Buffer(leaf_numbers, &d_leaf_cluster_starts,
                              &leaf_cluster_start_capacity);
    Reserve_Device_Int_Buffer(leaf_numbers, &d_leaf_cluster_ends,
                              &leaf_cluster_end_capacity);
    Launch_Device_Kernel(Build_Leaf_Cluster_Ranges,
                         (leaf_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         leaf_numbers, cluster_size, d_leaf_atom_offsets,
                         d_leaf_cluster_starts, d_leaf_cluster_ends);

    Reserve_Device_Int_Buffer(cluster_numbers + 1, &d_cluster_offsets,
                              &cluster_capacity);
    Reserve_Device_UInt_Buffer(cluster_numbers, &d_cluster_valid_masks,
                               &cluster_valid_mask_capacity);
    Reserve_Device_UInt_Buffer(cluster_numbers, &d_cluster_local_masks,
                               &cluster_local_mask_capacity);
    Reserve_Device_Vector_Buffer(cluster_numbers, &d_cluster_centers,
                                 &cluster_center_capacity);
    Reserve_Device_Vector_Buffer(cluster_numbers, &d_cluster_extents,
                                 &cluster_extent_capacity);
    Reserve_Device_Float_Buffer(cluster_numbers, &d_cluster_radii,
                                &cluster_radius_capacity);

    Launch_Device_Kernel(Build_Global_Cluster_Metadata,
                         (cluster_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         cluster_numbers, leaf_atom_total,
                         direct_local_atom_numbers, cluster_size,
                         d_sort_permutation, crd, cell, rcell,
                         d_cluster_offsets, d_cluster_valid_masks,
                         d_cluster_local_masks, d_cluster_centers,
                         d_cluster_extents,
                         d_cluster_radii);

    Reserve_Device_Int_Buffer(leaf_numbers, &d_leaf_all_local,
                              &leaf_all_local_capacity);
    Launch_Device_Kernel(Build_Leaf_All_Local_Flags,
                         (leaf_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         leaf_numbers, d_leaf_cluster_starts,
                         d_leaf_cluster_ends, d_cluster_local_masks,
                         d_leaf_all_local);

    const uint64_t* cluster_molecule_signatures = NULL;
    const int* cluster_molecule_ids = NULL;
    if (atom_to_molecule != NULL)
    {
        Reserve_Device_U64_Buffer(cluster_numbers,
                                  &d_cluster_molecule_signatures,
                                  &cluster_molecule_signature_capacity);
        Reserve_Device_Int_Buffer(cluster_numbers * cluster_size,
                                  &d_cluster_molecule_ids,
                                  &cluster_molecule_id_capacity);
        Launch_Device_Kernel(
            Build_Cluster_Molecule_Metadata,
            (cluster_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, cluster_numbers,
            cluster_size, d_sort_permutation, d_cluster_offsets,
            d_cluster_valid_masks, atom_to_molecule,
            d_cluster_molecule_signatures, d_cluster_molecule_ids);
        cluster_molecule_signatures = d_cluster_molecule_signatures;
        cluster_molecule_ids = d_cluster_molecule_ids;
    }

    const bool spatial_supercluster_grouping =
        Clustered_Use_Spatial_Supercluster_Grouping();
    if (spatial_supercluster_grouping)
    {
        const float spatial_supercluster_link_cutoff =
            Clustered_Spatial_Supercluster_Link_Cutoff();
        Reserve_Device_Int_Buffer(cluster_numbers, &d_sci_candidate_leaf_counts,
                                  &sci_candidate_leaf_count_capacity);
        Reserve_Device_Int_Buffer(cluster_numbers + 1, &d_candidate_sci_offsets,
                                  &candidate_offset_capacity);
        Launch_Device_Kernel(
            Build_Spatial_Group_Start_Flags,
            (cluster_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, cluster_numbers,
            super_cluster_clusters, spatial_supercluster_link_cutoff, cell,
            rcell, d_cluster_valid_masks, d_cluster_centers, d_cluster_extents,
            d_sci_candidate_leaf_counts);
        super_cluster_numbers =
            Exclusive_Scan_Counts(this, cluster_numbers,
                                  d_sci_candidate_leaf_counts,
                                  d_candidate_sci_offsets);
        Reserve_Device_Int_Buffer(super_cluster_numbers + 1,
                                  &d_super_cluster_offsets,
                                  &super_cluster_capacity);
        Launch_Device_Kernel(
            Build_Group_Offsets_From_Start_Flags,
            (cluster_numbers + 1 + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, cluster_numbers,
            super_cluster_numbers, d_sci_candidate_leaf_counts,
            d_candidate_sci_offsets, d_super_cluster_offsets);
    }
    else
    {
        super_cluster_numbers =
            (cluster_numbers + super_cluster_clusters - 1) /
            super_cluster_clusters;
        Reserve_Device_Int_Buffer(super_cluster_numbers + 1,
                                  &d_super_cluster_offsets,
                                  &super_cluster_capacity);
        Launch_Device_Kernel(Build_Fixed_Group_Offsets,
                             (super_cluster_numbers + 1 +
                              CONTROLLER::device_max_thread - 1) /
                                 CONTROLLER::device_max_thread,
                             CONTROLLER::device_max_thread, 0, NULL,
                             super_cluster_numbers + 1, super_cluster_clusters,
                             cluster_numbers, d_super_cluster_offsets);
    }

    Reserve_Device_Int_Buffer(super_cluster_numbers, &d_super_cluster_has_local,
                              &super_cluster_has_local_capacity);
    Reserve_Device_Int_Buffer(cluster_numbers, &d_cluster_to_supercluster,
                              &cluster_to_supercluster_capacity);
    Launch_Device_Kernel(Build_Cluster_To_Supercluster,
                         (super_cluster_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         super_cluster_numbers, d_super_cluster_offsets,
                         d_cluster_to_supercluster);
    Reserve_Device_Vector_Buffer(super_cluster_numbers, &d_super_cluster_centers,
                                 &super_cluster_center_capacity);
    Reserve_Device_Vector_Buffer(super_cluster_numbers, &d_super_cluster_sizes,
                                 &super_cluster_size_capacity);
    Launch_Device_Kernel(Build_Supercluster_Metadata,
                         (super_cluster_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         super_cluster_numbers, super_cluster_clusters,
                         build_cutoff, rcell, d_super_cluster_offsets,
                         d_cluster_valid_masks, d_cluster_local_masks,
                         d_cluster_centers, d_cluster_extents,
                         d_super_cluster_has_local,
                         d_super_cluster_centers, d_super_cluster_sizes);

    Reserve_Device_Int_Buffer(super_cluster_numbers, &d_sci_candidate_leaf_counts,
                              &sci_candidate_leaf_count_capacity);
    Reserve_Device_Int_Buffer(super_cluster_numbers + 1,
                              &d_sci_candidate_leaf_offsets,
                              &sci_candidate_leaf_offset_capacity);
    Launch_Device_Kernel(Build_Local_Supercluster_Flags,
                         (super_cluster_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         super_cluster_numbers, d_super_cluster_has_local,
                         d_sci_candidate_leaf_counts);
    sci_numbers = Exclusive_Scan_Counts(this, super_cluster_numbers,
                                        d_sci_candidate_leaf_counts,
                                        d_sci_candidate_leaf_offsets);
    if (sci_numbers <= 0)
    {
        cjpacked_numbers = 0;
        candidate_leaf_numbers = 0;
        exclusion_pool_numbers = 0;
        if (Clustered_Trace_Warp_Records_Enabled())
        {
            fprintf(stderr,
                    "[clustered build early-return] step=%d stage=no-super-sci super_cluster_numbers=%d\n",
                    md_info.sys.steps, super_cluster_numbers);
        }
        Commit_Clustered_Build_Cache(this, crd, cutoff);
        return;
    }
    const int super_sci_numbers = sci_numbers;
    const bool dense_shift_partitioned_candidates =
        Clustered_Use_Shift_Partitioned_Builder();
    const bool sparse_shift_candidates =
        !dense_shift_partitioned_candidates &&
        Clustered_Use_Sparse_Shift_Candidate_Builder();
    const bool fixed_shift_candidates =
        dense_shift_partitioned_candidates || sparse_shift_candidates;
    const bool central_candidate_halfshell_culling =
        fixed_shift_candidates &&
        Clustered_Use_Central_Candidate_Halfshell_Culling();
    const bool fixed_shift_leaf_screening =
        fixed_shift_candidates && Clustered_Use_Fixed_Shift_Leaf_Screening();
    candidate_leaf_cluster_stride =
        fixed_shift_leaf_screening
            ? (cornerstone_leaf_size + 2 * cluster_size - 2) / cluster_size
            : 0;
    int candidate_sci_numbers = dense_shift_partitioned_candidates
                                    ? super_sci_numbers *
                                          kClusteredShiftCount
                                    : super_sci_numbers;
    const int* candidate_sci_supercluster_ids = NULL;
    const int* candidate_shift_ids = NULL;

    Reserve_Device_Int_Buffer(sci_numbers, &d_sci_supercluster_ids,
                              &sci_capacity);
    Launch_Device_Kernel(Fill_Local_Supercluster_Ids,
                         (super_cluster_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         super_cluster_numbers, d_super_cluster_has_local,
                         d_sci_candidate_leaf_offsets, d_sci_supercluster_ids);
    candidate_sci_supercluster_ids = d_sci_supercluster_ids;

    if (sparse_shift_candidates)
    {
        Launch_Device_Kernel(
            Count_Supercluster_Active_Shifts,
            (super_sci_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, super_sci_numbers,
            d_sci_supercluster_ids, d_super_cluster_centers,
            d_super_cluster_sizes, d_sci_candidate_leaf_counts);
        candidate_sci_numbers = Exclusive_Scan_Counts(
            this, super_sci_numbers, d_sci_candidate_leaf_counts,
            d_sci_candidate_leaf_offsets);
        if (candidate_sci_numbers <= 0)
        {
            cjpacked_numbers = 0;
            candidate_leaf_numbers = 0;
            exclusion_pool_numbers = 0;
            if (Clustered_Trace_Warp_Records_Enabled())
            {
                fprintf(stderr,
                        "[clustered build early-return] step=%d stage=no-candidate-sci super_sci=%d\n",
                        md_info.sys.steps, super_sci_numbers);
            }
            Commit_Clustered_Build_Cache(this, crd, cutoff);
            return;
        }
        Reserve_Device_Int_Buffer(candidate_sci_numbers, &d_candidate_sci_offsets,
                                  &candidate_offset_capacity);
        Reserve_Device_Int_Buffer(candidate_sci_numbers, &d_candidate_shift_ids,
                                  &candidate_shift_capacity);
        Launch_Device_Kernel(
            Fill_Supercluster_Active_Shifts,
            (super_sci_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, super_sci_numbers,
            d_sci_supercluster_ids, d_super_cluster_centers,
            d_super_cluster_sizes, d_sci_candidate_leaf_offsets,
            d_candidate_sci_offsets, d_candidate_shift_ids);
        candidate_sci_supercluster_ids = d_candidate_sci_offsets;
        candidate_shift_ids = d_candidate_shift_ids;
    }
#ifndef USE_CPU
    {
        const cudaError_t prepare_candidate_sync_err = cudaDeviceSynchronize();
        if (prepare_candidate_sync_err != cudaSuccess)
        {
            fprintf(stderr,
                    "[clustered build sync] prepare-candidate-worklist: %s\n",
                    cudaGetErrorString(prepare_candidate_sync_err));
        }
    }
    if (Clustered_Trace_Warp_Records_Enabled())
    {
        std::vector<int> h_candidate_super_ids((size_t)super_sci_numbers, -1);
        deviceMemcpy(h_candidate_super_ids.data(), d_sci_supercluster_ids,
                     sizeof(int) * super_sci_numbers,
                     deviceMemcpyDeviceToHost);
        int min_super_id = super_cluster_numbers;
        int max_super_id = -1;
        int bad_super_id_count = 0;
        for (int sci = 0; sci < super_sci_numbers; sci += 1)
        {
            const int super_id = h_candidate_super_ids[(size_t)sci];
            min_super_id = IntMin(min_super_id, super_id);
            max_super_id = IntMax(max_super_id, super_id);
            if (super_id < 0 || super_id >= super_cluster_numbers)
            {
                bad_super_id_count += 1;
            }
        }
        fprintf(stderr,
                "[clustered local sci ids] super_sci=%d min=%d max=%d bad=%d\n",
                super_sci_numbers, min_super_id, max_super_id,
                bad_super_id_count);
    }
#endif

    Reserve_Device_Int_Buffer(candidate_sci_numbers, &d_sci_candidate_leaf_counts,
                              &sci_candidate_leaf_count_capacity);
    Reserve_Device_Int_Buffer(candidate_sci_numbers + 1,
                              &d_sci_candidate_leaf_offsets,
                              &sci_candidate_leaf_offset_capacity);
    if (fixed_shift_candidates)
    {
        Launch_Device_Kernel(
            Count_Supercluster_Candidate_Leaves_Fixed_Shift,
            (candidate_sci_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, candidate_sci_numbers,
            candidate_sci_supercluster_ids, d_super_cluster_centers,
            d_super_cluster_sizes, d_super_cluster_offsets,
            d_leaf_cluster_starts, d_leaf_cluster_ends, d_leaf_all_local,
            cell,
            build_cutoff, d_cluster_centers, d_cluster_extents,
            d_cluster_valid_masks, d_cluster_local_masks,
            rawPtr(cornerstone_state->octree.prefixes),
            rawPtr(cornerstone_state->octree.childOffsets),
            rawPtr(cornerstone_state->octree.parents),
            rawPtr(cornerstone_state->octree.internalToLeaf),
            candidate_shift_ids, central_candidate_halfshell_culling,
            fixed_shift_leaf_screening, use_morton_sfc,
            d_sci_candidate_leaf_counts);
    }
    else
    {
        Launch_Device_Kernel(
            Count_Supercluster_Candidate_Leaves,
            (candidate_sci_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, candidate_sci_numbers,
            candidate_sci_supercluster_ids, d_super_cluster_centers,
            d_super_cluster_sizes,
            rawPtr(cornerstone_state->octree.prefixes),
            rawPtr(cornerstone_state->octree.childOffsets),
            rawPtr(cornerstone_state->octree.parents),
            rawPtr(cornerstone_state->octree.internalToLeaf),
            use_morton_sfc, d_sci_candidate_leaf_counts);
    }
#ifndef USE_CPU
    Clustered_Debug_Device_Sync_If_Tracing(
        "Count_Supercluster_Candidate_Leaves");
#endif
    candidate_leaf_numbers = Exclusive_Scan_Counts(
        this, candidate_sci_numbers, d_sci_candidate_leaf_counts,
        d_sci_candidate_leaf_offsets);
    if (candidate_leaf_numbers <= 0)
    {
        cjpacked_numbers = 0;
        exclusion_pool_numbers = 0;
        if (Clustered_Trace_Warp_Records_Enabled())
        {
            fprintf(stderr,
                    "[clustered build early-return] step=%d stage=no-candidate-leaves candidate_sci=%d\n",
                    md_info.sys.steps, candidate_sci_numbers);
        }
        Commit_Clustered_Build_Cache(this, crd, cutoff);
        return;
    }

    Reserve_Device_Int_Buffer(candidate_leaf_numbers, &d_sci_candidate_leaf_ids,
                              &candidate_leaf_capacity);
    if (fixed_shift_candidates)
    {
        Launch_Device_Kernel(
            Fill_Supercluster_Candidate_Leaves_Fixed_Shift,
            (candidate_sci_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, candidate_sci_numbers,
            candidate_sci_supercluster_ids, d_super_cluster_centers,
            d_super_cluster_sizes, d_super_cluster_offsets,
            d_leaf_cluster_starts, d_leaf_cluster_ends, d_leaf_all_local,
            cell,
            build_cutoff, d_cluster_centers, d_cluster_extents,
            d_cluster_valid_masks, d_cluster_local_masks,
            rawPtr(cornerstone_state->octree.prefixes),
            rawPtr(cornerstone_state->octree.childOffsets),
            rawPtr(cornerstone_state->octree.parents),
            rawPtr(cornerstone_state->octree.internalToLeaf),
            candidate_shift_ids, d_sci_candidate_leaf_offsets,
            central_candidate_halfshell_culling, fixed_shift_leaf_screening,
            use_morton_sfc, d_sci_candidate_leaf_ids);
    }
    else
    {
        Launch_Device_Kernel(
            Fill_Supercluster_Candidate_Leaves,
            (candidate_sci_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, candidate_sci_numbers,
            candidate_sci_supercluster_ids, d_super_cluster_centers,
            d_super_cluster_sizes,
            rawPtr(cornerstone_state->octree.prefixes),
            rawPtr(cornerstone_state->octree.childOffsets),
            rawPtr(cornerstone_state->octree.parents),
            rawPtr(cornerstone_state->octree.internalToLeaf),
            d_sci_candidate_leaf_offsets, use_morton_sfc,
            d_sci_candidate_leaf_ids);
    }
#ifndef USE_CPU
    Clustered_Debug_Device_Sync_If_Tracing(
        "Fill_Supercluster_Candidate_Leaves");
#endif
    if (fixed_shift_leaf_screening)
    {
        const int candidate_leaf_mask_blocks =
            (candidate_sci_numbers +
             (kClusteredBuilderBlockSize / kClusteredBuilderWarpSize) - 1) /
            (kClusteredBuilderBlockSize / kClusteredBuilderWarpSize);
        Reserve_Device_UInt_Buffer(candidate_leaf_numbers *
                                       candidate_leaf_cluster_stride,
                                   &d_candidate_leaf_reach_masks,
                                   &candidate_leaf_mask_capacity);
        Launch_Device_Kernel(
            Build_Fixed_Shift_Candidate_Leaf_Masks,
            candidate_leaf_mask_blocks, kClusteredBuilderBlockSize, 0, NULL,
            candidate_sci_numbers, super_cluster_clusters, build_cutoff, cell,
            candidate_sci_supercluster_ids, d_super_cluster_offsets,
            d_cluster_to_supercluster, d_leaf_cluster_starts, d_leaf_cluster_ends,
            candidate_shift_ids,
            d_sci_candidate_leaf_offsets, d_sci_candidate_leaf_ids,
            d_cluster_valid_masks, d_cluster_local_masks, d_cluster_centers,
            d_cluster_extents, candidate_leaf_cluster_stride,
            d_candidate_leaf_reach_masks);
#ifndef USE_CPU
        Clustered_Debug_Device_Sync_If_Tracing(
            "Build_Fixed_Shift_Candidate_Leaf_Masks");
#endif
    }
    else
    {
        candidate_leaf_cluster_stride = 0;
    }

    this->candidate_sci_numbers = candidate_sci_numbers;
#ifndef USE_CPU
    if (Clustered_Trace_Warp_Records_Enabled())
    {
        Trace_Clustered_Builder_Stats(*this, candidate_sci_numbers,
                                      build_cutoff, cell, rcell);
    }
#endif
#ifdef USE_CPU
    HostClusteredBuildInput host_input = {};
    host_input.total_atom_numbers = total_atom_numbers;
    host_input.leaf_numbers = leaf_numbers;
    host_input.super_cluster_clusters = super_cluster_clusters;
    host_input.local_atom_numbers = local_atom_numbers;
    host_input.cluster_size = cluster_size;
    host_input.candidate_sci_numbers = candidate_sci_numbers;
    host_input.cutoff = build_cutoff;
    host_input.cell = cell;
    host_input.rcell = rcell;
    host_input.permutation.resize((size_t)total_atom_numbers);
    host_input.cluster_offsets.resize((size_t)cluster_numbers + 1);
    host_input.cluster_valid_masks.resize((size_t)cluster_numbers);
    host_input.cluster_local_masks.resize((size_t)cluster_numbers);
    host_input.cluster_centers.resize((size_t)cluster_numbers);
    host_input.cluster_extents.resize((size_t)cluster_numbers);
    host_input.cluster_radii.resize((size_t)cluster_numbers);
    host_input.leaf_cluster_starts.resize((size_t)leaf_numbers);
    host_input.leaf_cluster_ends.resize((size_t)leaf_numbers);
    host_input.super_cluster_offsets.resize((size_t)super_cluster_numbers + 1);
    host_input.cluster_to_supercluster.resize((size_t)cluster_numbers, -1);
    host_input.super_cluster_centers.resize((size_t)super_cluster_numbers);
    host_input.sci_supercluster_ids.resize((size_t)candidate_sci_numbers);
    host_input.candidate_leaf_offsets.resize((size_t)candidate_sci_numbers + 1);
    host_input.candidate_leaf_ids.resize((size_t)candidate_leaf_numbers);
    host_input.excluded_list_start.resize((size_t)local_atom_numbers, 0);
    host_input.excluded_numbers.resize((size_t)local_atom_numbers, 0);

    deviceMemcpy(host_input.permutation.data(), d_sort_permutation,
                 sizeof(int) * total_atom_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.cluster_offsets.data(), d_cluster_offsets,
                 sizeof(int) * (cluster_numbers + 1),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.cluster_valid_masks.data(), d_cluster_valid_masks,
                 sizeof(unsigned int) * cluster_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.cluster_local_masks.data(), d_cluster_local_masks,
                 sizeof(unsigned int) * cluster_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.cluster_centers.data(), d_cluster_centers,
                 sizeof(VECTOR) * cluster_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.cluster_extents.data(), d_cluster_extents,
                 sizeof(VECTOR) * cluster_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.cluster_radii.data(), d_cluster_radii,
                 sizeof(float) * cluster_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.leaf_cluster_starts.data(), d_leaf_cluster_starts,
                 sizeof(int) * leaf_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.leaf_cluster_ends.data(), d_leaf_cluster_ends,
                 sizeof(int) * leaf_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.super_cluster_offsets.data(), d_super_cluster_offsets,
                 sizeof(int) * (super_cluster_numbers + 1),
                 deviceMemcpyDeviceToHost);
    for (int super_i = 0; super_i < super_cluster_numbers; super_i += 1)
    {
        const int cluster_start = host_input.super_cluster_offsets[(size_t)super_i];
        const int cluster_end =
            host_input.super_cluster_offsets[(size_t)super_i + 1];
        for (int cluster_i = cluster_start; cluster_i < cluster_end;
             cluster_i += 1)
        {
            host_input.cluster_to_supercluster[(size_t)cluster_i] = super_i;
        }
    }
    deviceMemcpy(host_input.super_cluster_centers.data(),
                 d_super_cluster_centers,
                 sizeof(VECTOR) * super_cluster_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.sci_supercluster_ids.data(), d_sci_supercluster_ids,
                 sizeof(int) * candidate_sci_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.candidate_leaf_offsets.data(),
                 d_sci_candidate_leaf_offsets,
                 sizeof(int) * (candidate_sci_numbers + 1),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(host_input.candidate_leaf_ids.data(), d_sci_candidate_leaf_ids,
                 sizeof(int) * candidate_leaf_numbers,
                 deviceMemcpyDeviceToHost);
    if (local_atom_numbers > 0 && d_excluded_list_start != NULL &&
        d_excluded_numbers != NULL)
    {
        deviceMemcpy(host_input.excluded_list_start.data(), d_excluded_list_start,
                     sizeof(int) * local_atom_numbers,
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(host_input.excluded_numbers.data(), d_excluded_numbers,
                     sizeof(int) * local_atom_numbers,
                     deviceMemcpyDeviceToHost);
        const int total_excluded =
            host_input.excluded_list_start.back() +
            host_input.excluded_numbers.back();
        host_input.excluded_list.resize((size_t)IntMax(total_excluded, 0));
        if (total_excluded > 0 && d_excluded_list != NULL)
        {
            deviceMemcpy(host_input.excluded_list.data(), d_excluded_list,
                         sizeof(int) * total_excluded,
                         deviceMemcpyDeviceToHost);
        }
    }

    std::vector<LJ_CLUSTERED_SCI> host_scis;
    std::vector<LJ_CLUSTERED_CJ_PACKED> host_cjpacked;
    std::vector<unsigned long long> host_exclusion_pool;
    Build_Nbnxm_Payload_On_Host(host_input, &host_scis, &host_cjpacked,
                                &host_exclusion_pool);
    std::stable_sort(
        host_scis.begin(), host_scis.end(),
        [](const LJ_CLUSTERED_SCI& lhs, const LJ_CLUSTERED_SCI& rhs)
        {
            return (lhs.cjpacked_end - lhs.cjpacked_begin) >
                   (rhs.cjpacked_end - rhs.cjpacked_begin);
        });

    sci_numbers = static_cast<int>(host_scis.size());
    cjpacked_numbers = static_cast<int>(host_cjpacked.size());
    exclusion_pool_numbers = static_cast<int>(host_exclusion_pool.size());
    if (sci_numbers <= 0 || cjpacked_numbers <= 0)
    {
        if (Clustered_Trace_Warp_Records_Enabled())
        {
            fprintf(stderr,
                    "[clustered build early-return] step=%d stage=no-final-payload sci=%d cjpacked=%d excl=%d\n",
                    md_info.sys.steps, sci_numbers, cjpacked_numbers,
                    exclusion_pool_numbers);
        }
        Commit_Clustered_Build_Cache(this, crd, cutoff);
        return;
    }

    Reserve_Device_Buffer(sci_numbers, &d_nbnxm_sci, &nbnxm_sci_capacity);
    Reserve_Device_Buffer(cjpacked_numbers, &d_nbnxm_cjpacked,
                          &nbnxm_cjpacked_capacity);
    deviceMemcpy(d_nbnxm_sci, host_scis.data(),
                 sizeof(LJ_CLUSTERED_SCI) * sci_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_nbnxm_cjpacked, host_cjpacked.data(),
                 sizeof(LJ_CLUSTERED_CJ_PACKED) * cjpacked_numbers,
                 deviceMemcpyHostToDevice);
    if (exclusion_pool_numbers > 0)
    {
        Reserve_Device_PairMask_Buffer(exclusion_pool_numbers,
                                       &d_exclusion_mask_pool,
                                       &exclusion_capacity);
        deviceMemcpy(d_exclusion_mask_pool, host_exclusion_pool.data(),
                     sizeof(unsigned long long) * exclusion_pool_numbers,
                     deviceMemcpyHostToDevice);
    }
#else
    const int sci_shift_numbers = fixed_shift_candidates
                                      ? candidate_sci_numbers
                                      : candidate_sci_numbers *
                                            kClusteredShiftCount;
    Reserve_Device_Int_Buffer(sci_shift_numbers, &d_sci_shift_flags,
                              &sci_shift_capacity);
    Reserve_Device_Int_Buffer(sci_shift_numbers, &d_cjpacked_counts,
                              &cjpacked_count_capacity);
    Reserve_Device_Int_Buffer(sci_shift_numbers, &d_exclusion_counts,
                              &exclusion_count_capacity);
    Reserve_Device_Int_Buffer(sci_shift_numbers + 1, &d_sci_shift_offsets,
                              &sci_shift_offset_capacity);
    Reserve_Device_Int_Buffer(sci_shift_numbers + 1, &d_cjpacked_group_offsets,
                              &cjpacked_group_offset_capacity);
    Reserve_Device_Int_Buffer(sci_shift_numbers + 1, &d_exclusion_offsets,
                              &exclusion_offset_capacity);
    const int builder_warps_per_block =
        kClusteredBuilderBlockSize / kClusteredBuilderWarpSize;
    const int candidate_sci_blocks =
        (candidate_sci_numbers + builder_warps_per_block - 1) /
        builder_warps_per_block;

    Launch_Device_Kernel(
        Count_Nbnxm_Payload_From_Candidate_Leaves,
        candidate_sci_blocks, kClusteredBuilderBlockSize, 0, NULL,
        candidate_sci_numbers,
        sci_shift_numbers, cluster_size, super_cluster_clusters,
        local_atom_numbers, build_cutoff, cell, rcell, d_sort_permutation,
        d_cluster_offsets, d_leaf_cluster_starts, d_leaf_cluster_ends,
        d_super_cluster_offsets, d_cluster_to_supercluster,
        candidate_sci_supercluster_ids, d_super_cluster_centers,
        candidate_shift_ids,
        d_sci_candidate_leaf_offsets, d_sci_candidate_leaf_ids,
        candidate_leaf_cluster_stride,
        fixed_shift_leaf_screening ? d_candidate_leaf_reach_masks : NULL,
        d_cluster_valid_masks, d_cluster_local_masks, d_cluster_centers,
        d_cluster_extents, d_cluster_radii, cluster_molecule_signatures,
        cluster_molecule_ids,
        d_excluded_list_start, d_excluded_list, d_excluded_numbers,
        fixed_shift_candidates,
        d_sci_shift_flags, d_cjpacked_counts, d_exclusion_counts);
#ifndef USE_CPU
    Clustered_Debug_Device_Sync_If_Tracing(
        "Count_Nbnxm_Payload_From_Candidate_Leaves");
#endif

    sci_numbers = Exclusive_Scan_Counts(this, sci_shift_numbers,
                                        d_sci_shift_flags,
                                        d_sci_shift_offsets);
    cjpacked_numbers = Exclusive_Scan_Counts(
        this, sci_shift_numbers, d_cjpacked_counts,
        d_cjpacked_group_offsets);
    exclusion_pool_numbers =
        Exclusive_Scan_Counts(this, sci_shift_numbers, d_exclusion_counts,
                              d_exclusion_offsets);
    if (Clustered_Trace_Warp_Records_Enabled())
    {
        fprintf(stderr,
                "[clustered payload count] step=%d leaves=%d clusters=%d "
                "superclusters=%d candidate_sci=%d candidate_leaves=%d "
                "sci_shift=%d sci=%d cjpacked=%d excl=%d blocks=%d "
                "fixed_shift=%d dense_shift=%d sparse_shift=%d\n",
                md_info.sys.steps, leaf_numbers, cluster_numbers,
                super_cluster_numbers, candidate_sci_numbers,
                candidate_leaf_numbers, sci_shift_numbers, sci_numbers,
                cjpacked_numbers, exclusion_pool_numbers,
                candidate_sci_blocks, fixed_shift_candidates ? 1 : 0,
                dense_shift_partitioned_candidates ? 1 : 0,
                sparse_shift_candidates ? 1 : 0);
    }
    if (sci_numbers <= 0 || cjpacked_numbers <= 0)
    {
        Commit_Clustered_Build_Cache(this, crd, cutoff);
        return;
    }

    const bool run_reduced_staged_count =
        need_gmxpacked_payload && runtime_gmxpacked_direct_requested &&
        Clustered_Gmxpacked_Reduced_Build_Enabled();
    const bool build_gmxpacked_direct_candidate =
        need_gmxpacked_payload && runtime_gmxpacked_direct_requested &&
        Clustered_Gmxpacked_Direct_Candidate_Build_Enabled() &&
        !run_reduced_staged_count;
    const bool run_early_record_analysis =
        Clustered_Gmxpacked_Early_Record_Analyze_Enabled();

    Reserve_Device_Buffer(sci_numbers, &d_nbnxm_sci, &nbnxm_sci_capacity);
    Reserve_Device_Buffer(cjpacked_numbers, &d_nbnxm_cjpacked,
                          &nbnxm_cjpacked_capacity);
    if (exclusion_pool_numbers > 0)
    {
        Reserve_Device_PairMask_Buffer(exclusion_pool_numbers,
                                       &d_exclusion_mask_pool,
                                       &exclusion_capacity);
    }
    if (build_gmxpacked_direct_candidate)
    {
        gmxpacked_sci_numbers = sci_numbers;
        gmxpacked_cjpacked_numbers = cjpacked_numbers;
        gmxpacked_split_exclusion_numbers =
            cjpacked_numbers * kClusteredWarpSplitCount;
        gmxpacked_exclusion_numbers = 1 + gmxpacked_split_exclusion_numbers;
        Reserve_Device_Buffer(gmxpacked_sci_numbers, &d_gmxpacked_sci,
                              &gmxpacked_sci_capacity);
        Reserve_Device_Buffer(gmxpacked_cjpacked_numbers,
                              &d_gmxpacked_cjpacked,
                              &gmxpacked_cjpacked_capacity);
        Reserve_Device_Buffer(gmxpacked_exclusion_numbers,
                              &d_gmxpacked_exclusions,
                              &gmxpacked_exclusion_capacity);
        Launch_Device_Kernel(
            Initialize_Gmxpacked_Exclusion_Rows,
            (gmxpacked_exclusion_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, gmxpacked_exclusion_numbers,
            d_gmxpacked_exclusions);
    }

    int* d_early_record_analysis_raw_count = NULL;
    int early_record_analysis_count_capacity = 0;
    ClusteredEarlyRecordAnalyzeEntry* d_early_record_analysis_entries = NULL;
    int early_record_analysis_entry_capacity = 0;
    const int early_record_analysis_capacity =
        run_early_record_analysis ? cjpacked_numbers * kClusteredJGroupSize : 0;
    if (early_record_analysis_capacity > 0)
    {
        Reserve_Device_Int_Buffer(1, &d_early_record_analysis_raw_count,
                                  &early_record_analysis_count_capacity);
        deviceMemset(d_early_record_analysis_raw_count, 0, sizeof(int));
        Reserve_Device_Buffer(early_record_analysis_capacity,
                              &d_early_record_analysis_entries,
                              &early_record_analysis_entry_capacity);
    }

    const bool dedup_exclusion_masks =
        Clustered_Dedup_Exclusion_Masks_Enabled();
    Launch_Device_Kernel(
        Fill_Nbnxm_Payload_From_Candidate_Leaves,
        candidate_sci_blocks, kClusteredBuilderBlockSize, 0, NULL,
        candidate_sci_numbers,
        sci_shift_numbers, cluster_size, super_cluster_clusters, cluster_numbers,
        local_atom_numbers, build_cutoff, cell, rcell, d_sort_permutation,
        d_cluster_offsets, d_leaf_cluster_starts, d_leaf_cluster_ends,
        d_super_cluster_offsets, d_cluster_to_supercluster,
        candidate_sci_supercluster_ids, d_super_cluster_centers,
        candidate_shift_ids,
        d_sci_candidate_leaf_offsets, d_sci_candidate_leaf_ids,
        candidate_leaf_cluster_stride,
        fixed_shift_leaf_screening ? d_candidate_leaf_reach_masks : NULL,
        d_cluster_valid_masks, d_cluster_local_masks, d_cluster_centers,
        d_cluster_extents, d_cluster_radii, cluster_molecule_signatures,
        cluster_molecule_ids,
        d_excluded_list_start, d_excluded_list, d_excluded_numbers,
        fixed_shift_candidates, dedup_exclusion_masks,
        d_sci_shift_flags, d_sci_shift_offsets, d_cjpacked_counts,
        d_cjpacked_group_offsets, d_exclusion_offsets,
        build_gmxpacked_direct_candidate, d_nbnxm_sci, d_nbnxm_cjpacked,
        d_exclusion_mask_pool, d_gmxpacked_sci, d_gmxpacked_cjpacked,
        d_gmxpacked_exclusions, early_record_analysis_capacity,
        d_early_record_analysis_raw_count, d_early_record_analysis_entries);
#ifndef USE_CPU
    Clustered_Debug_Device_Sync_If_Tracing(
        "Fill_Nbnxm_Payload_From_Candidate_Leaves");
    if (early_record_analysis_capacity > 0)
    {
        Analyze_Clustered_Early_Record_Stream(
            this, early_record_analysis_capacity,
            d_early_record_analysis_raw_count,
            d_early_record_analysis_entries);
        Free_Single_Device_Pointer((void**)&d_early_record_analysis_raw_count);
        Free_Single_Device_Pointer((void**)&d_early_record_analysis_entries);
    }
#endif
    if (Clustered_Block_Sort_J_Entries_Enabled())
    {
        Rebuild_Payload_From_Sorted_J_Entries(this, true);
        sci_numbers = this->sci_numbers;
        cjpacked_numbers = this->cjpacked_numbers;
    }
    else if (Clustered_Sort_J_Entries_Enabled())
    {
        Rebuild_Payload_From_Sorted_J_Entries(this, false);
        sci_numbers = this->sci_numbers;
        cjpacked_numbers = this->cjpacked_numbers;
    }
    if (Clustered_Sort_CjPacked_Enabled())
    {
        Reorder_CjPacked_By_Cluster_J(this);
    }
    Stable_Sort_Sci_By_Workload(sci_numbers, d_nbnxm_sci, d_sci_shift_flags,
                                this);

    if (!runtime_gmxpacked_direct_requested && need_virial &&
        !prefer_full_record_builder)
    {
        Build_Grouped_Sci_Metadata(this);
    }
#endif
    if (!runtime_gmxpacked_direct_requested &&
        Clustered_Outer_Inner_Prune_Enabled(this) && cjpacked_numbers > 0)
    {
        Capture_Clustered_Outer_Imask(this);
    }
    if (!runtime_gmxpacked_direct_requested && cjpacked_numbers > 0)
    {
        Reserve_Device_U64_Buffer(cjpacked_numbers * kClusteredJGroupSize,
                                  &d_pair_shift_bits, &pair_shift_capacity);
    }
    if (need_gmxpacked_payload && !build_gmxpacked_direct_candidate)
    {
        Build_Gmxpacked_Payload(this);
        if (runtime_gmxpacked_direct_requested && gmxpacked_cjpacked_numbers > 0)
        {
            Reserve_Device_U64_Buffer(gmxpacked_cjpacked_numbers *
                                          kClusteredJGroupSize,
                                      &d_pair_shift_bits, &pair_shift_capacity);
            Refresh_Gmxpacked_Pair_Shift_Metadata(this, rcell);
        }
    }
    else if (build_gmxpacked_direct_candidate && gmxpacked_cjpacked_numbers > 0)
    {
        Reserve_Device_U64_Buffer(gmxpacked_cjpacked_numbers * kClusteredJGroupSize,
                                  &d_pair_shift_bits, &pair_shift_capacity);
        Refresh_Gmxpacked_Pair_Shift_Metadata(this, rcell);
        Clustered_Debug_Device_Sync_If_Tracing(
            "Fill_Nbnxm_Payload_From_Candidate_Leaves direct-gmxpacked");
        if (Clustered_Trace_Warp_Records_Enabled())
        {
            fprintf(stderr,
                    "[clustered primary gmxpacked payload] step=%d "
                    "step_builds=%d total_builds=%lld compact_sci=%d "
                    "compact_cj=%d compact_excl=%d split_excl=%d "
                    "compat_native_sci=%d compat_native_cj=%d "
                    "compat_native_excl_pool=%d source=direct-candidate\n",
                    md_info.sys.steps, primary_payload_build_count_this_step,
                    primary_payload_build_count_total, gmxpacked_sci_numbers,
                    gmxpacked_cjpacked_numbers, gmxpacked_exclusion_numbers,
                    gmxpacked_split_exclusion_numbers, sci_numbers,
                    cjpacked_numbers, exclusion_pool_numbers);
            fflush(stderr);
        }
    }
#ifndef USE_CPU
    if (build_warp_records && sci_numbers > 0 && cjpacked_numbers > 0)
    {
        Rebuild_Clustered_Forceonly_Warp_Records(this);
        if (need_virial && prefer_full_record_builder && !grouped_sci_ready &&
            forceonly_warp_record_numbers == 0 && sci_numbers > 0)
        {
            Build_Grouped_Sci_Metadata(this);
        }
    }
#endif
    if (grouped_sci_ready)
    {
        Analyze_Grouped_Clustered_J_Reuse(*this);
    }
    (void)leaf_atom_total;
    Commit_Clustered_Build_Cache(this, crd, cutoff);
}

void LJ_CLUSTER_LAYOUT::Clear()
{
#ifndef USE_CPU
    if (controller != NULL)
    {
        working_device = controller->working_device;
    }
    Bind_Clustered_Working_Device(&working_device);
#endif
    Free_Single_Device_Pointer((void**)&d_sort_permutation);
    Free_Single_Device_Pointer((void**)&d_sort_keys);
    Free_Single_Device_Pointer((void**)&d_cached_crd);
    Free_Single_Device_Pointer((void**)&d_need_rebuild);
    Free_Single_Device_Pointer((void**)&d_global_atom_to_molecule);
    Free_Single_Device_Pointer((void**)&d_local_atom_to_molecule);
    Free_Single_Device_Pointer((void**)&d_cluster_offsets);
    Free_Single_Device_Pointer((void**)&d_cluster_valid_masks);
    Free_Single_Device_Pointer((void**)&d_cluster_local_masks);
    Free_Single_Device_Pointer((void**)&d_cluster_centers);
    Free_Single_Device_Pointer((void**)&d_cluster_extents);
    Free_Single_Device_Pointer((void**)&d_cluster_radii);
    Free_Single_Device_Pointer((void**)&d_cluster_molecule_signatures);
    Free_Single_Device_Pointer((void**)&d_cluster_molecule_ids);
    Free_Single_Device_Pointer((void**)&d_leaf_atom_offsets);
    Free_Single_Device_Pointer((void**)&d_leaf_cluster_starts);
    Free_Single_Device_Pointer((void**)&d_leaf_cluster_ends);
    Free_Single_Device_Pointer((void**)&d_leaf_all_local);
    Free_Single_Device_Pointer((void**)&d_super_cluster_offsets);
    Free_Single_Device_Pointer((void**)&d_cluster_to_supercluster);
    Free_Single_Device_Pointer((void**)&d_super_cluster_has_local);
    Free_Single_Device_Pointer((void**)&d_super_cluster_centers);
    Free_Single_Device_Pointer((void**)&d_super_cluster_sizes);
    Free_Single_Device_Pointer((void**)&d_sci_supercluster_ids);
    Free_Single_Device_Pointer((void**)&d_sci_candidate_leaf_counts);
    Free_Single_Device_Pointer((void**)&d_sci_candidate_leaf_offsets);
    Free_Single_Device_Pointer((void**)&d_sci_candidate_leaf_ids);
    Free_Single_Device_Pointer((void**)&d_candidate_leaf_reach_masks);
    Free_Single_Device_Pointer((void**)&d_candidate_sci_offsets);
    Free_Single_Device_Pointer((void**)&d_candidate_shift_ids);
    Free_Single_Device_Pointer((void**)&d_grouped_sci_offsets);
    Free_Single_Device_Pointer((void**)&d_grouped_sci_ids);
    Free_Single_Device_Pointer((void**)&d_cjpacked_counts);
    Free_Single_Device_Pointer((void**)&d_exclusion_counts);
    Free_Single_Device_Pointer((void**)&d_exclusion_offsets);
    Free_Single_Device_Pointer((void**)&d_sci_shift_flags);
    Free_Single_Device_Pointer((void**)&d_sci_shift_offsets);
    Free_Single_Device_Pointer((void**)&d_cjpacked_group_offsets);
    Free_Single_Device_Pointer((void**)&d_outer_imask);
    Free_Single_Device_Pointer((void**)&d_exclusion_mask_pool);
    Free_Single_Device_Pointer((void**)&d_nbnxm_sci);
    Free_Single_Device_Pointer((void**)&d_nbnxm_cjpacked);
    Free_Gmxpacked_Payload(this);
    Free_Single_Device_Pointer((void**)&d_cjpacked_sort_indices);
    Free_Single_Device_Pointer((void**)&d_cjpacked_sort_buffer);
    Free_Single_Device_Pointer((void**)&d_j_entries);
    Free_Single_Device_Pointer((void**)&d_j_entry_buffer);
    Free_Single_Device_Pointer((void**)&d_jentry_counts);
    Free_Single_Device_Pointer((void**)&d_jentry_offsets);
    Free_Single_Device_Pointer((void**)&d_jentry_indices);
    Free_Single_Device_Pointer((void**)&d_nbnxm_warp_j_records);
    Free_Single_Device_Pointer((void**)&d_forceonly_warp_record_counts);
    Free_Single_Device_Pointer((void**)&d_forceonly_warp_record_offsets);
    Free_Single_Device_Pointer((void**)&d_forceonly_warp_j_records);
    Free_Single_Device_Pointer((void**)&d_pair_shift_bits);
    Free_Single_Device_Pointer((void**)&d_sort_key_buffer);
    Free_Single_Device_Pointer((void**)&d_sort_value_buffer);
    Free_Single_Device_Pointer((void**)&d_sort_temp_storage);
    Free_Single_Device_Pointer((void**)&d_reduce_temp_storage);
    Free_Single_Device_Pointer((void**)&d_scan_temp_storage);
    Free_Single_Device_Pointer((void**)&d_scan_total);

    delete cornerstone_state;
    cornerstone_state = NULL;

    permutation_capacity = 0;
    sort_key_capacity = 0;
    cluster_capacity = 0;
    cluster_valid_mask_capacity = 0;
    cluster_local_mask_capacity = 0;
    cluster_center_capacity = 0;
    cluster_extent_capacity = 0;
    cluster_radius_capacity = 0;
    leaf_capacity = 0;
    leaf_cluster_start_capacity = 0;
    leaf_cluster_end_capacity = 0;
    leaf_all_local_capacity = 0;
    super_cluster_capacity = 0;
    super_cluster_has_local_capacity = 0;
    super_cluster_center_capacity = 0;
    super_cluster_size_capacity = 0;
    sci_capacity = 0;
    sci_candidate_leaf_count_capacity = 0;
    sci_candidate_leaf_offset_capacity = 0;
    candidate_leaf_capacity = 0;
    candidate_leaf_mask_capacity = 0;
    cjpacked_capacity = 0;
    exclusion_scan_capacity = 0;
    exclusion_capacity = 0;
    scan_capacity = 0;
    sci_shift_capacity = 0;
    cjpacked_count_capacity = 0;
    exclusion_count_capacity = 0;
    sci_shift_offset_capacity = 0;
    cjpacked_group_offset_capacity = 0;
    exclusion_offset_capacity = 0;
    candidate_offset_capacity = 0;
    candidate_shift_capacity = 0;
    nbnxm_sci_capacity = 0;
    nbnxm_cjpacked_capacity = 0;
    gmxpacked_sci_capacity = 0;
    gmxpacked_cjpacked_capacity = 0;
    gmxpacked_exclusion_capacity = 0;
    cjpacked_sort_index_capacity = 0;
    cjpacked_sort_buffer_capacity = 0;
    jentry_capacity = 0;
    jentry_buffer_capacity = 0;
    jentry_count_capacity = 0;
    jentry_offset_capacity = 0;
    jentry_index_capacity = 0;
    nbnxm_warp_record_capacity = 0;
    forceonly_warp_record_numbers = 0;
    forceonly_warp_record_capacity = 0;
    forceonly_warp_record_count_capacity = 0;
    forceonly_warp_record_offset_capacity = 0;
    pair_shift_capacity = 0;
    outer_imask_capacity = 0;
    global_atom_to_molecule_capacity = 0;
    local_atom_to_molecule_capacity = 0;
    cluster_molecule_signature_capacity = 0;
    cluster_molecule_id_capacity = 0;
    sort_key_buffer_bytes = 0;
    sort_value_buffer_bytes = 0;
    sort_temp_storage_bytes = 0;
    reduce_temp_storage_bytes = 0;
    scan_temp_storage_bytes = 0;
    cached_crd_capacity = 0;
    rebuild_flag_capacity = 0;
    local_atom_numbers = 0;
    direct_local_atom_numbers = 0;
    ghost_numbers = 0;
    total_atom_numbers = 0;
    cluster_numbers = 0;
    super_cluster_numbers = 0;
    local_cluster_numbers = 0;
    candidate_sci_numbers = 0;
    sci_numbers = 0;
    cjpacked_numbers = 0;
    gmxpacked_sci_numbers = 0;
    gmxpacked_cjpacked_numbers = 0;
    gmxpacked_exclusion_numbers = 0;
    gmxpacked_split_exclusion_numbers = 0;
    candidate_leaf_numbers = 0;
    candidate_leaf_cluster_stride = 0;
    exclusion_pool_numbers = 0;
    grouped_sci_ready = false;
    rebuild_dirty = true;
    cache_ready = false;
    cached_cutoff = -1.0f;
    primary_payload_time_recorder = NULL;
    primary_payload_build_step = -1;
    primary_payload_build_count_this_step = 0;
    primary_payload_build_count_total = 0;
    Invalidate_Clustered_Legacy_Neighbor_View(this);
    d_atom_local = NULL;
    d_excluded_list_start = NULL;
    d_excluded_list = NULL;
    d_excluded_numbers = NULL;
}

bool Ensure_Legacy_Neighbor_View_From_Clustered_Payload(
    LJ_CLUSTERED_DIRECT_CACHE* cache,
    const LJ_CLUSTERED_LEGACY_NEIGHBOR_VIEW_REQUEST& request,
    ATOM_GROUP* d_legacy_nl, int max_neighbor_numbers,
    int* d_neighbor_list_overflow, const char** fallback_reason)
{
    const char* reason = "clustered legacy adapter not evaluated";
    auto fail = [&](const char* why) -> bool
    {
        reason = why;
        if (cache != NULL)
        {
            Invalidate_Clustered_Legacy_Neighbor_View(&cache->layout);
        }
        if (fallback_reason != NULL)
        {
            *fallback_reason = reason;
        }
        return false;
    };

    if (cache == NULL || !cache->initialized)
    {
        return fail("clustered payload cache is absent or uninitialized");
    }
    LJ_CLUSTER_LAYOUT& layout = cache->layout;
    Invalidate_Clustered_Legacy_Neighbor_View(&layout);
    if (!layout.Use_Clustered_Direct())
    {
        return fail("clustered direct payload is disabled");
    }
    if (!request.request_half)
    {
        return fail("requested legacy view does not require a half-list");
    }
    if (request.request_full)
    {
        return fail("full-list ATOM_GROUP semantics are unproven for clustered payloads");
    }
    if (request.contains_non_lj_consumer)
    {
        return fail("non-LJ half-list consumers are not proven against clustered ownership");
    }
    if (d_legacy_nl == NULL || max_neighbor_numbers <= 0 ||
        d_neighbor_list_overflow == NULL)
    {
        return fail("legacy ATOM_GROUP output storage is unavailable");
    }
    if (!layout.cache_ready || layout.gmxpacked_sci_numbers <= 0 ||
        layout.gmxpacked_cjpacked_numbers <= 0 ||
        layout.d_gmxpacked_sci == NULL || layout.d_gmxpacked_cjpacked == NULL ||
        layout.d_cluster_offsets == NULL || layout.d_cluster_valid_masks == NULL ||
        layout.d_cluster_local_masks == NULL || layout.d_sort_permutation == NULL ||
        layout.d_super_cluster_offsets == NULL)
    {
        return fail("compact clustered payload is absent or not ready");
    }
    if (request.local_atom_numbers != layout.local_atom_numbers ||
        request.ghost_numbers != layout.ghost_numbers)
    {
        return fail("legacy local/ghost domain differs from clustered payload domain");
    }
    if (request.require_all_local_atoms &&
        layout.direct_local_atom_numbers != request.local_atom_numbers)
    {
        return fail("clustered payload covers only the direct local subset");
    }
    if (request.require_local_ghost_pairs && layout.total_atom_numbers !=
                                             request.local_atom_numbers +
                                                 request.ghost_numbers)
    {
        return fail("clustered payload total does not cover local plus ghost atoms");
    }
    if (request.require_exclusions &&
        (layout.d_atom_local != request.d_atom_local ||
         layout.d_excluded_list_start != request.d_excluded_list_start ||
         layout.d_excluded_list != request.d_excluded_list ||
         layout.d_excluded_numbers != request.d_excluded_numbers))
    {
        return fail("clustered exclusion metadata does not match requested legacy view");
    }

    const float requested_cutoff_skin = request.cutoff + request.skin;
    const float payload_cutoff_skin =
        layout.cached_cutoff +
        (Clustered_Outer_Inner_Prune_Enabled(&layout) ? layout.rebuild_skin
                                                      : 0.0f);
    if (requested_cutoff_skin < 0.0f ||
        payload_cutoff_skin + 1.0e-4f < requested_cutoff_skin)
    {
        return fail("clustered cutoff/skin is not a semantic superset");
    }

    layout.legacy_neighbor_view_payload_build_count =
        layout.primary_payload_build_count_total;
    layout.legacy_neighbor_view_step = md_info.sys.steps;
    layout.legacy_neighbor_view_cutoff_skin = requested_cutoff_skin;
    return fail(
        "half-list pair-set proof is missing; grid legacy build remains required");
}

void LJ_CLUSTERED_DIRECT_CACHE::Initial(CONTROLLER* controller,
                                        const char* module_name,
                                        bool ordered_layout_enabled)
{
    if (initialized)
    {
        return;
    }
    initialized = true;
    layout.Initial(controller, module_name, ordered_layout_enabled);
    payload_gather_time_recorder =
        controller->Get_Time_Recorder("clustered coordinate gather");
    direct_kernel_time_recorder =
        controller->Get_Time_Recorder("clustered direct kernel");
    coordinate_gather_step = -1;
    coordinate_gather_count_this_step = 0;
    coordinate_gather_count_total = 0;
}

void LJ_CLUSTERED_DIRECT_CACHE::Refresh_Metadata(
    int local_atom_numbers, int direct_local_atom_numbers, int ghost_numbers,
    const int* d_atom_local,
    const int* d_excluded_list_start, const int* d_excluded_list,
    const int* d_excluded_numbers)
{
    if (!initialized)
    {
        return;
    }
    layout.Refresh_Metadata(local_atom_numbers, direct_local_atom_numbers,
                            ghost_numbers, d_atom_local,
                            d_excluded_list_start, d_excluded_list,
                            d_excluded_numbers);
}

void LJ_CLUSTERED_DIRECT_CACHE::Build(const VECTOR* crd, LTMatrix3 cell,
                                      LTMatrix3 rcell, float cutoff,
                                      bool need_virial,
                                      bool prefer_full_warp_record,
                                      bool need_gmxpacked_payload,
                                      bool need_aux_clustered_metadata,
                                      bool runtime_gmxpacked_direct_requested)
{
    if (!initialized)
    {
        return;
    }
    layout.Build(crd, cell, rcell, cutoff, need_virial,
                  prefer_full_warp_record, need_gmxpacked_payload,
                  need_aux_clustered_metadata,
                  runtime_gmxpacked_direct_requested);
}

void LJ_CLUSTERED_DIRECT_CACHE::Gather_Plain(const VECTOR* crd,
                                             const float* charge,
                                             const VECTOR_LJ* lj_type_src,
                                             LTMatrix3 cell, LTMatrix3 rcell)
{
    if (!initialized || !layout.Use_Clustered_Direct() ||
        layout.total_atom_numbers <= 0 || crd == NULL || charge == NULL ||
        lj_type_src == NULL)
    {
        return;
    }
#ifndef USE_CPU
    if (layout.controller != NULL)
    {
        layout.working_device = layout.controller->working_device;
    }
    Bind_Clustered_Working_Device(&layout.working_device);
#endif
    const int gather_step_count = Note_Clustered_Step_Counter(
        md_info.sys.steps, &coordinate_gather_step,
        &coordinate_gather_count_this_step, &coordinate_gather_count_total);
    if (Clustered_Trace_Warp_Records_Enabled())
    {
        fprintf(stderr,
                "[clustered coordinate gather] step=%d step_gathers=%d "
                "total_gathers=%lld source=plain-crd-charge atoms=%d\n",
                md_info.sys.steps, gather_step_count,
                coordinate_gather_count_total, layout.total_atom_numbers);
        fflush(stderr);
    }
    ClusteredRecorderScope gather_scope(payload_gather_time_recorder);
    Launch_Device_Kernel(
        Refresh_Current_Cluster_Centers_From_Crd,
        (layout.cluster_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout.cluster_numbers,
        layout.d_sort_permutation, layout.d_cluster_offsets, crd, cell, rcell,
        layout.d_cluster_centers);
    Reserve_Plain_Gather_Scratch(this);
    Launch_Device_Kernel(
        Gather_Sorted_LJ_Direct_Scratch_From_Plain,
        (layout.total_atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout.total_atom_numbers,
        layout.cluster_numbers, layout.d_sort_permutation,
        layout.d_cluster_offsets, layout.d_cluster_centers, cell, rcell, crd,
        charge, lj_type_src, d_sorted_atom_ids, d_sorted_xq,
        d_sorted_lj_type);
    Refresh_Plain_Gather_Dependent_Metadata(this, cell, rcell);
}

void LJ_CLUSTERED_DIRECT_CACHE::Gather_Plain(const VECTOR_LJ* src,
                                             LTMatrix3 cell, LTMatrix3 rcell)
{
    if (!initialized || !layout.Use_Clustered_Direct() ||
        layout.total_atom_numbers <= 0 || src == NULL)
    {
        return;
    }
#ifndef USE_CPU
    if (layout.controller != NULL)
    {
        layout.working_device = layout.controller->working_device;
    }
    Bind_Clustered_Working_Device(&layout.working_device);
#endif
    const int gather_step_count = Note_Clustered_Step_Counter(
        md_info.sys.steps, &coordinate_gather_step,
        &coordinate_gather_count_this_step, &coordinate_gather_count_total);
    if (Clustered_Trace_Warp_Records_Enabled())
    {
        fprintf(stderr,
                "[clustered coordinate gather] step=%d step_gathers=%d "
                "total_gathers=%lld source=compat-vector-lj atoms=%d\n",
                md_info.sys.steps, gather_step_count,
                coordinate_gather_count_total, layout.total_atom_numbers);
        fflush(stderr);
    }
    ClusteredRecorderScope gather_scope(payload_gather_time_recorder);
    Launch_Device_Kernel(
        Refresh_Current_Cluster_Centers<VECTOR_LJ>,
        (layout.cluster_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout.cluster_numbers,
        layout.d_sort_permutation, layout.d_cluster_offsets, src, cell, rcell,
        layout.d_cluster_centers);
    Reserve_Plain_Gather_Scratch(this);
    Launch_Device_Kernel(
        Gather_Sorted_LJ_Direct_Scratch,
        (layout.total_atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout.total_atom_numbers,
        layout.cluster_numbers, layout.d_sort_permutation,
        layout.d_cluster_offsets, layout.d_cluster_centers, cell, rcell, src,
        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type);
    Refresh_Plain_Gather_Dependent_Metadata(this, cell, rcell);
}

void LJ_CLUSTERED_DIRECT_CACHE::Gather_Soft_Core(
    const VECTOR_LJ_SOFT_TYPE* src)
{
    if (!initialized || !layout.Use_Clustered_Direct() ||
        layout.total_atom_numbers <= 0)
    {
        return;
    }
#ifndef USE_CPU
    if (layout.controller != NULL)
    {
        layout.working_device = layout.controller->working_device;
    }
    Bind_Clustered_Working_Device(&layout.working_device);
#endif
    ClusteredRecorderScope gather_scope(payload_gather_time_recorder);
    Reserve_Device_Buffer(layout.total_atom_numbers, &d_sorted_soft_crd,
                          &scratch_capacity);
    Launch_Device_Kernel(
        Gather_Sorted_Soft_Core_Scratch,
        (layout.total_atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, layout.total_atom_numbers,
        layout.d_sort_permutation, src, d_sorted_soft_crd);
}

bool LJ_CLUSTERED_DIRECT_CACHE::Coordinate_Gather_Ready_For_Current_Step() const
{
    return coordinate_gather_step == md_info.sys.steps &&
           coordinate_gather_count_this_step > 0;
}

void LJ_CLUSTERED_DIRECT_CACHE::Clear()
{
    if (!initialized)
    {
        return;
    }
#ifndef USE_CPU
    if (layout.controller != NULL)
    {
        layout.working_device = layout.controller->working_device;
    }
    Bind_Clustered_Working_Device(&layout.working_device);
#endif
    Free_Single_Device_Pointer((void**)&d_sorted_atom_ids);
    Free_Single_Device_Pointer((void**)&d_sorted_xq);
    Free_Single_Device_Pointer((void**)&d_sorted_lj_type);
    Free_Single_Device_Pointer((void**)&d_sorted_frc);
    Free_Single_Device_Pointer((void**)&d_sorted_frc_x);
    Free_Single_Device_Pointer((void**)&d_sorted_frc_y);
    Free_Single_Device_Pointer((void**)&d_sorted_frc_z);
    Free_Single_Device_Pointer((void**)&d_sorted_soft_crd);
    scratch_capacity = 0;
    payload_gather_time_recorder = NULL;
    direct_kernel_time_recorder = NULL;
    coordinate_gather_step = -1;
    coordinate_gather_count_this_step = 0;
    coordinate_gather_count_total = 0;
    layout.Clear();
    initialized = false;
}

namespace
{
LJ_CLUSTERED_DIRECT_CACHE g_shared_clustered_direct_cache;
}

LJ_CLUSTERED_DIRECT_CACHE* Acquire_Shared_LJ_Clustered_Direct_Cache(
    CONTROLLER* controller, const char* module_name,
    bool ordered_layout_enabled)
{
    g_shared_clustered_direct_cache.Initial(controller, module_name,
                                            ordered_layout_enabled);
    return &g_shared_clustered_direct_cache;
}

void Release_Shared_LJ_Clustered_Direct_Cache()
{
    g_shared_clustered_direct_cache.Clear();
}
