#include "gather_experiment.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace nbnxm_microbench
{
namespace
{

constexpr int kClusterSize = 8;
constexpr int kProductionBlockSize = 1024;

struct Vec3
{
    float x;
    float y;
    float z;
};

struct Matrix3
{
    float a11;
    float a21;
    float a22;
    float a31;
    float a32;
    float a33;
};

struct VectorLj
{
    Vec3 crd;
    int lj_type;
    float charge;
};

struct GatherOutputs
{
    std::vector<Vec3> centers;
    std::vector<Vec3> fractional_centers;
    std::vector<Vec3> fractional_extents;
    std::vector<int> atom_ids;
    std::vector<Float4POD> xq;
    std::vector<int> lj_type;
    std::vector<Float2POD> lj_comb;
};

struct DeviceState
{
    int* permutation = nullptr;
    int* cluster_offsets = nullptr;
    VectorLj* src = nullptr;
    Float2POD* lj_ab = nullptr;
    Vec3* centers = nullptr;
    Vec3* fractional_centers = nullptr;
    Vec3* fractional_extents = nullptr;
    int* atom_ids = nullptr;
    Float4POD* xq = nullptr;
    int* lj_type = nullptr;
    Float2POD* lj_comb = nullptr;
};

enum class GatherMode
{
    reference,
    fused,
};

template <typename T>
void CheckCuda(T code, const char* what)
{
    if (code != cudaSuccess)
    {
        std::fprintf(stderr, "%s failed: %s\n", what,
                     cudaGetErrorString(code));
        std::exit(1);
    }
}

__host__ __device__ __forceinline__ Vec3 Add(Vec3 a, Vec3 b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ __forceinline__ Vec3 Subtract(Vec3 a, Vec3 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ __forceinline__ Vec3 Scale(float scale, Vec3 value)
{
    return {scale * value.x, scale * value.y, scale * value.z};
}

__host__ __device__ __forceinline__ Vec3 Multiply(Vec3 value, Matrix3 matrix)
{
    return {value.x * matrix.a11 + value.y * matrix.a21 +
                value.z * matrix.a31,
            value.y * matrix.a22 + value.z * matrix.a32,
            value.z * matrix.a33};
}

__host__ __device__ __forceinline__ Vec3 PeriodicDisplacement(
    Vec3 a, Vec3 b, Matrix3 cell, Matrix3 reciprocal_cell)
{
    const Vec3 dr = Subtract(a, b);
    Vec3 shift = Multiply(dr, reciprocal_cell);
    shift.x = floorf(shift.x + 0.5f);
    shift.y = floorf(shift.y + 0.5f);
    shift.z = floorf(shift.z + 0.5f);
    return Subtract(dr, Multiply(shift, cell));
}

__host__ __device__ __forceinline__ Vec3 PeriodicCoordinate(
    Vec3 value, Matrix3 cell, Matrix3 reciprocal_cell)
{
    Vec3 shift = Multiply(value, reciprocal_cell);
    shift.x = floorf(shift.x);
    shift.y = floorf(shift.y);
    shift.z = floorf(shift.z);
    return Subtract(value, Multiply(shift, cell));
}

__host__ __device__ __forceinline__ Vec3 FractionalCenter(
    Vec3 center, Matrix3 reciprocal_cell)
{
    Vec3 fractional = Multiply(center, reciprocal_cell);
    fractional.x -= floorf(fractional.x);
    fractional.y -= floorf(fractional.y);
    fractional.z -= floorf(fractional.z);
    return fractional;
}

__host__ __device__ __forceinline__ Vec3 FractionalExtent(
    Vec3 extent, Matrix3 reciprocal_cell)
{
    return {
        fabsf(extent.x * reciprocal_cell.a11) +
            fabsf(extent.y * reciprocal_cell.a21) +
            fabsf(extent.z * reciprocal_cell.a31),
        fabsf(extent.y * reciprocal_cell.a22) +
            fabsf(extent.z * reciprocal_cell.a32),
        fabsf(extent.z * reciprocal_cell.a33)};
}

__host__ __device__ __forceinline__ int GetLjType(int a, int b)
{
    int y = b - a;
    const int sign = y >> 31;
    y = (y ^ sign) - sign;
    const int sum = b + a;
    const int hi = (sum + y) >> 1;
    const int lo = (sum - y) >> 1;
    return (hi * (hi + 1) >> 1) + lo;
}

__device__ __forceinline__ Float2POD GatherCombination(
    int lj_type, const Float2POD* lj_ab)
{
    Float2POD combination = {};
    if (lj_ab != nullptr)
    {
        const Float2POD self = lj_ab[GetLjType(lj_type, lj_type)];
        combination.x = sqrtf(fmaxf(self.y, 0.0f));
        combination.y = sqrtf(fmaxf(self.x, 0.0f));
    }
    return combination;
}

__global__ void GatherExperimentBaselineGeometry(
    int cluster_count, const int* permutation, const int* cluster_offsets,
    const VectorLj* src, Matrix3 cell, Matrix3 reciprocal_cell, Vec3* centers,
    Vec3* fractional_centers, Vec3* fractional_extents)
{
    const int cluster_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster_i >= cluster_count)
    {
        return;
    }
    const int start = cluster_offsets[cluster_i];
    const int end = cluster_offsets[cluster_i + 1];
    const int count = end > start ? end - start : 0;
    Vec3 center = {};
    Vec3 extent = {};
    if (count > 0)
    {
        const Vec3 anchor = src[permutation[start]].crd;
        for (int sorted_i = start; sorted_i < end; ++sorted_i)
        {
            const Vec3 pos = src[permutation[sorted_i]].crd;
            center = Add(center,
                         Add(anchor, PeriodicDisplacement(
                                         pos, anchor, cell, reciprocal_cell)));
        }
        center = PeriodicCoordinate(Scale(1.0f / count, center), cell,
                                    reciprocal_cell);
        for (int sorted_i = start; sorted_i < end; ++sorted_i)
        {
            const Vec3 pos = src[permutation[sorted_i]].crd;
            const Vec3 dr =
                PeriodicDisplacement(pos, center, cell, reciprocal_cell);
            extent.x = fmaxf(extent.x, fabsf(dr.x));
            extent.y = fmaxf(extent.y, fabsf(dr.y));
            extent.z = fmaxf(extent.z, fabsf(dr.z));
        }
    }
    centers[cluster_i] = center;
    fractional_centers[cluster_i] =
        FractionalCenter(center, reciprocal_cell);
    fractional_extents[cluster_i] =
        FractionalExtent(extent, reciprocal_cell);
}

__device__ __forceinline__ int FindClusterForSortedIndex(
    int sorted_i, int cluster_count, const int* cluster_offsets)
{
    int lo = 0;
    int hi = cluster_count;
    while (lo + 1 < hi)
    {
        const int mid = (lo + hi) >> 1;
        if (cluster_offsets[mid] <= sorted_i)
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }
    return lo;
}

__global__ void GatherExperimentBaselineSorted(
    int atom_count, int cluster_count, const int* permutation,
    const int* cluster_offsets, const Vec3* centers, Matrix3 cell,
    Matrix3 reciprocal_cell, const VectorLj* src, int* atom_ids,
    Float4POD* xq, int* lj_type, const Float2POD* lj_ab,
    Float2POD* lj_comb)
{
    const int sorted_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_i >= atom_count)
    {
        return;
    }
    const int cluster_i = FindClusterForSortedIndex(
        sorted_i, cluster_count, cluster_offsets);
    const int atom_i = permutation[sorted_i];
    const VectorLj atom = src[atom_i];
    const Vec3 center = centers[cluster_i];
    const Vec3 shifted = Add(
        center, PeriodicDisplacement(atom.crd, center, cell, reciprocal_cell));
    atom_ids[sorted_i] = atom_i;
    xq[sorted_i] = {shifted.x, shifted.y, shifted.z, atom.charge};
    lj_type[sorted_i] = atom.lj_type;
    if (lj_comb != nullptr)
    {
        lj_comb[sorted_i] = GatherCombination(atom.lj_type, lj_ab);
    }
}

__device__ __forceinline__ unsigned int GatherSubgroupMask()
{
    const int warp_lane = threadIdx.x & 31;
    const int subgroup_base = warp_lane & ~(kClusterSize - 1);
    return 0xffu << subgroup_base;
}

__device__ __forceinline__ Vec3 BroadcastFromSubgroupLeader(
    unsigned int mask, Vec3 value)
{
    value.x = __shfl_sync(mask, value.x, 0, kClusterSize);
    value.y = __shfl_sync(mask, value.y, 0, kClusterSize);
    value.z = __shfl_sync(mask, value.z, 0, kClusterSize);
    return value;
}

__device__ __forceinline__ Vec3 ReduceSubgroupSum(unsigned int mask,
                                                   Vec3 value)
{
    for (int delta = kClusterSize >> 1; delta > 0; delta >>= 1)
    {
        value.x += __shfl_down_sync(mask, value.x, delta, kClusterSize);
        value.y += __shfl_down_sync(mask, value.y, delta, kClusterSize);
        value.z += __shfl_down_sync(mask, value.z, delta, kClusterSize);
    }
    return value;
}

__device__ __forceinline__ Vec3 ReduceSubgroupMax(unsigned int mask,
                                                   Vec3 value)
{
    for (int delta = kClusterSize >> 1; delta > 0; delta >>= 1)
    {
        value.x = fmaxf(
            value.x,
            __shfl_down_sync(mask, value.x, delta, kClusterSize));
        value.y = fmaxf(
            value.y,
            __shfl_down_sync(mask, value.y, delta, kClusterSize));
        value.z = fmaxf(
            value.z,
            __shfl_down_sync(mask, value.z, delta, kClusterSize));
    }
    return value;
}

__global__ void GatherExperimentFused(
    int cluster_count, const int* permutation, const int* cluster_offsets,
    const VectorLj* src, Matrix3 cell, Matrix3 reciprocal_cell, Vec3* centers,
    Vec3* fractional_centers, Vec3* fractional_extents, int* atom_ids,
    Float4POD* xq, int* lj_type, const Float2POD* lj_ab,
    Float2POD* lj_comb)
{
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int cluster_i = global_thread / kClusterSize;
    const int sublane = threadIdx.x & (kClusterSize - 1);
    const unsigned int subgroup_mask = GatherSubgroupMask();
    const bool valid_cluster = cluster_i < cluster_count;
    const int start = valid_cluster ? cluster_offsets[cluster_i] : 0;
    const int end = valid_cluster ? cluster_offsets[cluster_i + 1] : 0;
    const int count = end > start ? end - start : 0;
    const bool active = sublane < count;

    Vec3 center = {};
    Vec3 extent = {};
    Vec3 anchor = {};
    if (sublane == 0 && count > 0)
    {
        anchor = src[permutation[start]].crd;
    }
    anchor = BroadcastFromSubgroupLeader(subgroup_mask, anchor);
    Vec3 unwrapped = {};
    if (active)
    {
        const Vec3 pos = src[permutation[start + sublane]].crd;
        unwrapped = Add(
            anchor,
            PeriodicDisplacement(pos, anchor, cell, reciprocal_cell));
    }
    const Vec3 sum = ReduceSubgroupSum(subgroup_mask, unwrapped);
    if (sublane == 0 && count > 0)
    {
        center = PeriodicCoordinate(Scale(1.0f / count, sum), cell,
                                    reciprocal_cell);
    }
    center = BroadcastFromSubgroupLeader(subgroup_mask, center);
    Vec3 lane_extent = {};
    if (active)
    {
        const Vec3 pos = src[permutation[start + sublane]].crd;
        const Vec3 dr =
            PeriodicDisplacement(pos, center, cell, reciprocal_cell);
        lane_extent = {fabsf(dr.x), fabsf(dr.y), fabsf(dr.z)};
    }
    extent = ReduceSubgroupMax(subgroup_mask, lane_extent);

    if (sublane == 0 && valid_cluster)
    {
        centers[cluster_i] = center;
        fractional_centers[cluster_i] =
            FractionalCenter(center, reciprocal_cell);
        fractional_extents[cluster_i] =
            FractionalExtent(extent, reciprocal_cell);
    }
    if (active)
    {
        const int sorted_i = start + sublane;
        const int atom_i = permutation[sorted_i];
        const VectorLj atom = src[atom_i];
        const Vec3 shifted = Add(
            center,
            PeriodicDisplacement(atom.crd, center, cell, reciprocal_cell));
        atom_ids[sorted_i] = atom_i;
        xq[sorted_i] = {shifted.x, shifted.y, shifted.z, atom.charge};
        lj_type[sorted_i] = atom.lj_type;
        if (lj_comb != nullptr)
        {
            lj_comb[sorted_i] = GatherCombination(atom.lj_type, lj_ab);
        }
    }
}

Matrix3 MakeMatrix(const LTMatrix3POD& source)
{
    return {source.a11, source.a21, source.a22,
            source.a31, source.a32, source.a33};
}

Matrix3 Invert(Matrix3 matrix)
{
    Matrix3 inverse = {};
    inverse.a33 = 1.0f / matrix.a33;
    inverse.a32 = -matrix.a32 * inverse.a33 / matrix.a22;
    inverse.a31 =
        (matrix.a32 * matrix.a21 * inverse.a33 - matrix.a31 * matrix.a22) /
        (matrix.a11 * matrix.a22 * matrix.a33);
    inverse.a22 = 1.0f / matrix.a22;
    inverse.a21 = -matrix.a21 * inverse.a22 / matrix.a11;
    inverse.a11 = 1.0f / matrix.a11;
    return inverse;
}

template <typename T>
T* CopyToDevice(const std::vector<T>& values)
{
    T* pointer = nullptr;
    if (!values.empty())
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&pointer),
                             sizeof(T) * values.size()),
                  "cudaMalloc(input)");
        CheckCuda(cudaMemcpy(pointer, values.data(), sizeof(T) * values.size(),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(input)");
    }
    return pointer;
}

template <typename T>
T* AllocateDevice(size_t count)
{
    T* pointer = nullptr;
    if (count != 0)
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&pointer),
                             sizeof(T) * count),
                  "cudaMalloc(output)");
    }
    return pointer;
}

template <typename T>
std::vector<T> CopyFromDevice(const T* pointer, size_t count)
{
    std::vector<T> values(count);
    if (count != 0)
    {
        CheckCuda(cudaMemcpy(values.data(), pointer, sizeof(T) * count,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(output)");
    }
    return values;
}

void LaunchGather(GatherMode mode, const DeviceState& device,
                  int atom_count, int cluster_count, Matrix3 cell,
                  Matrix3 reciprocal_cell, int block_size, bool use_comb)
{
    const Float2POD* lj_ab = use_comb ? device.lj_ab : nullptr;
    Float2POD* lj_comb = use_comb ? device.lj_comb : nullptr;
    if (mode == GatherMode::reference)
    {
        GatherExperimentBaselineGeometry<<<
            (cluster_count + kProductionBlockSize - 1) /
                kProductionBlockSize,
            kProductionBlockSize>>>(
            cluster_count, device.permutation, device.cluster_offsets,
            device.src, cell, reciprocal_cell, device.centers,
            device.fractional_centers, device.fractional_extents);
        GatherExperimentBaselineSorted<<<
            (atom_count + kProductionBlockSize - 1) / kProductionBlockSize,
            kProductionBlockSize>>>(
            atom_count, cluster_count, device.permutation,
            device.cluster_offsets, device.centers, cell, reciprocal_cell,
            device.src, device.atom_ids, device.xq, device.lj_type, lj_ab,
            lj_comb);
    }
    else
    {
        const int total_threads = cluster_count * kClusterSize;
        const int grid = (total_threads + block_size - 1) / block_size;
        GatherExperimentFused<<<grid, block_size>>>(
            cluster_count, device.permutation, device.cluster_offsets,
            device.src, cell, reciprocal_cell, device.centers,
            device.fractional_centers, device.fractional_extents,
            device.atom_ids, device.xq, device.lj_type, lj_ab, lj_comb);
    }
    CheckCuda(cudaGetLastError(), "launch gather experiment");
}

double TimeGather(GatherMode mode, const DeviceState& device, int atom_count,
                  int cluster_count, Matrix3 cell, Matrix3 reciprocal_cell,
                  int block_size, bool use_comb, int warmup, int iters)
{
    for (int i = 0; i < warmup; ++i)
    {
        LaunchGather(mode, device, atom_count, cluster_count, cell,
                     reciprocal_cell, block_size, use_comb);
    }
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CheckCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    CheckCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    CheckCuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < iters; ++i)
    {
        LaunchGather(mode, device, atom_count, cluster_count, cell,
                     reciprocal_cell, block_size, use_comb);
    }
    CheckCuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float total_ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&total_ms, start, stop),
              "cudaEventElapsedTime");
    CheckCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    CheckCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    return static_cast<double>(total_ms) / iters;
}

GatherOutputs CaptureGather(GatherMode mode, const DeviceState& device,
                            int atom_count, int cluster_count, Matrix3 cell,
                            Matrix3 reciprocal_cell, int block_size,
                            bool use_comb)
{
    LaunchGather(mode, device, atom_count, cluster_count, cell,
                 reciprocal_cell, block_size, use_comb);
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(capture)");
    GatherOutputs outputs;
    outputs.centers = CopyFromDevice(device.centers, cluster_count);
    outputs.fractional_centers =
        CopyFromDevice(device.fractional_centers, cluster_count);
    outputs.fractional_extents =
        CopyFromDevice(device.fractional_extents, cluster_count);
    outputs.atom_ids = CopyFromDevice(device.atom_ids, atom_count);
    outputs.xq = CopyFromDevice(device.xq, atom_count);
    outputs.lj_type = CopyFromDevice(device.lj_type, atom_count);
    if (use_comb)
    {
        outputs.lj_comb = CopyFromDevice(device.lj_comb, atom_count);
    }
    return outputs;
}

void AccumulateDifference(float actual, float reference, double* max_abs,
                          double* max_scaled)
{
    if (!std::isfinite(actual) || !std::isfinite(reference))
    {
        *max_abs = std::numeric_limits<double>::infinity();
        *max_scaled = std::numeric_limits<double>::infinity();
        return;
    }
    const double difference =
        std::fabs(static_cast<double>(actual) - reference);
    *max_abs = std::max(*max_abs, difference);
    *max_scaled = std::max(
        *max_scaled,
        difference / (1.0 + std::fabs(static_cast<double>(reference))));
}

bool CompareGatherOutputs(const GatherOutputs& actual,
                          const GatherOutputs& reference, const char* label,
                          bool use_comb)
{
    size_t integer_mismatches = 0;
    double max_abs = 0.0;
    double max_scaled = 0.0;
    for (size_t i = 0; i < reference.centers.size(); ++i)
    {
        const Vec3 actual_values[] = {actual.centers[i],
                                      actual.fractional_centers[i],
                                      actual.fractional_extents[i]};
        const Vec3 reference_values[] = {reference.centers[i],
                                         reference.fractional_centers[i],
                                         reference.fractional_extents[i]};
        for (int field = 0; field < 3; ++field)
        {
            AccumulateDifference(actual_values[field].x,
                                 reference_values[field].x, &max_abs,
                                 &max_scaled);
            AccumulateDifference(actual_values[field].y,
                                 reference_values[field].y, &max_abs,
                                 &max_scaled);
            AccumulateDifference(actual_values[field].z,
                                 reference_values[field].z, &max_abs,
                                 &max_scaled);
        }
    }
    for (size_t i = 0; i < reference.xq.size(); ++i)
    {
        integer_mismatches += actual.atom_ids[i] != reference.atom_ids[i];
        integer_mismatches += actual.lj_type[i] != reference.lj_type[i];
        AccumulateDifference(actual.xq[i].x, reference.xq[i].x, &max_abs,
                             &max_scaled);
        AccumulateDifference(actual.xq[i].y, reference.xq[i].y, &max_abs,
                             &max_scaled);
        AccumulateDifference(actual.xq[i].z, reference.xq[i].z, &max_abs,
                             &max_scaled);
        AccumulateDifference(actual.xq[i].w, reference.xq[i].w, &max_abs,
                             &max_scaled);
        if (use_comb)
        {
            AccumulateDifference(actual.lj_comb[i].x,
                                 reference.lj_comb[i].x, &max_abs,
                                 &max_scaled);
            AccumulateDifference(actual.lj_comb[i].y,
                                 reference.lj_comb[i].y, &max_abs,
                                 &max_scaled);
        }
    }
    const bool matched = integer_mismatches == 0 && max_abs <= 1.0e-4 &&
                         max_scaled <= 2.0e-5;
    std::printf(
        "gather_oracle mode=%s matched=%d integer_mismatches=%zu "
        "max_abs=%.9g max_scaled=%.9g\n",
        label, matched ? 1 : 0, integer_mismatches, max_abs, max_scaled);
    return matched;
}

void FreeDeviceState(DeviceState* device)
{
    CheckCuda(cudaFree(device->permutation), "cudaFree(permutation)");
    CheckCuda(cudaFree(device->cluster_offsets),
              "cudaFree(cluster_offsets)");
    CheckCuda(cudaFree(device->src), "cudaFree(src)");
    CheckCuda(cudaFree(device->lj_ab), "cudaFree(lj_ab)");
    CheckCuda(cudaFree(device->centers), "cudaFree(centers)");
    CheckCuda(cudaFree(device->fractional_centers),
              "cudaFree(fractional_centers)");
    CheckCuda(cudaFree(device->fractional_extents),
              "cudaFree(fractional_extents)");
    CheckCuda(cudaFree(device->atom_ids), "cudaFree(atom_ids)");
    CheckCuda(cudaFree(device->xq), "cudaFree(xq)");
    CheckCuda(cudaFree(device->lj_type), "cudaFree(lj_type)");
    CheckCuda(cudaFree(device->lj_comb), "cudaFree(lj_comb)");
    *device = {};
}

}  // namespace

int RunGatherExperiment(const SpongeGmxpackedForceOnlySnapshot& snapshot,
                        int warmup, int iters, int block_size)
{
    const int atom_count = static_cast<int>(snapshot.header.total_atom_numbers);
    const int cluster_count = static_cast<int>(snapshot.header.cluster_numbers);
    if (atom_count <= 0 || cluster_count <= 0 || warmup < 0 || iters <= 0 ||
        block_size <= 0 || block_size > 1024 || block_size % 32 != 0 ||
        snapshot.sorted_atom_ids.size() != static_cast<size_t>(atom_count) ||
        snapshot.sorted_xq.size() != static_cast<size_t>(atom_count) ||
        snapshot.sorted_lj_type.size() != static_cast<size_t>(atom_count) ||
        (snapshot.cluster_offsets.size() != static_cast<size_t>(cluster_count) &&
         snapshot.cluster_offsets.size() !=
             static_cast<size_t>(cluster_count + 1)))
    {
        std::fprintf(
            stderr,
            "gather experiment input is incomplete or invalid: atoms=%d "
            "clusters=%d warmup=%d iters=%d block=%d ids=%zu xq=%zu "
            "types=%zu offsets=%zu expected_offsets=%d\n",
            atom_count, cluster_count, warmup, iters, block_size,
            snapshot.sorted_atom_ids.size(), snapshot.sorted_xq.size(),
            snapshot.sorted_lj_type.size(), snapshot.cluster_offsets.size(),
            cluster_count + 1);
        return 1;
    }

    std::vector<int> cluster_offsets = snapshot.cluster_offsets;
    if (cluster_offsets.size() == static_cast<size_t>(cluster_count))
    {
        cluster_offsets.push_back(atom_count);
    }
    bool valid_offsets = cluster_offsets.front() == 0 &&
                         cluster_offsets.back() == atom_count;
    for (int cluster_i = 0; cluster_i < cluster_count && valid_offsets;
         ++cluster_i)
    {
        const int start = cluster_offsets[cluster_i];
        const int end = cluster_offsets[cluster_i + 1];
        valid_offsets = start >= 0 && start <= end && end <= atom_count &&
                        end - start <= kClusterSize;
    }
    if (!valid_offsets)
    {
        std::fprintf(stderr,
                     "gather experiment cluster offsets are invalid or a "
                     "cluster exceeds %d atoms\n",
                     kClusterSize);
        return 1;
    }

    // The force-only snapshot contains the published sorted payload rather
    // than the original coordinate source. Scatter it back through the stored
    // permutation to make one deterministic input for the old two-kernel
    // reference and the fused candidate. Production A/B remains the end-to-end
    // correctness gate.
    std::vector<VectorLj> raw(static_cast<size_t>(atom_count));
    std::vector<unsigned char> seen(static_cast<size_t>(atom_count), 0);
    int max_lj_type = -1;
    for (int sorted_i = 0; sorted_i < atom_count; ++sorted_i)
    {
        const int atom_i = snapshot.sorted_atom_ids[sorted_i];
        if (atom_i < 0 || atom_i >= atom_count || seen[atom_i] != 0)
        {
            std::fprintf(stderr,
                         "gather experiment permutation is not bijective at "
                         "sorted_i=%d atom_i=%d\n",
                         sorted_i, atom_i);
            return 1;
        }
        seen[atom_i] = 1;
        const Float4POD xq = snapshot.sorted_xq[sorted_i];
        const int lj_type = snapshot.sorted_lj_type[sorted_i];
        if (lj_type < 0)
        {
            std::fprintf(stderr,
                         "gather experiment LJ type is negative at "
                         "sorted_i=%d\n",
                         sorted_i);
            return 1;
        }
        max_lj_type = std::max(max_lj_type, lj_type);
        raw[atom_i] = {{xq.x, xq.y, xq.z},
                       lj_type, xq.w};
    }

    const Matrix3 cell = MakeMatrix(snapshot.header.cell);
    if (!std::isfinite(cell.a11) || !std::isfinite(cell.a22) ||
        !std::isfinite(cell.a33) || std::fabs(cell.a11) <= 1.0e-12f ||
        std::fabs(cell.a22) <= 1.0e-12f ||
        std::fabs(cell.a33) <= 1.0e-12f)
    {
        std::fprintf(stderr, "gather experiment cell is invalid\n");
        return 1;
    }
    const Matrix3 reciprocal_cell = Invert(cell);
    const bool use_comb = snapshot.header.use_lj_comb != 0u;
    const size_t required_lj_ab =
        static_cast<size_t>(GetLjType(max_lj_type, max_lj_type)) + 1;
    if (use_comb && snapshot.lj_ab.size() < required_lj_ab)
    {
        std::fprintf(stderr,
                     "gather experiment LJ table is incomplete: have=%zu "
                     "need=%zu\n",
                     snapshot.lj_ab.size(), required_lj_ab);
        return 1;
    }

    DeviceState device;
    device.permutation = CopyToDevice(snapshot.sorted_atom_ids);
    device.cluster_offsets = CopyToDevice(cluster_offsets);
    device.src = CopyToDevice(raw);
    if (use_comb)
    {
        device.lj_ab = CopyToDevice(snapshot.lj_ab);
    }
    device.centers = AllocateDevice<Vec3>(cluster_count);
    device.fractional_centers = AllocateDevice<Vec3>(cluster_count);
    device.fractional_extents = AllocateDevice<Vec3>(cluster_count);
    device.atom_ids = AllocateDevice<int>(atom_count);
    device.xq = AllocateDevice<Float4POD>(atom_count);
    device.lj_type = AllocateDevice<int>(atom_count);
    if (use_comb)
    {
        device.lj_comb = AllocateDevice<Float2POD>(atom_count);
    }

    const GatherOutputs reference = CaptureGather(
        GatherMode::reference, device, atom_count, cluster_count, cell,
        reciprocal_cell, block_size, use_comb);
    const GatherOutputs fused = CaptureGather(
        GatherMode::fused, device, atom_count, cluster_count, cell,
        reciprocal_cell, block_size, use_comb);
    const bool fused_ok =
        CompareGatherOutputs(fused, reference, "fused", use_comb);

    const double reference_ms = TimeGather(
        GatherMode::reference, device, atom_count, cluster_count, cell,
        reciprocal_cell, block_size, use_comb, warmup, iters);
    const double fused_ms = TimeGather(
        GatherMode::fused, device, atom_count, cluster_count, cell,
        reciprocal_cell, block_size, use_comb, warmup, iters);
    std::printf(
        "gather_experiment mode=reference block=%d avg_ms=%.9f "
        "speedup=1.000000\n",
        kProductionBlockSize, reference_ms);
    std::printf(
        "gather_experiment mode=fused block=%d avg_ms=%.9f "
        "speedup=%.6f\n",
        block_size, fused_ms, reference_ms / fused_ms);

    FreeDeviceState(&device);
    return fused_ok ? 0 : 1;
}

}  // namespace nbnxm_microbench
