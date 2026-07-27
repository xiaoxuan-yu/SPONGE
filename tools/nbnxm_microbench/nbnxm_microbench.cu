#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Lennard_Jones_force/clustered_lj_count_experiments.h"
#include "canonical_pair_oracle.h"
#include "nbnxm_microbench_snapshot.h"

int CONTROLLER::MPI_rank = 0;

void RunGromacsProduction(
    const nbnxm_microbench::GromacsPairlistSnapshot& snapshot,
    int warmup, int iters, const char* snapshotLabel);

namespace
{

using nbnxm_microbench::Float2POD;
using nbnxm_microbench::Float4POD;
using nbnxm_microbench::GromacsCjPackedPOD;
using nbnxm_microbench::GromacsExclPOD;
using nbnxm_microbench::GromacsPairlistSnapshot;
using nbnxm_microbench::GromacsSciPOD;
using nbnxm_microbench::LTMatrix3POD;
using nbnxm_microbench::SpongeClusteredFullOutputSnapshot;
using nbnxm_microbench::SpongeForceOnlySnapshot;
using nbnxm_microbench::SpongeGmxpackedCjPOD;
using nbnxm_microbench::SpongeGmxpackedExclusionPOD;
using nbnxm_microbench::SpongeGmxpackedForceOnlySnapshot;
using nbnxm_microbench::SpongeGmxpackedFullOutputSnapshot;
using nbnxm_microbench::SpongeGmxpackedSciPOD;
using nbnxm_microbench::SpongeGmxpackedSplitPOD;
using nbnxm_microbench::SpongeSciPOD;
using nbnxm_microbench::SpongeWarpJRecordPOD;

constexpr unsigned int kFullMask = 0xffffffffu;
constexpr int kClusterSize = 8;
constexpr int kSuperClusterClusters = 8;
constexpr int kWarpSplitCount = 2;
constexpr int kSplitJClusterSize = kClusterSize / kWarpSplitCount;
constexpr int kJGroupSize = kSplitJClusterSize;
constexpr int kShiftCount = 27;
constexpr int kCentralShiftId = 13;
constexpr int kPairShiftBits = 5;
constexpr uint64_t kPairShiftMask = (1ull << kPairShiftBits) - 1ull;
constexpr float kTwoDividedBySqrtPi = 1.1283791670218446f;
constexpr float kOracleCutoffGuard =
    4.0f * std::numeric_limits<float>::epsilon();

struct Vec3
{
    float x;
    float y;
    float z;
};

struct DiffStats
{
    double max_abs = 0.0;
    double max_scaled = 0.0;
    double rms = 0.0;
};

DiffStats CompareFloatArrays(const std::vector<float>& actual,
                             const std::vector<float>& reference);
DiffStats CompareForceArrays(const std::vector<Float4POD>& actual,
                             const std::vector<Float4POD>& reference);

enum class SpongeGmxTransformMode
{
    baseline,
    shiftMajorSfc,
    allBlockSuperJ,
    allCjPackedBlockSuperJ,
    centralBlockSuperJ,
    centralCjPackedBlockSuperJ,
    centralBlockSuperJClusterJ,
    allBlockSuperJClusterJ,
    centralTailBlockSuperJ256,
    centralTailBlockSuperJ320,
    centralBlockSuperJTailGap16_256,
    centralBlockSuperJTailGap16_320,
    splitSuperJ,
    splitSuperJGap16,
    centralSplitSuperJGap16,
    centralOnlySplitSuperJ,
    centralOnlySplitSuperJGap16,
    centralTailGapOnly16_192,
    centralTailSplitSuperJGap16_256,
    centralTailSplitSuperJGap16_384,
    centralTailGapOnly16_256,
    centralTailBackwardOnly256,
    centralTailBackwardOrGap32_256,
    centralTailGapOnly16_320,
    centralTailGapOnly16_384,
    centralTailGapOnly16_256Max1,
    centralTailGapOnly16_256Total320,
    centralTailGapOnly16_256Total384,
    centralTailGapOnly16_256Total448,
    centralRunAware256,
    centralRunAware320,
    centralRunAware384,
    centralRunAware512,
    centralRunAware640,
    centralJointSuperJClusterJ320,
    centralJointSuperJClusterJ384,
    centralJointSuperJClusterJ448,
    geometryShiftBucket,
    referenceZone,
    pairShiftBucket,
    padEmptyCj,
    exactCutoffImask,
};

const char* SpongeGmxTransformName(SpongeGmxTransformMode mode);

bool IsCentralJointSuperJClusterJLayout(SpongeGmxTransformMode mode)
{
    return mode == SpongeGmxTransformMode::centralJointSuperJClusterJ320 ||
           mode == SpongeGmxTransformMode::centralJointSuperJClusterJ384 ||
           mode == SpongeGmxTransformMode::centralJointSuperJClusterJ448;
}

size_t CentralJointTargetRecords(SpongeGmxTransformMode mode)
{
    switch (mode)
    {
        case SpongeGmxTransformMode::centralJointSuperJClusterJ320:
            return 320;
        case SpongeGmxTransformMode::centralJointSuperJClusterJ384:
            return 384;
        case SpongeGmxTransformMode::centralJointSuperJClusterJ448:
            return 448;
        default:
            return static_cast<size_t>(-1);
    }                                                                    \
}

struct LTMatrix3
{
    float a11;
    float a21;
    float a22;
    float a31;
    float a32;
    float a33;
};

DiffStats CompareVirialArrays(const std::vector<LTMatrix3>& actual,
                              const std::vector<LTMatrix3POD>& reference);

struct VectorLj
{
    Vec3 crd;
    int lj_type;
    float charge;
};

struct SnapshotRecordBuilderStats
{
    size_t sourceRecords = 0;
    size_t aggregateRows = 0;
    size_t compactSci = 0;
    size_t compactCjPacked = 0;
    size_t compactExcl = 0;
};

void ClearGromacsPairlistSnapshotPreserveCapacity(
    GromacsPairlistSnapshot* snapshot)
{
    snapshot->header = {};
    snapshot->cluster_offsets.clear();
    snapshot->cluster_valid_masks.clear();
    snapshot->cluster_local_masks.clear();
    snapshot->sci.clear();
    snapshot->cjpacked.clear();
    snapshot->excl.clear();
    snapshot->sorted_atom_ids.clear();
    snapshot->sorted_xq.clear();
    snapshot->sorted_lj_type.clear();
    snapshot->sorted_lj_comb.clear();
    snapshot->lj_ab.clear();
}

template <typename T>
inline void CheckCuda(T code, const char* what)
{
    if (code != cudaSuccess)
    {
        std::fprintf(stderr, "%s failed: %s\n", what,
                     cudaGetErrorString(code));
        std::exit(1);
    }                                                                    \
}

static inline LTMatrix3 MakeMatrix(const LTMatrix3POD& pod)
{
    return {pod.a11, pod.a21, pod.a22, pod.a31, pod.a32, pod.a33};
}

static inline float4 MakeFloat4(const Float4POD& pod)
{
    return make_float4(pod.x, pod.y, pod.z, pod.w);
}

static inline float2 MakeFloat2(const Float2POD& pod)
{
    return make_float2(pod.x, pod.y);
}

__host__ __device__ inline Vec3 operator+(Vec3 a, Vec3 b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ inline Vec3 operator-(Vec3 a, Vec3 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ inline Vec3 operator*(float a, Vec3 b)
{
    return {a * b.x, a * b.y, a * b.z};
}

__host__ __device__ inline Vec3 operator/(Vec3 a, float b)
{
    return {a.x / b, a.y / b, a.z / b};
}

__host__ __device__ inline LTMatrix3 operator+(LTMatrix3 a, LTMatrix3 b)
{
    return {a.a11 + b.a11, a.a21 + b.a21, a.a22 + b.a22,
            a.a31 + b.a31, a.a32 + b.a32, a.a33 + b.a33};
}

__host__ __device__ inline LTMatrix3 operator-(LTMatrix3 a, LTMatrix3 b)
{
    return {a.a11 - b.a11, a.a21 - b.a21, a.a22 - b.a22,
            a.a31 - b.a31, a.a32 - b.a32, a.a33 - b.a33};
}

__host__ __device__ inline LTMatrix3 operator*(float scalar, LTMatrix3 value)
{
    return {scalar * value.a11, scalar * value.a21, scalar * value.a22,
            scalar * value.a31, scalar * value.a32, scalar * value.a33};
}

__host__ __device__ __forceinline__ float Dot(Vec3 a, Vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline LTMatrix3 GetVirialFromForceDis(Vec3 veca,
                                                           Vec3 vecb)
{
    return {veca.x * vecb.x,
            veca.x * vecb.y + veca.y * vecb.x,
            veca.y * vecb.y,
            veca.x * vecb.z + veca.z * vecb.x,
            veca.y * vecb.z + veca.z * vecb.y,
            veca.z * vecb.z};
}

__host__ __device__ inline void AccumulateVirialFromForceDis(
    LTMatrix3* target, Vec3 force, Vec3 displacement, float scale)
{
    target->a11 += scale * force.x * displacement.x;
    target->a21 += scale *
                   (force.x * displacement.y + force.y * displacement.x);
    target->a22 += scale * force.y * displacement.y;
    target->a31 += scale *
                   (force.x * displacement.z + force.z * displacement.x);
    target->a32 += scale *
                   (force.y * displacement.z + force.z * displacement.y);
    target->a33 += scale * force.z * displacement.z;
}

__device__ inline void AtomicAddVirial(LTMatrix3* target, LTMatrix3 value)
{
    atomicAdd(&target->a11, value.a11);
    atomicAdd(&target->a21, value.a21);
    atomicAdd(&target->a22, value.a22);
    atomicAdd(&target->a31, value.a31);
    atomicAdd(&target->a32, value.a32);
    atomicAdd(&target->a33, value.a33);
}

__host__ __device__ inline Vec3 operator*(Vec3 vec, LTMatrix3 mat)
{
    return {vec.x * mat.a11 + vec.y * mat.a21 + vec.z * mat.a31,
            vec.y * mat.a22 + vec.z * mat.a32,
            vec.z * mat.a33};
}

inline LTMatrix3 InvertCellMatrix(LTMatrix3 cell)
{
    LTMatrix3 inverse = {};
    inverse.a11 = 1.0f / cell.a11;
    inverse.a22 = 1.0f / cell.a22;
    inverse.a33 = 1.0f / cell.a33;
    inverse.a21 = -cell.a21 / (cell.a11 * cell.a22);
    inverse.a32 = -cell.a32 / (cell.a22 * cell.a33);
    inverse.a31 = (cell.a21 * cell.a32 - cell.a31 * cell.a22) /
                  (cell.a11 * cell.a22 * cell.a33);
    return inverse;
}

inline bool CellLooksUsable(LTMatrix3 cell)
{
    return std::isfinite(cell.a11) && std::isfinite(cell.a22) &&
           std::isfinite(cell.a33) && std::fabs(cell.a11) > 1.0e-12f &&
           std::fabs(cell.a22) > 1.0e-12f &&
           std::fabs(cell.a33) > 1.0e-12f;
}

inline LTMatrix3 MakeOrthorhombicCellFromShiftVec(
    const std::array<Float4POD, 27>& shiftvec)
{
    LTMatrix3 cell = {};
    for (const Float4POD& shift : shiftvec)
    {
        cell.a11 = std::max(cell.a11, std::fabs(shift.x));
        cell.a22 = std::max(cell.a22, std::fabs(shift.y));
        cell.a33 = std::max(cell.a33, std::fabs(shift.z));
    }                                                                    \
    return cell;
}

inline Vec3 MinimumImageDelta(Vec3 delta, LTMatrix3 cell, LTMatrix3 rcell)
{
    Vec3 fractional = delta * rcell;
    fractional.x -= std::nearbyint(fractional.x);
    fractional.y -= std::nearbyint(fractional.y);
    fractional.z -= std::nearbyint(fractional.z);
    return fractional * cell;
}

__host__ __device__ inline int GetLjType(int a, int b)
{
    int y = (b - a);
    int x = y >> 31;
    y = (y ^ x) - x;
    x = b + a;
    int z = (x + y) >> 1;
    x = (x - y) >> 1;
    return (z * (z + 1) >> 1) + x;
}

__host__ __device__ inline VectorLj MakePackedLjAtom(const float4 xq,
                                                     int lj_type)
{
    return {{xq.x, xq.y, xq.z}, lj_type, xq.w};
}

__host__ __device__ inline Vec3 ShiftVectorFromId(int shift_id,
                                                  LTMatrix3 cell)
{
    const int sx = shift_id / 9 - 1;
    const int sy = (shift_id % 9) / 3 - 1;
    const int sz = shift_id % 3 - 1;
    return Vec3{static_cast<float>(sx), static_cast<float>(sy),
                static_cast<float>(sz)} *
           cell;
}

__host__ __device__ inline int GetPairShiftId(uint64_t packed_shift_bits,
                                              int i_local)
{
    return static_cast<int>(
        (packed_shift_bits >> (static_cast<uint64_t>(i_local) * kPairShiftBits)) &
        kPairShiftMask);
}

__host__ __device__ inline Vec3 GetShiftedDisplacement(VectorLj r2,
                                                       VectorLj r1,
                                                       Vec3 shift_vec)
{
    return (r2.crd - r1.crd) - shift_vec;
}

__device__ __forceinline__ float GetLjForceAbs(float inv_r2, float inv_r6,
                                               float A, float B)
{
    return (B - A * inv_r6) * inv_r6 * inv_r2;
}

__device__ inline float GetDirectCoulombForceAbs(float charge_product,
                                                 float inv_r,
                                                 float inv_r2,
                                                 float beta_dr)
{
    return charge_product * inv_r * inv_r2 *
           (beta_dr * kTwoDividedBySqrtPi * expf(-beta_dr * beta_dr) +
            erfcf(beta_dr));
}

__host__ __device__ __forceinline__ float PmeCorrF(float z2)
{
    constexpr float FN6 = -1.7357322914161492954e-8F;
    constexpr float FN5 = 1.4703624142580877519e-6F;
    constexpr float FN4 = -0.000053401640219807709149F;
    constexpr float FN3 = 0.0010054721316683106153F;
    constexpr float FN2 = -0.019278317264888380590F;
    constexpr float FN1 = 0.069670166153766424023F;
    constexpr float FN0 = -0.75225204789749321333F;

    constexpr float FD4 = 0.0011193462567257629232F;
    constexpr float FD3 = 0.014866955030185295499F;
    constexpr float FD2 = 0.11583842382862377919F;
    constexpr float FD1 = 0.50736591960530292870F;
    constexpr float FD0 = 1.0F;

    const float z4 = z2 * z2;

    float polyFD0 = FD4 * z4 + FD2;
    const float polyFD1 = FD3 * z4 + FD1;
    polyFD0 = polyFD0 * z4 + FD0;
    polyFD0 = polyFD1 * z2 + polyFD0;
    polyFD0 = 1.0F / polyFD0;

    float polyFN0 = FN6 * z4 + FN4;
    float polyFN1 = FN5 * z4 + FN3;
    polyFN0 = polyFN0 * z4 + FN2;
    polyFN1 = polyFN1 * z4 + FN1;
    polyFN0 = polyFN0 * z4 + FN0;
    polyFN0 = polyFN1 * z2 + polyFN0;
    return polyFN0 * polyFD0;
}

__device__ __forceinline__ float GetDirectCoulombForceAbsPmeCorrF(
    float charge_product, float inv_r, float inv_r2, float beta2_r2,
    float beta3)
{
    return charge_product *
           (inv_r * inv_r2 + PmeCorrF(beta2_r2) * beta3);
}

__device__ __forceinline__ float GetLjEnergy(float inv_r6, float A, float B)
{
    return (0.083333333f * A * inv_r6 - 0.166666667f * B) * inv_r6;
}

__device__ __forceinline__ float GetDirectCoulombEnergy(float charge_product,
                                                        float inv_r,
                                                        float beta_dr)
{
    return charge_product * erfcf(beta_dr) * inv_r;
}

__device__ inline unsigned int SubgroupMask(int lane, int subgroup_width)
{
    const unsigned int subgroup =
        static_cast<unsigned int>(lane / subgroup_width);
    const unsigned int width_mask =
        (1u << static_cast<unsigned int>(subgroup_width)) - 1u;
    return width_mask << (subgroup * static_cast<unsigned int>(subgroup_width));
}

__device__ inline int BroadcastSubgroupInt(int value, int lane,
                                           int subgroup_width)
{
    return __shfl_sync(SubgroupMask(lane, subgroup_width), value,
                       (lane / subgroup_width) * subgroup_width,
                       subgroup_width);
}

__device__ inline float4 BroadcastSubgroupFloat4(float4 value, int lane,
                                                 int subgroup_width)
{
    const unsigned int mask = SubgroupMask(lane, subgroup_width);
    const int src_lane = (lane / subgroup_width) * subgroup_width;
    value.x = __shfl_sync(mask, value.x, src_lane, subgroup_width);
    value.y = __shfl_sync(mask, value.y, src_lane, subgroup_width);
    value.z = __shfl_sync(mask, value.z, src_lane, subgroup_width);
    value.w = __shfl_sync(mask, value.w, src_lane, subgroup_width);
    return value;
}

__device__ inline float2 BroadcastSubgroupFloat2(float2 value, int lane,
                                                 int subgroup_width)
{
    const unsigned int mask = SubgroupMask(lane, subgroup_width);
    const int src_lane = (lane / subgroup_width) * subgroup_width;
    value.x = __shfl_sync(mask, value.x, src_lane, subgroup_width);
    value.y = __shfl_sync(mask, value.y, src_lane, subgroup_width);
    return value;
}

__device__ inline uint64_t BroadcastSubgroupU64(uint64_t value, int lane,
                                                int subgroup_width)
{
    const unsigned int mask = SubgroupMask(lane, subgroup_width);
    const int src_lane = (lane / subgroup_width) * subgroup_width;
    unsigned int lo = static_cast<unsigned int>(value & 0xffffffffu);
    unsigned int hi = static_cast<unsigned int>(value >> 32);
    lo = __shfl_sync(mask, lo, src_lane, subgroup_width);
    hi = __shfl_sync(mask, hi, src_lane, subgroup_width);
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

__device__ inline uint64_t BroadcastWarpU64(uint64_t value, int src_lane)
{
    unsigned int lo = static_cast<unsigned int>(value & 0xffffffffu);
    unsigned int hi = static_cast<unsigned int>(value >> 32);
    lo = __shfl_sync(kFullMask, lo, src_lane, warpSize);
    hi = __shfl_sync(kFullMask, hi, src_lane, warpSize);
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

__device__ __forceinline__ unsigned int SubgroupMask8(int lane)
{
    return 0xffu << static_cast<unsigned int>(lane & ~0x7);
}

__device__ __forceinline__ int BroadcastSubgroupInt8(int value, int lane)
{
    return __shfl_sync(SubgroupMask8(lane), value, lane & ~0x7, kClusterSize);
}

__device__ __forceinline__ uint64_t BroadcastSubgroupU648(uint64_t value,
                                                          int lane)
{
    unsigned int lo = static_cast<unsigned int>(value & 0xffffffffu);
    unsigned int hi = static_cast<unsigned int>(value >> 32);
    const unsigned int mask = SubgroupMask8(lane);
    const int src_lane = lane & ~0x7;
    lo = __shfl_sync(mask, lo, src_lane, kClusterSize);
    hi = __shfl_sync(mask, hi, src_lane, kClusterSize);
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

template <typename T>
__device__ __forceinline__ T LoadReadOnly(const T* ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

Float2POD MakeLjAB(float c12, float c6)
{
    Float2POD ab = {};
    ab.x = c12;
    ab.y = c6;
    return ab;
}

uint64_t MakeUniformPairShiftWord(int shiftId)
{
    uint64_t packed = 0;
    for (int i = 0; i < kSuperClusterClusters; ++i)
    {
        packed |= (static_cast<uint64_t>(shiftId) & kPairShiftMask)
                  << (static_cast<uint64_t>(i) * kPairShiftBits);
    }                                                                    \
    return packed;
}

std::vector<Float2POD> BuildTriangularLjTableFromGromacsComb(
    const GromacsPairlistSnapshot& snapshot)
{
    int numTypes = 0;
    for (int type : snapshot.sorted_lj_type)
    {
        numTypes = std::max(numTypes, type + 1);
    }                                                                    \
    std::vector<Float2POD> perType(static_cast<size_t>(numTypes));
    std::vector<unsigned char> seen(static_cast<size_t>(numTypes), 0u);
    for (size_t i = 0; i < snapshot.sorted_lj_type.size(); ++i)
    {
        const int type = snapshot.sorted_lj_type[i];
        if (type >= 0 && type < numTypes && !seen[static_cast<size_t>(type)])
        {
            perType[static_cast<size_t>(type)] = snapshot.sorted_lj_comb[i];
            seen[static_cast<size_t>(type)] = 1u;
        }
    }                                                                    \
    std::vector<Float2POD> ljAb(
        static_cast<size_t>(numTypes * (numTypes + 1) / 2));
    for (int i = 0; i < numTypes; ++i)
    {
        for (int j = 0; j < numTypes; ++j)
        {
            const int pairType = GetLjType(i, j);
            const float c6 =
                perType[static_cast<size_t>(i)].x *
                perType[static_cast<size_t>(j)].x;
            const float c12 =
                perType[static_cast<size_t>(i)].y *
                perType[static_cast<size_t>(j)].y;
            ljAb[static_cast<size_t>(pairType)] = MakeLjAB(c12, c6);
        }
    }
    return ljAb;
}

std::vector<Float2POD> BuildTriangularLjTableFromGromacsFull(
    const GromacsPairlistSnapshot& snapshot)
{
    int numTypes = static_cast<int>(snapshot.header.num_types);
    if (numTypes <= 0)
    {
        for (int type : snapshot.sorted_lj_type)
        {
            numTypes = std::max(numTypes, type + 1);
        }
    }
    std::vector<Float2POD> ljAb(
        static_cast<size_t>(numTypes * (numTypes + 1) / 2));
    for (int i = 0; i < numTypes; ++i)
    {
        for (int j = 0; j < numTypes; ++j)
        {
            const int pairType = GetLjType(i, j);
            ljAb[static_cast<size_t>(pairType)] =
                snapshot.lj_ab[static_cast<size_t>(i * numTypes + j)];
        }
    }
    return ljAb;
}

std::vector<Float2POD> BuildSortedLjCombFromSpongeSnapshot(
    const SpongeForceOnlySnapshot& snapshot)
{
    int numTypes = 0;
    for (int type : snapshot.sorted_lj_type)
    {
        numTypes = std::max(numTypes, type + 1);
    }

    std::vector<Float2POD> perType(static_cast<size_t>(numTypes));
    for (int type = 0; type < numTypes; ++type)
    {
        const Float2POD selfAb =
            snapshot.lj_ab[static_cast<size_t>(GetLjType(type, type))];
        Float2POD comb = {};
        comb.x = std::sqrt(std::max(selfAb.y, 0.0f));
        comb.y = std::sqrt(std::max(selfAb.x, 0.0f));
        perType[static_cast<size_t>(type)] = comb;
    }

    std::vector<Float2POD> sortedLjComb(snapshot.sorted_lj_type.size());
    for (size_t i = 0; i < snapshot.sorted_lj_type.size(); ++i)
    {
        const int type = snapshot.sorted_lj_type[i];
        if (type >= 0 && type < numTypes)
        {
            sortedLjComb[i] = perType[static_cast<size_t>(type)];
        }
    }
    return sortedLjComb;
}

void MaterializeImplicitGromacsClusters(
    const GromacsPairlistSnapshot& snapshot, std::vector<int>* clusterOffsets,
    std::vector<unsigned int>* clusterValidMasks,
    std::vector<unsigned int>* clusterLocalMasks)
{
    if (!snapshot.cluster_offsets.empty() &&
        !snapshot.cluster_valid_masks.empty() &&
        !snapshot.cluster_local_masks.empty())
    {
        *clusterOffsets = snapshot.cluster_offsets;
        *clusterValidMasks = snapshot.cluster_valid_masks;
        *clusterLocalMasks = snapshot.cluster_local_masks;
        return;
    }

    const size_t totalAtoms = snapshot.sorted_xq.size();
    const size_t clusterCount =
        (totalAtoms + kClusterSize - 1) / static_cast<size_t>(kClusterSize);
    clusterOffsets->resize(clusterCount);
    clusterValidMasks->resize(clusterCount);
    clusterLocalMasks->resize(clusterCount);
    for (size_t cluster = 0; cluster < clusterCount; ++cluster)
    {
        (*clusterOffsets)[cluster] =
            static_cast<int>(cluster * static_cast<size_t>(kClusterSize));
        const size_t atomsRemaining =
            totalAtoms - std::min(totalAtoms,
                                  cluster * static_cast<size_t>(kClusterSize));
        const size_t atomsInCluster =
            std::min(static_cast<size_t>(kClusterSize), atomsRemaining);
        const unsigned int validMask =
            atomsInCluster == static_cast<size_t>(kClusterSize)
                ? ((1u << kClusterSize) - 1u)
                : ((1u << static_cast<unsigned int>(atomsInCluster)) - 1u);
        (*clusterValidMasks)[cluster] = validMask;
        (*clusterLocalMasks)[cluster] = validMask;
    }
}

SpongeForceOnlySnapshot ConvertGromacsSnapshotToSponge(
    const GromacsPairlistSnapshot& snapshot)
{
    std::vector<int> clusterOffsets;
    std::vector<unsigned int> clusterValidMasks;
    std::vector<unsigned int> clusterLocalMasks;
    MaterializeImplicitGromacsClusters(snapshot, &clusterOffsets,
                                       &clusterValidMasks,
                                       &clusterLocalMasks);

    SpongeForceOnlySnapshot converted = {};
    converted.header.file = nbnxm_microbench::MakeFileHeader(
        nbnxm_microbench::SnapshotKind::spongeForceOnly);
    converted.header.cluster_size = snapshot.header.cluster_size;
    converted.header.super_cluster_clusters =
        snapshot.header.super_cluster_clusters;
    converted.header.warp_split_count = snapshot.header.cluster_pair_split;
    converted.header.cluster_numbers = clusterOffsets.size();
    converted.header.super_cluster_numbers =
        (converted.header.cluster_numbers + kSuperClusterClusters - 1) /
        kSuperClusterClusters;
    converted.header.sci_numbers = snapshot.sci.size();
    converted.header.total_atom_numbers = snapshot.sorted_xq.size();
    converted.header.local_atom_numbers = snapshot.header.local_atom_numbers;
    converted.header.cutoff = snapshot.header.cutoff;
    converted.header.pme_beta = snapshot.header.pme_beta;
    converted.header.cell = snapshot.header.cell;

    converted.cluster_offsets = std::move(clusterOffsets);
    converted.cluster_valid_masks = std::move(clusterValidMasks);
    converted.cluster_local_masks = std::move(clusterLocalMasks);
    converted.super_cluster_offsets.resize(
        static_cast<size_t>(converted.header.super_cluster_numbers + 1));
    for (size_t i = 0; i < converted.super_cluster_offsets.size(); ++i)
    {
        converted.super_cluster_offsets[i] = std::min(
            static_cast<int>(i * static_cast<size_t>(kSuperClusterClusters)),
            static_cast<int>(converted.cluster_offsets.size()));
    }
    converted.sci.resize(snapshot.sci.size());
    converted.record_offsets.resize(snapshot.sci.size() + 1);
    converted.pair_shift_bits.resize(27);
    for (int shift = 0; shift < static_cast<int>(converted.pair_shift_bits.size());
         ++shift)
    {
        converted.pair_shift_bits[static_cast<size_t>(shift)] =
            MakeUniformPairShiftWord(shift);
    }
    std::array<int, kShiftCount> gromacsToSpongeShift = {};
    for (int gmxShift = 0; gmxShift < kShiftCount; ++gmxShift)
    {
        gromacsToSpongeShift[static_cast<size_t>(gmxShift)] = gmxShift;
    }
    int gromacsZeroShift = kCentralShiftId;
    float bestShiftNorm2 = std::numeric_limits<float>::infinity();
    for (int shift = 0; shift < kShiftCount; ++shift)
    {
        const Float4POD& shiftVec =
            snapshot.header.shiftvec[static_cast<size_t>(shift)];
        const float norm2 =
            shiftVec.x * shiftVec.x + shiftVec.y * shiftVec.y +
            shiftVec.z * shiftVec.z;
        if (norm2 < bestShiftNorm2)
        {
            bestShiftNorm2 = norm2;
            gromacsZeroShift = shift;
        }
    }
    if (gromacsZeroShift >= 0 && gromacsZeroShift < kShiftCount &&
        gromacsZeroShift != kCentralShiftId)
    {
        std::swap(gromacsToSpongeShift[static_cast<size_t>(gromacsZeroShift)],
                  gromacsToSpongeShift[static_cast<size_t>(kCentralShiftId)]);
    }
    if (!snapshot.sorted_atom_ids.empty())
    {
        converted.sorted_atom_ids = snapshot.sorted_atom_ids;
    }
    else
    {
        converted.sorted_atom_ids.resize(snapshot.sorted_xq.size());
        std::iota(converted.sorted_atom_ids.begin(),
                  converted.sorted_atom_ids.end(), 0);
    }
    converted.sorted_xq = snapshot.sorted_xq;
    converted.sorted_lj_type = snapshot.sorted_lj_type;
    converted.lj_ab = snapshot.lj_ab.empty()
                              ? BuildTriangularLjTableFromGromacsComb(snapshot)
                              : BuildTriangularLjTableFromGromacsFull(snapshot);

    std::vector<SpongeWarpJRecordPOD> records;
    records.reserve(snapshot.cjpacked.size() * kWarpSplitCount *
                    kJGroupSize);
    for (size_t sciIdx = 0; sciIdx < snapshot.sci.size(); ++sciIdx)
    {
        const auto& sci = snapshot.sci[sciIdx];
        const int spongeShift =
            (sci.shift >= 0 && sci.shift < kShiftCount)
                ? gromacsToSpongeShift[static_cast<size_t>(sci.shift)]
                : sci.shift;
        converted.sci[sciIdx] = {
            sci.sci,
            spongeShift,
            0,
            0,
        };
        converted.record_offsets[sciIdx] = static_cast<int>(records.size());
        for (int jPacked = sci.cjPackedBegin; jPacked < sci.cjPackedEnd;
             ++jPacked)
        {
            const auto& packed = snapshot.cjpacked[static_cast<size_t>(jPacked)];
            for (int split = 0; split < kWarpSplitCount; ++split)
            {
                const unsigned int imask32 = packed.imei[split].imask;
                const int exclIndex = packed.imei[split].excl_ind;
                for (int jm = 0; jm < kJGroupSize; ++jm)
                {
                    const unsigned int imask8 =
                        (imask32 >> (jm * kSuperClusterClusters)) & 0xffu;
                    const int clusterJ = packed.cj[jm];
                    if (clusterJ < 0 || imask8 == 0)
                    {
                        continue;
                    }
                    SpongeWarpJRecordPOD record = {};
                    record.cluster_j = clusterJ;
                    record.sorted_j_base =
                        converted.cluster_offsets[static_cast<size_t>(clusterJ)] +
                        split * kSplitJClusterSize;
                    record.pair_shift_index = spongeShift;
                    record.valid_mask = static_cast<unsigned char>(
                        (converted.cluster_valid_masks[static_cast<size_t>(clusterJ)] >>
                         (split * kSplitJClusterSize)) &
                        ((1u << kSplitJClusterSize) - 1u));
                    record.imask = static_cast<unsigned char>(imask8);
                    record.local_mask = static_cast<unsigned char>(
                        (converted.cluster_local_masks[static_cast<size_t>(clusterJ)] >>
                         (split * kSplitJClusterSize)) &
                        ((1u << kSplitJClusterSize) - 1u));
                    record.j_lane_base = static_cast<unsigned char>(
                        split * kSplitJClusterSize);
                    if (exclIndex >= 0 &&
                        static_cast<size_t>(exclIndex) < snapshot.excl.size())
                    {
                        const auto& excl =
                            snapshot.excl[static_cast<size_t>(exclIndex)];
                        for (int jLocal = 0; jLocal < kSplitJClusterSize; ++jLocal)
                        {
                            for (int iLane = 0; iLane < kClusterSize; ++iLane)
                            {
                                const unsigned int wexcl =
                                    excl.pair[static_cast<size_t>(
                                        jLocal * kClusterSize + iLane)];
                                unsigned char pairExclMask = 0u;
                                for (int iLocal = 0; iLocal < kSuperClusterClusters;
                                     ++iLocal)
                                {
                                    const unsigned int imaskBit =
                                        1u << static_cast<unsigned int>(iLocal);
                                    const unsigned int pairBit =
                                        1u << static_cast<unsigned int>(
                                                  jm * kSuperClusterClusters +
                                                  iLocal);
                                    if ((imask8 & imaskBit) != 0u &&
                                        (wexcl & pairBit) == 0u)
                                    {
                                        pairExclMask |= static_cast<unsigned char>(
                                            imaskBit);
                                    }
                                }
                                record.pair_excl[static_cast<size_t>(
                                    jLocal * kClusterSize + iLane)] =
                                    pairExclMask;
                            }
                        }
                    }
                    records.push_back(record);
                }
            }
        }
    }
    converted.record_offsets[snapshot.sci.size()] =
        static_cast<int>(records.size());
    converted.records = std::move(records);
    converted.header.record_numbers = converted.records.size();
    converted.header.pair_shift_word_numbers =
        converted.pair_shift_bits.size();
    converted.header.lj_param_numbers = converted.lj_ab.size();
    return converted;
}

GromacsPairlistSnapshot ConvertSpongeSnapshotToGromacs(
    const SpongeForceOnlySnapshot& snapshot,
    SpongeGmxTransformMode transformMode,
    const GromacsPairlistSnapshot* referenceSnapshot = nullptr,
    double exactImaskRadiusScale = 1.0,
    SnapshotRecordBuilderStats* builderStats = nullptr,
    GromacsPairlistSnapshot* builderWorkspace = nullptr)
{
    GromacsPairlistSnapshot localConverted = {};
    GromacsPairlistSnapshot& converted =
        builderWorkspace != nullptr ? *builderWorkspace : localConverted;
    if (builderWorkspace != nullptr)
    {
        ClearGromacsPairlistSnapshotPreserveCapacity(builderWorkspace);
    }
    const bool builderStatsOnly = builderWorkspace != nullptr &&
                                  transformMode == SpongeGmxTransformMode::baseline &&
                                  referenceSnapshot == nullptr &&
                                  exactImaskRadiusScale == 1.0;
    const size_t superClusterCount = snapshot.super_cluster_offsets.size() - 1;
    if (builderStats != nullptr)
    {
        *builderStats = {};
        builderStats->sourceRecords = snapshot.records.size();
    }
    converted.header.file = nbnxm_microbench::MakeFileHeader(
        nbnxm_microbench::SnapshotKind::gromacsPairlist);
    converted.header.cluster_size = snapshot.header.cluster_size;
    converted.header.super_cluster_clusters =
        snapshot.header.super_cluster_clusters;
    converted.header.cluster_pair_split = snapshot.header.warp_split_count;
    converted.header.j_group_size = kJGroupSize;
    converted.header.elec_type = 4u;
    // Match the prepared GROMACS water snapshot path, which uses the
    // LJ-combination kernel and per-atom LJ-comb parameters.
    converted.header.vdw_type = 1u;
    converted.header.num_threads_z = 1u;
    converted.header.compute_energy = 0u;
    converted.header.compute_virial = 0u;
    converted.header.use_prune_kernel = 0u;
    converted.header.cluster_numbers =
        superClusterCount * static_cast<size_t>(kSuperClusterClusters);
    converted.header.sci_numbers = 0;
    converted.header.total_atom_numbers =
        converted.header.cluster_numbers * static_cast<size_t>(kClusterSize);
    converted.header.local_atom_numbers = converted.header.total_atom_numbers;
    converted.header.cutoff = snapshot.header.cutoff;
    converted.header.pme_beta = snapshot.header.pme_beta;
    converted.header.epsfac = 1.0f;
    converted.header.cell = snapshot.header.cell;

    std::vector<int> denseClusterIndexForOriginalCluster(
        snapshot.cluster_offsets.size(), -1);
    std::vector<int> superClusterForOriginalCluster(
        snapshot.cluster_offsets.size(), -1);
    std::vector<int> localClusterForOriginalCluster(
        snapshot.cluster_offsets.size(), -1);
    if (!builderStatsOnly)
    {
        converted.cluster_offsets.resize(
            static_cast<size_t>(converted.header.cluster_numbers), 0);
        converted.cluster_valid_masks.resize(
            static_cast<size_t>(converted.header.cluster_numbers), 0u);
        converted.cluster_local_masks.resize(
            static_cast<size_t>(converted.header.cluster_numbers), 0u);
        converted.sorted_atom_ids.resize(
            static_cast<size_t>(converted.header.total_atom_numbers), -1);
        converted.sorted_xq.resize(
            static_cast<size_t>(converted.header.total_atom_numbers));
        converted.sorted_lj_type.resize(
            static_cast<size_t>(converted.header.total_atom_numbers), 0);
        converted.sorted_lj_comb.resize(
            static_cast<size_t>(converted.header.total_atom_numbers), {});
    }
    const std::vector<Float2POD> sortedLjComb = builderStatsOnly
        ? std::vector<Float2POD>{}
        : BuildSortedLjCombFromSpongeSnapshot(snapshot);
    for (size_t superI = 0; superI < superClusterCount; ++superI)
    {
        const int clusterBegin = snapshot.super_cluster_offsets[superI];
        const int clusterEnd = snapshot.super_cluster_offsets[superI + 1];
        for (int iLocal = 0; iLocal < kSuperClusterClusters; ++iLocal)
        {
            const size_t denseIndex =
                superI * static_cast<size_t>(kSuperClusterClusters) +
                static_cast<size_t>(iLocal);
            const int clusterI = clusterBegin + iLocal;
            if (clusterI >= clusterEnd)
            {
                continue;
            }
            denseClusterIndexForOriginalCluster[static_cast<size_t>(clusterI)] =
                static_cast<int>(denseIndex);
            superClusterForOriginalCluster[static_cast<size_t>(clusterI)] =
                static_cast<int>(superI);
            localClusterForOriginalCluster[static_cast<size_t>(clusterI)] =
                iLocal;
            const int dstAtomBase =
                static_cast<int>(denseIndex * static_cast<size_t>(kClusterSize));
            if (builderStatsOnly)
            {
                continue;
            }
            converted.cluster_offsets[denseIndex] = dstAtomBase;
            const unsigned int validMask =
                snapshot.cluster_valid_masks[static_cast<size_t>(clusterI)];
            const unsigned int localMask =
                snapshot.cluster_local_masks[static_cast<size_t>(clusterI)];
            converted.cluster_valid_masks[denseIndex] = validMask;
            converted.cluster_local_masks[denseIndex] = localMask;
            const int srcAtomBase =
                snapshot.cluster_offsets[static_cast<size_t>(clusterI)];
            for (int lane = 0; lane < kClusterSize; ++lane)
            {
                if ((validMask & (1u << static_cast<unsigned int>(lane))) == 0u)
                {
                    continue;
                }
                converted.sorted_xq[static_cast<size_t>(dstAtomBase + lane)] =
                    snapshot.sorted_xq[static_cast<size_t>(srcAtomBase + lane)];
                converted.sorted_atom_ids[static_cast<size_t>(dstAtomBase + lane)] =
                    snapshot.sorted_atom_ids[static_cast<size_t>(srcAtomBase + lane)];
                converted.sorted_lj_type[static_cast<size_t>(dstAtomBase + lane)] =
                    snapshot.sorted_lj_type[static_cast<size_t>(srcAtomBase + lane)];
                converted.sorted_lj_comb[static_cast<size_t>(dstAtomBase + lane)] =
                    sortedLjComb[static_cast<size_t>(srcAtomBase + lane)];
            }
        }
    }

    if (!builderStatsOnly)
    {
        int numTypes = 0;
        for (int type : converted.sorted_lj_type)
        {
            numTypes = std::max(numTypes, type + 1);
        }
        converted.header.num_types = static_cast<uint32_t>(numTypes);
        converted.lj_ab.resize(static_cast<size_t>(numTypes * numTypes));
        for (int i = 0; i < numTypes; ++i)
        {
            for (int j = 0; j < numTypes; ++j)
            {
                converted.lj_ab[static_cast<size_t>(i * numTypes + j)] =
                    snapshot.lj_ab[static_cast<size_t>(GetLjType(i, j))];
            }
        }
        converted.header.lj_param_numbers = converted.lj_ab.size();
    }

    if (!builderStatsOnly)
    {
        for (int shift = 0; shift < 27; ++shift)
        {
            const Vec3 shiftVec = ShiftVectorFromId(shift, MakeMatrix(snapshot.header.cell));
            converted.header.shiftvec[static_cast<size_t>(shift)] = {
                shiftVec.x, shiftVec.y, shiftVec.z, 0.0f};
        }
    }

    converted.sci.clear();
    converted.cjpacked.clear();
    converted.excl.clear();
    converted.cjpacked.reserve(snapshot.records.size() / kJGroupSize + 1);
    converted.excl.reserve(snapshot.records.size() / kJGroupSize + 1);
    GromacsExclPOD noExcl = {};
    for (unsigned int& pair : noExcl.pair)
    {
        pair = 0xffffffffu;
    }
    converted.excl.push_back(noExcl);
    if (transformMode != SpongeGmxTransformMode::baseline)
    {
        std::printf("sponge-gmx-transform=%s\n",
                    SpongeGmxTransformName(transformMode));
    }

    struct ReferenceZoneEntry
    {
        int shift = kCentralShiftId;
        uint64_t order = 0;
    };
    auto makeReferenceKey = [](int superI, int clusterJ) -> uint64_t
    {
        return (static_cast<uint64_t>(static_cast<uint32_t>(superI)) << 32) |
               static_cast<uint32_t>(clusterJ);
    };
    std::unordered_map<uint64_t, ReferenceZoneEntry> referenceZones;
    if (transformMode == SpongeGmxTransformMode::referenceZone)
    {
        if (referenceSnapshot == nullptr)
        {
            std::fprintf(stderr,
                         "reference-zone transform requires "
                         "--reference-gmx-snapshot\n");
            std::exit(1);
        }
        uint64_t referenceOrder = 0;
        for (const GromacsSciPOD& refSci : referenceSnapshot->sci)
        {
            for (int packedIdx = refSci.cjPackedBegin;
                 packedIdx < refSci.cjPackedEnd; ++packedIdx)
            {
                const GromacsCjPackedPOD& packed =
                    referenceSnapshot->cjpacked[static_cast<size_t>(packedIdx)];
                for (int jm = 0; jm < kJGroupSize; ++jm)
                {
                    const int refClusterJ = packed.cj[jm];
                    if (refClusterJ < 0)
                    {
                        continue;
                    }
                    bool active = false;
                    for (int split = 0; split < kWarpSplitCount; ++split)
                    {
                        const unsigned int imask =
                            (packed.imei[split].imask >>
                             (jm * kSuperClusterClusters)) &
                            ((1u << kSuperClusterClusters) - 1u);
                        active = active || imask != 0u;
                    }
                    if (!active)
                    {
                        continue;
                    }
                    const uint64_t key =
                        makeReferenceKey(refSci.sci, refClusterJ);
                    referenceZones.emplace(key,
                                           ReferenceZoneEntry{refSci.shift,
                                                              referenceOrder});
                    ++referenceOrder;
                }
            }
        }
    }

    const bool needsGeometryShiftCenters =
        transformMode == SpongeGmxTransformMode::geometryShiftBucket;
    std::vector<Vec3> clusterCenters;
    std::vector<Vec3> superCenters;
    if (needsGeometryShiftCenters)
    {
        clusterCenters.assign(snapshot.cluster_offsets.size(),
                              Vec3{0.0f, 0.0f, 0.0f});
        std::vector<int> clusterCenterCounts(snapshot.cluster_offsets.size(), 0);
        for (size_t clusterIdx = 0; clusterIdx < snapshot.cluster_offsets.size();
             ++clusterIdx)
        {
            const unsigned int validMask = snapshot.cluster_valid_masks[clusterIdx];
            const int atomBase = snapshot.cluster_offsets[clusterIdx];
            Vec3 sum{0.0f, 0.0f, 0.0f};
            int count = 0;
            for (int lane = 0; lane < kClusterSize; ++lane)
            {
                if ((validMask & (1u << static_cast<unsigned int>(lane))) == 0u)
                {
                    continue;
                }
                const Float4POD& xq =
                    snapshot.sorted_xq[static_cast<size_t>(atomBase + lane)];
                sum = sum + Vec3{xq.x, xq.y, xq.z};
                ++count;
            }
            if (count > 0)
            {
                const float invCount = 1.0f / static_cast<float>(count);
                clusterCenters[clusterIdx] =
                    Vec3{sum.x * invCount, sum.y * invCount, sum.z * invCount};
                clusterCenterCounts[clusterIdx] = count;
            }
        }

        superCenters.assign(superClusterCount, Vec3{0.0f, 0.0f, 0.0f});
        for (size_t superI = 0; superI < superClusterCount; ++superI)
        {
            Vec3 sum{0.0f, 0.0f, 0.0f};
            int count = 0;
            for (int clusterI = snapshot.super_cluster_offsets[superI];
                 clusterI < snapshot.super_cluster_offsets[superI + 1]; ++clusterI)
            {
                const int clusterCount =
                    clusterCenterCounts[static_cast<size_t>(clusterI)];
                if (clusterCount == 0)
                {
                    continue;
                }
                sum = sum +
                      static_cast<float>(clusterCount) *
                          clusterCenters[static_cast<size_t>(clusterI)];
                count += clusterCount;
            }
            if (count > 0)
            {
                const float invCount = 1.0f / static_cast<float>(count);
                superCenters[superI] =
                    Vec3{sum.x * invCount, sum.y * invCount, sum.z * invCount};
            }
        }
    }
    const LTMatrix3 cell = MakeMatrix(snapshot.header.cell);
    auto nearestShiftIdForCenters = [&](Vec3 superCenter, Vec3 clusterCenter)
    {
        int bestShift = kCentralShiftId;
        float bestR2 = std::numeric_limits<float>::max();
        for (int shift = 0; shift < kShiftCount; ++shift)
        {
            const Vec3 dr =
                clusterCenter - ShiftVectorFromId(shift, cell) - superCenter;
            const float r2 = Dot(dr, dr);
            if (r2 < bestR2)
            {
                bestR2 = r2;
                bestShift = shift;
            }
        }
        return bestShift;
    };
    const float exactImaskRadius =
        snapshot.header.cutoff * static_cast<float>(exactImaskRadiusScale);
    const float cutoffSq = exactImaskRadius * exactImaskRadius;
    auto imaskHasAnyCutoffPair = [&](const SpongeWarpJRecordPOD& record,
                                     const SpongeSciPOD& sci, int iLocal) -> bool
    {
        const int clusterI =
            snapshot.super_cluster_offsets[static_cast<size_t>(sci.supercluster_id)] +
            iLocal;
        if (clusterI < 0 ||
            static_cast<size_t>(clusterI) >= snapshot.cluster_offsets.size())
        {
            return false;
        }
        const unsigned int validMaskI =
            snapshot.cluster_valid_masks[static_cast<size_t>(clusterI)];
        const unsigned int localMaskI =
            snapshot.cluster_local_masks[static_cast<size_t>(clusterI)];
        const unsigned int validMaskJ =
            snapshot.cluster_valid_masks[static_cast<size_t>(record.cluster_j)];
        const unsigned int localMaskJ =
            snapshot.cluster_local_masks[static_cast<size_t>(record.cluster_j)];
        const int atomBaseI = snapshot.cluster_offsets[static_cast<size_t>(clusterI)];
        const int atomBaseJ =
            snapshot.cluster_offsets[static_cast<size_t>(record.cluster_j)];
        const Vec3 shiftVec = ShiftVectorFromId(sci.shift_id, cell);
        for (int jLocal = 0; jLocal < kSplitJClusterSize; ++jLocal)
        {
            const int jLane = static_cast<int>(record.j_lane_base) + jLocal;
            if ((validMaskJ & (1u << static_cast<unsigned int>(jLane))) == 0u)
            {
                continue;
            }
            const Float4POD& xj =
                snapshot.sorted_xq[static_cast<size_t>(atomBaseJ + jLane)];
            for (int iLane = 0; iLane < kClusterSize; ++iLane)
            {
                if ((validMaskI & (1u << static_cast<unsigned int>(iLane))) == 0u ||
                    (localMaskI & (1u << static_cast<unsigned int>(iLane))) == 0u)
                {
                    continue;
                }
                if ((record.pair_excl[static_cast<size_t>(
                         jLocal * kClusterSize + iLane)] &
                     (1u << static_cast<unsigned int>(iLocal))) != 0u)
                {
                    continue;
                }
                if (sci.shift_id == kCentralShiftId && clusterI == record.cluster_j &&
                    (localMaskJ & (1u << static_cast<unsigned int>(jLane))) != 0u &&
                    jLane <= iLane)
                {
                    continue;
                }
                const Float4POD& xi =
                    snapshot.sorted_xq[static_cast<size_t>(atomBaseI + iLane)];
                const Vec3 dr = {shiftVec.x + xi.x - xj.x,
                                 shiftVec.y + xi.y - xj.y,
                                 shiftVec.z + xi.z - xj.z};
                const float dr2 = Dot(dr, dr);
                if (dr2 < cutoffSq && dr2 != 0.0f)
                {
                    return true;
                }
            }
        }
        return false;
    };
    auto exactCutoffImask = [&](const SpongeWarpJRecordPOD& record,
                                const SpongeSciPOD& sci) -> unsigned char
    {
        unsigned char filtered = 0u;
        for (int iLocal = 0; iLocal < kSuperClusterClusters; ++iLocal)
        {
            const unsigned char bit =
                static_cast<unsigned char>(1u << static_cast<unsigned int>(iLocal));
            if ((record.imask & bit) != 0u &&
                imaskHasAnyCutoffPair(record, sci, iLocal))
            {
                filtered |= bit;
            }
        }
        return filtered;
    };

    struct AggregatedCluster
    {
        int clusterJ = -1;
        int bucketShift = -1;
        std::array<unsigned char, kWarpSplitCount> hasSplit = {0u, 0u};
        std::array<unsigned char, kWarpSplitCount> imask = {0u, 0u};
        std::array<std::array<unsigned char, kClusterSize * kSplitJClusterSize>,
                   kWarpSplitCount>
            pairExcl = {};
    };

    auto entryHasActivePairExclusion =
        [](const AggregatedCluster& entry, int split) -> bool
    {
        for (int jLocal = 0; jLocal < kSplitJClusterSize; ++jLocal)
        {
            for (int iLane = 0; iLane < kClusterSize; ++iLane)
            {
                const unsigned char pairExclMask =
                    entry.pairExcl[static_cast<size_t>(split)]
                                  [static_cast<size_t>(
                                      jLocal * kClusterSize + iLane)];
                if ((pairExclMask &
                     entry.imask[static_cast<size_t>(split)]) != 0u)
                {
                    return true;
                }
            }
        }
        return false;
    };

    auto shouldStartNewSegment =
        [transformMode](const AggregatedCluster& prev,
                        const AggregatedCluster& current, int shiftId,
                        size_t currentSegmentLength,
                        size_t segmentsStarted,
                        size_t totalSciLength) -> bool
    {
        if (transformMode == SpongeGmxTransformMode::baseline ||
            transformMode == SpongeGmxTransformMode::shiftMajorSfc ||
            transformMode == SpongeGmxTransformMode::allBlockSuperJ ||
            transformMode == SpongeGmxTransformMode::allCjPackedBlockSuperJ ||
            transformMode == SpongeGmxTransformMode::centralBlockSuperJ ||
            transformMode == SpongeGmxTransformMode::centralCjPackedBlockSuperJ ||
            transformMode == SpongeGmxTransformMode::centralBlockSuperJClusterJ ||
            transformMode == SpongeGmxTransformMode::allBlockSuperJClusterJ ||
            transformMode == SpongeGmxTransformMode::centralTailBlockSuperJ256 ||
            transformMode == SpongeGmxTransformMode::centralTailBlockSuperJ320)
        {
            return false;
        }
        const int prevSuperJ = prev.clusterJ / kSuperClusterClusters;
        const int currentSuperJ = current.clusterJ / kSuperClusterClusters;
        const int deltaClusterJ = current.clusterJ - prev.clusterJ;
        const bool changedSuperJ = currentSuperJ != prevSuperJ;
        const bool backwardGap = deltaClusterJ <= 0;
        const bool largeGap16 = deltaClusterJ > 16;
        const bool largeGap32 = deltaClusterJ > 32;
        const bool badGap =
            backwardGap || largeGap16;
        const bool centralTail192 =
            shiftId == kCentralShiftId && currentSegmentLength >= 192;
        const bool centralTail256 =
            shiftId == kCentralShiftId && currentSegmentLength >= 256;
        const bool centralTail320 =
            shiftId == kCentralShiftId && currentSegmentLength >= 320;
        const bool centralTail384 =
            shiftId == kCentralShiftId && currentSegmentLength >= 384;
        const bool totalTail320 = totalSciLength >= 320;
        const bool totalTail384 = totalSciLength >= 384;
        const bool totalTail448 = totalSciLength >= 448;
        switch (transformMode)
        {
            case SpongeGmxTransformMode::splitSuperJ:
                return changedSuperJ;
            case SpongeGmxTransformMode::splitSuperJGap16:
                return changedSuperJ || badGap;
            case SpongeGmxTransformMode::centralSplitSuperJGap16:
                if (shiftId == kCentralShiftId)
                {
                    return changedSuperJ || badGap;
                }
                return changedSuperJ;
            case SpongeGmxTransformMode::centralOnlySplitSuperJ:
                if (shiftId == kCentralShiftId)
                {
                    return changedSuperJ;
                }
                return false;
            case SpongeGmxTransformMode::centralOnlySplitSuperJGap16:
                if (shiftId == kCentralShiftId)
                {
                    return changedSuperJ || badGap;
                }
                return false;
            case SpongeGmxTransformMode::centralTailGapOnly16_192:
                return centralTail192 && badGap;
            case SpongeGmxTransformMode::centralTailSplitSuperJGap16_256:
                return centralTail256 && (changedSuperJ || badGap);
            case SpongeGmxTransformMode::centralTailSplitSuperJGap16_384:
                return centralTail384 && (changedSuperJ || badGap);
            case SpongeGmxTransformMode::centralTailGapOnly16_256:
                return centralTail256 && badGap;
            case SpongeGmxTransformMode::centralTailBackwardOnly256:
                return centralTail256 && backwardGap;
            case SpongeGmxTransformMode::centralTailBackwardOrGap32_256:
                return centralTail256 && (backwardGap || largeGap32);
            case SpongeGmxTransformMode::centralTailGapOnly16_320:
                return centralTail320 && badGap;
            case SpongeGmxTransformMode::centralTailGapOnly16_384:
                return centralTail384 && badGap;
            case SpongeGmxTransformMode::centralTailGapOnly16_256Max1:
                return segmentsStarted == 0 && centralTail256 && badGap;
            case SpongeGmxTransformMode::centralTailGapOnly16_256Total320:
                return totalTail320 && centralTail256 && badGap;
            case SpongeGmxTransformMode::centralTailGapOnly16_256Total384:
                return totalTail384 && centralTail256 && badGap;
            case SpongeGmxTransformMode::centralTailGapOnly16_256Total448:
                return totalTail448 && centralTail256 && badGap;
            case SpongeGmxTransformMode::centralBlockSuperJTailGap16_256:
                return centralTail256 && badGap;
            case SpongeGmxTransformMode::centralBlockSuperJTailGap16_320:
                return centralTail320 && badGap;
            case SpongeGmxTransformMode::allBlockSuperJ:
            case SpongeGmxTransformMode::allCjPackedBlockSuperJ:
            case SpongeGmxTransformMode::centralBlockSuperJ:
            case SpongeGmxTransformMode::centralCjPackedBlockSuperJ:
            case SpongeGmxTransformMode::centralBlockSuperJClusterJ:
            case SpongeGmxTransformMode::allBlockSuperJClusterJ:
            case SpongeGmxTransformMode::centralTailBlockSuperJ256:
            case SpongeGmxTransformMode::centralTailBlockSuperJ320:
            case SpongeGmxTransformMode::shiftMajorSfc:
            case SpongeGmxTransformMode::baseline:
            default:
                return false;
        }
    };

    std::vector<size_t> sciOrder(snapshot.sci.size());
    std::iota(sciOrder.begin(), sciOrder.end(), 0);
    if (transformMode == SpongeGmxTransformMode::shiftMajorSfc)
    {
        std::stable_sort(
            sciOrder.begin(), sciOrder.end(),
            [&snapshot](size_t lhs, size_t rhs)
            {
                const auto& a = snapshot.sci[lhs];
                const auto& b = snapshot.sci[rhs];
                if (a.shift_id != b.shift_id)
                {
                    return a.shift_id < b.shift_id;
                }
                if (a.supercluster_id != b.supercluster_id)
                {
                    return a.supercluster_id < b.supercluster_id;
                }
                return lhs < rhs;
            });
    }
    uint64_t totalReferenceZoneMatchedEntries = 0;
    uint64_t totalReferenceZoneUnmatchedEntries = 0;
    uint64_t totalPairShiftEntries = 0;
    uint64_t totalPairShiftSplitEntries = 0;
    uint64_t totalExactImaskBits = 0;
    uint64_t keptExactImaskBits = 0;
    uint64_t droppedExactRecords = 0;
    std::vector<AggregatedCluster> aggregated;
    std::unordered_map<int, size_t> clusterToIndex;

    for (size_t orderedIdx = 0; orderedIdx < sciOrder.size(); ++orderedIdx)
    {
        const size_t sciIdx = sciOrder[orderedIdx];
        const auto& sci = snapshot.sci[sciIdx];

        aggregated.clear();
        clusterToIndex.clear();
        const int recordBegin = snapshot.record_offsets[sciIdx];
        const int recordEnd = snapshot.record_offsets[sciIdx + 1];
        const size_t recordCount =
            static_cast<size_t>(std::max(recordEnd - recordBegin, 0));
        if (recordCount > aggregated.capacity())
        {
            aggregated.reserve(recordCount);
        }
        if (static_cast<float>(recordCount) >
            static_cast<float>(clusterToIndex.bucket_count()) *
                clusterToIndex.max_load_factor())
        {
            clusterToIndex.reserve(recordCount);
        }
        for (int recordIdx = recordBegin; recordIdx < recordEnd; ++recordIdx)
        {
            const auto& record = snapshot.records[static_cast<size_t>(recordIdx)];
            const int split = record.j_lane_base / kSplitJClusterSize;
            unsigned char activeImask = record.imask;
            if (transformMode == SpongeGmxTransformMode::exactCutoffImask)
            {
                activeImask = exactCutoffImask(record, sci);
                totalExactImaskBits +=
                    static_cast<uint64_t>(std::popcount(record.imask));
                keptExactImaskBits +=
                    static_cast<uint64_t>(std::popcount(activeImask));
                if (activeImask == 0u)
                {
                    ++droppedExactRecords;
                    continue;
                }
            }
            if (transformMode == SpongeGmxTransformMode::pairShiftBucket)
            {
                uint64_t shiftBits = 0ull;
                if (record.pair_shift_index >= 0 &&
                    static_cast<size_t>(record.pair_shift_index) <
                        snapshot.pair_shift_bits.size())
                {
                    shiftBits =
                        snapshot.pair_shift_bits[static_cast<size_t>(
                            record.pair_shift_index)];
                }
                for (int shift = 0; shift < kShiftCount; ++shift)
                {
                    unsigned char splitImask = 0u;
                    for (int iLocal = 0; iLocal < kSuperClusterClusters; ++iLocal)
                    {
                        if ((activeImask &
                             (1u << static_cast<unsigned int>(iLocal))) == 0u)
                        {
                            continue;
                        }
                        if (GetPairShiftId(shiftBits, iLocal) == shift)
                        {
                            splitImask |=
                                static_cast<unsigned char>(
                                    1u << static_cast<unsigned int>(iLocal));
                        }
                    }
                    if (splitImask == 0u)
                    {
                        continue;
                    }
                    const int bucketedClusterJ =
                        record.cluster_j * kShiftCount + shift;
                    auto [it, inserted] =
                        clusterToIndex.emplace(bucketedClusterJ,
                                               aggregated.size());
                    if (inserted)
                    {
                        aggregated.push_back({});
                        aggregated.back().clusterJ = record.cluster_j;
                        aggregated.back().bucketShift = shift;
                    }
                    auto& entry = aggregated[it->second];
                    entry.hasSplit[static_cast<size_t>(split)] = 1u;
                    entry.imask[static_cast<size_t>(split)] |= splitImask;
                    for (size_t i = 0;
                         i < entry.pairExcl[static_cast<size_t>(split)].size();
                         ++i)
                    {
                        entry.pairExcl[static_cast<size_t>(split)][i] |=
                            record.pair_excl[i];
                    }
                }
                continue;
            }
            auto [it, inserted] =
                clusterToIndex.emplace(record.cluster_j, aggregated.size());
            if (inserted)
            {
                aggregated.push_back({});
                aggregated.back().clusterJ = record.cluster_j;
                aggregated.back().bucketShift = sci.shift_id;
            }
            auto& entry = aggregated[it->second];
            entry.hasSplit[static_cast<size_t>(split)] = 1u;
            entry.imask[static_cast<size_t>(split)] |= activeImask;
            for (size_t i = 0; i < entry.pairExcl[static_cast<size_t>(split)].size();
                 ++i)
            {
                entry.pairExcl[static_cast<size_t>(split)][i] |=
                    record.pair_excl[i];
            }
        }
        if (builderStats != nullptr)
        {
            builderStats->aggregateRows += aggregated.size();
        }

        auto shouldReorderBlocks = [&](int shiftId,
                                       size_t recordCount) -> bool
        {
            switch (transformMode)
            {
                case SpongeGmxTransformMode::allBlockSuperJ:
                    return true;
                case SpongeGmxTransformMode::allCjPackedBlockSuperJ:
                    return false;
                case SpongeGmxTransformMode::centralBlockSuperJ:
                    return shiftId == kCentralShiftId;
                case SpongeGmxTransformMode::centralCjPackedBlockSuperJ:
                    return false;
                case SpongeGmxTransformMode::centralBlockSuperJClusterJ:
                    return shiftId == kCentralShiftId;
                case SpongeGmxTransformMode::allBlockSuperJClusterJ:
                    return true;
                case SpongeGmxTransformMode::centralTailBlockSuperJ256:
                    return shiftId == kCentralShiftId && recordCount >= 256;
                case SpongeGmxTransformMode::centralTailBlockSuperJ320:
                    return shiftId == kCentralShiftId && recordCount >= 320;
                case SpongeGmxTransformMode::centralBlockSuperJTailGap16_256:
                    return shiftId == kCentralShiftId && recordCount >= 256;
                case SpongeGmxTransformMode::centralBlockSuperJTailGap16_320:
                    return shiftId == kCentralShiftId && recordCount >= 320;
                default:
                    return false;
            }
        };
        auto shouldReorderCjPackedGroups = [&](int shiftId,
                                               size_t recordCount) -> bool
        {
            (void)recordCount;
            switch (transformMode)
            {
                case SpongeGmxTransformMode::allCjPackedBlockSuperJ:
                    return true;
                case SpongeGmxTransformMode::centralCjPackedBlockSuperJ:
                    return shiftId == kCentralShiftId;
                default:
                    return false;
            }
        };
        auto shouldSortWithinBlock = [&](int shiftId,
                                         size_t recordCount) -> bool
        {
            switch (transformMode)
            {
                case SpongeGmxTransformMode::centralBlockSuperJClusterJ:
                    return shiftId == kCentralShiftId;
                case SpongeGmxTransformMode::allBlockSuperJClusterJ:
                    return true;
                default:
                    return false;
            }
        };
        auto isCentralRunAware = [&]() -> bool
        {
            return sci.shift_id == kCentralShiftId &&
                   (transformMode == SpongeGmxTransformMode::centralRunAware256 ||
                    transformMode == SpongeGmxTransformMode::centralRunAware320 ||
                    transformMode == SpongeGmxTransformMode::centralRunAware384 ||
                    transformMode == SpongeGmxTransformMode::centralRunAware512 ||
                    transformMode == SpongeGmxTransformMode::centralRunAware640);
        };
        auto isJointSuperJClusterJLayout = [&]() -> bool
        {
            return IsCentralJointSuperJClusterJLayout(transformMode);
        };
        auto runAwareTargetRecords = [&]() -> size_t
        {
            switch (transformMode)
            {
                case SpongeGmxTransformMode::centralRunAware256:
                    return 256;
                case SpongeGmxTransformMode::centralRunAware320:
                    return 320;
                case SpongeGmxTransformMode::centralRunAware384:
                    return 384;
                case SpongeGmxTransformMode::centralRunAware512:
                    return 512;
                case SpongeGmxTransformMode::centralRunAware640:
                    return 640;
                default:
                    return static_cast<size_t>(-1);
            }
        };
        auto jointLayoutTargetRecords = [&]() -> size_t
        {
            return CentralJointTargetRecords(transformMode);
        };
        auto activeSplitRecords = [](const AggregatedCluster& entry) -> size_t
        {
            return static_cast<size_t>(entry.hasSplit[0] != 0u) +
                   static_cast<size_t>(entry.hasSplit[1] != 0u);
        };
        auto originalSuperJ = [&](int clusterJ) -> int
        {
            if (clusterJ >= 0 &&
                static_cast<size_t>(clusterJ) <
                    superClusterForOriginalCluster.size())
            {
                const int superJ =
                    superClusterForOriginalCluster[static_cast<size_t>(clusterJ)];
                if (superJ >= 0)
                {
                    return superJ;
                }
            }
            return clusterJ / kSuperClusterClusters;
        };
        auto originalLocalJ = [&](int clusterJ) -> int
        {
            if (clusterJ >= 0 &&
                static_cast<size_t>(clusterJ) <
                    localClusterForOriginalCluster.size())
            {
                const int localJ =
                    localClusterForOriginalCluster[static_cast<size_t>(clusterJ)];
                if (localJ >= 0)
                {
                    return localJ;
                }
            }
            return clusterJ % kSuperClusterClusters;
        };
        auto sortBySuperJClusterJ =
            [&](const AggregatedCluster& lhs,
                const AggregatedCluster& rhs) -> bool
        {
            const int lhsSuperJ = originalSuperJ(lhs.clusterJ);
            const int rhsSuperJ = originalSuperJ(rhs.clusterJ);
            if (lhsSuperJ != rhsSuperJ)
            {
                return lhsSuperJ < rhsSuperJ;
            }
            const int lhsLocalJ = originalLocalJ(lhs.clusterJ);
            const int rhsLocalJ = originalLocalJ(rhs.clusterJ);
            if (lhsLocalJ != rhsLocalJ)
            {
                return lhsLocalJ < rhsLocalJ;
            }
            return lhs.clusterJ < rhs.clusterJ;
        };
        auto emitConvertedSci = [&](int superclusterId, int shiftId,
                                    const std::vector<AggregatedCluster>& entries,
                                    size_t begin, size_t end)
        {
            GromacsSciPOD convertedSci = {
                superclusterId,
                shiftId,
                static_cast<int>(converted.cjpacked.size()),
                static_cast<int>(converted.cjpacked.size()),
            };
            for (size_t base = begin; base < end; base += kJGroupSize)
            {
                GromacsCjPackedPOD packed = {};
                for (int jm = 0; jm < kJGroupSize; ++jm)
                {
                    packed.cj[jm] = -1;
                }
                for (int split = 0; split < kWarpSplitCount; ++split)
                {
                    GromacsExclPOD excl = {};
                    unsigned int splitImask = 0u;
                    bool splitHasPairExclusion = false;
                    for (int jm = 0; jm < kJGroupSize; ++jm)
                    {
                        const size_t idx = base + static_cast<size_t>(jm);
                        if (idx >= end)
                        {
                            continue;
                        }
                        const auto& entry = entries[idx];
                        packed.cj[jm] =
                            denseClusterIndexForOriginalCluster[static_cast<size_t>(
                                entry.clusterJ)];
                        if (entry.hasSplit[static_cast<size_t>(split)] == 0u)
                        {
                            continue;
                        }
                        splitHasPairExclusion |=
                            entryHasActivePairExclusion(entry, split);
                        splitImask |=
                            static_cast<unsigned int>(
                                entry.imask[static_cast<size_t>(split)])
                            << (jm * kSuperClusterClusters);
                        for (int jLocal = 0; jLocal < kSplitJClusterSize; ++jLocal)
                        {
                            for (int iLane = 0; iLane < kClusterSize; ++iLane)
                            {
                                const unsigned char pairExclMask =
                                    entry.pairExcl[static_cast<size_t>(split)]
                                                 [static_cast<size_t>(
                                                     jLocal * kClusterSize + iLane)];
                                unsigned int wexcl = 0u;
                                for (int iLocal = 0;
                                     iLocal < kSuperClusterClusters; ++iLocal)
                                {
                                    const unsigned int imaskBit =
                                        1u << static_cast<unsigned int>(iLocal);
                                    const unsigned int pairBit =
                                        1u << static_cast<unsigned int>(
                                                  jm * kSuperClusterClusters +
                                                  iLocal);
                                    if ((entry.imask[static_cast<size_t>(split)] &
                                         imaskBit) != 0u &&
                                        (pairExclMask & imaskBit) == 0u)
                                    {
                                        wexcl |= pairBit;
                                    }
                                }
                                excl.pair[static_cast<size_t>(
                                    jLocal * kClusterSize + iLane)] |= wexcl;
                            }
                        }
                    }
                    packed.imei[split].imask = splitImask;
                    if (splitHasPairExclusion)
                    {
                        packed.imei[split].excl_ind =
                            static_cast<int>(converted.excl.size());
                        converted.excl.push_back(excl);
                    }
                    else
                    {
                        packed.imei[split].excl_ind = 0;
                    }
                }
                if (transformMode == SpongeGmxTransformMode::padEmptyCj)
                {
                    int fillCluster = -1;
                    for (int jm = 0; jm < kJGroupSize; ++jm)
                    {
                        if (packed.cj[jm] >= 0)
                        {
                            fillCluster = packed.cj[jm];
                            break;
                        }
                    }
                    if (fillCluster >= 0)
                    {
                        for (int jm = 0; jm < kJGroupSize; ++jm)
                        {
                            if (packed.cj[jm] < 0)
                            {
                                packed.cj[jm] = fillCluster;
                            }
                        }
                    }
                }
                converted.cjpacked.push_back(packed);
            }
            convertedSci.cjPackedEnd =
                static_cast<int>(converted.cjpacked.size());
            converted.sci.push_back(convertedSci);
        };
        if (transformMode == SpongeGmxTransformMode::pairShiftBucket &&
            !aggregated.empty())
        {
            std::array<std::vector<AggregatedCluster>, kShiftCount> shiftBuckets;
            uint64_t splitEntries = 0;
            uint64_t totalEntries = 0;
            for (const AggregatedCluster& entry : aggregated)
            {
                int targetShift = entry.bucketShift;
                if (targetShift < 0 || targetShift >= kShiftCount)
                {
                    targetShift = sci.shift_id;
                }
                shiftBuckets[static_cast<size_t>(targetShift)].push_back(entry);
                ++totalEntries;
                if (targetShift != sci.shift_id)
                {
                    ++splitEntries;
                }
            }
            for (int shift = 0; shift < kShiftCount; ++shift)
            {
                auto& bucket = shiftBuckets[static_cast<size_t>(shift)];
                if (bucket.empty())
                {
                    continue;
                }
                emitConvertedSci(sci.supercluster_id, shift, bucket, 0,
                                 bucket.size());
            }
            totalPairShiftEntries += totalEntries;
            totalPairShiftSplitEntries += splitEntries;
            continue;
        }
        if (transformMode == SpongeGmxTransformMode::referenceZone &&
            !aggregated.empty())
        {
            struct ReferenceBucketEntry
            {
                AggregatedCluster entry;
                uint64_t order;
                int clusterJ;
            };
            std::array<std::vector<ReferenceBucketEntry>, kShiftCount> shiftBuckets;
            uint64_t fallbackOrder = referenceZones.size();
            uint64_t matchedEntries = 0;
            uint64_t unmatchedEntries = 0;
            for (const AggregatedCluster& entry : aggregated)
            {
                const int denseClusterJ =
                    denseClusterIndexForOriginalCluster[static_cast<size_t>(
                        entry.clusterJ)];
                int targetShift = sci.shift_id;
                uint64_t targetOrder = fallbackOrder++;
                if (denseClusterJ >= 0)
                {
                    const auto refIt = referenceZones.find(
                        makeReferenceKey(sci.supercluster_id, denseClusterJ));
                    if (refIt != referenceZones.end())
                    {
                        targetShift = refIt->second.shift;
                        targetOrder = refIt->second.order;
                        ++matchedEntries;
                    }
                    else
                    {
                        ++unmatchedEntries;
                    }
                }
                else
                {
                    ++unmatchedEntries;
                }
                if (targetShift < 0 || targetShift >= kShiftCount)
                {
                    targetShift = sci.shift_id;
                }
                shiftBuckets[static_cast<size_t>(targetShift)].push_back(
                    ReferenceBucketEntry{entry, targetOrder, denseClusterJ});
            }
            for (int shift = 0; shift < kShiftCount; ++shift)
            {
                auto& bucket = shiftBuckets[static_cast<size_t>(shift)];
                if (bucket.empty())
                {
                    continue;
                }
                std::stable_sort(
                    bucket.begin(), bucket.end(),
                    [](const ReferenceBucketEntry& lhs,
                       const ReferenceBucketEntry& rhs)
                    {
                        if (lhs.order != rhs.order)
                        {
                            return lhs.order < rhs.order;
                        }
                        return lhs.clusterJ < rhs.clusterJ;
                    });
                std::vector<AggregatedCluster> entries;
                entries.reserve(bucket.size());
                for (const ReferenceBucketEntry& item : bucket)
                {
                    entries.push_back(item.entry);
                }
                emitConvertedSci(sci.supercluster_id, shift, entries, 0,
                                 entries.size());
            }
            totalReferenceZoneMatchedEntries += matchedEntries;
            totalReferenceZoneUnmatchedEntries += unmatchedEntries;
            continue;
        }
        if (transformMode == SpongeGmxTransformMode::geometryShiftBucket &&
            !aggregated.empty())
        {
            std::array<std::vector<AggregatedCluster>, kShiftCount> shiftBuckets;
            for (const AggregatedCluster& entry : aggregated)
            {
                const int targetShift = nearestShiftIdForCenters(
                    superCenters[static_cast<size_t>(sci.supercluster_id)],
                    clusterCenters[static_cast<size_t>(entry.clusterJ)]);
                shiftBuckets[static_cast<size_t>(targetShift)].push_back(entry);
            }
            for (int shift = 0; shift < kShiftCount; ++shift)
            {
                auto& bucket = shiftBuckets[static_cast<size_t>(shift)];
                if (bucket.empty())
                {
                    continue;
                }
                std::stable_sort(
                    bucket.begin(), bucket.end(),
                    [](const AggregatedCluster& lhs, const AggregatedCluster& rhs)
                    {
                        const int lhsSuperJ =
                            lhs.clusterJ / kSuperClusterClusters;
                        const int rhsSuperJ =
                            rhs.clusterJ / kSuperClusterClusters;
                        if (lhsSuperJ != rhsSuperJ)
                        {
                            return lhsSuperJ < rhsSuperJ;
                        }
                        return lhs.clusterJ < rhs.clusterJ;
                    });
                emitConvertedSci(sci.supercluster_id, shift, bucket, 0,
                                 bucket.size());
            }
            continue;
        }
        std::vector<std::pair<size_t, size_t>> runAwareSegments;
        if (isJointSuperJClusterJLayout() &&
            sci.shift_id == kCentralShiftId && aggregated.size() > 1)
        {
            std::stable_sort(aggregated.begin(), aggregated.end(),
                             sortBySuperJClusterJ);
            {
                struct SuperJRun
                {
                    size_t begin;
                    size_t end;
                    size_t activeRecords;
                };
                std::vector<SuperJRun> runs;
                runs.reserve(aggregated.size());
                size_t runBegin = 0;
                while (runBegin < aggregated.size())
                {
                    const int superJ =
                        originalSuperJ(aggregated[runBegin].clusterJ);
                    size_t runEnd = runBegin + 1;
                    size_t activeRecords = activeSplitRecords(aggregated[runBegin]);
                    while (runEnd < aggregated.size() &&
                           originalSuperJ(aggregated[runEnd].clusterJ) == superJ)
                    {
                        activeRecords += activeSplitRecords(aggregated[runEnd]);
                        ++runEnd;
                    }
                    runs.push_back({runBegin, runEnd, activeRecords});
                    runBegin = runEnd;
                }

                const size_t targetRecords = jointLayoutTargetRecords();
                constexpr size_t minTailRecords = 192;
                size_t segmentBegin = 0;
                size_t currentIndex = 0;
                size_t accumulatedRecords = 0;
                for (const SuperJRun& run : runs)
                {
                    if (currentIndex > segmentBegin &&
                        accumulatedRecords >= targetRecords)
                    {
                        runAwareSegments.emplace_back(segmentBegin, currentIndex);
                        segmentBegin = currentIndex;
                        accumulatedRecords = 0;
                    }
                    currentIndex = run.end;
                    accumulatedRecords += run.activeRecords;
                }
                if (segmentBegin < currentIndex)
                {
                    if (!runAwareSegments.empty() &&
                        accumulatedRecords < minTailRecords)
                    {
                        runAwareSegments.back().second = currentIndex;
                    }
                    else
                    {
                        runAwareSegments.emplace_back(segmentBegin, currentIndex);
                    }
                }
            }
        }
        else if (shouldReorderCjPackedGroups(sci.shift_id, aggregated.size()) &&
            aggregated.size() > 1)
        {
            struct CjPackedGroup
            {
                int superJ;
                int firstClusterJ;
                size_t begin;
                size_t end;
            };
            std::vector<CjPackedGroup> groups;
            groups.reserve((aggregated.size() + kJGroupSize - 1) / kJGroupSize);
            for (size_t begin = 0; begin < aggregated.size();
                 begin += kJGroupSize)
            {
                const size_t end =
                    std::min(begin + static_cast<size_t>(kJGroupSize),
                             aggregated.size());
                groups.push_back(
                    {aggregated[begin].clusterJ / kSuperClusterClusters,
                     aggregated[begin].clusterJ, begin, end});
            }
            std::vector<CjPackedGroup> reorderedGroups = groups;
            std::stable_sort(
                reorderedGroups.begin(), reorderedGroups.end(),
                [](const CjPackedGroup& lhs, const CjPackedGroup& rhs)
                {
                    if (lhs.superJ != rhs.superJ)
                    {
                        return lhs.superJ < rhs.superJ;
                    }
                    return lhs.firstClusterJ < rhs.firstClusterJ;
                });
            bool changed = false;
            for (size_t i = 0; i < groups.size(); ++i)
            {
                if (groups[i].begin != reorderedGroups[i].begin)
                {
                    changed = true;
                    break;
                }
            }
            if (changed)
            {
                std::vector<AggregatedCluster> reordered;
                reordered.reserve(aggregated.size());
                for (const auto& group : reorderedGroups)
                {
                    reordered.insert(reordered.end(),
                                     aggregated.begin() + group.begin,
                                     aggregated.begin() + group.end);
                }
                aggregated.swap(reordered);
            }
        }
        else if (shouldReorderBlocks(sci.shift_id, aggregated.size()) &&
                 aggregated.size() > 1)
        {
            struct SuperJBlock
            {
                int superJ;
                int firstClusterJ;
                size_t begin;
                size_t end;
            };
            std::vector<SuperJBlock> blocks;
            blocks.reserve(aggregated.size());
            size_t begin = 0;
            while (begin < aggregated.size())
            {
                const int superJ =
                    aggregated[begin].clusterJ / kSuperClusterClusters;
                size_t end = begin + 1;
                while (end < aggregated.size() &&
                       aggregated[end].clusterJ / kSuperClusterClusters == superJ)
                {
                    ++end;
                }
                blocks.push_back(
                    {superJ, aggregated[begin].clusterJ, begin, end});
                begin = end;
            }
            std::vector<SuperJBlock> reorderedBlocks = blocks;
            std::stable_sort(
                reorderedBlocks.begin(), reorderedBlocks.end(),
                [](const SuperJBlock& lhs, const SuperJBlock& rhs)
                {
                    if (lhs.superJ != rhs.superJ)
                    {
                        return lhs.superJ < rhs.superJ;
                    }
                    return lhs.firstClusterJ < rhs.firstClusterJ;
                });
            bool changed = false;
            for (size_t i = 0; i < blocks.size(); ++i)
            {
                if (blocks[i].begin != reorderedBlocks[i].begin)
                {
                    changed = true;
                    break;
                }
            }
            if (changed)
            {
                std::vector<AggregatedCluster> reordered;
                reordered.reserve(aggregated.size());
                for (const auto& block : reorderedBlocks)
                {
                    if (shouldSortWithinBlock(sci.shift_id, aggregated.size()) &&
                        block.end - block.begin > 1)
                    {
                        std::vector<AggregatedCluster> sortedBlock(
                            aggregated.begin() + block.begin,
                            aggregated.begin() + block.end);
                        std::stable_sort(
                            sortedBlock.begin(), sortedBlock.end(),
                            [](const AggregatedCluster& lhs,
                               const AggregatedCluster& rhs)
                            {
                                return lhs.clusterJ < rhs.clusterJ;
                            });
                        reordered.insert(reordered.end(), sortedBlock.begin(),
                                         sortedBlock.end());
                    }
                    else
                    {
                        reordered.insert(reordered.end(),
                                         aggregated.begin() + block.begin,
                                         aggregated.begin() + block.end);
                    }
                }
                aggregated.swap(reordered);
            }
            else if (shouldSortWithinBlock(sci.shift_id, aggregated.size()))
            {
                size_t blockBegin = 0;
                while (blockBegin < aggregated.size())
                {
                    const int superJ =
                        aggregated[blockBegin].clusterJ / kSuperClusterClusters;
                    size_t blockEnd = blockBegin + 1;
                    while (blockEnd < aggregated.size() &&
                           aggregated[blockEnd].clusterJ / kSuperClusterClusters ==
                               superJ)
                    {
                        ++blockEnd;
                    }
                    if (blockEnd - blockBegin > 1)
                    {
                        std::stable_sort(
                            aggregated.begin() + static_cast<std::ptrdiff_t>(blockBegin),
                            aggregated.begin() + static_cast<std::ptrdiff_t>(blockEnd),
                            [](const AggregatedCluster& lhs,
                               const AggregatedCluster& rhs)
                            {
                                return lhs.clusterJ < rhs.clusterJ;
                            });
                    }
                    blockBegin = blockEnd;
                }
            }
        }
        else if (isCentralRunAware() && aggregated.size() > 1)
        {
            struct RunBlock
            {
                int superJ;
                int firstClusterJ;
                size_t begin;
                size_t end;
                size_t activeRecords;
            };
            std::vector<RunBlock> blocks;
            blocks.reserve(aggregated.size());
            size_t begin = 0;
            while (begin < aggregated.size())
            {
                const int superJ =
                    aggregated[begin].clusterJ / kSuperClusterClusters;
                size_t end = begin + 1;
                size_t activeRecords = activeSplitRecords(aggregated[begin]);
                while (end < aggregated.size() &&
                       aggregated[end].clusterJ / kSuperClusterClusters == superJ)
                {
                    activeRecords += activeSplitRecords(aggregated[end]);
                    ++end;
                }
                blocks.push_back(
                    {superJ, aggregated[begin].clusterJ, begin, end,
                     activeRecords});
                begin = end;
            }
            std::stable_sort(
                blocks.begin(), blocks.end(),
                [](const RunBlock& lhs, const RunBlock& rhs)
                {
                    if (lhs.superJ != rhs.superJ)
                    {
                        return lhs.superJ < rhs.superJ;
                    }
                    return lhs.firstClusterJ < rhs.firstClusterJ;
                });
            std::vector<AggregatedCluster> reordered;
            reordered.reserve(aggregated.size());
            const size_t targetRecords = runAwareTargetRecords();
            size_t segmentBegin = 0;
            size_t currentIndex = 0;
            size_t accumulatedRecords = 0;
            for (const RunBlock& block : blocks)
            {
                if (currentIndex > segmentBegin &&
                    accumulatedRecords >= targetRecords)
                {
                    runAwareSegments.emplace_back(segmentBegin, currentIndex);
                    segmentBegin = currentIndex;
                    accumulatedRecords = 0;
                }
                reordered.insert(reordered.end(), aggregated.begin() + block.begin,
                                 aggregated.begin() + block.end);
                currentIndex += block.end - block.begin;
                accumulatedRecords += block.activeRecords;
            }
            if (segmentBegin < currentIndex)
            {
                runAwareSegments.emplace_back(segmentBegin, currentIndex);
            }
            aggregated.swap(reordered);
        }

        std::vector<std::pair<size_t, size_t>> segments;
        if (!runAwareSegments.empty())
        {
            segments = std::move(runAwareSegments);
        }
        else if (!aggregated.empty())
        {
            size_t segmentBegin = 0;
            size_t segmentsStarted = 0;
            for (size_t idx = 1; idx < aggregated.size(); ++idx)
            {
                if (shouldStartNewSegment(aggregated[idx - 1], aggregated[idx],
                                          sci.shift_id, idx - segmentBegin,
                                          segmentsStarted,
                                          aggregated.size()))
                {
                    segments.emplace_back(segmentBegin, idx);
                    segmentBegin = idx;
                    ++segmentsStarted;
                }
            }
            segments.emplace_back(segmentBegin, aggregated.size());
        }

        for (const auto& segment : segments)
        {
            GromacsSciPOD convertedSci = {
                sci.supercluster_id,
                sci.shift_id,
                static_cast<int>(converted.cjpacked.size()),
                static_cast<int>(converted.cjpacked.size()),
            };
            for (size_t base = segment.first; base < segment.second;
                 base += kJGroupSize)
            {
                for (int jm = 0; jm < kJGroupSize; ++jm)
                {
                    (void)jm;
                }
                GromacsCjPackedPOD packed = {};
                for (int jm = 0; jm < kJGroupSize; ++jm)
                {
                    packed.cj[jm] = -1;
                }
                for (int split = 0; split < kWarpSplitCount; ++split)
                {
                    GromacsExclPOD excl = {};
                    unsigned int splitImask = 0u;
                    bool splitHasPairExclusion = false;
                    for (int jm = 0; jm < kJGroupSize; ++jm)
                    {
                        const size_t idx = base + static_cast<size_t>(jm);
                        if (idx >= segment.second)
                        {
                            continue;
                        }
                        const auto& entry = aggregated[idx];
                        packed.cj[jm] =
                            denseClusterIndexForOriginalCluster[static_cast<size_t>(
                                entry.clusterJ)];
                        if (entry.hasSplit[static_cast<size_t>(split)] == 0u)
                        {
                            continue;
                        }
                        splitHasPairExclusion |=
                            entryHasActivePairExclusion(entry, split);
                        splitImask |=
                            static_cast<unsigned int>(
                                entry.imask[static_cast<size_t>(split)])
                            << (jm * kSuperClusterClusters);
                        for (int jLocal = 0; jLocal < kSplitJClusterSize; ++jLocal)
                        {
                            for (int iLane = 0; iLane < kClusterSize; ++iLane)
                            {
                                const unsigned char pairExclMask =
                                    entry.pairExcl[static_cast<size_t>(split)]
                                                 [static_cast<size_t>(
                                                     jLocal * kClusterSize + iLane)];
                                unsigned int wexcl = 0u;
                                for (int iLocal = 0;
                                     iLocal < kSuperClusterClusters; ++iLocal)
                                {
                                    const unsigned int imaskBit =
                                        1u << static_cast<unsigned int>(iLocal);
                                    const unsigned int pairBit =
                                        1u << static_cast<unsigned int>(
                                                  jm * kSuperClusterClusters +
                                                  iLocal);
                                    if ((entry.imask[static_cast<size_t>(split)] &
                                         imaskBit) != 0u &&
                                        (pairExclMask & imaskBit) == 0u)
                                    {
                                        wexcl |= pairBit;
                                    }
                                }
                                excl.pair[static_cast<size_t>(
                                    jLocal * kClusterSize + iLane)] |= wexcl;
                            }
                        }
                    }
                    packed.imei[split].imask = splitImask;
                    if (splitHasPairExclusion)
                    {
                        packed.imei[split].excl_ind =
                            static_cast<int>(converted.excl.size());
                        converted.excl.push_back(excl);
                    }
                    else
                    {
                        packed.imei[split].excl_ind = 0;
                    }
                }
                if (transformMode == SpongeGmxTransformMode::padEmptyCj)
                {
                    int fillCluster = -1;
                    for (int jm = 0; jm < kJGroupSize; ++jm)
                    {
                        if (packed.cj[jm] >= 0)
                        {
                            fillCluster = packed.cj[jm];
                            break;
                        }
                    }
                    if (fillCluster >= 0)
                    {
                        for (int jm = 0; jm < kJGroupSize; ++jm)
                        {
                            if (packed.cj[jm] < 0)
                            {
                                packed.cj[jm] = fillCluster;
                            }
                        }
                    }
                }
                converted.cjpacked.push_back(packed);
            }
            convertedSci.cjPackedEnd =
                static_cast<int>(converted.cjpacked.size());
            converted.sci.push_back(convertedSci);
        }
    }

    if (transformMode == SpongeGmxTransformMode::referenceZone)
    {
        const uint64_t totalEntries =
            totalReferenceZoneMatchedEntries +
            totalReferenceZoneUnmatchedEntries;
        std::printf("transform=reference-zone matches=%llu unmatched=%llu "
                    "match_ratio=%.6f\n",
                    static_cast<unsigned long long>(
                        totalReferenceZoneMatchedEntries),
                    static_cast<unsigned long long>(
                        totalReferenceZoneUnmatchedEntries),
                    totalEntries > 0
                        ? static_cast<double>(
                              totalReferenceZoneMatchedEntries) /
                              static_cast<double>(totalEntries)
                        : 0.0);
    }
    if (transformMode == SpongeGmxTransformMode::pairShiftBucket)
    {
        std::printf("transform=pair-shift-bucket entries=%llu "
                    "shift_changed_entries=%llu shift_changed_ratio=%.6f\n",
                    static_cast<unsigned long long>(totalPairShiftEntries),
                    static_cast<unsigned long long>(totalPairShiftSplitEntries),
                    totalPairShiftEntries > 0
                        ? static_cast<double>(totalPairShiftSplitEntries) /
                              static_cast<double>(totalPairShiftEntries)
                        : 0.0);
    }
    if (transformMode == SpongeGmxTransformMode::exactCutoffImask)
    {
        const uint64_t droppedBits = totalExactImaskBits - keptExactImaskBits;
        std::printf("transform=exact-cutoff-imask radius_scale=%.6f "
                    "imask_bits=%llu kept_bits=%llu "
                    "dropped_bits=%llu dropped_bit_ratio=%.6f "
                    "dropped_records=%llu\n",
                    exactImaskRadiusScale,
                    static_cast<unsigned long long>(totalExactImaskBits),
                    static_cast<unsigned long long>(keptExactImaskBits),
                    static_cast<unsigned long long>(droppedBits),
                    totalExactImaskBits > 0
                        ? static_cast<double>(droppedBits) /
                              static_cast<double>(totalExactImaskBits)
                        : 0.0,
                    static_cast<unsigned long long>(droppedExactRecords));
    }

    converted.header.sci_numbers = converted.sci.size();
    converted.header.cjpacked_numbers = converted.cjpacked.size();
    converted.header.excl_numbers = converted.excl.size();
    if (builderStats != nullptr)
    {
        builderStats->compactSci = converted.header.sci_numbers;
        builderStats->compactCjPacked = converted.header.cjpacked_numbers;
        builderStats->compactExcl = converted.header.excl_numbers;
    }
    if (builderWorkspace != nullptr)
    {
        return {};
    }
    return converted;
}

SpongeForceOnlySnapshot ApplySpongeLayoutTransform(
    const SpongeForceOnlySnapshot& snapshot, SpongeGmxTransformMode transformMode)
{
    if (!IsCentralJointSuperJClusterJLayout(transformMode))
    {
        return snapshot;
    }

    SpongeForceOnlySnapshot transformed = snapshot;
    transformed.sci.clear();
    transformed.record_offsets.clear();
    transformed.records.clear();
    transformed.record_offsets.push_back(0);

    const size_t superClusterCount =
        snapshot.super_cluster_offsets.empty()
            ? 0
            : snapshot.super_cluster_offsets.size() - 1;
    std::vector<int> superClusterForOriginalCluster(
        snapshot.cluster_offsets.size(), -1);
    std::vector<int> localClusterForOriginalCluster(
        snapshot.cluster_offsets.size(), -1);
    for (size_t superI = 0; superI < superClusterCount; ++superI)
    {
        const int clusterBegin = snapshot.super_cluster_offsets[superI];
        const int clusterEnd = snapshot.super_cluster_offsets[superI + 1];
        for (int clusterI = clusterBegin; clusterI < clusterEnd; ++clusterI)
        {
            if (clusterI < 0 ||
                static_cast<size_t>(clusterI) >= snapshot.cluster_offsets.size())
            {
                continue;
            }
            superClusterForOriginalCluster[static_cast<size_t>(clusterI)] =
                static_cast<int>(superI);
            localClusterForOriginalCluster[static_cast<size_t>(clusterI)] =
                clusterI - clusterBegin;
        }
    }

    auto originalSuperJ = [&](int clusterJ) -> int
    {
        if (clusterJ >= 0 &&
            static_cast<size_t>(clusterJ) < superClusterForOriginalCluster.size())
        {
            const int superJ =
                superClusterForOriginalCluster[static_cast<size_t>(clusterJ)];
            if (superJ >= 0)
            {
                return superJ;
            }
        }
        return clusterJ / kSuperClusterClusters;
    };
    auto originalLocalJ = [&](int clusterJ) -> int
    {
        if (clusterJ >= 0 &&
            static_cast<size_t>(clusterJ) < localClusterForOriginalCluster.size())
        {
            const int localJ =
                localClusterForOriginalCluster[static_cast<size_t>(clusterJ)];
            if (localJ >= 0)
            {
                return localJ;
            }
        }
        return clusterJ % kSuperClusterClusters;
    };
    auto appendSegment = [&](const SpongeSciPOD& sci,
                             const std::vector<int>& orderedRecords,
                             size_t begin, size_t end)
    {
        SpongeSciPOD emitted = sci;
        emitted.cjpacked_begin = static_cast<int>(transformed.records.size());
        for (size_t idx = begin; idx < end; ++idx)
        {
            transformed.records.push_back(
                snapshot.records[static_cast<size_t>(orderedRecords[idx])]);
        }
        emitted.cjpacked_end = static_cast<int>(transformed.records.size());
        transformed.sci.push_back(emitted);
        transformed.record_offsets.push_back(
            static_cast<int>(transformed.records.size()));
    };
    auto appendOriginalSci = [&](const SpongeSciPOD& sci, int recordBegin,
                                int recordEnd)
    {
        SpongeSciPOD emitted = sci;
        emitted.cjpacked_begin = static_cast<int>(transformed.records.size());
        for (int recordIdx = recordBegin; recordIdx < recordEnd; ++recordIdx)
        {
            transformed.records.push_back(
                snapshot.records[static_cast<size_t>(recordIdx)]);
        }
        emitted.cjpacked_end = static_cast<int>(transformed.records.size());
        transformed.sci.push_back(emitted);
        transformed.record_offsets.push_back(
            static_cast<int>(transformed.records.size()));
    };

    const size_t targetRecords = CentralJointTargetRecords(transformMode);
    constexpr size_t minTailRecords = 192;
    for (size_t sciIdx = 0; sciIdx < snapshot.sci.size(); ++sciIdx)
    {
        const SpongeSciPOD& sci = snapshot.sci[sciIdx];
        const int recordBegin = snapshot.record_offsets[sciIdx];
        const int recordEnd = snapshot.record_offsets[sciIdx + 1];
        if (sci.shift_id != kCentralShiftId || recordEnd - recordBegin <= 1)
        {
            appendOriginalSci(sci, recordBegin, recordEnd);
            continue;
        }

        std::vector<int> orderedRecords(
            static_cast<size_t>(recordEnd - recordBegin));
        std::iota(orderedRecords.begin(), orderedRecords.end(), recordBegin);
        std::stable_sort(
            orderedRecords.begin(), orderedRecords.end(),
            [&](int lhsIdx, int rhsIdx)
            {
                const SpongeWarpJRecordPOD& lhs =
                    snapshot.records[static_cast<size_t>(lhsIdx)];
                const SpongeWarpJRecordPOD& rhs =
                    snapshot.records[static_cast<size_t>(rhsIdx)];
                const int lhsSuperJ = originalSuperJ(lhs.cluster_j);
                const int rhsSuperJ = originalSuperJ(rhs.cluster_j);
                if (lhsSuperJ != rhsSuperJ)
                {
                    return lhsSuperJ < rhsSuperJ;
                }
                const int lhsLocalJ = originalLocalJ(lhs.cluster_j);
                const int rhsLocalJ = originalLocalJ(rhs.cluster_j);
                if (lhsLocalJ != rhsLocalJ)
                {
                    return lhsLocalJ < rhsLocalJ;
                }
                if (lhs.cluster_j != rhs.cluster_j)
                {
                    return lhs.cluster_j < rhs.cluster_j;
                }
                if (lhs.j_lane_base != rhs.j_lane_base)
                {
                    return lhs.j_lane_base < rhs.j_lane_base;
                }
                return lhsIdx < rhsIdx;
            });

        struct SuperJRun
        {
            size_t begin;
            size_t end;
            size_t activeRecords;
        };
        std::vector<SuperJRun> runs;
        runs.reserve(orderedRecords.size());
        size_t runBegin = 0;
        while (runBegin < orderedRecords.size())
        {
            const int superJ = originalSuperJ(
                snapshot.records[static_cast<size_t>(orderedRecords[runBegin])]
                    .cluster_j);
            size_t runEnd = runBegin + 1;
            while (runEnd < orderedRecords.size() &&
                   originalSuperJ(snapshot
                                      .records[static_cast<size_t>(
                                          orderedRecords[runEnd])]
                                      .cluster_j) == superJ)
            {
                ++runEnd;
            }
            runs.push_back({runBegin, runEnd, runEnd - runBegin});
            runBegin = runEnd;
        }

        std::vector<std::pair<size_t, size_t>> segments;
        size_t segmentBegin = 0;
        size_t currentIndex = 0;
        size_t accumulatedRecords = 0;
        for (const SuperJRun& run : runs)
        {
            if (currentIndex > segmentBegin && accumulatedRecords >= targetRecords)
            {
                segments.emplace_back(segmentBegin, currentIndex);
                segmentBegin = currentIndex;
                accumulatedRecords = 0;
            }
            currentIndex = run.end;
            accumulatedRecords += run.activeRecords;
        }
        if (segmentBegin < currentIndex)
        {
            if (!segments.empty() && accumulatedRecords < minTailRecords)
            {
                segments.back().second = currentIndex;
            }
            else
            {
                segments.emplace_back(segmentBegin, currentIndex);
            }
        }
        for (const auto& segment : segments)
        {
            appendSegment(sci, orderedRecords, segment.first, segment.second);
        }
    }

    transformed.header.sci_numbers = transformed.sci.size();
    transformed.header.record_numbers = transformed.records.size();
    return transformed;
}

__device__ inline void ReduceSubgroupVec(float& x, float& y, float& z,
                                         int lane, int subgroup_width)
{
    const unsigned int mask = SubgroupMask(lane, subgroup_width);
    for (int delta = subgroup_width >> 1; delta > 0; delta >>= 1)
    {
        x += __shfl_down_sync(mask, x, delta, subgroup_width);
        y += __shfl_down_sync(mask, y, delta, subgroup_width);
        z += __shfl_down_sync(mask, z, delta, subgroup_width);
    }
}

__device__ inline void ReduceWarpOverJ(float& x, float& y, float& z,
                                       int subgroup_width)
{
    for (int delta = warpSize >> 1; delta >= subgroup_width; delta >>= 1)
    {
        x += __shfl_down_sync(kFullMask, x, delta, warpSize);
        y += __shfl_down_sync(kFullMask, y, delta, warpSize);
        z += __shfl_down_sync(kFullMask, z, delta, warpSize);
    }
}

__device__ inline float ReduceWarpFloatOverJ(float value, int subgroup_width)
{
    for (int delta = warpSize >> 1; delta >= subgroup_width; delta >>= 1)
    {
        value += __shfl_down_sync(kFullMask, value, delta, warpSize);
    }
    return value;
}

__device__ inline LTMatrix3 ReduceWarpVirialOverJ(LTMatrix3 value,
                                                  int subgroup_width)
{
    for (int delta = warpSize >> 1; delta >= subgroup_width; delta >>= 1)
    {
        value.a11 += __shfl_down_sync(kFullMask, value.a11, delta, warpSize);
        value.a21 += __shfl_down_sync(kFullMask, value.a21, delta, warpSize);
        value.a22 += __shfl_down_sync(kFullMask, value.a22, delta, warpSize);
        value.a31 += __shfl_down_sync(kFullMask, value.a31, delta, warpSize);
        value.a32 += __shfl_down_sync(kFullMask, value.a32, delta, warpSize);
        value.a33 += __shfl_down_sync(kFullMask, value.a33, delta, warpSize);
    }
    return value;
}

__device__ inline float ReduceWarpFloatAll(float value)
{
    for (int delta = warpSize >> 1; delta > 0; delta >>= 1)
    {
        value += __shfl_down_sync(kFullMask, value, delta, warpSize);
    }
    return value;
}

__device__ inline LTMatrix3 ReduceWarpVirialAll(LTMatrix3 value)
{
    for (int delta = warpSize >> 1; delta > 0; delta >>= 1)
    {
        value.a11 += __shfl_down_sync(kFullMask, value.a11, delta, warpSize);
        value.a21 += __shfl_down_sync(kFullMask, value.a21, delta, warpSize);
        value.a22 += __shfl_down_sync(kFullMask, value.a22, delta, warpSize);
        value.a31 += __shfl_down_sync(kFullMask, value.a31, delta, warpSize);
        value.a32 += __shfl_down_sync(kFullMask, value.a32, delta, warpSize);
        value.a33 += __shfl_down_sync(kFullMask, value.a33, delta, warpSize);
    }
    return value;
}

__device__ __forceinline__ float4 PackVirialLo(LTMatrix3 value)
{
    return {value.a11, value.a21, value.a22, value.a31};
}

__device__ __forceinline__ float2 PackVirialHi(LTMatrix3 value)
{
    return {value.a32, value.a33};
}

__device__ __forceinline__ LTMatrix3 UnpackVirial(float4 lo, float2 hi)
{
    return {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y};
}

template <bool enabled, int size>
struct ReplayEnergyBuffer
{
};

template <int size>
struct ReplayEnergyBuffer<true, size>
{
    float values[size] = {};

    __device__ __forceinline__ float& operator[](int idx)
    {
        return values[idx];
    }
};

template <bool total_output, bool need_energy, int size>
struct ReplayFullOutputBuffer;

template <bool need_energy, int size>
struct ReplayFullOutputBuffer<false, need_energy, size>
{
    ReplayEnergyBuffer<need_energy, size> energy_lj;
    ReplayEnergyBuffer<need_energy, size> energy_coulomb;
    LTMatrix3 virial[size];

    __device__ __forceinline__ ReplayFullOutputBuffer()
    {
        for (int i = 0; i < size; ++i)
        {
            virial[i] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
};

template <bool need_energy, int size>
struct ReplayFullOutputBuffer<true, need_energy, size>
{
    float energy_lj_total = 0.0f;
    float energy_coulomb_total = 0.0f;
    LTMatrix3 virial_total = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
};

template <bool total_output>
struct GmxPackedPerAtomSharedScratch
{
    char dummy = 0;
};

template <bool enabled, int size>
struct GmxPackedCentralDirectScratch
{
    char dummy = 0;
};

template <int size>
struct GmxPackedCentralDirectScratch<true, size>
{
    LTMatrix3 virial[kWarpSplitCount * size];
    float energy[kWarpSplitCount * size];
    float direct_cf_energy[kWarpSplitCount * size];
    float lj_energy[kWarpSplitCount * size];
};

template <bool need_energy, bool need_virial, bool total_output>
using GmxPackedForceStorage =
    std::conditional_t<total_output && (need_energy || need_virial),
                       float3,
                       float4>;

template <>
struct GmxPackedPerAtomSharedScratch<false>
{
    int atom_ids[kClusterSize * kSuperClusterClusters];
    float4 warp1_i_force[kSuperClusterClusters][kClusterSize];
    float warp1_i_energy_lj[kSuperClusterClusters][kClusterSize];
    float warp1_i_energy_coulomb[kSuperClusterClusters][kClusterSize];
    float4 warp1_i_virial_lo[kSuperClusterClusters][kClusterSize];
    float2 warp1_i_virial_hi[kSuperClusterClusters][kClusterSize];
};

__device__ __forceinline__ void ReduceWarpOverJ8(float& x, float& y, float& z)
{
    x += __shfl_down_sync(kFullMask, x, 16, warpSize);
    y += __shfl_down_sync(kFullMask, y, 16, warpSize);
    z += __shfl_down_sync(kFullMask, z, 16, warpSize);
    x += __shfl_down_sync(kFullMask, x, 8, warpSize);
    y += __shfl_down_sync(kFullMask, y, 8, warpSize);
    z += __shfl_down_sync(kFullMask, z, 8, warpSize);
}

__device__ __forceinline__ float ReduceWarpIToComponent8(float x, float y,
                                                         float z,
                                                         int warp_j_local)
{
    x += __shfl_down_sync(kFullMask, x, kClusterSize, warpSize);
    y += __shfl_up_sync(kFullMask, y, kClusterSize, warpSize);
    z += __shfl_down_sync(kFullMask, z, kClusterSize, warpSize);
    if (warp_j_local & 1)
    {
        x = y;
    }
    x += __shfl_down_sync(kFullMask, x, 2 * kClusterSize, warpSize);
    z += __shfl_up_sync(kFullMask, z, 2 * kClusterSize, warpSize);
    if (warp_j_local & 2)
    {
        x = z;
    }
    return x;
}

__device__ inline float ReduceSubgroupVecToComponent(float x, float y, float z,
                                                     int i_lane, int lane,
                                                     int subgroup_width)
{
    const unsigned int mask = SubgroupMask(lane, subgroup_width);
    x += __shfl_down_sync(mask, x, 1, subgroup_width);
    y += __shfl_up_sync(mask, y, 1, subgroup_width);
    z += __shfl_down_sync(mask, z, 1, subgroup_width);
    if (i_lane & 1)
    {
        x = y;
    }
    x += __shfl_down_sync(mask, x, 2, subgroup_width);
    z += __shfl_up_sync(mask, z, 2, subgroup_width);
    if (i_lane & 2)
    {
        x = z;
    }
    x += __shfl_down_sync(mask, x, 4, subgroup_width);
    return x;
}

__device__ __forceinline__ float ReduceSubgroupVecToComponent8(float x, float y,
                                                               float z,
                                                               int i_lane,
                                                               int lane)
{
    const unsigned int mask = SubgroupMask8(lane);
    x += __shfl_down_sync(mask, x, 1, kClusterSize);
    y += __shfl_up_sync(mask, y, 1, kClusterSize);
    z += __shfl_down_sync(mask, z, 1, kClusterSize);
    if (i_lane & 1)
    {
        x = y;
    }
    x += __shfl_down_sync(mask, x, 2, kClusterSize);
    z += __shfl_up_sync(mask, z, 2, kClusterSize);
    if (i_lane & 2)
    {
        x = z;
    }
    x += __shfl_down_sync(mask, x, 4, kClusterSize);
    return x;
}

__device__ __forceinline__ float ReduceSubgroupFloat8(float value, int lane)
{
    const unsigned int mask = SubgroupMask8(lane);
    value += __shfl_down_sync(mask, value, 1, kClusterSize);
    value += __shfl_down_sync(mask, value, 2, kClusterSize);
    value += __shfl_down_sync(mask, value, 4, kClusterSize);
    return value;
}

// Legacy experimental non-gmxpacked force-only path. Kept only as archived code;
// excluded from the active build after cleanup.
#if 0
template <bool force_soa, bool use_lj_comb, bool gmx_dense>
__global__ __launch_bounds__(kClusterSize * kSuperClusterClusters, 13)
void SpongeForceOnlyReplayKernel(
    int sci_numbers, int cluster_size, int super_cluster_clusters,
    int local_atom_numbers, const int* cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const int* super_cluster_offsets,
    const SpongeSciPOD* sci_entries, const int* compact_record_offsets,
    const SpongeWarpJRecordPOD* compact_records, const uint64_t* pair_shift_bits,
    const int* sorted_atom_ids, const float4* sorted_xq,
    const int* sorted_lj_type, const float2* sorted_lj_comb, LTMatrix3 cell,
    const float2* lj_ab_packed, float cutoff, float* frc_x, float* frc_y,
    float* frc_z, float pme_beta)
{
    constexpr int max_super_cluster_atoms = kClusterSize * kSuperClusterClusters;
    constexpr int max_block_warps = 2;
    const int sci = blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (sci >= sci_numbers ||
        tid >= super_cluster_clusters * cluster_size)
    {
        return;
    }

    const SpongeSciPOD sci_entry = sci_entries[sci];
    const int super_i = sci_entry.supercluster_id;
    const int cluster_i_start = super_cluster_offsets[super_i];
    const int cluster_i_end =
        gmx_dense ? (cluster_i_start + kSuperClusterClusters)
                  : super_cluster_offsets[super_i + 1];
    const int record_begin = compact_record_offsets[sci];
    const int record_end = compact_record_offsets[sci + 1];
    const bool sci_is_central = sci_entry.shift_id == kCentralShiftId;
    const float cutoff_sq = cutoff * cutoff;
    const float beta2 = pme_beta * pme_beta;
    const float beta3 = beta2 * pme_beta;

    __shared__ float4 shared_i_xq[max_super_cluster_atoms];
    __shared__ int shared_i_lj_type[max_super_cluster_atoms];
    __shared__ float2 shared_i_lj_comb[max_super_cluster_atoms];
    __shared__ int shared_i_sorted_ids[max_super_cluster_atoms];
    __shared__ unsigned int shared_i_valid_masks[kSuperClusterClusters];
    __shared__ unsigned int shared_i_local_masks[kSuperClusterClusters];
    __shared__ float4 warp1_i_force[kSuperClusterClusters][kClusterSize];

    const int i_lane = threadIdx.x;
    const int j_lane = threadIdx.y;
    const int lane = tid & (warpSize - 1);
    const int warp_id = tid / warpSize;
    const int active_cluster_count = cluster_i_end - cluster_i_start;
    const int i_slot = j_lane * cluster_size + i_lane;
    const int warp_j_base = warp_id * kSplitJClusterSize;
    const int warp_j_local = j_lane - warp_j_base;

#define SPONGE_REPLAY_I_LOCAL_LIST(OP) \
    OP(0)                              \
    OP(1)                              \
    OP(2)                              \
    OP(3)                              \
    OP(4)                              \
    OP(5)                              \
    OP(6)                              \
    OP(7)

#define SPONGE_REPLAY_DECLARE_FCI(I) \
    float fci_x_##I = 0.0f;          \
    float fci_y_##I = 0.0f;          \
    float fci_z_##I = 0.0f;
    SPONGE_REPLAY_I_LOCAL_LIST(SPONGE_REPLAY_DECLARE_FCI)
#undef SPONGE_REPLAY_DECLARE_FCI

    if constexpr (!gmx_dense)
    {
        if (j_lane == 0)
        {
            if (i_lane < active_cluster_count)
            {
                const int cluster_i = cluster_i_start + i_lane;
                shared_i_valid_masks[i_lane] = cluster_valid_masks[cluster_i];
                shared_i_local_masks[i_lane] = cluster_local_masks[cluster_i];
            }
            else if (i_lane < kSuperClusterClusters)
            {
                shared_i_valid_masks[i_lane] = 0u;
                shared_i_local_masks[i_lane] = 0u;
            }
        }
        if (j_lane < active_cluster_count)
        {
            const int cluster_i = cluster_i_start + j_lane;
            if ((cluster_valid_masks[cluster_i] & (1u << i_lane)) != 0u)
            {
                const int sorted_i = cluster_offsets[cluster_i] + i_lane;
                shared_i_xq[i_slot] = sorted_xq[sorted_i];
                if constexpr (use_lj_comb)
                {
                    shared_i_lj_comb[i_slot] = LoadReadOnly(sorted_lj_comb + sorted_i);
                }
                else
                {
                    shared_i_lj_type[i_slot] = sorted_lj_type[sorted_i];
                }
                shared_i_sorted_ids[i_slot] = sorted_i;
            }
            else
            {
                shared_i_sorted_ids[i_slot] = -1;
            }
        }
    }
    else
    {
        const int cluster_i = cluster_i_start + j_lane;
        const int sorted_i = cluster_offsets[cluster_i] + i_lane;
        shared_i_xq[i_slot] = sorted_xq[sorted_i];
        if constexpr (use_lj_comb)
        {
            shared_i_lj_comb[i_slot] = LoadReadOnly(sorted_lj_comb + sorted_i);
        }
        else
        {
            shared_i_lj_type[i_slot] = sorted_lj_type[sorted_i];
        }
        shared_i_sorted_ids[i_slot] = sorted_i;
    }
    __syncthreads();

    const unsigned int i_lane_mask = 1u << static_cast<unsigned int>(i_lane);
    unsigned int active_i_mask = 0u;
    if constexpr (!gmx_dense)
    {
#define SPONGE_REPLAY_CACHE_ACTIVE_I(I)                                 \
        if ((I) < active_cluster_count)                                 \
        {                                                               \
            const unsigned int valid_mask_i = shared_i_valid_masks[I];  \
            const unsigned int local_mask_i = shared_i_local_masks[I];  \
            if ((valid_mask_i & i_lane_mask) != 0u &&                   \
                (local_mask_i & i_lane_mask) != 0u)                     \
            {                                                           \
                active_i_mask |= (1u << static_cast<unsigned int>(I));  \
            }                                                           \
        }
        SPONGE_REPLAY_I_LOCAL_LIST(SPONGE_REPLAY_CACHE_ACTIVE_I)
#undef SPONGE_REPLAY_CACHE_ACTIVE_I
    }

#define SPONGE_REPLAY_DECLARE_I_LJ_TYPE(I) int cached_i_lj_type_##I = 0;
    SPONGE_REPLAY_I_LOCAL_LIST(SPONGE_REPLAY_DECLARE_I_LJ_TYPE)
#undef SPONGE_REPLAY_DECLARE_I_LJ_TYPE

#define SPONGE_REPLAY_CACHE_I_LJ_TYPE(I)                                  \
    if ((I) < active_cluster_count &&                                     \
        (active_i_mask & (1u << static_cast<unsigned int>(I))) != 0u)    \
    {                                                                     \
        if constexpr (!use_lj_comb)                                       \
        {                                                                 \
            cached_i_lj_type_##I =                                        \
                shared_i_lj_type[(I) * cluster_size + i_lane];            \
        }                                                                 \
    }
    SPONGE_REPLAY_I_LOCAL_LIST(SPONGE_REPLAY_CACHE_I_LJ_TYPE)
#undef SPONGE_REPLAY_CACHE_I_LJ_TYPE

    if constexpr (gmx_dense)
    {
        active_i_mask = 0xffu;
    }

    uint64_t sci_shift_bits = 0ull;
    if constexpr (gmx_dense)
    {
        if (lane == 0)
        {
            sci_shift_bits = pair_shift_bits[sci_entry.shift_id];
        }
        sci_shift_bits = BroadcastWarpU64(sci_shift_bits, 0);
    }

    for (int record_idx = record_begin + warp_id; record_idx < record_end;
         record_idx += max_block_warps)
    {
        const SpongeWarpJRecordPOD* record = compact_records + record_idx;
        int cluster_j = 0;
        unsigned int valid_mask_j = 0u;
        unsigned int imask = 0u;
        if (lane == 0)
        {
            cluster_j = record->cluster_j;
            valid_mask_j = gmx_dense ? ((1u << kSplitJClusterSize) - 1u)
                                     : record->valid_mask;
            imask = record->imask;
        }
        cluster_j = __shfl_sync(kFullMask, cluster_j, 0, warpSize);
        valid_mask_j = __shfl_sync(kFullMask, valid_mask_j, 0, warpSize);
        imask = __shfl_sync(kFullMask, imask, 0, warpSize);

        uint64_t shift_bits = sci_shift_bits;
        unsigned int local_mask_j = 0u;
        unsigned int j_lane_base = 0u;
        if (lane == 0 && !gmx_dense && record->pair_shift_index >= 0)
        {
            shift_bits = pair_shift_bits[record->pair_shift_index];
        }
        if (lane == 0)
        {
            local_mask_j = gmx_dense ? ((1u << kSplitJClusterSize) - 1u)
                                     : record->local_mask;
            j_lane_base = record->j_lane_base;
        }
        if constexpr (!gmx_dense)
        {
            shift_bits = BroadcastWarpU64(shift_bits, 0);
            local_mask_j = __shfl_sync(kFullMask, local_mask_j, 0, warpSize);
        }
        j_lane_base = __shfl_sync(kFullMask, j_lane_base, 0, warpSize);

        if (cluster_j < 0 ||
            (valid_mask_j & (1u << static_cast<unsigned int>(warp_j_local))) == 0u)
        {
            continue;
        }

        int sorted_j = -1;
        int absolute_j_lane = -1;
        float4 r2_xq = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        int r2_lj_type = 0;
        float2 r2_lj_comb = make_float2(0.0f, 0.0f);
        if (i_lane == 0)
        {
            sorted_j = record->sorted_j_base + warp_j_local;
            absolute_j_lane = static_cast<int>(j_lane_base) + warp_j_local;
            r2_xq = sorted_xq[sorted_j];
            if constexpr (use_lj_comb)
            {
                r2_lj_comb = LoadReadOnly(sorted_lj_comb + sorted_j);
            }
            else
            {
                r2_lj_type = sorted_lj_type[sorted_j];
            }
        }
        sorted_j = BroadcastSubgroupInt(sorted_j, lane, cluster_size);
        absolute_j_lane = BroadcastSubgroupInt(absolute_j_lane, lane, cluster_size);
        r2_xq = BroadcastSubgroupFloat4(r2_xq, lane, cluster_size);
        if constexpr (use_lj_comb)
        {
            r2_lj_comb = BroadcastSubgroupFloat2(r2_lj_comb, lane, cluster_size);
        }
        else
        {
            r2_lj_type = BroadcastSubgroupInt(r2_lj_type, lane, cluster_size);
        }

        const int atom_j_is_local =
            gmx_dense ? 1
                      : ((local_mask_j &
                          (1u << static_cast<unsigned int>(warp_j_local))) != 0u
                             ? 1
                             : 0);
        const unsigned char pair_excl_mask =
            record->pair_excl[warp_j_local * cluster_size + i_lane];
        float fcj_x = 0.0f;
        float fcj_y = 0.0f;
        float fcj_z = 0.0f;

#define SPONGE_REPLAY_ACCUMULATE_I(I)                                   \
        if ((I) < active_cluster_count &&                               \
            (imask & (1u << static_cast<unsigned int>(I))) != 0u &&     \
            (active_i_mask &                                             \
             (1u << static_cast<unsigned int>(I))) != 0u)               \
        {                                                               \
            const int cluster_i = cluster_i_start + (I);                \
            if (!(sci_is_central && cluster_i == cluster_j &&           \
                  atom_j_is_local != 0 &&                               \
                  absolute_j_lane <= i_lane) &&                         \
                (pair_excl_mask &                                       \
                 (1u << static_cast<unsigned int>(I))) == 0u)           \
            {                                                           \
                const float4 r1_xq =                                    \
                    shared_i_xq[(I) * cluster_size + i_lane];           \
                const Vec3 pair_shift =                                 \
                    ShiftVectorFromId(GetPairShiftId(shift_bits, I),    \
                                      cell);                            \
                const Vec3 dr = {                                       \
                    r2_xq.x - r1_xq.x - pair_shift.x,                   \
                    r2_xq.y - r1_xq.y - pair_shift.y,                   \
                    r2_xq.z - r1_xq.z - pair_shift.z};                  \
                const float dr2 = Dot(dr, dr);                          \
                if (dr2 < cutoff_sq && dr2 != 0.0f)                     \
                {                                                       \
                    const float inv_r = rsqrtf(dr2);                    \
                    const float inv_r2 = inv_r * inv_r;                 \
                    const float inv_r6 =                                \
                        inv_r2 * inv_r2 * inv_r2;                       \
                    const float charge_product =                        \
                        r1_xq.w * r2_xq.w;                              \
                    float frc_abs = 0.0f;                               \
                    if constexpr (use_lj_comb)                          \
                    {                                                   \
                        const float2 ljcp_i =                           \
                            shared_i_lj_comb[(I) * cluster_size + i_lane]; \
                        const float c6 = ljcp_i.x * r2_lj_comb.x;       \
                        const float c12 = ljcp_i.y * r2_lj_comb.y;      \
                        frc_abs = GetLjForceAbs(inv_r2, inv_r6,         \
                                                c12, c6);               \
                    }                                                   \
                    else                                                \
                    {                                                   \
                        const int lj_index =                            \
                            GetLjType(cached_i_lj_type_##I,             \
                                      r2_lj_type);                      \
                        const float2 AB = LoadReadOnly(lj_ab_packed +   \
                                                        lj_index);      \
                        frc_abs = GetLjForceAbs(inv_r2, inv_r6,         \
                                                AB.x, AB.y);            \
                    }                                                   \
                    frc_abs -= GetDirectCoulombForceAbsPmeCorrF(        \
                        charge_product, inv_r, inv_r2, beta2 * dr2,     \
                        beta3);                                         \
                    const float frc_x_local = frc_abs * dr.x;           \
                    const float frc_y_local = frc_abs * dr.y;           \
                    const float frc_z_local = frc_abs * dr.z;           \
                    fci_x_##I += frc_x_local;                           \
                    fci_y_##I += frc_y_local;                           \
                    fci_z_##I += frc_z_local;                           \
                    if (atom_j_is_local != 0)                           \
                    {                                                   \
                        fcj_x -= frc_x_local;                           \
                        fcj_y -= frc_y_local;                           \
                        fcj_z -= frc_z_local;                           \
                    }                                                   \
                }                                                       \
            }                                                           \
        }
        SPONGE_REPLAY_I_LOCAL_LIST(SPONGE_REPLAY_ACCUMULATE_I)
#undef SPONGE_REPLAY_ACCUMULATE_I

        if (atom_j_is_local != 0)
        {
            ReduceSubgroupVec(fcj_x, fcj_y, fcj_z, lane, cluster_size);
            if (i_lane == 0)
            {
                atomicAdd(frc_x + sorted_j, fcj_x);
                atomicAdd(frc_y + sorted_j, fcj_y);
                atomicAdd(frc_z + sorted_j, fcj_z);
            }
        }
    }

#define SPONGE_REPLAY_REDUCE_I(I)                                         \
    if ((I) < active_cluster_count)                                       \
    {                                                                     \
        const bool active_i =                                             \
            (active_i_mask & (1u << static_cast<unsigned int>(I))) != 0u; \
        float reduced_x = active_i ? fci_x_##I : 0.0f;                    \
        float reduced_y = active_i ? fci_y_##I : 0.0f;                    \
        float reduced_z = active_i ? fci_z_##I : 0.0f;                    \
        ReduceWarpOverJ(reduced_x, reduced_y, reduced_z, cluster_size);   \
        if (lane < cluster_size)                                          \
        {                                                                 \
            if (warp_id == 0)                                             \
            {                                                             \
                fci_x_##I = reduced_x;                                    \
                fci_y_##I = reduced_y;                                    \
                fci_z_##I = reduced_z;                                    \
            }                                                             \
            else                                                          \
            {                                                             \
                warp1_i_force[I][lane] =                                  \
                    make_float4(reduced_x, reduced_y, reduced_z, 0.0f);   \
            }                                                             \
        }                                                                 \
    }
    SPONGE_REPLAY_I_LOCAL_LIST(SPONGE_REPLAY_REDUCE_I)
#undef SPONGE_REPLAY_REDUCE_I
    __syncthreads();
    if (warp_id == 0 && j_lane == 0)
    {
 #define SPONGE_REPLAY_WRITEBACK_I(I)                                     \
        if ((I) < active_cluster_count)                                   \
        {                                                                 \
            const bool active_i =                                         \
                (active_i_mask &                                          \
                 (1u << static_cast<unsigned int>(I))) != 0u;             \
            if (active_i)                                                 \
            {                                                             \
                const int sorted_i =                                      \
                    shared_i_sorted_ids[(I) * cluster_size + i_lane];     \
                const float4 warp1_force = warp1_i_force[I][i_lane];      \
                atomicAdd(frc_x + sorted_i, fci_x_##I + warp1_force.x);   \
                atomicAdd(frc_y + sorted_i, fci_y_##I + warp1_force.y);   \
                atomicAdd(frc_z + sorted_i, fci_z_##I + warp1_force.z);   \
            }                                                             \
        }
        SPONGE_REPLAY_I_LOCAL_LIST(SPONGE_REPLAY_WRITEBACK_I)
#undef SPONGE_REPLAY_WRITEBACK_I
    }

#undef SPONGE_REPLAY_I_LOCAL_LIST
}
#endif

template <bool need_energy, bool total_output>
__global__ __launch_bounds__(kClusterSize * kSuperClusterClusters, 13)
void SpongeVirialReplayWarpRecordKernel(
    int sci_numbers, int cluster_size, int super_cluster_clusters,
    int local_atom_numbers, const int* cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const int* super_cluster_offsets,
    const SpongeSciPOD* sci_entries, const int* compact_record_offsets,
    const SpongeWarpJRecordPOD* compact_records,
    const uint64_t* pair_shift_bits, const int* sorted_atom_ids,
    const float4* sorted_xq, const int* sorted_lj_type, LTMatrix3 cell,
    const float2* lj_ab_packed, float cutoff, float* frc_x, float* frc_y,
    float* frc_z, float pme_beta, float* atom_energy,
    LTMatrix3* atom_virial, float* atom_direct_cf_energy, float* atom_lj_ene)
{
    constexpr int max_super_cluster_atoms = kClusterSize * kSuperClusterClusters;
    constexpr int max_block_warps = 2;
    const int sci = blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (sci >= sci_numbers || tid >= super_cluster_clusters * cluster_size)
    {
        return;
    }

    (void)local_atom_numbers;
    const SpongeSciPOD sci_entry = sci_entries[sci];
    const int super_i = sci_entry.supercluster_id;
    const int cluster_i_start = super_cluster_offsets[super_i];
    const int cluster_i_end = super_cluster_offsets[super_i + 1];
    const int record_begin = compact_record_offsets[sci];
    const int record_end = compact_record_offsets[sci + 1];
    const bool sci_is_central = sci_entry.shift_id == kCentralShiftId;
    const float cutoff_sq = cutoff * cutoff;
    const float beta2 = pme_beta * pme_beta;
    const float beta3 = beta2 * pme_beta;

    __shared__ float4 shared_i_xq[max_super_cluster_atoms];
    __shared__ int shared_i_lj_type[max_super_cluster_atoms];
    __shared__ int shared_i_atom_ids[max_super_cluster_atoms];
    __shared__ int shared_i_sorted_ids[max_super_cluster_atoms];
    __shared__ unsigned int shared_i_valid_masks[kSuperClusterClusters];
    __shared__ unsigned int shared_i_local_masks[kSuperClusterClusters];
    __shared__ float4 warp1_i_force[kSuperClusterClusters][kClusterSize];
    __shared__ float warp1_i_energy_lj[kSuperClusterClusters][kClusterSize];
    __shared__ float warp1_i_energy_coulomb[kSuperClusterClusters][kClusterSize];
    __shared__ float4 warp1_i_virial_lo[kSuperClusterClusters][kClusterSize];
    __shared__ float2 warp1_i_virial_hi[kSuperClusterClusters][kClusterSize];
    __shared__ float4 shared_total_virial_lo[max_block_warps];
    __shared__ float2 shared_total_virial_hi[max_block_warps];
    __shared__ float shared_total_energy_lj[max_block_warps];
    __shared__ float shared_total_energy_coulomb[max_block_warps];

    const int i_lane = threadIdx.x;
    const int j_lane = threadIdx.y;
    const int lane = tid & (warpSize - 1);
    const int warp_id = tid / warpSize;
    const int active_cluster_count = cluster_i_end - cluster_i_start;
    const int i_slot = j_lane * cluster_size + i_lane;
    const int warp_j_base = warp_id * kSplitJClusterSize;
    const int warp_j_local = j_lane - warp_j_base;

#define SPONGE_FULL_I_LOCAL_LIST(OP) \
    OP(0)                            \
    OP(1)                            \
    OP(2)                            \
    OP(3)                            \
    OP(4)                            \
    OP(5)                            \
    OP(6)                            \
    OP(7)

#define SPONGE_FULL_DECLARE_FCI(I) \
    float fci_x_##I = 0.0f;         \
    float fci_y_##I = 0.0f;         \
    float fci_z_##I = 0.0f;
    SPONGE_FULL_I_LOCAL_LIST(SPONGE_FULL_DECLARE_FCI)
#undef SPONGE_FULL_DECLARE_FCI
    ReplayFullOutputBuffer<total_output, need_energy, kSuperClusterClusters>
        output_buf;

    if (j_lane == 0)
    {
        if (i_lane < active_cluster_count)
        {
            const int cluster_i = cluster_i_start + i_lane;
            shared_i_valid_masks[i_lane] = cluster_valid_masks[cluster_i];
            shared_i_local_masks[i_lane] = cluster_local_masks[cluster_i];
        }
        else if (i_lane < kSuperClusterClusters)
        {
            shared_i_valid_masks[i_lane] = 0u;
            shared_i_local_masks[i_lane] = 0u;
        }
    }
    if (j_lane < active_cluster_count)
    {
        const int cluster_i = cluster_i_start + j_lane;
        if ((cluster_valid_masks[cluster_i] & (1u << i_lane)) != 0u)
        {
            const int sorted_atom_i = cluster_offsets[cluster_i] + i_lane;
            shared_i_xq[i_slot] = sorted_xq[sorted_atom_i];
            shared_i_lj_type[i_slot] = sorted_lj_type[sorted_atom_i];
            shared_i_atom_ids[i_slot] = sorted_atom_ids[sorted_atom_i];
            shared_i_sorted_ids[i_slot] = sorted_atom_i;
        }
        else
        {
            shared_i_atom_ids[i_slot] = -1;
            shared_i_sorted_ids[i_slot] = -1;
        }
    }
    __syncthreads();

    const unsigned int i_lane_mask = 1u << static_cast<unsigned int>(i_lane);
    unsigned int active_i_mask = 0u;
#define SPONGE_FULL_CACHE_ACTIVE_I(I)                                    \
    if ((I) < active_cluster_count)                                      \
    {                                                                    \
        const unsigned int valid_mask_i = shared_i_valid_masks[I];       \
        const unsigned int local_mask_i = shared_i_local_masks[I];       \
        if ((valid_mask_i & i_lane_mask) != 0u &&                        \
            (local_mask_i & i_lane_mask) != 0u)                          \
        {                                                                \
            active_i_mask |= (1u << static_cast<unsigned int>(I));       \
        }                                                                \
    }
    SPONGE_FULL_I_LOCAL_LIST(SPONGE_FULL_CACHE_ACTIVE_I)
#undef SPONGE_FULL_CACHE_ACTIVE_I

#define SPONGE_FULL_DECLARE_I_LJ_TYPE(I) int cached_i_lj_type_##I = 0;
    SPONGE_FULL_I_LOCAL_LIST(SPONGE_FULL_DECLARE_I_LJ_TYPE)
#undef SPONGE_FULL_DECLARE_I_LJ_TYPE

#define SPONGE_FULL_CACHE_I_LJ_TYPE(I)                                  \
    if ((I) < active_cluster_count &&                                   \
        (active_i_mask & (1u << static_cast<unsigned int>(I))) != 0u)  \
    {                                                                   \
        cached_i_lj_type_##I =                                          \
            shared_i_lj_type[(I) * cluster_size + i_lane];              \
    }
    SPONGE_FULL_I_LOCAL_LIST(SPONGE_FULL_CACHE_I_LJ_TYPE)
#undef SPONGE_FULL_CACHE_I_LJ_TYPE

    for (int record_idx = record_begin + warp_id; record_idx < record_end;
         record_idx += max_block_warps)
    {
        const SpongeWarpJRecordPOD* record = compact_records + record_idx;
        int cluster_j = 0;
        int pair_shift_index = -1;
        unsigned int valid_mask_j = 0u;
        unsigned int imask = 0u;
        if (lane == 0)
        {
            cluster_j = record->cluster_j;
            pair_shift_index = record->pair_shift_index;
            valid_mask_j = record->valid_mask;
            imask = record->imask;
        }
        cluster_j = __shfl_sync(kFullMask, cluster_j, 0, warpSize);
        pair_shift_index = __shfl_sync(kFullMask, pair_shift_index, 0, warpSize);
        valid_mask_j = __shfl_sync(kFullMask, valid_mask_j, 0, warpSize);
        imask = __shfl_sync(kFullMask, imask, 0, warpSize);
        uint64_t shift_bits = 0ull;
        unsigned int local_mask_j = 0u;
        unsigned int j_lane_base = 0u;
        if (lane == 0 && pair_shift_index >= 0)
        {
            shift_bits = pair_shift_bits[pair_shift_index];
        }
        if (lane == 0)
        {
            local_mask_j = record->local_mask;
            j_lane_base = record->j_lane_base;
        }
        shift_bits = BroadcastWarpU64(shift_bits, 0);
        local_mask_j = __shfl_sync(kFullMask, local_mask_j, 0, warpSize);
        j_lane_base = __shfl_sync(kFullMask, j_lane_base, 0, warpSize);

        if (cluster_j < 0 ||
            (valid_mask_j & (1u << static_cast<unsigned int>(warp_j_local))) ==
                0u)
        {
            continue;
        }

        int sorted_j = -1;
        int atom_j = -1;
        int absolute_j_lane = -1;
        float4 r2_xq = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        int r2_lj_type = 0;
        if (i_lane == 0)
        {
            sorted_j = record->sorted_j_base + warp_j_local;
            atom_j = sorted_atom_ids[sorted_j];
            absolute_j_lane = static_cast<int>(j_lane_base) + warp_j_local;
            r2_xq = sorted_xq[sorted_j];
            r2_lj_type = sorted_lj_type[sorted_j];
        }
        sorted_j = BroadcastSubgroupInt(sorted_j, lane, cluster_size);
        atom_j = BroadcastSubgroupInt(atom_j, lane, cluster_size);
        absolute_j_lane = BroadcastSubgroupInt(absolute_j_lane, lane, cluster_size);
        r2_xq = BroadcastSubgroupFloat4(r2_xq, lane, cluster_size);
        r2_lj_type = BroadcastSubgroupInt(r2_lj_type, lane, cluster_size);
        const int atom_j_is_local =
            (local_mask_j & (1u << static_cast<unsigned int>(warp_j_local))) !=
                    0u
                ? 1
                : 0;
        const VectorLj r2 = MakePackedLjAtom(r2_xq, r2_lj_type);
        const unsigned char pair_excl_mask =
            record->pair_excl[warp_j_local * cluster_size + i_lane];
        float fcj_x = 0.0f;
        float fcj_y = 0.0f;
        float fcj_z = 0.0f;

#define SPONGE_FULL_ACCUMULATE_I(I)                                         \
        if ((I) < active_cluster_count &&                                   \
            (imask & (1u << static_cast<unsigned int>(I))) != 0u &&         \
            (active_i_mask &                                                \
             (1u << static_cast<unsigned int>(I))) != 0u)                   \
        {                                                                   \
            const int cluster_i = cluster_i_start + (I);                    \
            if (!(sci_is_central && cluster_i == cluster_j &&               \
                  atom_j_is_local != 0 && absolute_j_lane <= i_lane) &&     \
                (pair_excl_mask &                                           \
                 (1u << static_cast<unsigned int>(I))) == 0u)               \
            {                                                               \
                const float4 r1_xq =                                        \
                    shared_i_xq[(I) * cluster_size + i_lane];               \
                const int r1_lj_type =                                      \
                    shared_i_lj_type[(I) * cluster_size + i_lane];          \
                const VectorLj r1 = MakePackedLjAtom(r1_xq, r1_lj_type);    \
                const Vec3 pair_shift =                                     \
                    ShiftVectorFromId(GetPairShiftId(shift_bits, I), cell); \
                const Vec3 dr = GetShiftedDisplacement(r2, r1, pair_shift); \
                const float dr2 = Dot(dr, dr);                              \
                if (dr2 < cutoff_sq && dr2 != 0.0f)                         \
                {                                                           \
                    const float inv_r = rsqrtf(dr2);                        \
                    const float inv_r2 = inv_r * inv_r;                     \
                    const float inv_r6 = inv_r2 * inv_r2 * inv_r2;          \
                    const float beta_dr = pme_beta * (dr2 * inv_r);         \
                    const float charge_product = r1.charge * r2.charge;     \
                    const int atom_pair_lj_type =                           \
                        GetLjType(r1.lj_type, r2.lj_type);                  \
                    const float2 AB = lj_ab_packed[atom_pair_lj_type];      \
                    const float ij_factor =                                 \
                        atom_j_is_local != 0 ? 1.0f : 0.5f;                 \
                    float frc_abs =                                         \
                        GetLjForceAbs(inv_r2, inv_r6, AB.x, AB.y);          \
                    frc_abs -= GetDirectCoulombForceAbsPmeCorrF(            \
                        charge_product, inv_r, inv_r2, beta2 * dr2, beta3); \
                    const float frc_x_local = frc_abs * dr.x;               \
                    const float frc_y_local = frc_abs * dr.y;               \
                    const float frc_z_local = frc_abs * dr.z;               \
                    fci_x_##I += frc_x_local;                               \
                    fci_y_##I += frc_y_local;                               \
                    fci_z_##I += frc_z_local;                               \
                    if (atom_j_is_local != 0)                               \
                    {                                                       \
                        fcj_x -= frc_x_local;                               \
                        fcj_y -= frc_y_local;                               \
                        fcj_z -= frc_z_local;                               \
                    }                                                       \
                    const LTMatrix3 pair_virial =                           \
                        -ij_factor *                                        \
                        GetVirialFromForceDis({frc_x_local, frc_y_local,    \
                                               frc_z_local},                 \
                                              dr);                          \
                    if constexpr (total_output)                             \
                    {                                                       \
                        output_buf.virial_total =                           \
                            output_buf.virial_total + pair_virial;          \
                    }                                                       \
                    else                                                    \
                    {                                                       \
                        output_buf.virial[I] =                              \
                            output_buf.virial[I] + pair_virial;             \
                    }                                                       \
                    if constexpr (need_energy)                              \
                    {                                                       \
                        const float pair_energy_lj =                        \
                            ij_factor * GetLjEnergy(inv_r6, AB.x, AB.y);    \
                        const float pair_energy_coulomb =                   \
                            ij_factor *                                     \
                            GetDirectCoulombEnergy(charge_product, inv_r,   \
                                                    beta_dr);               \
                        if constexpr (total_output)                         \
                        {                                                   \
                            output_buf.energy_lj_total += pair_energy_lj;   \
                            output_buf.energy_coulomb_total +=              \
                                pair_energy_coulomb;                        \
                        }                                                   \
                        else                                                \
                        {                                                   \
                            output_buf.energy_lj[I] += pair_energy_lj;      \
                            output_buf.energy_coulomb[I] +=                 \
                                pair_energy_coulomb;                        \
                        }                                                   \
                    }                                                       \
                }                                                           \
            }                                                               \
        }
        SPONGE_FULL_I_LOCAL_LIST(SPONGE_FULL_ACCUMULATE_I)
#undef SPONGE_FULL_ACCUMULATE_I

        if (atom_j_is_local != 0)
        {
            ReduceSubgroupVec(fcj_x, fcj_y, fcj_z, lane, cluster_size);
            if (i_lane == 0)
            {
                atomicAdd(frc_x + sorted_j, fcj_x);
                atomicAdd(frc_y + sorted_j, fcj_y);
                atomicAdd(frc_z + sorted_j, fcj_z);
            }
        }
    }

#define SPONGE_FULL_REDUCE_I(I)                                            \
    if ((I) < active_cluster_count)                                        \
    {                                                                      \
        const bool active_i =                                              \
            (active_i_mask & (1u << static_cast<unsigned int>(I))) != 0u;  \
        if constexpr (!total_output)                                       \
        {                                                                  \
            LTMatrix3 reduced_virial =                                     \
                active_i ? output_buf.virial[I]                            \
                         : LTMatrix3{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; \
            reduced_virial = ReduceWarpVirialOverJ(reduced_virial,         \
                                                   cluster_size);          \
            if (lane < cluster_size)                                       \
            {                                                              \
                if (warp_id == 0)                                          \
                {                                                          \
                    output_buf.virial[I] = reduced_virial;                 \
                }                                                          \
                else                                                       \
                {                                                          \
                    warp1_i_virial_lo[I][lane] = PackVirialLo(reduced_virial); \
                    warp1_i_virial_hi[I][lane] = PackVirialHi(reduced_virial); \
                }                                                          \
            }                                                              \
        }                                                                  \
        float reduced_x = active_i ? fci_x_##I : 0.0f;                     \
        float reduced_y = active_i ? fci_y_##I : 0.0f;                     \
        float reduced_z = active_i ? fci_z_##I : 0.0f;                     \
        ReduceWarpOverJ(reduced_x, reduced_y, reduced_z, cluster_size);    \
        if (lane < cluster_size)                                           \
        {                                                                  \
            if (warp_id == 0)                                              \
            {                                                              \
                fci_x_##I = reduced_x;                                     \
                fci_y_##I = reduced_y;                                     \
                fci_z_##I = reduced_z;                                     \
            }                                                              \
            else                                                           \
            {                                                              \
                warp1_i_force[I][lane] =                                   \
                    make_float4(reduced_x, reduced_y, reduced_z, 0.0f);    \
            }                                                              \
        }                                                                  \
        if constexpr (need_energy)                                         \
        {                                                                  \
            if constexpr (!total_output)                                   \
            {                                                              \
                float reduced_lj = active_i ? output_buf.energy_lj[I]      \
                                            : 0.0f;                        \
                float reduced_coulomb = active_i                           \
                                            ? output_buf.energy_coulomb[I] \
                                            : 0.0f;                        \
                reduced_lj = ReduceWarpFloatOverJ(reduced_lj, cluster_size); \
                reduced_coulomb = ReduceWarpFloatOverJ(reduced_coulomb,    \
                                                       cluster_size);       \
                if (lane < cluster_size)                                   \
                {                                                          \
                    if (warp_id == 0)                                      \
                    {                                                      \
                        output_buf.energy_lj[I] = reduced_lj;              \
                        output_buf.energy_coulomb[I] = reduced_coulomb;    \
                    }                                                      \
                    else                                                   \
                    {                                                      \
                        warp1_i_energy_lj[I][lane] = reduced_lj;           \
                        warp1_i_energy_coulomb[I][lane] = reduced_coulomb; \
                    }                                                      \
                }                                                          \
            }                                                              \
        }                                                                  \
    }
    SPONGE_FULL_I_LOCAL_LIST(SPONGE_FULL_REDUCE_I)
#undef SPONGE_FULL_REDUCE_I
    __syncthreads();

    if (warp_id == 0 && j_lane == 0)
    {
#define SPONGE_FULL_WRITEBACK_I(I)                                         \
        if ((I) < active_cluster_count)                                    \
        {                                                                  \
            const bool active_i =                                          \
                (active_i_mask &                                           \
                 (1u << static_cast<unsigned int>(I))) != 0u;              \
            if (active_i)                                                  \
            {                                                              \
                const int atom_i =                                         \
                    shared_i_atom_ids[(I) * cluster_size + i_lane];        \
                const int sorted_i =                                       \
                    shared_i_sorted_ids[(I) * cluster_size + i_lane];      \
                const float4 warp1_force = warp1_i_force[I][i_lane];       \
                atomicAdd(frc_x + sorted_i, fci_x_##I + warp1_force.x);    \
                atomicAdd(frc_y + sorted_i, fci_y_##I + warp1_force.y);    \
                atomicAdd(frc_z + sorted_i, fci_z_##I + warp1_force.z);    \
                if constexpr (!total_output)                               \
                {                                                          \
                    if constexpr (need_energy)                             \
                    {                                                      \
                        const float total_energy_lj =                      \
                            output_buf.energy_lj[I] +                      \
                            warp1_i_energy_lj[I][i_lane];                  \
                        const float total_energy_coulomb =                 \
                            output_buf.energy_coulomb[I] +                 \
                            warp1_i_energy_coulomb[I][i_lane];             \
                        atomicAdd(atom_energy + atom_i,                    \
                                  total_energy_lj + total_energy_coulomb); \
                        atomicAdd(atom_lj_ene + atom_i, total_energy_lj);  \
                        atomicAdd(atom_direct_cf_energy + atom_i,          \
                                  total_energy_coulomb);                   \
                    }                                                      \
                    AtomicAddVirial(                                       \
                        atom_virial + atom_i,                              \
                        output_buf.virial[I] +                             \
                            UnpackVirial(warp1_i_virial_lo[I][i_lane],     \
                                         warp1_i_virial_hi[I][i_lane]));   \
                }                                                          \
            }                                                              \
        }
        SPONGE_FULL_I_LOCAL_LIST(SPONGE_FULL_WRITEBACK_I)
#undef SPONGE_FULL_WRITEBACK_I
    }

    if constexpr (total_output)
    {
        LTMatrix3 reduced_total_virial =
            ReduceWarpVirialAll(output_buf.virial_total);
        float reduced_total_energy_lj = 0.0f;
        float reduced_total_energy_coulomb = 0.0f;
        if constexpr (need_energy)
        {
            reduced_total_energy_lj = ReduceWarpFloatAll(output_buf.energy_lj_total);
            reduced_total_energy_coulomb =
                ReduceWarpFloatAll(output_buf.energy_coulomb_total);
        }
        if (lane == 0)
        {
            shared_total_virial_lo[warp_id] = PackVirialLo(reduced_total_virial);
            shared_total_virial_hi[warp_id] = PackVirialHi(reduced_total_virial);
            if constexpr (need_energy)
            {
                shared_total_energy_lj[warp_id] = reduced_total_energy_lj;
                shared_total_energy_coulomb[warp_id] =
                    reduced_total_energy_coulomb;
            }
        }
        __syncthreads();
        if (tid == 0)
        {
            LTMatrix3 block_total_virial =
                UnpackVirial(shared_total_virial_lo[0], shared_total_virial_hi[0]);
            for (int warp = 1; warp < max_block_warps; ++warp)
            {
                block_total_virial =
                    block_total_virial +
                    UnpackVirial(shared_total_virial_lo[warp],
                                 shared_total_virial_hi[warp]);
            }
            AtomicAddVirial(atom_virial, block_total_virial);
            if constexpr (need_energy)
            {
                float block_total_energy_lj = 0.0f;
                float block_total_energy_coulomb = 0.0f;
                for (int warp = 0; warp < max_block_warps; ++warp)
                {
                    block_total_energy_lj += shared_total_energy_lj[warp];
                    block_total_energy_coulomb += shared_total_energy_coulomb[warp];
                }
                atomicAdd(atom_energy, block_total_energy_lj + block_total_energy_coulomb);
                atomicAdd(atom_lj_ene, block_total_energy_lj);
                atomicAdd(atom_direct_cf_energy, block_total_energy_coulomb);
            }
        }
    }

#undef SPONGE_FULL_I_LOCAL_LIST
}

// Legacy dense/compact experimental kernels. Excluded from the active build
// after cleanup; native fulloutput and gmxpacked are the supported paths.
#if 0
__global__ __launch_bounds__(kClusterSize * kSuperClusterClusters, 13)
void SpongeForceOnlyReplayDenseKernel(
    int sci_numbers, const SpongeSciPOD* sci_entries, const int* compact_record_offsets,
    const SpongeDenseWarpJRecordPOD* compact_records, const float4* dense_i_xq,
    const float2* dense_i_lj_comb, const float4* sorted_xq,
    const float2* sorted_lj_comb, LTMatrix3 cell, float cutoff, float4* frc_xyz,
    float pme_beta)
{
    constexpr int max_super_cluster_atoms = kClusterSize * kSuperClusterClusters;
    constexpr int max_block_warps = 2;
    const int sci = blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (sci >= sci_numbers || tid >= max_super_cluster_atoms)
    {
        return;
    }

    const SpongeSciPOD sci_entry = sci_entries[sci];
    const int super_i = sci_entry.supercluster_id;
    const int sorted_super_i_base = super_i * max_super_cluster_atoms;
    const int record_begin = compact_record_offsets[sci];
    const int record_end = compact_record_offsets[sci + 1];
    const float cutoff_sq = cutoff * cutoff;
    const float beta2 = pme_beta * pme_beta;
    const float beta3 = beta2 * pme_beta;
    constexpr float min_distance_sq = 3.82e-07f;

    __shared__ float4 shared_i_xq[max_super_cluster_atoms];
    __shared__ float2 shared_i_lj_comb[max_super_cluster_atoms];
    __shared__ Vec3 shared_sci_shift_vec;

    const int i_lane = threadIdx.x;
    const int j_lane = threadIdx.y;
    const int lane = tid & (warpSize - 1);
    const int warp_id = tid / warpSize;
    const int i_slot = j_lane * kClusterSize + i_lane;
    const int warp_j_base = warp_id * kSplitJClusterSize;
    const int warp_j_local = j_lane - warp_j_base;

#define SPONGE_REPLAY_DENSE_I_LOCAL_LIST(OP) \
    OP(0)                                    \
    OP(1)                                    \
    OP(2)                                    \
    OP(3)                                    \
    OP(4)                                    \
    OP(5)                                    \
    OP(6)                                    \
    OP(7)

#define SPONGE_REPLAY_DENSE_DECLARE_FCI(I) \
    float fci_x_##I = 0.0f;                \
    float fci_y_##I = 0.0f;                \
    float fci_z_##I = 0.0f;
    SPONGE_REPLAY_DENSE_I_LOCAL_LIST(SPONGE_REPLAY_DENSE_DECLARE_FCI)
#undef SPONGE_REPLAY_DENSE_DECLARE_FCI

    const int sorted_i = sorted_super_i_base + j_lane * kClusterSize + i_lane;
    shared_i_xq[i_slot] = LoadReadOnly(dense_i_xq + sorted_i);
    shared_i_lj_comb[i_slot] = LoadReadOnly(dense_i_lj_comb + sorted_i);

    if (tid == 0)
    {
        shared_sci_shift_vec = ShiftVectorFromId(sci_entry.shift_id, cell);
    }
    __syncthreads();
    const Vec3 sci_shift_vec = shared_sci_shift_vec;

    for (int record_idx = record_begin + warp_id; record_idx < record_end;
         record_idx += max_block_warps)
    {
        const SpongeDenseWarpJRecordPOD* record = compact_records + record_idx;
        int sorted_j_base = -1;
        if (lane == 0)
        {
            sorted_j_base = record->sorted_j_base;
        }
        sorted_j_base = __shfl_sync(0xFFFFFFFFu, sorted_j_base, 0);
        if (sorted_j_base < 0)
        {
            continue;
        }

        const unsigned int active_mask =
            static_cast<unsigned int>(record->active_masks[warp_j_local][i_lane]);
        const int sorted_j = sorted_j_base + warp_j_local;
        const float4 r2_xq = LoadReadOnly(sorted_xq + sorted_j);
        const float2 r2_lj_comb = LoadReadOnly(sorted_lj_comb + sorted_j);

        const float shifted_j_x = r2_xq.x - sci_shift_vec.x;
        const float shifted_j_y = r2_xq.y - sci_shift_vec.y;
        const float shifted_j_z = r2_xq.z - sci_shift_vec.z;
        const float qj = r2_xq.w;
        const float lj_j_x = r2_lj_comb.x;
        const float lj_j_y = r2_lj_comb.y;

        float fcj_x = 0.0f;
        float fcj_y = 0.0f;
        float fcj_z = 0.0f;

#define SPONGE_REPLAY_DENSE_ACCUMULATE_I(I)                                  \
        if ((active_mask & (1u << static_cast<unsigned int>(I))) != 0u)      \
        {                                                                    \
            const float4 r1_xq = shared_i_xq[(I) * kClusterSize + i_lane];   \
            const float dx = shifted_j_x - r1_xq.x;                          \
            const float dy = shifted_j_y - r1_xq.y;                          \
            const float dz = shifted_j_z - r1_xq.z;                          \
            const float dr2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));           \
            if (dr2 < cutoff_sq)                                             \
            {                                                                \
                const float r2 = fmaxf(dr2, min_distance_sq);                \
                const float inv_r = rsqrtf(r2);                              \
                const float inv_r2 = inv_r * inv_r;                          \
                const float inv_r6 = inv_r2 * inv_r2 * inv_r2;               \
                const float charge_product = r1_xq.w * qj;                   \
                const float2 ljcp_i =                                        \
                    shared_i_lj_comb[(I) * kClusterSize + i_lane];           \
                const float c6 = ljcp_i.x * lj_j_x;                          \
                const float c12 = ljcp_i.y * lj_j_y;                         \
                const float lj_term = fmaf(c12, inv_r6, -c6);               \
                const float lj_f_invr = (inv_r6 * inv_r2) * lj_term;        \
                const float coulomb_corr =                                   \
                    fmaf(PmeCorrF(beta2 * r2), beta3, inv_r * inv_r2);      \
                const float F_invr =                                         \
                    fmaf(-charge_product, coulomb_corr, lj_f_invr);         \
                const float fij_x = F_invr * dx;                             \
                const float fij_y = F_invr * dy;                             \
                const float fij_z = F_invr * dz;                             \
                fci_x_##I += fij_x;                                          \
                fci_y_##I += fij_y;                                          \
                fci_z_##I += fij_z;                                          \
                fcj_x -= fij_x;                                              \
                fcj_y -= fij_y;                                              \
                fcj_z -= fij_z;                                              \
            }                                                                \
        }

        SPONGE_REPLAY_DENSE_I_LOCAL_LIST(SPONGE_REPLAY_DENSE_ACCUMULATE_I)
#undef SPONGE_REPLAY_DENSE_ACCUMULATE_I

        const float fcj_component =
            ReduceSubgroupVecToComponent8(fcj_x, fcj_y, fcj_z, i_lane, lane);
        if (i_lane < 3)
        {
            float* frc_j = reinterpret_cast<float*>(frc_xyz + sorted_j);
            atomicAdd(frc_j + i_lane, fcj_component);
        }
    }

#define SPONGE_REPLAY_DENSE_REDUCE_I(I)                                   \
    float reduced_x_##I = fci_x_##I;                                      \
    float reduced_y_##I = fci_y_##I;                                      \
    float reduced_z_##I = fci_z_##I;                                      \
    const float reduced_component_##I =                                   \
        ReduceWarpIToComponent8(reduced_x_##I, reduced_y_##I,             \
                                reduced_z_##I, warp_j_local);             \
    if (warp_j_local < 3)                                                 \
    {                                                                     \
        const int sorted_i_local =                                        \
            sorted_super_i_base + (I) * kClusterSize + i_lane;            \
        float* frc_i = reinterpret_cast<float*>(frc_xyz + sorted_i_local); \
        atomicAdd(frc_i + warp_j_local, reduced_component_##I);           \
    }
    SPONGE_REPLAY_DENSE_I_LOCAL_LIST(SPONGE_REPLAY_DENSE_REDUCE_I)
#undef SPONGE_REPLAY_DENSE_REDUCE_I

#undef SPONGE_REPLAY_DENSE_I_LOCAL_LIST
}

__global__ __launch_bounds__(kClusterSize * kSuperClusterClusters, 13)
void SpongeForceOnlyReplayCompactDenseIKernel(
    int sci_numbers, const SpongeSciPOD* sci_entries, const int* record_offsets,
    const SpongeWarpJRecordPOD* records, const int* super_cluster_offsets,
    const float4* dense_i_xq, const float2* dense_i_lj_comb,
    const float4* sorted_xq, const float2* sorted_lj_comb, LTMatrix3 cell,
    float cutoff, float4* frc_xyz, float pme_beta)
{
    constexpr int max_super_cluster_atoms = kClusterSize * kSuperClusterClusters;
    constexpr int max_block_warps = 2;
    const int sci = blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (sci >= sci_numbers || tid >= max_super_cluster_atoms)
    {
        return;
    }

    const SpongeSciPOD sci_entry = sci_entries[sci];
    const int super_i = sci_entry.supercluster_id;
    const int cluster_i_start = super_cluster_offsets[super_i];
    const int sorted_super_i_base = super_i * max_super_cluster_atoms;
    const int record_begin = record_offsets[sci];
    const int record_end = record_offsets[sci + 1];
    const float cutoff_sq = cutoff * cutoff;
    const float beta2 = pme_beta * pme_beta;
    const float beta3 = beta2 * pme_beta;
    constexpr float min_distance_sq = 3.82e-07f;

    __shared__ float4 shared_i_xq[max_super_cluster_atoms];
    __shared__ float2 shared_i_lj_comb[max_super_cluster_atoms];
    __shared__ Vec3 shared_sci_shift_vec;

    const int i_lane = threadIdx.x;
    const int j_lane = threadIdx.y;
    const int lane = tid & (warpSize - 1);
    const int warp_id = tid / warpSize;
    const int i_slot = j_lane * kClusterSize + i_lane;
    const int warp_j_base = warp_id * kSplitJClusterSize;
    const int warp_j_local = j_lane - warp_j_base;

#define SPONGE_REPLAY_COMPACT_DENSEI_I_LOCAL_LIST(OP) \
    OP(0)                                             \
    OP(1)                                             \
    OP(2)                                             \
    OP(3)                                             \
    OP(4)                                             \
    OP(5)                                             \
    OP(6)                                             \
    OP(7)

#define SPONGE_REPLAY_COMPACT_DENSEI_DECLARE_FCI(I) \
    float fci_x_##I = 0.0f;                         \
    float fci_y_##I = 0.0f;                         \
    float fci_z_##I = 0.0f;
    SPONGE_REPLAY_COMPACT_DENSEI_I_LOCAL_LIST(
        SPONGE_REPLAY_COMPACT_DENSEI_DECLARE_FCI)
#undef SPONGE_REPLAY_COMPACT_DENSEI_DECLARE_FCI

    const int sorted_i = sorted_super_i_base + j_lane * kClusterSize + i_lane;
    shared_i_xq[i_slot] = LoadReadOnly(dense_i_xq + sorted_i);
    shared_i_lj_comb[i_slot] = LoadReadOnly(dense_i_lj_comb + sorted_i);

    if (tid == 0)
    {
        shared_sci_shift_vec = ShiftVectorFromId(sci_entry.shift_id, cell);
    }
    __syncthreads();
    const Vec3 sci_shift_vec = shared_sci_shift_vec;
    const bool sci_is_central = sci_entry.shift_id == kCentralShiftId;

    for (int record_idx = record_begin + warp_id; record_idx < record_end;
         record_idx += max_block_warps)
    {
        const SpongeWarpJRecordPOD* record = records + record_idx;
        int sorted_j_base = -1;
        int cluster_j = -1;
        unsigned int imask = 0u;
        unsigned int j_lane_base = 0u;
        if (lane == 0)
        {
            sorted_j_base = record->sorted_j_base;
            cluster_j = record->cluster_j;
            imask = static_cast<unsigned int>(record->imask);
            j_lane_base = static_cast<unsigned int>(record->j_lane_base);
        }
        sorted_j_base = __shfl_sync(0xFFFFFFFFu, sorted_j_base, 0);
        cluster_j = __shfl_sync(0xFFFFFFFFu, cluster_j, 0);
        imask = __shfl_sync(0xFFFFFFFFu, imask, 0);
        j_lane_base = __shfl_sync(0xFFFFFFFFu, j_lane_base, 0);
        if (sorted_j_base < 0 || imask == 0u)
        {
            continue;
        }

        const unsigned int pair_excl =
            static_cast<unsigned int>(
                record->pair_excl[warp_j_local * kClusterSize + i_lane]);
        unsigned int active_mask = imask & ~pair_excl;
        const int absolute_j_lane = static_cast<int>(j_lane_base) + warp_j_local;
        if (sci_is_central && absolute_j_lane <= i_lane)
        {
            const int central_i = cluster_j - cluster_i_start;
            if (static_cast<unsigned int>(central_i) <
                static_cast<unsigned int>(kSuperClusterClusters))
            {
                active_mask &= ~(1u << static_cast<unsigned int>(central_i));
            }
        }
        if (active_mask == 0u)
        {
            continue;
        }

        const int sorted_j = sorted_j_base + warp_j_local;
        const float4 r2_xq = LoadReadOnly(sorted_xq + sorted_j);
        const float2 r2_lj_comb = LoadReadOnly(sorted_lj_comb + sorted_j);

        const float shifted_j_x = r2_xq.x - sci_shift_vec.x;
        const float shifted_j_y = r2_xq.y - sci_shift_vec.y;
        const float shifted_j_z = r2_xq.z - sci_shift_vec.z;
        const float qj = r2_xq.w;
        const float lj_j_x = r2_lj_comb.x;
        const float lj_j_y = r2_lj_comb.y;

        float fcj_x = 0.0f;
        float fcj_y = 0.0f;
        float fcj_z = 0.0f;

#define SPONGE_REPLAY_COMPACT_DENSEI_ACCUMULATE_I(I)                       \
        if ((active_mask & (1u << static_cast<unsigned int>(I))) != 0u)    \
        {                                                                  \
            const float4 r1_xq = shared_i_xq[(I) * kClusterSize + i_lane]; \
            const float dx = shifted_j_x - r1_xq.x;                        \
            const float dy = shifted_j_y - r1_xq.y;                        \
            const float dz = shifted_j_z - r1_xq.z;                        \
            const float dr2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));         \
            if (dr2 < cutoff_sq && dr2 != 0.0f)                            \
            {                                                              \
                const float r2 = fmaxf(dr2, min_distance_sq);              \
                const float inv_r = rsqrtf(r2);                            \
                const float inv_r2 = inv_r * inv_r;                        \
                const float inv_r6 = inv_r2 * inv_r2 * inv_r2;             \
                const float charge_product = r1_xq.w * qj;                 \
                const float2 ljcp_i =                                      \
                    shared_i_lj_comb[(I) * kClusterSize + i_lane];         \
                const float c6 = ljcp_i.x * lj_j_x;                        \
                const float c12 = ljcp_i.y * lj_j_y;                       \
                const float lj_term = fmaf(c12, inv_r6, -c6);              \
                const float lj_f_invr = (inv_r6 * inv_r2) * lj_term;       \
                const float coulomb_corr =                                 \
                    fmaf(PmeCorrF(beta2 * r2), beta3, inv_r * inv_r2);     \
                const float F_invr =                                       \
                    fmaf(-charge_product, coulomb_corr, lj_f_invr);        \
                const float fij_x = F_invr * dx;                           \
                const float fij_y = F_invr * dy;                           \
                const float fij_z = F_invr * dz;                           \
                fci_x_##I += fij_x;                                        \
                fci_y_##I += fij_y;                                        \
                fci_z_##I += fij_z;                                        \
                fcj_x -= fij_x;                                            \
                fcj_y -= fij_y;                                            \
                fcj_z -= fij_z;                                            \
            }                                                              \
        }

        SPONGE_REPLAY_COMPACT_DENSEI_I_LOCAL_LIST(
            SPONGE_REPLAY_COMPACT_DENSEI_ACCUMULATE_I)
#undef SPONGE_REPLAY_COMPACT_DENSEI_ACCUMULATE_I

        const float fcj_component =
            ReduceSubgroupVecToComponent8(fcj_x, fcj_y, fcj_z, i_lane, lane);
        if (i_lane < 3)
        {
            float* frc_j = reinterpret_cast<float*>(frc_xyz + sorted_j);
            atomicAdd(frc_j + i_lane, fcj_component);
        }
    }

#define SPONGE_REPLAY_COMPACT_DENSEI_REDUCE_I(I)                         \
    float reduced_x_##I = fci_x_##I;                                     \
    float reduced_y_##I = fci_y_##I;                                     \
    float reduced_z_##I = fci_z_##I;                                     \
    const float reduced_component_##I =                                  \
        ReduceWarpIToComponent8(reduced_x_##I, reduced_y_##I,            \
                                reduced_z_##I, warp_j_local);            \
    if (warp_j_local < 3)                                                \
    {                                                                    \
        const int sorted_i_local =                                       \
            sorted_super_i_base + (I) * kClusterSize + i_lane;           \
        float* frc_i = reinterpret_cast<float*>(frc_xyz + sorted_i_local); \
        atomicAdd(frc_i + warp_j_local, reduced_component_##I);          \
    }
    SPONGE_REPLAY_COMPACT_DENSEI_I_LOCAL_LIST(
        SPONGE_REPLAY_COMPACT_DENSEI_REDUCE_I)
#undef SPONGE_REPLAY_COMPACT_DENSEI_REDUCE_I

#undef SPONGE_REPLAY_COMPACT_DENSEI_I_LOCAL_LIST
}

__global__ __launch_bounds__(kClusterSize * kSuperClusterClusters, 16)
void SpongeForceOnlyReplayCompactIndexedKernel(
    int sci_numbers, const SpongeSciPOD* sci_entries, const int* record_offsets,
    const SpongeCompactIndexedRecordPOD* records,
    const SpongeCompactIndexedExclPOD* exclusions,
    const int* super_cluster_offsets, const float4* dense_i_xq,
    const float2* dense_i_lj_comb, const float4* sorted_xq,
    const float2* sorted_lj_comb, LTMatrix3 cell, float cutoff,
    float4* frc_xyz, float pme_beta)
{
    constexpr int max_super_cluster_atoms = kClusterSize * kSuperClusterClusters;
    constexpr int max_block_warps = 2;
    const int sci = blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (sci >= sci_numbers || tid >= max_super_cluster_atoms)
    {
        return;
    }

    const SpongeSciPOD sci_entry = sci_entries[sci];
    const int super_i = sci_entry.supercluster_id;
    const int cluster_i_start = super_cluster_offsets[super_i];
    const int sorted_super_i_base = super_i * max_super_cluster_atoms;
    const int record_begin = record_offsets[sci];
    const int record_end = record_offsets[sci + 1];
    const float cutoff_sq = cutoff * cutoff;
    const float beta2 = pme_beta * pme_beta;
    const float beta3 = beta2 * pme_beta;
    constexpr float min_distance_sq = 3.82e-07f;

    __shared__ float4 shared_i_xq[max_super_cluster_atoms];
    __shared__ float2 shared_i_lj_comb[max_super_cluster_atoms];
    __shared__ Vec3 shared_sci_shift_vec;

    const int i_lane = threadIdx.x;
    const int j_lane = threadIdx.y;
    const int lane = tid & (warpSize - 1);
    const int warp_id = tid / warpSize;
    const int i_slot = j_lane * kClusterSize + i_lane;
    const int warp_j_base = warp_id * kSplitJClusterSize;
    const int warp_j_local = j_lane - warp_j_base;

#define SPONGE_REPLAY_COMPACT_INDEXED_I_LOCAL_LIST(OP) \
    OP(0)                                              \
    OP(1)                                              \
    OP(2)                                              \
    OP(3)                                              \
    OP(4)                                              \
    OP(5)                                              \
    OP(6)                                              \
    OP(7)

#define SPONGE_REPLAY_COMPACT_INDEXED_DECLARE_FCI(I) \
    float fci_x_##I = 0.0f;                          \
    float fci_y_##I = 0.0f;                          \
    float fci_z_##I = 0.0f;
    SPONGE_REPLAY_COMPACT_INDEXED_I_LOCAL_LIST(
        SPONGE_REPLAY_COMPACT_INDEXED_DECLARE_FCI)
#undef SPONGE_REPLAY_COMPACT_INDEXED_DECLARE_FCI

    const int sorted_i = sorted_super_i_base + j_lane * kClusterSize + i_lane;
    shared_i_xq[i_slot] = LoadReadOnly(dense_i_xq + sorted_i);
    shared_i_lj_comb[i_slot] = LoadReadOnly(dense_i_lj_comb + sorted_i);
    if (tid == 0)
    {
        shared_sci_shift_vec = ShiftVectorFromId(sci_entry.shift_id, cell);
    }
    __syncthreads();

    const Vec3 sci_shift_vec = shared_sci_shift_vec;
    const bool sci_is_central = sci_entry.shift_id == kCentralShiftId;

    for (int record_idx = record_begin + warp_id; record_idx < record_end;
         record_idx += max_block_warps)
    {
        const SpongeCompactIndexedRecordPOD* record = records + record_idx;
        int sorted_j_base = -1;
        int cluster_j = -1;
        int excl_index = -1;
        unsigned int imask = 0u;
        unsigned int j_lane_base = 0u;
        if (lane == 0)
        {
            sorted_j_base = record->sorted_j_base;
            cluster_j = record->cluster_j;
            excl_index = record->excl_index;
            imask = static_cast<unsigned int>(record->imask);
            j_lane_base = static_cast<unsigned int>(record->j_lane_base);
        }
        sorted_j_base = __shfl_sync(0xFFFFFFFFu, sorted_j_base, 0);
        cluster_j = __shfl_sync(0xFFFFFFFFu, cluster_j, 0);
        excl_index = __shfl_sync(0xFFFFFFFFu, excl_index, 0);
        imask = __shfl_sync(0xFFFFFFFFu, imask, 0);
        j_lane_base = __shfl_sync(0xFFFFFFFFu, j_lane_base, 0);
        if (sorted_j_base < 0 || imask == 0u)
        {
            continue;
        }

        const int central_i = cluster_j - cluster_i_start;
        const bool has_central_self_record =
            sci_is_central &&
            static_cast<unsigned int>(central_i) <
                static_cast<unsigned int>(kSuperClusterClusters);
        const bool use_mask_slow_path =
            excl_index >= 0 || has_central_self_record;

        const int sorted_j = sorted_j_base + warp_j_local;
        const float4 r2_xq = LoadReadOnly(sorted_xq + sorted_j);
        const float2 r2_lj_comb = LoadReadOnly(sorted_lj_comb + sorted_j);
        const float shifted_j_x = r2_xq.x - sci_shift_vec.x;
        const float shifted_j_y = r2_xq.y - sci_shift_vec.y;
        const float shifted_j_z = r2_xq.z - sci_shift_vec.z;
        const float qj = r2_xq.w;
        const float lj_j_x = r2_lj_comb.x;
        const float lj_j_y = r2_lj_comb.y;

        float fcj_x = 0.0f;
        float fcj_y = 0.0f;
        float fcj_z = 0.0f;

#define SPONGE_REPLAY_COMPACT_INDEXED_COMPUTE_I(I)                         \
        {                                                                  \
            const float4 r1_xq =                                           \
                shared_i_xq[(I) * kClusterSize + i_lane];                  \
            const float dx = shifted_j_x - r1_xq.x;                        \
            const float dy = shifted_j_y - r1_xq.y;                        \
            const float dz = shifted_j_z - r1_xq.z;                        \
            const float dr2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));         \
            if (dr2 < cutoff_sq)                                           \
            {                                                              \
                const float r2 = fmaxf(dr2, min_distance_sq);              \
                const float inv_r = rsqrtf(r2);                            \
                const float inv_r2 = inv_r * inv_r;                        \
                const float inv_r6 = inv_r2 * inv_r2 * inv_r2;             \
                const float charge_product = r1_xq.w * qj;                 \
                const float2 ljcp_i =                                      \
                    shared_i_lj_comb[(I) * kClusterSize + i_lane];         \
                const float c6 = ljcp_i.x * lj_j_x;                        \
                const float c12 = ljcp_i.y * lj_j_y;                       \
                const float lj_term = fmaf(c12, inv_r6, -c6);              \
                const float lj_f_invr = (inv_r6 * inv_r2) * lj_term;       \
                const float coulomb_corr =                                 \
                    fmaf(PmeCorrF(beta2 * r2), beta3, inv_r * inv_r2);     \
                const float F_invr =                                       \
                    fmaf(-charge_product, coulomb_corr, lj_f_invr);        \
                const float fij_x = F_invr * dx;                           \
                const float fij_y = F_invr * dy;                           \
                const float fij_z = F_invr * dz;                           \
                fci_x_##I += fij_x;                                        \
                fci_y_##I += fij_y;                                        \
                fci_z_##I += fij_z;                                        \
                fcj_x -= fij_x;                                            \
                fcj_y -= fij_y;                                            \
                fcj_z -= fij_z;                                            \
            }                                                              \
        }

        if (!use_mask_slow_path)
        {
#define SPONGE_REPLAY_COMPACT_INDEXED_FAST_I(I)                         \
            if ((imask & (1u << static_cast<unsigned int>(I))) != 0u)    \
            {                                                            \
                SPONGE_REPLAY_COMPACT_INDEXED_COMPUTE_I(I)              \
            }
            SPONGE_REPLAY_COMPACT_INDEXED_I_LOCAL_LIST(
                SPONGE_REPLAY_COMPACT_INDEXED_FAST_I)
#undef SPONGE_REPLAY_COMPACT_INDEXED_FAST_I
        }
        else
        {
            unsigned int pair_excl = 0u;
            if (excl_index >= 0)
            {
                pair_excl = static_cast<unsigned int>(
                    exclusions[excl_index].pair_excl[warp_j_local][i_lane]);
            }
            const int absolute_j_lane =
                static_cast<int>(j_lane_base) + warp_j_local;
            const bool central_self_lane =
                has_central_self_record && absolute_j_lane <= i_lane;
#define SPONGE_REPLAY_COMPACT_INDEXED_SLOW_I(I)                           \
            {                                                             \
                constexpr unsigned int i_bit_##I =                        \
                    1u << static_cast<unsigned int>(I);                   \
                if ((imask & i_bit_##I) != 0u &&                          \
                    (pair_excl & i_bit_##I) == 0u &&                      \
                    !(central_self_lane && central_i == (I)))             \
                {                                                         \
                    SPONGE_REPLAY_COMPACT_INDEXED_COMPUTE_I(I)           \
                }                                                         \
            }
            SPONGE_REPLAY_COMPACT_INDEXED_I_LOCAL_LIST(
                SPONGE_REPLAY_COMPACT_INDEXED_SLOW_I)
#undef SPONGE_REPLAY_COMPACT_INDEXED_SLOW_I
        }
#undef SPONGE_REPLAY_COMPACT_INDEXED_COMPUTE_I

        const float fcj_component =
            ReduceSubgroupVecToComponent8(fcj_x, fcj_y, fcj_z, i_lane, lane);
        if (i_lane < 3)
        {
            float* frc_j = reinterpret_cast<float*>(frc_xyz + sorted_j);
            atomicAdd(frc_j + i_lane, fcj_component);
        }
    }

#define SPONGE_REPLAY_COMPACT_INDEXED_REDUCE_I(I)                         \
    float reduced_x_##I = fci_x_##I;                                      \
    float reduced_y_##I = fci_y_##I;                                      \
    float reduced_z_##I = fci_z_##I;                                      \
    const float reduced_component_##I =                                   \
        ReduceWarpIToComponent8(reduced_x_##I, reduced_y_##I,             \
                                reduced_z_##I, warp_j_local);             \
    if (warp_j_local < 3)                                                 \
    {                                                                     \
        const int sorted_i_local =                                        \
            sorted_super_i_base + (I) * kClusterSize + i_lane;            \
        float* frc_i = reinterpret_cast<float*>(frc_xyz + sorted_i_local); \
        atomicAdd(frc_i + warp_j_local, reduced_component_##I);           \
    }
    SPONGE_REPLAY_COMPACT_INDEXED_I_LOCAL_LIST(
        SPONGE_REPLAY_COMPACT_INDEXED_REDUCE_I)
#undef SPONGE_REPLAY_COMPACT_INDEXED_REDUCE_I

#undef SPONGE_REPLAY_COMPACT_INDEXED_I_LOCAL_LIST
}
#endif

template <bool need_energy, bool need_virial, bool total_output,
          bool partial_output = false, bool central_direct_output = false,
          bool gated_partial_output = false>
__global__ __launch_bounds__(kClusterSize * kSuperClusterClusters,
                             total_output
                                 ? ((need_energy && need_virial) ? 12 : 14)
                                 : 9)
void SpongeGmxPackedLjCombKernel(
    int sci_numbers, const GromacsSciPOD* sci_entries,
    const GromacsCjPackedPOD* cjpacked_entries,
    const GromacsExclPOD* excl_entries,
    const unsigned int* cluster_local_masks, const int* sorted_atom_ids,
    const int* partial_sci_bases, const unsigned char* atom_partial_flags,
    const float4* sorted_xq, const float2* sorted_lj_comb,
    const float4* shiftvec, float cutoff,
    GmxPackedForceStorage<need_energy, need_virial, total_output>* frc_xyz,
    float pme_beta, int central_shift_id,
    LTMatrix3* atom_virial, float3* shift_force,
    float* atom_energy, float* atom_direct_cf_energy, float* atom_lj_ene,
    LTMatrix3* direct_atom_virial = nullptr,
    float* direct_atom_energy = nullptr,
    float* direct_atom_direct_cf_energy = nullptr,
    float* direct_atom_lj_ene = nullptr)
{
    static_assert(!partial_output || !total_output,
                  "partial output is only meaningful for per-atom output");
    static_assert(!central_direct_output || partial_output,
                  "central direct output requires partial output mode");
    static_assert(!gated_partial_output || partial_output,
                  "gated partial output requires partial output mode");
    static_assert(!gated_partial_output || !central_direct_output,
                  "gated partial output is only implemented for regular partial slots");
    constexpr int max_super_cluster_atoms = kClusterSize * kSuperClusterClusters;
    constexpr int max_block_warps = 2;
    const int sci = blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (sci >= sci_numbers || tid >= max_super_cluster_atoms)
    {
        return;
    }

    const int i_lane = threadIdx.x;
    const int j_lane = threadIdx.y;
    const int lane = tid & (warpSize - 1);
    const int split = tid / warpSize;
    const int split_j_lane = j_lane - split * kSplitJClusterSize;
    const int i_slot = j_lane * kClusterSize + i_lane;

    const GromacsSciPOD sci_entry = sci_entries[sci];
    const int sorted_super_i_base =
        sci_entry.sci * kSuperClusterClusters * kClusterSize;
    const float4 shift = shiftvec[sci_entry.shift];
    const float cutoff_sq = cutoff * cutoff;
    const float beta2 = pme_beta * pme_beta;
    const float beta3 = beta2 * pme_beta;
    constexpr float min_distance_sq = 3.82e-07f;
    const bool doCalcShift = need_virial && (sci_entry.shift != central_shift_id);
    const bool directCentralOutput =
        central_direct_output && sci_entry.shift == central_shift_id;

#define SPONGE_REPLAY_GMXPACKED_I_LOCAL_LIST(OP) \
    OP(0)                                        \
    OP(1)                                        \
    OP(2)                                        \
    OP(3)                                        \
    OP(4)                                        \
    OP(5)                                        \
    OP(6)                                        \
    OP(7)

#define SPONGE_REPLAY_GMXPACKED_JM_LIST(OP) \
    OP(0)                                    \
    OP(1)                                    \
    OP(2)                                    \
    OP(3)

    __shared__ GmxPackedPerAtomSharedScratch<total_output> perAtomShared;
    __shared__ GmxPackedCentralDirectScratch<central_direct_output,
                                             max_super_cluster_atoms>
        centralDirectScratch;
    __shared__ float4 shared_i_xq[max_super_cluster_atoms];
    __shared__ float2 shared_i_lj_comb[max_super_cluster_atoms];

#define SPONGE_REPLAY_GMXPACKED_DECLARE_FCI(I) \
    float fci_x_##I = 0.0f;                    \
    float fci_y_##I = 0.0f;                    \
    float fci_z_##I = 0.0f;
    SPONGE_REPLAY_GMXPACKED_I_LOCAL_LIST(
        SPONGE_REPLAY_GMXPACKED_DECLARE_FCI)
#undef SPONGE_REPLAY_GMXPACKED_DECLARE_FCI

    const int sorted_i = sorted_super_i_base + i_slot;
    float4 shifted_i_xq = sorted_xq[sorted_i];
    shifted_i_xq.x += shift.x;
    shifted_i_xq.y += shift.y;
    shifted_i_xq.z += shift.z;
    shared_i_xq[i_slot] = shifted_i_xq;
    shared_i_lj_comb[i_slot] = sorted_lj_comb[sorted_i];
    if constexpr (!total_output)
    {
        perAtomShared.atom_ids[i_slot] = sorted_atom_ids[sorted_i];
    }
    __syncthreads();

    ReplayFullOutputBuffer<total_output, need_energy, kSuperClusterClusters>
        output_buf;
    float fshift_component = 0.0f;

    for (int packed_idx = sci_entry.cjPackedBegin;
         packed_idx < sci_entry.cjPackedEnd; ++packed_idx)
    {
        const unsigned int imask =
            cjpacked_entries[packed_idx].imei[split].imask;
        const int excl_index =
            cjpacked_entries[packed_idx].imei[split].excl_ind;
        if (imask == 0u)
        {
            continue;
        }

        unsigned int pair_bits = 0xffffffffu;
        if (excl_index != 0)
        {
            pair_bits =
                excl_entries[excl_index].pair[split_j_lane * kClusterSize + i_lane];
        }
        const unsigned int effective_mask = imask & pair_bits;

#define SPONGE_REPLAY_GMXPACKED_COMPUTE_I(I)                              \
            {                                                             \
                const float4 r1_xq =                                      \
                    shared_i_xq[(I) * kClusterSize + i_lane];             \
                const float dx = shifted_j_x - r1_xq.x;                   \
                const float dy = shifted_j_y - r1_xq.y;                   \
                const float dz = shifted_j_z - r1_xq.z;                   \
                const float dr2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));    \
                if (dr2 < cutoff_sq)                                      \
                {                                                         \
                    const float r2 = fmaxf(dr2, min_distance_sq);         \
                    const float inv_r = rsqrtf(r2);                       \
                    const float inv_r2 = inv_r * inv_r;                   \
                    const float inv_r6 = inv_r2 * inv_r2 * inv_r2;        \
                    const float beta_dr = pme_beta * (r2 * inv_r);        \
                    const float charge_product = r1_xq.w * qj;            \
                    const float2 ljcp_i =                                 \
                        shared_i_lj_comb[(I) * kClusterSize + i_lane];    \
                    const float c6 = ljcp_i.x * lj_j_x;                   \
                    const float c12 = ljcp_i.y * lj_j_y;                  \
                    float frc_abs =                                       \
                        GetLjForceAbs(inv_r2, inv_r6, c12, c6);           \
                    frc_abs -= GetDirectCoulombForceAbsPmeCorrF(          \
                        charge_product, inv_r, inv_r2, beta2 * r2, beta3);\
                    const float fij_x = frc_abs * dx;                     \
                    const float fij_y = frc_abs * dy;                     \
                    const float fij_z = frc_abs * dz;                     \
                    fci_x_##I += fij_x;                                   \
                    fci_y_##I += fij_y;                                   \
                    fci_z_##I += fij_z;                                   \
                    fcj_x -= fij_x;                                       \
                    fcj_y -= fij_y;                                       \
                    fcj_z -= fij_z;                                       \
                    if constexpr (need_virial && !total_output)           \
                    {                                                     \
                        const LTMatrix3 pair_virial =                     \
                            -ij_factor *                                  \
                            GetVirialFromForceDis(                        \
                                {fij_x, fij_y, fij_z}, {dx, dy, dz});     \
                        output_buf.virial[I] =                            \
                            output_buf.virial[I] + pair_virial;           \
                    }                                                     \
                    if constexpr (need_energy)                            \
                    {                                                     \
                        const float pair_energy_lj =                      \
                            ij_factor * GetLjEnergy(inv_r6, c12, c6);     \
                        const float pair_energy_coulomb =                 \
                            ij_factor * GetDirectCoulombEnergy(           \
                                charge_product, inv_r, beta_dr);          \
                        if constexpr (total_output)                       \
                        {                                                 \
                            output_buf.energy_lj_total += pair_energy_lj; \
                            output_buf.energy_coulomb_total +=            \
                                pair_energy_coulomb;                      \
                        }                                                 \
                        else                                              \
                        {                                                 \
                            output_buf.energy_lj[I] += pair_energy_lj;    \
                            output_buf.energy_coulomb[I] +=               \
                                pair_energy_coulomb;                      \
                        }                                                 \
                    }                                                     \
                }                                                         \
            }

#define SPONGE_REPLAY_GMXPACKED_MASKED_I(I)                              \
            {                                                            \
                const unsigned int packed_bit_##I =                      \
                    base_mask << static_cast<unsigned int>(I);           \
                if ((effective_mask & packed_bit_##I) != 0u)             \
                {                                                        \
                    SPONGE_REPLAY_GMXPACKED_COMPUTE_I(I)                 \
                }                                                        \
            }

#define SPONGE_REPLAY_GMXPACKED_PROCESS_JM(JM)                            \
        {                                                                 \
            constexpr unsigned int jm_mask =                              \
                ((1u << kSuperClusterClusters) - 1u) <<                  \
                ((JM) * kSuperClusterClusters);                           \
            if ((imask & jm_mask) != 0u)                                  \
            {                                                             \
                const int cluster_j = cjpacked_entries[packed_idx].cj[JM]; \
                if (cluster_j >= 0)                                       \
                {                                                         \
                    const int sorted_j = cluster_j * kClusterSize + j_lane; \
                    const float4 r2_xq = sorted_xq[sorted_j];               \
                    const float2 r2_lj_comb =                             \
                        sorted_lj_comb[sorted_j];                          \
                    const float shifted_j_x = r2_xq.x;                    \
                    const float shifted_j_y = r2_xq.y;                    \
                    const float shifted_j_z = r2_xq.z;                    \
                    const float qj = r2_xq.w;                             \
                    const float lj_j_x = r2_lj_comb.x;                   \
                    const float lj_j_y = r2_lj_comb.y;                   \
                    const float ij_factor =                              \
                        ((cluster_local_masks[cluster_j] &               \
                          (1u << static_cast<unsigned int>(j_lane))) != 0u) \
                            ? 1.0f                                       \
                            : 0.5f;                                      \
                                                                          \
                    constexpr unsigned int base_mask =                    \
                        1u << ((JM) * kSuperClusterClusters);             \
                                                                          \
                    float fcj_x = 0.0f;                                  \
                    float fcj_y = 0.0f;                                  \
                    float fcj_z = 0.0f;                                  \
                                                                          \
                    SPONGE_REPLAY_GMXPACKED_I_LOCAL_LIST(                \
                        SPONGE_REPLAY_GMXPACKED_MASKED_I)                \
                                                                          \
                    const float fcj_component =                           \
                        ReduceSubgroupVecToComponent8(fcj_x, fcj_y, fcj_z, \
                                                       i_lane, lane);      \
                    if (i_lane < 3)                                       \
                    {                                                     \
                        float* frc_j =                                    \
                            reinterpret_cast<float*>(frc_xyz + sorted_j); \
                        atomicAdd(frc_j + i_lane, fcj_component);         \
                    }                                                     \
                }                                                         \
            }                                                             \
        }

        SPONGE_REPLAY_GMXPACKED_JM_LIST(SPONGE_REPLAY_GMXPACKED_PROCESS_JM)
#undef SPONGE_REPLAY_GMXPACKED_PROCESS_JM
#undef SPONGE_REPLAY_GMXPACKED_MASKED_I
#undef SPONGE_REPLAY_GMXPACKED_COMPUTE_I
    }

#define SPONGE_REPLAY_GMXPACKED_REDUCE_I(I)                                \
    if constexpr (need_virial && !total_output)                            \
    {                                                                      \
        LTMatrix3 reduced_virial = output_buf.virial[I];                   \
        reduced_virial = ReduceWarpVirialOverJ(reduced_virial,             \
                                               kClusterSize);              \
        if (lane < kClusterSize)                                           \
        {                                                                  \
            const int atom_i =                                             \
                perAtomShared.atom_ids[(I) * kClusterSize + i_lane];       \
            if (atom_i >= 0)                                               \
            {                                                              \
                if constexpr (partial_output)                              \
                {                                                          \
                    const int local_slot = (I) * kClusterSize + i_lane;     \
                    const bool atomUsesPartial =                            \
                        !gated_partial_output ||                            \
                        atom_partial_flags[atom_i] != 0u;                  \
                    if constexpr (gated_partial_output)                     \
                    {                                                      \
                        if (!atomUsesPartial)                               \
                        {                                                  \
                            AtomicAddVirial(direct_atom_virial + atom_i,   \
                                            reduced_virial);               \
                        }                                                  \
                        else                                               \
                        {                                                  \
                            const int partial_slot =                       \
                                (sci * kWarpSplitCount + split) *          \
                                    max_super_cluster_atoms +              \
                                local_slot;                                \
                            atom_virial[partial_slot] = reduced_virial;    \
                        }                                                  \
                    }                                                      \
                    else if constexpr (central_direct_output)               \
                    {                                                      \
                        if (directCentralOutput)                            \
                        {                                                  \
                            centralDirectScratch                           \
                                .virial[split * max_super_cluster_atoms +  \
                                        local_slot] = reduced_virial;       \
                        }                                                  \
                        else                                               \
                        {                                                  \
                            const int partial_slot =                       \
                                partial_sci_bases[sci] +                   \
                                split * max_super_cluster_atoms +           \
                                local_slot;                                \
                            atom_virial[partial_slot] = reduced_virial;    \
                        }                                                  \
                    }                                                      \
                    else                                                   \
                    {                                                      \
                        const int partial_slot =                           \
                            (sci * kWarpSplitCount + split) *              \
                                max_super_cluster_atoms +                  \
                            local_slot;                                    \
                        atom_virial[partial_slot] = reduced_virial;        \
                    }                                                      \
                }                                                          \
                else                                                       \
                {                                                          \
                    AtomicAddVirial(atom_virial + atom_i, reduced_virial); \
                }                                                          \
            }                                                              \
        }                                                                  \
    }                                                                      \
    float reduced_x_##I = fci_x_##I;                                       \
    float reduced_y_##I = fci_y_##I;                                       \
    float reduced_z_##I = fci_z_##I;                                       \
    if constexpr (!total_output)                                           \
    {                                                                      \
        ReduceWarpOverJ(reduced_x_##I, reduced_y_##I,                      \
                        reduced_z_##I, kClusterSize);                      \
        if (lane < kClusterSize)                                           \
        {                                                                  \
            const int atom_i =                                             \
                perAtomShared.atom_ids[(I) * kClusterSize + i_lane];       \
            if (atom_i >= 0)                                               \
            {                                                              \
                const int sorted_i_local = sorted_super_i_base +           \
                                           (I) * kClusterSize + i_lane;    \
                float* frc_i =                                             \
                    reinterpret_cast<float*>(frc_xyz + sorted_i_local);    \
                atomicAdd(frc_i + 0, reduced_x_##I);                       \
                atomicAdd(frc_i + 1, reduced_y_##I);                       \
                atomicAdd(frc_i + 2, reduced_z_##I);                       \
            }                                                              \
        }                                                                  \
    }                                                                      \
    else                                                                   \
    {                                                                      \
        const float reduced_component_##I =                                \
            ReduceWarpIToComponent8(reduced_x_##I, reduced_y_##I,          \
                                    reduced_z_##I, split_j_lane);          \
        if (split_j_lane < 3)                                              \
        {                                                                  \
            const int sorted_i_local = sorted_super_i_base +               \
                                       (I) * kClusterSize + i_lane;        \
            float* frc_i =                                                 \
                reinterpret_cast<float*>(frc_xyz + sorted_i_local);        \
            atomicAdd(frc_i + split_j_lane, reduced_component_##I);        \
            if constexpr (need_virial)                                     \
            {                                                              \
                if (doCalcShift)                                           \
                {                                                          \
                    fshift_component += reduced_component_##I;             \
                }                                                          \
            }                                                              \
        }                                                                  \
    }                                                                      \
    if constexpr (need_energy && !total_output)                            \
    {                                                                      \
        float reduced_lj = ReduceWarpFloatOverJ(output_buf.energy_lj[I],   \
                                                kClusterSize);             \
        float reduced_coulomb =                                            \
            ReduceWarpFloatOverJ(output_buf.energy_coulomb[I],             \
                                 kClusterSize);                            \
        if (lane < kClusterSize)                                           \
        {                                                                  \
            const int atom_i =                                             \
                perAtomShared.atom_ids[(I) * kClusterSize + i_lane];       \
            if (atom_i >= 0)                                               \
            {                                                              \
                if constexpr (partial_output)                              \
                {                                                          \
                    const int local_slot = (I) * kClusterSize + i_lane;     \
                    const bool atomUsesPartial =                            \
                        !gated_partial_output ||                            \
                        atom_partial_flags[atom_i] != 0u;                  \
                    if constexpr (gated_partial_output)                     \
                    {                                                      \
                        if (!atomUsesPartial)                               \
                        {                                                  \
                            atomicAdd(direct_atom_energy + atom_i,         \
                                      reduced_lj + reduced_coulomb);        \
                            atomicAdd(direct_atom_lj_ene + atom_i,         \
                                      reduced_lj);                         \
                            atomicAdd(direct_atom_direct_cf_energy +        \
                                          atom_i,                           \
                                      reduced_coulomb);                    \
                        }                                                  \
                        else                                               \
                        {                                                  \
                            const int partial_slot =                       \
                                (sci * kWarpSplitCount + split) *          \
                                    max_super_cluster_atoms +              \
                                local_slot;                                \
                            atom_energy[partial_slot] =                    \
                                reduced_lj + reduced_coulomb;              \
                            atom_lj_ene[partial_slot] = reduced_lj;        \
                            atom_direct_cf_energy[partial_slot] =           \
                                reduced_coulomb;                           \
                        }                                                  \
                    }                                                      \
                    else if constexpr (central_direct_output)               \
                    {                                                      \
                        if (directCentralOutput)                            \
                        {                                                  \
                            const int scratch_slot =                        \
                                split * max_super_cluster_atoms +           \
                                local_slot;                                \
                            centralDirectScratch.energy[scratch_slot] =     \
                                reduced_lj + reduced_coulomb;              \
                            centralDirectScratch.lj_energy[scratch_slot] =  \
                                reduced_lj;                                \
                            centralDirectScratch                            \
                                .direct_cf_energy[scratch_slot] =           \
                                reduced_coulomb;                           \
                        }                                                  \
                        else                                               \
                        {                                                  \
                            const int partial_slot =                       \
                                partial_sci_bases[sci] +                   \
                                split * max_super_cluster_atoms +           \
                                local_slot;                                \
                            atom_energy[partial_slot] =                    \
                                reduced_lj + reduced_coulomb;              \
                            atom_lj_ene[partial_slot] = reduced_lj;        \
                            atom_direct_cf_energy[partial_slot] =           \
                                reduced_coulomb;                           \
                        }                                                  \
                    }                                                      \
                    else                                                   \
                    {                                                      \
                        const int partial_slot =                           \
                            (sci * kWarpSplitCount + split) *              \
                                max_super_cluster_atoms +                  \
                            local_slot;                                    \
                        atom_energy[partial_slot] =                        \
                            reduced_lj + reduced_coulomb;                  \
                        atom_lj_ene[partial_slot] = reduced_lj;            \
                        atom_direct_cf_energy[partial_slot] =              \
                            reduced_coulomb;                               \
                    }                                                      \
                }                                                          \
                else                                                       \
                {                                                          \
                    atomicAdd(atom_energy + atom_i,                        \
                              reduced_lj + reduced_coulomb);               \
                    atomicAdd(atom_lj_ene + atom_i, reduced_lj);           \
                    atomicAdd(atom_direct_cf_energy + atom_i,              \
                              reduced_coulomb);                            \
                }                                                          \
            }                                                              \
        }                                                                  \
    }
    SPONGE_REPLAY_GMXPACKED_I_LOCAL_LIST(SPONGE_REPLAY_GMXPACKED_REDUCE_I)

#define SPONGE_REPLAY_GMXPACKED_COMBINE_CENTRAL_I(I)                       \
    if (split == 0 && lane < kClusterSize)                                  \
    {                                                                      \
        const int local_slot = (I) * kClusterSize + i_lane;                 \
        const int atom_i = perAtomShared.atom_ids[local_slot];              \
        if (atom_i >= 0)                                                    \
        {                                                                  \
            if constexpr (need_virial)                                      \
            {                                                              \
                LTMatrix3 virial = centralDirectScratch.virial[local_slot]; \
                const LTMatrix3 other = centralDirectScratch.virial[        \
                    max_super_cluster_atoms + local_slot];                  \
                virial.a11 += other.a11;                                   \
                virial.a21 += other.a21;                                   \
                virial.a22 += other.a22;                                   \
                virial.a31 += other.a31;                                   \
                virial.a32 += other.a32;                                   \
                virial.a33 += other.a33;                                   \
                direct_atom_virial[atom_i] = virial;                       \
            }                                                              \
            if constexpr (need_energy)                                      \
            {                                                              \
                direct_atom_energy[atom_i] =                                \
                    centralDirectScratch.energy[local_slot] +               \
                    centralDirectScratch                                    \
                        .energy[max_super_cluster_atoms + local_slot];      \
                direct_atom_lj_ene[atom_i] =                                \
                    centralDirectScratch.lj_energy[local_slot] +            \
                    centralDirectScratch                                    \
                        .lj_energy[max_super_cluster_atoms + local_slot];   \
                direct_atom_direct_cf_energy[atom_i] =                      \
                    centralDirectScratch.direct_cf_energy[local_slot] +     \
                    centralDirectScratch.direct_cf_energy[                  \
                        max_super_cluster_atoms + local_slot];              \
            }                                                              \
        }                                                                  \
    }

    if constexpr (partial_output && central_direct_output)
    {
        if (directCentralOutput)
        {
            __syncthreads();
            SPONGE_REPLAY_GMXPACKED_I_LOCAL_LIST(
                SPONGE_REPLAY_GMXPACKED_COMBINE_CENTRAL_I)
            __syncthreads();
        }
    }
#undef SPONGE_REPLAY_GMXPACKED_COMBINE_CENTRAL_I
#undef SPONGE_REPLAY_GMXPACKED_REDUCE_I

    if constexpr (total_output)
    {
        float reduced_total_energy_lj = 0.0f;
        float reduced_total_energy_coulomb = 0.0f;
        if constexpr (need_energy)
        {
            reduced_total_energy_lj =
                ReduceWarpFloatAll(output_buf.energy_lj_total);
            reduced_total_energy_coulomb =
                ReduceWarpFloatAll(output_buf.energy_coulomb_total);
        }
        if constexpr (need_virial)
        {
            if (doCalcShift && split_j_lane < 3)
            {
                const float reduced_fshift =
                    ReduceSubgroupFloat8(fshift_component, lane);
                if (i_lane == 0)
                {
                    atomicAdd(&shift_force[sci_entry.shift].x + split_j_lane,
                              reduced_fshift);
                }
            }
        }
        if constexpr (need_energy)
        {
            if (lane == 0)
            {
                atomicAdd(atom_energy,
                          reduced_total_energy_lj + reduced_total_energy_coulomb);
                atomicAdd(atom_lj_ene, reduced_total_energy_lj);
                atomicAdd(atom_direct_cf_energy, reduced_total_energy_coulomb);
            }
        }
    }

#undef SPONGE_REPLAY_GMXPACKED_JM_LIST
#undef SPONGE_REPLAY_GMXPACKED_I_LOCAL_LIST
}

template <bool need_energy, bool need_virial, bool accumulate_output = false>
__global__ void ReduceGmxPackedPartialOutputsKernel(
    int atom_count, const int* atom_partial_offsets, const int* atom_partial_slots,
    const LTMatrix3* partial_virial, const float* partial_energy,
    const float* partial_direct_cf_energy, const float* partial_lj_energy,
    LTMatrix3* atom_virial, float* atom_energy, float* atom_direct_cf_energy,
    float* atom_lj_energy)
{
    const int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= atom_count)
    {
        return;
    }
    const int begin = atom_partial_offsets[atom];
    const int end = atom_partial_offsets[atom + 1];
    if constexpr (need_virial)
    {
        LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int entry = begin; entry < end; ++entry)
        {
            const LTMatrix3 value = partial_virial[atom_partial_slots[entry]];
            virial.a11 += value.a11;
            virial.a21 += value.a21;
            virial.a22 += value.a22;
            virial.a31 += value.a31;
            virial.a32 += value.a32;
            virial.a33 += value.a33;
        }
        if constexpr (accumulate_output)
        {
            LTMatrix3 current = atom_virial[atom];
            current.a11 += virial.a11;
            current.a21 += virial.a21;
            current.a22 += virial.a22;
            current.a31 += virial.a31;
            current.a32 += virial.a32;
            current.a33 += virial.a33;
            atom_virial[atom] = current;
        }
        else
        {
            atom_virial[atom] = virial;
        }
    }
    if constexpr (need_energy)
    {
        float energy = 0.0f;
        float direct = 0.0f;
        float lj = 0.0f;
        for (int entry = begin; entry < end; ++entry)
        {
            const int slot = atom_partial_slots[entry];
            energy += partial_energy[slot];
            direct += partial_direct_cf_energy[slot];
            lj += partial_lj_energy[slot];
        }
        if constexpr (accumulate_output)
        {
            atom_energy[atom] += energy;
            atom_direct_cf_energy[atom] += direct;
            atom_lj_energy[atom] += lj;
        }
        else
        {
            atom_energy[atom] = energy;
            atom_direct_cf_energy[atom] = direct;
            atom_lj_energy[atom] = lj;
        }
    }
}

template <bool need_energy, bool need_virial, bool accumulate_output,
          int entry_count>
__global__ void ReduceGmxPackedPartialOutputsFixedKernel(
    int reduce_atom_count, const int* reduce_atoms, const int* reduce_slots,
    const LTMatrix3* partial_virial, const float* partial_energy,
    const float* partial_direct_cf_energy, const float* partial_lj_energy,
    LTMatrix3* atom_virial, float* atom_energy, float* atom_direct_cf_energy,
    float* atom_lj_energy)
{
    const int reduce_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (reduce_idx >= reduce_atom_count)
    {
        return;
    }
    const int atom = reduce_atoms[reduce_idx];
    const int slot_base = reduce_idx * entry_count;
    if constexpr (need_virial)
    {
        LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int entry = 0; entry < entry_count; ++entry)
        {
            const LTMatrix3 value =
                partial_virial[reduce_slots[slot_base + entry]];
            virial.a11 += value.a11;
            virial.a21 += value.a21;
            virial.a22 += value.a22;
            virial.a31 += value.a31;
            virial.a32 += value.a32;
            virial.a33 += value.a33;
        }
        if constexpr (accumulate_output)
        {
            LTMatrix3 current = atom_virial[atom];
            current.a11 += virial.a11;
            current.a21 += virial.a21;
            current.a22 += virial.a22;
            current.a31 += virial.a31;
            current.a32 += virial.a32;
            current.a33 += virial.a33;
            atom_virial[atom] = current;
        }
        else
        {
            atom_virial[atom] = virial;
        }
    }
    if constexpr (need_energy)
    {
        float energy = 0.0f;
        float direct = 0.0f;
        float lj = 0.0f;
#pragma unroll
        for (int entry = 0; entry < entry_count; ++entry)
        {
            const int slot = reduce_slots[slot_base + entry];
            energy += partial_energy[slot];
            direct += partial_direct_cf_energy[slot];
            lj += partial_lj_energy[slot];
        }
        if constexpr (accumulate_output)
        {
            atom_energy[atom] += energy;
            atom_direct_cf_energy[atom] += direct;
            atom_lj_energy[atom] += lj;
        }
        else
        {
            atom_energy[atom] = energy;
            atom_direct_cf_energy[atom] = direct;
            atom_lj_energy[atom] = lj;
        }
    }
}

template <bool need_energy, bool need_virial, bool accumulate_output = false>
__global__ void ReduceGmxPackedPartialOutputsGenericAtomsKernel(
    int reduce_atom_count, const int* reduce_atoms,
    const int* atom_partial_offsets, const int* atom_partial_slots,
    const LTMatrix3* partial_virial, const float* partial_energy,
    const float* partial_direct_cf_energy, const float* partial_lj_energy,
    LTMatrix3* atom_virial, float* atom_energy, float* atom_direct_cf_energy,
    float* atom_lj_energy)
{
    const int reduce_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (reduce_idx >= reduce_atom_count)
    {
        return;
    }
    const int atom = reduce_atoms[reduce_idx];
    const int begin = atom_partial_offsets[atom];
    const int end = atom_partial_offsets[atom + 1];
    if constexpr (need_virial)
    {
        LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int entry = begin; entry < end; ++entry)
        {
            const LTMatrix3 value = partial_virial[atom_partial_slots[entry]];
            virial.a11 += value.a11;
            virial.a21 += value.a21;
            virial.a22 += value.a22;
            virial.a31 += value.a31;
            virial.a32 += value.a32;
            virial.a33 += value.a33;
        }
        if constexpr (accumulate_output)
        {
            LTMatrix3 current = atom_virial[atom];
            current.a11 += virial.a11;
            current.a21 += virial.a21;
            current.a22 += virial.a22;
            current.a31 += virial.a31;
            current.a32 += virial.a32;
            current.a33 += virial.a33;
            atom_virial[atom] = current;
        }
        else
        {
            atom_virial[atom] = virial;
        }
    }
    if constexpr (need_energy)
    {
        float energy = 0.0f;
        float direct = 0.0f;
        float lj = 0.0f;
        for (int entry = begin; entry < end; ++entry)
        {
            const int slot = atom_partial_slots[entry];
            energy += partial_energy[slot];
            direct += partial_direct_cf_energy[slot];
            lj += partial_lj_energy[slot];
        }
        if constexpr (accumulate_output)
        {
            atom_energy[atom] += energy;
            atom_direct_cf_energy[atom] += direct;
            atom_lj_energy[atom] += lj;
        }
        else
        {
            atom_energy[atom] = energy;
            atom_direct_cf_energy[atom] = direct;
            atom_lj_energy[atom] = lj;
        }
    }
}

template <typename ForceTarget>
__device__ inline void AtomicAddProductionForceComponent(ForceTarget* frc,
                                                         int atom,
                                                         int component,
                                                         float value)
{
    float* force_components = reinterpret_cast<float*>(frc + atom);
    atomicAdd(force_components + component, value);
}

// The default production-gmxpacked microbench path instantiates the production
// warp-record evaluator below. These adapters preserve its helper contract
// without linking the full Lennard_Jones_force host translation unit.
static __host__ __device__ __forceinline__ int
Clustered_Gmxpacked_Get_LJ_Type_MinMax(int a, int b)
{
    const int hi = a > b ? a : b;
    const int lo = a > b ? b : a;
    return (hi * (hi + 1) >> 1) + lo;
}

template <typename T>
static __device__ __forceinline__ T Clustered_Load_ReadOnly(const T* ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

static __host__ __device__ __forceinline__ VECTOR
Clustered_Shift_Vector_From_Id(int shift_id, LTMatrix3 cell)
{
    const Vec3 shift = ShiftVectorFromId(shift_id, cell);
    return VECTOR(shift.x, shift.y, shift.z);
}

static __host__ __device__ __forceinline__ int
Clustered_Get_Pair_Shift_Id(uint64_t packed_shift_bits, int i_local)
{
    return GetPairShiftId(packed_shift_bits, i_local);
}

template <bool enabled, int size>
struct Clustered_Energy_Buffer
{
    float unused = 0.0f;

    __device__ __forceinline__ float& operator[](int)
    {
        return unused;
    }
};

template <int size>
struct Clustered_Energy_Buffer<true, size>
{
    float values[size];

    __device__ __forceinline__ void Clear()
    {
        for (int i = 0; i < size; ++i)
        {
            values[i] = 0.0f;
        }
    }

    __device__ __forceinline__ float& operator[](int idx)
    {
        return values[idx];
    }
};

template <bool total_output, bool need_energy, int size>
struct Clustered_Full_Record_Output_Buffer;

template <bool need_energy, int size>
struct Clustered_Full_Record_Output_Buffer<false, need_energy, size>
{
    Clustered_Energy_Buffer<need_energy, size> energy_lj;
    Clustered_Energy_Buffer<need_energy, size> energy_coulomb;
    LTMatrix3 virial[size];

    __device__ __forceinline__ Clustered_Full_Record_Output_Buffer(
        bool store_energy, bool store_virial)
    {
        if constexpr (need_energy)
        {
            if (store_energy)
            {
                energy_lj.Clear();
                energy_coulomb.Clear();
            }
        }
        if (store_virial)
        {
            for (int i = 0; i < size; ++i)
            {
                virial[i] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            }
        }
    }
};

template <bool need_energy, int size>
struct Clustered_Full_Record_Output_Buffer<true, need_energy, size>
{
    float energy_lj_total;
    float energy_coulomb_total;
    LTMatrix3 virial_total;

    __device__ __forceinline__ Clustered_Full_Record_Output_Buffer(
        bool store_energy, bool store_virial)
    {
        if constexpr (need_energy)
        {
            if (store_energy)
            {
                energy_lj_total = 0.0f;
                energy_coulomb_total = 0.0f;
            }
        }
        if (store_virial)
        {
            virial_total = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
};

static __device__ __forceinline__ unsigned int
Clustered_Microbench_Subgroup_Mask(int lane, int subgroup_width)
{
    const unsigned int subgroup =
        static_cast<unsigned int>(lane / subgroup_width);
    const unsigned int width_mask =
        (1u << static_cast<unsigned int>(subgroup_width)) - 1u;
    return width_mask << (subgroup * static_cast<unsigned int>(subgroup_width));
}

static __device__ __forceinline__ float
Reduce_Clustered_Subgroup_Vector_To_Component(float x, float y, float z,
                                               int component_lane, int lane,
                                               int subgroup_width)
{
    const unsigned int mask =
        Clustered_Microbench_Subgroup_Mask(lane, subgroup_width);
    for (int delta = subgroup_width >> 1; delta > 0; delta >>= 1)
    {
        x += __shfl_down_sync(mask, x, delta, subgroup_width);
        y += __shfl_down_sync(mask, y, delta, subgroup_width);
        z += __shfl_down_sync(mask, z, delta, subgroup_width);
    }
    const int leader = lane - lane % subgroup_width;
    x = __shfl_sync(mask, x, leader, warpSize);
    y = __shfl_sync(mask, y, leader, warpSize);
    z = __shfl_sync(mask, z, leader, warpSize);
    return component_lane == 0 ? x : (component_lane == 1 ? y : z);
}

static __device__ __forceinline__ float
Reduce_Clustered_Warp_I_To_Component(float x, float y, float z, int i_lane,
                                      int component_lane, int subgroup_width)
{
    for (int delta = warpSize >> 1; delta >= subgroup_width; delta >>= 1)
    {
        x += __shfl_down_sync(kFullMask, x, delta, warpSize);
        y += __shfl_down_sync(kFullMask, y, delta, warpSize);
        z += __shfl_down_sync(kFullMask, z, delta, warpSize);
    }
    x = __shfl_sync(kFullMask, x, i_lane, warpSize);
    y = __shfl_sync(kFullMask, y, i_lane, warpSize);
    z = __shfl_sync(kFullMask, z, i_lane, warpSize);
    return component_lane == 0 ? x : (component_lane == 1 ? y : z);
}

static __device__ __forceinline__ void Clustered_Atomic_Add_Force_Component(
    VECTOR* frc, int atom_index, int component, float value)
{
    float* force_component = reinterpret_cast<float*>(frc + atom_index);
    atomicAdd(force_component + component, value);
}

static __device__ __forceinline__ float Reduce_Clustered_Warp_Float_Over_J(
    float value, int subgroup_width)
{
    for (int delta = warpSize >> 1; delta >= subgroup_width; delta >>= 1)
    {
        value += __shfl_down_sync(kFullMask, value, delta, warpSize);
    }
    return value;
}

static __device__ __forceinline__ LTMatrix3
Reduce_Clustered_Warp_Virial_Over_J(LTMatrix3 value, int subgroup_width)
{
    for (int delta = warpSize >> 1; delta >= subgroup_width; delta >>= 1)
    {
        value.a11 +=
            __shfl_down_sync(kFullMask, value.a11, delta, warpSize);
        value.a21 +=
            __shfl_down_sync(kFullMask, value.a21, delta, warpSize);
        value.a22 +=
            __shfl_down_sync(kFullMask, value.a22, delta, warpSize);
        value.a31 +=
            __shfl_down_sync(kFullMask, value.a31, delta, warpSize);
        value.a32 +=
            __shfl_down_sync(kFullMask, value.a32, delta, warpSize);
        value.a33 +=
            __shfl_down_sync(kFullMask, value.a33, delta, warpSize);
    }
    return value;
}

static __device__ __forceinline__ float4 Pack_Clustered_Virial_Lo(
    LTMatrix3 value)
{
    return PackVirialLo(value);
}

static __device__ __forceinline__ float2 Pack_Clustered_Virial_Hi(
    LTMatrix3 value)
{
    return PackVirialHi(value);
}

static __device__ __forceinline__ LTMatrix3 Unpack_Clustered_Virial(
    float4 lo, float2 hi)
{
    return UnpackVirial(lo, hi);
}

static __device__ __forceinline__ void atomicAdd(LTMatrix3* target,
                                                  LTMatrix3 value)
{
    ::atomicAdd(&target->a11, value.a11);
    ::atomicAdd(&target->a21, value.a21);
    ::atomicAdd(&target->a22, value.a22);
    ::atomicAdd(&target->a31, value.a31);
    ::atomicAdd(&target->a32, value.a32);
    ::atomicAdd(&target->a33, value.a33);
}

static __device__ __forceinline__ float atomicAdd(float* target, float value)
{
    return ::atomicAdd(target, value);
}

static __device__ __forceinline__ float Get_Clustered_LJ_Force_Abs(
    float inv_r2, float inv_r6, float A, float B)
{
    return GetLjForceAbs(inv_r2, inv_r6, A, B);
}

static __device__ __forceinline__ float Get_Clustered_LJ_Energy(
    float inv_r6, float A, float B)
{
    return GetLjEnergy(inv_r6, A, B);
}

static __device__ __forceinline__ float Get_Clustered_Direct_Coulomb_Energy(
    float charge_product, float inv_r, float beta_dr)
{
    return GetDirectCoulombEnergy(charge_product, inv_r, beta_dr);
}

static __device__ __forceinline__ float
Get_Clustered_Direct_Coulomb_Force_Abs_PME_Corr(
    float charge_product, float inv_r, float inv_r2, float beta2_r2,
    float beta3)
{
    return GetDirectCoulombForceAbsPmeCorrF(
        charge_product, inv_r, inv_r2, beta2_r2, beta3);
}

#include "Lennard_Jones_force/clustered_lj_warp_record_kernel.cuh"

static_assert(sizeof(SpongeGmxpackedSciPOD) ==
              sizeof(LJ_CLUSTERED_GMXPACKED_SCI));
static_assert(sizeof(SpongeGmxpackedCjPOD) ==
              sizeof(LJ_CLUSTERED_GMXPACKED_CJ));
static_assert(sizeof(SpongeGmxpackedExclusionPOD) ==
              sizeof(LJ_CLUSTERED_GMXPACKED_EXCLUSION));

template <typename ForceTarget>
__device__ inline void StoreProductionForceComponent(ForceTarget* frc,
                                                     int atom,
                                                     int component,
                                                     float value)
{
    volatile float* force_components =
        reinterpret_cast<volatile float*>(frc + atom);
    force_components[component] = value;
}

template <bool sci_shift_only>
__global__ __launch_bounds__(kClusterSize * kSuperClusterClusters)
void SpongeProductionGmxpackedCutoffMaskKernel(
    int sci_numbers, int cluster_numbers,
    const SpongeGmxpackedSciPOD* sci_entries,
    const SpongeGmxpackedCjPOD* cjpacked_entries,
    const SpongeGmxpackedExclusionPOD* exclusion_entries,
    const uint64_t* pair_shift_bits, const int* sci_shift_safe_flags,
    int sci_shift_safe_value, const float4* sorted_xq, LTMatrix3 cell,
    float cutoff, uint64_t* cutoff_pass_masks)
{
    const int sci = blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (sci >= sci_numbers ||
        tid >= kClusterSize * kSuperClusterClusters)
    {
        return;
    }
    if (sci_shift_safe_flags != nullptr &&
        sci_shift_safe_flags[sci] != sci_shift_safe_value)
    {
        return;
    }

    const int i_lane = threadIdx.x;
    const int j_lane = threadIdx.y;
    const int lane = tid & (warpSize - 1);
    const int split = tid / warpSize;
    const int split_j_lane = j_lane - split * kSplitJClusterSize;
    const SpongeGmxpackedSciPOD sci_entry = sci_entries[sci];
    const int cluster_i_start =
        sci_entry.supercluster_id * kSuperClusterClusters;
    const Vec3 sci_shift = ShiftVectorFromId(sci_entry.shift_id, cell);
    const float cutoff_sq = cutoff * cutoff * (1.0f + kOracleCutoffGuard);

    __shared__ float4 shared_i_xq[kClusterSize * kSuperClusterClusters];
    const int i_slot = j_lane * kClusterSize + i_lane;
    const int sorted_i =
        (cluster_i_start + j_lane) * kClusterSize + i_lane;
    float4 i_xq = sorted_xq[sorted_i];
    if constexpr (sci_shift_only)
    {
        i_xq.x += sci_shift.x;
        i_xq.y += sci_shift.y;
        i_xq.z += sci_shift.z;
    }
    shared_i_xq[i_slot] = i_xq;
    __syncthreads();

    for (int packed_idx = sci_entry.cjpacked_begin;
         packed_idx < sci_entry.cjpacked_end; ++packed_idx)
    {
        const SpongeGmxpackedCjPOD* packed = cjpacked_entries + packed_idx;
        const unsigned int imask = packed->split[split].imask;
        unsigned int cutoff_pass_mask = 0u;
        if (imask != 0u)
        {
            const int exclusion_index =
                packed->split[split].exclusion_index;
            unsigned int pair_bits = 0xffffffffu;
            if (exclusion_index != 0 && exclusion_entries != nullptr)
            {
                pair_bits = exclusion_entries[exclusion_index]
                                .pair[split_j_lane * kClusterSize + i_lane];
            }
            const unsigned int effective_mask = imask & pair_bits;
            for (int jm = 0; jm < kJGroupSize; ++jm)
            {
                const unsigned int jm_mask =
                    ((1u << kSuperClusterClusters) - 1u)
                    << (jm * kSuperClusterClusters);
                if ((imask & jm_mask) == 0u)
                {
                    continue;
                }
                const int cluster_j = packed->cj[jm];
                if (cluster_j < 0 || cluster_j >= cluster_numbers)
                {
                    continue;
                }
                uint64_t shift_bits = 0ull;
                if constexpr (!sci_shift_only)
                {
                    shift_bits =
                        pair_shift_bits != nullptr
                            ? pair_shift_bits[packed_idx * kJGroupSize + jm]
                            : 0ull;
                }
                const int sorted_j = cluster_j * kClusterSize + j_lane;
                const float4 r2_xq = sorted_xq[sorted_j];
                for (int i_local = 0; i_local < kSuperClusterClusters;
                     ++i_local)
                {
                    const unsigned int packed_bit =
                        1u << (jm * kSuperClusterClusters + i_local);
                    const bool active_pair =
                        (effective_mask & packed_bit) != 0u;
                    bool cutoff_pass = false;
                    if (active_pair)
                    {
                        const float4 r1_xq =
                            shared_i_xq[i_local * kClusterSize + i_lane];
                        float dx = r2_xq.x - r1_xq.x;
                        float dy = r2_xq.y - r1_xq.y;
                        float dz = r2_xq.z - r1_xq.z;
                        if constexpr (!sci_shift_only)
                        {
                            const Vec3 pair_shift =
                                pair_shift_bits != nullptr
                                    ? ShiftVectorFromId(
                                          GetPairShiftId(shift_bits, i_local),
                                          cell)
                                    : sci_shift;
                            dx -= pair_shift.x;
                            dy -= pair_shift.y;
                            dz -= pair_shift.z;
                        }
                        const float dr2 =
                            fmaf(dx, dx, fmaf(dy, dy, dz * dz));
                        cutoff_pass = dr2 < cutoff_sq;
                    }
                    if (__ballot_sync(kFullMask, cutoff_pass) != 0u)
                    {
                        cutoff_pass_mask |= packed_bit;
                    }
                }
            }
        }
        if (lane == 0)
        {
            unsigned int* cutoff_pass_masks_u32 =
                reinterpret_cast<unsigned int*>(cutoff_pass_masks);
            cutoff_pass_masks_u32[packed_idx * kWarpSplitCount + split] =
                cutoff_pass_mask;
        }
    }
}

template <bool need_energy, bool need_virial, bool total_output,
          bool compact_force_storage, bool sci_shift_only,
          typename ForceTarget, bool virial_from_shift = false,
          bool use_lj_comb = true, bool skip_force_writeback = false,
          bool local_i_mask8 = false, bool skip_empty_effective_jm = false,
          bool dense_no_exclusion_fast = false,
          bool attribution_force_all_i = false,
          bool attribution_no_cutoff_branch = false,
          bool oracle_cutoff_sidecar = false, int sci_work_parts = 1,
          bool contiguous_sci_work = false, bool fused_sits = false,
          bool sits_correction_only = false>
__global__ __launch_bounds__(kClusterSize * kSuperClusterClusters,
                             use_lj_comb
                                 ? (total_output
                                        ? ((need_energy && need_virial) ? 12
                                                                        : 14)
                                        : 9)
                                 : ((need_virial && !total_output &&
                                     sci_work_parts > 1)
                                        ? 10
                                        : (total_output ? 12 : 13)))
void SpongeProductionGmxpackedReplayKernel(
    int sci_numbers, int cluster_numbers, const int* cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const int* super_cluster_offsets,
    const SpongeGmxpackedSciPOD* sci_entries,
    const SpongeGmxpackedCjPOD* cjpacked_entries,
    const SpongeGmxpackedExclusionPOD* exclusion_entries,
    const uint64_t* pair_shift_bits, const int* sci_shift_safe_flags,
    int sci_shift_safe_value, const int* sorted_atom_ids,
    const float4* sorted_xq, const int* sorted_lj_type,
    const float2* sorted_lj_comb, LTMatrix3 cell,
    const float2* lj_ab_packed, float cutoff, ForceTarget* frc,
    float pme_beta, float* atom_energy, LTMatrix3* atom_virial,
    float* atom_direct_cf_energy, float* atom_lj_energy,
    float3* shift_force, bool store_energy = need_energy,
    bool store_virial = need_virial,
    ForceTarget* frc_enhancing = nullptr, int selective_atom_end = 0,
    float pwwp_factor = 0.0f)
{
    static_assert(!sits_correction_only || fused_sits,
                  "SITS correction-only replay requires the SITS output policy");
    static_assert(sci_work_parts > 0, "SCI work partition count must be positive");
    static_assert(sci_work_parts == 1 ||
                      ((!need_energy && !need_virial && !total_output) ||
                       (need_virial && !total_output &&
                        compact_force_storage)),
                  "SCI work partitioning requires force-only output or the "
                  "packed-AB compact full path");
    static_assert(!contiguous_sci_work || sci_work_parts > 1,
                  "contiguous SCI work requires multiple work parts");
    static_assert(!virial_from_shift ||
                      (need_virial && total_output && sci_shift_only),
                  "shift-force virial replay requires fixed-shift total virial");
    (void)cluster_offsets;
    (void)cluster_valid_masks;
    (void)cluster_local_masks;
    (void)super_cluster_offsets;
    static_assert(!virial_from_shift || use_lj_comb,
                  "table replay does not yet support shift-force virial mode");
    if constexpr (use_lj_comb)
    {
        (void)sorted_lj_type;
        (void)lj_ab_packed;
    }
    else
    {
        (void)sorted_lj_comb;
    }
    if constexpr (!fused_sits)
    {
        (void)frc_enhancing;
        (void)selective_atom_end;
        (void)pwwp_factor;
    }
    constexpr int max_super_cluster_atoms = kClusterSize * kSuperClusterClusters;
    constexpr int max_block_warps = 2;
    int sci = 0;
    int sci_work_part = 0;
    if constexpr (sci_work_parts == 1)
    {
        sci = blockIdx.x;
    }
    else
    {
        sci = blockIdx.x / sci_work_parts;
        sci_work_part = blockIdx.x % sci_work_parts;
    }
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (sci >= sci_numbers || tid >= max_super_cluster_atoms)
    {
        return;
    }
    if (sci_shift_safe_flags != nullptr &&
        sci_shift_safe_flags[sci] != sci_shift_safe_value)
    {
        return;
    }

    const int i_lane = threadIdx.x;
    const int j_lane = threadIdx.y;
    const int lane = tid & (warpSize - 1);
    const int split = tid / warpSize;
    const int split_j_lane = j_lane - split * kSplitJClusterSize;
    const int i_slot = j_lane * kClusterSize + i_lane;

    const SpongeGmxpackedSciPOD sci_entry = sci_entries[sci];
    const int cluster_i_start = sci_entry.supercluster_id * kSuperClusterClusters;
    const float cutoff_sq = cutoff * cutoff;
    const float beta2 = pme_beta * pme_beta;
    const float beta3 = beta2 * pme_beta;
    constexpr float min_distance_sq = 3.82e-07f;
    const Vec3 sci_shift = ShiftVectorFromId(sci_entry.shift_id, cell);

#define SPONGE_PROD_GMXPACKED_I_LIST(OP) \
    OP(0)                                \
    OP(1)                                \
    OP(2)                                \
    OP(3)                                \
    OP(4)                                \
    OP(5)                                \
    OP(6)                                \
    OP(7)

#define SPONGE_PROD_GMXPACKED_JM_LIST(OP) \
    OP(0)                                 \
    OP(1)                                 \
    OP(2)                                 \
    OP(3)

    __shared__ float4 shared_i_xq[max_super_cluster_atoms];
    __shared__ int shared_i_lj_type[max_super_cluster_atoms];
    __shared__ float2 shared_i_lj_comb[max_super_cluster_atoms];
    __shared__ float4 shared_total_virial_lo[max_block_warps];
    __shared__ float2 shared_total_virial_hi[max_block_warps];
    __shared__ float shared_total_energy_lj[max_block_warps];
    __shared__ float shared_total_energy_coulomb[max_block_warps];
    __shared__ float4
        shared_split_virial_lo[2][kSuperClusterClusters][kClusterSize];
    __shared__ float2
        shared_split_virial_hi[2][kSuperClusterClusters][kClusterSize];

    const int sorted_i = (cluster_i_start + j_lane) * kClusterSize + i_lane;
    float4 i_xq = sorted_xq[sorted_i];
    if constexpr (sci_shift_only)
    {
        i_xq.x += sci_shift.x;
        i_xq.y += sci_shift.y;
        i_xq.z += sci_shift.z;
    }
    shared_i_xq[i_slot] = i_xq;
    if constexpr (use_lj_comb)
    {
        shared_i_lj_comb[i_slot] = sorted_lj_comb[sorted_i];
    }
    else
    {
        shared_i_lj_type[i_slot] = sorted_lj_type[sorted_i];
    }
    __syncthreads();

    const auto process_packed = [&](auto energy_tag, auto virial_tag) {
    constexpr bool compute_energy =
        need_energy && decltype(energy_tag)::value;
    constexpr bool compute_virial =
        need_virial && decltype(virial_tag)::value;

#define SPONGE_PROD_DECLARE_FCI(I) \
    float fci_x_##I = 0.0f;        \
    float fci_y_##I = 0.0f;        \
    float fci_z_##I = 0.0f;
    SPONGE_PROD_GMXPACKED_I_LIST(SPONGE_PROD_DECLARE_FCI)
#undef SPONGE_PROD_DECLARE_FCI

    ReplayFullOutputBuffer<total_output, compute_energy,
                           kSuperClusterClusters>
        output_buf;
    float fshift_component = 0.0f;

    const int packed_count =
        sci_entry.cjpacked_end - sci_entry.cjpacked_begin;
    const int packed_begin =
        contiguous_sci_work
            ? sci_entry.cjpacked_begin +
                  packed_count * sci_work_part / sci_work_parts
            : sci_entry.cjpacked_begin + sci_work_part;
    const int packed_end =
        contiguous_sci_work
            ? sci_entry.cjpacked_begin +
                  packed_count * (sci_work_part + 1) / sci_work_parts
            : sci_entry.cjpacked_end;
    constexpr int packed_stride =
        contiguous_sci_work ? 1 : sci_work_parts;
    for (int packed_idx = packed_begin; packed_idx < packed_end;
         packed_idx += packed_stride)
    {
        const SpongeGmxpackedCjPOD* packed = cjpacked_entries + packed_idx;
        unsigned int imask = packed->split[split].imask;
        if constexpr (oracle_cutoff_sidecar)
        {
            const uint64_t* oracle_pass_masks =
                reinterpret_cast<const uint64_t*>(shift_force);
            imask &= static_cast<unsigned int>(
                oracle_pass_masks[packed_idx] >> (split * 32));
        }
        if (imask == 0u)
        {
            continue;
        }
        const int exclusion_index = packed->split[split].exclusion_index;
        unsigned int pair_bits = 0xffffffffu;
        if constexpr (sci_shift_only)
        {
            if (exclusion_index != 0)
            {
                pair_bits = exclusion_entries[exclusion_index]
                                .pair[split_j_lane * kClusterSize + i_lane];
            }
        }
        else if (exclusion_index != 0 && exclusion_entries != nullptr)
        {
            pair_bits = exclusion_entries[exclusion_index]
                            .pair[split_j_lane * kClusterSize + i_lane];
        }
        const unsigned int effective_mask = imask & pair_bits;

#define SPONGE_PROD_PAIR_FORCE_BODY(I)                                      \
    {                                                                      \
        const float r2 = fmaxf(dr2, min_distance_sq);                      \
        const float inv_r = rsqrtf(r2);                                    \
        const float inv_r2 = inv_r * inv_r;                                \
        const float inv_r6 = inv_r2 * inv_r2 * inv_r2;                    \
        const float beta_dr = pme_beta * (r2 * inv_r);                    \
        const float charge_product = r1_xq.w * qj;                        \
        float c6 = 0.0f;                                                   \
        float c12 = 0.0f;                                                  \
        if constexpr (use_lj_comb)                                         \
        {                                                                  \
            const float2 ljcp_i =                                          \
                shared_i_lj_comb[(I) * kClusterSize + i_lane];             \
            c6 = ljcp_i.x * lj_j_x;                                        \
            c12 = ljcp_i.y * lj_j_y;                                       \
        }                                                                  \
        else                                                               \
        {                                                                  \
            const int r1_lj_type =                                         \
                shared_i_lj_type[(I) * kClusterSize + i_lane];             \
            const int lj_index = GetLjType(r1_lj_type, r2_lj_type);        \
            const float2 AB = lj_ab_packed[lj_index];                     \
            c12 = AB.x;                                                    \
            c6 = AB.y;                                                     \
        }                                                                  \
        float frc_abs = GetLjForceAbs(inv_r2, inv_r6, c12, c6);           \
        frc_abs -= GetDirectCoulombForceAbsPmeCorrF(                       \
            charge_product, inv_r, inv_r2, beta2 * r2, beta3);             \
        const float fij_x = frc_abs * dx;                                  \
        const float fij_y = frc_abs * dy;                                  \
        const float fij_z = frc_abs * dz;                                  \
        if constexpr (!sits_correction_only)                               \
        {                                                                  \
            fci_x_##I += fij_x;                                            \
            fci_y_##I += fij_y;                                            \
            fci_z_##I += fij_z;                                            \
            fcj_x -= fij_x;                                                \
            fcj_y -= fij_y;                                                \
            fcj_z -= fij_z;                                                \
        }                                                                  \
        if constexpr (fused_sits)                                          \
        {                                                                  \
            const int sorted_i_pair =                                      \
                (cluster_i_start + (I)) * kClusterSize + i_lane;           \
            const int atom_i_pair = sorted_atom_ids[sorted_i_pair];        \
            const int atom_j_pair = sorted_atom_ids[sorted_j];             \
            const int mark_sum =                                           \
                (atom_i_pair < selective_atom_end ? 0 : 1) +              \
                (atom_j_pair < selective_atom_end ? 0 : 1);               \
            const float selective_factor =                                \
                mark_sum == 0 ? 1.0f                                      \
                              : (mark_sum == 1 ? pwwp_factor : 0.0f);      \
            if (atom_i_pair >= 0 && atom_i_pair < selective_atom_end)      \
            {                                                              \
                AtomicAddProductionForceComponent(                         \
                    frc_enhancing, atom_i_pair, 0,                         \
                    selective_factor * fij_x);                             \
                AtomicAddProductionForceComponent(                         \
                    frc_enhancing, atom_i_pair, 1,                         \
                    selective_factor * fij_y);                             \
                AtomicAddProductionForceComponent(                         \
                    frc_enhancing, atom_i_pair, 2,                         \
                    selective_factor * fij_z);                             \
            }                                                              \
            if (atom_j_pair >= 0 && atom_j_pair < selective_atom_end)      \
            {                                                              \
                AtomicAddProductionForceComponent(                         \
                    frc_enhancing, atom_j_pair, 0,                        \
                    -selective_factor * fij_x);                            \
                AtomicAddProductionForceComponent(                         \
                    frc_enhancing, atom_j_pair, 1,                        \
                    -selective_factor * fij_y);                            \
                AtomicAddProductionForceComponent(                         \
                    frc_enhancing, atom_j_pair, 2,                        \
                    -selective_factor * fij_z);                            \
            }                                                              \
        }                                                                  \
        if constexpr (compute_virial && !virial_from_shift)                \
        {                                                                  \
            if constexpr (total_output)                                    \
            {                                                              \
                output_buf.virial_total.a11 -= fij_x * dx;                 \
                output_buf.virial_total.a21 -= fij_x * dy + fij_y * dx;    \
                output_buf.virial_total.a22 -= fij_y * dy;                 \
                output_buf.virial_total.a31 -= fij_x * dz + fij_z * dx;    \
                output_buf.virial_total.a32 -= fij_y * dz + fij_z * dy;    \
                output_buf.virial_total.a33 -= fij_z * dz;                 \
            }                                                              \
            else                                                           \
            {                                                              \
                output_buf.virial[I].a11 -= fij_x * dx;                    \
                output_buf.virial[I].a21 -= fij_x * dy + fij_y * dx;       \
                output_buf.virial[I].a22 -= fij_y * dy;                    \
                output_buf.virial[I].a31 -= fij_x * dz + fij_z * dx;       \
                output_buf.virial[I].a32 -= fij_y * dz + fij_z * dy;       \
                output_buf.virial[I].a33 -= fij_z * dz;                    \
            }                                                              \
        }                                                                  \
        if constexpr (compute_energy)                                      \
        {                                                                  \
            const float pair_lj_energy = GetLjEnergy(inv_r6, c12, c6);    \
            const float pair_coulomb_energy =                              \
                GetDirectCoulombEnergy(charge_product, inv_r, beta_dr);    \
            if constexpr (total_output)                                    \
            {                                                              \
                output_buf.energy_lj_total += pair_lj_energy;              \
                output_buf.energy_coulomb_total += pair_coulomb_energy;    \
            }                                                              \
            else                                                           \
            {                                                              \
                output_buf.energy_lj[I] += pair_lj_energy;                 \
                output_buf.energy_coulomb[I] += pair_coulomb_energy;       \
            }                                                              \
        }                                                                  \
    }

#define SPONGE_PROD_COMPUTE_I_BODY(I)                                      \
    {                                                                      \
            const float4 r1_xq =                                           \
                shared_i_xq[(I) * kClusterSize + i_lane];                 \
            float dx = shifted_j_x - r1_xq.x;                              \
            float dy = shifted_j_y - r1_xq.y;                              \
            float dz = shifted_j_z - r1_xq.z;                              \
            if constexpr (!sci_shift_only)                                 \
            {                                                              \
                const Vec3 pair_shift =                                    \
                    pair_shift_bits != nullptr                             \
                        ? ShiftVectorFromId(GetPairShiftId(shift_bits, I), \
                                            cell)                          \
                        : sci_shift;                                       \
                dx -= pair_shift.x;                                        \
                dy -= pair_shift.y;                                        \
                dz -= pair_shift.z;                                        \
            }                                                              \
            const float dr2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));         \
            if constexpr (attribution_no_cutoff_branch)                    \
            {                                                              \
                SPONGE_PROD_PAIR_FORCE_BODY(I)                             \
            }                                                              \
            else if (dr2 < cutoff_sq)                                      \
            {                                                              \
                SPONGE_PROD_PAIR_FORCE_BODY(I)                             \
            }                                                              \
    }

#define SPONGE_PROD_COMPUTE_I(I)                                           \
    {                                                                      \
        constexpr unsigned int packed_bit =                                \
            base_mask << static_cast<unsigned int>(I);                     \
        constexpr unsigned int local_bit =                                 \
            1u << static_cast<unsigned int>(I);                            \
        if constexpr (attribution_force_all_i)                              \
        {                                                                  \
            SPONGE_PROD_COMPUTE_I_BODY(I)                                  \
        }                                                                  \
        else                                                               \
        {                                                                  \
            const bool active_pair =                                        \
                local_i_mask8 ? ((active_i_mask & local_bit) != 0u)         \
                              : ((effective_mask & packed_bit) != 0u);      \
            if (active_pair)                                               \
            {                                                              \
                SPONGE_PROD_COMPUTE_I_BODY(I)                              \
            }                                                              \
        }                                                                  \
    }

#define SPONGE_PROD_COMPUTE_I_DENSE(I)                                     \
    {                                                                      \
        SPONGE_PROD_COMPUTE_I_BODY(I)                                      \
    }

#define SPONGE_PROD_PROCESS_JM(JM)                                         \
    {                                                                      \
        constexpr unsigned int base_mask =                                 \
            1u << ((JM) * kSuperClusterClusters);                         \
        constexpr unsigned int jm_mask =                                   \
            ((1u << kSuperClusterClusters) - 1u)                           \
            << ((JM) * kSuperClusterClusters);                             \
        const unsigned int active_i_mask =                                 \
            (effective_mask >> ((JM) * kSuperClusterClusters)) &           \
            ((1u << kSuperClusterClusters) - 1u);                          \
        const bool process_jm = skip_empty_effective_jm                    \
                                    ? __any_sync(kFullMask,                \
                                                 active_i_mask != 0u)      \
                                    : ((imask & jm_mask) != 0u);           \
        if (process_jm)                                                    \
        {                                                                  \
            const int cluster_j = packed->cj[JM];                          \
            if (cluster_j >= 0 &&                                          \
                (sci_shift_only || cluster_j < cluster_numbers))           \
            {                                                              \
                uint64_t shift_bits = 0ull;                                \
                if constexpr (!sci_shift_only)                             \
                {                                                          \
                    shift_bits = pair_shift_bits != nullptr                \
                                     ? pair_shift_bits[packed_idx *        \
                                                       kJGroupSize + (JM)] \
                                     : 0ull;                               \
                }                                                          \
                const int sorted_j = cluster_j * kClusterSize + j_lane;    \
                const float4 r2_xq = sorted_xq[sorted_j];                  \
                const float shifted_j_x = r2_xq.x;                         \
                const float shifted_j_y = r2_xq.y;                         \
                const float shifted_j_z = r2_xq.z;                         \
                const float qj = r2_xq.w;                                  \
                int r2_lj_type = 0;                                        \
                float lj_j_x = 0.0f;                                       \
                float lj_j_y = 0.0f;                                       \
                if constexpr (use_lj_comb)                                 \
                {                                                          \
                    const float2 r2_lj_comb = sorted_lj_comb[sorted_j];    \
                    lj_j_x = r2_lj_comb.x;                                 \
                    lj_j_y = r2_lj_comb.y;                                 \
                }                                                          \
                else                                                       \
                {                                                          \
                    r2_lj_type = sorted_lj_type[sorted_j];                 \
                }                                                          \
                float fcj_x = 0.0f;                                        \
                float fcj_y = 0.0f;                                        \
                float fcj_z = 0.0f;                                        \
                const bool dense_no_exclusion =                            \
                    dense_no_exclusion_fast && exclusion_index == 0 &&      \
                    active_i_mask == ((1u << kSuperClusterClusters) - 1u); \
                if (dense_no_exclusion)                                    \
                {                                                          \
                    SPONGE_PROD_GMXPACKED_I_LIST(                          \
                        SPONGE_PROD_COMPUTE_I_DENSE)                       \
                }                                                          \
                else                                                       \
                {                                                          \
                    SPONGE_PROD_GMXPACKED_I_LIST(SPONGE_PROD_COMPUTE_I)   \
                }                                                          \
                if constexpr (!sits_correction_only)                       \
                {                                                          \
                    const float fcj_component =                            \
                        ReduceSubgroupVecToComponent8(                      \
                            fcj_x, fcj_y, fcj_z, i_lane, lane);             \
                    if (i_lane < 3)                                        \
                    {                                                      \
                        const int force_index =                            \
                            compact_force_storage                          \
                                ? sorted_j                                 \
                                : sorted_atom_ids[sorted_j];                \
                        if constexpr (skip_force_writeback)                \
                        {                                                  \
                            StoreProductionForceComponent(                 \
                                frc, force_index, i_lane, fcj_component);   \
                        }                                                  \
                        else                                               \
                        {                                                  \
                            AtomicAddProductionForceComponent(             \
                                frc, force_index, i_lane, fcj_component);   \
                        }                                                  \
                    }                                                      \
                }                                                          \
            }                                                              \
        }                                                                  \
    }
        SPONGE_PROD_GMXPACKED_JM_LIST(SPONGE_PROD_PROCESS_JM)
#undef SPONGE_PROD_PROCESS_JM
#undef SPONGE_PROD_COMPUTE_I_DENSE
#undef SPONGE_PROD_COMPUTE_I
#undef SPONGE_PROD_COMPUTE_I_BODY
#undef SPONGE_PROD_PAIR_FORCE_BODY
    }

#define SPONGE_PROD_REDUCE_I(I)                                            \
    {                                                                      \
        float reduced_x = fci_x_##I;                                       \
        float reduced_y = fci_y_##I;                                       \
        float reduced_z = fci_z_##I;                                       \
        const float reduced_component =                                    \
            ReduceWarpIToComponent8(reduced_x, reduced_y, reduced_z,       \
                                    split_j_lane);                         \
        if (split_j_lane < 3)                                              \
        {                                                                  \
            const int sorted_i_local =                                     \
                (cluster_i_start + (I)) * kClusterSize + i_lane;           \
            const int force_index = compact_force_storage                  \
                                        ? sorted_i_local                   \
                                        : sorted_atom_ids[sorted_i_local]; \
            if constexpr (skip_force_writeback)                            \
            {                                                              \
                StoreProductionForceComponent(                             \
                    frc, force_index, split_j_lane, reduced_component);    \
            }                                                              \
            else                                                           \
            {                                                              \
                AtomicAddProductionForceComponent(                         \
                    frc, force_index, split_j_lane, reduced_component);    \
            }                                                              \
            if constexpr (compute_virial && virial_from_shift)             \
            {                                                              \
                if (sci_entry.shift_id != kCentralShiftId)                 \
                {                                                          \
                    fshift_component += reduced_component;                 \
                }                                                          \
            }                                                              \
        }                                                                  \
        if constexpr (compute_virial && !total_output)                     \
        {                                                                  \
            if (store_virial)                                              \
            {                                                              \
                LTMatrix3 reduced_virial = output_buf.virial[I];           \
                reduced_virial =                                           \
                    ReduceWarpVirialOverJ(reduced_virial, kClusterSize);   \
                if constexpr (sci_work_parts == 2)                         \
                {                                                          \
                    if (lane < kClusterSize)                               \
                    {                                                      \
                        shared_split_virial_lo[split][I][i_lane] =         \
                            PackVirialLo(reduced_virial);                   \
                        shared_split_virial_hi[split][I][i_lane] =         \
                            PackVirialHi(reduced_virial);                   \
                    }                                                      \
                }                                                          \
                else if (lane < kClusterSize)                              \
                {                                                          \
                    const int sorted_i_local =                             \
                        (cluster_i_start + (I)) * kClusterSize + i_lane;   \
                    const int atom_i = sorted_atom_ids[sorted_i_local];    \
                    if (atom_i >= 0)                                       \
                    {                                                      \
                        AtomicAddVirial(atom_virial + atom_i,              \
                                       reduced_virial);                    \
                    }                                                      \
                }                                                          \
            }                                                              \
        }                                                                  \
        if constexpr (compute_energy && !total_output)                     \
        {                                                                  \
            if (store_energy)                                              \
            {                                                              \
                float reduced_lj = output_buf.energy_lj[I];                \
                float reduced_coulomb = output_buf.energy_coulomb[I];      \
                reduced_lj =                                               \
                    ReduceWarpFloatOverJ(reduced_lj, kClusterSize);        \
                reduced_coulomb =                                          \
                    ReduceWarpFloatOverJ(reduced_coulomb, kClusterSize);   \
                if (lane < kClusterSize)                                   \
                {                                                          \
                    const int sorted_i_local =                             \
                        (cluster_i_start + (I)) * kClusterSize + i_lane;   \
                    const int atom_i = sorted_atom_ids[sorted_i_local];    \
                    if (atom_i >= 0)                                       \
                    {                                                      \
                        atomicAdd(atom_energy + atom_i,                    \
                                  reduced_lj + reduced_coulomb);          \
                        atomicAdd(atom_lj_energy + atom_i, reduced_lj);    \
                        atomicAdd(atom_direct_cf_energy + atom_i,          \
                                  reduced_coulomb);                        \
                    }                                                      \
                }                                                          \
            }                                                              \
        }                                                                  \
    }
    if constexpr (!sits_correction_only)
    {
        SPONGE_PROD_GMXPACKED_I_LIST(SPONGE_PROD_REDUCE_I)
    }
#undef SPONGE_PROD_REDUCE_I

    if constexpr (compute_virial && !total_output && sci_work_parts == 2)
    {
        if (store_virial)
        {
            __syncthreads();
#define SPONGE_PROD_WRITE_MERGED_VIRIAL(I)                                \
    if (split == 0 && lane < kClusterSize)                                 \
    {                                                                       \
        const int sorted_i_local =                                          \
            (cluster_i_start + (I)) * kClusterSize + i_lane;                \
        const int atom_i = sorted_atom_ids[sorted_i_local];                 \
        if (atom_i >= 0)                                                    \
        {                                                                   \
            const LTMatrix3 merged_virial =                                 \
                UnpackVirial(shared_split_virial_lo[0][I][i_lane],          \
                             shared_split_virial_hi[0][I][i_lane]) +         \
                UnpackVirial(shared_split_virial_lo[1][I][i_lane],          \
                             shared_split_virial_hi[1][I][i_lane]);          \
            AtomicAddVirial(atom_virial + atom_i, merged_virial);           \
        }                                                                   \
    }
            SPONGE_PROD_GMXPACKED_I_LIST(
                SPONGE_PROD_WRITE_MERGED_VIRIAL)
#undef SPONGE_PROD_WRITE_MERGED_VIRIAL
        }
    }

    if constexpr (total_output)
    {
        LTMatrix3 reduced_total_virial = {0.0f, 0.0f, 0.0f,
                                          0.0f, 0.0f, 0.0f};
        if constexpr (compute_virial && !virial_from_shift)
        {
            reduced_total_virial =
                ReduceWarpVirialAll(output_buf.virial_total);
        }
        float reduced_total_energy_lj = 0.0f;
        float reduced_total_energy_coulomb = 0.0f;
        if constexpr (compute_energy)
        {
            reduced_total_energy_lj =
                ReduceWarpFloatAll(output_buf.energy_lj_total);
            reduced_total_energy_coulomb =
                ReduceWarpFloatAll(output_buf.energy_coulomb_total);
        }
        if (lane == 0)
        {
            if constexpr (compute_virial && !virial_from_shift)
            {
                shared_total_virial_lo[split] =
                    PackVirialLo(reduced_total_virial);
                shared_total_virial_hi[split] =
                    PackVirialHi(reduced_total_virial);
            }
            if constexpr (compute_energy)
            {
                shared_total_energy_lj[split] = reduced_total_energy_lj;
                shared_total_energy_coulomb[split] =
                    reduced_total_energy_coulomb;
            }
        }
        __syncthreads();
        if (tid == 0)
        {
            if constexpr (compute_virial && !virial_from_shift)
            {
                LTMatrix3 block_total_virial = UnpackVirial(
                    shared_total_virial_lo[0], shared_total_virial_hi[0]);
                for (int warp = 1; warp < max_block_warps; warp += 1)
                {
                    block_total_virial =
                        block_total_virial +
                        UnpackVirial(shared_total_virial_lo[warp],
                                     shared_total_virial_hi[warp]);
                }
                AtomicAddVirial(atom_virial, block_total_virial);
            }
            if constexpr (compute_energy)
            {
                float block_total_energy_lj = 0.0f;
                float block_total_energy_coulomb = 0.0f;
                for (int warp = 0; warp < max_block_warps; warp += 1)
                {
                    block_total_energy_lj += shared_total_energy_lj[warp];
                    block_total_energy_coulomb +=
                        shared_total_energy_coulomb[warp];
                }
                atomicAdd(atom_energy,
                          block_total_energy_lj + block_total_energy_coulomb);
                atomicAdd(atom_lj_energy, block_total_energy_lj);
                atomicAdd(atom_direct_cf_energy, block_total_energy_coulomb);
            }
        }
    }
    if constexpr (compute_virial && virial_from_shift)
    {
        if (sci_entry.shift_id != kCentralShiftId && split_j_lane < 3)
        {
            const float reduced_fshift =
                ReduceSubgroupFloat8(fshift_component, lane);
            if (i_lane == 0)
            {
                atomicAdd(&shift_force[sci_entry.shift_id].x + split_j_lane,
                          reduced_fshift);
            }
        }
    }
    };

    if constexpr (!need_energy && !need_virial)
    {
        process_packed(std::false_type{}, std::false_type{});
    }
    else if (store_energy)
    {
        process_packed(std::true_type{}, std::true_type{});
    }
    else
    {
        process_packed(std::false_type{}, std::true_type{});
    }

#undef SPONGE_PROD_GMXPACKED_JM_LIST
#undef SPONGE_PROD_GMXPACKED_I_LIST
}

template <bool sorted_force_storage, bool use_shiftvec>
__global__ __launch_bounds__(kClusterSize * kSuperClusterClusters, 9)
void SpongeProductionGmxpackedSpecializedSafeForceKernel(
    int sci_numbers, int cluster_numbers,
    const SpongeGmxpackedSciPOD* sci_entries,
    const SpongeGmxpackedCjPOD* cjpacked_entries,
    const SpongeGmxpackedExclusionPOD* exclusion_entries,
    const int* sorted_atom_ids, const float4* sorted_xq,
    const float2* sorted_lj_comb, const float4* shiftvec, LTMatrix3 cell,
    float cutoff,
    float4* frc, float pme_beta)
{
    constexpr int max_super_cluster_atoms = kClusterSize * kSuperClusterClusters;
    const int sci = blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (sci >= sci_numbers || tid >= max_super_cluster_atoms)
    {
        return;
    }

    const int i_lane = threadIdx.x;
    const int j_lane = threadIdx.y;
    const int lane = tid & (warpSize - 1);
    const int split = tid / warpSize;
    const int split_j_lane = j_lane - split * kSplitJClusterSize;
    const int i_slot = j_lane * kClusterSize + i_lane;

    const SpongeGmxpackedSciPOD sci_entry = sci_entries[sci];
    const int cluster_i_start = sci_entry.supercluster_id * kSuperClusterClusters;
    Vec3 sci_shift = {};
    if constexpr (use_shiftvec)
    {
        const float4 shift = shiftvec[sci_entry.shift_id];
        sci_shift = {shift.x, shift.y, shift.z};
    }
    else
    {
        sci_shift = ShiftVectorFromId(sci_entry.shift_id, cell);
    }
    const float cutoff_sq = cutoff * cutoff;
    const float beta2 = pme_beta * pme_beta;
    const float beta3 = beta2 * pme_beta;
    constexpr float min_distance_sq = 3.82e-07f;

#define SPONGE_SPEC_SAFE_I_LIST(OP) \
    OP(0)                           \
    OP(1)                           \
    OP(2)                           \
    OP(3)                           \
    OP(4)                           \
    OP(5)                           \
    OP(6)                           \
    OP(7)

#define SPONGE_SPEC_SAFE_JM_LIST(OP) \
    OP(0)                            \
    OP(1)                            \
    OP(2)                            \
    OP(3)

    __shared__ float4 shared_i_xq[max_super_cluster_atoms];
    __shared__ float2 shared_i_lj_comb[max_super_cluster_atoms];

#define SPONGE_SPEC_SAFE_DECLARE_FCI(I) \
    float fci_x_##I = 0.0f;             \
    float fci_y_##I = 0.0f;             \
    float fci_z_##I = 0.0f;
    SPONGE_SPEC_SAFE_I_LIST(SPONGE_SPEC_SAFE_DECLARE_FCI)
#undef SPONGE_SPEC_SAFE_DECLARE_FCI

    const int sorted_i = (cluster_i_start + j_lane) * kClusterSize + i_lane;
    float4 i_xq = sorted_xq[sorted_i];
    i_xq.x += sci_shift.x;
    i_xq.y += sci_shift.y;
    i_xq.z += sci_shift.z;
    shared_i_xq[i_slot] = i_xq;
    shared_i_lj_comb[i_slot] = sorted_lj_comb[sorted_i];
    __syncthreads();

    for (int packed_idx = sci_entry.cjpacked_begin;
         packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
    {
        const SpongeGmxpackedCjPOD* packed = cjpacked_entries + packed_idx;
        const unsigned int imask = packed->split[split].imask;
        if (imask == 0u)
        {
            continue;
        }
        const int exclusion_index = packed->split[split].exclusion_index;
        unsigned int pair_bits = 0xffffffffu;
        if (exclusion_index != 0)
        {
            pair_bits = exclusion_entries[exclusion_index]
                            .pair[split_j_lane * kClusterSize + i_lane];
        }
        const unsigned int effective_mask = imask & pair_bits;

#define SPONGE_SPEC_SAFE_COMPUTE_I(I)                                      \
    {                                                                      \
        constexpr unsigned int packed_bit =                                \
            base_mask << static_cast<unsigned int>(I);                     \
        if ((effective_mask & packed_bit) != 0u)                           \
        {                                                                  \
            const float4 r1_xq =                                           \
                shared_i_xq[(I) * kClusterSize + i_lane];                 \
            const float dx = shifted_j_x - r1_xq.x;                        \
            const float dy = shifted_j_y - r1_xq.y;                        \
            const float dz = shifted_j_z - r1_xq.z;                        \
            const float dr2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));         \
            if (dr2 < cutoff_sq && dr2 != 0.0f)                            \
            {                                                              \
                const float r2 = fmaxf(dr2, min_distance_sq);              \
                const float inv_r = rsqrtf(r2);                            \
                const float inv_r2 = inv_r * inv_r;                        \
                const float inv_r6 = inv_r2 * inv_r2 * inv_r2;             \
                const float charge_product = r1_xq.w * qj;                \
                const float2 ljcp_i =                                      \
                    shared_i_lj_comb[(I) * kClusterSize + i_lane];        \
                const float c6 = ljcp_i.x * lj_j_x;                       \
                const float c12 = ljcp_i.y * lj_j_y;                      \
                float frc_abs = GetLjForceAbs(inv_r2, inv_r6, c12, c6);   \
                frc_abs -= GetDirectCoulombForceAbsPmeCorrF(               \
                    charge_product, inv_r, inv_r2, beta2 * r2, beta3);     \
                const float fij_x = frc_abs * dx;                          \
                const float fij_y = frc_abs * dy;                          \
                const float fij_z = frc_abs * dz;                          \
                fci_x_##I += fij_x;                                        \
                fci_y_##I += fij_y;                                        \
                fci_z_##I += fij_z;                                        \
                fcj_x -= fij_x;                                            \
                fcj_y -= fij_y;                                            \
                fcj_z -= fij_z;                                            \
            }                                                              \
        }                                                                  \
    }

#define SPONGE_SPEC_SAFE_PROCESS_JM(JM)                                    \
    {                                                                      \
        constexpr unsigned int base_mask =                                 \
            1u << ((JM) * kSuperClusterClusters);                         \
        constexpr unsigned int jm_mask =                                   \
            ((1u << kSuperClusterClusters) - 1u)                           \
            << ((JM) * kSuperClusterClusters);                             \
        if ((imask & jm_mask) != 0u)                                       \
        {                                                                  \
            const int cluster_j = packed->cj[JM];                          \
            if (cluster_j >= 0 && cluster_j < cluster_numbers)             \
            {                                                              \
                const int sorted_j = cluster_j * kClusterSize + j_lane;    \
                const float4 r2_xq = sorted_xq[sorted_j];                  \
                const float2 r2_lj_comb = sorted_lj_comb[sorted_j];        \
                const float shifted_j_x = r2_xq.x;                         \
                const float shifted_j_y = r2_xq.y;                         \
                const float shifted_j_z = r2_xq.z;                         \
                const float qj = r2_xq.w;                                  \
                const float lj_j_x = r2_lj_comb.x;                         \
                const float lj_j_y = r2_lj_comb.y;                         \
                float fcj_x = 0.0f;                                        \
                float fcj_y = 0.0f;                                        \
                float fcj_z = 0.0f;                                        \
                SPONGE_SPEC_SAFE_I_LIST(SPONGE_SPEC_SAFE_COMPUTE_I)       \
                const float fcj_component =                                \
                    ReduceSubgroupVecToComponent8(fcj_x, fcj_y, fcj_z,     \
                                                   i_lane, lane);          \
                if (i_lane < 3)                                            \
                {                                                          \
                    const int force_index = sorted_force_storage           \
                                                ? sorted_j                 \
                                                : sorted_atom_ids[sorted_j];\
                    AtomicAddProductionForceComponent(                     \
                        frc, force_index, i_lane, fcj_component);          \
                }                                                          \
            }                                                              \
        }                                                                  \
    }
        SPONGE_SPEC_SAFE_JM_LIST(SPONGE_SPEC_SAFE_PROCESS_JM)
#undef SPONGE_SPEC_SAFE_PROCESS_JM
#undef SPONGE_SPEC_SAFE_COMPUTE_I
    }

#define SPONGE_SPEC_SAFE_REDUCE_I(I)                                       \
    {                                                                      \
        float reduced_x = fci_x_##I;                                       \
        float reduced_y = fci_y_##I;                                       \
        float reduced_z = fci_z_##I;                                       \
        const float reduced_component =                                    \
            ReduceWarpIToComponent8(reduced_x, reduced_y, reduced_z,       \
                                    split_j_lane);                         \
        if (split_j_lane < 3)                                              \
        {                                                                  \
            const int sorted_i_local =                                     \
                (cluster_i_start + (I)) * kClusterSize + i_lane;           \
            const int force_index = sorted_force_storage                   \
                                        ? sorted_i_local                   \
                                        : sorted_atom_ids[sorted_i_local]; \
            AtomicAddProductionForceComponent(                             \
                frc, force_index, split_j_lane, reduced_component);        \
        }                                                                  \
    }
    SPONGE_SPEC_SAFE_I_LIST(SPONGE_SPEC_SAFE_REDUCE_I)
#undef SPONGE_SPEC_SAFE_REDUCE_I

#undef SPONGE_SPEC_SAFE_JM_LIST
#undef SPONGE_SPEC_SAFE_I_LIST
}

// Legacy internal GROMACS-format replay kernel; not used by the kept paths.
#if 0
__global__ void GromacsPairlistReplayKernel(
    int sci_numbers, int local_atom_numbers, const int* cluster_offsets,
    const unsigned int* cluster_valid_masks, const unsigned int* cluster_local_masks,
    const GromacsSciPOD* sci_entries, const GromacsCjPackedPOD* cjpacked_entries,
    const GromacsExclPOD* excl_entries, const float4* sorted_xq,
    const int* sorted_lj_type, const float2* lj_ab_packed,
    const float4* shiftvec, float cutoff, float pme_beta,
    float* frc_x, float* frc_y, float* frc_z)
{
    const int sci = blockIdx.x;
    const int i_lane = threadIdx.x;
    const int j_lane = threadIdx.y;
    const int lane = j_lane * blockDim.x + i_lane;
    const int split = j_lane / kSplitJClusterSize;
    const int split_j_lane = j_lane & (kSplitJClusterSize - 1);
    if (sci >= sci_numbers)
    {
        return;
    }

    const GromacsSciPOD sci_entry = sci_entries[sci];
    const float4 shift = shiftvec[sci_entry.shift];
    const float cutoff_sq = cutoff * cutoff;
    const int cluster_i_start = sci_entry.sci * kSuperClusterClusters;

    for (int packed_idx = sci_entry.cjPackedBegin; packed_idx < sci_entry.cjPackedEnd;
         ++packed_idx)
    {
        const GromacsCjPackedPOD packed = cjpacked_entries[packed_idx];
        for (int jm = 0; jm < kJGroupSize; ++jm)
        {
            const int cluster_j = packed.cj[jm];
            if (cluster_j < 0)
            {
                continue;
            }
            const unsigned int valid_mask_j = cluster_valid_masks[cluster_j];
            const unsigned int local_mask_j = cluster_local_masks[cluster_j];
            if ((valid_mask_j & (1u << static_cast<unsigned int>(j_lane))) == 0u)
            {
                continue;
            }
            const int sorted_j = cluster_offsets[cluster_j] + j_lane;
            const VectorLj r2 =
                MakePackedLjAtom(sorted_xq[sorted_j], sorted_lj_type[sorted_j]);
            const GromacsExclPOD excl = excl_entries[packed.imei[split].excl_ind];
            const unsigned int pair_bits =
                excl.pair[split_j_lane * kClusterSize + i_lane];
            for (int im = 0; im < kSuperClusterClusters; ++im)
            {
                if (((packed.imei[split].imask >>
                      (jm * kSuperClusterClusters + im)) &
                     1u) == 0u)
                {
                    continue;
                }
                const int cluster_i = cluster_i_start + im;
                const unsigned int valid_mask_i = cluster_valid_masks[cluster_i];
                const unsigned int local_mask_i = cluster_local_masks[cluster_i];
                if ((valid_mask_i &
                     (1u << static_cast<unsigned int>(i_lane))) == 0u ||
                    (local_mask_i &
                     (1u << static_cast<unsigned int>(i_lane))) == 0u)
                {
                    continue;
                }
                if (((pair_bits >> (jm * kSuperClusterClusters + im)) & 1u) == 0u)
                {
                    continue;
                }
                if (sci_entry.shift == kCentralShiftId && cluster_i == cluster_j &&
                    (local_mask_j & (1u << static_cast<unsigned int>(j_lane))) != 0u &&
                    j_lane <= i_lane)
                {
                    continue;
                }
                const int sorted_i = cluster_offsets[cluster_i] + i_lane;
                const VectorLj r1 = MakePackedLjAtom(sorted_xq[sorted_i],
                                                     sorted_lj_type[sorted_i]);
                const Vec3 dr = {shift.x + r1.crd.x - r2.crd.x,
                                 shift.y + r1.crd.y - r2.crd.y,
                                 shift.z + r1.crd.z - r2.crd.z};
                const float dr2 = Dot(dr, dr);
                if (dr2 < cutoff_sq && dr2 != 0.0f)
                {
                    const float inv_r = rsqrtf(dr2);
                    const float inv_r2 = inv_r * inv_r;
                    const float inv_r6 = inv_r2 * inv_r2 * inv_r2;
                    const float beta_dr = pme_beta * (dr2 * inv_r);
                    const float charge_product = r1.charge * r2.charge;
                    const int lj_index = GetLjType(r1.lj_type, r2.lj_type);
                    const float2 AB = lj_ab_packed[lj_index];
                    float frc_abs = GetLjForceAbs(inv_r2, inv_r6, AB.x, AB.y);
                    frc_abs -= GetDirectCoulombForceAbs(charge_product, inv_r,
                                                        inv_r2, beta_dr);
                    const float fx = frc_abs * dr.x;
                    const float fy = frc_abs * dr.y;
                    const float fz = frc_abs * dr.z;
                    atomicAdd(frc_x + sorted_i, fx);
                    atomicAdd(frc_y + sorted_i, fy);
                    atomicAdd(frc_z + sorted_i, fz);
                    if ((local_mask_j &
                         (1u << static_cast<unsigned int>(j_lane))) != 0u)
                    {
                        atomicAdd(frc_x + sorted_j, -fx);
                        atomicAdd(frc_y + sorted_j, -fy);
                        atomicAdd(frc_z + sorted_j, -fz);
                    }
                }
            }
        }
    }
}
#endif

struct Arguments
{
    std::string kernel;
    std::string snapshot;
    std::string referenceGmxSnapshot;
    std::string spongeLjMode = "comb-gmxpacked";
    std::string spongeGmxTransform = "baseline";
    double exactImaskRadiusScale = 1.0;
    int warmup = 50;
    int iters = 200;
    int refreshBlockSize = 128;
    int sitsAtomEnd = -1;
    float sitsPwwpFactor = 1.0f;
    bool analyze = false;
    bool pairOracle = false;
    bool computeEnergy = false;
    bool computeVirial = false;
};

enum class ProductionGmxpackedReplayMode
{
    split,
    fullOutputSciSplit2,
    fullOutputSciSplit4,
    fullOutputSciSplit8,
    fusedSitsForceOnly,
    sparseSitsForceOnly,
    compactForce,
    sortedForce,
    sortedForceSciSplit2,
    sortedForceSciSplit3,
    sortedForceSciSplit4,
    sortedForceLocalIMask8,
    sortedForceActiveIMask8,
    sortedForceOracleImask,
    sortedForceOracleSidecar,
    sortedForceDeviceSidecar,
    sortedForceDenseNoExcl,
    sortedForceAttrAllI,
    sortedForceAttrNoCutoff,
    sortedForceAttrAllINoCutoff,
    sortedForceNoWriteback,
    safeOnly,
    specializedSafe,
    specializedSortedForce,
    specializedShiftvec,
    shiftVirial,
};

const char* ProductionGmxpackedReplayModeName(
    ProductionGmxpackedReplayMode mode)
{
    switch (mode)
    {
        case ProductionGmxpackedReplayMode::split:
            return "split";
        case ProductionGmxpackedReplayMode::fullOutputSciSplit2:
            return "full-output-sci-split2";
        case ProductionGmxpackedReplayMode::fullOutputSciSplit4:
            return "full-output-sci-split4";
        case ProductionGmxpackedReplayMode::fullOutputSciSplit8:
            return "full-output-sci-split8";
        case ProductionGmxpackedReplayMode::fusedSitsForceOnly:
            return "fused-sits-force-only";
        case ProductionGmxpackedReplayMode::sparseSitsForceOnly:
            return "sparse-sits-force-only";
        case ProductionGmxpackedReplayMode::compactForce:
            return "compact-force";
        case ProductionGmxpackedReplayMode::sortedForce:
            return "sorted-force";
        case ProductionGmxpackedReplayMode::sortedForceSciSplit2:
            return "sorted-force-sci-split2";
        case ProductionGmxpackedReplayMode::sortedForceSciSplit3:
            return "sorted-force-sci-split3";
        case ProductionGmxpackedReplayMode::sortedForceSciSplit4:
            return "sorted-force-sci-split4";
        case ProductionGmxpackedReplayMode::sortedForceLocalIMask8:
            return "sorted-force-local-i-mask8";
        case ProductionGmxpackedReplayMode::sortedForceActiveIMask8:
            return "sorted-force-active-i-mask8";
        case ProductionGmxpackedReplayMode::sortedForceOracleImask:
            return "sorted-force-oracle-imask";
        case ProductionGmxpackedReplayMode::sortedForceOracleSidecar:
            return "sorted-force-oracle-sidecar";
        case ProductionGmxpackedReplayMode::sortedForceDeviceSidecar:
            return "sorted-force-device-sidecar";
        case ProductionGmxpackedReplayMode::sortedForceDenseNoExcl:
            return "sorted-force-dense-noexcl";
        case ProductionGmxpackedReplayMode::sortedForceAttrAllI:
            return "sorted-force-attr-all-i";
        case ProductionGmxpackedReplayMode::sortedForceAttrNoCutoff:
            return "sorted-force-attr-no-cutoff";
        case ProductionGmxpackedReplayMode::sortedForceAttrAllINoCutoff:
            return "sorted-force-attr-all-i-no-cutoff";
        case ProductionGmxpackedReplayMode::sortedForceNoWriteback:
            return "sorted-force-no-atomic";
        case ProductionGmxpackedReplayMode::safeOnly:
            return "safe-only";
        case ProductionGmxpackedReplayMode::specializedSafe:
            return "specialized-safe";
        case ProductionGmxpackedReplayMode::specializedSortedForce:
            return "specialized-sorted-force";
        case ProductionGmxpackedReplayMode::specializedShiftvec:
            return "specialized-shiftvec";
        case ProductionGmxpackedReplayMode::shiftVirial:
            return "shift-virial";
    }
    return "unknown";
}

void PrintUsage(const char* argv0)
{
    std::fprintf(stderr,
                 "Usage: %s (--kernel sponge|gmx|builder | --builder) --snapshot PATH "
                 "[--sponge-lj-mode fulloutput|comb-gmxpacked|production-gmxpacked|"
                 "production-gmxpacked-full-sci-split2|"
                 "production-gmxpacked-full-sci-split4|"
                 "production-gmxpacked-full-sci-split8|"
                 "production-gmxpacked-fused-sits-force-only|"
                 "production-gmxpacked-sparse-sits-force-only|"
                 "comb-gmxpacked-peratom|comb-gmxpacked-partial-reduce|"
                 "comb-gmxpacked-partial-reduce-specialized|"
                 "comb-gmxpacked-partial-reduce-threshold4|"
                 "comb-gmxpacked-central-direct-partial-reduce|"
                 "production-gmxpacked-compact-force|production-gmxpacked-sorted-force|"
                 "production-gmxpacked-sorted-force-sci-split2|"
                 "production-gmxpacked-sorted-force-sci-split3|"
                 "production-gmxpacked-sorted-force-sci-split4|"
                 "production-gmxpacked-sorted-force-local-i-mask8|"
                 "production-gmxpacked-sorted-force-active-i-mask8|"
                 "production-gmxpacked-sorted-force-oracle-imask|"
                 "production-gmxpacked-sorted-force-oracle-sidecar|"
                 "production-gmxpacked-sorted-force-device-sidecar|"
                 "production-gmxpacked-sorted-force-dense-noexcl|"
                 "production-gmxpacked-sorted-force-attr-all-i|"
                 "production-gmxpacked-sorted-force-attr-no-cutoff|"
                 "production-gmxpacked-sorted-force-attr-all-i-no-cutoff|"
                 "production-gmxpacked-sorted-force-no-atomic|"
                 "production-gmxpacked-safe-only|production-gmxpacked-specialized|"
                 "production-gmxpacked-specialized-sorted-force|"
                 "production-gmxpacked-specialized-shiftvec|"
                 "production-gmxpacked-shift-virial|"
                 "production-gmxpacked-refresh|"
                 "production-gmxpacked-record-stream-inner-active|"
                 "production-gmxpacked-collect-traversal|"
                 "production-gmxpacked-collect-screen|"
                 "production-gmxpacked-collect-emit|"
                 "production-gmxpacked-collect-coop-traversal|"
                 "production-gmxpacked-collect-coop-screen|"
                 "production-gmxpacked-collect-coop-emit|"
                 "production-gmxpacked-collect-screen-stats|"
                 "production-gmxpacked-collect-coop-screen-stats] "
                 "[--sponge-gmx-transform baseline] "
                 "[--warmup N] [--iters N] [--refresh-block-size N] "
                 "[--sits-atom-end N] [--sits-pwwp-factor X] [--compute-energy] "
                 "[--compute-virial] [--analyze] [--pair-oracle]\n",
                 argv0);
}

SpongeGmxTransformMode ParseSpongeGmxTransform(const std::string& value)
{
    if (value == "baseline")
    {
        return SpongeGmxTransformMode::baseline;
    }
    std::fprintf(stderr,
                 "unsupported sponge gmx transform after cleanup: %s (only baseline remains)\n",
                 value.c_str());
    std::exit(1);
}

const char* SpongeGmxTransformName(SpongeGmxTransformMode mode)
{
    (void)mode;
    return "baseline";
}

Arguments ParseArguments(int argc, char** argv)
{
    Arguments args;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view flag(argv[i]);
        if (flag == "--kernel" && i + 1 < argc)
        {
            args.kernel = argv[++i];
        }
        else if (flag == "--builder")
        {
            args.kernel = "builder";
        }
        else if (flag == "--snapshot" && i + 1 < argc)
        {
            args.snapshot = argv[++i];
        }
        else if (flag == "--reference-gmx-snapshot" && i + 1 < argc)
        {
            args.referenceGmxSnapshot = argv[++i];
        }
        else if (flag == "--sponge-lj-mode" && i + 1 < argc)
        {
            args.spongeLjMode = argv[++i];
        }
        else if (flag == "--sponge-gmx-transform" && i + 1 < argc)
        {
            args.spongeGmxTransform = argv[++i];
        }
        else if (flag == "--exact-imask-radius-scale" && i + 1 < argc)
        {
            args.exactImaskRadiusScale = std::atof(argv[++i]);
        }
        else if (flag == "--warmup" && i + 1 < argc)
        {
            args.warmup = std::atoi(argv[++i]);
        }
        else if (flag == "--iters" && i + 1 < argc)
        {
            args.iters = std::atoi(argv[++i]);
        }
        else if (flag == "--refresh-block-size" && i + 1 < argc)
        {
            args.refreshBlockSize = std::atoi(argv[++i]);
        }
        else if (flag == "--sits-atom-end" && i + 1 < argc)
        {
            args.sitsAtomEnd = std::atoi(argv[++i]);
        }
        else if (flag == "--sits-pwwp-factor" && i + 1 < argc)
        {
            args.sitsPwwpFactor =
                static_cast<float>(std::atof(argv[++i]));
        }
        else if (flag == "--analyze")
        {
            args.analyze = true;
        }
        else if (flag == "--pair-oracle")
        {
            args.pairOracle = true;
        }
        else if (flag == "--compute-energy")
        {
            args.computeEnergy = true;
        }
        else if (flag == "--compute-virial")
        {
            args.computeVirial = true;
        }
        else
        {
            PrintUsage(argv[0]);
            std::exit(1);
        }
    }
    if (args.kernel.empty() || args.snapshot.empty())
    {
        PrintUsage(argv[0]);
        std::exit(1);
    }
    if (args.refreshBlockSize <= 0 || args.refreshBlockSize > 1024)
    {
        std::fprintf(stderr, "--refresh-block-size must be in [1, 1024]\n");
        std::exit(1);
    }
    if (args.sitsAtomEnd < -1)
    {
        std::fprintf(stderr, "--sits-atom-end must be non-negative\n");
        std::exit(1);
    }
    if (!std::isfinite(args.sitsPwwpFactor))
    {
        std::fprintf(stderr, "--sits-pwwp-factor must be finite\n");
        std::exit(1);
    }
    return args;
}

inline int HostPopcount(unsigned int value)
{
    return std::popcount(value);
}

double PercentileFromSorted(const std::vector<int>& sortedValues, double q)
{
    if (sortedValues.empty())
    {
        return 0.0;
    }
    const double clampedQ = std::clamp(q, 0.0, 1.0);
    const double position =
        clampedQ * static_cast<double>(sortedValues.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, sortedValues.size() - 1);
    const double weight = position - static_cast<double>(lower);
    return static_cast<double>(sortedValues[lower]) * (1.0 - weight) +
           static_cast<double>(sortedValues[upper]) * weight;
}

double PercentileFromSortedU32(const std::vector<uint32_t>& sortedValues,
                               double q)
{
    if (sortedValues.empty())
    {
        return 0.0;
    }
    const double clampedQ = std::clamp(q, 0.0, 1.0);
    const double position =
        clampedQ * static_cast<double>(sortedValues.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, sortedValues.size() - 1);
    const double weight = position - static_cast<double>(lower);
    return static_cast<double>(sortedValues[lower]) * (1.0 - weight) +
           static_cast<double>(sortedValues[upper]) * weight;
}

double AverageFromValues(const std::vector<int>& values)
{
    if (values.empty())
    {
        return 0.0;
    }
    const uint64_t total =
        std::accumulate(values.begin(), values.end(), uint64_t{0});
    return static_cast<double>(total) / static_cast<double>(values.size());
}

double PercentileFromSortedD(const std::vector<double>& sortedValues, double q)
{
    if (sortedValues.empty())
    {
        return 0.0;
    }
    const double clampedQ = std::clamp(q, 0.0, 1.0);
    const double position =
        clampedQ * static_cast<double>(sortedValues.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, sortedValues.size() - 1);
    const double weight = position - static_cast<double>(lower);
    return sortedValues[lower] * (1.0 - weight) +
           sortedValues[upper] * weight;
}

double AverageFromValuesD(const std::vector<double>& values)
{
    if (values.empty())
    {
        return 0.0;
    }
    const double total = std::accumulate(values.begin(), values.end(), 0.0);
    return total / static_cast<double>(values.size());
}

struct ClusterGeometryStats
{
    std::vector<double> radii;
    std::vector<double> aabbDiag;
};

ClusterGeometryStats CollectMinimumImageClusterGeometry(
    const std::vector<Float4POD>& sortedXq,
    const std::vector<int>& clusterOffsets,
    const std::vector<unsigned int>& clusterValidMasks,
    LTMatrix3 cell)
{
    ClusterGeometryStats stats;
    stats.radii.reserve(clusterOffsets.size());
    stats.aabbDiag.reserve(clusterOffsets.size());
    if (!CellLooksUsable(cell))
    {
        return stats;
    }
    const LTMatrix3 rcell = InvertCellMatrix(cell);
    for (size_t cluster = 0; cluster < clusterOffsets.size(); ++cluster)
    {
        if (cluster >= clusterValidMasks.size())
        {
            continue;
        }
        const unsigned int validMask = clusterValidMasks[cluster];
        if (validMask == 0u)
        {
            continue;
        }
        const int atomBase = clusterOffsets[cluster];
        int referenceLane = -1;
        for (int lane = 0; lane < kClusterSize; ++lane)
        {
            if ((validMask & (1u << static_cast<unsigned int>(lane))) != 0u)
            {
                referenceLane = lane;
                break;
            }
        }
        if (referenceLane < 0 ||
            static_cast<size_t>(atomBase + referenceLane) >= sortedXq.size())
        {
            continue;
        }
        const Float4POD& referenceAtom =
            sortedXq[static_cast<size_t>(atomBase + referenceLane)];
        const Vec3 reference{referenceAtom.x, referenceAtom.y, referenceAtom.z};
        std::array<Vec3, kClusterSize> unwrappedAtoms = {};
        Vec3 sum{0.0f, 0.0f, 0.0f};
        int validCount = 0;
        for (int lane = 0; lane < kClusterSize; ++lane)
        {
            if ((validMask & (1u << static_cast<unsigned int>(lane))) == 0u)
            {
                continue;
            }
            const size_t atomIndex = static_cast<size_t>(atomBase + lane);
            if (atomIndex >= sortedXq.size())
            {
                continue;
            }
            const Float4POD& atom = sortedXq[atomIndex];
            const Vec3 wrapped{atom.x, atom.y, atom.z};
            const Vec3 unwrapped =
                reference + MinimumImageDelta(wrapped - reference, cell, rcell);
            unwrappedAtoms[static_cast<size_t>(lane)] = unwrapped;
            sum = sum + unwrapped;
            validCount += 1;
        }
        if (validCount == 0)
        {
            continue;
        }
        const Vec3 center = sum / static_cast<float>(validCount);
        Vec3 minPoint{std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max()};
        Vec3 maxPoint{-std::numeric_limits<float>::max(),
                      -std::numeric_limits<float>::max(),
                      -std::numeric_limits<float>::max()};
        double maxRadiusSq = 0.0;
        for (int lane = 0; lane < kClusterSize; ++lane)
        {
            if ((validMask & (1u << static_cast<unsigned int>(lane))) == 0u)
            {
                continue;
            }
            const Vec3 atom = unwrappedAtoms[static_cast<size_t>(lane)];
            minPoint.x = std::min(minPoint.x, atom.x);
            minPoint.y = std::min(minPoint.y, atom.y);
            minPoint.z = std::min(minPoint.z, atom.z);
            maxPoint.x = std::max(maxPoint.x, atom.x);
            maxPoint.y = std::max(maxPoint.y, atom.y);
            maxPoint.z = std::max(maxPoint.z, atom.z);
            const double dx = static_cast<double>(atom.x - center.x);
            const double dy = static_cast<double>(atom.y - center.y);
            const double dz = static_cast<double>(atom.z - center.z);
            maxRadiusSq = std::max(maxRadiusSq, dx * dx + dy * dy + dz * dz);
        }
        const double ax = static_cast<double>(maxPoint.x - minPoint.x);
        const double ay = static_cast<double>(maxPoint.y - minPoint.y);
        const double az = static_cast<double>(maxPoint.z - minPoint.z);
        stats.radii.push_back(std::sqrt(maxRadiusSq));
        stats.aabbDiag.push_back(std::sqrt(ax * ax + ay * ay + az * az));
    }
    std::sort(stats.radii.begin(), stats.radii.end());
    std::sort(stats.aabbDiag.begin(), stats.aabbDiag.end());
    return stats;
}

void PrintClusterGeometryStats(const char* label,
                               const ClusterGeometryStats& stats)
{
    std::printf(
        "analysis=%s method=minimum_image cluster_radius_avg=%.6f "
        "cluster_radius_p50=%.6f cluster_radius_p90=%.6f "
        "cluster_radius_p99=%.6f cluster_radius_max=%.6f "
        "cluster_aabb_diag_avg=%.6f cluster_aabb_diag_p50=%.6f "
        "cluster_aabb_diag_p90=%.6f cluster_aabb_diag_p99=%.6f "
        "cluster_aabb_diag_max=%.6f\n",
        label,
        AverageFromValuesD(stats.radii),
        PercentileFromSortedD(stats.radii, 0.50),
        PercentileFromSortedD(stats.radii, 0.90),
        PercentileFromSortedD(stats.radii, 0.99),
        stats.radii.empty() ? 0.0 : stats.radii.back(),
        AverageFromValuesD(stats.aabbDiag),
        PercentileFromSortedD(stats.aabbDiag, 0.50),
        PercentileFromSortedD(stats.aabbDiag, 0.90),
        PercentileFromSortedD(stats.aabbDiag, 0.99),
        stats.aabbDiag.empty() ? 0.0 : stats.aabbDiag.back());
}

void AnalyzeSpongeWaterClusterComposition(
    const SpongeForceOnlySnapshot& snapshot)
{
    if (snapshot.sorted_atom_ids.empty() ||
        snapshot.header.total_atom_numbers % 3u != 0u)
    {
        return;
    }
    uint64_t validAtoms = 0;
    uint64_t partialMoleculeAtoms = 0;
    uint64_t fullMoleculeAtoms = 0;
    uint64_t adjacentLanePairs = 0;
    uint64_t adjacentSameMoleculePairs = 0;
    uint64_t adjacentConsecutiveAtomPairs = 0;
    std::vector<int> uniqueMoleculesPerCluster;
    std::vector<int> fullMoleculesPerCluster;
    std::vector<int> partialMoleculesPerCluster;
    uniqueMoleculesPerCluster.reserve(snapshot.cluster_offsets.size());
    fullMoleculesPerCluster.reserve(snapshot.cluster_offsets.size());
    partialMoleculesPerCluster.reserve(snapshot.cluster_offsets.size());
    for (size_t cluster = 0; cluster < snapshot.cluster_offsets.size(); ++cluster)
    {
        if (cluster >= snapshot.cluster_valid_masks.size())
        {
            continue;
        }
        const unsigned int validMask = snapshot.cluster_valid_masks[cluster];
        if (validMask == 0u)
        {
            continue;
        }
        const int atomBase = snapshot.cluster_offsets[cluster];
        std::unordered_map<int, int> moleculeAtomCounts;
        int previousAtomId = -1;
        for (int lane = 0; lane < kClusterSize; ++lane)
        {
            if ((validMask & (1u << static_cast<unsigned int>(lane))) == 0u)
            {
                continue;
            }
            const size_t atomIndex = static_cast<size_t>(atomBase + lane);
            if (atomIndex >= snapshot.sorted_atom_ids.size())
            {
                continue;
            }
            const int atomId = snapshot.sorted_atom_ids[atomIndex];
            const int moleculeId = atomId / 3;
            moleculeAtomCounts[moleculeId] += 1;
            validAtoms += 1;
            if (previousAtomId >= 0)
            {
                adjacentLanePairs += 1;
                if (previousAtomId / 3 == moleculeId)
                {
                    adjacentSameMoleculePairs += 1;
                }
                if (atomId == previousAtomId + 1)
                {
                    adjacentConsecutiveAtomPairs += 1;
                }
            }
            previousAtomId = atomId;
        }
        int fullMolecules = 0;
        int partialMolecules = 0;
        for (const auto& item : moleculeAtomCounts)
        {
            if (item.second == 3)
            {
                fullMolecules += 1;
                fullMoleculeAtoms += 3;
            }
            else
            {
                partialMolecules += 1;
                partialMoleculeAtoms += static_cast<uint64_t>(item.second);
            }
        }
        uniqueMoleculesPerCluster.push_back(
            static_cast<int>(moleculeAtomCounts.size()));
        fullMoleculesPerCluster.push_back(fullMolecules);
        partialMoleculesPerCluster.push_back(partialMolecules);
    }
    std::sort(uniqueMoleculesPerCluster.begin(),
              uniqueMoleculesPerCluster.end());
    std::sort(fullMoleculesPerCluster.begin(), fullMoleculesPerCluster.end());
    std::sort(partialMoleculesPerCluster.begin(),
              partialMoleculesPerCluster.end());
    std::printf(
        "analysis=sponge-water-cluster inferred_atoms_per_molecule=3 "
        "avg_unique_molecules_per_cluster=%.6f p50_unique=%.2f "
        "p90_unique=%.2f avg_full_molecules_per_cluster=%.6f "
        "avg_partial_molecules_per_cluster=%.6f "
        "partial_molecule_atom_ratio=%.6f full_molecule_atom_ratio=%.6f "
        "adjacent_same_molecule_ratio=%.6f "
        "adjacent_consecutive_atom_ratio=%.6f\n",
        AverageFromValues(uniqueMoleculesPerCluster),
        PercentileFromSorted(uniqueMoleculesPerCluster, 0.50),
        PercentileFromSorted(uniqueMoleculesPerCluster, 0.90),
        AverageFromValues(fullMoleculesPerCluster),
        AverageFromValues(partialMoleculesPerCluster),
        validAtoms > 0
            ? static_cast<double>(partialMoleculeAtoms) /
                  static_cast<double>(validAtoms)
            : 0.0,
        validAtoms > 0
            ? static_cast<double>(fullMoleculeAtoms) /
                  static_cast<double>(validAtoms)
            : 0.0,
        adjacentLanePairs > 0
            ? static_cast<double>(adjacentSameMoleculePairs) /
                  static_cast<double>(adjacentLanePairs)
            : 0.0,
        adjacentLanePairs > 0
            ? static_cast<double>(adjacentConsecutiveAtomPairs) /
                  static_cast<double>(adjacentLanePairs)
            : 0.0);
}

std::vector<Vec3> BuildSpongeClusterCenters(const SpongeForceOnlySnapshot& snapshot)
{
    std::vector<Vec3> centers(snapshot.cluster_offsets.size(), {0.0f, 0.0f, 0.0f});
    for (size_t cluster = 0; cluster < snapshot.cluster_offsets.size(); ++cluster)
    {
        const unsigned int validMask = snapshot.cluster_valid_masks[cluster];
        const int validCount = HostPopcount(validMask);
        if (validCount == 0)
        {
            continue;
        }
        Vec3 sum = {0.0f, 0.0f, 0.0f};
        const int atomBase = snapshot.cluster_offsets[cluster];
        for (int lane = 0; lane < kClusterSize; ++lane)
        {
            if ((validMask & (1u << static_cast<unsigned int>(lane))) == 0u)
            {
                continue;
            }
            const Float4POD& xq =
                snapshot.sorted_xq[static_cast<size_t>(atomBase + lane)];
            sum.x += xq.x;
            sum.y += xq.y;
            sum.z += xq.z;
        }
        const float invCount = 1.0f / static_cast<float>(validCount);
        centers[cluster] = invCount * sum;
    }
    return centers;
}

std::vector<Vec3> BuildSpongeSuperclusterCenters(
    const SpongeForceOnlySnapshot& snapshot, const std::vector<Vec3>& clusterCenters)
{
    const size_t superCount = snapshot.super_cluster_offsets.empty()
                                  ? 0
                                  : snapshot.super_cluster_offsets.size() - 1;
    std::vector<Vec3> centers(superCount, {0.0f, 0.0f, 0.0f});
    for (size_t superI = 0; superI < superCount; ++superI)
    {
        const int clusterBegin = snapshot.super_cluster_offsets[superI];
        const int clusterEnd = snapshot.super_cluster_offsets[superI + 1];
        Vec3 sum = {0.0f, 0.0f, 0.0f};
        int atomCount = 0;
        for (int cluster = clusterBegin; cluster < clusterEnd; ++cluster)
        {
            const int validCount =
                HostPopcount(snapshot.cluster_valid_masks[static_cast<size_t>(cluster)]);
            if (validCount == 0)
            {
                continue;
            }
            sum.x += clusterCenters[static_cast<size_t>(cluster)].x *
                     static_cast<float>(validCount);
            sum.y += clusterCenters[static_cast<size_t>(cluster)].y *
                     static_cast<float>(validCount);
            sum.z += clusterCenters[static_cast<size_t>(cluster)].z *
                     static_cast<float>(validCount);
            atomCount += validCount;
        }
        if (atomCount > 0)
        {
            const float invCount = 1.0f / static_cast<float>(atomCount);
            centers[superI] = invCount * sum;
        }
    }
    return centers;
}

int DetermineCanonicalShiftFromCenters(
    Vec3 centerI, Vec3 centerJ, const std::array<Vec3, kShiftCount>& shiftVecs)
{
    const Vec3 dr = centerJ - centerI;
    int bestShift = kCentralShiftId;
    float bestD2 = std::numeric_limits<float>::infinity();
    for (int shiftId = 0; shiftId < kShiftCount; ++shiftId)
    {
        const Vec3 shifted = dr - shiftVecs[static_cast<size_t>(shiftId)];
        const float d2 = Dot(shifted, shifted);
        if (d2 < bestD2)
        {
            bestD2 = d2;
            bestShift = shiftId;
        }
    }
    return bestShift;
}

int FindZeroShiftId(const std::array<Float4POD, kShiftCount>& shiftvec)
{
    int bestShift = kCentralShiftId;
    float bestNorm2 = std::numeric_limits<float>::infinity();
    for (int shift = 0; shift < kShiftCount; ++shift)
    {
        const Float4POD& v = shiftvec[static_cast<size_t>(shift)];
        const float norm2 = v.x * v.x + v.y * v.y + v.z * v.z;
        if (norm2 < bestNorm2)
        {
            bestNorm2 = norm2;
            bestShift = shift;
        }
    }
    return bestShift;
}

void AnalyzeSpongeSnapshot(const SpongeForceOnlySnapshot& snapshot)
{
    uint64_t total_cluster_valid_atoms = 0;
    uint64_t total_valid_j_atoms = 0;
    uint64_t total_local_j_atoms = 0;
    uint64_t total_active_i_clusters = 0;
    uint64_t total_active_i_atoms = 0;
    uint64_t total_potential_atom_pairs = 0;
    uint64_t total_excluded_i_atoms = 0;
    uint64_t full_valid_records = 0;
    uint64_t central_only_records = 0;
    uint64_t shifted_only_records = 0;
    uint64_t mixed_shift_records = 0;
    uint64_t central_only_potential_atom_pairs = 0;
    uint64_t shifted_only_potential_atom_pairs = 0;
    uint64_t mixed_shift_potential_atom_pairs = 0;
    uint64_t uniform_shift_records = 0;
    uint64_t uniform_shift_match_sci_records = 0;
    uint64_t uniform_shift_mismatch_sci_records = 0;
    uint64_t mixed_shift_id_records = 0;
    uint64_t uniform_shift_potential_atom_pairs = 0;
    uint64_t uniform_shift_match_sci_potential_atom_pairs = 0;
    uint64_t uniform_shift_mismatch_sci_potential_atom_pairs = 0;
    uint64_t mixed_shift_id_potential_atom_pairs = 0;
    uint64_t total_distinct_shift_ids = 0;
    int max_distinct_shift_ids = 0;
    std::array<uint64_t, kShiftCount + 1> distinct_shift_hist = {};
    const int supercluster_count = snapshot.super_cluster_offsets.empty()
                                       ? 0
                                       : static_cast<int>(
                                             snapshot.super_cluster_offsets.size()) -
                                             1;
    std::vector<unsigned int> supercluster_shift_masks(
        static_cast<size_t>(std::max(supercluster_count, 0)), 0u);
    std::vector<int> supercluster_sci_counts(
        static_cast<size_t>(std::max(supercluster_count, 0)), 0);
    std::vector<uint64_t> supercluster_record_counts(
        static_cast<size_t>(std::max(supercluster_count, 0)), 0ull);
    std::vector<uint64_t> supercluster_pair_counts(
        static_cast<size_t>(std::max(supercluster_count, 0)), 0ull);
    std::vector<int> supercluster_sizes;
    std::vector<int> active_i_clusters_per_record;
    std::vector<int> active_i_clusters_per_central_record;
    std::vector<int> active_i_clusters_per_shifted_record;
    std::vector<int> records_per_sci;
    std::vector<int> records_per_central_sci;
    std::vector<int> records_per_shifted_sci;
    std::vector<int> dense_active_bits_per_sci;
    std::vector<int> dense_active_bits_per_central_sci;
    std::vector<int> dense_active_bits_per_shifted_sci;
    uint64_t central_sci = 0;
    uint64_t shifted_sci = 0;
    uint64_t central_sci_records = 0;
    uint64_t shifted_sci_records = 0;
    uint64_t central_sci_pairs = 0;
    uint64_t shifted_sci_pairs = 0;
    uint64_t total_superj_blocks = 0;
    uint64_t canonical_match_blocks = 0;
    uint64_t canonical_mismatch_blocks = 0;
    uint64_t central_blocks = 0;
    uint64_t shifted_blocks = 0;
    uint64_t central_blocks_with_shifted_canonical = 0;
    uint64_t shifted_blocks_with_central_canonical = 0;
    uint64_t block_cluster_entries = 0;
    uint64_t block_split_records = 0;
    uint64_t total_imask_bits = 0;
    uint64_t cutoff_supported_imask_bits = 0;
    uint64_t cutoff_empty_imask_bits = 0;
    uint64_t central_imask_bits = 0;
    uint64_t central_cutoff_empty_imask_bits = 0;
    uint64_t shifted_imask_bits = 0;
    uint64_t shifted_cutoff_empty_imask_bits = 0;
    uint64_t total_true_cutoff_pairs_for_imask_bits = 0;
    std::array<uint64_t, 7> true_cutoff_pair_hist = {};
    const unsigned int full_valid_mask = (1u << kSplitJClusterSize) - 1u;
    const LTMatrix3 cell = MakeMatrix(snapshot.header.cell);
    const float cutoff_sq = snapshot.header.cutoff * snapshot.header.cutoff;
    const ClusterGeometryStats clusterGeometry =
        CollectMinimumImageClusterGeometry(
            snapshot.sorted_xq, snapshot.cluster_offsets,
            snapshot.cluster_valid_masks, cell);
    const std::vector<Vec3> clusterCenters = BuildSpongeClusterCenters(snapshot);
    const std::vector<Vec3> superclusterCenters =
        BuildSpongeSuperclusterCenters(snapshot, clusterCenters);
    supercluster_sizes.reserve(static_cast<size_t>(std::max(supercluster_count, 0)));
    for (int super_i = 0; super_i < supercluster_count; ++super_i)
    {
        supercluster_sizes.push_back(
            snapshot.super_cluster_offsets[static_cast<size_t>(super_i + 1)] -
            snapshot.super_cluster_offsets[static_cast<size_t>(super_i)]);
    }
    active_i_clusters_per_record.reserve(snapshot.records.size());
    active_i_clusters_per_central_record.reserve(snapshot.records.size());
    active_i_clusters_per_shifted_record.reserve(snapshot.records.size());
    records_per_sci.reserve(snapshot.sci.size());
    records_per_central_sci.reserve(snapshot.sci.size());
    records_per_shifted_sci.reserve(snapshot.sci.size());
    dense_active_bits_per_sci.reserve(snapshot.sci.size());
    dense_active_bits_per_central_sci.reserve(snapshot.sci.size());
    dense_active_bits_per_shifted_sci.reserve(snapshot.sci.size());
    std::array<Vec3, kShiftCount> shiftVecs = {};
    for (int shiftId = 0; shiftId < kShiftCount; ++shiftId)
    {
        shiftVecs[static_cast<size_t>(shiftId)] =
            ShiftVectorFromId(shiftId, cell);
    }
    auto truePairHistBin = [](int pairCount) -> int
    {
        if (pairCount <= 0)
        {
            return 0;
        }
        if (pairCount == 1)
        {
            return 1;
        }
        if (pairCount == 2)
        {
            return 2;
        }
        if (pairCount <= 4)
        {
            return 3;
        }
        if (pairCount <= 8)
        {
            return 4;
        }
        if (pairCount <= 16)
        {
            return 5;
        }
        return 6;
    };
    auto countTrueCutoffPairsForImaskBit =
        [&](const SpongeWarpJRecordPOD& record, const SpongeSciPOD& sci,
            int i_local) -> int
    {
        if (sci.supercluster_id < 0 ||
            static_cast<size_t>(sci.supercluster_id + 1) >=
                snapshot.super_cluster_offsets.size())
        {
            return 0;
        }
        if (record.cluster_j < 0 ||
            static_cast<size_t>(record.cluster_j) >= snapshot.cluster_offsets.size())
        {
            return 0;
        }
        const int cluster_i =
            snapshot.super_cluster_offsets[static_cast<size_t>(sci.supercluster_id)] +
            i_local;
        if (cluster_i < 0 ||
            static_cast<size_t>(cluster_i) >= snapshot.cluster_offsets.size())
        {
            return 0;
        }
        const unsigned int valid_mask_i =
            snapshot.cluster_valid_masks[static_cast<size_t>(cluster_i)];
        const unsigned int local_mask_i =
            snapshot.cluster_local_masks[static_cast<size_t>(cluster_i)];
        const unsigned int active_i_mask = valid_mask_i & local_mask_i;
        const unsigned int valid_mask_j =
            snapshot.cluster_valid_masks[static_cast<size_t>(record.cluster_j)];
        const unsigned int local_mask_j =
            snapshot.cluster_local_masks[static_cast<size_t>(record.cluster_j)];
        const int atom_base_i =
            snapshot.cluster_offsets[static_cast<size_t>(cluster_i)];
        const int atom_base_j =
            snapshot.cluster_offsets[static_cast<size_t>(record.cluster_j)];
        const Vec3 shift_vec =
            (sci.shift_id >= 0 && sci.shift_id < kShiftCount)
                ? shiftVecs[static_cast<size_t>(sci.shift_id)]
                : Vec3{0.0f, 0.0f, 0.0f};
        int pair_count = 0;
        for (int j_local = 0; j_local < kSplitJClusterSize; ++j_local)
        {
            const int j_lane = static_cast<int>(record.j_lane_base) + j_local;
            if (j_lane < 0 || j_lane >= kClusterSize ||
                (valid_mask_j & (1u << static_cast<unsigned int>(j_lane))) == 0u)
            {
                continue;
            }
            const Float4POD& xj =
                snapshot.sorted_xq[static_cast<size_t>(atom_base_j + j_lane)];
            for (int i_lane = 0; i_lane < kClusterSize; ++i_lane)
            {
                if ((active_i_mask & (1u << static_cast<unsigned int>(i_lane))) ==
                    0u)
                {
                    continue;
                }
                if ((record.pair_excl[static_cast<size_t>(
                         j_local * kClusterSize + i_lane)] &
                     (1u << static_cast<unsigned int>(i_local))) != 0u)
                {
                    continue;
                }
                if (sci.shift_id == kCentralShiftId && cluster_i == record.cluster_j &&
                    (local_mask_j & (1u << static_cast<unsigned int>(j_lane))) !=
                        0u &&
                    j_lane <= i_lane)
                {
                    continue;
                }
                const Float4POD& xi =
                    snapshot.sorted_xq[static_cast<size_t>(atom_base_i + i_lane)];
                const Vec3 dr = {shift_vec.x + xi.x - xj.x,
                                 shift_vec.y + xi.y - xj.y,
                                 shift_vec.z + xi.z - xj.z};
                const float dr2 = Dot(dr, dr);
                if (dr2 < cutoff_sq && dr2 != 0.0f)
                {
                    pair_count += 1;
                }
            }
        }
        return pair_count;
    };
    for (unsigned int valid_mask : snapshot.cluster_valid_masks)
    {
        total_cluster_valid_atoms += static_cast<uint64_t>(HostPopcount(valid_mask));
    }

    for (size_t sci_idx = 0; sci_idx < snapshot.sci.size(); ++sci_idx)
    {
        const SpongeSciPOD& sci = snapshot.sci[sci_idx];
        const int super_i = sci.supercluster_id;
        const bool sci_is_central = sci.shift_id == kCentralShiftId;
        if (sci_is_central)
        {
            central_sci += 1;
        }
        else
        {
            shifted_sci += 1;
        }
        if (super_i >= 0 && super_i < supercluster_count)
        {
            supercluster_sci_counts[static_cast<size_t>(super_i)] += 1;
            if (sci.shift_id >= 0 && sci.shift_id < kShiftCount)
            {
                supercluster_shift_masks[static_cast<size_t>(super_i)] |=
                    (1u << static_cast<unsigned int>(sci.shift_id));
            }
        }
        const int cluster_i_start = snapshot.super_cluster_offsets[super_i];
        const int cluster_i_end = snapshot.super_cluster_offsets[super_i + 1];
        const int active_cluster_count = cluster_i_end - cluster_i_start;
        const int record_begin = snapshot.record_offsets[sci_idx];
        const int record_end = snapshot.record_offsets[sci_idx + 1];
        const uint64_t sci_record_count =
            static_cast<uint64_t>(std::max(record_end - record_begin, 0));
        records_per_sci.push_back(record_end - record_begin);
        if (sci_is_central)
        {
            central_sci_records += sci_record_count;
            records_per_central_sci.push_back(record_end - record_begin);
        }
        else
        {
            shifted_sci_records += sci_record_count;
            records_per_shifted_sci.push_back(record_end - record_begin);
        }
        if (super_i >= 0 && super_i < supercluster_count)
        {
            supercluster_record_counts[static_cast<size_t>(super_i)] +=
                sci_record_count;
        }

        struct AggregatedCluster
        {
            int clusterJ = -1;
            unsigned int splitMask = 0u;
        };
        std::vector<AggregatedCluster> aggregated;
        std::unordered_map<int, size_t> clusterToIndex;
        aggregated.reserve(static_cast<size_t>(std::max(record_end - record_begin, 0)));
        int sci_dense_active_bits = 0;
        for (int record_idx = record_begin; record_idx < record_end; ++record_idx)
        {
            const SpongeWarpJRecordPOD& record = snapshot.records[record_idx];
            const int split = record.j_lane_base / kSplitJClusterSize;
            auto [it, inserted] =
                clusterToIndex.emplace(record.cluster_j, aggregated.size());
            if (inserted)
            {
                aggregated.push_back({record.cluster_j, 0u});
            }
            aggregated[it->second].splitMask |=
                (1u << static_cast<unsigned int>(split));
        }
        size_t blockBegin = 0;
        while (blockBegin < aggregated.size())
        {
            const int superJ =
                aggregated[blockBegin].clusterJ / kSuperClusterClusters;
            size_t blockEnd = blockBegin + 1;
            while (blockEnd < aggregated.size() &&
                   aggregated[blockEnd].clusterJ / kSuperClusterClusters == superJ)
            {
                blockEnd += 1;
            }
            total_superj_blocks += 1;
            block_cluster_entries += static_cast<uint64_t>(blockEnd - blockBegin);
            for (size_t idx = blockBegin; idx < blockEnd; ++idx)
            {
                block_split_records +=
                    static_cast<uint64_t>(HostPopcount(aggregated[idx].splitMask));
            }
            const int canonicalShift =
                (super_i >= 0 && super_i < static_cast<int>(superclusterCenters.size()) &&
                 superJ >= 0 && superJ < static_cast<int>(superclusterCenters.size()))
                    ? DetermineCanonicalShiftFromCenters(
                          superclusterCenters[static_cast<size_t>(super_i)],
                          superclusterCenters[static_cast<size_t>(superJ)], shiftVecs)
                    : sci.shift_id;
            if (sci.shift_id == canonicalShift)
            {
                canonical_match_blocks += 1;
            }
            else
            {
                canonical_mismatch_blocks += 1;
            }
            if (sci_is_central)
            {
                central_blocks += 1;
                if (canonicalShift != kCentralShiftId)
                {
                    central_blocks_with_shifted_canonical += 1;
                }
            }
            else
            {
                shifted_blocks += 1;
                if (canonicalShift == kCentralShiftId)
                {
                    shifted_blocks_with_central_canonical += 1;
                }
            }
            blockBegin = blockEnd;
        }

        for (int record_idx = record_begin; record_idx < record_end; ++record_idx)
        {
            const SpongeWarpJRecordPOD& record = snapshot.records[record_idx];
            const unsigned int valid_mask_j = record.valid_mask;
            const unsigned int local_mask_j = record.local_mask;
            const unsigned int active_i_mask = record.imask;
            const int valid_j_atoms = HostPopcount(valid_mask_j);
            const int local_j_atoms =
                HostPopcount(valid_mask_j & local_mask_j);
            int active_i_clusters = 0;
            int active_i_atoms = 0;
            int excluded_i_atoms = 0;
            if (valid_mask_j == full_valid_mask)
            {
                full_valid_records += 1;
            }
            for (int i_local = 0; i_local < active_cluster_count; ++i_local)
            {
                if ((active_i_mask & (1u << static_cast<unsigned int>(i_local))) == 0u)
                {
                    continue;
                }
                active_i_clusters += 1;
                const int true_cutoff_pairs =
                    countTrueCutoffPairsForImaskBit(record, sci, i_local);
                total_imask_bits += 1;
                total_true_cutoff_pairs_for_imask_bits +=
                    static_cast<uint64_t>(true_cutoff_pairs);
                true_cutoff_pair_hist[static_cast<size_t>(
                    truePairHistBin(true_cutoff_pairs))] += 1;
                if (true_cutoff_pairs > 0)
                {
                    cutoff_supported_imask_bits += 1;
                }
                else
                {
                    cutoff_empty_imask_bits += 1;
                }
                if (sci_is_central)
                {
                    central_imask_bits += 1;
                    if (true_cutoff_pairs == 0)
                    {
                        central_cutoff_empty_imask_bits += 1;
                    }
                }
                else
                {
                    shifted_imask_bits += 1;
                    if (true_cutoff_pairs == 0)
                    {
                        shifted_cutoff_empty_imask_bits += 1;
                    }
                }
                const int cluster_i = cluster_i_start + i_local;
                const unsigned int i_local_mask =
                    snapshot.cluster_valid_masks[cluster_i] &
                    snapshot.cluster_local_masks[cluster_i];
                active_i_atoms += HostPopcount(i_local_mask);
                for (int j_local = 0; j_local < kSplitJClusterSize; ++j_local)
                {
                    if ((valid_mask_j & (1u << static_cast<unsigned int>(j_local))) == 0u)
                    {
                        continue;
                    }
                    for (int i_lane = 0; i_lane < kClusterSize; ++i_lane)
                    {
                        if ((i_local_mask & (1u << static_cast<unsigned int>(i_lane))) == 0u)
                        {
                            continue;
                        }
                        const unsigned char excl_mask =
                            record.pair_excl[j_local * kClusterSize + i_lane];
                        if ((excl_mask & (1u << static_cast<unsigned int>(i_local))) != 0u)
                        {
                            excluded_i_atoms += 1;
                        }
                    }
                }
            }
            active_i_clusters_per_record.push_back(active_i_clusters);
            if (sci_is_central)
            {
                active_i_clusters_per_central_record.push_back(active_i_clusters);
            }
            else
            {
                active_i_clusters_per_shifted_record.push_back(active_i_clusters);
            }
            total_valid_j_atoms += static_cast<uint64_t>(valid_j_atoms);
            total_local_j_atoms += static_cast<uint64_t>(local_j_atoms);
            total_active_i_clusters += static_cast<uint64_t>(active_i_clusters);
            total_active_i_atoms += static_cast<uint64_t>(active_i_atoms);
            total_potential_atom_pairs +=
                static_cast<uint64_t>(active_i_atoms) *
                static_cast<uint64_t>(valid_j_atoms);
            if (sci_is_central)
            {
                central_sci_pairs +=
                    static_cast<uint64_t>(active_i_atoms) *
                    static_cast<uint64_t>(valid_j_atoms);
            }
            else
            {
                shifted_sci_pairs +=
                    static_cast<uint64_t>(active_i_atoms) *
                    static_cast<uint64_t>(valid_j_atoms);
            }
            if (super_i >= 0 && super_i < supercluster_count)
            {
                supercluster_pair_counts[static_cast<size_t>(super_i)] +=
                    static_cast<uint64_t>(active_i_atoms) *
                    static_cast<uint64_t>(valid_j_atoms);
            }
            total_excluded_i_atoms += static_cast<uint64_t>(excluded_i_atoms);
            for (int j_local = 0; j_local < kSplitJClusterSize; ++j_local)
            {
                const int absolute_j_lane =
                    static_cast<int>(record.j_lane_base) + j_local;
                for (int i_lane = 0; i_lane < kClusterSize; ++i_lane)
                {
                    unsigned int activeMask =
                        static_cast<unsigned int>(record.imask) &
                        ~static_cast<unsigned int>(
                            record.pair_excl[j_local * kClusterSize + i_lane]);
                    if (sci_is_central && absolute_j_lane <= i_lane)
                    {
                        const int central_i = record.cluster_j - cluster_i_start;
                        if (static_cast<unsigned int>(central_i) <
                            static_cast<unsigned int>(kSuperClusterClusters))
                        {
                            activeMask &=
                                ~(1u << static_cast<unsigned int>(central_i));
                        }
                    }
                    sci_dense_active_bits += HostPopcount(activeMask & 0xffu);
                }
            }
            bool has_central_shift = false;
            bool has_shifted_pair = false;
            uint64_t record_shift_bits = 0;
            if (record.pair_shift_index >= 0 &&
                static_cast<size_t>(record.pair_shift_index) <
                    snapshot.pair_shift_bits.size())
            {
                record_shift_bits =
                    snapshot.pair_shift_bits[static_cast<size_t>(
                        record.pair_shift_index)];
            }
            unsigned int record_shift_mask = 0u;
            int uniform_shift_id = -1;
            for (int i_local = 0; i_local < active_cluster_count; ++i_local)
            {
                if ((active_i_mask & (1u << static_cast<unsigned int>(i_local))) ==
                    0u)
                {
                    continue;
                }
                const int pair_shift_id = static_cast<int>(
                    (record_shift_bits >>
                     (static_cast<uint64_t>(i_local) * kPairShiftBits)) &
                    kPairShiftMask);
                if (pair_shift_id == kCentralShiftId)
                {
                    has_central_shift = true;
                }
                else
                {
                    has_shifted_pair = true;
                }
                if (pair_shift_id >= 0 && pair_shift_id < kShiftCount)
                {
                    record_shift_mask |=
                        (1u << static_cast<unsigned int>(pair_shift_id));
                    if (uniform_shift_id < 0)
                    {
                        uniform_shift_id = pair_shift_id;
                    }
                }
            }
            const uint64_t record_potential_pairs =
                static_cast<uint64_t>(active_i_atoms) *
                static_cast<uint64_t>(valid_j_atoms);
            const int distinct_shift_ids = HostPopcount(record_shift_mask);
            total_distinct_shift_ids +=
                static_cast<uint64_t>(distinct_shift_ids);
            max_distinct_shift_ids =
                std::max(max_distinct_shift_ids, distinct_shift_ids);
            if (distinct_shift_ids >= 0 && distinct_shift_ids <= kShiftCount)
            {
                distinct_shift_hist[static_cast<size_t>(distinct_shift_ids)] += 1;
            }
            if (distinct_shift_ids <= 1)
            {
                uniform_shift_records += 1;
                uniform_shift_potential_atom_pairs += record_potential_pairs;
                if (uniform_shift_id == sci.shift_id)
                {
                    uniform_shift_match_sci_records += 1;
                    uniform_shift_match_sci_potential_atom_pairs +=
                        record_potential_pairs;
                }
                else
                {
                    uniform_shift_mismatch_sci_records += 1;
                    uniform_shift_mismatch_sci_potential_atom_pairs +=
                        record_potential_pairs;
                }
            }
            else
            {
                mixed_shift_id_records += 1;
                mixed_shift_id_potential_atom_pairs += record_potential_pairs;
            }
            if (has_central_shift && has_shifted_pair)
            {
                mixed_shift_records += 1;
                mixed_shift_potential_atom_pairs += record_potential_pairs;
            }
            else if (has_shifted_pair)
            {
                shifted_only_records += 1;
                shifted_only_potential_atom_pairs += record_potential_pairs;
            }
            else
            {
                central_only_records += 1;
                central_only_potential_atom_pairs += record_potential_pairs;
            }
        }
        dense_active_bits_per_sci.push_back(sci_dense_active_bits);
        if (sci_is_central)
        {
            dense_active_bits_per_central_sci.push_back(sci_dense_active_bits);
        }
        else
        {
            dense_active_bits_per_shifted_sci.push_back(sci_dense_active_bits);
        }
    }

    const double record_count = static_cast<double>(snapshot.records.size());
    const double sci_count = static_cast<double>(snapshot.sci.size());
    const double cluster_count =
        static_cast<double>(snapshot.cluster_offsets.size());
    std::printf(
        "analysis=sponge clusters=%zu avg_valid_atoms_per_cluster=%.6f "
        "records=%zu sci=%zu atoms=%llu full_valid_records=%llu "
        "avg_records_per_sci=%.6f "
        "full_valid_ratio=%.6f avg_valid_j=%.6f avg_local_j=%.6f "
        "avg_active_i_clusters=%.6f avg_active_i_atoms=%.6f "
        "potential_atom_pairs=%llu avg_potential_pairs_per_record=%.6f "
        "central_only_records=%llu shifted_only_records=%llu "
        "mixed_shift_records=%llu "
        "central_only_pair_ratio=%.6f shifted_only_pair_ratio=%.6f "
        "mixed_shift_pair_ratio=%.6f "
        "excluded_i_atoms=%llu avg_excluded_i_atoms_per_record=%.6f\n",
        snapshot.cluster_offsets.size(),
        cluster_count > 0.0
            ? static_cast<double>(total_cluster_valid_atoms) / cluster_count
            : 0.0,
        snapshot.records.size(), snapshot.sci.size(),
        static_cast<unsigned long long>(snapshot.header.total_atom_numbers),
        static_cast<unsigned long long>(full_valid_records),
        sci_count > 0.0 ? record_count / sci_count : 0.0,
        record_count > 0.0 ? static_cast<double>(full_valid_records) / record_count
                           : 0.0,
        record_count > 0.0
            ? static_cast<double>(total_valid_j_atoms) / record_count
            : 0.0,
        record_count > 0.0
            ? static_cast<double>(total_local_j_atoms) / record_count
            : 0.0,
        record_count > 0.0
            ? static_cast<double>(total_active_i_clusters) / record_count
            : 0.0,
        record_count > 0.0
            ? static_cast<double>(total_active_i_atoms) / record_count
            : 0.0,
        static_cast<unsigned long long>(total_potential_atom_pairs),
        record_count > 0.0
            ? static_cast<double>(total_potential_atom_pairs) / record_count
            : 0.0,
        static_cast<unsigned long long>(central_only_records),
        static_cast<unsigned long long>(shifted_only_records),
        static_cast<unsigned long long>(mixed_shift_records),
        total_potential_atom_pairs > 0
            ? static_cast<double>(central_only_potential_atom_pairs) /
                  static_cast<double>(total_potential_atom_pairs)
            : 0.0,
        total_potential_atom_pairs > 0
            ? static_cast<double>(shifted_only_potential_atom_pairs) /
                  static_cast<double>(total_potential_atom_pairs)
            : 0.0,
        total_potential_atom_pairs > 0
            ? static_cast<double>(mixed_shift_potential_atom_pairs) /
                  static_cast<double>(total_potential_atom_pairs)
            : 0.0,
        static_cast<unsigned long long>(total_excluded_i_atoms),
        record_count > 0.0
            ? static_cast<double>(total_excluded_i_atoms) / record_count
            : 0.0);
    std::printf(
        "analysis=sponge-shift uniform_records=%llu "
        "uniform_record_ratio=%.6f uniform_match_sci_records=%llu "
        "uniform_match_sci_ratio=%.6f "
        "uniform_mismatch_sci_records=%llu "
        "uniform_mismatch_sci_ratio=%.6f "
        "mixed_shift_id_records=%llu mixed_shift_id_ratio=%.6f "
        "uniform_pair_ratio=%.6f uniform_match_sci_pair_ratio=%.6f "
        "uniform_mismatch_sci_pair_ratio=%.6f "
        "mixed_shift_id_pair_ratio=%.6f avg_distinct_shift_ids=%.6f "
        "max_distinct_shift_ids=%d hist=[1:%llu 2:%llu 3:%llu 4:%llu 5+:%llu]\n",
        static_cast<unsigned long long>(uniform_shift_records),
        record_count > 0.0
            ? static_cast<double>(uniform_shift_records) / record_count
            : 0.0,
        static_cast<unsigned long long>(uniform_shift_match_sci_records),
        record_count > 0.0
            ? static_cast<double>(uniform_shift_match_sci_records) / record_count
            : 0.0,
        static_cast<unsigned long long>(uniform_shift_mismatch_sci_records),
        record_count > 0.0
            ? static_cast<double>(uniform_shift_mismatch_sci_records) / record_count
            : 0.0,
        static_cast<unsigned long long>(mixed_shift_id_records),
        record_count > 0.0
            ? static_cast<double>(mixed_shift_id_records) / record_count
            : 0.0,
        total_potential_atom_pairs > 0
            ? static_cast<double>(uniform_shift_potential_atom_pairs) /
                  static_cast<double>(total_potential_atom_pairs)
            : 0.0,
        total_potential_atom_pairs > 0
            ? static_cast<double>(uniform_shift_match_sci_potential_atom_pairs) /
                  static_cast<double>(total_potential_atom_pairs)
            : 0.0,
        total_potential_atom_pairs > 0
            ? static_cast<double>(
                  uniform_shift_mismatch_sci_potential_atom_pairs) /
                  static_cast<double>(total_potential_atom_pairs)
            : 0.0,
        total_potential_atom_pairs > 0
            ? static_cast<double>(mixed_shift_id_potential_atom_pairs) /
                  static_cast<double>(total_potential_atom_pairs)
            : 0.0,
        record_count > 0.0
            ? static_cast<double>(total_distinct_shift_ids) / record_count
            : 0.0,
        max_distinct_shift_ids,
        static_cast<unsigned long long>(distinct_shift_hist[1]),
        static_cast<unsigned long long>(distinct_shift_hist[2]),
        static_cast<unsigned long long>(distinct_shift_hist[3]),
        static_cast<unsigned long long>(distinct_shift_hist[4]),
        static_cast<unsigned long long>(
            (distinct_shift_hist.size() > 5)
                ? std::accumulate(distinct_shift_hist.begin() + 5,
                                  distinct_shift_hist.end(), uint64_t{0})
                : uint64_t{0}));
    const uint64_t thin_supported_imask_bits =
        true_cutoff_pair_hist[1] + true_cutoff_pair_hist[2];
    std::printf(
        "analysis=sponge-imask-tightness imask_bits=%llu "
        "cutoff_supported_bits=%llu cutoff_supported_ratio=%.6f "
        "cutoff_empty_bits=%llu cutoff_empty_ratio=%.6f "
        "thin_supported_bits=%llu thin_supported_ratio=%.6f "
        "avg_true_cutoff_pairs_per_imask_bit=%.6f "
        "central_empty_ratio=%.6f shifted_empty_ratio=%.6f "
        "true_pair_hist=[0:%llu 1:%llu 2:%llu 3-4:%llu 5-8:%llu "
        "9-16:%llu 17+:%llu]\n",
        static_cast<unsigned long long>(total_imask_bits),
        static_cast<unsigned long long>(cutoff_supported_imask_bits),
        total_imask_bits > 0
            ? static_cast<double>(cutoff_supported_imask_bits) /
                  static_cast<double>(total_imask_bits)
            : 0.0,
        static_cast<unsigned long long>(cutoff_empty_imask_bits),
        total_imask_bits > 0
            ? static_cast<double>(cutoff_empty_imask_bits) /
                  static_cast<double>(total_imask_bits)
            : 0.0,
        static_cast<unsigned long long>(thin_supported_imask_bits),
        cutoff_supported_imask_bits > 0
            ? static_cast<double>(thin_supported_imask_bits) /
                  static_cast<double>(cutoff_supported_imask_bits)
            : 0.0,
        total_imask_bits > 0
            ? static_cast<double>(total_true_cutoff_pairs_for_imask_bits) /
                  static_cast<double>(total_imask_bits)
            : 0.0,
        central_imask_bits > 0
            ? static_cast<double>(central_cutoff_empty_imask_bits) /
                  static_cast<double>(central_imask_bits)
            : 0.0,
        shifted_imask_bits > 0
            ? static_cast<double>(shifted_cutoff_empty_imask_bits) /
                  static_cast<double>(shifted_imask_bits)
            : 0.0,
        static_cast<unsigned long long>(true_cutoff_pair_hist[0]),
        static_cast<unsigned long long>(true_cutoff_pair_hist[1]),
        static_cast<unsigned long long>(true_cutoff_pair_hist[2]),
        static_cast<unsigned long long>(true_cutoff_pair_hist[3]),
        static_cast<unsigned long long>(true_cutoff_pair_hist[4]),
        static_cast<unsigned long long>(true_cutoff_pair_hist[5]),
        static_cast<unsigned long long>(true_cutoff_pair_hist[6]));

    uint64_t active_superclusters = 0;
    uint64_t total_active_shifts = 0;
    int max_active_shifts = 0;
    std::array<uint64_t, 6> active_shift_hist = {};
    for (int super_i = 0; super_i < supercluster_count; ++super_i)
    {
        const int sci_count_for_super =
            supercluster_sci_counts[static_cast<size_t>(super_i)];
        if (sci_count_for_super <= 0)
        {
            continue;
        }
        active_superclusters += 1;
        const int active_shifts = HostPopcount(
            supercluster_shift_masks[static_cast<size_t>(super_i)]);
        total_active_shifts += static_cast<uint64_t>(active_shifts);
        max_active_shifts = std::max(max_active_shifts, active_shifts);
        const int hist_bin =
            std::min(std::max(active_shifts, 0), static_cast<int>(active_shift_hist.size()) - 1);
        active_shift_hist[static_cast<size_t>(hist_bin)] += 1;
    }
    std::printf(
        "analysis=sponge-sci-shift superclusters_with_sci=%llu "
        "central_sci=%llu shifted_sci=%llu central_sci_ratio=%.6f "
        "shifted_sci_ratio=%.6f avg_active_shifts_per_supercluster=%.6f "
        "max_active_shifts_per_supercluster=%d "
        "avg_sci_per_supercluster=%.6f avg_records_per_central_sci=%.6f "
        "avg_records_per_shifted_sci=%.6f avg_pairs_per_central_sci=%.6f "
        "avg_pairs_per_shifted_sci=%.6f shift_hist=[1:%llu 2:%llu 3:%llu 4:%llu 5+:%llu]\n",
        static_cast<unsigned long long>(active_superclusters),
        static_cast<unsigned long long>(central_sci),
        static_cast<unsigned long long>(shifted_sci),
        sci_count > 0.0 ? static_cast<double>(central_sci) / sci_count : 0.0,
        sci_count > 0.0 ? static_cast<double>(shifted_sci) / sci_count : 0.0,
        active_superclusters > 0
            ? static_cast<double>(total_active_shifts) /
                  static_cast<double>(active_superclusters)
            : 0.0,
        max_active_shifts,
        active_superclusters > 0
            ? static_cast<double>(snapshot.sci.size()) /
                  static_cast<double>(active_superclusters)
            : 0.0,
        central_sci > 0
            ? static_cast<double>(central_sci_records) /
                  static_cast<double>(central_sci)
            : 0.0,
        shifted_sci > 0
            ? static_cast<double>(shifted_sci_records) /
                  static_cast<double>(shifted_sci)
            : 0.0,
        central_sci > 0
            ? static_cast<double>(central_sci_pairs) /
                  static_cast<double>(central_sci)
            : 0.0,
        shifted_sci > 0
            ? static_cast<double>(shifted_sci_pairs) /
                  static_cast<double>(shifted_sci)
            : 0.0,
        static_cast<unsigned long long>(active_shift_hist[1]),
        static_cast<unsigned long long>(active_shift_hist[2]),
        static_cast<unsigned long long>(active_shift_hist[3]),
        static_cast<unsigned long long>(active_shift_hist[4]),
        static_cast<unsigned long long>(
            active_shift_hist[5]));
    std::printf(
        "analysis=sponge-superj-shift blocks=%llu canonical_match_blocks=%llu "
        "canonical_match_ratio=%.6f canonical_mismatch_blocks=%llu "
        "canonical_mismatch_ratio=%.6f central_blocks=%llu shifted_blocks=%llu "
        "central_blocks_with_shifted_canonical=%llu "
        "central_shift_escape_ratio=%.6f "
        "shifted_blocks_with_central_canonical=%llu "
        "shifted_to_central_ratio=%.6f avg_cluster_entries_per_block=%.6f "
        "avg_split_records_per_block=%.6f\n",
        static_cast<unsigned long long>(total_superj_blocks),
        static_cast<unsigned long long>(canonical_match_blocks),
        total_superj_blocks > 0
            ? static_cast<double>(canonical_match_blocks) /
                  static_cast<double>(total_superj_blocks)
            : 0.0,
        static_cast<unsigned long long>(canonical_mismatch_blocks),
        total_superj_blocks > 0
            ? static_cast<double>(canonical_mismatch_blocks) /
                  static_cast<double>(total_superj_blocks)
            : 0.0,
        static_cast<unsigned long long>(central_blocks),
        static_cast<unsigned long long>(shifted_blocks),
        static_cast<unsigned long long>(central_blocks_with_shifted_canonical),
        central_blocks > 0
            ? static_cast<double>(central_blocks_with_shifted_canonical) /
                  static_cast<double>(central_blocks)
            : 0.0,
        static_cast<unsigned long long>(shifted_blocks_with_central_canonical),
        shifted_blocks > 0
            ? static_cast<double>(shifted_blocks_with_central_canonical) /
                  static_cast<double>(shifted_blocks)
            : 0.0,
        total_superj_blocks > 0
            ? static_cast<double>(block_cluster_entries) /
                  static_cast<double>(total_superj_blocks)
            : 0.0,
        total_superj_blocks > 0
            ? static_cast<double>(block_split_records) /
                  static_cast<double>(total_superj_blocks)
            : 0.0);

    std::sort(supercluster_sizes.begin(), supercluster_sizes.end());
    std::sort(active_i_clusters_per_record.begin(),
              active_i_clusters_per_record.end());
    std::sort(active_i_clusters_per_central_record.begin(),
              active_i_clusters_per_central_record.end());
    std::sort(active_i_clusters_per_shifted_record.begin(),
              active_i_clusters_per_shifted_record.end());
    std::sort(records_per_sci.begin(), records_per_sci.end());
    std::sort(records_per_central_sci.begin(), records_per_central_sci.end());
    std::sort(records_per_shifted_sci.begin(), records_per_shifted_sci.end());
    std::sort(dense_active_bits_per_sci.begin(),
              dense_active_bits_per_sci.end());
    std::sort(dense_active_bits_per_central_sci.begin(),
              dense_active_bits_per_central_sci.end());
    std::sort(dense_active_bits_per_shifted_sci.begin(),
              dense_active_bits_per_shifted_sci.end());

    std::array<uint64_t, kSuperClusterClusters + 1> supercluster_size_hist = {};
    for (int size : supercluster_sizes)
    {
        const int bin = std::min(std::max(size, 0), kSuperClusterClusters);
        supercluster_size_hist[static_cast<size_t>(bin)] += 1;
    }

    std::printf(
        "analysis=sponge-shape superclusters=%d "
        "supercluster_size_avg=%.6f p50=%.2f p90=%.2f p99=%.2f "
        "size_hist=[1:%llu 2:%llu 3:%llu 4:%llu 5:%llu 6:%llu 7:%llu 8:%llu] "
        "avg_active_i_clusters_per_record=%.6f p50=%.2f p90=%.2f "
        "central_avg_active_i_clusters_per_record=%.6f central_p90=%.2f "
        "shifted_avg_active_i_clusters_per_record=%.6f shifted_p90=%.2f\n",
        supercluster_count,
        AverageFromValues(supercluster_sizes),
        PercentileFromSorted(supercluster_sizes, 0.50),
        PercentileFromSorted(supercluster_sizes, 0.90),
        PercentileFromSorted(supercluster_sizes, 0.99),
        static_cast<unsigned long long>(supercluster_size_hist[1]),
        static_cast<unsigned long long>(supercluster_size_hist[2]),
        static_cast<unsigned long long>(supercluster_size_hist[3]),
        static_cast<unsigned long long>(supercluster_size_hist[4]),
        static_cast<unsigned long long>(supercluster_size_hist[5]),
        static_cast<unsigned long long>(supercluster_size_hist[6]),
        static_cast<unsigned long long>(supercluster_size_hist[7]),
        static_cast<unsigned long long>(supercluster_size_hist[8]),
        AverageFromValues(active_i_clusters_per_record),
        PercentileFromSorted(active_i_clusters_per_record, 0.50),
        PercentileFromSorted(active_i_clusters_per_record, 0.90),
        AverageFromValues(active_i_clusters_per_central_record),
        PercentileFromSorted(active_i_clusters_per_central_record, 0.90),
        AverageFromValues(active_i_clusters_per_shifted_record),
        PercentileFromSorted(active_i_clusters_per_shifted_record, 0.90));
    PrintClusterGeometryStats("sponge-cluster-geom", clusterGeometry);
    AnalyzeSpongeWaterClusterComposition(snapshot);
    std::printf(
        "analysis=sponge-sci-shape avg_records_per_sci=%.6f p50=%.2f "
        "p90=%.2f p99=%.2f central_p90_records=%.2f shifted_p90_records=%.2f "
        "avg_dense_active_bits_per_sci=%.6f p90_dense_active_bits_per_sci=%.2f "
        "central_avg_dense_active_bits_per_sci=%.6f central_p90=%.2f "
        "shifted_avg_dense_active_bits_per_sci=%.6f shifted_p90=%.2f\n",
        AverageFromValues(records_per_sci),
        PercentileFromSorted(records_per_sci, 0.50),
        PercentileFromSorted(records_per_sci, 0.90),
        PercentileFromSorted(records_per_sci, 0.99),
        PercentileFromSorted(records_per_central_sci, 0.90),
        PercentileFromSorted(records_per_shifted_sci, 0.90),
        AverageFromValues(dense_active_bits_per_sci),
        PercentileFromSorted(dense_active_bits_per_sci, 0.90),
        AverageFromValues(dense_active_bits_per_central_sci),
        PercentileFromSorted(dense_active_bits_per_central_sci, 0.90),
        AverageFromValues(dense_active_bits_per_shifted_sci),
        PercentileFromSorted(dense_active_bits_per_shifted_sci, 0.90));
}

void AnalyzeGromacsForceWriteLocality(const GromacsPairlistSnapshot& snapshot)
{
    struct WriteTraceStats
    {
        uint64_t sectorEvents = 0;
        uint64_t jSectorEvents = 0;
        uint64_t iSectorEvents = 0;
        uint64_t uniqueSectors = 0;
        uint64_t reuseEvents = 0;
        uint64_t reuseDistanceSum = 0;
        uint64_t shortReuse64 = 0;
        uint64_t shortReuse256 = 0;
        uint64_t shortReuse1024 = 0;
        uint64_t sameSectorTransitions = 0;
        uint64_t contiguousForwardTransitions = 0;
        uint64_t backwardTransitions = 0;
        uint64_t largeJumpTransitions = 0;
        uint64_t transitionAbsDeltaSum = 0;
        uint32_t maxReuseDistance = 0;
        int64_t previousSector = -1;
        std::vector<int64_t> lastTouch;
        std::vector<uint32_t> reuseDistances;
    };

    const uint64_t sectorCount =
        (snapshot.header.total_atom_numbers + 1ull) / 2ull;
    WriteTraceStats stats;
    stats.lastTouch.assign(static_cast<size_t>(sectorCount), -1);
    stats.reuseDistances.reserve(snapshot.cjpacked.size() * 2);

    auto touchSector = [&](uint64_t sector, bool isJWrite)
    {
        if (sector >= sectorCount)
        {
            return;
        }
        if (stats.previousSector >= 0)
        {
            const int64_t delta =
                static_cast<int64_t>(sector) - stats.previousSector;
            if (delta == 0)
            {
                stats.sameSectorTransitions += 1;
            }
            if (delta == 1)
            {
                stats.contiguousForwardTransitions += 1;
            }
            if (delta < 0)
            {
                stats.backwardTransitions += 1;
            }
            if (std::llabs(delta) >
                (kSuperClusterClusters * kClusterSize / 2))
            {
                stats.largeJumpTransitions += 1;
            }
            stats.transitionAbsDeltaSum +=
                static_cast<uint64_t>(std::llabs(delta));
        }
        stats.previousSector = static_cast<int64_t>(sector);

        int64_t& last = stats.lastTouch[static_cast<size_t>(sector)];
        if (last >= 0)
        {
            const uint64_t distance =
                stats.sectorEvents - static_cast<uint64_t>(last);
            const uint32_t clampedDistance =
                static_cast<uint32_t>(std::min<uint64_t>(
                    distance, std::numeric_limits<uint32_t>::max()));
            stats.reuseEvents += 1;
            stats.reuseDistanceSum += distance;
            stats.maxReuseDistance =
                std::max(stats.maxReuseDistance, clampedDistance);
            stats.reuseDistances.push_back(clampedDistance);
            if (distance <= 64)
            {
                stats.shortReuse64 += 1;
            }
            if (distance <= 256)
            {
                stats.shortReuse256 += 1;
            }
            if (distance <= 1024)
            {
                stats.shortReuse1024 += 1;
            }
        }
        else
        {
            stats.uniqueSectors += 1;
        }
        last = static_cast<int64_t>(stats.sectorEvents);
        stats.sectorEvents += 1;
        if (isJWrite)
        {
            stats.jSectorEvents += 1;
        }
        else
        {
            stats.iSectorEvents += 1;
        }
    };

    for (const GromacsSciPOD& sci : snapshot.sci)
    {
        for (int packedIdx = sci.cjPackedBegin; packedIdx < sci.cjPackedEnd;
             ++packedIdx)
        {
            const GromacsCjPackedPOD& packed =
                snapshot.cjpacked[static_cast<size_t>(packedIdx)];
            for (int split = 0; split < kWarpSplitCount; ++split)
            {
                const unsigned int imask = packed.imei[split].imask;
                for (int jm = 0; jm < kJGroupSize; ++jm)
                {
                    const int clusterJ = packed.cj[jm];
                    if (clusterJ < 0)
                    {
                        continue;
                    }
                    const unsigned int jmMask =
                        ((1u << kSuperClusterClusters) - 1u)
                        << (jm * kSuperClusterClusters);
                    if ((imask & jmMask) == 0u)
                    {
                        continue;
                    }
                    const uint64_t atomBase =
                        static_cast<uint64_t>(clusterJ) * kClusterSize +
                        static_cast<uint64_t>(split) * kSplitJClusterSize;
                    for (int localPair = 0; localPair < kSplitJClusterSize;
                         localPair += 2)
                    {
                        touchSector((atomBase + localPair) / 2ull, true);
                    }
                }
            }
        }

        const uint64_t superAtomBase =
            static_cast<uint64_t>(sci.sci) * kSuperClusterClusters *
            kClusterSize;
        for (int split = 0; split < kWarpSplitCount; ++split)
        {
            for (int localCluster = 0; localCluster < kSuperClusterClusters;
                 ++localCluster)
            {
                const uint64_t clusterAtomBase =
                    superAtomBase +
                    static_cast<uint64_t>(localCluster) * kClusterSize;
                for (int localPair = 0; localPair < kClusterSize;
                     localPair += 2)
                {
                    touchSector((clusterAtomBase + localPair) / 2ull, false);
                }
            }
        }
    }

    std::sort(stats.reuseDistances.begin(), stats.reuseDistances.end());
    const double transitions =
        static_cast<double>(std::max<uint64_t>(stats.sectorEvents - 1, 1));
    const double reuseEvents =
        static_cast<double>(std::max<uint64_t>(stats.reuseEvents, 1));
    std::printf(
        "analysis=gromacs-write-locality sector_events=%llu j_sector_events=%llu "
        "i_sector_events=%llu unique_sectors=%llu sector_reuse_ratio=%.6f "
        "reuse_distance_avg=%.6f reuse_distance_p50=%.2f "
        "reuse_distance_p90=%.2f reuse_distance_p99=%.2f "
        "short_reuse64_ratio=%.6f short_reuse256_ratio=%.6f "
        "short_reuse1024_ratio=%.6f same_sector_transition_ratio=%.6f "
        "contiguous_sector_transition_ratio=%.6f backward_sector_transition_ratio=%.6f "
        "large_sector_jump_ratio=%.6f avg_abs_sector_delta=%.6f "
        "max_reuse_distance=%u\n",
        static_cast<unsigned long long>(stats.sectorEvents),
        static_cast<unsigned long long>(stats.jSectorEvents),
        static_cast<unsigned long long>(stats.iSectorEvents),
        static_cast<unsigned long long>(stats.uniqueSectors),
        stats.sectorEvents > 0
            ? static_cast<double>(stats.reuseEvents) /
                  static_cast<double>(stats.sectorEvents)
            : 0.0,
        static_cast<double>(stats.reuseDistanceSum) / reuseEvents,
        PercentileFromSortedU32(stats.reuseDistances, 0.50),
        PercentileFromSortedU32(stats.reuseDistances, 0.90),
        PercentileFromSortedU32(stats.reuseDistances, 0.99),
        static_cast<double>(stats.shortReuse64) / reuseEvents,
        static_cast<double>(stats.shortReuse256) / reuseEvents,
        static_cast<double>(stats.shortReuse1024) / reuseEvents,
        static_cast<double>(stats.sameSectorTransitions) / transitions,
        static_cast<double>(stats.contiguousForwardTransitions) / transitions,
        static_cast<double>(stats.backwardTransitions) / transitions,
        static_cast<double>(stats.largeJumpTransitions) / transitions,
        static_cast<double>(stats.transitionAbsDeltaSum) / transitions,
        stats.maxReuseDistance);
}

void AnalyzeGromacsPerAtomOutputWriters(const GromacsPairlistSnapshot& snapshot)
{
    const auto atomCount =
        static_cast<size_t>(snapshot.header.total_atom_numbers);
    if (atomCount == 0 || snapshot.sorted_atom_ids.empty())
    {
        std::printf(
            "analysis=gmxpacked-peratom-output-writers atoms=%zu valid_write_events=0 "
            "atoms_written=0 atoms_unique_writer=0 atoms_multi_writer=0 "
            "unique_writer_ratio=0.000000 multi_writer_ratio=0.000000 "
            "writer_count_avg=0.000000 writer_count_p50=0.00 "
            "writer_count_p90=0.00 writer_count_p99=0.00 "
            "writer_count_max=0 central_events=0 shifted_events=0\n",
            atomCount);
        return;
    }

    const int centralShiftId = FindZeroShiftId(snapshot.header.shiftvec);
    std::vector<uint32_t> writerCounts(atomCount, 0u);
    std::vector<uint32_t> centralWriterCounts(atomCount, 0u);
    std::vector<uint32_t> shiftedWriterCounts(atomCount, 0u);
    uint64_t validWriteEvents = 0;
    uint64_t centralEvents = 0;
    uint64_t shiftedEvents = 0;
    uint64_t invalidSlots = 0;
    uint64_t outOfRangeSlots = 0;

    for (const GromacsSciPOD& sci : snapshot.sci)
    {
        const int sortedSuperIBase =
            sci.sci * kSuperClusterClusters * kClusterSize;
        const bool central = (sci.shift == centralShiftId);
        for (int localCluster = 0; localCluster < kSuperClusterClusters;
             ++localCluster)
        {
            for (int lane = 0; lane < kClusterSize; ++lane)
            {
                const int sortedI =
                    sortedSuperIBase + localCluster * kClusterSize + lane;
                if (sortedI < 0 ||
                    static_cast<size_t>(sortedI) >= snapshot.sorted_atom_ids.size())
                {
                    outOfRangeSlots += 1;
                    continue;
                }
                const int atom = snapshot.sorted_atom_ids[static_cast<size_t>(sortedI)];
                if (atom < 0)
                {
                    invalidSlots += 1;
                    continue;
                }
                if (static_cast<size_t>(atom) >= atomCount)
                {
                    outOfRangeSlots += 1;
                    continue;
                }
                writerCounts[static_cast<size_t>(atom)] += 1u;
                if (central)
                {
                    centralWriterCounts[static_cast<size_t>(atom)] += 1u;
                    centralEvents += 1;
                }
                else
                {
                    shiftedWriterCounts[static_cast<size_t>(atom)] += 1u;
                    shiftedEvents += 1;
                }
                validWriteEvents += 1;
            }
        }
    }

    std::vector<uint32_t> nonzeroCounts;
    nonzeroCounts.reserve(atomCount);
    uint64_t atomsUniqueWriter = 0;
    uint64_t atomsMultiWriter = 0;
    uint64_t atomsCentralMultiWriter = 0;
    uint64_t atomsShiftedMultiWriter = 0;
    uint64_t totalWriterCount = 0;
    uint32_t maxWriterCount = 0;
    uint32_t maxCentralWriterCount = 0;
    uint32_t maxShiftedWriterCount = 0;
    std::array<uint64_t, 9> writerHist = {};
    for (size_t atom = 0; atom < atomCount; ++atom)
    {
        const uint32_t count = writerCounts[atom];
        maxCentralWriterCount =
            std::max(maxCentralWriterCount, centralWriterCounts[atom]);
        maxShiftedWriterCount =
            std::max(maxShiftedWriterCount, shiftedWriterCounts[atom]);
        if (centralWriterCounts[atom] > 1u)
        {
            atomsCentralMultiWriter += 1;
        }
        if (shiftedWriterCounts[atom] > 1u)
        {
            atomsShiftedMultiWriter += 1;
        }
        if (count == 0u)
        {
            continue;
        }
        nonzeroCounts.push_back(count);
        totalWriterCount += count;
        maxWriterCount = std::max(maxWriterCount, count);
        writerHist[std::min<size_t>(count, writerHist.size() - 1)] += 1;
        if (count == 1u)
        {
            atomsUniqueWriter += 1;
        }
        else
        {
            atomsMultiWriter += 1;
        }
    }
    std::sort(nonzeroCounts.begin(), nonzeroCounts.end());
    const uint64_t atomsWritten = static_cast<uint64_t>(nonzeroCounts.size());
    const double atomsWrittenDenom =
        static_cast<double>(std::max<uint64_t>(atomsWritten, 1));

    std::printf(
        "analysis=gmxpacked-peratom-output-writers atoms=%zu sci=%zu "
        "valid_write_events=%llu central_events=%llu shifted_events=%llu "
        "invalid_slots=%llu out_of_range_slots=%llu atoms_written=%llu "
        "atoms_unique_writer=%llu atoms_multi_writer=%llu "
        "unique_writer_ratio=%.6f multi_writer_ratio=%.6f "
        "writer_count_avg=%.6f writer_count_p50=%.2f writer_count_p90=%.2f "
        "writer_count_p99=%.2f writer_count_max=%u central_multi_atoms=%llu "
        "shifted_multi_atoms=%llu central_writer_max=%u shifted_writer_max=%u "
        "hist1=%llu hist2=%llu hist3=%llu hist4=%llu hist5=%llu hist6=%llu "
        "hist7=%llu hist8plus=%llu\n",
        atomCount, snapshot.sci.size(),
        static_cast<unsigned long long>(validWriteEvents),
        static_cast<unsigned long long>(centralEvents),
        static_cast<unsigned long long>(shiftedEvents),
        static_cast<unsigned long long>(invalidSlots),
        static_cast<unsigned long long>(outOfRangeSlots),
        static_cast<unsigned long long>(atomsWritten),
        static_cast<unsigned long long>(atomsUniqueWriter),
        static_cast<unsigned long long>(atomsMultiWriter),
        static_cast<double>(atomsUniqueWriter) / atomsWrittenDenom,
        static_cast<double>(atomsMultiWriter) / atomsWrittenDenom,
        atomsWritten > 0
            ? static_cast<double>(totalWriterCount) /
                  static_cast<double>(atomsWritten)
            : 0.0,
        PercentileFromSortedU32(nonzeroCounts, 0.50),
        PercentileFromSortedU32(nonzeroCounts, 0.90),
        PercentileFromSortedU32(nonzeroCounts, 0.99), maxWriterCount,
        static_cast<unsigned long long>(atomsCentralMultiWriter),
        static_cast<unsigned long long>(atomsShiftedMultiWriter),
        maxCentralWriterCount, maxShiftedWriterCount,
        static_cast<unsigned long long>(writerHist[1]),
        static_cast<unsigned long long>(writerHist[2]),
        static_cast<unsigned long long>(writerHist[3]),
        static_cast<unsigned long long>(writerHist[4]),
        static_cast<unsigned long long>(writerHist[5]),
        static_cast<unsigned long long>(writerHist[6]),
        static_cast<unsigned long long>(writerHist[7]),
        static_cast<unsigned long long>(writerHist[8]));
}

void AnalyzeGromacsSnapshot(const GromacsPairlistSnapshot& snapshot)
{
    struct GromacsSciEntryTrace
    {
        int sciIdx = -1;
        int superI = -1;
        int shift = -1;
        int cjpackedSpan = 0;
        int activeJClusters = 0;
        int activeRecordsForSci = 0;
        int firstClusterJ = -1;
        int lastClusterJ = -1;
        int contiguousTransitions = 0;
        int totalTransitions = 0;
        int backwardTransitions = 0;
        int sameSuperTransitions = 0;
        int largeJumpTransitions = 0;
        int maxAbsDelta = 0;
        uint64_t absClusterJDelta = 0;
    };
    const size_t cluster_count =
        snapshot.cluster_offsets.empty()
            ? (snapshot.sorted_xq.size() + kClusterSize - 1) /
                  static_cast<size_t>(kClusterSize)
            : snapshot.cluster_offsets.size();
    std::vector<int> geometryClusterOffsets = snapshot.cluster_offsets;
    std::vector<unsigned int> geometryClusterValidMasks =
        snapshot.cluster_valid_masks;
    if (geometryClusterOffsets.empty())
    {
        geometryClusterOffsets.resize(cluster_count);
        geometryClusterValidMasks.resize(cluster_count, 0u);
        for (size_t cluster = 0; cluster < cluster_count; ++cluster)
        {
            geometryClusterOffsets[cluster] =
                static_cast<int>(cluster * static_cast<size_t>(kClusterSize));
            unsigned int validMask = 0u;
            for (int lane = 0; lane < kClusterSize; ++lane)
            {
                const size_t atomIndex =
                    cluster * static_cast<size_t>(kClusterSize) +
                    static_cast<size_t>(lane);
                if (atomIndex < snapshot.sorted_xq.size())
                {
                    validMask |= 1u << static_cast<unsigned int>(lane);
                }
            }
            geometryClusterValidMasks[cluster] = validMask;
        }
    }
    LTMatrix3 geometryCell = MakeMatrix(snapshot.header.cell);
    if (!CellLooksUsable(geometryCell))
    {
        geometryCell =
            MakeOrthorhombicCellFromShiftVec(snapshot.header.shiftvec);
    }
    const ClusterGeometryStats clusterGeometry =
        CollectMinimumImageClusterGeometry(
            snapshot.sorted_xq, geometryClusterOffsets,
            geometryClusterValidMasks, geometryCell);
    uint64_t total_valid_j_atoms = 0;
    uint64_t total_local_j_atoms = 0;
    uint64_t total_active_i_clusters = 0;
    uint64_t total_active_i_atoms = 0;
    uint64_t total_potential_atom_pairs = 0;
    uint64_t active_j_clusters = 0;
    uint64_t active_split_j_clusters = 0;
    uint64_t active_records = 0;
    uint64_t central_only_records = 0;
    uint64_t shifted_only_records = 0;
    uint64_t central_only_potential_atom_pairs = 0;
    uint64_t shifted_only_potential_atom_pairs = 0;
    int max_super_i = -1;
    for (const GromacsSciPOD& sci : snapshot.sci)
    {
        max_super_i = std::max(max_super_i, sci.sci);
    }
    const int supercluster_count = max_super_i + 1;
    std::vector<unsigned int> supercluster_shift_masks(
        static_cast<size_t>(std::max(supercluster_count, 0)), 0u);
    std::vector<int> supercluster_sci_counts(
        static_cast<size_t>(std::max(supercluster_count, 0)), 0);
    uint64_t central_sci = 0;
    uint64_t shifted_sci = 0;
    uint64_t central_sci_records = 0;
    uint64_t shifted_sci_records = 0;
    uint64_t central_sci_pairs = 0;
    uint64_t shifted_sci_pairs = 0;
    uint64_t split_entries_with_excl = 0;
    uint64_t central_same_cluster_split_records = 0;
    std::vector<unsigned char> used_excl(snapshot.excl.size(), 0u);
    uint64_t full_cjpacked_groups = 0;
    uint64_t partial_cjpacked_groups = 0;
    uint64_t total_active_j_slots = 0;
    uint64_t total_contiguous_transitions = 0;
    uint64_t total_backward_transitions = 0;
    uint64_t total_same_super_transitions = 0;
    uint64_t total_large_jump_transitions = 0;
    uint64_t total_abs_cluster_j_delta = 0;
    int64_t total_signed_cluster_j_delta = 0;
    std::array<uint64_t, 2> kind_sci_counts = {};
    std::array<uint64_t, 2> kind_cjpacked_spans = {};
    std::array<uint64_t, 2> kind_active_j_clusters = {};
    std::array<uint64_t, 2> kind_active_records = {};
    std::array<uint64_t, 2> kind_transitions = {};
    std::array<uint64_t, 2> kind_contiguous_transitions = {};
    std::array<uint64_t, 2> kind_backward_transitions = {};
    std::array<uint64_t, 2> kind_same_super_transitions = {};
    std::array<uint64_t, 2> kind_large_jump_transitions = {};
    std::array<uint64_t, 2> kind_abs_cluster_j_delta = {};
    uint64_t sci_order_transitions = 0;
    uint64_t sci_order_same_shift = 0;
    uint64_t sci_order_same_super = 0;
    uint64_t sci_order_backward_super = 0;
    uint64_t sci_order_large_super_jump = 0;
    uint64_t sci_order_abs_super_delta = 0;
    int previousSciSuper = -1;
    int previousSciShift = -1;
    std::array<uint64_t, kJGroupSize + 1> active_j_per_cjpacked_hist = {};
    std::vector<int> cjpacked_per_sci;
    std::vector<int> active_j_per_sci;
    std::vector<int> active_records_per_sci;
    std::array<std::vector<int>, 2> kind_cjpacked_per_sci;
    std::array<std::vector<int>, 2> kind_active_j_per_sci;
    std::array<std::vector<int>, 2> kind_active_records_per_sci;
    std::vector<GromacsSciEntryTrace> sci_entry_traces;
    cjpacked_per_sci.reserve(snapshot.sci.size());
    active_j_per_sci.reserve(snapshot.sci.size());
    active_records_per_sci.reserve(snapshot.sci.size());
    kind_cjpacked_per_sci[0].reserve(snapshot.sci.size());
    kind_cjpacked_per_sci[1].reserve(snapshot.sci.size());
    kind_active_j_per_sci[0].reserve(snapshot.sci.size());
    kind_active_j_per_sci[1].reserve(snapshot.sci.size());
    kind_active_records_per_sci[0].reserve(snapshot.sci.size());
    kind_active_records_per_sci[1].reserve(snapshot.sci.size());
    sci_entry_traces.reserve(snapshot.sci.size());
    const bool full_clusters =
        (snapshot.header.total_atom_numbers % kClusterSize) == 0;
    const int centralShiftId = FindZeroShiftId(snapshot.header.shiftvec);

    for (size_t sci_idx = 0; sci_idx < snapshot.sci.size(); ++sci_idx)
    {
        const GromacsSciPOD& sci = snapshot.sci[sci_idx];
        if (previousSciSuper >= 0)
        {
            const int superDelta = sci.sci - previousSciSuper;
            sci_order_transitions += 1;
            sci_order_abs_super_delta +=
                static_cast<uint64_t>(std::abs(superDelta));
            if (sci.shift == previousSciShift)
            {
                sci_order_same_shift += 1;
            }
            if (sci.sci == previousSciSuper)
            {
                sci_order_same_super += 1;
            }
            if (superDelta < 0)
            {
                sci_order_backward_super += 1;
            }
            if (std::abs(superDelta) > kSuperClusterClusters)
            {
                sci_order_large_super_jump += 1;
            }
        }
        previousSciSuper = sci.sci;
        previousSciShift = sci.shift;
        GromacsSciEntryTrace entryTrace = {};
        entryTrace.sciIdx = static_cast<int>(sci_idx);
        entryTrace.superI = sci.sci;
        entryTrace.shift = sci.shift;
        entryTrace.cjpackedSpan = sci.cjPackedEnd - sci.cjPackedBegin;
        const bool is_central = (sci.shift == centralShiftId);
        const int kind_idx = is_central ? 0 : 1;
        if (is_central)
        {
            central_sci += 1;
        }
        else
        {
            shifted_sci += 1;
        }
        if (sci.sci >= 0 && sci.sci < supercluster_count)
        {
            supercluster_sci_counts[static_cast<size_t>(sci.sci)] += 1;
            if (sci.shift >= 0 && sci.shift < kShiftCount)
            {
                supercluster_shift_masks[static_cast<size_t>(sci.sci)] |=
                    (1u << static_cast<unsigned int>(sci.shift));
            }
        }
        uint64_t sci_active_records = 0;
        uint64_t sci_potential_pairs = 0;
        int previousClusterJ = -1;
        for (int packed_idx = sci.cjPackedBegin; packed_idx < sci.cjPackedEnd;
             ++packed_idx)
        {
            const GromacsCjPackedPOD& packed = snapshot.cjpacked[packed_idx];
            int activeJInPacked = 0;
            for (int jm = 0; jm < kJGroupSize; ++jm)
            {
                const int cluster_j = packed.cj[jm];
                if (cluster_j < 0)
                {
                    continue;
                }
                activeJInPacked += 1;
                active_j_clusters += 1;
                total_active_j_slots += 1;
                entryTrace.activeJClusters += 1;
                if (entryTrace.firstClusterJ < 0)
                {
                    entryTrace.firstClusterJ = cluster_j;
                }
                entryTrace.lastClusterJ = cluster_j;
                if (previousClusterJ >= 0)
                {
                    const int delta = cluster_j - previousClusterJ;
                    entryTrace.totalTransitions += 1;
                    total_signed_cluster_j_delta += delta;
                    total_abs_cluster_j_delta +=
                        static_cast<uint64_t>(std::abs(delta));
                    entryTrace.absClusterJDelta +=
                        static_cast<uint64_t>(std::abs(delta));
                    entryTrace.maxAbsDelta =
                        std::max(entryTrace.maxAbsDelta, std::abs(delta));
                    if (delta == 1)
                    {
                        entryTrace.contiguousTransitions += 1;
                        total_contiguous_transitions += 1;
                    }
                    if (delta < 0)
                    {
                        entryTrace.backwardTransitions += 1;
                        total_backward_transitions += 1;
                    }
                    if ((previousClusterJ / kSuperClusterClusters) ==
                        (cluster_j / kSuperClusterClusters))
                    {
                        entryTrace.sameSuperTransitions += 1;
                        total_same_super_transitions += 1;
                    }
                    if (std::abs(delta) > kSuperClusterClusters)
                    {
                        entryTrace.largeJumpTransitions += 1;
                        total_large_jump_transitions += 1;
                    }
                }
                previousClusterJ = cluster_j;
                for (int split = 0; split < kWarpSplitCount; ++split)
                {
                    const unsigned int imask = packed.imei[split].imask;
                    const int exclIndex = packed.imei[split].excl_ind;
                    if (exclIndex != 0)
                    {
                        split_entries_with_excl += 1;
                        if (exclIndex > 0 &&
                            static_cast<size_t>(exclIndex) < used_excl.size())
                        {
                            used_excl[static_cast<size_t>(exclIndex)] = 1u;
                        }
                    }
                    active_split_j_clusters += 1;
                    const unsigned int jm_imask =
                        (imask >> (jm * kSuperClusterClusters)) &
                        ((1u << kSuperClusterClusters) - 1u);
                    if (is_central)
                    {
                        const int central_i =
                            cluster_j - sci.sci * kSuperClusterClusters;
                        if (central_i >= 0 &&
                            central_i < kSuperClusterClusters &&
                            ((jm_imask >> static_cast<unsigned int>(central_i)) &
                             1u) != 0u)
                        {
                            central_same_cluster_split_records += 1;
                        }
                    }
                    if (jm_imask != 0u)
                    {
                        active_records += 1;
                        sci_active_records += 1;
                        entryTrace.activeRecordsForSci += 1;
                        if (is_central)
                        {
                            central_only_records += 1;
                        }
                        else
                        {
                            shifted_only_records += 1;
                        }
                    }
                    const int active_i_clusters = HostPopcount(jm_imask);
                    const int valid_j_atoms = kSplitJClusterSize;
                    const int local_j_atoms = valid_j_atoms;
                    const int active_i_atoms =
                        full_clusters ? active_i_clusters * kClusterSize : 0;
                    total_valid_j_atoms += static_cast<uint64_t>(valid_j_atoms);
                    total_local_j_atoms += static_cast<uint64_t>(local_j_atoms);
                    total_active_i_clusters +=
                        static_cast<uint64_t>(active_i_clusters);
                    total_active_i_atoms += static_cast<uint64_t>(active_i_atoms);
                    total_potential_atom_pairs +=
                        static_cast<uint64_t>(active_i_atoms) *
                        static_cast<uint64_t>(valid_j_atoms);
                    sci_potential_pairs +=
                        static_cast<uint64_t>(active_i_atoms) *
                        static_cast<uint64_t>(valid_j_atoms);
                    if (is_central)
                    {
                        central_only_potential_atom_pairs +=
                            static_cast<uint64_t>(active_i_atoms) *
                            static_cast<uint64_t>(valid_j_atoms);
                    }
                    else
                    {
                        shifted_only_potential_atom_pairs +=
                            static_cast<uint64_t>(active_i_atoms) *
                            static_cast<uint64_t>(valid_j_atoms);
                    }
                }
            }
            active_j_per_cjpacked_hist[static_cast<size_t>(activeJInPacked)] += 1;
            if (activeJInPacked == kJGroupSize)
            {
                full_cjpacked_groups += 1;
            }
            else if (activeJInPacked > 0)
            {
                partial_cjpacked_groups += 1;
            }
        }
        cjpacked_per_sci.push_back(entryTrace.cjpackedSpan);
        active_j_per_sci.push_back(entryTrace.activeJClusters);
        active_records_per_sci.push_back(entryTrace.activeRecordsForSci);
        kind_sci_counts[static_cast<size_t>(kind_idx)] += 1;
        kind_cjpacked_spans[static_cast<size_t>(kind_idx)] +=
            static_cast<uint64_t>(entryTrace.cjpackedSpan);
        kind_active_j_clusters[static_cast<size_t>(kind_idx)] +=
            static_cast<uint64_t>(entryTrace.activeJClusters);
        kind_active_records[static_cast<size_t>(kind_idx)] +=
            static_cast<uint64_t>(entryTrace.activeRecordsForSci);
        kind_transitions[static_cast<size_t>(kind_idx)] +=
            static_cast<uint64_t>(entryTrace.totalTransitions);
        kind_contiguous_transitions[static_cast<size_t>(kind_idx)] +=
            static_cast<uint64_t>(entryTrace.contiguousTransitions);
        kind_backward_transitions[static_cast<size_t>(kind_idx)] +=
            static_cast<uint64_t>(entryTrace.backwardTransitions);
        kind_same_super_transitions[static_cast<size_t>(kind_idx)] +=
            static_cast<uint64_t>(entryTrace.sameSuperTransitions);
        kind_large_jump_transitions[static_cast<size_t>(kind_idx)] +=
            static_cast<uint64_t>(entryTrace.largeJumpTransitions);
        kind_abs_cluster_j_delta[static_cast<size_t>(kind_idx)] +=
            entryTrace.absClusterJDelta;
        kind_cjpacked_per_sci[static_cast<size_t>(kind_idx)].push_back(
            entryTrace.cjpackedSpan);
        kind_active_j_per_sci[static_cast<size_t>(kind_idx)].push_back(
            entryTrace.activeJClusters);
        kind_active_records_per_sci[static_cast<size_t>(kind_idx)].push_back(
            entryTrace.activeRecordsForSci);
        sci_entry_traces.push_back(entryTrace);
        if (is_central)
        {
            central_sci_records += sci_active_records;
            central_sci_pairs += sci_potential_pairs;
        }
        else
        {
            shifted_sci_records += sci_active_records;
            shifted_sci_pairs += sci_potential_pairs;
        }
    }

    std::printf(
        "analysis=gromacs clusters=%zu avg_local_atoms_per_cluster=%.6f "
        "central_shift_id=%d sci=%zu cjpacked=%zu active_j_clusters=%llu "
        "active_split_j_clusters=%llu active_records=%llu "
        "avg_records_per_sci=%.6f central_only_records=%llu shifted_only_records=%llu "
        "central_only_pair_ratio=%.6f shifted_only_pair_ratio=%.6f "
        "full_clusters=%d atoms=%llu "
        "avg_valid_j=%.6f avg_local_j=%.6f avg_active_i_clusters=%.6f "
        "avg_active_i_atoms=%.6f potential_atom_pairs=%llu "
        "avg_potential_pairs_per_record=%.6f\n",
        cluster_count,
        cluster_count > 0
            ? static_cast<double>(snapshot.header.local_atom_numbers) /
                  static_cast<double>(cluster_count)
            : 0.0,
        centralShiftId, snapshot.sci.size(), snapshot.cjpacked.size(),
        static_cast<unsigned long long>(active_j_clusters),
        static_cast<unsigned long long>(active_split_j_clusters),
        static_cast<unsigned long long>(active_records),
        snapshot.sci.size() > 0
            ? static_cast<double>(active_records) /
                  static_cast<double>(snapshot.sci.size())
            : 0.0,
        static_cast<unsigned long long>(central_only_records),
        static_cast<unsigned long long>(shifted_only_records),
        total_potential_atom_pairs > 0
            ? static_cast<double>(central_only_potential_atom_pairs) /
                  static_cast<double>(total_potential_atom_pairs)
            : 0.0,
        total_potential_atom_pairs > 0
            ? static_cast<double>(shifted_only_potential_atom_pairs) /
                  static_cast<double>(total_potential_atom_pairs)
            : 0.0,
        full_clusters ? 1 : 0,
        static_cast<unsigned long long>(snapshot.header.total_atom_numbers),
        active_j_clusters > 0
            ? static_cast<double>(total_valid_j_atoms) /
                  static_cast<double>(active_j_clusters)
            : 0.0,
        active_j_clusters > 0
            ? static_cast<double>(total_local_j_atoms) /
                  static_cast<double>(active_j_clusters)
            : 0.0,
        active_split_j_clusters > 0
            ? static_cast<double>(total_active_i_clusters) /
                  static_cast<double>(active_split_j_clusters)
            : 0.0,
        active_split_j_clusters > 0
            ? static_cast<double>(total_active_i_atoms) /
                  static_cast<double>(active_split_j_clusters)
            : 0.0,
        static_cast<unsigned long long>(total_potential_atom_pairs),
        active_records > 0
            ? static_cast<double>(total_potential_atom_pairs) /
                  static_cast<double>(active_records)
            : 0.0);
    PrintClusterGeometryStats("gromacs-cluster-geom", clusterGeometry);
    AnalyzeGromacsForceWriteLocality(snapshot);
    AnalyzeGromacsPerAtomOutputWriters(snapshot);
    const uint64_t uniqueUsedExcl = std::accumulate(
        used_excl.begin(), used_excl.end(), uint64_t{0});
    std::printf(
        "analysis=gromacs-exclusion excl_entries=%zu unique_used_excl=%llu "
        "split_entries_with_excl=%llu central_same_cluster_split_records=%llu "
        "central_same_cluster_ratio=%.6f\n",
        snapshot.excl.size(), static_cast<unsigned long long>(uniqueUsedExcl),
        static_cast<unsigned long long>(split_entries_with_excl),
        static_cast<unsigned long long>(central_same_cluster_split_records),
        active_records > 0
            ? static_cast<double>(central_same_cluster_split_records) /
                  static_cast<double>(active_records)
            : 0.0);

    uint64_t active_superclusters = 0;
    uint64_t total_active_shifts = 0;
    int max_active_shifts = 0;
    std::array<uint64_t, 6> active_shift_hist = {};
    for (int super_i = 0; super_i < supercluster_count; ++super_i)
    {
        const int sci_count_for_super =
            supercluster_sci_counts[static_cast<size_t>(super_i)];
        if (sci_count_for_super <= 0)
        {
            continue;
        }
        active_superclusters += 1;
        const int active_shifts = HostPopcount(
            supercluster_shift_masks[static_cast<size_t>(super_i)]);
        total_active_shifts += static_cast<uint64_t>(active_shifts);
        max_active_shifts = std::max(max_active_shifts, active_shifts);
        const int hist_bin =
            std::min(std::max(active_shifts, 0), static_cast<int>(active_shift_hist.size()) - 1);
        active_shift_hist[static_cast<size_t>(hist_bin)] += 1;
    }
    std::printf(
        "analysis=gromacs-sci-shift superclusters_with_sci=%llu "
        "central_sci=%llu shifted_sci=%llu central_sci_ratio=%.6f "
        "shifted_sci_ratio=%.6f avg_active_shifts_per_supercluster=%.6f "
        "max_active_shifts_per_supercluster=%d "
        "avg_sci_per_supercluster=%.6f avg_records_per_central_sci=%.6f "
        "avg_records_per_shifted_sci=%.6f avg_pairs_per_central_sci=%.6f "
        "avg_pairs_per_shifted_sci=%.6f shift_hist=[1:%llu 2:%llu 3:%llu 4:%llu 5+:%llu]\n",
        static_cast<unsigned long long>(active_superclusters),
        static_cast<unsigned long long>(central_sci),
        static_cast<unsigned long long>(shifted_sci),
        snapshot.sci.size() > 0
            ? static_cast<double>(central_sci) /
                  static_cast<double>(snapshot.sci.size())
            : 0.0,
        snapshot.sci.size() > 0
            ? static_cast<double>(shifted_sci) /
                  static_cast<double>(snapshot.sci.size())
            : 0.0,
        active_superclusters > 0
            ? static_cast<double>(total_active_shifts) /
                  static_cast<double>(active_superclusters)
            : 0.0,
        max_active_shifts,
        active_superclusters > 0
            ? static_cast<double>(snapshot.sci.size()) /
                  static_cast<double>(active_superclusters)
            : 0.0,
        central_sci > 0
            ? static_cast<double>(central_sci_records) /
                  static_cast<double>(central_sci)
            : 0.0,
        shifted_sci > 0
            ? static_cast<double>(shifted_sci_records) /
                  static_cast<double>(shifted_sci)
            : 0.0,
        central_sci > 0
            ? static_cast<double>(central_sci_pairs) /
                  static_cast<double>(central_sci)
            : 0.0,
        shifted_sci > 0
            ? static_cast<double>(shifted_sci_pairs) /
                  static_cast<double>(shifted_sci)
            : 0.0,
        static_cast<unsigned long long>(active_shift_hist[1]),
        static_cast<unsigned long long>(active_shift_hist[2]),
        static_cast<unsigned long long>(active_shift_hist[3]),
        static_cast<unsigned long long>(active_shift_hist[4]),
        static_cast<unsigned long long>(active_shift_hist[5]));

    std::sort(cjpacked_per_sci.begin(), cjpacked_per_sci.end());
    std::sort(active_j_per_sci.begin(), active_j_per_sci.end());
    std::sort(active_records_per_sci.begin(), active_records_per_sci.end());
    for (size_t kind = 0; kind < kind_cjpacked_per_sci.size(); ++kind)
    {
        std::sort(kind_cjpacked_per_sci[kind].begin(),
                  kind_cjpacked_per_sci[kind].end());
        std::sort(kind_active_j_per_sci[kind].begin(),
                  kind_active_j_per_sci[kind].end());
        std::sort(kind_active_records_per_sci[kind].begin(),
                  kind_active_records_per_sci[kind].end());
    }
    std::sort(
        sci_entry_traces.begin(), sci_entry_traces.end(),
        [](const GromacsSciEntryTrace& lhs, const GromacsSciEntryTrace& rhs)
        {
            if (lhs.activeRecordsForSci != rhs.activeRecordsForSci)
            {
                return lhs.activeRecordsForSci > rhs.activeRecordsForSci;
            }
            return lhs.cjpackedSpan > rhs.cjpackedSpan;
        });
    const double total_transitions =
        static_cast<double>(std::max<uint64_t>(
            1ull,
            std::accumulate(
                sci_entry_traces.begin(), sci_entry_traces.end(), 0ull,
                [](uint64_t acc, const GromacsSciEntryTrace& entry)
                { return acc + static_cast<uint64_t>(entry.totalTransitions); })));
    std::printf(
        "analysis=gromacs-entry avg_cjpacked_per_sci=%.6f "
        "p50_cjpacked_per_sci=%.2f p90_cjpacked_per_sci=%.2f "
        "p99_cjpacked_per_sci=%.2f avg_active_j_per_sci=%.6f "
        "p50_active_j_per_sci=%.2f p90_active_j_per_sci=%.2f "
        "avg_active_records_per_sci=%.6f p90_active_records_per_sci=%.2f "
        "avg_active_j_per_cjpacked=%.6f full_cjpacked_ratio=%.6f "
        "partial_cjpacked_ratio=%.6f contiguous_j_ratio=%.6f "
        "same_j_super_ratio=%.6f backward_jump_ratio=%.6f "
        "large_jump_ratio=%.6f avg_abs_cluster_j_delta=%.6f "
        "avg_signed_cluster_j_delta=%.6f active_j_hist=[1:%llu 2:%llu 3:%llu 4:%llu]\n",
        snapshot.sci.size() > 0
            ? static_cast<double>(snapshot.cjpacked.size()) /
                  static_cast<double>(snapshot.sci.size())
            : 0.0,
        PercentileFromSorted(cjpacked_per_sci, 0.50),
        PercentileFromSorted(cjpacked_per_sci, 0.90),
        PercentileFromSorted(cjpacked_per_sci, 0.99),
        snapshot.sci.size() > 0
            ? static_cast<double>(active_j_clusters) /
                  static_cast<double>(snapshot.sci.size())
            : 0.0,
        PercentileFromSorted(active_j_per_sci, 0.50),
        PercentileFromSorted(active_j_per_sci, 0.90),
        snapshot.sci.size() > 0
            ? static_cast<double>(active_records) /
                  static_cast<double>(snapshot.sci.size())
            : 0.0,
        PercentileFromSorted(active_records_per_sci, 0.90),
        snapshot.cjpacked.size() > 0
            ? static_cast<double>(total_active_j_slots) /
                  static_cast<double>(snapshot.cjpacked.size())
            : 0.0,
        snapshot.cjpacked.size() > 0
            ? static_cast<double>(full_cjpacked_groups) /
                  static_cast<double>(snapshot.cjpacked.size())
            : 0.0,
        snapshot.cjpacked.size() > 0
            ? static_cast<double>(partial_cjpacked_groups) /
                  static_cast<double>(snapshot.cjpacked.size())
            : 0.0,
        static_cast<double>(total_contiguous_transitions) / total_transitions,
        static_cast<double>(total_same_super_transitions) / total_transitions,
        static_cast<double>(total_backward_transitions) / total_transitions,
        static_cast<double>(total_large_jump_transitions) / total_transitions,
        static_cast<double>(total_abs_cluster_j_delta) / total_transitions,
        static_cast<double>(total_signed_cluster_j_delta) / total_transitions,
        static_cast<unsigned long long>(active_j_per_cjpacked_hist[1]),
        static_cast<unsigned long long>(active_j_per_cjpacked_hist[2]),
        static_cast<unsigned long long>(active_j_per_cjpacked_hist[3]),
        static_cast<unsigned long long>(active_j_per_cjpacked_hist[4]));
    for (size_t kind = 0; kind < 2; ++kind)
    {
        const uint64_t sci_count = kind_sci_counts[kind];
        const double kind_trans =
            static_cast<double>(std::max<uint64_t>(kind_transitions[kind], 1));
        std::printf(
            "analysis=gromacs-entry-kind kind=%s sci=%llu "
            "active_records=%llu avg_active_records_per_sci=%.6f "
            "p90_active_records_per_sci=%.2f avg_cjpacked_per_sci=%.6f "
            "p90_cjpacked_per_sci=%.2f avg_active_j_per_sci=%.6f "
            "p90_active_j_per_sci=%.2f contiguous_j_ratio=%.6f "
            "backward_jump_ratio=%.6f same_j_super_ratio=%.6f "
            "large_jump_ratio=%.6f avg_abs_cluster_j_delta=%.6f\n",
            kind == 0 ? "central" : "shifted",
            static_cast<unsigned long long>(sci_count),
            static_cast<unsigned long long>(kind_active_records[kind]),
            sci_count > 0
                ? static_cast<double>(kind_active_records[kind]) /
                      static_cast<double>(sci_count)
                : 0.0,
            PercentileFromSorted(kind_active_records_per_sci[kind], 0.90),
            sci_count > 0
                ? static_cast<double>(kind_cjpacked_spans[kind]) /
                      static_cast<double>(sci_count)
                : 0.0,
            PercentileFromSorted(kind_cjpacked_per_sci[kind], 0.90),
            sci_count > 0
                ? static_cast<double>(kind_active_j_clusters[kind]) /
                      static_cast<double>(sci_count)
                : 0.0,
            PercentileFromSorted(kind_active_j_per_sci[kind], 0.90),
            static_cast<double>(kind_contiguous_transitions[kind]) /
                kind_trans,
            static_cast<double>(kind_backward_transitions[kind]) / kind_trans,
            static_cast<double>(kind_same_super_transitions[kind]) / kind_trans,
            static_cast<double>(kind_large_jump_transitions[kind]) / kind_trans,
            static_cast<double>(kind_abs_cluster_j_delta[kind]) / kind_trans);
    }
    std::printf(
        "analysis=gromacs-sci-order transitions=%llu same_shift_ratio=%.6f "
        "same_super_ratio=%.6f backward_super_ratio=%.6f "
        "large_super_jump_ratio=%.6f avg_abs_super_delta=%.6f\n",
        static_cast<unsigned long long>(sci_order_transitions),
        sci_order_transitions > 0
            ? static_cast<double>(sci_order_same_shift) /
                  static_cast<double>(sci_order_transitions)
            : 0.0,
        sci_order_transitions > 0
            ? static_cast<double>(sci_order_same_super) /
                  static_cast<double>(sci_order_transitions)
            : 0.0,
        sci_order_transitions > 0
            ? static_cast<double>(sci_order_backward_super) /
                  static_cast<double>(sci_order_transitions)
            : 0.0,
        sci_order_transitions > 0
            ? static_cast<double>(sci_order_large_super_jump) /
                  static_cast<double>(sci_order_transitions)
            : 0.0,
        sci_order_transitions > 0
            ? static_cast<double>(sci_order_abs_super_delta) /
                  static_cast<double>(sci_order_transitions)
            : 0.0);
    for (size_t rank = 0;
         rank < std::min<size_t>(4, sci_entry_traces.size()); ++rank)
    {
        const GromacsSciEntryTrace& entry = sci_entry_traces[rank];
        const double local_transition_count =
            static_cast<double>(std::max(entry.totalTransitions, 1));
        std::printf(
            "analysis=gromacs-entry-top rank=%zu sci_idx=%d super=%d shift=%d "
            "cjpacked=%d active_j=%d active_records=%d first_j=%d last_j=%d "
            "contiguous_ratio=%.6f backward_ratio=%.6f same_super_ratio=%.6f "
            "large_jump_ratio=%.6f max_abs_delta=%d\n",
            rank + 1, entry.sciIdx, entry.superI, entry.shift,
            entry.cjpackedSpan, entry.activeJClusters,
            entry.activeRecordsForSci, entry.firstClusterJ, entry.lastClusterJ,
            static_cast<double>(entry.contiguousTransitions) /
                local_transition_count,
            static_cast<double>(entry.backwardTransitions) /
                local_transition_count,
            static_cast<double>(entry.sameSuperTransitions) /
                local_transition_count,
            static_cast<double>(entry.largeJumpTransitions) /
                local_transition_count,
            entry.maxAbsDelta);
    }
    for (size_t rank = 0;
         rank < std::min<size_t>(2, sci_entry_traces.size()); ++rank)
    {
        const GromacsSciEntryTrace& entry = sci_entry_traces[rank];
        std::array<int, 16> previewClusters = {};
        std::array<int, 15> previewDeltas = {};
        int previewCount = 0;
        const GromacsSciPOD& sci = snapshot.sci[static_cast<size_t>(entry.sciIdx)];
        for (int packed_idx = sci.cjPackedBegin; packed_idx < sci.cjPackedEnd;
             ++packed_idx)
        {
            const GromacsCjPackedPOD& packed = snapshot.cjpacked[packed_idx];
            for (int jm = 0; jm < kJGroupSize; ++jm)
            {
                const int cluster_j = packed.cj[jm];
                if (cluster_j < 0)
                {
                    continue;
                }
                if (previewCount < static_cast<int>(previewClusters.size()))
                {
                    previewClusters[static_cast<size_t>(previewCount)] = cluster_j;
                    if (previewCount > 0)
                    {
                        previewDeltas[static_cast<size_t>(previewCount - 1)] =
                            cluster_j - previewClusters[static_cast<size_t>(
                                            previewCount - 1)];
                    }
                }
                previewCount += 1;
            }
            if (previewCount >= static_cast<int>(previewClusters.size()))
            {
                break;
            }
        }
        std::printf(
            "analysis=gromacs-entry-seq rank=%zu sci_idx=%d preview_clusters=[",
            rank + 1, entry.sciIdx);
        const int clustersToPrint =
            std::min(previewCount, static_cast<int>(previewClusters.size()));
        for (int i = 0; i < clustersToPrint; ++i)
        {
            if (i > 0)
            {
                std::printf(" ");
            }
            std::printf("%d", previewClusters[static_cast<size_t>(i)]);
        }
        std::printf("] preview_deltas=[");
        for (int i = 0; i < std::max(0, clustersToPrint - 1); ++i)
        {
            if (i > 0)
            {
                std::printf(" ");
            }
            std::printf("%d", previewDeltas[static_cast<size_t>(i)]);
        }
        std::printf("]\n");
    }
}

template <typename T>
T* CopyVectorToDevice(const std::vector<T>& values)
{
    T* ptr = nullptr;
    if (!values.empty())
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&ptr),
                             sizeof(T) * values.size()),
                  "cudaMalloc");
        CheckCuda(cudaMemcpy(ptr, values.data(), sizeof(T) * values.size(),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy");
    }
    return ptr;
}

template <typename T>
std::vector<T> CopyVectorFromDevice(const T* ptr, size_t count)
{
    std::vector<T> values(count);
    if (count != 0)
    {
        CheckCuda(cudaMemcpy(values.data(), ptr, sizeof(T) * count,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(device vector to host)");
    }
    return values;
}

void PrintRefreshVerification(const SpongeGmxpackedForceOnlySnapshot& snapshot,
                              const uint64_t* d_pair_shift_bits,
                              const int* d_sci_shift_safe_flags,
                              const char* label)
{
    const std::vector<uint64_t> pair_shift_bits =
        CopyVectorFromDevice(d_pair_shift_bits, snapshot.pair_shift_bits.size());
    size_t pair_mismatches = 0;
    size_t first_pair_mismatch = static_cast<size_t>(-1);
    for (size_t i = 0; i < snapshot.pair_shift_bits.size(); ++i)
    {
        if (pair_shift_bits[i] != snapshot.pair_shift_bits[i])
        {
            if (first_pair_mismatch == static_cast<size_t>(-1))
            {
                first_pair_mismatch = i;
            }
            pair_mismatches += 1;
        }
    }

    size_t flag_mismatches = 0;
    size_t first_flag_mismatch = static_cast<size_t>(-1);
    if (d_sci_shift_safe_flags != nullptr &&
        snapshot.sci_shift_safe_flags.size() == snapshot.sci.size())
    {
        const std::vector<int> safe_flags = CopyVectorFromDevice(
            d_sci_shift_safe_flags, snapshot.sci_shift_safe_flags.size());
        for (size_t i = 0; i < snapshot.sci_shift_safe_flags.size(); ++i)
        {
            if (safe_flags[i] != snapshot.sci_shift_safe_flags[i])
            {
                if (first_flag_mismatch == static_cast<size_t>(-1))
                {
                    first_flag_mismatch = i;
                }
                flag_mismatches += 1;
            }
        }
    }

    std::printf(
        "verify=%s pair_shift_mismatches=%zu first_pair_mismatch=%zd",
        label, pair_mismatches,
        first_pair_mismatch == static_cast<size_t>(-1)
            ? -1
            : static_cast<ptrdiff_t>(first_pair_mismatch));
    if (first_pair_mismatch != static_cast<size_t>(-1))
    {
        std::printf(" pair_ref=0x%016llx pair_got=0x%016llx",
                    static_cast<unsigned long long>(
                        snapshot.pair_shift_bits[first_pair_mismatch]),
                    static_cast<unsigned long long>(
                        pair_shift_bits[first_pair_mismatch]));
    }
    if (snapshot.sci_shift_safe_flags.size() == snapshot.sci.size())
    {
        std::printf(" sci_safe_flag_mismatches=%zu first_flag_mismatch=%zd",
                    flag_mismatches,
                    first_flag_mismatch == static_cast<size_t>(-1)
                        ? -1
                        : static_cast<ptrdiff_t>(first_flag_mismatch));
    }
    std::printf("\n");
}

int GetOptionalEnvInt(const char* name, int defaultValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return defaultValue;
    }
    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : defaultValue;
}

struct GmxPackedPartialOutputIndex
{
    int partialCount = 0;
    std::vector<int> sciPartialBases;
    std::vector<unsigned char> atomPartialFlags;
    std::vector<int> atomPartialOffsets;
    std::vector<int> atomPartialSlots;
    std::vector<int> reduceAtoms2;
    std::vector<int> reduceSlots2;
    std::vector<int> reduceAtoms4;
    std::vector<int> reduceSlots4;
    std::vector<int> reduceAtoms6;
    std::vector<int> reduceSlots6;
    std::vector<int> reduceAtoms8;
    std::vector<int> reduceSlots8;
    std::vector<int> reduceGenericAtoms;
};

GmxPackedPartialOutputIndex BuildGmxPackedPartialOutputIndex(
    const GromacsPairlistSnapshot& snapshot, bool shiftedOnly = false,
    int centralShiftId = kCentralShiftId,
    int writerCountThreshold = -1)
{
    GmxPackedPartialOutputIndex index;
    const auto atomCount =
        static_cast<size_t>(snapshot.header.total_atom_numbers);
    index.sciPartialBases.resize(snapshot.sci.size(), -1);
    index.atomPartialFlags.resize(atomCount, 1u);
    std::vector<std::vector<int>> slotsByAtom(atomCount);
    for (size_t sciIdx = 0; sciIdx < snapshot.sci.size(); ++sciIdx)
    {
        const GromacsSciPOD& sci = snapshot.sci[sciIdx];
        if (shiftedOnly && sci.shift == centralShiftId)
        {
            continue;
        }
        index.sciPartialBases[sciIdx] = index.partialCount;
        index.partialCount +=
            kWarpSplitCount * kSuperClusterClusters * kClusterSize;
        const int sortedSuperIBase =
            sci.sci * kSuperClusterClusters * kClusterSize;
        const int partialBase = index.sciPartialBases[sciIdx];
        for (int localCluster = 0; localCluster < kSuperClusterClusters;
             ++localCluster)
        {
            for (int lane = 0; lane < kClusterSize; ++lane)
            {
                const int localSlot = localCluster * kClusterSize + lane;
                const int sortedI = sortedSuperIBase + localSlot;
                if (sortedI < 0 ||
                    static_cast<size_t>(sortedI) >= snapshot.sorted_atom_ids.size())
                {
                    continue;
                }
                const int atom = snapshot.sorted_atom_ids[static_cast<size_t>(sortedI)];
                if (atom < 0 || static_cast<size_t>(atom) >= atomCount)
                {
                    continue;
                }
                for (int split = 0; split < kWarpSplitCount; ++split)
                {
                    slotsByAtom[static_cast<size_t>(atom)].push_back(
                        partialBase +
                        split * kSuperClusterClusters * kClusterSize +
                        localSlot);
                }
            }
        }
    }
    if (writerCountThreshold >= 0)
    {
        for (size_t atom = 0; atom < atomCount; ++atom)
        {
            const int writerCount =
                static_cast<int>(slotsByAtom[atom].size()) / kWarpSplitCount;
            const bool usePartial = writerCount > writerCountThreshold;
            index.atomPartialFlags[atom] = usePartial ? 1u : 0u;
            if (!usePartial)
            {
                slotsByAtom[atom].clear();
            }
        }
    }
    index.atomPartialOffsets.resize(atomCount + 1, 0);
    for (size_t atom = 0; atom < atomCount; ++atom)
    {
        index.atomPartialOffsets[atom + 1] =
            index.atomPartialOffsets[atom] +
            static_cast<int>(slotsByAtom[atom].size());
    }
    index.atomPartialSlots.reserve(
        static_cast<size_t>(index.atomPartialOffsets.back()));
    for (const std::vector<int>& slots : slotsByAtom)
    {
        index.atomPartialSlots.insert(index.atomPartialSlots.end(), slots.begin(),
                                      slots.end());
    }
    auto appendFixedBucket = [&](std::vector<int>& atoms,
                                 std::vector<int>& slots,
                                 size_t atom,
                                 int begin,
                                 int entryCount)
    {
        atoms.push_back(static_cast<int>(atom));
        slots.insert(slots.end(),
                     index.atomPartialSlots.begin() + begin,
                     index.atomPartialSlots.begin() + begin + entryCount);
    };
    for (size_t atom = 0; atom < atomCount; ++atom)
    {
        const int begin = index.atomPartialOffsets[atom];
        const int end = index.atomPartialOffsets[atom + 1];
        const int entryCount = end - begin;
        switch (entryCount)
        {
        case 0:
            break;
        case 2:
            appendFixedBucket(index.reduceAtoms2, index.reduceSlots2, atom,
                              begin, entryCount);
            break;
        case 4:
            appendFixedBucket(index.reduceAtoms4, index.reduceSlots4, atom,
                              begin, entryCount);
            break;
        case 6:
            appendFixedBucket(index.reduceAtoms6, index.reduceSlots6, atom,
                              begin, entryCount);
            break;
        case 8:
            appendFixedBucket(index.reduceAtoms8, index.reduceSlots8, atom,
                              begin, entryCount);
            break;
        default:
            index.reduceGenericAtoms.push_back(static_cast<int>(atom));
            break;
        }
    }
    return index;
}

template <bool need_energy, bool need_virial, bool accumulate_output>
void LaunchGmxPackedPartialOutputReduceBuckets(
    const GmxPackedPartialOutputIndex& index, int reduceThreads,
    const int* d_reduce_atoms_2, const int* d_reduce_slots_2,
    const int* d_reduce_atoms_4, const int* d_reduce_slots_4,
    const int* d_reduce_atoms_6, const int* d_reduce_slots_6,
    const int* d_reduce_atoms_8, const int* d_reduce_slots_8,
    const int* d_reduce_generic_atoms, const int* d_atom_partial_offsets,
    const int* d_atom_partial_slots, const LTMatrix3* d_partial_virial,
    const float* d_partial_energy, const float* d_partial_direct_cf_energy,
    const float* d_partial_lj_energy, LTMatrix3* d_atom_virial,
    float* d_atom_energy, float* d_atom_direct_cf_energy, float* d_atom_lj_energy)
{
    const dim3 reduceBlock(reduceThreads, 1, 1);
    auto launchGrid = [reduceThreads](size_t count)
    {
        return dim3(static_cast<unsigned int>(
                        (count + static_cast<size_t>(reduceThreads) - 1) /
                        static_cast<size_t>(reduceThreads)),
                    1, 1);
    };
    if (!index.reduceAtoms2.empty())
    {
        ReduceGmxPackedPartialOutputsFixedKernel<need_energy, need_virial,
                                                 accumulate_output, 2>
            <<<launchGrid(index.reduceAtoms2.size()), reduceBlock>>>(
                static_cast<int>(index.reduceAtoms2.size()), d_reduce_atoms_2,
                d_reduce_slots_2, d_partial_virial, d_partial_energy,
                d_partial_direct_cf_energy, d_partial_lj_energy, d_atom_virial,
                d_atom_energy, d_atom_direct_cf_energy, d_atom_lj_energy);
    }
    if (!index.reduceAtoms4.empty())
    {
        ReduceGmxPackedPartialOutputsFixedKernel<need_energy, need_virial,
                                                 accumulate_output, 4>
            <<<launchGrid(index.reduceAtoms4.size()), reduceBlock>>>(
                static_cast<int>(index.reduceAtoms4.size()), d_reduce_atoms_4,
                d_reduce_slots_4, d_partial_virial, d_partial_energy,
                d_partial_direct_cf_energy, d_partial_lj_energy, d_atom_virial,
                d_atom_energy, d_atom_direct_cf_energy, d_atom_lj_energy);
    }
    if (!index.reduceAtoms6.empty())
    {
        ReduceGmxPackedPartialOutputsFixedKernel<need_energy, need_virial,
                                                 accumulate_output, 6>
            <<<launchGrid(index.reduceAtoms6.size()), reduceBlock>>>(
                static_cast<int>(index.reduceAtoms6.size()), d_reduce_atoms_6,
                d_reduce_slots_6, d_partial_virial, d_partial_energy,
                d_partial_direct_cf_energy, d_partial_lj_energy, d_atom_virial,
                d_atom_energy, d_atom_direct_cf_energy, d_atom_lj_energy);
    }
    if (!index.reduceAtoms8.empty())
    {
        ReduceGmxPackedPartialOutputsFixedKernel<need_energy, need_virial,
                                                 accumulate_output, 8>
            <<<launchGrid(index.reduceAtoms8.size()), reduceBlock>>>(
                static_cast<int>(index.reduceAtoms8.size()), d_reduce_atoms_8,
                d_reduce_slots_8, d_partial_virial, d_partial_energy,
                d_partial_direct_cf_energy, d_partial_lj_energy, d_atom_virial,
                d_atom_energy, d_atom_direct_cf_energy, d_atom_lj_energy);
    }
    if (!index.reduceGenericAtoms.empty())
    {
        ReduceGmxPackedPartialOutputsGenericAtomsKernel<need_energy, need_virial,
                                                        accumulate_output>
            <<<launchGrid(index.reduceGenericAtoms.size()), reduceBlock>>>(
                static_cast<int>(index.reduceGenericAtoms.size()),
                d_reduce_generic_atoms, d_atom_partial_offsets,
                d_atom_partial_slots, d_partial_virial, d_partial_energy,
                d_partial_direct_cf_energy, d_partial_lj_energy, d_atom_virial,
                d_atom_energy, d_atom_direct_cf_energy, d_atom_lj_energy);
    }
}

void AnalyzeSpongeProductionGmxpackedSnapshot(
    const SpongeGmxpackedForceOnlySnapshot& snapshot)
{
    const size_t safe_sci = static_cast<size_t>(
        std::count_if(snapshot.sci_shift_safe_flags.begin(),
                      snapshot.sci_shift_safe_flags.end(),
                      [](int value) { return value != 0; }));
    std::printf(
        "analysis=sponge-production-gmxpacked clusters=%llu superclusters=%llu "
        "sci=%zu cjpacked=%zu excl=%zu pair_shift_words=%zu atoms=%llu "
        "safe_sci=%zu unsafe_sci=%zu cutoff=%.6f pme_beta=%.6f "
        "builder_metadata=%u leaves=%zu octree_nodes=%zu candidate_sci=%zu "
        "candidate_leaves=%zu\n",
        static_cast<unsigned long long>(snapshot.header.cluster_numbers),
        static_cast<unsigned long long>(snapshot.header.super_cluster_numbers),
        snapshot.sci.size(), snapshot.cjpacked.size(), snapshot.excl.size(),
        snapshot.pair_shift_bits.size(),
        static_cast<unsigned long long>(snapshot.header.total_atom_numbers),
        safe_sci, snapshot.sci.size() - safe_sci, snapshot.header.cutoff,
        snapshot.header.pme_beta,
        snapshot.candidate_leaf_offsets.empty() ? 0u : 1u,
        snapshot.leaf_cluster_starts.size(), snapshot.octree_prefixes.size(),
        snapshot.candidate_leaf_offsets.empty()
            ? 0u
            : snapshot.candidate_leaf_offsets.size() - 1,
        snapshot.candidate_leaf_ids.size());

    std::array<unsigned long long, 33> imaskPopcount = {};
    std::array<unsigned long long, 9> jmImaskPopcount = {};
    std::array<unsigned long long, 9> jmEffectivePopcount = {};
    std::array<unsigned long long, 9> threadEffectivePopcount = {};
    unsigned long long splitEntries = 0;
    unsigned long long nonzeroImask = 0;
    unsigned long long excludedSplits = 0;
    unsigned long long jmSlots = 0;
    unsigned long long validJmSlots = 0;
    unsigned long long invalidJmSlots = 0;
    unsigned long long nonzeroJmImask = 0;
    unsigned long long nonzeroJmEffective = 0;
    unsigned long long jmKilledByExclusion = 0;
    unsigned long long jmDenseAllThreads = 0;
    unsigned long long jmUniformSparse = 0;
    unsigned long long jmMixed = 0;
    unsigned long long jmEmpty = 0;
    unsigned long long jmNoExclusion = 0;
    unsigned long long jmDenseNoExclusion = 0;
    unsigned long long jmUniformSparseNoExclusion = 0;
    unsigned long long splitAllEmpty = 0;
    unsigned long long splitAllDense = 0;
    unsigned long long splitAllGeneric = 0;
    unsigned long long splitDenseOnlyWithEmpty = 0;
    unsigned long long splitGenericOnlyWithEmpty = 0;
    unsigned long long splitMixedDenseGeneric = 0;
    unsigned long long sciEmpty = 0;
    unsigned long long sciDenseOnly = 0;
    unsigned long long sciGenericOnly = 0;
    unsigned long long sciMixedDenseGeneric = 0;
    unsigned long long sciDenseFracGe25 = 0;
    unsigned long long sciDenseFracGe50 = 0;
    unsigned long long sciDenseFracGe75 = 0;
    unsigned long long denseRuns = 0;
    unsigned long long genericRuns = 0;
    unsigned long long classTransitions = 0;
    int previousNonemptyClass = 0;

    for (const SpongeGmxpackedSciPOD& sci : snapshot.sci)
    {
        unsigned long long sciDense = 0;
        unsigned long long sciGeneric = 0;
        for (int packedIdx = sci.cjpacked_begin; packedIdx < sci.cjpacked_end;
             ++packedIdx)
        {
            const SpongeGmxpackedCjPOD& packed =
                snapshot.cjpacked[static_cast<size_t>(packedIdx)];
            for (int split = 0; split < kWarpSplitCount; ++split)
            {
                unsigned long long splitDense = 0;
                unsigned long long splitGeneric = 0;
                unsigned long long splitEmpty = 0;
                const SpongeGmxpackedSplitPOD& splitEntry = packed.split[split];
                const unsigned int imask = splitEntry.imask;
                ++splitEntries;
                imaskPopcount[std::popcount(imask)] += 1ull;
                if (imask != 0u)
                {
                    ++nonzeroImask;
                }
                if (splitEntry.exclusion_index != 0)
                {
                    ++excludedSplits;
                }

                for (int jm = 0; jm < kJGroupSize; ++jm)
                {
                    const unsigned int imask8 =
                        (imask >> (jm * kSuperClusterClusters)) & 0xffu;
                    jmImaskPopcount[std::popcount(imask8)] += 1ull;
                    ++jmSlots;
                    unsigned int effectiveMaskOr = 0u;
                    bool allThreadsDense = true;
                    bool uniformThreadMask = true;
                    bool haveFirstThreadMask = false;
                    unsigned int firstThreadMask = 0u;
                    const int clusterJ = packed.cj[jm];
                    const bool validClusterJ =
                        clusterJ >= 0 &&
                        clusterJ < static_cast<int>(snapshot.header.cluster_numbers);
                    if (validClusterJ)
                    {
                        ++validJmSlots;
                        for (int splitJLane = 0;
                             splitJLane < kSplitJClusterSize; ++splitJLane)
                        {
                            for (int iLane = 0; iLane < kClusterSize; ++iLane)
                            {
                                unsigned int pairBits = 0xffffffffu;
                                if (splitEntry.exclusion_index != 0)
                                {
                                    pairBits =
                                        snapshot.excl[static_cast<size_t>(
                                                          splitEntry
                                                              .exclusion_index)]
                                            .pair[(splitJLane * kClusterSize) +
                                                  iLane];
                                }
                                const unsigned int effectiveMask =
                                    imask & pairBits;
                                const unsigned int threadMask =
                                    (effectiveMask >>
                                     (jm * kSuperClusterClusters)) &
                                    0xffu;
                                effectiveMaskOr |= threadMask;
                                allThreadsDense =
                                    allThreadsDense && threadMask == 0xffu;
                                if (!haveFirstThreadMask)
                                {
                                    firstThreadMask = threadMask;
                                    haveFirstThreadMask = true;
                                }
                                else if (threadMask != firstThreadMask)
                                {
                                    uniformThreadMask = false;
                                }
                                threadEffectivePopcount[std::popcount(
                                    threadMask)] += 1ull;
                            }
                        }
                    }
                    else
                    {
                        ++invalidJmSlots;
                        allThreadsDense = false;
                    }
                    const int effectivePopcount =
                        std::popcount(effectiveMaskOr);
                    if (imask8 != 0u)
                    {
                        ++nonzeroJmImask;
                    }
                    if (effectiveMaskOr != 0u)
                    {
                        ++nonzeroJmEffective;
                    }
                    if (imask8 != 0u && effectiveMaskOr == 0u)
                    {
                        ++jmKilledByExclusion;
                    }
                    jmEffectivePopcount[static_cast<size_t>(
                        effectivePopcount)] += 1ull;
                    const bool nonempty = effectiveMaskOr != 0u;
                    const bool dense = validClusterJ && allThreadsDense;
                    const bool uniformSparse =
                        validClusterJ && nonempty && !dense && uniformThreadMask;
                    const bool noExclusion = splitEntry.exclusion_index == 0;
                    if (!nonempty)
                    {
                        ++jmEmpty;
                        ++splitEmpty;
                    }
                    else if (dense)
                    {
                        ++jmDenseAllThreads;
                        ++splitDense;
                        ++sciDense;
                        if (noExclusion)
                        {
                            ++jmDenseNoExclusion;
                        }
                    }
                    else
                    {
                        ++splitGeneric;
                        ++sciGeneric;
                        if (uniformSparse)
                        {
                            ++jmUniformSparse;
                            if (noExclusion)
                            {
                                ++jmUniformSparseNoExclusion;
                            }
                        }
                        else
                        {
                            ++jmMixed;
                        }
                    }
                    if (validClusterJ && noExclusion)
                    {
                        ++jmNoExclusion;
                    }
                    const int nonemptyClass = dense ? 1 : (nonempty ? 2 : 0);
                    if (nonemptyClass != 0)
                    {
                        if (nonemptyClass == 1)
                        {
                            ++denseRuns;
                        }
                        else
                        {
                            ++genericRuns;
                        }
                        if (previousNonemptyClass != 0 &&
                            previousNonemptyClass != nonemptyClass)
                        {
                            ++classTransitions;
                        }
                        previousNonemptyClass = nonemptyClass;
                    }
                }
                if (splitDense == 0 && splitGeneric == 0)
                {
                    ++splitAllEmpty;
                }
                else if (splitDense != 0 && splitGeneric == 0 && splitEmpty == 0)
                {
                    ++splitAllDense;
                }
                else if (splitDense == 0 && splitGeneric != 0 && splitEmpty == 0)
                {
                    ++splitAllGeneric;
                }
                else if (splitDense != 0 && splitGeneric == 0)
                {
                    ++splitDenseOnlyWithEmpty;
                }
                else if (splitDense == 0 && splitGeneric != 0)
                {
                    ++splitGenericOnlyWithEmpty;
                }
                else
                {
                    ++splitMixedDenseGeneric;
                }
            }
        }
        const unsigned long long sciNonempty = sciDense + sciGeneric;
        if (sciNonempty == 0)
        {
            ++sciEmpty;
        }
        else if (sciDense != 0 && sciGeneric == 0)
        {
            ++sciDenseOnly;
        }
        else if (sciDense == 0 && sciGeneric != 0)
        {
            ++sciGenericOnly;
        }
        else
        {
            ++sciMixedDenseGeneric;
        }
        if (sciNonempty != 0)
        {
            const double denseFraction =
                static_cast<double>(sciDense) /
                static_cast<double>(sciNonempty);
            if (denseFraction >= 0.25)
            {
                ++sciDenseFracGe25;
            }
            if (denseFraction >= 0.50)
            {
                ++sciDenseFracGe50;
            }
            if (denseFraction >= 0.75)
            {
                ++sciDenseFracGe75;
            }
        }
    }

    std::printf(
        "analysis=sponge-production-gmxpacked-mask split_entries=%llu "
        "nonzero_imask=%llu excluded_splits=%llu jm_slots=%llu "
        "nonzero_jm_imask=%llu nonzero_jm_effective=%llu "
        "jm_killed_by_exclusion=%llu\n",
        splitEntries, nonzeroImask, excludedSplits, jmSlots, nonzeroJmImask,
        nonzeroJmEffective, jmKilledByExclusion);
    std::printf("analysis=sponge-production-gmxpacked-imask-popcount");
    for (size_t i = 0; i < imaskPopcount.size(); ++i)
    {
        if (imaskPopcount[i] != 0ull)
        {
            std::printf(" pc%zu=%llu", i, imaskPopcount[i]);
        }
    }
    std::printf("\n");
    std::printf("analysis=sponge-production-gmxpacked-jm-imask-popcount");
    for (size_t i = 0; i < jmImaskPopcount.size(); ++i)
    {
        if (jmImaskPopcount[i] != 0ull)
        {
            std::printf(" pc%zu=%llu", i, jmImaskPopcount[i]);
        }
    }
    std::printf("\n");
    std::printf("analysis=sponge-production-gmxpacked-effective-jm-popcount");
    for (size_t i = 0; i < jmEffectivePopcount.size(); ++i)
    {
        if (jmEffectivePopcount[i] != 0ull)
        {
            std::printf(" pc%zu=%llu", i, jmEffectivePopcount[i]);
        }
    }
    std::printf("\n");
    std::printf("analysis=sponge-production-gmxpacked-thread-mask-popcount");
    for (size_t i = 0; i < threadEffectivePopcount.size(); ++i)
    {
        if (threadEffectivePopcount[i] != 0ull)
        {
            std::printf(" pc%zu=%llu", i, threadEffectivePopcount[i]);
        }
    }
    std::printf("\n");
    std::printf(
        "analysis=sponge-production-gmxpacked-jm-class valid=%llu invalid=%llu "
        "empty=%llu dense_all_threads=%llu uniform_sparse=%llu mixed=%llu "
        "no_exclusion=%llu dense_no_exclusion=%llu "
        "uniform_sparse_no_exclusion=%llu dense_runs=%llu generic_runs=%llu "
        "class_transitions=%llu\n",
        validJmSlots, invalidJmSlots, jmEmpty, jmDenseAllThreads,
        jmUniformSparse, jmMixed, jmNoExclusion, jmDenseNoExclusion,
        jmUniformSparseNoExclusion, denseRuns, genericRuns, classTransitions);
    std::printf(
        "analysis=sponge-production-gmxpacked-split-class all_empty=%llu "
        "all_dense=%llu all_generic=%llu dense_only_with_empty=%llu "
        "generic_only_with_empty=%llu mixed_dense_generic=%llu\n",
        splitAllEmpty, splitAllDense, splitAllGeneric, splitDenseOnlyWithEmpty,
        splitGenericOnlyWithEmpty, splitMixedDenseGeneric);
    std::printf(
        "analysis=sponge-production-gmxpacked-sci-class empty=%llu "
        "dense_only=%llu generic_only=%llu mixed_dense_generic=%llu "
        "dense_frac_ge25=%llu dense_frac_ge50=%llu dense_frac_ge75=%llu\n",
        sciEmpty, sciDenseOnly, sciGenericOnly, sciMixedDenseGeneric,
        sciDenseFracGe25, sciDenseFracGe50, sciDenseFracGe75);
}

GromacsPairlistSnapshot ConvertSpongeGmxpackedSnapshotToGromacs(
    const SpongeGmxpackedForceOnlySnapshot& snapshot,
    bool computeEnergy = false,
    bool computeVirial = false)
{
    GromacsPairlistSnapshot converted = {};
    converted.header.file = nbnxm_microbench::MakeFileHeader(
        nbnxm_microbench::SnapshotKind::gromacsPairlist);
    converted.header.cluster_size = snapshot.header.cluster_size;
    converted.header.super_cluster_clusters =
        snapshot.header.super_cluster_clusters;
    converted.header.cluster_pair_split = snapshot.header.warp_split_count;
    converted.header.j_group_size = snapshot.header.j_group_size;
    converted.header.elec_type = 4u;
    converted.header.vdw_type = 1u;
    converted.header.num_threads_z = 1u;
    converted.header.compute_energy = computeEnergy ? 1u : 0u;
    converted.header.compute_virial = computeVirial ? 1u : 0u;
    converted.header.use_prune_kernel = 0u;
    converted.header.cluster_numbers = snapshot.header.cluster_numbers;
    converted.header.sci_numbers = snapshot.sci.size();
    converted.header.cjpacked_numbers = snapshot.cjpacked.size();
    converted.header.excl_numbers = snapshot.excl.size();
    converted.header.total_atom_numbers = snapshot.header.total_atom_numbers;
    converted.header.local_atom_numbers = snapshot.header.local_atom_numbers;
    converted.header.lj_param_numbers = snapshot.lj_ab.size();
    converted.header.cutoff = snapshot.header.cutoff;
    converted.header.pme_beta = snapshot.header.pme_beta;
    converted.header.epsfac = 1.0f;
    converted.header.cell = snapshot.header.cell;
    int numTypes = 0;
    for (int type : snapshot.sorted_lj_type)
    {
        numTypes = std::max(numTypes, type + 1);
    }
    converted.header.num_types = static_cast<uint32_t>(numTypes);
    const LTMatrix3 cell = MakeMatrix(snapshot.header.cell);
    for (int shift = 0; shift < kShiftCount; ++shift)
    {
        const Vec3 shiftVec = ShiftVectorFromId(shift, cell);
        converted.header.shiftvec[static_cast<size_t>(shift)] = {
            shiftVec.x, shiftVec.y, shiftVec.z, 0.0f};
    }

    converted.cluster_offsets = snapshot.cluster_offsets;
    converted.cluster_valid_masks = snapshot.cluster_valid_masks;
    converted.cluster_local_masks = snapshot.cluster_local_masks;
    converted.sorted_atom_ids = snapshot.sorted_atom_ids;
    converted.sorted_xq = snapshot.sorted_xq;
    converted.sorted_lj_type = snapshot.sorted_lj_type;
    converted.sorted_lj_comb = snapshot.sorted_lj_comb;
    converted.lj_ab = snapshot.lj_ab;

    converted.sci.reserve(snapshot.sci.size());
    for (const SpongeGmxpackedSciPOD& sci : snapshot.sci)
    {
        converted.sci.push_back({sci.supercluster_id, sci.shift_id,
                                 sci.cjpacked_begin, sci.cjpacked_end});
    }

    converted.cjpacked.reserve(snapshot.cjpacked.size());
    for (const SpongeGmxpackedCjPOD& cj : snapshot.cjpacked)
    {
        GromacsCjPackedPOD packed = {};
        std::memcpy(packed.cj, cj.cj, sizeof(packed.cj));
        for (int split = 0; split < kWarpSplitCount; ++split)
        {
            packed.imei[split].imask = cj.split[split].imask;
            packed.imei[split].excl_ind = cj.split[split].exclusion_index;
        }
        converted.cjpacked.push_back(packed);
    }

    converted.excl.reserve(snapshot.excl.size());
    for (const SpongeGmxpackedExclusionPOD& exclusion : snapshot.excl)
    {
        GromacsExclPOD excl = {};
        std::memcpy(excl.pair, exclusion.pair, sizeof(excl.pair));
        converted.excl.push_back(excl);
    }
    return converted;
}

std::vector<uint64_t> BuildGmxpackedOracleCutoffPassMasks(
    const SpongeGmxpackedForceOnlySnapshot& snapshot,
    uint64_t* source_sites_out, uint64_t* kept_sites_out,
    uint64_t* conservative_oob_sites_out)
{
    std::vector<uint64_t> pass_masks(snapshot.cjpacked.size(), 0ull);
    uint64_t source_sites = 0ull;
    uint64_t kept_sites = 0ull;
    uint64_t conservative_oob_sites = 0ull;
    const LTMatrix3 cell = MakeMatrix(snapshot.header.cell);
    const float cutoff_sq = snapshot.header.cutoff * snapshot.header.cutoff;
    const float conservative_cutoff_sq =
        cutoff_sq * (1.0f + kOracleCutoffGuard);

    for (size_t sci_index = 0; sci_index < snapshot.sci.size(); ++sci_index)
    {
        const SpongeGmxpackedSciPOD& sci = snapshot.sci[sci_index];
        const bool sci_shift_safe =
            sci_index < snapshot.sci_shift_safe_flags.size() &&
            snapshot.sci_shift_safe_flags[sci_index] != 0;
        const Vec3 sci_shift = ShiftVectorFromId(sci.shift_id, cell);
        const int cluster_i_start = sci.supercluster_id * kSuperClusterClusters;
        for (int packed_index = sci.cjpacked_begin;
             packed_index < sci.cjpacked_end; ++packed_index)
        {
            if (packed_index < 0 ||
                static_cast<size_t>(packed_index) >= snapshot.cjpacked.size())
            {
                continue;
            }
            const SpongeGmxpackedCjPOD& packed =
                snapshot.cjpacked[static_cast<size_t>(packed_index)];
            for (int split = 0; split < kWarpSplitCount; ++split)
            {
                const unsigned int imask = packed.split[split].imask;
                const int exclusion_index =
                    packed.split[split].exclusion_index;
                if (imask == 0u)
                {
                    continue;
                }
                for (int jm = 0; jm < kJGroupSize; ++jm)
                {
                    const int cluster_j = packed.cj[jm];
                    const uint64_t pair_shift_word_index =
                        static_cast<uint64_t>(packed_index) * kJGroupSize + jm;
                    const uint64_t pair_shift_word =
                        pair_shift_word_index < snapshot.pair_shift_bits.size()
                            ? snapshot.pair_shift_bits[
                                  static_cast<size_t>(pair_shift_word_index)]
                            : 0ull;
                    for (int i_local = 0; i_local < kSuperClusterClusters;
                         ++i_local)
                    {
                        const unsigned int packed_bit =
                            1u << (jm * kSuperClusterClusters + i_local);
                        if ((imask & packed_bit) == 0u)
                        {
                            continue;
                        }
                        ++source_sites;
                        bool any_pass = false;
                        bool conservative_oob = cluster_j < 0;
                        for (int split_j_lane = 0;
                             split_j_lane < kSplitJClusterSize && !any_pass;
                             ++split_j_lane)
                        {
                            const int j_lane =
                                split * kSplitJClusterSize + split_j_lane;
                            const uint64_t sorted_j =
                                static_cast<uint64_t>(cluster_j) * kClusterSize +
                                j_lane;
                            for (int i_lane = 0;
                                 i_lane < kClusterSize && !any_pass; ++i_lane)
                            {
                                unsigned int pair_bits = 0xffffffffu;
                                if (exclusion_index != 0)
                                {
                                    if (exclusion_index < 0 ||
                                        static_cast<size_t>(exclusion_index) >=
                                            snapshot.excl.size())
                                    {
                                        conservative_oob = true;
                                        break;
                                    }
                                    pair_bits =
                                        snapshot.excl[static_cast<size_t>(
                                                          exclusion_index)]
                                            .pair[split_j_lane * kClusterSize +
                                                  i_lane];
                                }
                                if ((pair_bits & packed_bit) == 0u)
                                {
                                    continue;
                                }
                                const uint64_t sorted_i =
                                    static_cast<uint64_t>(cluster_i_start +
                                                          i_local) *
                                        kClusterSize +
                                    i_lane;
                                if (sorted_i >= snapshot.sorted_xq.size() ||
                                    sorted_j >= snapshot.sorted_xq.size())
                                {
                                    conservative_oob = true;
                                    break;
                                }
                                Vec3 pair_shift = sci_shift;
                                if (!sci_shift_safe &&
                                    pair_shift_word_index <
                                        snapshot.pair_shift_bits.size())
                                {
                                    pair_shift = ShiftVectorFromId(
                                        GetPairShiftId(pair_shift_word, i_local),
                                        cell);
                                }
                                const Float4POD& r1 = snapshot.sorted_xq[
                                    static_cast<size_t>(sorted_i)];
                                const Float4POD& r2 = snapshot.sorted_xq[
                                    static_cast<size_t>(sorted_j)];
                                const float dx = r2.x - r1.x - pair_shift.x;
                                const float dy = r2.y - r1.y - pair_shift.y;
                                const float dz = r2.z - r1.z - pair_shift.z;
                                const float dr2 = std::fma(
                                    dx, dx, std::fma(dy, dy, dz * dz));
                                any_pass = dr2 < conservative_cutoff_sq;
                            }
                            if (conservative_oob)
                            {
                                break;
                            }
                        }
                        if (any_pass || conservative_oob)
                        {
                            const int bit_index =
                                split * 32 + jm * kSuperClusterClusters +
                                i_local;
                            pass_masks[static_cast<size_t>(packed_index)] |=
                                1ull << static_cast<unsigned int>(bit_index);
                            ++kept_sites;
                            if (conservative_oob)
                            {
                                ++conservative_oob_sites;
                            }
                        }
                    }
                }
            }
        }
    }
    if (source_sites_out != nullptr)
    {
        *source_sites_out = source_sites;
    }
    if (kept_sites_out != nullptr)
    {
        *kept_sites_out = kept_sites;
    }
    if (conservative_oob_sites_out != nullptr)
    {
        *conservative_oob_sites_out = conservative_oob_sites;
    }
    return pass_masks;
}

struct SparseSitsGmxpackedStream
{
    std::vector<SpongeGmxpackedSciPOD> sci;
    std::vector<SpongeGmxpackedCjPOD> cjpacked;
    std::vector<uint64_t> pair_shift_bits;
    std::vector<int> sci_shift_safe_flags;
    uint64_t source_tile_bits = 0;
    uint64_t selected_tile_bits = 0;
};

SparseSitsGmxpackedStream BuildSparseSitsGmxpackedStream(
    const SpongeGmxpackedForceOnlySnapshot& snapshot, int selective_atom_end)
{
    SparseSitsGmxpackedStream result;
    const int cluster_numbers =
        static_cast<int>(snapshot.header.cluster_numbers);
    std::vector<unsigned int> selected_lane_masks(
        static_cast<size_t>(std::max(cluster_numbers, 0)), 0u);
    for (int cluster = 0; cluster < cluster_numbers; ++cluster)
    {
        unsigned int selected_mask = 0u;
        for (int lane = 0; lane < kClusterSize; ++lane)
        {
            const size_t sorted =
                static_cast<size_t>(cluster * kClusterSize + lane);
            if (sorted >= snapshot.sorted_atom_ids.size())
            {
                continue;
            }
            const int atom = snapshot.sorted_atom_ids[sorted];
            if (atom >= 0 && atom < selective_atom_end)
            {
                selected_mask |= 1u << static_cast<unsigned int>(lane);
            }
        }
        selected_lane_masks[static_cast<size_t>(cluster)] = selected_mask;
    }

    result.sci.reserve(snapshot.sci.size());
    result.cjpacked.reserve(snapshot.cjpacked.size());
    result.pair_shift_bits.reserve(snapshot.pair_shift_bits.size());
    result.sci_shift_safe_flags.reserve(snapshot.sci_shift_safe_flags.size());
    for (size_t sci_index = 0; sci_index < snapshot.sci.size(); ++sci_index)
    {
        const SpongeGmxpackedSciPOD& source_sci = snapshot.sci[sci_index];
        SpongeGmxpackedSciPOD compact_sci = source_sci;
        compact_sci.cjpacked_begin =
            static_cast<int>(result.cjpacked.size());
        const int cluster_i_start =
            source_sci.supercluster_id * kSuperClusterClusters;
        for (int packed_index = source_sci.cjpacked_begin;
             packed_index < source_sci.cjpacked_end; ++packed_index)
        {
            if (packed_index < 0 ||
                static_cast<size_t>(packed_index) >= snapshot.cjpacked.size())
            {
                continue;
            }
            SpongeGmxpackedCjPOD compact_packed =
                snapshot.cjpacked[static_cast<size_t>(packed_index)];
            bool packed_selected = false;
            for (int split = 0; split < kWarpSplitCount; ++split)
            {
                const unsigned int source_imask =
                    compact_packed.split[split].imask;
                result.source_tile_bits +=
                    static_cast<uint64_t>(std::popcount(source_imask));
                unsigned int selected_imask = 0u;
                const unsigned int split_j_lane_mask =
                    ((1u << kSplitJClusterSize) - 1u)
                    << static_cast<unsigned int>(split *
                                                  kSplitJClusterSize);
                for (int jm = 0; jm < kJGroupSize; ++jm)
                {
                    const int cluster_j = compact_packed.cj[jm];
                    const unsigned int selected_j_lanes =
                        cluster_j >= 0 && cluster_j < cluster_numbers
                            ? selected_lane_masks[
                                  static_cast<size_t>(cluster_j)]
                            : 0u;
                    for (int i_local = 0;
                         i_local < kSuperClusterClusters; ++i_local)
                    {
                        const unsigned int packed_bit =
                            1u << static_cast<unsigned int>(
                                jm * kSuperClusterClusters + i_local);
                        if ((source_imask & packed_bit) == 0u)
                        {
                            continue;
                        }
                        const int cluster_i = cluster_i_start + i_local;
                        const unsigned int selected_i_lanes =
                            cluster_i >= 0 && cluster_i < cluster_numbers
                                ? selected_lane_masks[
                                      static_cast<size_t>(cluster_i)]
                                : 0u;
                        if (selected_i_lanes != 0u ||
                            (selected_j_lanes & split_j_lane_mask) != 0u)
                        {
                            selected_imask |= packed_bit;
                        }
                    }
                }
                compact_packed.split[split].imask = selected_imask;
                result.selected_tile_bits +=
                    static_cast<uint64_t>(std::popcount(selected_imask));
                packed_selected = packed_selected || selected_imask != 0u;
            }
            if (!packed_selected)
            {
                continue;
            }
            result.cjpacked.push_back(compact_packed);
            for (int jm = 0; jm < kJGroupSize; ++jm)
            {
                const size_t shift_index =
                    static_cast<size_t>(packed_index * kJGroupSize + jm);
                result.pair_shift_bits.push_back(
                    shift_index < snapshot.pair_shift_bits.size()
                        ? snapshot.pair_shift_bits[shift_index]
                        : 0ull);
            }
        }
        compact_sci.cjpacked_end =
            static_cast<int>(result.cjpacked.size());
        if (compact_sci.cjpacked_begin == compact_sci.cjpacked_end)
        {
            continue;
        }
        result.sci.push_back(compact_sci);
        result.sci_shift_safe_flags.push_back(
            sci_index < snapshot.sci_shift_safe_flags.size()
                ? snapshot.sci_shift_safe_flags[sci_index]
                : 0);
    }
    return result;
}

void RunSpongeProductionGmxpacked(
    const SpongeGmxpackedForceOnlySnapshot& snapshot, int warmup, int iters,
    bool computeEnergy, bool computeVirial, const char* snapshotLabel,
    ProductionGmxpackedReplayMode replayMode =
        ProductionGmxpackedReplayMode::split,
    const SpongeGmxpackedFullOutputSnapshot* fullOutputReference = nullptr,
    int sitsAtomEnd = -1, float sitsPwwpFactor = 1.0f)
{
    if (snapshot.header.cluster_size != kClusterSize ||
        snapshot.header.super_cluster_clusters != kSuperClusterClusters ||
        snapshot.header.warp_split_count != kWarpSplitCount ||
        snapshot.header.j_group_size != kJGroupSize)
    {
        std::fprintf(stderr,
                     "production-gmxpacked replay requires 8x8 gmxpacked "
                     "payload, got cluster=%u super=%u split=%u jgroup=%u "
                     "use_lj_comb=%u\n",
                     snapshot.header.cluster_size,
                     snapshot.header.super_cluster_clusters,
                     snapshot.header.warp_split_count,
                     snapshot.header.j_group_size,
                     snapshot.header.use_lj_comb);
        std::exit(1);
    }
    if ((computeEnergy || computeVirial) && snapshot.sci.empty())
    {
        std::fprintf(stderr,
                     "production-gmxpacked energy/virial replay needs a non-empty payload\n");
        std::exit(1);
    }

    const int sci_numbers = static_cast<int>(snapshot.header.sci_numbers);
    const int cluster_numbers =
        static_cast<int>(snapshot.header.cluster_numbers);
    const int total_atom_numbers =
        static_cast<int>(snapshot.header.total_atom_numbers);
    const bool fusedSitsForceOnly =
        replayMode == ProductionGmxpackedReplayMode::fusedSitsForceOnly;
    const bool sparseSitsForceOnly =
        replayMode == ProductionGmxpackedReplayMode::sparseSitsForceOnly;
    const bool sitsForceOnly = fusedSitsForceOnly || sparseSitsForceOnly;
    if (sitsForceOnly &&
        (computeEnergy || computeVirial || sitsAtomEnd < 0 ||
         sitsAtomEnd > total_atom_numbers))
    {
        std::fprintf(
            stderr,
            "fused SITS force-only replay requires force-only output and "
            "--sits-atom-end in [0, %d]\n",
            total_atom_numbers);
        std::exit(1);
    }
    if (fullOutputReference != nullptr &&
        (fullOutputReference->header.compute_energy !=
             static_cast<uint32_t>(computeEnergy) ||
         fullOutputReference->header.compute_virial !=
             static_cast<uint32_t>(computeVirial) ||
         fullOutputReference->header.total_output != 0u ||
         fullOutputReference->header.force_soa != 0u ||
         fullOutputReference->reference_force.size() !=
             static_cast<size_t>(total_atom_numbers)))
    {
        std::fprintf(
            stderr,
            "production-gmxpacked full-output reference contract mismatch "
            "(energy=%u/%u virial=%u/%u total_output=%u force_soa=%u "
            "force_refs=%zu atoms=%d)\n",
            fullOutputReference->header.compute_energy,
            computeEnergy ? 1u : 0u,
            fullOutputReference->header.compute_virial,
            computeVirial ? 1u : 0u,
            fullOutputReference->header.total_output,
            fullOutputReference->header.force_soa,
            fullOutputReference->reference_force.size(),
            total_atom_numbers);
        std::exit(1);
    }
    const bool useLjComb = snapshot.header.use_lj_comb != 0u;
    if (sparseSitsForceOnly && useLjComb)
    {
        std::fprintf(
            stderr,
            "sparse SITS force-only replay currently targets packed AB-table snapshots only\n");
        std::exit(1);
    }
    const bool compactForceOnly =
        replayMode == ProductionGmxpackedReplayMode::compactForce;
    const bool sortedForceOnly =
        replayMode == ProductionGmxpackedReplayMode::sortedForce ||
        replayMode == ProductionGmxpackedReplayMode::sparseSitsForceOnly ||
        replayMode == ProductionGmxpackedReplayMode::sortedForceSciSplit2 ||
        replayMode == ProductionGmxpackedReplayMode::sortedForceSciSplit3 ||
        replayMode == ProductionGmxpackedReplayMode::sortedForceSciSplit4 ||
        replayMode == ProductionGmxpackedReplayMode::sortedForceLocalIMask8 ||
        replayMode == ProductionGmxpackedReplayMode::sortedForceActiveIMask8 ||
        replayMode == ProductionGmxpackedReplayMode::sortedForceOracleImask ||
        replayMode == ProductionGmxpackedReplayMode::sortedForceOracleSidecar ||
        replayMode == ProductionGmxpackedReplayMode::sortedForceDeviceSidecar ||
        replayMode == ProductionGmxpackedReplayMode::sortedForceDenseNoExcl ||
        replayMode == ProductionGmxpackedReplayMode::sortedForceAttrAllI ||
        replayMode == ProductionGmxpackedReplayMode::sortedForceAttrNoCutoff ||
        replayMode == ProductionGmxpackedReplayMode::sortedForceAttrAllINoCutoff ||
        replayMode == ProductionGmxpackedReplayMode::sortedForceNoWriteback;
    const bool specializedSortedForceOnly =
        replayMode == ProductionGmxpackedReplayMode::specializedSortedForce;
    const bool specializedShiftvecOnly =
        replayMode == ProductionGmxpackedReplayMode::specializedShiftvec;
    const bool shiftVirialMode =
        replayMode == ProductionGmxpackedReplayMode::shiftVirial;
    const bool totalOutput = shiftVirialMode && computeVirial;
    const int scalar_output_numbers = totalOutput ? 1 : total_atom_numbers;
    if ((compactForceOnly || sortedForceOnly || specializedSortedForceOnly ||
         specializedShiftvecOnly) &&
        (computeEnergy || computeVirial))
    {
        std::fprintf(stderr,
                     "production-gmxpacked sorted/compact force replay is force-only; "
                     "use production-gmxpacked for energy/virial\n");
        std::exit(1);
    }
    if (shiftVirialMode && (!totalOutput || !computeVirial))
    {
        std::fprintf(stderr,
                     "production-gmxpacked shift-virial replay requires "
                     "--compute-virial\n");
        std::exit(1);
    }
    if (!useLjComb &&
        (replayMode == ProductionGmxpackedReplayMode::specializedSafe ||
         replayMode == ProductionGmxpackedReplayMode::specializedSortedForce ||
         replayMode == ProductionGmxpackedReplayMode::specializedShiftvec ||
         replayMode == ProductionGmxpackedReplayMode::shiftVirial))
    {
        std::fprintf(stderr,
                     "%s replay currently supports LJ-comb snapshots only\n",
                     ProductionGmxpackedReplayModeName(replayMode));
        std::exit(1);
    }
    if (!useLjComb && snapshot.header.lj_type_matrix_stride != 0u)
    {
        std::fprintf(stderr,
                     "production-gmxpacked replay currently supports packed "
                     "AB-table snapshots only, got matrix_stride=%u\n",
                     snapshot.header.lj_type_matrix_stride);
        std::exit(1);
    }
    const size_t safe_sci = static_cast<size_t>(
        std::count_if(snapshot.sci_shift_safe_flags.begin(),
                      snapshot.sci_shift_safe_flags.end(),
                      [](int value) { return value != 0; }));
    const bool allSciSafe = safe_sci == snapshot.sci.size();
    const SparseSitsGmxpackedStream sparse_sits_stream =
        sparseSitsForceOnly
            ? BuildSparseSitsGmxpackedStream(snapshot, sitsAtomEnd)
            : SparseSitsGmxpackedStream{};
    if ((replayMode == ProductionGmxpackedReplayMode::safeOnly ||
         replayMode == ProductionGmxpackedReplayMode::specializedSafe ||
         replayMode == ProductionGmxpackedReplayMode::specializedSortedForce ||
         replayMode == ProductionGmxpackedReplayMode::specializedShiftvec ||
         replayMode == ProductionGmxpackedReplayMode::shiftVirial) &&
        !allSciSafe)
    {
        std::fprintf(stderr,
                     "%s replay requires all SCI entries to be safe, got "
                     "safe_sci=%zu sci=%zu\n",
                     ProductionGmxpackedReplayModeName(replayMode), safe_sci,
                     snapshot.sci.size());
        std::exit(1);
    }

    int* d_cluster_offsets = CopyVectorToDevice(snapshot.cluster_offsets);
    unsigned int* d_cluster_valid_masks =
        CopyVectorToDevice(snapshot.cluster_valid_masks);
    unsigned int* d_cluster_local_masks =
        CopyVectorToDevice(snapshot.cluster_local_masks);
    int* d_super_cluster_offsets =
        CopyVectorToDevice(snapshot.super_cluster_offsets);
    SpongeGmxpackedSciPOD* d_sci = CopyVectorToDevice(snapshot.sci);
    std::vector<SpongeGmxpackedCjPOD> oracle_cjpacked;
    std::vector<uint64_t> oracle_pass_masks;
    const bool use_oracle_imask =
        replayMode == ProductionGmxpackedReplayMode::sortedForceOracleImask;
    const bool use_oracle_sidecar =
        replayMode == ProductionGmxpackedReplayMode::sortedForceOracleSidecar;
    const bool use_device_sidecar =
        replayMode == ProductionGmxpackedReplayMode::sortedForceDeviceSidecar;
    const bool use_sci_split2 =
        replayMode == ProductionGmxpackedReplayMode::sortedForceSciSplit2;
    const bool use_sci_split3 =
        replayMode == ProductionGmxpackedReplayMode::sortedForceSciSplit3;
    const bool use_sci_split4 =
        replayMode == ProductionGmxpackedReplayMode::sortedForceSciSplit4;
    if ((use_sci_split2 || use_sci_split3 || use_sci_split4) && useLjComb)
    {
        std::fprintf(stderr,
                     "SCI-split sorted force currently targets packed AB-table snapshots only\n");
        std::exit(1);
    }
    if (use_oracle_imask || use_oracle_sidecar || use_device_sidecar)
    {
        uint64_t source_sites = 0ull;
        uint64_t kept_sites = 0ull;
        uint64_t conservative_oob_sites = 0ull;
        oracle_pass_masks = BuildGmxpackedOracleCutoffPassMasks(
            snapshot, &source_sites, &kept_sites,
            &conservative_oob_sites);
        if (use_oracle_imask)
        {
            oracle_cjpacked = snapshot.cjpacked;
            for (size_t packed_index = 0;
                 packed_index < oracle_cjpacked.size(); ++packed_index)
            {
                for (int split = 0; split < kWarpSplitCount; ++split)
                {
                    const unsigned int split_pass_mask =
                        static_cast<unsigned int>(
                            oracle_pass_masks[packed_index] >> (split * 32));
                    oracle_cjpacked[packed_index].split[split].imask &=
                        split_pass_mask;
                }
            }
        }
        const double kept_percent =
            source_sites == 0ull
                ? 0.0
                : 100.0 * static_cast<double>(kept_sites) /
                      static_cast<double>(source_sites);
        std::fprintf(
            stderr,
            "[gmxpacked oracle %s] source_sites=%llu kept_sites=%llu "
            "kept_pct=%.3f conservative_oob_sites=%llu\n",
            use_device_sidecar
                ? "device-reference"
                : (use_oracle_sidecar ? "sidecar" : "imask"),
            static_cast<unsigned long long>(source_sites),
            static_cast<unsigned long long>(kept_sites), kept_percent,
            static_cast<unsigned long long>(conservative_oob_sites));
    }
    const std::vector<SpongeGmxpackedCjPOD>& replay_cjpacked =
        oracle_cjpacked.empty() ? snapshot.cjpacked : oracle_cjpacked;
    SpongeGmxpackedCjPOD* d_cjpacked =
        CopyVectorToDevice(replay_cjpacked);
    uint64_t* d_oracle_pass_masks = nullptr;
    if (use_oracle_sidecar)
    {
        d_oracle_pass_masks = CopyVectorToDevice(oracle_pass_masks);
    }
    else if (use_device_sidecar)
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_oracle_pass_masks),
                             sizeof(uint64_t) * snapshot.cjpacked.size()),
                  "cudaMalloc(device_cutoff_pass_masks)");
    }
    SpongeGmxpackedExclusionPOD* d_excl =
        CopyVectorToDevice(snapshot.excl);
    uint64_t* d_pair_shift_bits =
        CopyVectorToDevice(snapshot.pair_shift_bits);
    int* d_sci_shift_safe_flags =
        CopyVectorToDevice(snapshot.sci_shift_safe_flags);
    SpongeGmxpackedSciPOD* d_sparse_sits_sci =
        CopyVectorToDevice(sparse_sits_stream.sci);
    SpongeGmxpackedCjPOD* d_sparse_sits_cjpacked =
        CopyVectorToDevice(sparse_sits_stream.cjpacked);
    uint64_t* d_sparse_sits_pair_shift_bits =
        CopyVectorToDevice(sparse_sits_stream.pair_shift_bits);
    int* d_sparse_sits_sci_shift_safe_flags =
        CopyVectorToDevice(sparse_sits_stream.sci_shift_safe_flags);
    int* d_sorted_atom_ids = CopyVectorToDevice(snapshot.sorted_atom_ids);
    std::vector<float4> sorted_xq(snapshot.sorted_xq.size());
    for (size_t i = 0; i < snapshot.sorted_xq.size(); ++i)
    {
        sorted_xq[i] = MakeFloat4(snapshot.sorted_xq[i]);
    }
    float4* d_sorted_xq = CopyVectorToDevice(sorted_xq);
    int* d_sorted_lj_type = CopyVectorToDevice(snapshot.sorted_lj_type);
    std::vector<float2> sorted_lj_comb(snapshot.sorted_lj_comb.size());
    for (size_t i = 0; i < snapshot.sorted_lj_comb.size(); ++i)
    {
        sorted_lj_comb[i] = MakeFloat2(snapshot.sorted_lj_comb[i]);
    }
    float2* d_sorted_lj_comb = CopyVectorToDevice(sorted_lj_comb);
    std::vector<float2> lj_ab(snapshot.lj_ab.size());
    for (size_t i = 0; i < snapshot.lj_ab.size(); ++i)
    {
        lj_ab[i] = MakeFloat2(snapshot.lj_ab[i]);
    }
    float2* d_lj_ab = CopyVectorToDevice(lj_ab);
    const LTMatrix3 cell = MakeMatrix(snapshot.header.cell);
    std::vector<float4> shiftvec(kShiftCount);
    for (int shift = 0; shift < kShiftCount; ++shift)
    {
        const Vec3 shift_vec = ShiftVectorFromId(shift, cell);
        shiftvec[static_cast<size_t>(shift)] =
            make_float4(shift_vec.x, shift_vec.y, shift_vec.z, 0.0f);
    }
    float4* d_shiftvec = CopyVectorToDevice(shiftvec);

    float4* d_force_regular = nullptr;
    float4* d_force_enhancing = nullptr;
    float3* d_force_compact = nullptr;
    VECTOR* d_force_shared_production = nullptr;
    const bool useSharedProductionForceOnly =
        replayMode == ProductionGmxpackedReplayMode::split &&
        !computeEnergy && !computeVirial;
    if (totalOutput || compactForceOnly || computeEnergy || computeVirial)
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_force_compact),
                             sizeof(float3) * total_atom_numbers),
                  "cudaMalloc(force_compact)");
    }
    else
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_force_regular),
                             sizeof(float4) * total_atom_numbers),
                  "cudaMalloc(force_regular)");
    }
    if (useSharedProductionForceOnly)
    {
        CheckCuda(
            cudaMalloc(reinterpret_cast<void**>(&d_force_shared_production),
                       sizeof(VECTOR) * total_atom_numbers),
            "cudaMalloc(force_shared_production)");
    }
    if (sitsForceOnly)
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_force_enhancing),
                             sizeof(float4) * total_atom_numbers),
                  "cudaMalloc(force_enhancing)");
    }
    float* d_atom_energy = nullptr;
    float* d_atom_direct_cf_energy = nullptr;
    float* d_atom_lj_energy = nullptr;
    LTMatrix3* d_atom_virial = nullptr;
    float3* d_shift_force = nullptr;
    if (computeEnergy)
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_atom_energy),
                             sizeof(float) * scalar_output_numbers),
                  "cudaMalloc(atom_energy)");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_atom_direct_cf_energy),
                             sizeof(float) * scalar_output_numbers),
                  "cudaMalloc(atom_direct_cf_energy)");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_atom_lj_energy),
                             sizeof(float) * scalar_output_numbers),
                  "cudaMalloc(atom_lj_energy)");
    }
    if (computeVirial && shiftVirialMode)
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_shift_force),
                             sizeof(float3) * shiftvec.size()),
                  "cudaMalloc(shift_force)");
    }
    else if (computeVirial)
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_atom_virial),
                             sizeof(LTMatrix3) * scalar_output_numbers),
                  "cudaMalloc(atom_virial)");
    }

    const dim3 block(kClusterSize, kClusterSize, 1);
    const dim3 grid(static_cast<unsigned int>(sci_numbers), 1, 1);
    const dim3 sci_split2_grid(
        static_cast<unsigned int>(sci_numbers * 2), 1, 1);
    const dim3 sci_split3_grid(
        static_cast<unsigned int>(sci_numbers * 3), 1, 1);
    const dim3 sci_split4_grid(
        static_cast<unsigned int>(sci_numbers * 4), 1, 1);
    const dim3 sci_split8_grid(
        static_cast<unsigned int>(sci_numbers * 8), 1, 1);
    constexpr int sparse_sits_work_parts = 8;
    const int sparse_sits_sci_numbers =
        static_cast<int>(sparse_sits_stream.sci.size());
    const size_t sparse_sits_safe_sci = static_cast<size_t>(
        std::count_if(sparse_sits_stream.sci_shift_safe_flags.begin(),
                      sparse_sits_stream.sci_shift_safe_flags.end(),
                      [](int value) { return value != 0; }));
    const bool sparse_sits_all_safe =
        sparse_sits_safe_sci == sparse_sits_stream.sci.size();
    const dim3 sparse_sits_grid(
        static_cast<unsigned int>(sparse_sits_sci_numbers *
                                  sparse_sits_work_parts),
        1, 1);
    auto clearOutputs = [&]() {
        if (d_force_compact != nullptr)
        {
            CheckCuda(cudaMemset(d_force_compact, 0,
                                 sizeof(float3) * total_atom_numbers),
                      "cudaMemset(force_compact)");
        }
        if (d_force_regular != nullptr)
        {
            CheckCuda(cudaMemset(d_force_regular, 0,
                                 sizeof(float4) * total_atom_numbers),
                      "cudaMemset(force_regular)");
        }
        if (d_force_enhancing != nullptr)
        {
            CheckCuda(cudaMemset(d_force_enhancing, 0,
                                 sizeof(float4) * total_atom_numbers),
                      "cudaMemset(force_enhancing)");
        }
        if (d_force_shared_production != nullptr)
        {
            CheckCuda(
                cudaMemset(d_force_shared_production, 0,
                           sizeof(VECTOR) * total_atom_numbers),
                "cudaMemset(force_shared_production)");
        }
        if (d_atom_energy != nullptr)
        {
            CheckCuda(cudaMemset(d_atom_energy, 0,
                                 sizeof(float) * scalar_output_numbers),
                      "cudaMemset(atom_energy)");
            CheckCuda(cudaMemset(d_atom_direct_cf_energy, 0,
                                 sizeof(float) * scalar_output_numbers),
                      "cudaMemset(atom_direct_cf_energy)");
            CheckCuda(cudaMemset(d_atom_lj_energy, 0,
                                 sizeof(float) * scalar_output_numbers),
                      "cudaMemset(atom_lj_energy)");
        }
        if (d_atom_virial != nullptr)
        {
            CheckCuda(cudaMemset(d_atom_virial, 0,
                                 sizeof(LTMatrix3) * scalar_output_numbers),
                      "cudaMemset(atom_virial)");
        }
        if (d_shift_force != nullptr)
        {
            CheckCuda(cudaMemset(d_shift_force, 0,
                                 sizeof(float3) * shiftvec.size()),
                      "cudaMemset(shift_force)");
        }
    };
    auto launchDeviceCutoffSidecar = [&]() {
        if (safe_sci != 0)
        {
            SpongeProductionGmxpackedCutoffMaskKernel<true><<<grid, block>>>(
                sci_numbers, cluster_numbers, d_sci, d_cjpacked, d_excl,
                nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags, 1,
                d_sorted_xq, cell, snapshot.header.cutoff,
                d_oracle_pass_masks);
            CheckCuda(
                cudaGetLastError(),
                "launch production-gmxpacked device cutoff-sidecar fast");
        }
        if (safe_sci != snapshot.sci.size())
        {
            SpongeProductionGmxpackedCutoffMaskKernel<false><<<grid, block>>>(
                sci_numbers, cluster_numbers, d_sci, d_cjpacked, d_excl,
                d_pair_shift_bits,
                safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                d_sorted_xq, cell, snapshot.header.cutoff,
                d_oracle_pass_masks);
            CheckCuda(
                cudaGetLastError(),
                "launch production-gmxpacked device cutoff-sidecar slow");
        }
    };
    auto launchForceOnlySplit = [&]() {
        const auto* shared_sci =
            reinterpret_cast<const LJ_CLUSTERED_GMXPACKED_SCI*>(d_sci);
        const auto* shared_cjpacked =
            reinterpret_cast<const LJ_CLUSTERED_GMXPACKED_CJ*>(d_cjpacked);
        const auto* shared_excl =
            reinterpret_cast<const LJ_CLUSTERED_GMXPACKED_EXCLUSION*>(d_excl);
        if (safe_sci != 0 && useLjComb)
        {
            Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                false, false, false, false, true, true, false, true>
                <<<grid, block>>>(
                    sci_numbers, kClusterSize, kSuperClusterClusters,
                    cluster_numbers, d_cluster_offsets, d_cluster_valid_masks,
                    d_cluster_local_masks, d_super_cluster_offsets, shared_sci,
                    shared_cjpacked, shared_excl, nullptr,
                    allSciSafe ? nullptr : d_sci_shift_safe_flags, 1,
                    d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                    d_sorted_lj_comb, cell, d_lj_ab, snapshot.header.cutoff,
                    d_force_shared_production, snapshot.header.pme_beta,
                    nullptr, nullptr, nullptr, nullptr, false, false);
        }
        else if (safe_sci != 0)
        {
            Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                false, false, false, false, false, true, false, true>
                <<<grid, block>>>(
                    sci_numbers, kClusterSize, kSuperClusterClusters,
                    cluster_numbers, d_cluster_offsets, d_cluster_valid_masks,
                    d_cluster_local_masks, d_super_cluster_offsets, shared_sci,
                    shared_cjpacked, shared_excl, nullptr,
                    allSciSafe ? nullptr : d_sci_shift_safe_flags, 1,
                    d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                    d_sorted_lj_comb, cell, d_lj_ab, snapshot.header.cutoff,
                    d_force_shared_production, snapshot.header.pme_beta,
                    nullptr, nullptr, nullptr, nullptr, false, false);
        }
        if (safe_sci != snapshot.sci.size() && useLjComb)
        {
            Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                false, false, false, false, true, true, false, false>
                <<<grid, block>>>(
                    sci_numbers, kClusterSize, kSuperClusterClusters,
                    cluster_numbers, d_cluster_offsets, d_cluster_valid_masks,
                    d_cluster_local_masks, d_super_cluster_offsets, shared_sci,
                    shared_cjpacked, shared_excl, d_pair_shift_bits,
                    safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                    d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                    d_sorted_lj_comb, cell, d_lj_ab, snapshot.header.cutoff,
                    d_force_shared_production, snapshot.header.pme_beta,
                    nullptr, nullptr, nullptr, nullptr, false, false);
        }
        else if (safe_sci != snapshot.sci.size())
        {
            Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                false, false, false, false, false, true, false, false>
                <<<grid, block>>>(
                    sci_numbers, kClusterSize, kSuperClusterClusters,
                    cluster_numbers, d_cluster_offsets, d_cluster_valid_masks,
                    d_cluster_local_masks, d_super_cluster_offsets, shared_sci,
                    shared_cjpacked, shared_excl, d_pair_shift_bits, nullptr, 0,
                    d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                    d_sorted_lj_comb, cell, d_lj_ab, snapshot.header.cutoff,
                    d_force_shared_production, snapshot.header.pme_beta,
                    nullptr, nullptr, nullptr, nullptr, false, false);
        }
        CheckCuda(cudaGetLastError(),
                  "launch shared production-gmxpacked force-only");
    };
    auto launchForceOnlyFusedSits = [&]() {
        if (safe_sci != 0)
        {
            if (useLjComb)
            {
                SpongeProductionGmxpackedReplayKernel<
                    false, false, false, false, true, float4, false, true,
                    false, false, false, false, false, false, false, 1, false,
                    true>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags,
                        1, d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr, false, false, d_force_enhancing,
                        sitsAtomEnd, sitsPwwpFactor);
            }
            else
            {
                SpongeProductionGmxpackedReplayKernel<
                    false, false, false, false, true, float4, false, false,
                    false, false, false, false, false, false, false, 4, false,
                    true>
                    <<<sci_split4_grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags,
                        1, d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr, false, false, d_force_enhancing,
                        sitsAtomEnd, sitsPwwpFactor);
            }
            CheckCuda(cudaGetLastError(),
                      "launch production-gmxpacked fused SITS fast");
        }
        if (safe_sci != snapshot.sci.size())
        {
            if (useLjComb)
            {
                SpongeProductionGmxpackedReplayKernel<
                    false, false, false, false, false, float4, false, true,
                    false, false, false, false, false, false, false, 1, false,
                    true>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        d_pair_shift_bits,
                        safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr, false, false, d_force_enhancing,
                        sitsAtomEnd, sitsPwwpFactor);
            }
            else
            {
                SpongeProductionGmxpackedReplayKernel<
                    false, false, false, false, false, float4, false, false,
                    false, false, false, false, false, false, false, 4, false,
                    true>
                    <<<sci_split4_grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        d_pair_shift_bits,
                        safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr, false, false, d_force_enhancing,
                        sitsAtomEnd, sitsPwwpFactor);
            }
            CheckCuda(cudaGetLastError(),
                      "launch production-gmxpacked fused SITS slow");
        }
    };
    auto launchForceOnlyCompact = [&]() {
        if (safe_sci != 0)
        {
            if (useLjComb)
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, true, float3>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags,
                        1, d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_compact,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            else
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, true, float3,
                                                      false, false>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags,
                        1, d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_compact,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            CheckCuda(cudaGetLastError(),
                      "launch production-gmxpacked compact fast");
        }
        if (safe_sci != snapshot.sci.size())
        {
            if (useLjComb)
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, false, float3>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        d_pair_shift_bits,
                        safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_compact,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            else
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, false, float3,
                                                      false, false>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        d_pair_shift_bits,
                        safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_compact,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            CheckCuda(cudaGetLastError(),
                      "launch production-gmxpacked compact slow");
        }
    };
    auto launchForceOnlySorted = [&]() {
        if (safe_sci != 0)
        {
            if (useLjComb)
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, true, float4>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags,
                        1, d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            else
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, true, float4,
                                                      false, false>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags,
                        1, d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            CheckCuda(cudaGetLastError(),
                      "launch production-gmxpacked sorted fast");
        }
        if (safe_sci != snapshot.sci.size())
        {
            if (useLjComb)
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, false, float4>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        d_pair_shift_bits,
                        safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            else
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, false, float4,
                                                      false, false>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        d_pair_shift_bits,
                        safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            CheckCuda(cudaGetLastError(),
                      "launch production-gmxpacked sorted slow");
        }
    };
    auto launchForceOnlySortedSciSplit2 = [&]() {
        if (safe_sci != 0)
        {
            SpongeProductionGmxpackedReplayKernel<
                false, false, false, true, true, float4, false, false, false,
                false, false, false, false, false, false, 2>
                <<<sci_split2_grid, block>>>(
                    sci_numbers, cluster_numbers, d_cluster_offsets,
                    d_cluster_valid_masks, d_cluster_local_masks,
                    d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                    nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags, 1,
                    d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                    d_sorted_lj_comb, cell, d_lj_ab, snapshot.header.cutoff,
                    d_force_regular, snapshot.header.pme_beta, nullptr, nullptr,
                    nullptr, nullptr, nullptr);
            CheckCuda(cudaGetLastError(),
                      "launch production-gmxpacked sorted sci-split2 fast");
        }
        if (safe_sci != snapshot.sci.size())
        {
            SpongeProductionGmxpackedReplayKernel<
                false, false, false, true, false, float4, false, false, false,
                false, false, false, false, false, false, 2>
                <<<sci_split2_grid, block>>>(
                    sci_numbers, cluster_numbers, d_cluster_offsets,
                    d_cluster_valid_masks, d_cluster_local_masks,
                    d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                    d_pair_shift_bits,
                    safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                    d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                    d_sorted_lj_comb, cell, d_lj_ab, snapshot.header.cutoff,
                    d_force_regular, snapshot.header.pme_beta, nullptr, nullptr,
                    nullptr, nullptr, nullptr);
            CheckCuda(cudaGetLastError(),
                      "launch production-gmxpacked sorted sci-split2 slow");
        }
    };
    auto launchForceOnlySortedSciSplit3 = [&]() {
        if (safe_sci != 0)
        {
            SpongeProductionGmxpackedReplayKernel<
                false, false, false, true, true, float4, false, false, false,
                false, false, false, false, false, false, 3, true>
                <<<sci_split3_grid, block>>>(
                    sci_numbers, cluster_numbers, d_cluster_offsets,
                    d_cluster_valid_masks, d_cluster_local_masks,
                    d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                    nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags, 1,
                    d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                    d_sorted_lj_comb, cell, d_lj_ab, snapshot.header.cutoff,
                    d_force_regular, snapshot.header.pme_beta, nullptr, nullptr,
                    nullptr, nullptr, nullptr);
            CheckCuda(cudaGetLastError(),
                      "launch production-gmxpacked sorted sci-split3 fast");
        }
        if (safe_sci != snapshot.sci.size())
        {
            SpongeProductionGmxpackedReplayKernel<
                false, false, false, true, false, float4, false, false, false,
                false, false, false, false, false, false, 3, true>
                <<<sci_split3_grid, block>>>(
                    sci_numbers, cluster_numbers, d_cluster_offsets,
                    d_cluster_valid_masks, d_cluster_local_masks,
                    d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                    d_pair_shift_bits,
                    safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                    d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                    d_sorted_lj_comb, cell, d_lj_ab, snapshot.header.cutoff,
                    d_force_regular, snapshot.header.pme_beta, nullptr, nullptr,
                    nullptr, nullptr, nullptr);
            CheckCuda(cudaGetLastError(),
                      "launch production-gmxpacked sorted sci-split3 slow");
        }
    };
    auto launchForceOnlySortedSciSplit4 = [&]() {
        if (safe_sci != 0)
        {
            SpongeProductionGmxpackedReplayKernel<
                false, false, false, true, true, float4, false, false, false,
                false, false, false, false, false, false, 4>
                <<<sci_split4_grid, block>>>(
                    sci_numbers, cluster_numbers, d_cluster_offsets,
                    d_cluster_valid_masks, d_cluster_local_masks,
                    d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                    nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags, 1,
                    d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                    d_sorted_lj_comb, cell, d_lj_ab, snapshot.header.cutoff,
                    d_force_regular, snapshot.header.pme_beta, nullptr, nullptr,
                    nullptr, nullptr, nullptr);
            CheckCuda(cudaGetLastError(),
                      "launch production-gmxpacked sorted sci-split4 fast");
        }
        if (safe_sci != snapshot.sci.size())
        {
            SpongeProductionGmxpackedReplayKernel<
                false, false, false, true, false, float4, false, false, false,
                false, false, false, false, false, false, 4>
                <<<sci_split4_grid, block>>>(
                    sci_numbers, cluster_numbers, d_cluster_offsets,
                    d_cluster_valid_masks, d_cluster_local_masks,
                    d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                    d_pair_shift_bits,
                    safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                    d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                    d_sorted_lj_comb, cell, d_lj_ab, snapshot.header.cutoff,
                    d_force_regular, snapshot.header.pme_beta, nullptr, nullptr,
                    nullptr, nullptr, nullptr);
            CheckCuda(cudaGetLastError(),
                      "launch production-gmxpacked sorted sci-split4 slow");
        }
    };
    auto launchSparseSitsCorrection = [&]() {
        if (sparse_sits_sci_numbers == 0)
        {
            return;
        }
        if (sparse_sits_safe_sci != 0)
        {
            SpongeProductionGmxpackedReplayKernel<
                false, false, false, false, true, float4, false, false,
                false, false, false, false, false, false, false,
                sparse_sits_work_parts, false, true, true>
                <<<sparse_sits_grid, block>>>(
                    sparse_sits_sci_numbers, cluster_numbers,
                    d_cluster_offsets, d_cluster_valid_masks,
                    d_cluster_local_masks, d_super_cluster_offsets,
                    d_sparse_sits_sci, d_sparse_sits_cjpacked, d_excl,
                    nullptr,
                    sparse_sits_all_safe
                        ? nullptr
                        : d_sparse_sits_sci_shift_safe_flags,
                    1, d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                    d_sorted_lj_comb, cell, d_lj_ab,
                    snapshot.header.cutoff, d_force_regular,
                    snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                    nullptr, nullptr, false, false, d_force_enhancing,
                    sitsAtomEnd, sitsPwwpFactor);
            CheckCuda(cudaGetLastError(),
                      "launch sparse SITS correction fast");
        }
        if (sparse_sits_safe_sci != sparse_sits_stream.sci.size())
        {
            SpongeProductionGmxpackedReplayKernel<
                false, false, false, false, false, float4, false, false,
                false, false, false, false, false, false, false,
                sparse_sits_work_parts, false, true, true>
                <<<sparse_sits_grid, block>>>(
                    sparse_sits_sci_numbers, cluster_numbers,
                    d_cluster_offsets, d_cluster_valid_masks,
                    d_cluster_local_masks, d_super_cluster_offsets,
                    d_sparse_sits_sci, d_sparse_sits_cjpacked, d_excl,
                    d_sparse_sits_pair_shift_bits,
                    sparse_sits_safe_sci == 0
                        ? nullptr
                        : d_sparse_sits_sci_shift_safe_flags,
                    0, d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                    d_sorted_lj_comb, cell, d_lj_ab,
                    snapshot.header.cutoff, d_force_regular,
                    snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                    nullptr, nullptr, false, false, d_force_enhancing,
                    sitsAtomEnd, sitsPwwpFactor);
            CheckCuda(cudaGetLastError(),
                      "launch sparse SITS correction slow");
        }
    };
    auto launchForceOnlySortedOracleSidecar = [&]() {
        if (safe_sci != 0)
        {
            if (useLjComb)
            {
                SpongeProductionGmxpackedReplayKernel<
                    false, false, false, true, true, float4, false, true,
                    false, false, false, false, false, false, true>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags,
                        1, d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr,
                        reinterpret_cast<float3*>(d_oracle_pass_masks));
            }
            else
            {
                SpongeProductionGmxpackedReplayKernel<
                    false, false, false, true, true, float4, false, false,
                    false, false, false, false, false, false, true>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags,
                        1, d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr,
                        reinterpret_cast<float3*>(d_oracle_pass_masks));
            }
            CheckCuda(
                cudaGetLastError(),
                "launch production-gmxpacked sorted oracle-sidecar fast");
        }
        if (safe_sci != snapshot.sci.size())
        {
            if (useLjComb)
            {
                SpongeProductionGmxpackedReplayKernel<
                    false, false, false, true, false, float4, false, true,
                    false, false, false, false, false, false, true>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        d_pair_shift_bits,
                        safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr,
                        reinterpret_cast<float3*>(d_oracle_pass_masks));
            }
            else
            {
                SpongeProductionGmxpackedReplayKernel<
                    false, false, false, true, false, float4, false, false,
                    false, false, false, false, false, false, true>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        d_pair_shift_bits,
                        safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr,
                        reinterpret_cast<float3*>(d_oracle_pass_masks));
            }
            CheckCuda(
                cudaGetLastError(),
                "launch production-gmxpacked sorted oracle-sidecar slow");
        }
    };
    auto launchForceOnlySortedLocalIMask8 = [&]() {
        if (safe_sci != 0)
        {
            if (useLjComb)
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, true, float4, false,
                                                      true, false, true, false>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags,
                        1, d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            else
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, true, float4, false,
                                                      false, false, true, false>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags,
                        1, d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            CheckCuda(cudaGetLastError(),
                      "launch production-gmxpacked sorted local-i-mask8 fast");
        }
        if (safe_sci != snapshot.sci.size())
        {
            if (useLjComb)
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, false, float4, false,
                                                      true, false, true, false>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        d_pair_shift_bits,
                        safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            else
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, false, float4, false,
                                                      false, false, true, false>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        d_pair_shift_bits,
                        safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            CheckCuda(cudaGetLastError(),
                      "launch production-gmxpacked sorted local-i-mask8 slow");
        }
    };
    auto launchForceOnlySortedActiveIMask8 = [&]() {
        if (safe_sci != 0)
        {
            if (useLjComb)
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, true, float4, false,
                                                      true, false, true, true>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags,
                        1, d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            else
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, true, float4, false,
                                                      false, false, true, true>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags,
                        1, d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            CheckCuda(cudaGetLastError(),
                      "launch production-gmxpacked sorted active-i-mask8 fast");
        }
        if (safe_sci != snapshot.sci.size())
        {
            if (useLjComb)
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, false, float4, false,
                                                      true, false, true, true>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        d_pair_shift_bits,
                        safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            else
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, false, float4, false,
                                                      false, false, true, true>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        d_pair_shift_bits,
                        safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            CheckCuda(cudaGetLastError(),
                      "launch production-gmxpacked sorted active-i-mask8 slow");
        }
    };
    auto launchForceOnlySortedDenseNoExcl = [&]() {
        if (safe_sci != 0)
        {
            if (useLjComb)
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, true, float4, false,
                                                      true, false, false, false,
                                                      true>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags,
                        1, d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            else
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, true, float4, false,
                                                      false, false, false, false,
                                                      true>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags,
                        1, d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            CheckCuda(cudaGetLastError(),
                      "launch production-gmxpacked sorted dense-noexcl fast");
        }
        if (safe_sci != snapshot.sci.size())
        {
            if (useLjComb)
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, false, float4, false,
                                                      true, false, false, false,
                                                      true>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        d_pair_shift_bits,
                        safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            else
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, false, float4, false,
                                                      false, false, false, false,
                                                      true>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        d_pair_shift_bits,
                        safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            CheckCuda(cudaGetLastError(),
                      "launch production-gmxpacked sorted dense-noexcl slow");
        }
    };
    auto launchForceOnlySortedAttribution =
        [&]<bool forceAllI, bool noCutoff>(const char* label)
    {
        if (safe_sci != 0)
        {
            if (useLjComb)
            {
                SpongeProductionGmxpackedReplayKernel<
                    false, false, false, true, true, float4, false, true,
                    false, false, false, false, forceAllI, noCutoff>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags,
                        1, d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            else
            {
                SpongeProductionGmxpackedReplayKernel<
                    false, false, false, true, true, float4, false, false,
                    false, false, false, false, forceAllI, noCutoff>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags,
                        1, d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            CheckCuda(cudaGetLastError(), label);
        }
        if (safe_sci != snapshot.sci.size())
        {
            if (useLjComb)
            {
                SpongeProductionGmxpackedReplayKernel<
                    false, false, false, true, false, float4, false, true,
                    false, false, false, false, forceAllI, noCutoff>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        d_pair_shift_bits,
                        safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            else
            {
                SpongeProductionGmxpackedReplayKernel<
                    false, false, false, true, false, float4, false, false,
                    false, false, false, false, forceAllI, noCutoff>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        d_pair_shift_bits,
                        safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            CheckCuda(cudaGetLastError(), label);
        }
    };
    auto launchForceOnlySortedNoWriteback = [&]() {
        if (safe_sci != 0)
        {
            if (useLjComb)
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, true, float4, false,
                                                      true, true>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags,
                        1, d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            else
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, true, float4, false,
                                                      false, true>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        nullptr, allSciSafe ? nullptr : d_sci_shift_safe_flags,
                        1, d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            CheckCuda(cudaGetLastError(),
                      "launch production-gmxpacked sorted no-atomic fast");
        }
        if (safe_sci != snapshot.sci.size())
        {
            if (useLjComb)
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, false, float4, false,
                                                      true, true>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        d_pair_shift_bits,
                        safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            else
            {
                SpongeProductionGmxpackedReplayKernel<false, false, false,
                                                      true, false, float4, false,
                                                      false, true>
                    <<<grid, block>>>(
                        sci_numbers, cluster_numbers, d_cluster_offsets,
                        d_cluster_valid_masks, d_cluster_local_masks,
                        d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                        d_pair_shift_bits,
                        safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                        d_sorted_lj_comb, cell, d_lj_ab,
                        snapshot.header.cutoff, d_force_regular,
                        snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            }
            CheckCuda(cudaGetLastError(),
                      "launch production-gmxpacked sorted no-atomic slow");
        }
    };
    auto launchForceOnlySafeOnly = [&]() {
        if (useLjComb)
        {
            SpongeProductionGmxpackedReplayKernel<false, false, false, false,
                                                  true, float4>
                <<<grid, block>>>(
                    sci_numbers, cluster_numbers, d_cluster_offsets,
                    d_cluster_valid_masks, d_cluster_local_masks,
                    d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                    nullptr, nullptr, 0, d_sorted_atom_ids, d_sorted_xq,
                    d_sorted_lj_type, d_sorted_lj_comb, cell, d_lj_ab,
                    snapshot.header.cutoff, d_force_regular,
                    snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                    nullptr, nullptr);
        }
        else
        {
            SpongeProductionGmxpackedReplayKernel<false, false, false, false,
                                                  true, float4, false, false>
                <<<grid, block>>>(
                    sci_numbers, cluster_numbers, d_cluster_offsets,
                    d_cluster_valid_masks, d_cluster_local_masks,
                    d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                    nullptr, nullptr, 0, d_sorted_atom_ids, d_sorted_xq,
                    d_sorted_lj_type, d_sorted_lj_comb, cell, d_lj_ab,
                    snapshot.header.cutoff, d_force_regular,
                    snapshot.header.pme_beta, nullptr, nullptr, nullptr,
                    nullptr, nullptr);
        }
        CheckCuda(cudaGetLastError(),
                  "launch production-gmxpacked safe-only");
    };
    auto launchForceOnlySpecialized = [&]() {
        SpongeProductionGmxpackedSpecializedSafeForceKernel<false, false>
            <<<grid, block>>>(
            sci_numbers, cluster_numbers, d_sci, d_cjpacked, d_excl,
            d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_comb, nullptr, cell,
            snapshot.header.cutoff, d_force_regular, snapshot.header.pme_beta);
        CheckCuda(cudaGetLastError(),
                  "launch production-gmxpacked specialized-safe");
    };
    auto launchForceOnlySpecializedSorted = [&]() {
        SpongeProductionGmxpackedSpecializedSafeForceKernel<true, false>
            <<<grid, block>>>(
            sci_numbers, cluster_numbers, d_sci, d_cjpacked, d_excl,
            d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_comb, nullptr, cell,
            snapshot.header.cutoff, d_force_regular, snapshot.header.pme_beta);
        CheckCuda(cudaGetLastError(),
                  "launch production-gmxpacked specialized-sorted");
    };
    auto launchForceOnlySpecializedShiftvec = [&]() {
        SpongeProductionGmxpackedSpecializedSafeForceKernel<false, true>
            <<<grid, block>>>(
            sci_numbers, cluster_numbers, d_sci, d_cjpacked, d_excl,
            d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_comb, d_shiftvec, cell,
            snapshot.header.cutoff, d_force_regular, snapshot.header.pme_beta);
        CheckCuda(cudaGetLastError(),
                  "launch production-gmxpacked specialized-shiftvec");
    };
    auto launchSharedProductionFull = [&]() {
        const auto* shared_sci =
            reinterpret_cast<const LJ_CLUSTERED_GMXPACKED_SCI*>(d_sci);
        const auto* shared_cjpacked =
            reinterpret_cast<const LJ_CLUSTERED_GMXPACKED_CJ*>(d_cjpacked);
        const auto* shared_excl =
            reinterpret_cast<const LJ_CLUSTERED_GMXPACKED_EXCLUSION*>(d_excl);
        VECTOR* shared_force = reinterpret_cast<VECTOR*>(d_force_compact);
        if (safe_sci != 0 && useLjComb)
        {
            Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                true, true, false, true, true, true, false, true>
                <<<grid, block>>>(
                    sci_numbers, kClusterSize, kSuperClusterClusters,
                    cluster_numbers, d_cluster_offsets, d_cluster_valid_masks,
                    d_cluster_local_masks, d_super_cluster_offsets, shared_sci,
                    shared_cjpacked, shared_excl, nullptr,
                    allSciSafe ? nullptr : d_sci_shift_safe_flags, 1,
                    d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                    d_sorted_lj_comb, cell, d_lj_ab, snapshot.header.cutoff,
                    shared_force, snapshot.header.pme_beta, d_atom_energy,
                    d_atom_virial, d_atom_direct_cf_energy, d_atom_lj_energy,
                    computeEnergy, computeVirial);
        }
        else if (safe_sci != 0)
        {
            Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                true, true, false, true, false, true, false, true>
                <<<grid, block>>>(
                    sci_numbers, kClusterSize, kSuperClusterClusters,
                    cluster_numbers, d_cluster_offsets, d_cluster_valid_masks,
                    d_cluster_local_masks, d_super_cluster_offsets, shared_sci,
                    shared_cjpacked, shared_excl, nullptr,
                    allSciSafe ? nullptr : d_sci_shift_safe_flags, 1,
                    d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                    d_sorted_lj_comb, cell, d_lj_ab, snapshot.header.cutoff,
                    shared_force, snapshot.header.pme_beta, d_atom_energy,
                    d_atom_virial, d_atom_direct_cf_energy, d_atom_lj_energy,
                    computeEnergy, computeVirial);
        }
        if (safe_sci != snapshot.sci.size() && useLjComb)
        {
            Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                true, true, false, true, true, true, false, false>
                <<<grid, block>>>(
                    sci_numbers, kClusterSize, kSuperClusterClusters,
                    cluster_numbers, d_cluster_offsets, d_cluster_valid_masks,
                    d_cluster_local_masks, d_super_cluster_offsets, shared_sci,
                    shared_cjpacked, shared_excl, d_pair_shift_bits,
                    safe_sci == 0 ? nullptr : d_sci_shift_safe_flags, 0,
                    d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                    d_sorted_lj_comb, cell, d_lj_ab, snapshot.header.cutoff,
                    shared_force, snapshot.header.pme_beta, d_atom_energy,
                    d_atom_virial, d_atom_direct_cf_energy, d_atom_lj_energy,
                    computeEnergy, computeVirial);
        }
        else if (safe_sci != snapshot.sci.size())
        {
            Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                true, true, false, true, false, true, false, false>
                <<<grid, block>>>(
                    sci_numbers, kClusterSize, kSuperClusterClusters,
                    cluster_numbers, d_cluster_offsets, d_cluster_valid_masks,
                    d_cluster_local_masks, d_super_cluster_offsets, shared_sci,
                    shared_cjpacked, shared_excl, d_pair_shift_bits, nullptr, 0,
                    d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type,
                    d_sorted_lj_comb, cell, d_lj_ab, snapshot.header.cutoff,
                    shared_force, snapshot.header.pme_beta, d_atom_energy,
                    d_atom_virial, d_atom_direct_cf_energy, d_atom_lj_energy,
                    computeEnergy, computeVirial);
        }
        CheckCuda(cudaGetLastError(),
                  "launch shared production-gmxpacked full");
    };
    auto launchEnergyVirialKernel =
        [&]<bool needEnergy, bool needVirial, bool totalOutputKernel,
            bool sciShiftOnly, bool virialFromShift, bool useLjCombKernel,
            int sciWorkParts = 1>(
            const uint64_t* pairShiftBits, const int* sciFlags,
            int sciSafeValue)
    {
        SpongeProductionGmxpackedReplayKernel<
            needEnergy, needVirial, totalOutputKernel, true, sciShiftOnly,
            float3, virialFromShift, useLjCombKernel, false, false, false,
            false, false, false, false, sciWorkParts, false>
            <<<(sciWorkParts == 8
                    ? sci_split8_grid
                    : (sciWorkParts == 4
                           ? sci_split4_grid
                           : (sciWorkParts == 2 ? sci_split2_grid : grid))),
                block>>>(
                sci_numbers, cluster_numbers, d_cluster_offsets,
                d_cluster_valid_masks, d_cluster_local_masks,
                d_super_cluster_offsets, d_sci, d_cjpacked, d_excl,
                pairShiftBits, sciFlags, sciSafeValue, d_sorted_atom_ids,
                d_sorted_xq, d_sorted_lj_type, d_sorted_lj_comb, cell,
                d_lj_ab, snapshot.header.cutoff, d_force_compact,
                snapshot.header.pme_beta, d_atom_energy, d_atom_virial,
                d_atom_direct_cf_energy, d_atom_lj_energy,
                virialFromShift ? d_shift_force : nullptr, computeEnergy,
                computeVirial);
    };
    auto launchEnergyVirialOneClass = [&](bool safeClass)
    {
        const uint64_t* pairShiftBits = safeClass ? nullptr : d_pair_shift_bits;
        const int* sciFlags =
            safeClass
                ? (allSciSafe ? nullptr : d_sci_shift_safe_flags)
                : (safe_sci == 0 ? nullptr : d_sci_shift_safe_flags);
        const int sciSafeValue = safeClass ? 1 : 0;
        if (shiftVirialMode)
        {
            if (computeEnergy && computeVirial)
            {
                launchEnergyVirialKernel
                    .template operator()<true, true, true, true, true, true>(
                        nullptr, nullptr, 1);
            }
            else
            {
                launchEnergyVirialKernel
                    .template operator()<false, true, true, true, true, true>(
                        nullptr, nullptr, 1);
            }
            return;
        }
        if (safeClass)
        {
            if (useLjComb)
            {
                launchEnergyVirialKernel
                    .template operator()<true, true, false, true, false, true>(
                        pairShiftBits, sciFlags, sciSafeValue);
            }
            else
            {
                if (replayMode ==
                    ProductionGmxpackedReplayMode::fullOutputSciSplit8)
                {
                    launchEnergyVirialKernel
                        .template operator()<true, true, false, true, false,
                                               false, 8>(
                            pairShiftBits, sciFlags, sciSafeValue);
                }
                else if (replayMode ==
                         ProductionGmxpackedReplayMode::fullOutputSciSplit4)
                {
                    launchEnergyVirialKernel
                        .template operator()<true, true, false, true, false,
                                               false, 4>(
                            pairShiftBits, sciFlags, sciSafeValue);
                }
                else
                {
                    launchEnergyVirialKernel
                        .template operator()<true, true, false, true, false,
                                               false, 2>(
                            pairShiftBits, sciFlags, sciSafeValue);
                }
            }
        }
        else
        {
            if (useLjComb)
            {
                launchEnergyVirialKernel
                    .template operator()<true, true, false, false, false, true>(
                        pairShiftBits, sciFlags, sciSafeValue);
            }
            else
            {
                if (replayMode ==
                    ProductionGmxpackedReplayMode::fullOutputSciSplit8)
                {
                    launchEnergyVirialKernel
                        .template operator()<true, true, false, false, false,
                                               false, 8>(
                            pairShiftBits, sciFlags, sciSafeValue);
                }
                else if (replayMode ==
                         ProductionGmxpackedReplayMode::fullOutputSciSplit4)
                {
                    launchEnergyVirialKernel
                        .template operator()<true, true, false, false, false,
                                               false, 4>(
                            pairShiftBits, sciFlags, sciSafeValue);
                }
                else
                {
                    launchEnergyVirialKernel
                        .template operator()<true, true, false, false, false,
                                               false, 2>(
                            pairShiftBits, sciFlags, sciSafeValue);
                }
            }
        }
    };
    auto launchEnergyVirial = [&]() {
        if (replayMode == ProductionGmxpackedReplayMode::split)
        {
            launchSharedProductionFull();
            return;
        }
        if (safe_sci != 0)
        {
            launchEnergyVirialOneClass(true);
        }
        if (safe_sci != snapshot.sci.size())
        {
            launchEnergyVirialOneClass(false);
        }
        CheckCuda(cudaGetLastError(),
                  "launch production-gmxpacked energy/virial");
    };
    auto launchKernel = [&]() {
        if (computeEnergy || computeVirial)
        {
            launchEnergyVirial();
        }
        else
        {
            if (sparseSitsForceOnly)
            {
                launchForceOnlySortedSciSplit4();
                launchSparseSitsCorrection();
            }
            else if (fusedSitsForceOnly)
            {
                launchForceOnlyFusedSits();
            }
            else if (replayMode ==
                     ProductionGmxpackedReplayMode::specializedSafe)
            {
                launchForceOnlySpecialized();
            }
            else if (replayMode ==
                     ProductionGmxpackedReplayMode::specializedSortedForce)
            {
                launchForceOnlySpecializedSorted();
            }
            else if (replayMode ==
                     ProductionGmxpackedReplayMode::specializedShiftvec)
            {
                launchForceOnlySpecializedShiftvec();
            }
            else if (replayMode == ProductionGmxpackedReplayMode::compactForce)
            {
                launchForceOnlyCompact();
            }
            else if (replayMode == ProductionGmxpackedReplayMode::sortedForce)
            {
                launchForceOnlySorted();
            }
            else if (replayMode ==
                     ProductionGmxpackedReplayMode::sortedForceSciSplit2)
            {
                launchForceOnlySortedSciSplit2();
            }
            else if (replayMode ==
                     ProductionGmxpackedReplayMode::sortedForceSciSplit3)
            {
                launchForceOnlySortedSciSplit3();
            }
            else if (replayMode ==
                     ProductionGmxpackedReplayMode::sortedForceSciSplit4)
            {
                launchForceOnlySortedSciSplit4();
            }
            else if (replayMode ==
                     ProductionGmxpackedReplayMode::sortedForceLocalIMask8)
            {
                launchForceOnlySortedLocalIMask8();
            }
            else if (replayMode ==
                     ProductionGmxpackedReplayMode::sortedForceActiveIMask8)
            {
                launchForceOnlySortedActiveIMask8();
            }
            else if (replayMode ==
                     ProductionGmxpackedReplayMode::sortedForceOracleImask)
            {
                launchForceOnlySorted();
            }
            else if (replayMode ==
                     ProductionGmxpackedReplayMode::sortedForceOracleSidecar)
            {
                launchForceOnlySortedOracleSidecar();
            }
            else if (replayMode ==
                     ProductionGmxpackedReplayMode::sortedForceDeviceSidecar)
            {
                launchDeviceCutoffSidecar();
                launchForceOnlySortedOracleSidecar();
            }
            else if (replayMode ==
                     ProductionGmxpackedReplayMode::sortedForceDenseNoExcl)
            {
                launchForceOnlySortedDenseNoExcl();
            }
            else if (replayMode ==
                     ProductionGmxpackedReplayMode::sortedForceAttrAllI)
            {
                launchForceOnlySortedAttribution
                    .template operator()<true, false>(
                        "launch production-gmxpacked sorted attr-all-i");
            }
            else if (replayMode ==
                     ProductionGmxpackedReplayMode::sortedForceAttrNoCutoff)
            {
                launchForceOnlySortedAttribution
                    .template operator()<false, true>(
                        "launch production-gmxpacked sorted attr-no-cutoff");
            }
            else if (replayMode ==
                     ProductionGmxpackedReplayMode::sortedForceAttrAllINoCutoff)
            {
                launchForceOnlySortedAttribution
                    .template operator()<true, true>(
                        "launch production-gmxpacked sorted attr-all-i-no-cutoff");
            }
            else if (replayMode ==
                     ProductionGmxpackedReplayMode::sortedForceNoWriteback)
            {
                launchForceOnlySortedNoWriteback();
            }
            else if (replayMode == ProductionGmxpackedReplayMode::safeOnly)
            {
                launchForceOnlySafeOnly();
            }
            else
            {
                launchForceOnlySplit();
            }
        }
    };

    clearOutputs();
    for (int i = 0; i < warmup; ++i)
    {
        launchKernel();
    }
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");
    clearOutputs();

    cudaEvent_t start, stop;
    CheckCuda(cudaEventCreate(&start), "cudaEventCreate");
    CheckCuda(cudaEventCreate(&stop), "cudaEventCreate");
    CheckCuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < iters; ++i)
    {
        launchKernel();
    }
    CheckCuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float total_ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&total_ms, start, stop),
              "cudaEventElapsedTime");
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(timed)");

    // Timed iterations intentionally accumulate into the same output buffers
    // so memset cost is not charged to the kernel. Re-run once from zero for
    // all correctness checks and reported observables.
    clearOutputs();
    launchKernel();
    CheckCuda(cudaDeviceSynchronize(),
              "cudaDeviceSynchronize(post_timing_validation)");

    DiffStats full_output_force_stats = {};
    DiffStats full_output_energy_stats = {};
    DiffStats full_output_direct_energy_stats = {};
    DiffStats full_output_lj_energy_stats = {};
    DiffStats full_output_virial_stats = {};
    bool full_output_reference_ok = true;
    if (fullOutputReference != nullptr)
    {
        std::vector<float3> sorted_force(
            static_cast<size_t>(total_atom_numbers));
        CheckCuda(cudaMemcpy(sorted_force.data(), d_force_compact,
                             sizeof(float3) * sorted_force.size(),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(full_output_force)");
        std::vector<Float4POD> replay_force(
            static_cast<size_t>(total_atom_numbers), Float4POD{});
        for (int sorted_i = 0; sorted_i < total_atom_numbers; ++sorted_i)
        {
            const int atom_i =
                snapshot.sorted_atom_ids[static_cast<size_t>(sorted_i)];
            if (atom_i < 0 || atom_i >= total_atom_numbers)
            {
                continue;
            }
            Float4POD& force = replay_force[static_cast<size_t>(atom_i)];
            force.x += sorted_force[static_cast<size_t>(sorted_i)].x;
            force.y += sorted_force[static_cast<size_t>(sorted_i)].y;
            force.z += sorted_force[static_cast<size_t>(sorted_i)].z;
        }
        full_output_force_stats = CompareForceArrays(
            replay_force, fullOutputReference->reference_force);

        if (computeEnergy)
        {
            std::vector<float> atom_energy(
                static_cast<size_t>(scalar_output_numbers));
            std::vector<float> direct_energy(
                static_cast<size_t>(scalar_output_numbers));
            std::vector<float> lj_energy(
                static_cast<size_t>(scalar_output_numbers));
            CheckCuda(cudaMemcpy(atom_energy.data(), d_atom_energy,
                                 sizeof(float) * atom_energy.size(),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(full_output_atom_energy)");
            CheckCuda(cudaMemcpy(direct_energy.data(),
                                 d_atom_direct_cf_energy,
                                 sizeof(float) * direct_energy.size(),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(full_output_direct_energy)");
            CheckCuda(cudaMemcpy(lj_energy.data(), d_atom_lj_energy,
                                 sizeof(float) * lj_energy.size(),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(full_output_lj_energy)");
            full_output_energy_stats = CompareFloatArrays(
                atom_energy, fullOutputReference->reference_atom_energy);
            full_output_direct_energy_stats = CompareFloatArrays(
                direct_energy,
                fullOutputReference->reference_direct_cf_energy);
            full_output_lj_energy_stats = CompareFloatArrays(
                lj_energy, fullOutputReference->reference_lj_energy);
        }
        if (computeVirial)
        {
            std::vector<LTMatrix3> atom_virial(
                static_cast<size_t>(scalar_output_numbers));
            CheckCuda(cudaMemcpy(atom_virial.data(), d_atom_virial,
                                 sizeof(LTMatrix3) * atom_virial.size(),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(full_output_atom_virial)");
            full_output_virial_stats = CompareVirialArrays(
                atom_virial,
                fullOutputReference->reference_atom_virial);
        }

        constexpr double reference_tolerance = 2.0e-5;
        full_output_reference_ok =
            std::isfinite(full_output_force_stats.max_scaled) &&
            full_output_force_stats.max_scaled <= reference_tolerance &&
            (!computeEnergy ||
             (std::isfinite(full_output_energy_stats.max_scaled) &&
              std::isfinite(full_output_direct_energy_stats.max_scaled) &&
              std::isfinite(full_output_lj_energy_stats.max_scaled) &&
              full_output_energy_stats.max_scaled <= reference_tolerance &&
              full_output_direct_energy_stats.max_scaled <=
                  reference_tolerance &&
              full_output_lj_energy_stats.max_scaled <=
                  reference_tolerance)) &&
            (!computeVirial ||
             (std::isfinite(full_output_virial_stats.max_scaled) &&
              full_output_virial_stats.max_scaled <= reference_tolerance));
        std::printf(
            "gmxpacked_fulloutput_reference matched=%u tolerance=%.3e "
            "force_max_abs=%.6e force_max_scaled=%.6e force_rms=%.6e "
            "energy_max_abs=%.6e energy_max_scaled=%.6e "
            "direct_energy_max_abs=%.6e direct_energy_max_scaled=%.6e "
            "lj_energy_max_abs=%.6e lj_energy_max_scaled=%.6e "
            "virial_max_abs=%.6e virial_max_scaled=%.6e\n",
            full_output_reference_ok ? 1u : 0u, reference_tolerance,
            full_output_force_stats.max_abs,
            full_output_force_stats.max_scaled,
            full_output_force_stats.rms,
            full_output_energy_stats.max_abs,
            full_output_energy_stats.max_scaled,
            full_output_direct_energy_stats.max_abs,
            full_output_direct_energy_stats.max_scaled,
            full_output_lj_energy_stats.max_abs,
            full_output_lj_energy_stats.max_scaled,
            full_output_virial_stats.max_abs,
            full_output_virial_stats.max_scaled);
    }

    bool fused_sits_validation_ok = true;
    double fused_sits_max_abs = 0.0;
    double fused_sits_max_scaled = 0.0;
    uint64_t fused_sits_unselected_nonzero = 0ull;
    if (sitsForceOnly)
    {
        clearOutputs();
        launchKernel();
        CheckCuda(cudaDeviceSynchronize(),
                  "cudaDeviceSynchronize(fused_sits_validation)");
        std::vector<float4> ordinary_force(
            static_cast<size_t>(total_atom_numbers));
        std::vector<float4> enhancing_force(
            static_cast<size_t>(total_atom_numbers));
        CheckCuda(cudaMemcpy(ordinary_force.data(), d_force_regular,
                             sizeof(float4) * ordinary_force.size(),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(fused_sits_ordinary_force)");
        CheckCuda(cudaMemcpy(enhancing_force.data(), d_force_enhancing,
                             sizeof(float4) * enhancing_force.size(),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(fused_sits_enhancing_force)");
        for (int atom = 0; atom < total_atom_numbers; ++atom)
        {
            const float4 ordinary =
                ordinary_force[static_cast<size_t>(atom)];
            const float4 enhancing =
                enhancing_force[static_cast<size_t>(atom)];
            const float values[3] = {enhancing.x, enhancing.y, enhancing.z};
            for (float value : values)
            {
                fused_sits_validation_ok =
                    fused_sits_validation_ok && std::isfinite(value);
            }
            if (atom >= sitsAtomEnd &&
                (enhancing.x != 0.0f || enhancing.y != 0.0f ||
                 enhancing.z != 0.0f))
            {
                ++fused_sits_unselected_nonzero;
            }
            if (sitsAtomEnd == total_atom_numbers)
            {
                const float ordinary_values[3] = {
                    ordinary.x, ordinary.y, ordinary.z};
                for (int component = 0; component < 3; ++component)
                {
                    const double abs_diff = std::abs(
                        static_cast<double>(values[component]) -
                        static_cast<double>(ordinary_values[component]));
                    const double scale = std::max(
                        1.0,
                        std::max(std::abs(
                                     static_cast<double>(values[component])),
                                 std::abs(static_cast<double>(
                                     ordinary_values[component]))));
                    fused_sits_max_abs =
                        std::max(fused_sits_max_abs, abs_diff);
                    fused_sits_max_scaled =
                        std::max(fused_sits_max_scaled, abs_diff / scale);
                }
            }
        }
        constexpr double fused_sits_tolerance = 2.0e-5;
        if (sparseSitsForceOnly)
        {
            clearOutputs();
            launchForceOnlyFusedSits();
            CheckCuda(cudaDeviceSynchronize(),
                      "cudaDeviceSynchronize(sparse_sits_reference)");
            std::vector<float4> fused_reference(
                static_cast<size_t>(total_atom_numbers));
            CheckCuda(cudaMemcpy(fused_reference.data(), d_force_enhancing,
                                 sizeof(float4) * fused_reference.size(),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(sparse_sits_fused_reference)");
            for (int atom = 0; atom < total_atom_numbers; ++atom)
            {
                const float* candidate = reinterpret_cast<const float*>(
                    &enhancing_force[static_cast<size_t>(atom)]);
                const float* reference = reinterpret_cast<const float*>(
                    &fused_reference[static_cast<size_t>(atom)]);
                for (int component = 0; component < 3; ++component)
                {
                    const double abs_diff = std::abs(
                        static_cast<double>(candidate[component]) -
                        static_cast<double>(reference[component]));
                    const double scale = std::max(
                        1.0, std::max(std::abs(static_cast<double>(
                                                  candidate[component])),
                                      std::abs(static_cast<double>(
                                          reference[component]))));
                    fused_sits_max_abs =
                        std::max(fused_sits_max_abs, abs_diff);
                    fused_sits_max_scaled =
                        std::max(fused_sits_max_scaled, abs_diff / scale);
                }
            }
            clearOutputs();
            launchKernel();
            CheckCuda(cudaDeviceSynchronize(),
                      "cudaDeviceSynchronize(sparse_sits_restore)");
        }
        fused_sits_validation_ok =
            fused_sits_validation_ok &&
            fused_sits_unselected_nonzero == 0ull &&
            fused_sits_max_scaled <= fused_sits_tolerance;
        std::printf(
            "%s_sits_force_reference matched=%u all_selected=%u "
            "tolerance=%.3e max_abs=%.6e max_scaled=%.6e "
            "unselected_nonzero=%llu selected_end=%d pwwp_factor=%.6g\n",
            sparseSitsForceOnly ? "sparse" : "fused",
            fused_sits_validation_ok ? 1u : 0u,
            sitsAtomEnd == total_atom_numbers ? 1u : 0u,
            fused_sits_tolerance, fused_sits_max_abs,
            fused_sits_max_scaled,
            static_cast<unsigned long long>(
                fused_sits_unselected_nonzero),
            sitsAtomEnd, sitsPwwpFactor);
    }

    bool device_validation_ok = true;
    if (use_device_sidecar)
    {
        std::vector<uint64_t> device_pass_masks(oracle_pass_masks.size());
        CheckCuda(cudaMemcpy(device_pass_masks.data(), d_oracle_pass_masks,
                             sizeof(uint64_t) * device_pass_masks.size(),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(device_cutoff_pass_masks)");
        uint64_t mask_mismatches = 0ull;
        uint64_t mask_missing_bits = 0ull;
        uint64_t mask_extra_bits = 0ull;
        size_t first_mask_mismatch = device_pass_masks.size();
        for (size_t i = 0; i < device_pass_masks.size(); ++i)
        {
            if (device_pass_masks[i] != oracle_pass_masks[i])
            {
                ++mask_mismatches;
                mask_missing_bits += static_cast<uint64_t>(std::popcount(
                    oracle_pass_masks[i] & ~device_pass_masks[i]));
                mask_extra_bits += static_cast<uint64_t>(std::popcount(
                    device_pass_masks[i] & ~oracle_pass_masks[i]));
                if (first_mask_mismatch == device_pass_masks.size())
                {
                    first_mask_mismatch = i;
                }
            }
        }

        std::vector<float4> classified_force(
            static_cast<size_t>(total_atom_numbers));
        clearOutputs();
        launchDeviceCutoffSidecar();
        launchForceOnlySortedOracleSidecar();
        CheckCuda(cudaDeviceSynchronize(),
                  "cudaDeviceSynchronize(force_validation_classified)");
        CheckCuda(cudaMemcpy(classified_force.data(), d_force_regular,
                             sizeof(float4) * classified_force.size(),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(classified_force)");
        clearOutputs();
        launchForceOnlySorted();
        CheckCuda(cudaDeviceSynchronize(),
                  "cudaDeviceSynchronize(force_validation_baseline)");
        std::vector<float4> baseline_force(
            static_cast<size_t>(total_atom_numbers));
        CheckCuda(cudaMemcpy(baseline_force.data(), d_force_regular,
                             sizeof(float4) * baseline_force.size(),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(baseline_force)");

        uint64_t force_bitwise_mismatches = 0ull;
        uint64_t force_tolerance_mismatches = 0ull;
        float force_max_abs = 0.0f;
        float force_max_scaled = 0.0f;
        for (size_t atom = 0; atom < baseline_force.size(); ++atom)
        {
            const float* actual = reinterpret_cast<const float*>(
                &classified_force[atom]);
            const float* reference = reinterpret_cast<const float*>(
                &baseline_force[atom]);
            for (int component = 0; component < 3; ++component)
            {
                if (actual[component] != reference[component])
                {
                    ++force_bitwise_mismatches;
                }
                const float abs_error =
                    std::fabs(actual[component] - reference[component]);
                const float scaled_error =
                    abs_error / (1.0f + std::fabs(reference[component]));
                force_max_abs = std::max(force_max_abs, abs_error);
                force_max_scaled = std::max(force_max_scaled, scaled_error);
                if (!std::isfinite(actual[component]) ||
                    abs_error >
                        1.0e-5f * (1.0f + std::fabs(reference[component])))
                {
                    ++force_tolerance_mismatches;
                }
            }
        }
        device_validation_ok =
            mask_missing_bits == 0ull && force_tolerance_mismatches == 0ull;
        std::fprintf(
            stderr,
            "[gmxpacked device-sidecar validate] mask_mismatches=%llu "
            "mask_missing_bits=%llu mask_extra_bits=%llu "
            "first_mask_mismatch=%lld force_bitwise_mismatches=%llu "
            "force_tolerance_mismatches=%llu force_max_abs=%.9g "
            "force_max_scaled=%.9g\n",
            static_cast<unsigned long long>(mask_mismatches),
            static_cast<unsigned long long>(mask_missing_bits),
            static_cast<unsigned long long>(mask_extra_bits),
            first_mask_mismatch == device_pass_masks.size()
                ? -1ll
                : static_cast<long long>(first_mask_mismatch),
            static_cast<unsigned long long>(force_bitwise_mismatches),
            static_cast<unsigned long long>(force_tolerance_mismatches),
            force_max_abs, force_max_scaled);
    }
    if (use_sci_split2 || use_sci_split3 || use_sci_split4)
    {
        const int sci_split_parts = use_sci_split4 ? 4 : (use_sci_split3 ? 3 : 2);
        std::vector<float4> split_force(
            static_cast<size_t>(total_atom_numbers));
        clearOutputs();
        if (use_sci_split4)
        {
            launchForceOnlySortedSciSplit4();
        }
        else if (use_sci_split3)
        {
            launchForceOnlySortedSciSplit3();
        }
        else
        {
            launchForceOnlySortedSciSplit2();
        }
        CheckCuda(cudaDeviceSynchronize(),
                  "cudaDeviceSynchronize(force_validation_sci_split2)");
        CheckCuda(cudaMemcpy(split_force.data(), d_force_regular,
                             sizeof(float4) * split_force.size(),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(sci_split2_force)");

        clearOutputs();
        launchForceOnlySorted();
        CheckCuda(cudaDeviceSynchronize(),
                  "cudaDeviceSynchronize(force_validation_baseline)");
        std::vector<float4> baseline_force(
            static_cast<size_t>(total_atom_numbers));
        CheckCuda(cudaMemcpy(baseline_force.data(), d_force_regular,
                             sizeof(float4) * baseline_force.size(),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(sci_split2_baseline_force)");

        uint64_t force_bitwise_mismatches = 0ull;
        uint64_t force_tolerance_mismatches = 0ull;
        float force_max_abs = 0.0f;
        float force_max_scaled = 0.0f;
        for (size_t atom = 0; atom < baseline_force.size(); ++atom)
        {
            const float* actual =
                reinterpret_cast<const float*>(&split_force[atom]);
            const float* reference =
                reinterpret_cast<const float*>(&baseline_force[atom]);
            for (int component = 0; component < 3; ++component)
            {
                if (actual[component] != reference[component])
                {
                    ++force_bitwise_mismatches;
                }
                const float abs_error =
                    std::fabs(actual[component] - reference[component]);
                const float scaled_error =
                    abs_error / (1.0f + std::fabs(reference[component]));
                force_max_abs = std::max(force_max_abs, abs_error);
                force_max_scaled = std::max(force_max_scaled, scaled_error);
                if (!std::isfinite(actual[component]) ||
                    abs_error >
                        1.0e-5f * (1.0f + std::fabs(reference[component])))
                {
                    ++force_tolerance_mismatches;
                }
            }
        }
        device_validation_ok = force_tolerance_mismatches == 0ull;
        std::fprintf(
            stderr,
            "[gmxpacked sci-split%d validate] force_bitwise_mismatches=%llu "
            "force_tolerance_mismatches=%llu force_max_abs=%.9g "
            "force_max_scaled=%.9g\n",
            sci_split_parts,
            static_cast<unsigned long long>(force_bitwise_mismatches),
            static_cast<unsigned long long>(force_tolerance_mismatches),
            force_max_abs, force_max_scaled);
    }

    float host_energy = 0.0f;
    float host_direct_energy = 0.0f;
    float host_lj_energy = 0.0f;
    LTMatrix3 host_virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (computeEnergy)
    {
        std::vector<float> host_atom_energy(
            static_cast<size_t>(scalar_output_numbers));
        std::vector<float> host_atom_direct_energy(
            static_cast<size_t>(scalar_output_numbers));
        std::vector<float> host_atom_lj_energy(
            static_cast<size_t>(scalar_output_numbers));
        CheckCuda(cudaMemcpy(host_atom_energy.data(), d_atom_energy,
                             sizeof(float) * scalar_output_numbers,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(atom_energy)");
        CheckCuda(cudaMemcpy(host_atom_direct_energy.data(),
                             d_atom_direct_cf_energy,
                             sizeof(float) * scalar_output_numbers,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(atom_direct_cf_energy)");
        CheckCuda(cudaMemcpy(host_atom_lj_energy.data(), d_atom_lj_energy,
                             sizeof(float) * scalar_output_numbers,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(atom_lj_energy)");
        for (int i = 0; i < scalar_output_numbers; ++i)
        {
            const size_t atom = static_cast<size_t>(i);
            host_energy += host_atom_energy[atom];
            host_direct_energy += host_atom_direct_energy[atom];
            host_lj_energy += host_atom_lj_energy[atom];
        }
    }
    if (computeVirial && shiftVirialMode)
    {
        std::vector<float3> host_shift_force(shiftvec.size());
        std::vector<float3> host_force_compact(
            static_cast<size_t>(total_atom_numbers));
        CheckCuda(cudaMemcpy(host_shift_force.data(), d_shift_force,
                             sizeof(float3) * shiftvec.size(),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(shift_force)");
        CheckCuda(cudaMemcpy(host_force_compact.data(), d_force_compact,
                             sizeof(float3) * total_atom_numbers,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(force_compact)");
        for (size_t i = 0; i < host_shift_force.size(); ++i)
        {
            AccumulateVirialFromForceDis(
                &host_virial,
                Vec3{host_shift_force[i].x, host_shift_force[i].y,
                     host_shift_force[i].z},
                Vec3{shiftvec[i].x, shiftvec[i].y, shiftvec[i].z}, 1.0f);
        }
        for (int i = 0; i < total_atom_numbers; ++i)
        {
            const size_t atom = static_cast<size_t>(i);
            AccumulateVirialFromForceDis(
                &host_virial,
                Vec3{host_force_compact[atom].x,
                     host_force_compact[atom].y,
                     host_force_compact[atom].z},
                Vec3{sorted_xq[atom].x, sorted_xq[atom].y,
                     sorted_xq[atom].z},
                1.0f);
        }
    }
    else if (computeVirial)
    {
        std::vector<LTMatrix3> host_atom_virial(
            static_cast<size_t>(scalar_output_numbers));
        CheckCuda(cudaMemcpy(host_atom_virial.data(), d_atom_virial,
                             sizeof(LTMatrix3) * scalar_output_numbers,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(atom_virial)");
        for (int i = 0; i < scalar_output_numbers; ++i)
        {
            const LTMatrix3& atom_virial =
                host_atom_virial[static_cast<size_t>(i)];
            host_virial.a11 += atom_virial.a11;
            host_virial.a21 += atom_virial.a21;
            host_virial.a22 += atom_virial.a22;
            host_virial.a31 += atom_virial.a31;
            host_virial.a32 += atom_virial.a32;
            host_virial.a33 += atom_virial.a33;
        }
    }
    const bool sane =
        device_validation_ok && full_output_reference_ok &&
        fused_sits_validation_ok &&
        (!computeEnergy || (std::isfinite(host_energy) &&
                            std::isfinite(host_direct_energy) &&
                            std::isfinite(host_lj_energy))) &&
        (!computeVirial || (std::isfinite(host_virial.a11) &&
                            std::isfinite(host_virial.a22) &&
                            std::isfinite(host_virial.a33)));
    const bool full_output_mode = computeEnergy || computeVirial;
    int reported_sci_work_parts = 1;
    bool reported_contiguous_sci_work = false;
    if (full_output_mode && !useLjComb && !shiftVirialMode &&
        replayMode != ProductionGmxpackedReplayMode::split)
    {
        reported_sci_work_parts =
            replayMode == ProductionGmxpackedReplayMode::fullOutputSciSplit8
                ? 8
                : (replayMode ==
                           ProductionGmxpackedReplayMode::fullOutputSciSplit4
                       ? 4
                       : 2);
    }
    else if (!full_output_mode &&
             replayMode ==
                 ProductionGmxpackedReplayMode::sortedForceSciSplit2)
    {
        reported_sci_work_parts = 2;
    }
    else if (!full_output_mode &&
             replayMode ==
                 ProductionGmxpackedReplayMode::sortedForceSciSplit3)
    {
        reported_sci_work_parts = 3;
        reported_contiguous_sci_work = true;
    }
    else if (!full_output_mode &&
             replayMode ==
                 ProductionGmxpackedReplayMode::sortedForceSciSplit4)
    {
        reported_sci_work_parts = 4;
    }
    else if (!full_output_mode && fusedSitsForceOnly && !useLjComb)
    {
        reported_sci_work_parts = 4;
    }
    else if (!full_output_mode && sparseSitsForceOnly)
    {
        reported_sci_work_parts = sparse_sits_work_parts;
    }
    if (sparseSitsForceOnly)
    {
        const double selected_tile_percent =
            sparse_sits_stream.source_tile_bits == 0
                ? 0.0
                : 100.0 *
                      static_cast<double>(
                          sparse_sits_stream.selected_tile_bits) /
                      static_cast<double>(
                          sparse_sits_stream.source_tile_bits);
        std::printf(
            "sparse_sits_stream sci=%zu/%zu cjpacked=%zu/%zu "
            "tile_bits=%llu/%llu tile_pct=%.3f work_parts=%d\n",
            sparse_sits_stream.sci.size(), snapshot.sci.size(),
            sparse_sits_stream.cjpacked.size(), snapshot.cjpacked.size(),
            static_cast<unsigned long long>(
                sparse_sits_stream.selected_tile_bits),
            static_cast<unsigned long long>(
                sparse_sits_stream.source_tile_bits),
            selected_tile_percent, sparse_sits_work_parts);
    }
    std::printf(
        "kernel=sponge_production_gmxpacked snapshot=%s avg_ms=%.6f iters=%d "
        "variant=%s implementation=%s output_mode=%s lj_mode=%s "
        "sci_work_parts=%d "
        "contiguous_sci_work=%u launches_per_iter=%d sci=%d cjpacked=%zu "
        "excl=%zu atoms=%d safe_sci=%zu unsafe_sci=%zu compute_energy=%u "
        "compute_virial=%u total_output=%u sanity=%s energy=%.6e "
        "direct_energy=%.6e "
        "lj_energy=%.6e virial_xx=%.6e virial_yy=%.6e virial_zz=%.6e\n",
        snapshotLabel, total_ms / static_cast<float>(iters), iters,
        ProductionGmxpackedReplayModeName(replayMode),
        replayMode == ProductionGmxpackedReplayMode::split
            ? "shared-production"
            : "microbench-experimental",
        full_output_mode ? "full" : "force-only",
        useLjComb ? "comb" : "packed-ab",
        reported_sci_work_parts,
        reported_contiguous_sci_work ? 1u : 0u,
        replayMode == ProductionGmxpackedReplayMode::split
            ? 1
            : (sparseSitsForceOnly
                   ? 2
                   : ((replayMode == ProductionGmxpackedReplayMode::fusedSitsForceOnly ||
         replayMode == ProductionGmxpackedReplayMode::compactForce ||
         replayMode == ProductionGmxpackedReplayMode::sortedForce ||
         replayMode == ProductionGmxpackedReplayMode::sortedForceSciSplit2 ||
         replayMode == ProductionGmxpackedReplayMode::sortedForceSciSplit3 ||
         replayMode == ProductionGmxpackedReplayMode::sortedForceSciSplit4 ||
         replayMode == ProductionGmxpackedReplayMode::sortedForceLocalIMask8 ||
         replayMode == ProductionGmxpackedReplayMode::sortedForceActiveIMask8 ||
         replayMode == ProductionGmxpackedReplayMode::sortedForceOracleImask ||
         replayMode == ProductionGmxpackedReplayMode::sortedForceOracleSidecar ||
         replayMode == ProductionGmxpackedReplayMode::sortedForceDeviceSidecar ||
         replayMode == ProductionGmxpackedReplayMode::sortedForceDenseNoExcl ||
         replayMode == ProductionGmxpackedReplayMode::sortedForceAttrAllI ||
         replayMode == ProductionGmxpackedReplayMode::sortedForceAttrNoCutoff ||
         replayMode == ProductionGmxpackedReplayMode::sortedForceAttrAllINoCutoff ||
         replayMode == ProductionGmxpackedReplayMode::sortedForceNoWriteback)
            ? static_cast<int>(
                  ((safe_sci != 0 ? 1 : 0) +
                   (safe_sci != snapshot.sci.size() ? 1 : 0)) *
                  (replayMode ==
                           ProductionGmxpackedReplayMode::sortedForceDeviceSidecar
                       ? 2
                       : 1))
            : 1)),
        sci_numbers, snapshot.cjpacked.size(),
        snapshot.excl.size(), total_atom_numbers, safe_sci,
        snapshot.sci.size() - safe_sci, computeEnergy ? 1u : 0u,
        computeVirial ? 1u : 0u, totalOutput ? 1u : 0u,
        sane ? "ok" : "bad", host_energy, host_direct_energy, host_lj_energy,
        host_virial.a11, host_virial.a22, host_virial.a33);

    CheckCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    CheckCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    CheckCuda(cudaFree(d_atom_virial), "cudaFree(atom_virial)");
    CheckCuda(cudaFree(d_atom_lj_energy), "cudaFree(atom_lj_energy)");
    CheckCuda(cudaFree(d_atom_direct_cf_energy),
              "cudaFree(atom_direct_cf_energy)");
    CheckCuda(cudaFree(d_atom_energy), "cudaFree(atom_energy)");
    CheckCuda(cudaFree(d_shift_force), "cudaFree(shift_force)");
    CheckCuda(cudaFree(d_force_compact), "cudaFree(force_compact)");
    CheckCuda(cudaFree(d_force_regular), "cudaFree(force_regular)");
    CheckCuda(cudaFree(d_force_enhancing), "cudaFree(force_enhancing)");
    CheckCuda(cudaFree(d_force_shared_production),
              "cudaFree(force_shared_production)");
    CheckCuda(cudaFree(d_shiftvec), "cudaFree(shiftvec)");
    CheckCuda(cudaFree(d_lj_ab), "cudaFree(lj_ab)");
    CheckCuda(cudaFree(d_sorted_lj_comb), "cudaFree(sorted_lj_comb)");
    CheckCuda(cudaFree(d_sorted_lj_type), "cudaFree(sorted_lj_type)");
    CheckCuda(cudaFree(d_sorted_xq), "cudaFree(sorted_xq)");
    CheckCuda(cudaFree(d_sorted_atom_ids), "cudaFree(sorted_atom_ids)");
    CheckCuda(cudaFree(d_sparse_sits_sci_shift_safe_flags),
              "cudaFree(sparse_sits_sci_shift_safe_flags)");
    CheckCuda(cudaFree(d_sparse_sits_pair_shift_bits),
              "cudaFree(sparse_sits_pair_shift_bits)");
    CheckCuda(cudaFree(d_sparse_sits_cjpacked),
              "cudaFree(sparse_sits_cjpacked)");
    CheckCuda(cudaFree(d_sparse_sits_sci), "cudaFree(sparse_sits_sci)");
    CheckCuda(cudaFree(d_sci_shift_safe_flags),
              "cudaFree(sci_shift_safe_flags)");
    CheckCuda(cudaFree(d_pair_shift_bits), "cudaFree(pair_shift_bits)");
    CheckCuda(cudaFree(d_excl), "cudaFree(excl)");
    CheckCuda(cudaFree(d_oracle_pass_masks), "cudaFree(oracle_pass_masks)");
    CheckCuda(cudaFree(d_cjpacked), "cudaFree(cjpacked)");
    CheckCuda(cudaFree(d_sci), "cudaFree(sci)");
    CheckCuda(cudaFree(d_super_cluster_offsets),
              "cudaFree(super_cluster_offsets)");
    CheckCuda(cudaFree(d_cluster_local_masks), "cudaFree(cluster_local_masks)");
    CheckCuda(cudaFree(d_cluster_valid_masks), "cudaFree(cluster_valid_masks)");
    CheckCuda(cudaFree(d_cluster_offsets), "cudaFree(cluster_offsets)");
    if (fullOutputReference != nullptr && !full_output_reference_ok)
    {
        std::exit(2);
    }
}

std::vector<Vec3> MakeVec3Vector(const std::vector<Float4POD>& values)
{
    std::vector<Vec3> converted(values.size());
    for (size_t i = 0; i < values.size(); ++i)
    {
        converted[i] = {values[i].x, values[i].y, values[i].z};
    }
    return converted;
}

std::vector<::VECTOR> MakeProbeVector(
    const std::vector<Float4POD>& values)
{
    std::vector<::VECTOR> converted(values.size());
    for (size_t i = 0; i < values.size(); ++i)
    {
        converted[i] = {values[i].x, values[i].y, values[i].z};
    }
    return converted;
}

::LTMatrix3 MakeProbeMatrix(const LTMatrix3POD& value)
{
    return {value.a11, value.a21, value.a22,
            value.a31, value.a32, value.a33};
}

::LTMatrix3 MakeProbeMatrix(const LTMatrix3& value)
{
    return {value.a11, value.a21, value.a22,
            value.a31, value.a32, value.a33};
}

std::vector<LJ_CLUSTERED_GMXPACKED_SCI> MakeProductionGmxpackedSciVector(
    const std::vector<SpongeGmxpackedSciPOD>& values)
{
    std::vector<LJ_CLUSTERED_GMXPACKED_SCI> converted(values.size());
    for (size_t i = 0; i < values.size(); ++i)
    {
        converted[i].supercluster_id = values[i].supercluster_id;
        converted[i].shift_id = values[i].shift_id;
        converted[i].cjpacked_begin = values[i].cjpacked_begin;
        converted[i].cjpacked_end = values[i].cjpacked_end;
    }
    return converted;
}

std::vector<LJ_CLUSTERED_GMXPACKED_CJ> MakeProductionGmxpackedCjVector(
    const std::vector<SpongeGmxpackedCjPOD>& values)
{
    std::vector<LJ_CLUSTERED_GMXPACKED_CJ> converted(values.size());
    for (size_t i = 0; i < values.size(); ++i)
    {
        std::memcpy(converted[i].cj, values[i].cj, sizeof(converted[i].cj));
        for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
        {
            converted[i].split[split].imask = values[i].split[split].imask;
            converted[i].split[split].exclusion_index =
                values[i].split[split].exclusion_index;
        }
    }
    return converted;
}

std::vector<LJ_CLUSTERED_GMXPACKED_EXCLUSION>
MakeProductionGmxpackedExclusionVector(
    const std::vector<SpongeGmxpackedExclusionPOD>& values)
{
    std::vector<LJ_CLUSTERED_GMXPACKED_EXCLUSION> converted(values.size());
    for (size_t i = 0; i < values.size(); ++i)
    {
        std::memcpy(converted[i].pair, values[i].pair, sizeof(converted[i].pair));
    }
    return converted;
}

std::vector<::VECTOR> MakeRefreshFractionalCenters(
    const std::vector<::VECTOR>& centers, const ::LTMatrix3& rcell)
{
    std::vector<::VECTOR> fractional(centers.size());
    for (size_t i = 0; i < centers.size(); ++i)
    {
        ::VECTOR value = centers[i] * rcell;
        value.x -= std::floor(value.x);
        value.y -= std::floor(value.y);
        value.z -= std::floor(value.z);
        fractional[i] = value;
    }
    return fractional;
}

std::vector<::VECTOR> MakeRefreshFractionalExtents(
    const std::vector<::VECTOR>& extents, const ::LTMatrix3& rcell)
{
    std::vector<::VECTOR> fractional(extents.size());
    for (size_t i = 0; i < extents.size(); ++i)
    {
        const ::VECTOR value = extents[i];
        fractional[i] = {
            std::fabs(value.x * rcell.a11) +
                std::fabs(value.y * rcell.a21) +
                std::fabs(value.z * rcell.a31),
            std::fabs(value.y * rcell.a22) +
                std::fabs(value.z * rcell.a32),
            std::fabs(value.z * rcell.a33)};
    }
    return fractional;
}

void RunSpongeProductionGmxpackedRefresh(
    const SpongeGmxpackedForceOnlySnapshot& snapshot, int warmup, int iters,
    const char* snapshotLabel, int refreshBlockSize)
{
    if (snapshot.cluster_centers.empty() ||
        snapshot.cluster_extents.size() != snapshot.cluster_centers.size())
    {
        std::fprintf(stderr,
                     "production-gmxpacked-refresh requires a snapshot with "
                     "builder metadata footer\n");
        std::exit(1);
    }
    const int sci_numbers = static_cast<int>(snapshot.sci.size());
    std::vector<::VECTOR> cluster_centers =
        MakeProbeVector(snapshot.cluster_centers);
    std::vector<::VECTOR> cluster_extents =
        MakeProbeVector(snapshot.cluster_extents);
    const ::LTMatrix3 rcell =
        MakeProbeMatrix(InvertCellMatrix(MakeMatrix(snapshot.header.cell)));
    std::vector<::VECTOR> cluster_fractional_centers =
        MakeRefreshFractionalCenters(cluster_centers, rcell);
    std::vector<::VECTOR> cluster_fractional_extents =
        MakeRefreshFractionalExtents(cluster_extents, rcell);
    std::vector<LJ_CLUSTERED_GMXPACKED_SCI> gmxpacked_sci =
        MakeProductionGmxpackedSciVector(snapshot.sci);
    std::vector<LJ_CLUSTERED_GMXPACKED_CJ> gmxpacked_cjpacked =
        MakeProductionGmxpackedCjVector(snapshot.cjpacked);
    std::vector<LJ_CLUSTERED_GMXPACKED_EXCLUSION> gmxpacked_excl =
        MakeProductionGmxpackedExclusionVector(snapshot.excl);
    int* d_super_cluster_offsets =
        CopyVectorToDevice(snapshot.super_cluster_offsets);
    ::VECTOR* d_cluster_fractional_centers =
        CopyVectorToDevice(cluster_fractional_centers);
    ::VECTOR* d_cluster_fractional_extents =
        CopyVectorToDevice(cluster_fractional_extents);
    unsigned int* d_cluster_valid_masks =
        CopyVectorToDevice(snapshot.cluster_valid_masks);
    unsigned int* d_cluster_local_masks =
        CopyVectorToDevice(snapshot.cluster_local_masks);
    LJ_CLUSTERED_GMXPACKED_SCI* d_sci = CopyVectorToDevice(gmxpacked_sci);
    LJ_CLUSTERED_GMXPACKED_CJ* d_cjpacked =
        CopyVectorToDevice(gmxpacked_cjpacked);
    LJ_CLUSTERED_GMXPACKED_EXCLUSION* d_excl =
        CopyVectorToDevice(gmxpacked_excl);
    uint64_t* d_pair_shift_bits = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_pair_shift_bits),
                         sizeof(uint64_t) * snapshot.pair_shift_bits.size()),
              "cudaMalloc(pair_shift_bits)");
    int* d_sci_shift_safe_flags = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_sci_shift_safe_flags),
                         sizeof(int) * snapshot.sci.size()),
              "cudaMalloc(sci_shift_safe_flags)");
    int* d_sci_shift_safe_count = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_sci_shift_safe_count),
                         sizeof(int)),
              "cudaMalloc(sci_shift_safe_count)");
    auto launchKernel = [&]() {
        CheckCuda(cudaMemset(d_sci_shift_safe_count, 0, sizeof(int)),
                  "cudaMemset(sci_shift_safe_count)");
        Refresh_Gmxpacked_Pair_Shift_Bits<<<sci_numbers, refreshBlockSize>>>(
            sci_numbers, d_super_cluster_offsets,
            d_cluster_fractional_centers, d_cluster_fractional_extents,
            d_cluster_valid_masks, d_cluster_local_masks, d_sci, d_cjpacked,
            d_excl, d_pair_shift_bits, NULL,
            d_sci_shift_safe_flags, d_sci_shift_safe_count, true);
        CheckCuda(cudaGetLastError(), "launch production-gmxpacked-refresh");
    };

    for (int i = 0; i < warmup; ++i)
    {
        launchKernel();
    }
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");

    cudaEvent_t start, stop;
    CheckCuda(cudaEventCreate(&start), "cudaEventCreate");
    CheckCuda(cudaEventCreate(&stop), "cudaEventCreate");
    CheckCuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < iters; ++i)
    {
        launchKernel();
    }
    CheckCuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float total_ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&total_ms, start, stop),
              "cudaEventElapsedTime");
    int safe_count = 0;
    CheckCuda(cudaMemcpy(&safe_count, d_sci_shift_safe_count, sizeof(int),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(sci_shift_safe_count)");
    std::printf(
        "kernel=sponge_production_gmxpacked_refresh snapshot=%s avg_ms=%.6f "
        "iters=%d block_size=%d sci=%d cjpacked=%zu safe_sci=%d unsafe_sci=%d\n",
        snapshotLabel, total_ms / static_cast<float>(iters), iters,
        refreshBlockSize, sci_numbers, snapshot.cjpacked.size(), safe_count,
        sci_numbers - safe_count);
    PrintRefreshVerification(snapshot, d_pair_shift_bits, d_sci_shift_safe_flags,
                             "sponge_production_gmxpacked_refresh");

    CheckCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    CheckCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    CheckCuda(cudaFree(d_sci_shift_safe_count),
              "cudaFree(sci_shift_safe_count)");
    CheckCuda(cudaFree(d_sci_shift_safe_flags),
              "cudaFree(sci_shift_safe_flags)");
    CheckCuda(cudaFree(d_pair_shift_bits), "cudaFree(pair_shift_bits)");
    CheckCuda(cudaFree(d_excl), "cudaFree(excl)");
    CheckCuda(cudaFree(d_cjpacked), "cudaFree(cjpacked)");
    CheckCuda(cudaFree(d_sci), "cudaFree(sci)");
    CheckCuda(cudaFree(d_cluster_local_masks), "cudaFree(cluster_local_masks)");
    CheckCuda(cudaFree(d_cluster_valid_masks), "cudaFree(cluster_valid_masks)");
    CheckCuda(cudaFree(d_cluster_fractional_extents),
              "cudaFree(cluster_fractional_extents)");
    CheckCuda(cudaFree(d_cluster_fractional_centers),
              "cudaFree(cluster_fractional_centers)");
    CheckCuda(cudaFree(d_super_cluster_offsets),
              "cudaFree(super_cluster_offsets)");
}

void RunSpongeProductionGmxpackedRefreshRootChildQueue2(
    const SpongeGmxpackedForceOnlySnapshot& snapshot, int warmup, int iters,
    const char* snapshotLabel, int refreshBlockSize)
{
    if (snapshot.cluster_centers.empty() ||
        snapshot.cluster_extents.size() != snapshot.cluster_centers.size() ||
        snapshot.candidate_leaf_offsets.empty() ||
        snapshot.octree_prefixes.empty())
    {
        std::fprintf(
            stderr,
            "production-gmxpacked refresh queue2 requires a snapshot with "
            "builder metadata footer\n");
        std::exit(1);
    }

    const int sci_numbers = static_cast<int>(snapshot.sci.size());
    const int candidate_sci_numbers =
        static_cast<int>(snapshot.candidate_leaf_offsets.size() - 1);
    const int taskCapacity = std::max(1, candidate_sci_numbers * 64);

    std::vector<::VECTOR> cluster_centers =
        MakeProbeVector(snapshot.cluster_centers);
    std::vector<::VECTOR> cluster_extents =
        MakeProbeVector(snapshot.cluster_extents);
    const ::LTMatrix3 rcell =
        MakeProbeMatrix(InvertCellMatrix(MakeMatrix(snapshot.header.cell)));
    std::vector<::VECTOR> cluster_fractional_centers =
        MakeRefreshFractionalCenters(cluster_centers, rcell);
    std::vector<::VECTOR> cluster_fractional_extents =
        MakeRefreshFractionalExtents(cluster_extents, rcell);
    std::vector<LJ_CLUSTERED_GMXPACKED_SCI> gmxpacked_sci =
        MakeProductionGmxpackedSciVector(snapshot.sci);
    std::vector<LJ_CLUSTERED_GMXPACKED_CJ> gmxpacked_cjpacked =
        MakeProductionGmxpackedCjVector(snapshot.cjpacked);
    std::vector<LJ_CLUSTERED_GMXPACKED_EXCLUSION> gmxpacked_excl =
        MakeProductionGmxpackedExclusionVector(snapshot.excl);
    std::vector<::VECTOR> super_cluster_centers =
        MakeProbeVector(snapshot.super_cluster_centers);
    std::vector<::VECTOR> super_cluster_sizes =
        MakeProbeVector(snapshot.super_cluster_sizes);

    int* d_super_cluster_offsets =
        CopyVectorToDevice(snapshot.super_cluster_offsets);
    ::VECTOR* d_cluster_fractional_centers =
        CopyVectorToDevice(cluster_fractional_centers);
    ::VECTOR* d_cluster_fractional_extents =
        CopyVectorToDevice(cluster_fractional_extents);
    unsigned int* d_cluster_valid_masks =
        CopyVectorToDevice(snapshot.cluster_valid_masks);
    unsigned int* d_cluster_local_masks =
        CopyVectorToDevice(snapshot.cluster_local_masks);
    LJ_CLUSTERED_GMXPACKED_SCI* d_sci = CopyVectorToDevice(gmxpacked_sci);
    LJ_CLUSTERED_GMXPACKED_CJ* d_cjpacked =
        CopyVectorToDevice(gmxpacked_cjpacked);
    LJ_CLUSTERED_GMXPACKED_EXCLUSION* d_excl =
        CopyVectorToDevice(gmxpacked_excl);
    uint64_t* d_pair_shift_bits = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_pair_shift_bits),
                         sizeof(uint64_t) * snapshot.pair_shift_bits.size()),
              "cudaMalloc(pair_shift_bits)");
    int* d_sci_shift_safe_flags = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_sci_shift_safe_flags),
                         sizeof(int) * snapshot.sci.size()),
              "cudaMalloc(sci_shift_safe_flags)");
    int* d_sci_shift_safe_count = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_sci_shift_safe_count),
                         sizeof(int)),
              "cudaMalloc(sci_shift_safe_count)");

    int* d_sci_supercluster_ids =
        CopyVectorToDevice(snapshot.sci_supercluster_ids);
    ::VECTOR* d_super_cluster_centers =
        CopyVectorToDevice(super_cluster_centers);
    ::VECTOR* d_super_cluster_sizes = CopyVectorToDevice(super_cluster_sizes);
    uint64_t* d_node_prefixes = CopyVectorToDevice(snapshot.octree_prefixes);
    int* d_child_offsets = CopyVectorToDevice(snapshot.octree_child_offsets);
    int* d_candidate_shift_ids =
        CopyVectorToDevice(snapshot.candidate_shift_ids);
    int* d_task_counter = nullptr;
    int* d_task_overflow = nullptr;
    int* d_task_sci_ids = nullptr;
    int* d_task_nodes = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_task_counter), sizeof(int)),
              "cudaMalloc(root_child_task_counter)");
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_task_overflow), sizeof(int)),
              "cudaMalloc(root_child_task_overflow)");
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_task_sci_ids),
                         sizeof(int) * taskCapacity),
              "cudaMalloc(root_child_task_sci_ids)");
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_task_nodes),
                         sizeof(int) * taskCapacity),
              "cudaMalloc(root_child_task_nodes)");

    constexpr int taskBuildBlockSize = 128;
    const int taskBuildItems = candidate_sci_numbers * 8;
    const int taskBuildBlocks =
        (taskBuildItems + taskBuildBlockSize - 1) / taskBuildBlockSize;

    auto launchKernel = [&]() {
        CheckCuda(cudaMemset(d_sci_shift_safe_count, 0, sizeof(int)),
                  "cudaMemset(sci_shift_safe_count)");
        Refresh_Gmxpacked_Pair_Shift_Bits<<<sci_numbers, refreshBlockSize>>>(
            sci_numbers, d_super_cluster_offsets,
            d_cluster_fractional_centers, d_cluster_fractional_extents,
            d_cluster_valid_masks, d_cluster_local_masks, d_sci, d_cjpacked,
            d_excl, d_pair_shift_bits, NULL,
            d_sci_shift_safe_flags, d_sci_shift_safe_count, true);
        CheckCuda(cudaGetLastError(), "launch production-gmxpacked-refresh");

        CheckCuda(cudaMemset(d_task_counter, 0, sizeof(int)),
                  "cudaMemset(root_child_task_counter)");
        CheckCuda(cudaMemset(d_task_overflow, 0, sizeof(int)),
                  "cudaMemset(root_child_task_overflow)");
        Launch_Clustered_Gmxpacked_Candidate_Leaf_Root_Child_Task_Build(
            taskBuildBlocks, taskBuildBlockSize, candidate_sci_numbers,
            d_sci_supercluster_ids, d_super_cluster_centers,
            d_super_cluster_sizes, d_node_prefixes, d_child_offsets,
            snapshot.candidate_shift_ids.empty() ? nullptr
                                                 : d_candidate_shift_ids,
            false, true, taskCapacity, d_task_counter, d_task_overflow,
            d_task_sci_ids, d_task_nodes, 2);
        CheckCuda(cudaGetLastError(), "launch root-child queue2 payload build");
    };

    for (int i = 0; i < warmup; ++i)
    {
        launchKernel();
    }
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");

    cudaEvent_t start, stop;
    CheckCuda(cudaEventCreate(&start), "cudaEventCreate");
    CheckCuda(cudaEventCreate(&stop), "cudaEventCreate");
    CheckCuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < iters; ++i)
    {
        launchKernel();
    }
    CheckCuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float total_ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&total_ms, start, stop),
              "cudaEventElapsedTime");

    int safe_count = 0;
    int task_count = 0;
    int task_overflow = 0;
    CheckCuda(cudaMemcpy(&safe_count, d_sci_shift_safe_count, sizeof(int),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(sci_shift_safe_count)");
    CheckCuda(cudaMemcpy(&task_count, d_task_counter, sizeof(int),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(root_child_task_counter)");
    CheckCuda(cudaMemcpy(&task_overflow, d_task_overflow, sizeof(int),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(root_child_task_overflow)");

    std::printf(
        "kernel=sponge_production_gmxpacked_refresh_rootchild_queue2 "
        "snapshot=%s avg_ms=%.6f iters=%d block_size=%d sci=%d cjpacked=%zu safe_sci=%d "
        "unsafe_sci=%d candidate_sci=%d root_child_tasks=%d "
        "root_child_task_overflow=%d root_child_task_capacity=%d\n",
        snapshotLabel, total_ms / static_cast<float>(iters), iters,
        refreshBlockSize, sci_numbers, snapshot.cjpacked.size(), safe_count,
        sci_numbers - safe_count, candidate_sci_numbers, task_count,
        task_overflow, taskCapacity);
    PrintRefreshVerification(snapshot, d_pair_shift_bits, d_sci_shift_safe_flags,
                             "sponge_production_gmxpacked_refresh_rootchild_queue2");

    CheckCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    CheckCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    CheckCuda(cudaFree(d_task_nodes), "cudaFree(root_child_task_nodes)");
    CheckCuda(cudaFree(d_task_sci_ids), "cudaFree(root_child_task_sci_ids)");
    CheckCuda(cudaFree(d_task_overflow), "cudaFree(root_child_task_overflow)");
    CheckCuda(cudaFree(d_task_counter), "cudaFree(root_child_task_counter)");
    CheckCuda(cudaFree(d_candidate_shift_ids), "cudaFree(candidate_shift_ids)");
    CheckCuda(cudaFree(d_child_offsets), "cudaFree(child_offsets)");
    CheckCuda(cudaFree(d_node_prefixes), "cudaFree(node_prefixes)");
    CheckCuda(cudaFree(d_super_cluster_sizes),
              "cudaFree(super_cluster_sizes)");
    CheckCuda(cudaFree(d_super_cluster_centers),
              "cudaFree(super_cluster_centers)");
    CheckCuda(cudaFree(d_sci_supercluster_ids),
              "cudaFree(sci_supercluster_ids)");
    CheckCuda(cudaFree(d_sci_shift_safe_count),
              "cudaFree(sci_shift_safe_count)");
    CheckCuda(cudaFree(d_sci_shift_safe_flags),
              "cudaFree(sci_shift_safe_flags)");
    CheckCuda(cudaFree(d_pair_shift_bits), "cudaFree(pair_shift_bits)");
    CheckCuda(cudaFree(d_excl), "cudaFree(excl)");
    CheckCuda(cudaFree(d_cjpacked), "cudaFree(cjpacked)");
    CheckCuda(cudaFree(d_sci), "cudaFree(sci)");
    CheckCuda(cudaFree(d_cluster_local_masks), "cudaFree(cluster_local_masks)");
    CheckCuda(cudaFree(d_cluster_valid_masks), "cudaFree(cluster_valid_masks)");
    CheckCuda(cudaFree(d_cluster_fractional_extents),
              "cudaFree(cluster_fractional_extents)");
    CheckCuda(cudaFree(d_cluster_fractional_centers),
              "cudaFree(cluster_fractional_centers)");
    CheckCuda(cudaFree(d_super_cluster_offsets),
              "cudaFree(super_cluster_offsets)");
}

void RunSpongeProductionGmxpackedRecordStreamInnerActive(
    const SpongeGmxpackedForceOnlySnapshot& snapshot, int warmup, int iters,
    const char* snapshotLabel)
{
    if (snapshot.sci.empty() || snapshot.cjpacked.empty() ||
        snapshot.cluster_centers.empty() || snapshot.sorted_atom_ids.empty() ||
        snapshot.sorted_xq.empty())
    {
        std::fprintf(
            stderr,
            "production-gmxpacked record-stream-inner-active replay requires "
            "a production gmxpacked snapshot with builder metadata and coords\n");
        std::exit(1);
    }

    const int sciNumbers = static_cast<int>(snapshot.sci.size());
    const int totalAtomNumbers =
        static_cast<int>(snapshot.header.total_atom_numbers);
    int sourceCapacity = 0;
    for (const SpongeGmxpackedSciPOD& sci : snapshot.sci)
    {
        for (int packedIdx = sci.cjpacked_begin; packedIdx < sci.cjpacked_end;
             packedIdx += 1)
        {
            const SpongeGmxpackedCjPOD& packed =
                snapshot.cjpacked[static_cast<size_t>(packedIdx)];
            for (int jm = 0; jm < kJGroupSize; jm += 1)
            {
                if (packed.cj[jm] < 0)
                {
                    continue;
                }
                for (int split = 0; split < kWarpSplitCount; split += 1)
                {
                    const unsigned int imask =
                        (packed.split[split].imask >>
                         (jm * kSuperClusterClusters)) &
                        ((1u << kSuperClusterClusters) - 1u);
                    sourceCapacity += imask != 0u ? 1 : 0;
                }
            }
        }
    }
    if (sourceCapacity <= 0)
    {
        std::fprintf(stderr,
                     "record-stream-inner-active replay found no sources\n");
        std::exit(1);
    }

    std::vector<::VECTOR> clusterCenters =
        MakeProbeVector(snapshot.cluster_centers);
    std::vector<::VECTOR> crd(static_cast<size_t>(totalAtomNumbers));
    for (size_t sortedIdx = 0; sortedIdx < snapshot.sorted_xq.size();
         sortedIdx += 1)
    {
        const int atomId = snapshot.sorted_atom_ids[sortedIdx];
        if (atomId < 0 || atomId >= totalAtomNumbers)
        {
            continue;
        }
        const Float4POD& xq = snapshot.sorted_xq[sortedIdx];
        crd[static_cast<size_t>(atomId)] = {xq.x, xq.y, xq.z};
    }
    std::vector<int> identityOffsets(static_cast<size_t>(sourceCapacity));
    std::iota(identityOffsets.begin(), identityOffsets.end(), 0);

    int* dClusterOffsets = CopyVectorToDevice(snapshot.cluster_offsets);
    int* dSuperClusterOffsets =
        CopyVectorToDevice(snapshot.super_cluster_offsets);
    unsigned int* dClusterLocalMasks =
        CopyVectorToDevice(snapshot.cluster_local_masks);
    ::VECTOR* dClusterCenters = CopyVectorToDevice(clusterCenters);
    ::VECTOR* dCrd = CopyVectorToDevice(crd);
    int* dPermutation = CopyVectorToDevice(snapshot.sorted_atom_ids);
    LJ_CLUSTERED_GMXPACKED_SCI* dSci =
        reinterpret_cast<LJ_CLUSTERED_GMXPACKED_SCI*>(
            CopyVectorToDevice(snapshot.sci));
    LJ_CLUSTERED_GMXPACKED_CJ* dCjpacked =
        reinterpret_cast<LJ_CLUSTERED_GMXPACKED_CJ*>(
            CopyVectorToDevice(snapshot.cjpacked));
    LJ_CLUSTERED_GMXPACKED_EXCLUSION* dExcl =
        reinterpret_cast<LJ_CLUSTERED_GMXPACKED_EXCLUSION*>(
            CopyVectorToDevice(snapshot.excl));
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* dSources = nullptr;
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* dActiveSources = nullptr;
    int* dSourceCursor = nullptr;
    int* dOverflowRows = nullptr;
    int* dActiveFlags = nullptr;
    unsigned int* dActiveImasks = nullptr;
    int* dActiveOffsets = CopyVectorToDevice(identityOffsets);
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&dSources),
                         sizeof(*dSources) * sourceCapacity),
              "cudaMalloc(record_stream_sources)");
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&dActiveSources),
                         sizeof(*dActiveSources) * sourceCapacity),
              "cudaMalloc(record_stream_active_sources)");
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&dSourceCursor), sizeof(int)),
              "cudaMalloc(record_stream_source_cursor)");
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&dOverflowRows), sizeof(int)),
              "cudaMalloc(record_stream_overflow_rows)");
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&dActiveFlags),
                         sizeof(int) * sourceCapacity),
              "cudaMalloc(record_stream_active_flags)");
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&dActiveImasks),
                         sizeof(unsigned int) * sourceCapacity),
              "cudaMalloc(record_stream_active_imasks)");

    CheckCuda(cudaMemset(dSourceCursor, 0, sizeof(int)),
              "cudaMemset(record_stream_source_cursor)");
    CheckCuda(cudaMemset(dOverflowRows, 0, sizeof(int)),
              "cudaMemset(record_stream_overflow_rows)");
    const int materializeBlockSize =
        GetOptionalEnvInt("SPONGE_RECORD_STREAM_MATERIALIZE_BLOCK_SIZE", 256);
    Launch_Clustered_Gmxpacked_Record_Stream_Source_Materialize_From_Gmxpacked(
        sciNumbers, materializeBlockSize, dSci, dCjpacked, dExcl,
        sourceCapacity, dSources, dSourceCursor, dOverflowRows);
    CheckCuda(cudaGetLastError(), "launch record-stream materialize");
    CheckCuda(cudaDeviceSynchronize(),
              "cudaDeviceSynchronize(record-stream materialize)");
    int sourceRows = 0;
    int overflowRows = 0;
    CheckCuda(cudaMemcpy(&sourceRows, dSourceCursor, sizeof(int),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(record_stream_source_cursor)");
    CheckCuda(cudaMemcpy(&overflowRows, dOverflowRows, sizeof(int),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(record_stream_overflow_rows)");
    if (sourceRows <= 0 || sourceRows > sourceCapacity || overflowRows != 0)
    {
        std::fprintf(stderr,
                     "record-stream materialize failed: source_rows=%d "
                     "overflow=%d capacity=%d\n",
                     sourceRows, overflowRows, sourceCapacity);
        std::exit(1);
    }

    const ::LTMatrix3 cell = MakeProbeMatrix(snapshot.header.cell);
    const LTMatrix3 hostRcell = InvertCellMatrix(MakeMatrix(snapshot.header.cell));
    const ::LTMatrix3 rcell = {hostRcell.a11, hostRcell.a21, hostRcell.a22,
                               hostRcell.a31, hostRcell.a32, hostRcell.a33};
    const float cutoffSq = snapshot.header.cutoff * snapshot.header.cutoff;
    const int innerBlockSize =
        GetOptionalEnvInt("SPONGE_RECORD_STREAM_INNER_ACTIVE_BLOCK_SIZE", 1024);

    auto launchCount = [&]() {
        Launch_Clustered_Gmxpacked_Record_Stream_Inner_Active_Count_Probe(
            sourceRows, innerBlockSize, dSources, dPermutation,
            dClusterOffsets, dSuperClusterOffsets, dClusterLocalMasks,
            dClusterCenters, dCrd, cell, rcell, cutoffSq, dActiveFlags,
            dActiveImasks);
        CheckCuda(cudaGetLastError(), "launch record-stream inner count");
    };
    auto launchFillRecompute = [&]() {
        Launch_Clustered_Gmxpacked_Record_Stream_Inner_Active_Fill_Probe(
            sourceRows, innerBlockSize, dSources, dPermutation,
            dClusterOffsets, dSuperClusterOffsets, dClusterLocalMasks,
            dClusterCenters, dCrd, cell, rcell, cutoffSq, dActiveOffsets,
            dActiveSources);
        CheckCuda(cudaGetLastError(),
                  "launch record-stream inner fill recompute");
    };
    auto launchFillCached = [&]() {
        Launch_Clustered_Gmxpacked_Record_Stream_Inner_Active_Fill_Cached_Probe(
            sourceRows, innerBlockSize, dSources, dActiveImasks,
            dActiveOffsets, dActiveSources);
        CheckCuda(cudaGetLastError(), "launch record-stream inner fill cached");
    };

    auto timeKernel = [&](const auto& launch) {
        for (int i = 0; i < warmup; i += 1)
        {
            launch();
        }
        CheckCuda(cudaDeviceSynchronize(),
                  "cudaDeviceSynchronize(record-stream warmup)");
        cudaEvent_t start, stop;
        CheckCuda(cudaEventCreate(&start), "cudaEventCreate");
        CheckCuda(cudaEventCreate(&stop), "cudaEventCreate");
        CheckCuda(cudaEventRecord(start), "cudaEventRecord(start)");
        for (int i = 0; i < iters; i += 1)
        {
            launch();
        }
        CheckCuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
        CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
        float totalMs = 0.0f;
        CheckCuda(cudaEventElapsedTime(&totalMs, start, stop),
                  "cudaEventElapsedTime");
        CheckCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
        CheckCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
        return totalMs / static_cast<float>(iters);
    };

    const float countMs = timeKernel(launchCount);
    const float fillRecomputeMs = timeKernel(launchFillRecompute);
    launchCount();
    CheckCuda(cudaDeviceSynchronize(),
              "cudaDeviceSynchronize(record-stream cached prep)");
    const float fillCachedMs = timeKernel(launchFillCached);
    const float pairRecomputeMs = timeKernel([&]() {
        launchCount();
        launchFillRecompute();
    });
    const float pairCachedMs = timeKernel([&]() {
        launchCount();
        launchFillCached();
    });

    launchCount();
    CheckCuda(cudaDeviceSynchronize(),
              "cudaDeviceSynchronize(record-stream final count)");
    std::vector<int> activeFlags(static_cast<size_t>(sourceRows));
    CheckCuda(cudaMemcpy(activeFlags.data(), dActiveFlags,
                         sizeof(int) * activeFlags.size(),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(record_stream_active_flags)");
    int activeRows = 0;
    for (int flag : activeFlags)
    {
        activeRows += flag != 0 ? 1 : 0;
    }

    std::printf(
        "kernel=sponge_production_gmxpacked_record_stream_inner_active "
        "snapshot=%s iters=%d source_rows=%d source_capacity=%d active_rows=%d "
        "overflow=%d inner_block_size=%d materialize_block_size=%d "
        "count_ms=%.6f fill_recompute_ms=%.6f fill_cached_ms=%.6f "
        "pair_recompute_ms=%.6f pair_cached_ms=%.6f\n",
        snapshotLabel, iters, sourceRows, sourceCapacity, activeRows,
        overflowRows, innerBlockSize, materializeBlockSize, countMs,
        fillRecomputeMs, fillCachedMs, pairRecomputeMs, pairCachedMs);

    CheckCuda(cudaFree(dActiveImasks), "cudaFree(record_stream_active_imasks)");
    CheckCuda(cudaFree(dActiveFlags), "cudaFree(record_stream_active_flags)");
    CheckCuda(cudaFree(dOverflowRows), "cudaFree(record_stream_overflow_rows)");
    CheckCuda(cudaFree(dSourceCursor), "cudaFree(record_stream_source_cursor)");
    CheckCuda(cudaFree(dActiveSources),
              "cudaFree(record_stream_active_sources)");
    CheckCuda(cudaFree(dSources), "cudaFree(record_stream_sources)");
    CheckCuda(cudaFree(dActiveOffsets),
              "cudaFree(record_stream_active_offsets)");
    CheckCuda(cudaFree(dExcl), "cudaFree(excl)");
    CheckCuda(cudaFree(dCjpacked), "cudaFree(cjpacked)");
    CheckCuda(cudaFree(dSci), "cudaFree(sci)");
    CheckCuda(cudaFree(dPermutation), "cudaFree(permutation)");
    CheckCuda(cudaFree(dCrd), "cudaFree(crd)");
    CheckCuda(cudaFree(dClusterCenters), "cudaFree(cluster_centers)");
    CheckCuda(cudaFree(dClusterLocalMasks), "cudaFree(cluster_local_masks)");
    CheckCuda(cudaFree(dSuperClusterOffsets),
              "cudaFree(super_cluster_offsets)");
    CheckCuda(cudaFree(dClusterOffsets), "cudaFree(cluster_offsets)");
}

struct CollectSuperStats
{
    long long leaves = 0;
    long long node_visits = 0;
    int sci = 0;
    int nonzero_sci = 0;
    int max_count = 0;
};

int CollectPercentile(std::vector<int> values, double p)
{
    if (values.empty())
    {
        return 0;
    }
    std::sort(values.begin(), values.end());
    const double clamped = std::min(std::max(p, 0.0), 1.0);
    const size_t idx = static_cast<size_t>(
        std::llround(clamped * static_cast<double>(values.size() - 1)));
    return values[idx];
}

double CollectTopShare(std::vector<int> values, long long total, double frac)
{
    if (values.empty() || total <= 0)
    {
        return 0.0;
    }
    std::sort(values.begin(), values.end(), std::greater<int>());
    const size_t take = std::max<size_t>(
        1, static_cast<size_t>(std::ceil(frac * values.size())));
    long long top = 0;
    for (size_t i = 0; i < std::min(take, values.size()); ++i)
    {
        top += values[i];
    }
    return static_cast<double>(top) / static_cast<double>(total);
}

void PrintCollectDistribution(
    const SpongeGmxpackedForceOnlySnapshot& snapshot, const char* snapshotLabel,
    const char* modeName, bool cooperativeTraversal,
    const std::vector<int>& counts,
    const std::vector<LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_STATS>*
        statsOrNull)
{
    std::array<unsigned long long, 12> hist = {};
    std::array<long long, kShiftCount> shiftLeaves = {};
    std::array<long long, kShiftCount> shiftNodeVisits = {};
    std::array<int, kShiftCount> shiftSci = {};
    std::array<int, kShiftCount> shiftNonzero = {};
    std::vector<CollectSuperStats> superStats(
        snapshot.sci_supercluster_ids.empty()
            ? 0
            : static_cast<size_t>(*std::max_element(
                    snapshot.sci_supercluster_ids.begin(),
                    snapshot.sci_supercluster_ids.end())) +
                  1);
    std::vector<int> nonzeroCounts;
    nonzeroCounts.reserve(counts.size());
    long long totalLeaves = 0;
    int nonzeroSci = 0;
    int maxLeaves = 0;
    int zeroSci = 0;

    auto histBin = [](int count) {
        if (count == 0) return 0;
        if (count == 1) return 1;
        if (count <= 4) return 2;
        if (count <= 8) return 3;
        if (count <= 16) return 4;
        if (count <= 32) return 5;
        if (count <= 64) return 6;
        if (count <= 128) return 7;
        if (count <= 256) return 8;
        if (count <= 512) return 9;
        if (count <= 1024) return 10;
        return 11;
    };

    for (size_t sci = 0; sci < counts.size(); ++sci)
    {
        const int count = counts[sci];
        totalLeaves += count;
        maxLeaves = std::max(maxLeaves, count);
        hist[static_cast<size_t>(histBin(count))] += 1;
        if (count > 0)
        {
            nonzeroSci += 1;
            nonzeroCounts.push_back(count);
        }
        else
        {
            zeroSci += 1;
        }

        const int sciBase =
            snapshot.candidate_shift_ids.empty()
                ? static_cast<int>(sci) / kShiftCount
                : static_cast<int>(sci);
        const int shiftId =
            snapshot.candidate_shift_ids.empty()
                ? static_cast<int>(sci) % kShiftCount
                : snapshot.candidate_shift_ids[sci];
        const int superI =
            sciBase >= 0 &&
                    static_cast<size_t>(sciBase) <
                        snapshot.sci_supercluster_ids.size()
                ? snapshot.sci_supercluster_ids[static_cast<size_t>(sciBase)]
                : -1;
        const long long nodeVisits =
            statsOrNull != nullptr ? (*statsOrNull)[sci].node_visits : 0;
        if (shiftId >= 0 && shiftId < kShiftCount)
        {
            shiftSci[static_cast<size_t>(shiftId)] += 1;
            shiftLeaves[static_cast<size_t>(shiftId)] += count;
            shiftNodeVisits[static_cast<size_t>(shiftId)] += nodeVisits;
            if (count > 0)
            {
                shiftNonzero[static_cast<size_t>(shiftId)] += 1;
            }
        }
        if (superI >= 0 && static_cast<size_t>(superI) < superStats.size())
        {
            CollectSuperStats& super = superStats[static_cast<size_t>(superI)];
            super.sci += 1;
            super.leaves += count;
            super.node_visits += nodeVisits;
            super.max_count = std::max(super.max_count, count);
            if (count > 0)
            {
                super.nonzero_sci += 1;
            }
        }
    }

    const double mean =
        counts.empty() ? 0.0
                       : static_cast<double>(totalLeaves) /
                             static_cast<double>(counts.size());
    std::printf(
        "analysis=collect_distribution snapshot=%s mode=%s cooperative=%u "
        "candidate_sci=%zu zero_sci=%d nonzero_sci=%d leaves=%lld "
        "count_mean=%.6f count_p50=%d count_p75=%d count_p90=%d "
        "count_p95=%d count_p99=%d count_max=%d "
        "count_nonzero_p50=%d count_nonzero_p90=%d count_nonzero_p99=%d "
        "hist=[0:%llu 1:%llu 2-4:%llu 5-8:%llu 9-16:%llu 17-32:%llu "
        "33-64:%llu 65-128:%llu 129-256:%llu 257-512:%llu "
        "513-1024:%llu 1025+:%llu]\n",
        snapshotLabel, modeName, cooperativeTraversal ? 1u : 0u,
        counts.size(), zeroSci, nonzeroSci, totalLeaves, mean,
        CollectPercentile(counts, 0.50), CollectPercentile(counts, 0.75),
        CollectPercentile(counts, 0.90), CollectPercentile(counts, 0.95),
        CollectPercentile(counts, 0.99), maxLeaves,
        CollectPercentile(nonzeroCounts, 0.50),
        CollectPercentile(nonzeroCounts, 0.90),
        CollectPercentile(nonzeroCounts, 0.99), hist[0], hist[1], hist[2],
        hist[3], hist[4], hist[5], hist[6], hist[7], hist[8], hist[9],
        hist[10], hist[11]);

    std::vector<int> order(counts.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return counts[static_cast<size_t>(a)] != counts[static_cast<size_t>(b)]
                   ? counts[static_cast<size_t>(a)] >
                         counts[static_cast<size_t>(b)]
                   : a < b;
    });
    const int topCount = std::min<int>(16, static_cast<int>(order.size()));
    for (int rank = 0; rank < topCount; ++rank)
    {
        const int sci = order[static_cast<size_t>(rank)];
        const int sciBase = snapshot.candidate_shift_ids.empty()
                                ? sci / kShiftCount
                                : sci;
        const int shiftId = snapshot.candidate_shift_ids.empty()
                                ? sci % kShiftCount
                                : snapshot.candidate_shift_ids[sci];
        const int superI =
            sciBase >= 0 &&
                    static_cast<size_t>(sciBase) <
                        snapshot.sci_supercluster_ids.size()
                ? snapshot.sci_supercluster_ids[static_cast<size_t>(sciBase)]
                : -1;
        LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_STATS emptyStats = {};
        const auto& stat =
            statsOrNull != nullptr ? (*statsOrNull)[static_cast<size_t>(sci)]
                                   : emptyStats;
        std::printf(
            "analysis=collect_top_sci snapshot=%s mode=%s cooperative=%u "
            "rank=%d sci=%d shift=%d super_i=%d leaves=%d node_visits=%d "
            "overlap_tests=%d endpoint_leaves=%d halfshell_rejects=%d "
            "screen_tests=%d screen_accepts=%d root_rejects=%d\n",
            snapshotLabel, modeName, cooperativeTraversal ? 1u : 0u, rank, sci,
            shiftId, superI, counts[static_cast<size_t>(sci)],
            stat.node_visits, stat.overlap_tests, stat.endpoint_leaves,
            stat.halfshell_rejects, stat.screen_tests, stat.screen_accepts,
            stat.root_rejects);
    }

    for (int shift = 0; shift < kShiftCount; ++shift)
    {
        std::printf(
            "analysis=collect_shift_distribution snapshot=%s mode=%s "
            "cooperative=%u shift=%d sci=%d nonzero_sci=%d leaves=%lld "
            "node_visits=%lld\n",
            snapshotLabel, modeName, cooperativeTraversal ? 1u : 0u, shift,
            shiftSci[static_cast<size_t>(shift)],
            shiftNonzero[static_cast<size_t>(shift)],
            shiftLeaves[static_cast<size_t>(shift)],
            shiftNodeVisits[static_cast<size_t>(shift)]);
    }

    std::vector<int> superOrder(superStats.size());
    std::iota(superOrder.begin(), superOrder.end(), 0);
    std::sort(superOrder.begin(), superOrder.end(), [&](int a, int b) {
        const auto& sa = superStats[static_cast<size_t>(a)];
        const auto& sb = superStats[static_cast<size_t>(b)];
        return sa.leaves != sb.leaves ? sa.leaves > sb.leaves : a < b;
    });
    const int topSuper = std::min<int>(16, static_cast<int>(superOrder.size()));
    for (int rank = 0; rank < topSuper; ++rank)
    {
        const int superI = superOrder[static_cast<size_t>(rank)];
        const auto& stat = superStats[static_cast<size_t>(superI)];
        std::printf(
            "analysis=collect_top_supercluster snapshot=%s mode=%s "
            "cooperative=%u rank=%d super_i=%d sci=%d nonzero_sci=%d "
            "leaves=%lld node_visits=%lld max_count=%d\n",
            snapshotLabel, modeName, cooperativeTraversal ? 1u : 0u, rank,
            superI, stat.sci, stat.nonzero_sci, stat.leaves,
            stat.node_visits, stat.max_count);
    }

    if (statsOrNull != nullptr)
    {
        std::vector<int> nodeVisits;
        std::vector<int> endpointLeaves;
        nodeVisits.reserve(statsOrNull->size());
        endpointLeaves.reserve(statsOrNull->size());
        long long totalNodeVisits = 0;
        long long totalOverlapTests = 0;
        long long totalEndpointLeaves = 0;
        long long totalHalfshellRejects = 0;
        long long totalScreenTests = 0;
        long long totalScreenAccepts = 0;
        int rootRejectSci = 0;
        for (const auto& stat : *statsOrNull)
        {
            nodeVisits.push_back(stat.node_visits);
            endpointLeaves.push_back(stat.endpoint_leaves);
            totalNodeVisits += stat.node_visits;
            totalOverlapTests += stat.overlap_tests;
            totalEndpointLeaves += stat.endpoint_leaves;
            totalHalfshellRejects += stat.halfshell_rejects;
            totalScreenTests += stat.screen_tests;
            totalScreenAccepts += stat.screen_accepts;
            rootRejectSci += stat.root_rejects > 0 ? 1 : 0;
        }
        std::printf(
            "analysis=collect_stats_summary snapshot=%s mode=%s "
            "cooperative=%u node_visits=%lld overlap_tests=%lld "
            "endpoint_leaves=%lld halfshell_rejects=%lld screen_tests=%lld "
            "screen_accepts=%lld root_reject_sci=%d node_p50=%d node_p90=%d "
            "node_p95=%d node_p99=%d node_top5_share=%.6f "
            "endpoint_top5_share=%.6f leaves_top5_share=%.6f\n",
            snapshotLabel, modeName, cooperativeTraversal ? 1u : 0u,
            totalNodeVisits, totalOverlapTests, totalEndpointLeaves,
            totalHalfshellRejects, totalScreenTests, totalScreenAccepts,
            rootRejectSci, CollectPercentile(nodeVisits, 0.50),
            CollectPercentile(nodeVisits, 0.90),
            CollectPercentile(nodeVisits, 0.95),
            CollectPercentile(nodeVisits, 0.99),
            CollectTopShare(nodeVisits, totalNodeVisits, 0.05),
            CollectTopShare(endpointLeaves, totalEndpointLeaves, 0.05),
            CollectTopShare(counts, totalLeaves, 0.05));
    }
}

void RunSpongeProductionGmxpackedCollect(
    const SpongeGmxpackedForceOnlySnapshot& snapshot, int warmup, int iters,
    const char* snapshotLabel, ClusteredGmxpackedCandidateLeafProbeMode mode,
    bool cooperativeTraversal, bool collectStats, bool activeCandidateList,
    bool rootChildSplit, bool rootChildTaskQueue, int rootChildTaskSplitDepth,
    bool rootChildDeviceCounter)
{
    if (snapshot.candidate_leaf_offsets.empty() ||
        snapshot.octree_prefixes.empty() ||
        snapshot.cluster_centers.empty() || snapshot.cluster_extents.empty())
    {
        std::fprintf(stderr,
                     "production-gmxpacked collect replay requires a snapshot "
                     "with builder metadata footer\n");
        std::exit(1);
    }
    const int candidate_sci_numbers =
        static_cast<int>(snapshot.candidate_leaf_offsets.size() - 1);
    std::vector<int> active_sci_indices;
    const int candidate_leaf_capacity = std::max<int>(
        1, static_cast<int>(!snapshot.candidate_leaf_ids.empty()
                                ? snapshot.candidate_leaf_ids.size()
                                : snapshot.cjpacked.size()));
    constexpr int defaultBuilderBlockSize = 128;
    constexpr int activeBuilderBlockSize = 128;
    const int builderBlockSize =
        activeCandidateList ? activeBuilderBlockSize : defaultBuilderBlockSize;
    const int groupsPerBlock = builderBlockSize / kClusterSize;

    std::vector<::VECTOR> cluster_centers =
        MakeProbeVector(snapshot.cluster_centers);
    std::vector<::VECTOR> cluster_extents =
        MakeProbeVector(snapshot.cluster_extents);
    std::vector<::VECTOR> super_cluster_centers =
        MakeProbeVector(snapshot.super_cluster_centers);
    std::vector<::VECTOR> super_cluster_sizes =
        MakeProbeVector(snapshot.super_cluster_sizes);
    int* d_sci_supercluster_ids =
        CopyVectorToDevice(snapshot.sci_supercluster_ids);
    ::VECTOR* d_super_cluster_centers =
        CopyVectorToDevice(super_cluster_centers);
    ::VECTOR* d_super_cluster_sizes = CopyVectorToDevice(super_cluster_sizes);
    int* d_super_cluster_offsets =
        CopyVectorToDevice(snapshot.super_cluster_offsets);
    int* d_leaf_cluster_starts =
        CopyVectorToDevice(snapshot.leaf_cluster_starts);
    int* d_leaf_cluster_ends = CopyVectorToDevice(snapshot.leaf_cluster_ends);
    int* d_leaf_all_local = CopyVectorToDevice(snapshot.leaf_all_local);
    ::VECTOR* d_cluster_centers = CopyVectorToDevice(cluster_centers);
    ::VECTOR* d_cluster_extents = CopyVectorToDevice(cluster_extents);
    unsigned int* d_cluster_valid_masks =
        CopyVectorToDevice(snapshot.cluster_valid_masks);
    unsigned int* d_cluster_local_masks =
        CopyVectorToDevice(snapshot.cluster_local_masks);
    uint64_t* d_node_prefixes = CopyVectorToDevice(snapshot.octree_prefixes);
    int* d_child_offsets = CopyVectorToDevice(snapshot.octree_child_offsets);
    int* d_parents = CopyVectorToDevice(snapshot.octree_parents);
    int* d_internal_to_leaf =
        CopyVectorToDevice(snapshot.octree_internal_to_leaf);
    int* d_candidate_shift_ids =
        CopyVectorToDevice(snapshot.candidate_shift_ids);
    int* d_root_child_task_sci_ids = nullptr;
    int* d_root_child_task_nodes = nullptr;
    int* d_root_child_task_counter = nullptr;
    int* d_root_child_task_overflow = nullptr;
    int* d_root_child_task_work_cursor = nullptr;
    int* d_root_child_task_leaf_counts = nullptr;
    int rootChildTaskCapacity = 0;
    int rootChildDeviceCounterBlocks = 0;
    int rootChildTaskCount = 0;
    int rootChildTaskOverflow = 0;
    float rootChildTaskBuildMs = 0.0f;
    float rootChildTaskBuildTotalMs = 0.0f;
    float rootChildTaskBuildRepeatGpuMs = 0.0f;
    double rootChildTaskBuildHostMs = 0.0;
    double rootChildTaskCounterCopyHostMs = 0.0;
    double rootChildTaskBuildRepeatHostMs = 0.0;
    int rootChildTaskBuildRepeatIters = 0;
    if (rootChildTaskQueue)
    {
        rootChildTaskCapacity = std::max(
            1, candidate_sci_numbers *
                   (rootChildTaskSplitDepth > 1 ? 64 : 8));
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_root_child_task_sci_ids),
                             sizeof(int) * rootChildTaskCapacity),
                  "cudaMalloc(root_child_task_sci_ids)");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_root_child_task_nodes),
                             sizeof(int) * rootChildTaskCapacity),
                  "cudaMalloc(root_child_task_nodes)");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_root_child_task_counter),
                             sizeof(int)),
                  "cudaMalloc(root_child_task_counter)");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_root_child_task_overflow),
                             sizeof(int)),
                  "cudaMalloc(root_child_task_overflow)");
        if (rootChildDeviceCounter)
        {
            CheckCuda(cudaMalloc(
                          reinterpret_cast<void**>(&d_root_child_task_work_cursor),
                          sizeof(int)),
                      "cudaMalloc(root_child_task_work_cursor)");
            CheckCuda(cudaMalloc(
                          reinterpret_cast<void**>(&d_root_child_task_leaf_counts),
                          sizeof(int) * rootChildTaskCapacity),
                      "cudaMalloc(root_child_task_leaf_counts)");
            rootChildDeviceCounterBlocks =
                GetOptionalEnvInt("SPONGE_ROOT_CHILD_DEVICE_COUNTER_BLOCKS",
                                  256);
        }
        constexpr int taskBuildBlockSize = 128;
        const int taskBuildItems = candidate_sci_numbers * 8;
        const int taskBuildBlocks =
            (taskBuildItems + taskBuildBlockSize - 1) / taskBuildBlockSize;
        auto launchTaskBuild = [&]() {
            CheckCuda(cudaMemset(d_root_child_task_counter, 0, sizeof(int)),
                      "cudaMemset(root_child_task_counter)");
            CheckCuda(cudaMemset(d_root_child_task_overflow, 0, sizeof(int)),
                      "cudaMemset(root_child_task_overflow)");
            Launch_Clustered_Gmxpacked_Candidate_Leaf_Root_Child_Task_Build(
                taskBuildBlocks, taskBuildBlockSize, candidate_sci_numbers,
                d_sci_supercluster_ids, d_super_cluster_centers,
                d_super_cluster_sizes, d_node_prefixes, d_child_offsets,
                snapshot.candidate_shift_ids.empty() ? nullptr
                                                     : d_candidate_shift_ids,
                false, true, rootChildTaskCapacity, d_root_child_task_counter,
                d_root_child_task_overflow, d_root_child_task_sci_ids,
                d_root_child_task_nodes, rootChildTaskSplitDepth);
            CheckCuda(cudaGetLastError(), "launch root-child task build");
        };
        cudaEvent_t taskTotalStart, taskKernelStart, taskStop;
        CheckCuda(cudaEventCreate(&taskTotalStart),
                  "cudaEventCreate(taskTotalStart)");
        CheckCuda(cudaEventCreate(&taskKernelStart),
                  "cudaEventCreate(taskKernelStart)");
        CheckCuda(cudaEventCreate(&taskStop), "cudaEventCreate(taskStop)");
        const auto taskBuildHostStart = std::chrono::steady_clock::now();
        CheckCuda(cudaEventRecord(taskTotalStart),
                  "cudaEventRecord(root_child_task_total_start)");
        CheckCuda(cudaMemset(d_root_child_task_counter, 0, sizeof(int)),
                  "cudaMemset(root_child_task_counter)");
        CheckCuda(cudaMemset(d_root_child_task_overflow, 0, sizeof(int)),
                  "cudaMemset(root_child_task_overflow)");
        CheckCuda(cudaEventRecord(taskKernelStart),
                  "cudaEventRecord(root_child_task_kernel_start)");
        Launch_Clustered_Gmxpacked_Candidate_Leaf_Root_Child_Task_Build(
            taskBuildBlocks, taskBuildBlockSize, candidate_sci_numbers,
            d_sci_supercluster_ids, d_super_cluster_centers,
            d_super_cluster_sizes, d_node_prefixes, d_child_offsets,
            snapshot.candidate_shift_ids.empty() ? nullptr
                                                 : d_candidate_shift_ids,
            false, true, rootChildTaskCapacity, d_root_child_task_counter,
            d_root_child_task_overflow, d_root_child_task_sci_ids,
            d_root_child_task_nodes, rootChildTaskSplitDepth);
        CheckCuda(cudaGetLastError(), "launch root-child task build");
        CheckCuda(cudaEventRecord(taskStop),
                  "cudaEventRecord(root_child_task_stop)");
        CheckCuda(cudaEventSynchronize(taskStop),
                  "cudaEventSynchronize(root_child_task_stop)");
        const auto taskBuildHostStop = std::chrono::steady_clock::now();
        CheckCuda(cudaEventElapsedTime(&rootChildTaskBuildTotalMs,
                                       taskTotalStart, taskStop),
                  "cudaEventElapsedTime(root_child_task_build_total)");
        CheckCuda(cudaEventElapsedTime(&rootChildTaskBuildMs, taskKernelStart,
                                       taskStop),
                  "cudaEventElapsedTime(root_child_task_build)");
        rootChildTaskBuildHostMs =
            std::chrono::duration<double, std::milli>(taskBuildHostStop -
                                                      taskBuildHostStart)
                .count();
        if (!rootChildDeviceCounter)
        {
            const auto counterCopyHostStart = std::chrono::steady_clock::now();
            CheckCuda(cudaMemcpy(&rootChildTaskCount, d_root_child_task_counter,
                                 sizeof(int), cudaMemcpyDeviceToHost),
                      "cudaMemcpy(root_child_task_counter)");
            CheckCuda(cudaMemcpy(&rootChildTaskOverflow,
                                 d_root_child_task_overflow, sizeof(int),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(root_child_task_overflow)");
            const auto counterCopyHostStop = std::chrono::steady_clock::now();
            rootChildTaskCounterCopyHostMs =
                std::chrono::duration<double, std::milli>(counterCopyHostStop -
                                                          counterCopyHostStart)
                    .count();
        }
        rootChildTaskBuildRepeatIters = GetOptionalEnvInt(
            "SPONGE_ROOT_CHILD_TASK_BUILD_BENCH_ITERS", 0);
        if (rootChildTaskBuildRepeatIters > 0)
        {
            constexpr int taskBuildRepeatWarmup = 8;
            for (int i = 0; i < taskBuildRepeatWarmup; i += 1)
            {
                launchTaskBuild();
            }
            CheckCuda(cudaDeviceSynchronize(),
                      "cudaDeviceSynchronize(root_child_task_repeat_warmup)");
            CheckCuda(cudaEventRecord(taskTotalStart),
                      "cudaEventRecord(root_child_task_repeat_start)");
            const auto repeatHostStart = std::chrono::steady_clock::now();
            for (int i = 0; i < rootChildTaskBuildRepeatIters; i += 1)
            {
                launchTaskBuild();
            }
            CheckCuda(cudaEventRecord(taskStop),
                      "cudaEventRecord(root_child_task_repeat_stop)");
            CheckCuda(cudaEventSynchronize(taskStop),
                      "cudaEventSynchronize(root_child_task_repeat_stop)");
            const auto repeatHostStop = std::chrono::steady_clock::now();
            float repeatTotalMs = 0.0f;
            CheckCuda(cudaEventElapsedTime(&repeatTotalMs, taskTotalStart,
                                           taskStop),
                      "cudaEventElapsedTime(root_child_task_repeat)");
            rootChildTaskBuildRepeatGpuMs =
                repeatTotalMs / static_cast<float>(rootChildTaskBuildRepeatIters);
            rootChildTaskBuildRepeatHostMs =
                std::chrono::duration<double, std::milli>(repeatHostStop -
                                                          repeatHostStart)
                    .count() /
                static_cast<double>(rootChildTaskBuildRepeatIters);
            if (!rootChildDeviceCounter)
            {
                CheckCuda(cudaMemcpy(&rootChildTaskCount,
                                     d_root_child_task_counter, sizeof(int),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy(root_child_task_counter_repeat)");
                CheckCuda(cudaMemcpy(&rootChildTaskOverflow,
                                     d_root_child_task_overflow, sizeof(int),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy(root_child_task_overflow_repeat)");
            }
        }
        CheckCuda(cudaEventDestroy(taskTotalStart),
                  "cudaEventDestroy(taskTotalStart)");
        CheckCuda(cudaEventDestroy(taskKernelStart),
                  "cudaEventDestroy(taskKernelStart)");
        CheckCuda(cudaEventDestroy(taskStop), "cudaEventDestroy(taskStop)");
        if (!rootChildDeviceCounter &&
            (rootChildTaskCount <= 0 || rootChildTaskOverflow != 0))
        {
            std::fprintf(stderr,
                         "root-child task build failed: tasks=%d overflow=%d "
                         "capacity=%d\n",
                         rootChildTaskCount, rootChildTaskOverflow,
                         rootChildTaskCapacity);
            std::exit(1);
        }
    }
    if (activeCandidateList)
    {
        int* d_full_counts = nullptr;
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_full_counts),
                             sizeof(int) * candidate_sci_numbers),
                  "cudaMalloc(active_full_counts)");
        CheckCuda(cudaMemset(d_full_counts, 0,
                             sizeof(int) * candidate_sci_numbers),
                  "cudaMemset(active_full_counts)");
        const int full_blocks =
            (candidate_sci_numbers + groupsPerBlock - 1) / groupsPerBlock;
        Launch_Clustered_Gmxpacked_Candidate_Leaf_Probe(
            mode, full_blocks, builderBlockSize, candidate_sci_numbers,
            d_sci_supercluster_ids, d_super_cluster_centers,
            d_super_cluster_sizes, d_super_cluster_offsets,
            d_leaf_cluster_starts, d_leaf_cluster_ends, d_leaf_all_local,
            MakeProbeMatrix(snapshot.header.cell), snapshot.header.cutoff,
            d_cluster_centers, d_cluster_extents, d_cluster_valid_masks,
            d_cluster_local_masks, d_node_prefixes, d_child_offsets, d_parents,
            d_internal_to_leaf,
            snapshot.candidate_shift_ids.empty() ? nullptr
                                                 : d_candidate_shift_ids,
            true, false, !cooperativeTraversal, cooperativeTraversal, false,
            candidate_leaf_capacity, d_full_counts, nullptr, nullptr, nullptr,
            nullptr, nullptr);
        CheckCuda(cudaGetLastError(), "launch active-list full collect");
        CheckCuda(cudaDeviceSynchronize(),
                  "cudaDeviceSynchronize(active-list full collect)");
        std::vector<int> full_counts(
            static_cast<size_t>(candidate_sci_numbers));
        CheckCuda(cudaMemcpy(full_counts.data(), d_full_counts,
                             sizeof(int) * full_counts.size(),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(active_full_counts)");
        CheckCuda(cudaFree(d_full_counts), "cudaFree(active_full_counts)");
        active_sci_indices.reserve(
            static_cast<size_t>(candidate_sci_numbers));
        for (int sci = 0; sci < candidate_sci_numbers; sci += 1)
        {
            if (full_counts[static_cast<size_t>(sci)] > 0)
            {
                active_sci_indices.push_back(sci);
            }
        }
        if (active_sci_indices.empty())
        {
            std::fprintf(stderr,
                         "active collect replay found no active candidate "
                         "SCI rows in full probe\n");
            std::exit(1);
        }
    }
    const int launch_sci_numbers =
        activeCandidateList ? static_cast<int>(active_sci_indices.size())
                            : candidate_sci_numbers;
    const int launch_work_items =
        rootChildTaskQueue
            ? (rootChildDeviceCounter ? rootChildTaskCapacity
                                      : rootChildTaskCount)
            : (rootChildSplit ? launch_sci_numbers * 8
                              : launch_sci_numbers);
    const int blocks =
        rootChildDeviceCounter
            ? rootChildDeviceCounterBlocks
            : (launch_work_items + groupsPerBlock - 1) / groupsPerBlock;
    int* d_active_sci_indices =
        activeCandidateList ? CopyVectorToDevice(active_sci_indices) : nullptr;
    const int output_count_numbers =
        rootChildTaskQueue ? candidate_sci_numbers : launch_sci_numbers;
    int* d_counts = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_counts),
                         sizeof(int) * output_count_numbers),
              "cudaMalloc(collect_counts)");
    LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_RECORD* d_records = nullptr;
    int* d_cursor = nullptr;
    int* d_overflow = nullptr;
    LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_STATS* d_stats = nullptr;
    const bool emit =
        mode == ClusteredGmxpackedCandidateLeafProbeMode::Emit;
    if (collectStats)
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_stats),
                             sizeof(*d_stats) * output_count_numbers),
                  "cudaMalloc(collect_stats)");
    }
    if (emit)
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_records),
                             sizeof(*d_records) * candidate_leaf_capacity),
                  "cudaMalloc(collect_records)");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_cursor), sizeof(int)),
                  "cudaMalloc(collect_cursor)");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_overflow),
                             sizeof(int)),
                  "cudaMalloc(collect_overflow)");
    }

    const ::LTMatrix3 probeCell = MakeProbeMatrix(snapshot.header.cell);
    const char* modeName =
        mode == ClusteredGmxpackedCandidateLeafProbeMode::Traversal
            ? "traversal"
            : (mode == ClusteredGmxpackedCandidateLeafProbeMode::Screen
                   ? "screen"
                   : "emit");
    auto launchKernel = [&]() {
        CheckCuda(cudaMemset(d_counts, 0,
                             sizeof(int) * output_count_numbers),
                  "cudaMemset(collect_counts)");
        if (collectStats)
        {
            CheckCuda(cudaMemset(d_stats, 0,
                                 sizeof(*d_stats) * output_count_numbers),
                      "cudaMemset(collect_stats)");
        }
        if (emit)
        {
            CheckCuda(cudaMemset(d_cursor, 0, sizeof(int)),
                      "cudaMemset(collect_cursor)");
            CheckCuda(cudaMemset(d_overflow, 0, sizeof(int)),
                      "cudaMemset(collect_overflow)");
        }
        if (rootChildDeviceCounter)
        {
            CheckCuda(cudaMemset(d_root_child_task_work_cursor, 0, sizeof(int)),
                      "cudaMemset(root_child_task_work_cursor)");
            Launch_Clustered_Gmxpacked_Candidate_Leaf_Root_Child_Device_Counter_Probe(
                blocks, builderBlockSize, candidate_sci_numbers,
                d_sci_supercluster_ids, d_super_cluster_centers,
                d_super_cluster_sizes, d_super_cluster_offsets,
                d_leaf_cluster_starts, d_leaf_cluster_ends, d_leaf_all_local,
                probeCell, snapshot.header.cutoff, d_cluster_centers,
                d_cluster_extents, d_cluster_valid_masks, d_cluster_local_masks,
                d_node_prefixes, d_child_offsets, d_parents, d_internal_to_leaf,
                snapshot.candidate_shift_ids.empty() ? nullptr
                                                     : d_candidate_shift_ids,
                true, false, !cooperativeTraversal, rootChildTaskCapacity,
                d_root_child_task_counter, d_root_child_task_work_cursor,
                d_counts, d_root_child_task_leaf_counts,
                d_root_child_task_sci_ids, d_root_child_task_nodes);
        }
        else
        {
            Launch_Clustered_Gmxpacked_Candidate_Leaf_Probe(
                mode, blocks, builderBlockSize,
                rootChildTaskQueue ? launch_work_items : launch_sci_numbers,
                d_sci_supercluster_ids, d_super_cluster_centers,
                d_super_cluster_sizes, d_super_cluster_offsets,
                d_leaf_cluster_starts, d_leaf_cluster_ends, d_leaf_all_local,
                probeCell, snapshot.header.cutoff, d_cluster_centers,
                d_cluster_extents, d_cluster_valid_masks, d_cluster_local_masks,
                d_node_prefixes, d_child_offsets, d_parents, d_internal_to_leaf,
                snapshot.candidate_shift_ids.empty() ? nullptr
                                                     : d_candidate_shift_ids,
                true, false, !cooperativeTraversal, cooperativeTraversal,
                rootChildSplit, candidate_leaf_capacity, d_counts,
                emit ? d_records : nullptr, emit ? d_cursor : nullptr,
                emit ? d_overflow : nullptr, collectStats ? d_stats : nullptr,
                d_active_sci_indices, d_root_child_task_sci_ids,
                d_root_child_task_nodes);
        }
        CheckCuda(cudaGetLastError(), "launch production-gmxpacked-collect");
    };

    for (int i = 0; i < warmup; ++i)
    {
        launchKernel();
    }
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");

    cudaEvent_t start, stop;
    CheckCuda(cudaEventCreate(&start), "cudaEventCreate");
    CheckCuda(cudaEventCreate(&stop), "cudaEventCreate");
    CheckCuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < iters; ++i)
    {
        launchKernel();
    }
    CheckCuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float total_ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&total_ms, start, stop),
              "cudaEventElapsedTime");
    if (rootChildDeviceCounter)
    {
        const auto counterCopyHostStart = std::chrono::steady_clock::now();
        CheckCuda(cudaMemcpy(&rootChildTaskCount, d_root_child_task_counter,
                             sizeof(int), cudaMemcpyDeviceToHost),
                  "cudaMemcpy(root_child_task_counter_post_collect)");
        CheckCuda(cudaMemcpy(&rootChildTaskOverflow, d_root_child_task_overflow,
                             sizeof(int), cudaMemcpyDeviceToHost),
                  "cudaMemcpy(root_child_task_overflow_post_collect)");
        const auto counterCopyHostStop = std::chrono::steady_clock::now();
        rootChildTaskCounterCopyHostMs =
            std::chrono::duration<double, std::milli>(counterCopyHostStop -
                                                      counterCopyHostStart)
                .count();
        if (rootChildTaskCount <= 0 || rootChildTaskOverflow != 0)
        {
            std::fprintf(stderr,
                         "root-child task build failed: tasks=%d overflow=%d "
                         "capacity=%d\n",
                         rootChildTaskCount, rootChildTaskOverflow,
                         rootChildTaskCapacity);
            std::exit(1);
        }
    }

    std::vector<int> counts(static_cast<size_t>(output_count_numbers));
    CheckCuda(cudaMemcpy(counts.data(), d_counts,
                         sizeof(int) * counts.size(),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(collect_counts)");
    long long totalLeaves = 0;
    int nonzeroSci = 0;
    int maxLeaves = 0;
    for (int count : counts)
    {
        totalLeaves += count;
        nonzeroSci += count > 0 ? 1 : 0;
        maxLeaves = std::max(maxLeaves, count);
    }
    std::vector<LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_STATS> stats;
    if (collectStats)
    {
        stats.resize(static_cast<size_t>(launch_sci_numbers));
        CheckCuda(cudaMemcpy(stats.data(), d_stats,
                             sizeof(*d_stats) * stats.size(),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(collect_stats)");
    }
    int cursor = -1;
    int overflow = -1;
    if (emit)
    {
        CheckCuda(cudaMemcpy(&cursor, d_cursor, sizeof(int),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(collect_cursor)");
        CheckCuda(cudaMemcpy(&overflow, d_overflow, sizeof(int),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(collect_overflow)");
    }
    std::printf(
        "kernel=sponge_production_gmxpacked_collect snapshot=%s avg_ms=%.6f "
        "iters=%d mode=%s cooperative=%u candidate_sci=%d leaves=%lld "
        "nonzero_sci=%d max_per_sci=%d cursor=%d overflow=%d capacity=%d "
        "active_list=%u root_child_split=%u work_items=%d "
        "root_child_queue=%u root_child_tasks=%d root_child_task_build_ms=%.6f "
        "root_child_task_total_ms=%.6f root_child_task_build_host_ms=%.6f "
        "root_child_task_counter_copy_host_ms=%.6f "
        "root_child_task_repeat_iters=%d root_child_task_repeat_gpu_ms=%.6f "
        "root_child_task_repeat_host_ms=%.6f "
        "root_child_task_overflow=%d root_child_task_split_depth=%d "
        "root_child_device_counter=%u root_child_device_counter_blocks=%d "
        "root_child_task_capacity=%d "
        "original_candidate_sci=%d\n",
        snapshotLabel, total_ms / static_cast<float>(iters), iters, modeName,
        cooperativeTraversal ? 1u : 0u, output_count_numbers, totalLeaves,
        nonzeroSci, maxLeaves, cursor, overflow, candidate_leaf_capacity,
        activeCandidateList ? 1u : 0u, rootChildSplit ? 1u : 0u,
        launch_work_items, rootChildTaskQueue ? 1u : 0u, rootChildTaskCount,
        rootChildTaskBuildMs, rootChildTaskBuildTotalMs,
        rootChildTaskBuildHostMs, rootChildTaskCounterCopyHostMs,
        rootChildTaskBuildRepeatIters, rootChildTaskBuildRepeatGpuMs,
        rootChildTaskBuildRepeatHostMs, rootChildTaskOverflow,
        rootChildTaskSplitDepth, rootChildDeviceCounter ? 1u : 0u,
        rootChildDeviceCounterBlocks, rootChildTaskCapacity,
        candidate_sci_numbers);
    PrintCollectDistribution(snapshot, snapshotLabel, modeName,
                             cooperativeTraversal, counts,
                             collectStats ? &stats : nullptr);

    CheckCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    CheckCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    CheckCuda(cudaFree(d_stats), "cudaFree(collect_stats)");
    CheckCuda(cudaFree(d_overflow), "cudaFree(collect_overflow)");
    CheckCuda(cudaFree(d_cursor), "cudaFree(collect_cursor)");
    CheckCuda(cudaFree(d_records), "cudaFree(collect_records)");
    CheckCuda(cudaFree(d_counts), "cudaFree(collect_counts)");
    CheckCuda(cudaFree(d_active_sci_indices),
              "cudaFree(active_sci_indices)");
    CheckCuda(cudaFree(d_root_child_task_work_cursor),
              "cudaFree(root_child_task_work_cursor)");
    CheckCuda(cudaFree(d_root_child_task_leaf_counts),
              "cudaFree(root_child_task_leaf_counts)");
    CheckCuda(cudaFree(d_root_child_task_overflow),
              "cudaFree(root_child_task_overflow)");
    CheckCuda(cudaFree(d_root_child_task_counter),
              "cudaFree(root_child_task_counter)");
    CheckCuda(cudaFree(d_root_child_task_nodes),
              "cudaFree(root_child_task_nodes)");
    CheckCuda(cudaFree(d_root_child_task_sci_ids),
              "cudaFree(root_child_task_sci_ids)");
    CheckCuda(cudaFree(d_candidate_shift_ids),
              "cudaFree(candidate_shift_ids)");
    CheckCuda(cudaFree(d_internal_to_leaf), "cudaFree(internal_to_leaf)");
    CheckCuda(cudaFree(d_parents), "cudaFree(parents)");
    CheckCuda(cudaFree(d_child_offsets), "cudaFree(child_offsets)");
    CheckCuda(cudaFree(d_node_prefixes), "cudaFree(node_prefixes)");
    CheckCuda(cudaFree(d_cluster_local_masks), "cudaFree(cluster_local_masks)");
    CheckCuda(cudaFree(d_cluster_valid_masks), "cudaFree(cluster_valid_masks)");
    CheckCuda(cudaFree(d_cluster_extents), "cudaFree(cluster_extents)");
    CheckCuda(cudaFree(d_cluster_centers), "cudaFree(cluster_centers)");
    CheckCuda(cudaFree(d_leaf_all_local), "cudaFree(leaf_all_local)");
    CheckCuda(cudaFree(d_leaf_cluster_ends), "cudaFree(leaf_cluster_ends)");
    CheckCuda(cudaFree(d_leaf_cluster_starts), "cudaFree(leaf_cluster_starts)");
    CheckCuda(cudaFree(d_super_cluster_offsets),
              "cudaFree(super_cluster_offsets)");
    CheckCuda(cudaFree(d_super_cluster_sizes),
              "cudaFree(super_cluster_sizes)");
    CheckCuda(cudaFree(d_super_cluster_centers),
              "cudaFree(super_cluster_centers)");
    CheckCuda(cudaFree(d_sci_supercluster_ids),
              "cudaFree(sci_supercluster_ids)");
}

SpongeForceOnlySnapshot ExtractPayloadFromFullOutputSnapshot(
    const SpongeClusteredFullOutputSnapshot& snapshot)
{
    SpongeForceOnlySnapshot payload = {};
    payload.header.file =
        nbnxm_microbench::MakeFileHeader(nbnxm_microbench::SnapshotKind::spongeForceOnly);
    payload.header.cluster_size = snapshot.header.cluster_size;
    payload.header.super_cluster_clusters =
        snapshot.header.super_cluster_clusters;
    payload.header.warp_split_count = snapshot.header.warp_split_count;
    payload.header.cluster_numbers = snapshot.header.cluster_numbers;
    payload.header.super_cluster_numbers = snapshot.header.super_cluster_numbers;
    payload.header.sci_numbers = snapshot.header.sci_numbers;
    payload.header.record_numbers = snapshot.header.record_numbers;
    payload.header.pair_shift_word_numbers = snapshot.header.pair_shift_word_numbers;
    payload.header.total_atom_numbers = snapshot.header.total_atom_numbers;
    payload.header.local_atom_numbers = snapshot.header.local_atom_numbers;
    payload.header.lj_param_numbers = snapshot.header.lj_param_numbers;
    payload.header.cutoff = snapshot.header.cutoff;
    payload.header.pme_beta = snapshot.header.pme_beta;
    payload.header.cell = snapshot.header.cell;
    payload.cluster_offsets = snapshot.cluster_offsets;
    payload.cluster_valid_masks = snapshot.cluster_valid_masks;
    payload.cluster_local_masks = snapshot.cluster_local_masks;
    payload.super_cluster_offsets = snapshot.super_cluster_offsets;
    payload.sci = snapshot.sci;
    payload.record_offsets = snapshot.record_offsets;
    payload.records = snapshot.records;
    payload.pair_shift_bits = snapshot.pair_shift_bits;
    payload.sorted_atom_ids = snapshot.sorted_atom_ids;
    payload.sorted_xq = snapshot.sorted_xq;
    payload.sorted_lj_type = snapshot.sorted_lj_type;
    payload.lj_ab = snapshot.lj_ab;
    return payload;
}

DiffStats CompareFloatArrays(const std::vector<float>& actual,
                             const std::vector<float>& reference)
{
    DiffStats stats = {};
    if (actual.size() != reference.size() || actual.empty())
    {
        stats.max_abs = std::numeric_limits<double>::infinity();
        stats.rms = std::numeric_limits<double>::infinity();
        return stats;
    }
    double sum_sq = 0.0;
    for (size_t i = 0; i < actual.size(); ++i)
    {
        const double diff = static_cast<double>(actual[i]) -
                            static_cast<double>(reference[i]);
        stats.max_abs = std::max(stats.max_abs, std::fabs(diff));
        stats.max_scaled = std::max(
            stats.max_scaled,
            std::fabs(diff) /
                (1.0 + std::fabs(static_cast<double>(reference[i]))));
        sum_sq += diff * diff;
    }
    stats.rms = std::sqrt(sum_sq / static_cast<double>(actual.size()));
    return stats;
}

DiffStats CompareForceArrays(const std::vector<Float4POD>& actual,
                             const std::vector<Float4POD>& reference)
{
    DiffStats stats = {};
    if (actual.size() != reference.size() || actual.empty())
    {
        stats.max_abs = std::numeric_limits<double>::infinity();
        stats.rms = std::numeric_limits<double>::infinity();
        return stats;
    }
    double sum_sq = 0.0;
    const double denom = static_cast<double>(actual.size()) * 3.0;
    for (size_t i = 0; i < actual.size(); ++i)
    {
        const double dx = static_cast<double>(actual[i].x) - reference[i].x;
        const double dy = static_cast<double>(actual[i].y) - reference[i].y;
        const double dz = static_cast<double>(actual[i].z) - reference[i].z;
        stats.max_abs =
            std::max({stats.max_abs, std::fabs(dx), std::fabs(dy), std::fabs(dz)});
        stats.max_scaled = std::max(
            {stats.max_scaled,
             std::fabs(dx) /
                 (1.0 + std::fabs(static_cast<double>(reference[i].x))),
             std::fabs(dy) /
                 (1.0 + std::fabs(static_cast<double>(reference[i].y))),
             std::fabs(dz) /
                 (1.0 + std::fabs(static_cast<double>(reference[i].z)))});
        sum_sq += dx * dx + dy * dy + dz * dz;
    }
    stats.rms = std::sqrt(sum_sq / denom);
    return stats;
}

DiffStats CompareVirialArrays(const std::vector<LTMatrix3>& actual,
                              const std::vector<LTMatrix3POD>& reference)
{
    DiffStats stats = {};
    if (actual.size() != reference.size() || actual.empty())
    {
        stats.max_abs = std::numeric_limits<double>::infinity();
        stats.rms = std::numeric_limits<double>::infinity();
        return stats;
    }
    double sum_sq = 0.0;
    const double denom = static_cast<double>(actual.size()) * 6.0;
    for (size_t i = 0; i < actual.size(); ++i)
    {
        const std::array<double, 6> diffs = {
            static_cast<double>(actual[i].a11) - reference[i].a11,
            static_cast<double>(actual[i].a21) - reference[i].a21,
            static_cast<double>(actual[i].a22) - reference[i].a22,
            static_cast<double>(actual[i].a31) - reference[i].a31,
            static_cast<double>(actual[i].a32) - reference[i].a32,
            static_cast<double>(actual[i].a33) - reference[i].a33,
        };
        const std::array<double, 6> reference_values = {
            reference[i].a11, reference[i].a21, reference[i].a22,
            reference[i].a31, reference[i].a32, reference[i].a33,
        };
        for (size_t component = 0; component < diffs.size(); ++component)
        {
            const double diff = diffs[component];
            stats.max_abs = std::max(stats.max_abs, std::fabs(diff));
            stats.max_scaled = std::max(
                stats.max_scaled,
                std::fabs(diff) /
                    (1.0 + std::fabs(reference_values[component])));
            sum_sq += diff * diff;
        }
    }
    stats.rms = std::sqrt(sum_sq / denom);
    return stats;
}

void RunSpongeFullOutput(const SpongeClusteredFullOutputSnapshot& snapshot,
                         int warmup, int iters, const char* snapshotLabel)
{
    if (snapshot.header.compute_virial == 0u)
    {
        std::fprintf(stderr,
                     "full-output replay requires compute_virial=1 in snapshot\n");
        std::exit(1);
    }
    if (snapshot.header.force_soa == 0u)
    {
        std::fprintf(stderr,
                     "full-output replay currently supports force_soa snapshots only\n");
        std::exit(1);
    }

    const auto sci_numbers = static_cast<int>(snapshot.header.sci_numbers);
    const auto total_atom_numbers =
        static_cast<int>(snapshot.header.total_atom_numbers);
    const bool need_energy = snapshot.header.compute_energy != 0u;
    const bool total_output = snapshot.header.total_output != 0u;
    const int scalar_output_numbers = total_output ? 1 : total_atom_numbers;

    float *frc_x = nullptr, *frc_y = nullptr, *frc_z = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&frc_x),
                         sizeof(float) * total_atom_numbers),
              "cudaMalloc");
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&frc_y),
                         sizeof(float) * total_atom_numbers),
              "cudaMalloc");
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&frc_z),
                         sizeof(float) * total_atom_numbers),
              "cudaMalloc");

    float* d_atom_energy = nullptr;
    float* d_atom_direct_cf_energy = nullptr;
    float* d_atom_lj_energy = nullptr;
    if (need_energy)
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_atom_energy),
                             sizeof(float) * scalar_output_numbers),
                  "cudaMalloc(atom_energy)");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_atom_direct_cf_energy),
                             sizeof(float) * scalar_output_numbers),
                  "cudaMalloc(atom_direct_cf_energy)");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_atom_lj_energy),
                             sizeof(float) * scalar_output_numbers),
                  "cudaMalloc(atom_lj_energy)");
    }
    LTMatrix3* d_atom_virial = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_atom_virial),
                         sizeof(LTMatrix3) * scalar_output_numbers),
              "cudaMalloc(atom_virial)");

    int* d_cluster_offsets = CopyVectorToDevice(snapshot.cluster_offsets);
    unsigned int* d_cluster_valid_masks =
        CopyVectorToDevice(snapshot.cluster_valid_masks);
    unsigned int* d_cluster_local_masks =
        CopyVectorToDevice(snapshot.cluster_local_masks);
    int* d_super_cluster_offsets =
        CopyVectorToDevice(snapshot.super_cluster_offsets);
    SpongeSciPOD* d_sci = CopyVectorToDevice(snapshot.sci);
    int* d_record_offsets = CopyVectorToDevice(snapshot.record_offsets);
    SpongeWarpJRecordPOD* d_records = CopyVectorToDevice(snapshot.records);
    uint64_t* d_pair_shift_bits = CopyVectorToDevice(snapshot.pair_shift_bits);
    int* d_sorted_atom_ids = CopyVectorToDevice(snapshot.sorted_atom_ids);
    std::vector<float4> sorted_xq(snapshot.sorted_xq.size());
    for (size_t i = 0; i < snapshot.sorted_xq.size(); ++i)
    {
        sorted_xq[i] = MakeFloat4(snapshot.sorted_xq[i]);
    }
    float4* d_sorted_xq = CopyVectorToDevice(sorted_xq);
    int* d_sorted_lj_type = CopyVectorToDevice(snapshot.sorted_lj_type);
    std::vector<float2> lj_ab(snapshot.lj_ab.size());
    for (size_t i = 0; i < snapshot.lj_ab.size(); ++i)
    {
        lj_ab[i] = MakeFloat2(snapshot.lj_ab[i]);
    }
    float2* d_lj_ab = CopyVectorToDevice(lj_ab);

    const dim3 block(kClusterSize, kClusterSize, 1);
    const dim3 grid(static_cast<unsigned int>(sci_numbers), 1, 1);
    const LTMatrix3 cell = MakeMatrix(snapshot.header.cell);

    auto launchKernel = [&]() {
        CheckCuda(cudaMemset(frc_x, 0, sizeof(float) * total_atom_numbers),
                  "cudaMemset(frc_x)");
        CheckCuda(cudaMemset(frc_y, 0, sizeof(float) * total_atom_numbers),
                  "cudaMemset(frc_y)");
        CheckCuda(cudaMemset(frc_z, 0, sizeof(float) * total_atom_numbers),
                  "cudaMemset(frc_z)");
        CheckCuda(cudaMemset(d_atom_virial, 0,
                             sizeof(LTMatrix3) * scalar_output_numbers),
                  "cudaMemset(atom_virial)");
        if (need_energy)
        {
            CheckCuda(cudaMemset(d_atom_energy, 0,
                                 sizeof(float) * scalar_output_numbers),
                      "cudaMemset(atom_energy)");
            CheckCuda(cudaMemset(d_atom_direct_cf_energy, 0,
                                 sizeof(float) * scalar_output_numbers),
                      "cudaMemset(atom_direct_cf_energy)");
            CheckCuda(cudaMemset(d_atom_lj_energy, 0,
                                 sizeof(float) * scalar_output_numbers),
                      "cudaMemset(atom_lj_energy)");
        }
        if (need_energy)
        {
            if (total_output)
            {
                SpongeVirialReplayWarpRecordKernel<true, true>
                    <<<grid, block>>>(
                        sci_numbers, snapshot.header.cluster_size,
                        snapshot.header.super_cluster_clusters,
                        static_cast<int>(snapshot.header.local_atom_numbers),
                        d_cluster_offsets, d_cluster_valid_masks,
                        d_cluster_local_masks, d_super_cluster_offsets, d_sci,
                        d_record_offsets, d_records, d_pair_shift_bits,
                        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type, cell,
                        d_lj_ab, snapshot.header.cutoff, frc_x, frc_y, frc_z,
                        snapshot.header.pme_beta, d_atom_energy, d_atom_virial,
                        d_atom_direct_cf_energy, d_atom_lj_energy);
            }
            else
            {
                SpongeVirialReplayWarpRecordKernel<true, false>
                    <<<grid, block>>>(
                        sci_numbers, snapshot.header.cluster_size,
                        snapshot.header.super_cluster_clusters,
                        static_cast<int>(snapshot.header.local_atom_numbers),
                        d_cluster_offsets, d_cluster_valid_masks,
                        d_cluster_local_masks, d_super_cluster_offsets, d_sci,
                        d_record_offsets, d_records, d_pair_shift_bits,
                        d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type, cell,
                        d_lj_ab, snapshot.header.cutoff, frc_x, frc_y, frc_z,
                        snapshot.header.pme_beta, d_atom_energy, d_atom_virial,
                        d_atom_direct_cf_energy, d_atom_lj_energy);
            }
        }
        else
        {
            SpongeVirialReplayWarpRecordKernel<false, false>
                <<<grid, block>>>(
                    sci_numbers, snapshot.header.cluster_size,
                    snapshot.header.super_cluster_clusters,
                    static_cast<int>(snapshot.header.local_atom_numbers),
                    d_cluster_offsets, d_cluster_valid_masks,
                    d_cluster_local_masks, d_super_cluster_offsets, d_sci,
                    d_record_offsets, d_records, d_pair_shift_bits,
                    d_sorted_atom_ids, d_sorted_xq, d_sorted_lj_type, cell,
                    d_lj_ab, snapshot.header.cutoff, frc_x, frc_y, frc_z,
                    snapshot.header.pme_beta, nullptr, d_atom_virial, nullptr,
                    nullptr);
        }
    };

    for (int i = 0; i < warmup; ++i)
    {
        launchKernel();
    }
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    cudaEvent_t start, stop;
    CheckCuda(cudaEventCreate(&start), "cudaEventCreate");
    CheckCuda(cudaEventCreate(&stop), "cudaEventCreate");
    CheckCuda(cudaEventRecord(start), "cudaEventRecord");
    for (int i = 0; i < iters; ++i)
    {
        launchKernel();
    }
    CheckCuda(cudaEventRecord(stop), "cudaEventRecord");
    CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float total_ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&total_ms, start, stop),
              "cudaEventElapsedTime");

    std::vector<float> host_frc_x(static_cast<size_t>(total_atom_numbers));
    std::vector<float> host_frc_y(static_cast<size_t>(total_atom_numbers));
    std::vector<float> host_frc_z(static_cast<size_t>(total_atom_numbers));
    CheckCuda(cudaMemcpy(host_frc_x.data(), frc_x,
                         sizeof(float) * total_atom_numbers,
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(frc_x)");
    CheckCuda(cudaMemcpy(host_frc_y.data(), frc_y,
                         sizeof(float) * total_atom_numbers,
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(frc_y)");
    CheckCuda(cudaMemcpy(host_frc_z.data(), frc_z,
                         sizeof(float) * total_atom_numbers,
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(frc_z)");
    std::vector<Float4POD> replay_force(static_cast<size_t>(total_atom_numbers),
                                        Float4POD{});
    for (int sorted_i = 0; sorted_i < total_atom_numbers; ++sorted_i)
    {
        const int atom_i = snapshot.sorted_atom_ids[static_cast<size_t>(sorted_i)];
        if (atom_i < 0 || atom_i >= total_atom_numbers)
        {
            continue;
        }
        replay_force[static_cast<size_t>(atom_i)].x +=
            host_frc_x[static_cast<size_t>(sorted_i)];
        replay_force[static_cast<size_t>(atom_i)].y +=
            host_frc_y[static_cast<size_t>(sorted_i)];
        replay_force[static_cast<size_t>(atom_i)].z +=
            host_frc_z[static_cast<size_t>(sorted_i)];
    }
    const DiffStats force_stats =
        CompareForceArrays(replay_force, snapshot.reference_force);

    DiffStats energy_stats = {};
    DiffStats direct_energy_stats = {};
    DiffStats lj_energy_stats = {};
    DiffStats virial_stats = {};

    std::vector<LTMatrix3> host_atom_virial(static_cast<size_t>(scalar_output_numbers));
    CheckCuda(cudaMemcpy(host_atom_virial.data(), d_atom_virial,
                         sizeof(LTMatrix3) * scalar_output_numbers,
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(atom_virial)");
    virial_stats =
        CompareVirialArrays(host_atom_virial, snapshot.reference_atom_virial);

    if (need_energy)
    {
        std::vector<float> host_atom_energy(static_cast<size_t>(scalar_output_numbers));
        std::vector<float> host_direct_cf_energy(static_cast<size_t>(scalar_output_numbers));
        std::vector<float> host_lj_energy(static_cast<size_t>(scalar_output_numbers));
        CheckCuda(cudaMemcpy(host_atom_energy.data(), d_atom_energy,
                             sizeof(float) * scalar_output_numbers,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(atom_energy)");
        CheckCuda(cudaMemcpy(host_direct_cf_energy.data(), d_atom_direct_cf_energy,
                             sizeof(float) * scalar_output_numbers,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(atom_direct_cf_energy)");
        CheckCuda(cudaMemcpy(host_lj_energy.data(), d_atom_lj_energy,
                             sizeof(float) * scalar_output_numbers,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(atom_lj_energy)");
        energy_stats = CompareFloatArrays(host_atom_energy,
                                          snapshot.reference_atom_energy);
        direct_energy_stats = CompareFloatArrays(
            host_direct_cf_energy, snapshot.reference_direct_cf_energy);
        lj_energy_stats =
            CompareFloatArrays(host_lj_energy, snapshot.reference_lj_energy);
    }

    std::printf(
        "kernel=sponge_fulloutput_warp_record snapshot=%s avg_ms=%.6f iters=%d sci=%d records=%zu atoms=%d force_max_abs=%.6e force_rms=%.6e virial_max_abs=%.6e virial_rms=%.6e energy_max_abs=%.6e energy_rms=%.6e direct_energy_max_abs=%.6e lj_energy_max_abs=%.6e total_output=%u\n",
        snapshotLabel, total_ms / static_cast<float>(iters), iters,
        sci_numbers, snapshot.records.size(), total_atom_numbers,
        force_stats.max_abs, force_stats.rms, virial_stats.max_abs,
        virial_stats.rms, energy_stats.max_abs, energy_stats.rms,
        direct_energy_stats.max_abs, lj_energy_stats.max_abs,
        snapshot.header.total_output);
}

void RunSpongeWithGmxPackedLjComb(
    const GromacsPairlistSnapshot& snapshot, int warmup, int iters,
    const char* snapshotLabel,
    const SpongeClusteredFullOutputSnapshot* referenceFullOutputSnapshot = nullptr,
    bool forcePerAtomOutput = false, bool usePartialReduce = false,
    bool useCentralDirectPartialReduce = false,
    bool useSpecializedPartialReduce = false,
    int partialReduceWriterThreshold = -1)
{
    std::vector<int> clusterOffsets;
    std::vector<unsigned int> clusterValidMasks;
    std::vector<unsigned int> clusterLocalMasks;
    MaterializeImplicitGromacsClusters(snapshot, &clusterOffsets,
                                       &clusterValidMasks,
                                       &clusterLocalMasks);
    std::vector<GromacsCjPackedPOD> normalizedCjpacked = snapshot.cjpacked;
    std::vector<GromacsExclPOD> normalizedExcl = snapshot.excl;
    if (normalizedExcl.empty())
    {
        GromacsExclPOD noExcl = {};
        for (unsigned int& pair : noExcl.pair)
        {
            pair = 0xffffffffu;
        }
        normalizedExcl.push_back(noExcl);
    }
    for (const GromacsSciPOD& sciEntry : snapshot.sci)
    {
        if (sciEntry.shift != kCentralShiftId)
        {
            continue;
        }
        const int superIClusterBase = sciEntry.sci * kSuperClusterClusters;
        for (int packedIdx = sciEntry.cjPackedBegin;
             packedIdx < sciEntry.cjPackedEnd; ++packedIdx)
        {
            GromacsCjPackedPOD& packed =
                normalizedCjpacked[static_cast<size_t>(packedIdx)];
            for (int split = 0; split < kWarpSplitCount; ++split)
            {
                const unsigned int imask = packed.imei[split].imask;
                if (imask == 0u)
                {
                    continue;
                }
                int exclIndex = packed.imei[split].excl_ind;
                if (exclIndex < 0 ||
                    static_cast<size_t>(exclIndex) >= normalizedExcl.size())
                {
                    exclIndex = 0;
                }
                GromacsExclPOD updated = normalizedExcl[static_cast<size_t>(exclIndex)];
                bool changed = false;
                for (int jm = 0; jm < kJGroupSize; ++jm)
                {
                    const int centralI = packed.cj[jm] - superIClusterBase;
                    if (centralI < 0 || centralI >= kSuperClusterClusters)
                    {
                        continue;
                    }
                    const unsigned int packedBit =
                        1u << static_cast<unsigned int>(
                            jm * kSuperClusterClusters + centralI);
                    if ((imask & packedBit) == 0u)
                    {
                        continue;
                    }
                    for (int splitJLane = 0; splitJLane < kSplitJClusterSize;
                         ++splitJLane)
                    {
                        const int globalJLane =
                            split * kSplitJClusterSize + splitJLane;
                        for (int iLane = 0; iLane < kClusterSize; ++iLane)
                        {
                            if (globalJLane > iLane)
                            {
                                continue;
                            }
                            unsigned int& pair =
                                updated.pair[splitJLane * kClusterSize + iLane];
                            if ((pair & packedBit) != 0u)
                            {
                                pair &= ~packedBit;
                                changed = true;
                            }
                        }
                    }
                }
                if (changed)
                {
                    packed.imei[split].excl_ind =
                        static_cast<int>(normalizedExcl.size());
                    normalizedExcl.push_back(updated);
                }
            }
        }
    }

    const auto sci_numbers = static_cast<int>(snapshot.header.sci_numbers);
    const auto total_atom_numbers =
        static_cast<int>(snapshot.header.total_atom_numbers);
    const bool needEnergy = snapshot.header.compute_energy != 0u;
    const bool needVirial = snapshot.header.compute_virial != 0u;
    const bool validateAgainstReference =
        referenceFullOutputSnapshot != nullptr;
    const bool totalOutput =
        !forcePerAtomOutput && !usePartialReduce &&
        (referenceFullOutputSnapshot == nullptr ||
         referenceFullOutputSnapshot->header.total_output != 0u);
    const int centralShiftId = FindZeroShiftId(snapshot.header.shiftvec);
    if (usePartialReduce && totalOutput)
    {
        std::fprintf(stderr,
                     "comb-gmxpacked partial-reduce requires per-atom output\n");
        std::exit(1);
    }
    if (useCentralDirectPartialReduce && !usePartialReduce)
    {
        std::fprintf(stderr,
                     "central-direct partial-reduce requires partial-reduce\n");
        std::exit(1);
    }
    const int output_atom_numbers = validateAgainstReference
                                        ? static_cast<int>(referenceFullOutputSnapshot
                                                               ->header.total_atom_numbers)
                                        : total_atom_numbers;
    const int scalar_output_numbers = totalOutput ? 1 : output_atom_numbers;
    const bool compactForceStorage = totalOutput && (needEnergy || needVirial);
    if (!totalOutput && snapshot.sorted_atom_ids.empty())
    {
        std::fprintf(stderr,
                     "comb-gmxpacked per-atom replay requires sorted_atom_ids\n");
        std::exit(1);
    }
    float3* frc_xyz_compact = nullptr;
    float4* frc_xyz_regular = nullptr;
    if (compactForceStorage)
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&frc_xyz_compact),
                             sizeof(float3) * total_atom_numbers),
                  "cudaMalloc(frc_xyz_compact)");
    }
    else
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&frc_xyz_regular),
                             sizeof(float4) * total_atom_numbers),
                  "cudaMalloc(frc_xyz_regular)");
    }
    float3* d_shift_force = nullptr;
    if (needVirial && totalOutput)
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_shift_force),
                             sizeof(float3) * snapshot.header.shiftvec.size()),
                  "cudaMalloc(shift_force)");
    }
    LTMatrix3* d_atom_virial = nullptr;
    if (needVirial && !totalOutput)
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_atom_virial),
                             sizeof(LTMatrix3) * scalar_output_numbers),
                  "cudaMalloc(atom_virial)");
    }
    float* d_atom_energy = nullptr;
    float* d_atom_direct_cf_energy = nullptr;
    float* d_atom_lj_energy = nullptr;
    if (needEnergy)
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_atom_energy),
                             sizeof(float) * scalar_output_numbers),
                  "cudaMalloc(atom_energy)");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_atom_direct_cf_energy),
                             sizeof(float) * scalar_output_numbers),
                  "cudaMalloc(atom_direct_cf_energy)");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_atom_lj_energy),
                             sizeof(float) * scalar_output_numbers),
                  "cudaMalloc(atom_lj_energy)");
    }
    GmxPackedPartialOutputIndex partialIndex;
    int* d_sci_partial_bases = nullptr;
    unsigned char* d_atom_partial_flags = nullptr;
    int* d_atom_partial_offsets = nullptr;
    int* d_atom_partial_slots = nullptr;
    int* d_reduce_atoms_2 = nullptr;
    int* d_reduce_slots_2 = nullptr;
    int* d_reduce_atoms_4 = nullptr;
    int* d_reduce_slots_4 = nullptr;
    int* d_reduce_atoms_6 = nullptr;
    int* d_reduce_slots_6 = nullptr;
    int* d_reduce_atoms_8 = nullptr;
    int* d_reduce_slots_8 = nullptr;
    int* d_reduce_generic_atoms = nullptr;
    LTMatrix3* d_partial_virial = nullptr;
    float* d_partial_energy = nullptr;
    float* d_partial_direct_cf_energy = nullptr;
    float* d_partial_lj_energy = nullptr;
    if (usePartialReduce)
    {
        partialIndex = BuildGmxPackedPartialOutputIndex(
            snapshot, useCentralDirectPartialReduce, centralShiftId,
            partialReduceWriterThreshold);
        if (useCentralDirectPartialReduce)
        {
            d_sci_partial_bases = CopyVectorToDevice(partialIndex.sciPartialBases);
        }
        if (partialReduceWriterThreshold >= 0)
        {
            d_atom_partial_flags =
                CopyVectorToDevice(partialIndex.atomPartialFlags);
        }
        d_atom_partial_offsets =
            CopyVectorToDevice(partialIndex.atomPartialOffsets);
        d_atom_partial_slots = CopyVectorToDevice(partialIndex.atomPartialSlots);
        d_reduce_atoms_2 = CopyVectorToDevice(partialIndex.reduceAtoms2);
        d_reduce_slots_2 = CopyVectorToDevice(partialIndex.reduceSlots2);
        d_reduce_atoms_4 = CopyVectorToDevice(partialIndex.reduceAtoms4);
        d_reduce_slots_4 = CopyVectorToDevice(partialIndex.reduceSlots4);
        d_reduce_atoms_6 = CopyVectorToDevice(partialIndex.reduceAtoms6);
        d_reduce_slots_6 = CopyVectorToDevice(partialIndex.reduceSlots6);
        d_reduce_atoms_8 = CopyVectorToDevice(partialIndex.reduceAtoms8);
        d_reduce_slots_8 = CopyVectorToDevice(partialIndex.reduceSlots8);
        d_reduce_generic_atoms =
            CopyVectorToDevice(partialIndex.reduceGenericAtoms);
        if (needVirial)
        {
            CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_partial_virial),
                                 sizeof(LTMatrix3) * partialIndex.partialCount),
                      "cudaMalloc(partial_virial)");
        }
        if (needEnergy)
        {
            CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_partial_energy),
                                 sizeof(float) * partialIndex.partialCount),
                      "cudaMalloc(partial_energy)");
            CheckCuda(cudaMalloc(reinterpret_cast<void**>(
                                      &d_partial_direct_cf_energy),
                                  sizeof(float) * partialIndex.partialCount),
                      "cudaMalloc(partial_direct_cf_energy)");
            CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_partial_lj_energy),
                                 sizeof(float) * partialIndex.partialCount),
                      "cudaMalloc(partial_lj_energy)");
        }
    }

    GromacsSciPOD* d_sci = CopyVectorToDevice(snapshot.sci);
    GromacsCjPackedPOD* d_cjpacked = CopyVectorToDevice(normalizedCjpacked);
    GromacsExclPOD* d_excl = CopyVectorToDevice(normalizedExcl);
    unsigned int* d_cluster_local_masks = CopyVectorToDevice(clusterLocalMasks);
    int* d_sorted_atom_ids = CopyVectorToDevice(snapshot.sorted_atom_ids);
    std::vector<float4> sorted_xq(snapshot.sorted_xq.size());
    for (size_t i = 0; i < snapshot.sorted_xq.size(); ++i)
    {
        sorted_xq[i] = MakeFloat4(snapshot.sorted_xq[i]);
    }
    float4* d_sorted_xq = CopyVectorToDevice(sorted_xq);
    std::vector<float2> sorted_lj_comb(snapshot.sorted_lj_comb.size());
    for (size_t i = 0; i < snapshot.sorted_lj_comb.size(); ++i)
    {
        sorted_lj_comb[i] = MakeFloat2(snapshot.sorted_lj_comb[i]);
    }
    float2* d_sorted_lj_comb = CopyVectorToDevice(sorted_lj_comb);
    std::vector<float4> shiftvec(snapshot.header.shiftvec.size());
    for (size_t i = 0; i < snapshot.header.shiftvec.size(); ++i)
    {
        shiftvec[i] = MakeFloat4(snapshot.header.shiftvec[i]);
    }
    float4* d_shiftvec = CopyVectorToDevice(shiftvec);

    const dim3 block(kClusterSize, kClusterSize, 1);
    const dim3 grid(static_cast<unsigned int>(sci_numbers), 1, 1);
    auto launchKernel = [&]() {
        if (compactForceStorage)
        {
            CheckCuda(cudaMemset(frc_xyz_compact, 0,
                                 sizeof(float3) * total_atom_numbers),
                      "cudaMemset(frc_xyz_compact)");
        }
        else
        {
            CheckCuda(cudaMemset(frc_xyz_regular, 0,
                                 sizeof(float4) * total_atom_numbers),
                      "cudaMemset(frc_xyz_regular)");
        }
        if (needVirial && totalOutput)
        {
            CheckCuda(cudaMemset(d_shift_force, 0,
                                 sizeof(float3) * shiftvec.size()),
                      "cudaMemset(shift_force)");
        }
        if (needVirial && !totalOutput)
        {
            CheckCuda(cudaMemset(d_atom_virial, 0,
                                 sizeof(LTMatrix3) * scalar_output_numbers),
                      "cudaMemset(atom_virial)");
        }
        if (usePartialReduce && needVirial && partialReduceWriterThreshold < 0)
        {
            CheckCuda(cudaMemset(d_partial_virial, 0,
                                 sizeof(LTMatrix3) * partialIndex.partialCount),
                      "cudaMemset(partial_virial)");
        }
        if (usePartialReduce && needEnergy && partialReduceWriterThreshold < 0)
        {
            CheckCuda(cudaMemset(d_partial_energy, 0,
                                 sizeof(float) * partialIndex.partialCount),
                      "cudaMemset(partial_energy)");
            CheckCuda(cudaMemset(d_partial_direct_cf_energy, 0,
                                 sizeof(float) * partialIndex.partialCount),
                      "cudaMemset(partial_direct_cf_energy)");
            CheckCuda(cudaMemset(d_partial_lj_energy, 0,
                                 sizeof(float) * partialIndex.partialCount),
                      "cudaMemset(partial_lj_energy)");
        }
        if (needEnergy && needVirial)
        {
            CheckCuda(cudaMemset(d_atom_energy, 0,
                                 sizeof(float) * scalar_output_numbers),
                      "cudaMemset(atom_energy)");
            CheckCuda(cudaMemset(d_atom_direct_cf_energy, 0,
                                 sizeof(float) * scalar_output_numbers),
                      "cudaMemset(atom_direct_cf_energy)");
            CheckCuda(cudaMemset(d_atom_lj_energy, 0,
                                 sizeof(float) * scalar_output_numbers),
                      "cudaMemset(atom_lj_energy)");
            if (totalOutput)
            {
                SpongeGmxPackedLjCombKernel<true, true, true><<<grid, block>>>(
                    sci_numbers, d_sci, d_cjpacked, d_excl,
                    d_cluster_local_masks, nullptr, nullptr, nullptr,
                    d_sorted_xq, d_sorted_lj_comb, d_shiftvec,
                    snapshot.header.cutoff, frc_xyz_compact,
                    snapshot.header.pme_beta, centralShiftId, nullptr,
                    d_shift_force, d_atom_energy, d_atom_direct_cf_energy,
                    d_atom_lj_energy);
            }
            else
            {
                if (usePartialReduce)
                {
                    if (useCentralDirectPartialReduce)
                    {
                        SpongeGmxPackedLjCombKernel<true, true, false, true,
                                                    true>
                            <<<grid, block>>>(
                                sci_numbers, d_sci, d_cjpacked, d_excl,
                                d_cluster_local_masks, d_sorted_atom_ids,
                                d_sci_partial_bases, nullptr, d_sorted_xq,
                                d_sorted_lj_comb, d_shiftvec,
                                snapshot.header.cutoff, frc_xyz_regular,
                                snapshot.header.pme_beta, centralShiftId,
                                d_partial_virial, nullptr, d_partial_energy,
                                d_partial_direct_cf_energy, d_partial_lj_energy,
                                d_atom_virial, d_atom_energy,
                                d_atom_direct_cf_energy, d_atom_lj_energy);
                    }
                    else if (partialReduceWriterThreshold >= 0)
                    {
                        SpongeGmxPackedLjCombKernel<true, true, false, true,
                                                    false, true>
                            <<<grid, block>>>(
                                sci_numbers, d_sci, d_cjpacked, d_excl,
                                d_cluster_local_masks, d_sorted_atom_ids,
                                nullptr, d_atom_partial_flags, d_sorted_xq,
                                d_sorted_lj_comb, d_shiftvec,
                                snapshot.header.cutoff, frc_xyz_regular,
                                snapshot.header.pme_beta, centralShiftId,
                                d_partial_virial, nullptr, d_partial_energy,
                                d_partial_direct_cf_energy, d_partial_lj_energy,
                                d_atom_virial, d_atom_energy,
                                d_atom_direct_cf_energy, d_atom_lj_energy);
                    }
                    else
                    {
                        SpongeGmxPackedLjCombKernel<true, true, false, true>
                            <<<grid, block>>>(
                                sci_numbers, d_sci, d_cjpacked, d_excl,
                                d_cluster_local_masks, d_sorted_atom_ids,
                                nullptr, nullptr, d_sorted_xq, d_sorted_lj_comb,
                                d_shiftvec, snapshot.header.cutoff,
                                frc_xyz_regular, snapshot.header.pme_beta,
                                centralShiftId, d_partial_virial, nullptr,
                                d_partial_energy,
                                d_partial_direct_cf_energy, d_partial_lj_energy);
                    }
                }
                else
                {
                    SpongeGmxPackedLjCombKernel<true, true, false>
                        <<<grid, block>>>(
                            sci_numbers, d_sci, d_cjpacked, d_excl,
                            d_cluster_local_masks, d_sorted_atom_ids,
                            nullptr, nullptr, d_sorted_xq, d_sorted_lj_comb,
                            d_shiftvec, snapshot.header.cutoff,
                            frc_xyz_regular, snapshot.header.pme_beta,
                            centralShiftId, d_atom_virial, nullptr, d_atom_energy,
                            d_atom_direct_cf_energy,
                            d_atom_lj_energy);
                }
            }
        }
        else if (needEnergy)
        {
            CheckCuda(cudaMemset(d_atom_energy, 0,
                                 sizeof(float) * scalar_output_numbers),
                      "cudaMemset(atom_energy)");
            CheckCuda(cudaMemset(d_atom_direct_cf_energy, 0,
                                 sizeof(float) * scalar_output_numbers),
                      "cudaMemset(atom_direct_cf_energy)");
            CheckCuda(cudaMemset(d_atom_lj_energy, 0,
                                 sizeof(float) * scalar_output_numbers),
                      "cudaMemset(atom_lj_energy)");
            if (totalOutput)
            {
                SpongeGmxPackedLjCombKernel<true, false, true><<<grid, block>>>(
                    sci_numbers, d_sci, d_cjpacked, d_excl,
                    d_cluster_local_masks, nullptr, nullptr, nullptr,
                    d_sorted_xq, d_sorted_lj_comb, d_shiftvec,
                    snapshot.header.cutoff, frc_xyz_compact,
                    snapshot.header.pme_beta, centralShiftId, nullptr, nullptr,
                    d_atom_energy, d_atom_direct_cf_energy, d_atom_lj_energy);
            }
            else
            {
                if (usePartialReduce)
                {
                    if (useCentralDirectPartialReduce)
                    {
                        SpongeGmxPackedLjCombKernel<true, false, false, true,
                                                    true>
                            <<<grid, block>>>(
                                sci_numbers, d_sci, d_cjpacked, d_excl,
                                d_cluster_local_masks, d_sorted_atom_ids,
                                d_sci_partial_bases, nullptr, d_sorted_xq,
                                d_sorted_lj_comb, d_shiftvec,
                                snapshot.header.cutoff, frc_xyz_regular,
                                snapshot.header.pme_beta, centralShiftId,
                                nullptr, nullptr, d_partial_energy,
                                d_partial_direct_cf_energy, d_partial_lj_energy,
                                nullptr, d_atom_energy,
                                d_atom_direct_cf_energy, d_atom_lj_energy);
                    }
                    else if (partialReduceWriterThreshold >= 0)
                    {
                        SpongeGmxPackedLjCombKernel<true, false, false, true,
                                                    false, true>
                            <<<grid, block>>>(
                                sci_numbers, d_sci, d_cjpacked, d_excl,
                                d_cluster_local_masks, d_sorted_atom_ids,
                                nullptr, d_atom_partial_flags, d_sorted_xq,
                                d_sorted_lj_comb, d_shiftvec,
                                snapshot.header.cutoff, frc_xyz_regular,
                                snapshot.header.pme_beta, centralShiftId,
                                nullptr, nullptr, d_partial_energy,
                                d_partial_direct_cf_energy, d_partial_lj_energy,
                                nullptr, d_atom_energy,
                                d_atom_direct_cf_energy, d_atom_lj_energy);
                    }
                    else
                    {
                        SpongeGmxPackedLjCombKernel<true, false, false, true>
                            <<<grid, block>>>(
                                sci_numbers, d_sci, d_cjpacked, d_excl,
                                d_cluster_local_masks, d_sorted_atom_ids,
                                nullptr, nullptr, d_sorted_xq, d_sorted_lj_comb,
                                d_shiftvec, snapshot.header.cutoff,
                                frc_xyz_regular, snapshot.header.pme_beta,
                                centralShiftId, nullptr, nullptr, d_partial_energy,
                                d_partial_direct_cf_energy, d_partial_lj_energy);
                    }
                }
                else
                {
                    SpongeGmxPackedLjCombKernel<true, false, false>
                        <<<grid, block>>>(
                            sci_numbers, d_sci, d_cjpacked, d_excl,
                            d_cluster_local_masks, d_sorted_atom_ids,
                            nullptr, nullptr, d_sorted_xq, d_sorted_lj_comb,
                            d_shiftvec, snapshot.header.cutoff,
                            frc_xyz_regular, snapshot.header.pme_beta,
                            centralShiftId, nullptr, nullptr, d_atom_energy,
                            d_atom_direct_cf_energy, d_atom_lj_energy);
                }
            }
        }
        else if (needVirial)
        {
            if (totalOutput)
            {
                SpongeGmxPackedLjCombKernel<false, true, true><<<grid, block>>>(
                    sci_numbers, d_sci, d_cjpacked, d_excl,
                    d_cluster_local_masks, nullptr, nullptr, nullptr,
                    d_sorted_xq, d_sorted_lj_comb, d_shiftvec,
                    snapshot.header.cutoff, frc_xyz_compact,
                    snapshot.header.pme_beta, centralShiftId, nullptr,
                    d_shift_force, nullptr, nullptr, nullptr);
            }
            else
            {
                if (usePartialReduce)
                {
                    if (useCentralDirectPartialReduce)
                    {
                        SpongeGmxPackedLjCombKernel<false, true, false, true,
                                                    true>
                            <<<grid, block>>>(
                                sci_numbers, d_sci, d_cjpacked, d_excl,
                                d_cluster_local_masks, d_sorted_atom_ids,
                                d_sci_partial_bases, nullptr, d_sorted_xq,
                                d_sorted_lj_comb, d_shiftvec,
                                snapshot.header.cutoff, frc_xyz_regular,
                                snapshot.header.pme_beta, centralShiftId,
                                d_partial_virial, nullptr, nullptr, nullptr,
                                nullptr, d_atom_virial, nullptr, nullptr,
                                nullptr);
                    }
                    else if (partialReduceWriterThreshold >= 0)
                    {
                        SpongeGmxPackedLjCombKernel<false, true, false, true,
                                                    false, true>
                            <<<grid, block>>>(
                                sci_numbers, d_sci, d_cjpacked, d_excl,
                                d_cluster_local_masks, d_sorted_atom_ids,
                                nullptr, d_atom_partial_flags, d_sorted_xq,
                                d_sorted_lj_comb, d_shiftvec,
                                snapshot.header.cutoff, frc_xyz_regular,
                                snapshot.header.pme_beta, centralShiftId,
                                d_partial_virial, nullptr, nullptr, nullptr,
                                nullptr, d_atom_virial, nullptr, nullptr,
                                nullptr);
                    }
                    else
                    {
                        SpongeGmxPackedLjCombKernel<false, true, false, true>
                            <<<grid, block>>>(
                                sci_numbers, d_sci, d_cjpacked, d_excl,
                                d_cluster_local_masks, d_sorted_atom_ids,
                                nullptr, nullptr, d_sorted_xq, d_sorted_lj_comb,
                                d_shiftvec, snapshot.header.cutoff,
                                frc_xyz_regular, snapshot.header.pme_beta,
                                centralShiftId, d_partial_virial, nullptr,
                                nullptr, nullptr, nullptr);
                    }
                }
                else
                {
                    SpongeGmxPackedLjCombKernel<false, true, false>
                        <<<grid, block>>>(
                            sci_numbers, d_sci, d_cjpacked, d_excl,
                            d_cluster_local_masks, d_sorted_atom_ids,
                            nullptr, nullptr, d_sorted_xq, d_sorted_lj_comb,
                            d_shiftvec, snapshot.header.cutoff,
                            frc_xyz_regular, snapshot.header.pme_beta,
                            centralShiftId, d_atom_virial, nullptr, nullptr,
                            nullptr, nullptr);
                }
            }
        }
        else
        {
            SpongeGmxPackedLjCombKernel<false, false, true><<<grid, block>>>(
                sci_numbers, d_sci, d_cjpacked, d_excl, d_cluster_local_masks,
                nullptr, nullptr, nullptr, d_sorted_xq, d_sorted_lj_comb,
                d_shiftvec, snapshot.header.cutoff, frc_xyz_regular,
                snapshot.header.pme_beta, centralShiftId,
                nullptr, nullptr, nullptr, nullptr, nullptr);
        }
        if (usePartialReduce)
        {
            const int reduceThreads = 256;
            if (useSpecializedPartialReduce || partialReduceWriterThreshold >= 0)
            {
                if (needEnergy && needVirial)
                {
                    if (useCentralDirectPartialReduce ||
                        partialReduceWriterThreshold >= 0)
                    {
                        LaunchGmxPackedPartialOutputReduceBuckets<true, true, true>(
                            partialIndex, reduceThreads, d_reduce_atoms_2,
                            d_reduce_slots_2, d_reduce_atoms_4, d_reduce_slots_4,
                            d_reduce_atoms_6, d_reduce_slots_6, d_reduce_atoms_8,
                            d_reduce_slots_8, d_reduce_generic_atoms,
                            d_atom_partial_offsets, d_atom_partial_slots,
                            d_partial_virial, d_partial_energy,
                            d_partial_direct_cf_energy, d_partial_lj_energy,
                            d_atom_virial, d_atom_energy,
                            d_atom_direct_cf_energy, d_atom_lj_energy);
                    }
                    else
                    {
                        LaunchGmxPackedPartialOutputReduceBuckets<true, true,
                                                                  false>(
                            partialIndex, reduceThreads, d_reduce_atoms_2,
                            d_reduce_slots_2, d_reduce_atoms_4, d_reduce_slots_4,
                            d_reduce_atoms_6, d_reduce_slots_6, d_reduce_atoms_8,
                            d_reduce_slots_8, d_reduce_generic_atoms,
                            d_atom_partial_offsets, d_atom_partial_slots,
                            d_partial_virial, d_partial_energy,
                            d_partial_direct_cf_energy, d_partial_lj_energy,
                            d_atom_virial, d_atom_energy,
                            d_atom_direct_cf_energy, d_atom_lj_energy);
                    }
                }
                else if (needEnergy)
                {
                    if (useCentralDirectPartialReduce ||
                        partialReduceWriterThreshold >= 0)
                    {
                        LaunchGmxPackedPartialOutputReduceBuckets<true, false,
                                                                  true>(
                            partialIndex, reduceThreads, d_reduce_atoms_2,
                            d_reduce_slots_2, d_reduce_atoms_4, d_reduce_slots_4,
                            d_reduce_atoms_6, d_reduce_slots_6, d_reduce_atoms_8,
                            d_reduce_slots_8, d_reduce_generic_atoms,
                            d_atom_partial_offsets, d_atom_partial_slots,
                            nullptr, d_partial_energy,
                            d_partial_direct_cf_energy, d_partial_lj_energy,
                            nullptr, d_atom_energy, d_atom_direct_cf_energy,
                            d_atom_lj_energy);
                    }
                    else
                    {
                        LaunchGmxPackedPartialOutputReduceBuckets<true, false,
                                                                  false>(
                            partialIndex, reduceThreads, d_reduce_atoms_2,
                            d_reduce_slots_2, d_reduce_atoms_4, d_reduce_slots_4,
                            d_reduce_atoms_6, d_reduce_slots_6, d_reduce_atoms_8,
                            d_reduce_slots_8, d_reduce_generic_atoms,
                            d_atom_partial_offsets, d_atom_partial_slots,
                            nullptr, d_partial_energy,
                            d_partial_direct_cf_energy, d_partial_lj_energy,
                            nullptr, d_atom_energy, d_atom_direct_cf_energy,
                            d_atom_lj_energy);
                    }
                }
                else if (needVirial)
                {
                    if (useCentralDirectPartialReduce ||
                        partialReduceWriterThreshold >= 0)
                    {
                        LaunchGmxPackedPartialOutputReduceBuckets<false, true,
                                                                  true>(
                            partialIndex, reduceThreads, d_reduce_atoms_2,
                            d_reduce_slots_2, d_reduce_atoms_4, d_reduce_slots_4,
                            d_reduce_atoms_6, d_reduce_slots_6, d_reduce_atoms_8,
                            d_reduce_slots_8, d_reduce_generic_atoms,
                            d_atom_partial_offsets, d_atom_partial_slots,
                            d_partial_virial, nullptr, nullptr, nullptr,
                            d_atom_virial, nullptr, nullptr, nullptr);
                    }
                    else
                    {
                        LaunchGmxPackedPartialOutputReduceBuckets<false, true,
                                                                  false>(
                            partialIndex, reduceThreads, d_reduce_atoms_2,
                            d_reduce_slots_2, d_reduce_atoms_4, d_reduce_slots_4,
                            d_reduce_atoms_6, d_reduce_slots_6, d_reduce_atoms_8,
                            d_reduce_slots_8, d_reduce_generic_atoms,
                            d_atom_partial_offsets, d_atom_partial_slots,
                            d_partial_virial, nullptr, nullptr, nullptr,
                            d_atom_virial, nullptr, nullptr, nullptr);
                    }
                }
            }
            else
            {
                const dim3 reduceBlock(reduceThreads, 1, 1);
                const dim3 reduceGrid(
                    static_cast<unsigned int>(
                        (scalar_output_numbers + reduceThreads - 1) /
                        reduceThreads),
                    1, 1);
                if (needEnergy && needVirial)
                {
                    if (useCentralDirectPartialReduce)
                    {
                        ReduceGmxPackedPartialOutputsKernel<true, true, true>
                            <<<reduceGrid, reduceBlock>>>(
                                scalar_output_numbers, d_atom_partial_offsets,
                                d_atom_partial_slots, d_partial_virial,
                                d_partial_energy, d_partial_direct_cf_energy,
                                d_partial_lj_energy, d_atom_virial,
                                d_atom_energy, d_atom_direct_cf_energy,
                                d_atom_lj_energy);
                    }
                    else
                    {
                        ReduceGmxPackedPartialOutputsKernel<true, true>
                            <<<reduceGrid, reduceBlock>>>(
                                scalar_output_numbers, d_atom_partial_offsets,
                                d_atom_partial_slots, d_partial_virial,
                                d_partial_energy, d_partial_direct_cf_energy,
                                d_partial_lj_energy, d_atom_virial,
                                d_atom_energy, d_atom_direct_cf_energy,
                                d_atom_lj_energy);
                    }
                }
                else if (needEnergy)
                {
                    if (useCentralDirectPartialReduce)
                    {
                        ReduceGmxPackedPartialOutputsKernel<true, false, true>
                            <<<reduceGrid, reduceBlock>>>(
                                scalar_output_numbers, d_atom_partial_offsets,
                                d_atom_partial_slots, nullptr, d_partial_energy,
                                d_partial_direct_cf_energy,
                                d_partial_lj_energy, nullptr, d_atom_energy,
                                d_atom_direct_cf_energy, d_atom_lj_energy);
                    }
                    else
                    {
                        ReduceGmxPackedPartialOutputsKernel<true, false>
                            <<<reduceGrid, reduceBlock>>>(
                                scalar_output_numbers, d_atom_partial_offsets,
                                d_atom_partial_slots, nullptr, d_partial_energy,
                                d_partial_direct_cf_energy,
                                d_partial_lj_energy, nullptr, d_atom_energy,
                                d_atom_direct_cf_energy, d_atom_lj_energy);
                    }
                }
                else if (needVirial)
                {
                    if (useCentralDirectPartialReduce)
                    {
                        ReduceGmxPackedPartialOutputsKernel<false, true, true>
                            <<<reduceGrid, reduceBlock>>>(
                                scalar_output_numbers, d_atom_partial_offsets,
                                d_atom_partial_slots, d_partial_virial, nullptr,
                                nullptr, nullptr, d_atom_virial, nullptr,
                                nullptr, nullptr);
                    }
                    else
                    {
                        ReduceGmxPackedPartialOutputsKernel<false, true>
                            <<<reduceGrid, reduceBlock>>>(
                                scalar_output_numbers, d_atom_partial_offsets,
                                d_atom_partial_slots, d_partial_virial, nullptr,
                                nullptr, nullptr, d_atom_virial, nullptr,
                                nullptr, nullptr);
                    }
                }
            }
        }
    };

    for (int i = 0; i < warmup; ++i)
    {
        launchKernel();
    }
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    cudaEvent_t start, stop;
    CheckCuda(cudaEventCreate(&start), "cudaEventCreate");
    CheckCuda(cudaEventCreate(&stop), "cudaEventCreate");
    CheckCuda(cudaEventRecord(start), "cudaEventRecord");
    for (int i = 0; i < iters; ++i)
    {
        launchKernel();
    }
    CheckCuda(cudaEventRecord(stop), "cudaEventRecord");
    CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float total_ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&total_ms, start, stop),
              "cudaEventElapsedTime");
    std::vector<float3> host_frc_xyz_compact;
    std::vector<float4> host_frc_xyz_regular;
    if (needVirial || validateAgainstReference)
    {
        if (compactForceStorage)
        {
            host_frc_xyz_compact.resize(static_cast<size_t>(total_atom_numbers));
            CheckCuda(cudaMemcpy(host_frc_xyz_compact.data(), frc_xyz_compact,
                                 sizeof(float3) * total_atom_numbers,
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(frc_xyz_compact)");
        }
        else
        {
            host_frc_xyz_regular.resize(static_cast<size_t>(total_atom_numbers));
            CheckCuda(cudaMemcpy(host_frc_xyz_regular.data(), frc_xyz_regular,
                                 sizeof(float4) * total_atom_numbers,
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(frc_xyz_regular)");
        }
    }
    DiffStats force_stats = {};
    DiffStats energy_stats = {};
    DiffStats direct_energy_stats = {};
    DiffStats lj_energy_stats = {};
    DiffStats virial_stats = {};

    LTMatrix3 host_virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float host_atom_energy = 0.0f;
    float host_direct_cf_energy = 0.0f;
    float host_lj_energy = 0.0f;
    std::vector<LTMatrix3> host_atom_virial;
    std::vector<float> host_atom_energy_vec;
    std::vector<float> host_direct_cf_energy_vec;
    std::vector<float> host_lj_energy_vec;
    if (needVirial && totalOutput)
    {
        std::vector<float3> host_shift_force(shiftvec.size());
        CheckCuda(cudaMemcpy(host_shift_force.data(), d_shift_force,
                             sizeof(float3) * shiftvec.size(),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(shift_force)");
        for (size_t i = 0; i < host_shift_force.size(); ++i)
        {
            AccumulateVirialFromForceDis(
                &host_virial,
                Vec3{host_shift_force[i].x, host_shift_force[i].y,
                     host_shift_force[i].z},
                Vec3{shiftvec[i].x, shiftvec[i].y, shiftvec[i].z}, 1.0f);
        }
        for (int i = 0; i < total_atom_numbers; ++i)
        {
            AccumulateVirialFromForceDis(
                &host_virial,
                Vec3{host_frc_xyz_compact[static_cast<size_t>(i)].x,
                     host_frc_xyz_compact[static_cast<size_t>(i)].y,
                     host_frc_xyz_compact[static_cast<size_t>(i)].z},
                Vec3{sorted_xq[static_cast<size_t>(i)].x,
                     sorted_xq[static_cast<size_t>(i)].y,
                     sorted_xq[static_cast<size_t>(i)].z},
                1.0f);
        }
    }
    else if (needVirial)
    {
        host_atom_virial.resize(static_cast<size_t>(scalar_output_numbers));
        CheckCuda(cudaMemcpy(host_atom_virial.data(), d_atom_virial,
                             sizeof(LTMatrix3) * scalar_output_numbers,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(atom_virial)");
    }
    if (needEnergy)
    {
        if (totalOutput)
        {
            CheckCuda(cudaMemcpy(&host_atom_energy, d_atom_energy, sizeof(float),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(atom_energy)");
            CheckCuda(cudaMemcpy(&host_direct_cf_energy, d_atom_direct_cf_energy,
                                 sizeof(float), cudaMemcpyDeviceToHost),
                      "cudaMemcpy(atom_direct_cf_energy)");
            CheckCuda(cudaMemcpy(&host_lj_energy, d_atom_lj_energy,
                                 sizeof(float), cudaMemcpyDeviceToHost),
                      "cudaMemcpy(atom_lj_energy)");
        }
        else
        {
            host_atom_energy_vec.resize(static_cast<size_t>(scalar_output_numbers));
            host_direct_cf_energy_vec.resize(
                static_cast<size_t>(scalar_output_numbers));
            host_lj_energy_vec.resize(static_cast<size_t>(scalar_output_numbers));
            CheckCuda(cudaMemcpy(host_atom_energy_vec.data(), d_atom_energy,
                                 sizeof(float) * scalar_output_numbers,
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(atom_energy)");
            CheckCuda(cudaMemcpy(host_direct_cf_energy_vec.data(),
                                 d_atom_direct_cf_energy,
                                 sizeof(float) * scalar_output_numbers,
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(atom_direct_cf_energy)");
            CheckCuda(cudaMemcpy(host_lj_energy_vec.data(), d_atom_lj_energy,
                                 sizeof(float) * scalar_output_numbers,
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(atom_lj_energy)");
        }
    }
    if (needVirial && !totalOutput)
    {
        host_virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (const LTMatrix3& value : host_atom_virial)
        {
            host_virial.a11 += value.a11;
            host_virial.a21 += value.a21;
            host_virial.a22 += value.a22;
            host_virial.a31 += value.a31;
            host_virial.a32 += value.a32;
            host_virial.a33 += value.a33;
        }
    }
    if (needEnergy && !totalOutput)
    {
        host_atom_energy = 0.0f;
        host_direct_cf_energy = 0.0f;
        host_lj_energy = 0.0f;
        for (float value : host_atom_energy_vec)
        {
            host_atom_energy += value;
        }
        for (float value : host_direct_cf_energy_vec)
        {
            host_direct_cf_energy += value;
        }
        for (float value : host_lj_energy_vec)
        {
            host_lj_energy += value;
        }
    }

    if (validateAgainstReference)
    {
        std::vector<Float4POD> replay_force(
            referenceFullOutputSnapshot->reference_force.size(), Float4POD{});
        for (int sorted_i = 0; sorted_i < total_atom_numbers; ++sorted_i)
        {
            const int atom_i = snapshot.sorted_atom_ids[static_cast<size_t>(sorted_i)];
            if (atom_i < 0 ||
                static_cast<size_t>(atom_i) >= replay_force.size())
            {
                continue;
            }
            replay_force[static_cast<size_t>(atom_i)].x +=
                compactForceStorage
                    ? host_frc_xyz_compact[static_cast<size_t>(sorted_i)].x
                    : host_frc_xyz_regular[static_cast<size_t>(sorted_i)].x;
            replay_force[static_cast<size_t>(atom_i)].y +=
                compactForceStorage
                    ? host_frc_xyz_compact[static_cast<size_t>(sorted_i)].y
                    : host_frc_xyz_regular[static_cast<size_t>(sorted_i)].y;
            replay_force[static_cast<size_t>(atom_i)].z +=
                compactForceStorage
                    ? host_frc_xyz_compact[static_cast<size_t>(sorted_i)].z
                    : host_frc_xyz_regular[static_cast<size_t>(sorted_i)].z;
        }
        force_stats = CompareForceArrays(replay_force,
                                         referenceFullOutputSnapshot->reference_force);
        if (needVirial)
        {
            if (totalOutput)
            {
                host_atom_virial = {host_virial};
            }
            virial_stats = CompareVirialArrays(
                host_atom_virial,
                referenceFullOutputSnapshot->reference_atom_virial);
        }
        if (needEnergy)
        {
            if (totalOutput)
            {
                host_atom_energy_vec = {host_atom_energy};
                host_direct_cf_energy_vec = {host_direct_cf_energy};
                host_lj_energy_vec = {host_lj_energy};
            }
            energy_stats = CompareFloatArrays(
                host_atom_energy_vec,
                referenceFullOutputSnapshot->reference_atom_energy);
            direct_energy_stats = CompareFloatArrays(
                host_direct_cf_energy_vec,
                referenceFullOutputSnapshot->reference_direct_cf_energy);
            lj_energy_stats = CompareFloatArrays(
                host_lj_energy_vec,
                referenceFullOutputSnapshot->reference_lj_energy);
        }
    }
    const bool sane = (!needEnergy ||
                       (!totalOutput ||
                        (std::isfinite(host_atom_energy) &&
                         std::isfinite(host_direct_cf_energy) &&
                         std::isfinite(host_lj_energy)))) &&
                       (!needVirial ||
                       (!totalOutput ||
                        (std::isfinite(host_virial.a11) &&
                         std::isfinite(host_virial.a21) &&
                         std::isfinite(host_virial.a22) &&
                         std::isfinite(host_virial.a31) &&
                         std::isfinite(host_virial.a32) &&
                         std::isfinite(host_virial.a33))));
    if (validateAgainstReference)
    {
        std::printf(
            "kernel=sponge_gmxpacked_ljcomb_fulloutput snapshot=%s avg_ms=%.6f iters=%d sci=%d cjpacked=%zu excl=%zu atoms=%d force_max_abs=%.6e force_rms=%.6e virial_max_abs=%.6e virial_rms=%.6e energy_max_abs=%.6e energy_rms=%.6e direct_energy_max_abs=%.6e lj_energy_max_abs=%.6e total_output=%u partial_reduce=%u central_direct=%u specialized_reduce=%u partial_threshold=%d\n",
            snapshotLabel, total_ms / static_cast<float>(iters), iters,
            sci_numbers, normalizedCjpacked.size(), normalizedExcl.size(),
            total_atom_numbers, force_stats.max_abs, force_stats.rms,
            virial_stats.max_abs, virial_stats.rms, energy_stats.max_abs,
            energy_stats.rms, direct_energy_stats.max_abs,
            lj_energy_stats.max_abs, totalOutput ? 1u : 0u,
            usePartialReduce ? 1u : 0u,
            useCentralDirectPartialReduce ? 1u : 0u,
            useSpecializedPartialReduce ? 1u : 0u,
            partialReduceWriterThreshold);
    }
    else if (needEnergy || needVirial)
    {
        std::printf("kernel=sponge_gmxpacked_ljcomb_totaloutput snapshot=%s avg_ms=%.6f "
                    "iters=%d sci=%d cjpacked=%zu excl=%zu atoms=%d compute_energy=%u compute_virial=%u "
                    "sanity=%s energy=%.6e direct_energy=%.6e lj_energy=%.6e virial_xx=%.6e virial_yy=%.6e virial_zz=%.6e "
                    "total_output=%u partial_reduce=%u central_direct=%u specialized_reduce=%u partial_threshold=%d\n",
                    snapshotLabel, total_ms / static_cast<float>(iters), iters,
                    sci_numbers, normalizedCjpacked.size(), normalizedExcl.size(),
                    total_atom_numbers, snapshot.header.compute_energy,
                    snapshot.header.compute_virial, sane ? "ok" : "bad",
                    host_atom_energy, host_direct_cf_energy, host_lj_energy,
                    host_virial.a11, host_virial.a22, host_virial.a33,
                    totalOutput ? 1u : 0u, usePartialReduce ? 1u : 0u,
                    useCentralDirectPartialReduce ? 1u : 0u,
                    useSpecializedPartialReduce ? 1u : 0u,
                    partialReduceWriterThreshold);
    }
    else
    {
        std::printf("kernel=sponge_gmxpacked_ljcomb snapshot=%s avg_ms=%.6f "
                    "iters=%d sci=%d cjpacked=%zu excl=%zu atoms=%d\n",
                    snapshotLabel, total_ms / static_cast<float>(iters), iters,
                    sci_numbers, normalizedCjpacked.size(), normalizedExcl.size(),
                    total_atom_numbers);
    }

    CheckCuda(cudaFree(d_partial_lj_energy), "cudaFree(partial_lj_energy)");
    CheckCuda(cudaFree(d_partial_direct_cf_energy),
              "cudaFree(partial_direct_cf_energy)");
    CheckCuda(cudaFree(d_partial_energy), "cudaFree(partial_energy)");
    CheckCuda(cudaFree(d_partial_virial), "cudaFree(partial_virial)");
    CheckCuda(cudaFree(d_reduce_generic_atoms), "cudaFree(reduce_generic_atoms)");
    CheckCuda(cudaFree(d_reduce_slots_8), "cudaFree(reduce_slots_8)");
    CheckCuda(cudaFree(d_reduce_atoms_8), "cudaFree(reduce_atoms_8)");
    CheckCuda(cudaFree(d_reduce_slots_6), "cudaFree(reduce_slots_6)");
    CheckCuda(cudaFree(d_reduce_atoms_6), "cudaFree(reduce_atoms_6)");
    CheckCuda(cudaFree(d_reduce_slots_4), "cudaFree(reduce_slots_4)");
    CheckCuda(cudaFree(d_reduce_atoms_4), "cudaFree(reduce_atoms_4)");
    CheckCuda(cudaFree(d_reduce_slots_2), "cudaFree(reduce_slots_2)");
    CheckCuda(cudaFree(d_reduce_atoms_2), "cudaFree(reduce_atoms_2)");
    CheckCuda(cudaFree(d_atom_partial_slots), "cudaFree(atom_partial_slots)");
    CheckCuda(cudaFree(d_atom_partial_offsets), "cudaFree(atom_partial_offsets)");
    CheckCuda(cudaFree(d_atom_partial_flags), "cudaFree(atom_partial_flags)");
    CheckCuda(cudaFree(d_sci_partial_bases), "cudaFree(sci_partial_bases)");
    CheckCuda(cudaFree(d_atom_lj_energy), "cudaFree(atom_lj_energy)");
    CheckCuda(cudaFree(d_atom_direct_cf_energy), "cudaFree(atom_direct_cf_energy)");
    CheckCuda(cudaFree(d_atom_energy), "cudaFree(atom_energy)");
    CheckCuda(cudaFree(d_atom_virial), "cudaFree(atom_virial)");
    CheckCuda(cudaFree(d_shift_force), "cudaFree(shift_force)");
    CheckCuda(cudaFree(d_cluster_local_masks), "cudaFree(cluster_local_masks)");
    CheckCuda(cudaFree(d_sorted_atom_ids), "cudaFree(sorted_atom_ids)");
    CheckCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    CheckCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    CheckCuda(cudaFree(d_shiftvec), "cudaFree(shiftvec)");
    CheckCuda(cudaFree(d_sorted_lj_comb), "cudaFree(sorted_lj_comb)");
    CheckCuda(cudaFree(d_sorted_xq), "cudaFree(sorted_xq)");
    CheckCuda(cudaFree(d_excl), "cudaFree(excl)");
    CheckCuda(cudaFree(d_cjpacked), "cudaFree(cjpacked)");
    CheckCuda(cudaFree(d_sci), "cudaFree(sci)");
    CheckCuda(cudaFree(frc_xyz_regular), "cudaFree(frc_xyz_regular)");
    CheckCuda(cudaFree(frc_xyz_compact), "cudaFree(frc_xyz_compact)");
}

void RunGromacs(const GromacsPairlistSnapshot& snapshot, int warmup, int iters)
{
    RunGromacsProduction(snapshot, warmup, iters, "native");
}

void RunSnapshotRecordBuilderMicrobench(
    const SpongeForceOnlySnapshot& snapshot,
    SpongeGmxTransformMode transformMode,
    double exactImaskRadiusScale,
    int warmup,
    int iters,
    const char* snapshotLabel)
{
    size_t builderSink = 0;
    GromacsPairlistSnapshot builderWorkspace;
    for (int i = 0; i < warmup; ++i)
    {
        SnapshotRecordBuilderStats stats;
        ConvertSpongeSnapshotToGromacs(snapshot, transformMode, nullptr,
                                       exactImaskRadiusScale, &stats,
                                       &builderWorkspace);
        builderSink = builderSink * 1315423911u + builderWorkspace.sci.size() +
                      builderWorkspace.cjpacked.size() +
                      builderWorkspace.excl.size() +
                      stats.aggregateRows;
    }

    SnapshotRecordBuilderStats finalStats;
    double totalMs = 0.0;
    for (int i = 0; i < iters; ++i)
    {
        SnapshotRecordBuilderStats stats;
        const auto start = std::chrono::steady_clock::now();
        ConvertSpongeSnapshotToGromacs(snapshot, transformMode, nullptr,
                                       exactImaskRadiusScale, &stats,
                                       &builderWorkspace);
        const auto stop = std::chrono::steady_clock::now();
        totalMs += std::chrono::duration<double, std::milli>(stop - start).count();
        builderSink = builderSink * 1315423911u + builderWorkspace.sci.size() +
                      builderWorkspace.cjpacked.size() +
                      builderWorkspace.excl.size() +
                      stats.aggregateRows;
        finalStats = stats;
    }

    if (builderSink == std::numeric_limits<size_t>::max())
    {
        std::fprintf(stderr, "builder sink overflowed unexpectedly\n");
        std::exit(1);
    }

    std::printf(
        "kernel=builder snapshot=%s avg_ms=%.6f iters=%d records=%zu aggregates=%zu sci=%zu cjpacked=%zu excl=%zu\n",
        snapshotLabel, totalMs / static_cast<double>(iters), iters,
        finalStats.sourceRecords, finalStats.aggregateRows,
        finalStats.compactSci, finalStats.compactCjPacked,
        finalStats.compactExcl);
}

} // namespace

int main(int argc, char** argv)
{
    const Arguments args = ParseArguments(argc, argv);
    const SpongeGmxTransformMode spongeGmxTransform =
        ParseSpongeGmxTransform(args.spongeGmxTransform);
    const bool combGmxPackedPerAtomMode =
        args.spongeLjMode == "comb-gmxpacked-peratom";
    const bool combGmxPackedPartialReduceMode =
        args.spongeLjMode == "comb-gmxpacked-partial-reduce";
    const bool combGmxPackedPartialReduceSpecializedMode =
        args.spongeLjMode == "comb-gmxpacked-partial-reduce-specialized";
    const bool combGmxPackedPartialReduceThreshold4Mode =
        args.spongeLjMode == "comb-gmxpacked-partial-reduce-threshold4";
    const bool combGmxPackedCentralDirectPartialReduceMode =
        args.spongeLjMode == "comb-gmxpacked-central-direct-partial-reduce";
    const int combGmxPackedPartialReduceThreshold =
        combGmxPackedPartialReduceThreshold4Mode ? 4 : -1;
    const bool combGmxPackedAnyPartialReduceMode =
        combGmxPackedPartialReduceMode ||
        combGmxPackedPartialReduceSpecializedMode ||
        combGmxPackedPartialReduceThreshold4Mode ||
        combGmxPackedCentralDirectPartialReduceMode;
    const bool combGmxPackedReplayMode =
        args.spongeLjMode == "comb-gmxpacked" || combGmxPackedPerAtomMode ||
        combGmxPackedAnyPartialReduceMode;
    if (args.kernel == "builder")
    {
        SpongeClusteredFullOutputSnapshot fullOutputSnapshot;
        if (nbnxm_microbench::ReadSpongeClusteredFullOutputSnapshot(
                args.snapshot, &fullOutputSnapshot))
        {
            const SpongeForceOnlySnapshot payload =
                ExtractPayloadFromFullOutputSnapshot(fullOutputSnapshot);
            if (args.analyze)
            {
                AnalyzeSpongeSnapshot(payload);
                if (args.spongeLjMode == "comb-gmxpacked")
                {
                    GromacsPairlistSnapshot converted =
                        ConvertSpongeSnapshotToGromacs(
                            payload, spongeGmxTransform, nullptr,
                            args.exactImaskRadiusScale);
                    AnalyzeGromacsPerAtomOutputWriters(converted);
                }
                return 0;
            }
            RunSnapshotRecordBuilderMicrobench(
                payload, spongeGmxTransform, args.exactImaskRadiusScale,
                args.warmup, args.iters, args.snapshot.c_str());
            return 0;
        }

        SpongeForceOnlySnapshot snapshot;
        if (nbnxm_microbench::ReadSpongeForceOnlySnapshot(args.snapshot,
                                                          &snapshot))
        {
            if (args.analyze)
            {
                AnalyzeSpongeSnapshot(snapshot);
                if (args.spongeLjMode == "comb-gmxpacked")
                {
                    const GromacsPairlistSnapshot converted =
                        ConvertSpongeSnapshotToGromacs(
                            snapshot, spongeGmxTransform, nullptr,
                            args.exactImaskRadiusScale);
                    AnalyzeGromacsPerAtomOutputWriters(converted);
                }
                return 0;
            }
            RunSnapshotRecordBuilderMicrobench(
                snapshot, spongeGmxTransform, args.exactImaskRadiusScale,
                args.warmup, args.iters, args.snapshot.c_str());
            return 0;
        }

        std::fprintf(
            stderr,
            "builder kernel requires a SPONGE force-only or full-output snapshot: %s\n",
            args.snapshot.c_str());
        return 1;
    }
    if (args.kernel == "sponge")
    {
        SpongeGmxpackedFullOutputSnapshot gmxpackedFullOutputSnapshot;
        if (nbnxm_microbench::ReadSpongeGmxpackedFullOutputSnapshot(
                args.snapshot, &gmxpackedFullOutputSnapshot))
        {
            if (args.pairOracle)
            {
                const nbnxm_microbench::CanonicalPairOracleResult result =
                    nbnxm_microbench::CompareCanonicalPairs(
                        gmxpackedFullOutputSnapshot.payload);
                std::printf(
                    "canonical_pair_oracle metadata_ready=%d matched=%d "
                    "payload=%zu oracle=%zu duplicates=%zu missing=%zu "
                    "extra=%zu\n",
                    result.metadata_ready ? 1 : 0, result.matched ? 1 : 0,
                    result.payload_pair_count, result.oracle_pair_count,
                    result.duplicate_payload_pairs, result.missing_pairs,
                    result.extra_pairs);
                if (!result.failure_reason.empty())
                {
                    std::fprintf(stderr, "pair oracle failure: %s\n",
                                 result.failure_reason.c_str());
                }
                return result.matched ? 0 : 2;
            }
            if (args.analyze)
            {
                AnalyzeSpongeProductionGmxpackedSnapshot(
                    gmxpackedFullOutputSnapshot.payload);
                return 0;
            }
            if (args.spongeLjMode == "fulloutput" ||
                args.spongeLjMode == "production-gmxpacked" ||
                args.spongeLjMode ==
                    "production-gmxpacked-full-sci-split2" ||
                args.spongeLjMode ==
                    "production-gmxpacked-full-sci-split4" ||
                args.spongeLjMode ==
                    "production-gmxpacked-full-sci-split8")
            {
                const ProductionGmxpackedReplayMode replay_mode =
                    args.spongeLjMode ==
                            "production-gmxpacked-full-sci-split8"
                        ? ProductionGmxpackedReplayMode::fullOutputSciSplit8
                        : (args.spongeLjMode ==
                                   "production-gmxpacked-full-sci-split4"
                               ? ProductionGmxpackedReplayMode::
                                     fullOutputSciSplit4
                               : (args.spongeLjMode ==
                                          "production-gmxpacked-full-sci-split2"
                                      ? ProductionGmxpackedReplayMode::
                                            fullOutputSciSplit2
                                      : ProductionGmxpackedReplayMode::split));
                RunSpongeProductionGmxpacked(
                    gmxpackedFullOutputSnapshot.payload, args.warmup,
                    args.iters,
                    gmxpackedFullOutputSnapshot.header.compute_energy != 0u,
                    gmxpackedFullOutputSnapshot.header.compute_virial != 0u,
                    "native-gmxpacked-fulloutput",
                    replay_mode,
                    &gmxpackedFullOutputSnapshot);
                return 0;
            }
            std::fprintf(
                stderr,
                "gmxpacked full-output snapshots support sponge-lj-mode "
                "fulloutput|production-gmxpacked|"
                "production-gmxpacked-full-sci-split2|"
                "production-gmxpacked-full-sci-split4|"
                "production-gmxpacked-full-sci-split8 only, got %s\n",
                args.spongeLjMode.c_str());
            return 1;
        }

        SpongeGmxpackedForceOnlySnapshot productionGmxpackedSnapshot;
        if (nbnxm_microbench::ReadSpongeGmxpackedForceOnlySnapshot(
                args.snapshot, &productionGmxpackedSnapshot))
        {
            if (args.pairOracle)
            {
                const nbnxm_microbench::CanonicalPairOracleResult result =
                    nbnxm_microbench::CompareCanonicalPairs(
                        productionGmxpackedSnapshot);
                std::printf(
                    "canonical_pair_oracle metadata_ready=%d matched=%d "
                    "payload=%zu oracle=%zu duplicates=%zu missing=%zu "
                    "extra=%zu\n",
                    result.metadata_ready ? 1 : 0, result.matched ? 1 : 0,
                    result.payload_pair_count, result.oracle_pair_count,
                    result.duplicate_payload_pairs, result.missing_pairs,
                    result.extra_pairs);
                if (!result.failure_reason.empty())
                {
                    std::fprintf(stderr, "pair oracle failure: %s\n",
                                 result.failure_reason.c_str());
                }
                for (const auto& pair : result.first_duplicates)
                {
                    std::fprintf(stderr, "duplicate=(%d,%d,shift=%d)\n",
                                 pair.global_i, pair.global_j,
                                 pair.shift_id);
                }
                for (const auto& occurrence :
                     result.first_duplicate_occurrences)
                {
                    std::fprintf(
                        stderr,
                        "duplicate_source=(%d,%d,shift=%d sci=%zu "
                        "packed=%zu split=%d jm=%d i_local=%d "
                        "cluster_i=%d cluster_j=%d sci_shift=%d "
                        "pair_shift=%d exclusion=%d imask=0x%08x "
                        "exclusion_hash=0x%016llx)\n",
                        occurrence.pair.global_i,
                        occurrence.pair.global_j,
                        occurrence.pair.shift_id,
                        occurrence.sci_index,
                        occurrence.packed_index,
                        occurrence.split,
                        occurrence.jm,
                        occurrence.i_local,
                        occurrence.cluster_i,
                        occurrence.cluster_j,
                        occurrence.sci_shift_id,
                        occurrence.pair_shift_id,
                        occurrence.exclusion_index,
                        occurrence.imask,
                        static_cast<unsigned long long>(
                            occurrence.exclusion_hash));
                }
                for (const auto& source :
                     result.duplicate_source_summaries)
                {
                    std::fprintf(
                        stderr,
                        "duplicate_source_summary=(sci=%zu packed=%zu "
                        "split=%d jm=%d i_local=%d cluster_i=%d "
                        "cluster_j=%d sci_shift=%d pair_shift=%d "
                        "exclusion=%d imask=0x%08x "
                        "exclusion_hash=0x%016llx accepted=%zu "
                        "duplicate=%zu)\n",
                        source.sci_index, source.packed_index,
                        source.split, source.jm, source.i_local,
                        source.cluster_i, source.cluster_j,
                        source.sci_shift_id, source.pair_shift_id,
                        source.exclusion_index, source.imask,
                        static_cast<unsigned long long>(
                            source.exclusion_hash),
                        source.accepted_pairs, source.duplicate_pairs);
                }
                for (const auto& pair : result.first_missing)
                {
                    std::fprintf(stderr, "missing=(%d,%d,shift=%d)\n",
                                 pair.global_i, pair.global_j,
                                 pair.shift_id);
                }
                for (const auto& pair : result.first_extra)
                {
                    std::fprintf(stderr, "extra=(%d,%d,shift=%d)\n",
                                 pair.global_i, pair.global_j,
                                 pair.shift_id);
                }
                return result.matched ? 0 : 2;
            }
            if (args.analyze)
            {
                AnalyzeSpongeProductionGmxpackedSnapshot(
                    productionGmxpackedSnapshot);
                const GromacsPairlistSnapshot converted =
                    ConvertSpongeGmxpackedSnapshotToGromacs(
                        productionGmxpackedSnapshot, args.computeEnergy,
                        args.computeVirial);
                AnalyzeGromacsPerAtomOutputWriters(converted);
                return 0;
            }
            if (args.spongeLjMode == "production-gmxpacked")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked");
                return 0;
            }
            if (args.spongeLjMode ==
                "production-gmxpacked-fused-sits-force-only")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-fused-sits-force-only",
                    ProductionGmxpackedReplayMode::fusedSitsForceOnly,
                    nullptr, args.sitsAtomEnd, args.sitsPwwpFactor);
                return 0;
            }
            if (args.spongeLjMode ==
                "production-gmxpacked-sparse-sits-force-only")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-sparse-sits-force-only",
                    ProductionGmxpackedReplayMode::sparseSitsForceOnly,
                    nullptr, args.sitsAtomEnd, args.sitsPwwpFactor);
                return 0;
            }
            if (args.spongeLjMode == "production-gmxpacked-compact-force")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-compact-force",
                    ProductionGmxpackedReplayMode::compactForce);
                return 0;
            }
            if (args.spongeLjMode == "production-gmxpacked-sorted-force")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-sorted-force",
                    ProductionGmxpackedReplayMode::sortedForce);
                return 0;
            }
            if (args.spongeLjMode ==
                "production-gmxpacked-sorted-force-sci-split2")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-sorted-force-sci-split2",
                    ProductionGmxpackedReplayMode::sortedForceSciSplit2);
                return 0;
            }
            if (args.spongeLjMode ==
                "production-gmxpacked-sorted-force-sci-split3")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-sorted-force-sci-split3",
                    ProductionGmxpackedReplayMode::sortedForceSciSplit3);
                return 0;
            }
            if (args.spongeLjMode ==
                "production-gmxpacked-sorted-force-sci-split4")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-sorted-force-sci-split4",
                    ProductionGmxpackedReplayMode::sortedForceSciSplit4);
                return 0;
            }
            if (args.spongeLjMode ==
                "production-gmxpacked-sorted-force-local-i-mask8")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-sorted-force-local-i-mask8",
                    ProductionGmxpackedReplayMode::sortedForceLocalIMask8);
                return 0;
            }
            if (args.spongeLjMode ==
                "production-gmxpacked-sorted-force-active-i-mask8")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-sorted-force-active-i-mask8",
                    ProductionGmxpackedReplayMode::sortedForceActiveIMask8);
                return 0;
            }
            if (args.spongeLjMode ==
                "production-gmxpacked-sorted-force-oracle-imask")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-sorted-force-oracle-imask",
                    ProductionGmxpackedReplayMode::sortedForceOracleImask);
                return 0;
            }
            if (args.spongeLjMode ==
                "production-gmxpacked-sorted-force-oracle-sidecar")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-sorted-force-oracle-sidecar",
                    ProductionGmxpackedReplayMode::sortedForceOracleSidecar);
                return 0;
            }
            if (args.spongeLjMode ==
                "production-gmxpacked-sorted-force-device-sidecar")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-sorted-force-device-sidecar",
                    ProductionGmxpackedReplayMode::sortedForceDeviceSidecar);
                return 0;
            }
            if (args.spongeLjMode ==
                "production-gmxpacked-sorted-force-dense-noexcl")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-sorted-force-dense-noexcl",
                    ProductionGmxpackedReplayMode::sortedForceDenseNoExcl);
                return 0;
            }
            if (args.spongeLjMode ==
                "production-gmxpacked-sorted-force-attr-all-i")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-sorted-force-attr-all-i",
                    ProductionGmxpackedReplayMode::sortedForceAttrAllI);
                return 0;
            }
            if (args.spongeLjMode ==
                "production-gmxpacked-sorted-force-attr-no-cutoff")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-sorted-force-attr-no-cutoff",
                    ProductionGmxpackedReplayMode::sortedForceAttrNoCutoff);
                return 0;
            }
            if (args.spongeLjMode ==
                "production-gmxpacked-sorted-force-attr-all-i-no-cutoff")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-sorted-force-attr-all-i-no-cutoff",
                    ProductionGmxpackedReplayMode::sortedForceAttrAllINoCutoff);
                return 0;
            }
            if (args.spongeLjMode ==
                "production-gmxpacked-sorted-force-no-atomic")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-sorted-force-no-atomic",
                    ProductionGmxpackedReplayMode::sortedForceNoWriteback);
                return 0;
            }
            if (args.spongeLjMode == "production-gmxpacked-safe-only")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-safe-only",
                    ProductionGmxpackedReplayMode::safeOnly);
                return 0;
            }
            if (args.spongeLjMode == "production-gmxpacked-specialized")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-specialized",
                    ProductionGmxpackedReplayMode::specializedSafe);
                return 0;
            }
            if (args.spongeLjMode ==
                "production-gmxpacked-specialized-sorted-force")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-specialized-sorted-force",
                    ProductionGmxpackedReplayMode::specializedSortedForce);
                return 0;
            }
            if (args.spongeLjMode == "production-gmxpacked-specialized-shiftvec")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-specialized-shiftvec",
                    ProductionGmxpackedReplayMode::specializedShiftvec);
                return 0;
            }
            if (args.spongeLjMode == "production-gmxpacked-shift-virial")
            {
                RunSpongeProductionGmxpacked(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.computeEnergy, args.computeVirial,
                    "native-production-gmxpacked-shift-virial",
                    ProductionGmxpackedReplayMode::shiftVirial);
                return 0;
            }
            if (args.spongeLjMode == "production-gmxpacked-refresh")
            {
                RunSpongeProductionGmxpackedRefresh(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    "native-production-gmxpacked-refresh",
                    args.refreshBlockSize);
                return 0;
            }
            if (args.spongeLjMode ==
                "production-gmxpacked-refresh-rootchild-queue2")
            {
                RunSpongeProductionGmxpackedRefreshRootChildQueue2(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    "native-production-gmxpacked-refresh-rootchild-queue2",
                    args.refreshBlockSize);
                return 0;
            }
            if (args.spongeLjMode ==
                "production-gmxpacked-record-stream-inner-active")
            {
                RunSpongeProductionGmxpackedRecordStreamInnerActive(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    "native-production-gmxpacked-record-stream-inner-active");
                return 0;
            }
            if (args.spongeLjMode == "production-gmxpacked-collect-traversal" ||
                args.spongeLjMode == "production-gmxpacked-collect-screen" ||
                args.spongeLjMode == "production-gmxpacked-collect-emit" ||
                args.spongeLjMode ==
                    "production-gmxpacked-collect-screen-active" ||
                args.spongeLjMode ==
                    "production-gmxpacked-collect-screen-active-rootchild" ||
                args.spongeLjMode ==
                    "production-gmxpacked-collect-screen-rootchild-queue" ||
                args.spongeLjMode ==
                    "production-gmxpacked-collect-screen-rootchild-queue2" ||
                args.spongeLjMode ==
                    "production-gmxpacked-collect-screen-rootchild-queue2-devicecounter" ||
                args.spongeLjMode ==
                    "production-gmxpacked-collect-coop-traversal" ||
                args.spongeLjMode == "production-gmxpacked-collect-coop-screen" ||
                args.spongeLjMode == "production-gmxpacked-collect-coop-emit" ||
                args.spongeLjMode ==
                    "production-gmxpacked-collect-screen-stats" ||
                args.spongeLjMode ==
                    "production-gmxpacked-collect-coop-screen-stats")
            {
                const bool cooperative =
                    args.spongeLjMode.find("collect-coop-") !=
                    std::string::npos;
                const bool collectStats =
                    args.spongeLjMode.find("-stats") != std::string::npos;
                const bool activeCandidateList =
                    args.spongeLjMode.find("-active") != std::string::npos;
                const bool rootChildSplit =
                    args.spongeLjMode.find("-rootchild") != std::string::npos;
                const bool rootChildTaskQueue =
                    args.spongeLjMode.find("-rootchild-queue") !=
                    std::string::npos;
                const bool rootChildDeviceCounter =
                    args.spongeLjMode.find("-devicecounter") !=
                    std::string::npos;
                const int rootChildTaskSplitDepth =
                    args.spongeLjMode.find("-rootchild-queue2") !=
                            std::string::npos
                        ? 2
                        : 1;
                const ClusteredGmxpackedCandidateLeafProbeMode collectMode =
                    args.spongeLjMode.find("collect-traversal") !=
                            std::string::npos ||
                            args.spongeLjMode.find("collect-coop-traversal") !=
                                std::string::npos
                        ? ClusteredGmxpackedCandidateLeafProbeMode::Traversal
                        : (args.spongeLjMode.find("collect-screen") !=
                                   std::string::npos ||
                                   args.spongeLjMode.find(
                                       "collect-coop-screen") !=
                                   std::string::npos
                               ? ClusteredGmxpackedCandidateLeafProbeMode::Screen
                               : ClusteredGmxpackedCandidateLeafProbeMode::Emit);
                if (rootChildSplit &&
                    (collectMode !=
                         ClusteredGmxpackedCandidateLeafProbeMode::Screen ||
                     cooperative || collectStats))
                {
                    std::fprintf(stderr,
                                 "rootchild collect replay currently supports "
                                 "screen active mode only\n");
                    return 1;
                }
                RunSpongeProductionGmxpackedCollect(
                    productionGmxpackedSnapshot, args.warmup, args.iters,
                    args.spongeLjMode.c_str(), collectMode, cooperative,
                    collectStats, activeCandidateList, rootChildSplit,
                    rootChildTaskQueue, rootChildTaskSplitDepth,
                    rootChildDeviceCounter);
                return 0;
            }
            if (combGmxPackedReplayMode)
            {
                GromacsPairlistSnapshot converted =
                    ConvertSpongeGmxpackedSnapshotToGromacs(
                        productionGmxpackedSnapshot, args.computeEnergy,
                        args.computeVirial);
                if ((combGmxPackedPerAtomMode ||
                     combGmxPackedAnyPartialReduceMode) &&
                    converted.header.compute_energy == 0u &&
                    converted.header.compute_virial == 0u)
                {
                    std::fprintf(stderr,
                                 "%s requires --compute-energy and/or --compute-virial\n",
                                 args.spongeLjMode.c_str());
                    return 1;
                }
                RunSpongeWithGmxPackedLjComb(
                    converted, args.warmup, args.iters,
                    combGmxPackedCentralDirectPartialReduceMode
                        ? "production-gmxpacked-converted-comb-gmxpacked-central-direct-partial-reduce"
                        : (combGmxPackedPartialReduceSpecializedMode
                               ? "production-gmxpacked-converted-comb-gmxpacked-partial-reduce-specialized"
                               : (combGmxPackedPartialReduceThreshold4Mode
                                      ? "production-gmxpacked-converted-comb-gmxpacked-partial-reduce-threshold4"
                               : (combGmxPackedPartialReduceMode
                               ? "production-gmxpacked-converted-comb-gmxpacked-partial-reduce"
                               : (combGmxPackedPerAtomMode
                                      ? "production-gmxpacked-converted-comb-gmxpacked-peratom"
                                      : "production-gmxpacked-converted-comb-gmxpacked")))),
                    nullptr,
                    combGmxPackedPerAtomMode ||
                        combGmxPackedAnyPartialReduceMode,
                    combGmxPackedAnyPartialReduceMode,
                    combGmxPackedCentralDirectPartialReduceMode,
                    combGmxPackedPartialReduceSpecializedMode,
                    combGmxPackedPartialReduceThreshold);
                return 0;
            }
            std::fprintf(stderr,
                         "production gmxpacked snapshots support "
                         "sponge-lj-mode production-gmxpacked|"
                         "production-gmxpacked-compact-force|"
                         "production-gmxpacked-sorted-force|"
                         "production-gmxpacked-sorted-force-sci-split2|"
                         "production-gmxpacked-sorted-force-sci-split3|"
                         "production-gmxpacked-sorted-force-sci-split4|"
                         "production-gmxpacked-safe-only|"
                         "production-gmxpacked-specialized|"
                         "production-gmxpacked-specialized-sorted-force|"
                         "production-gmxpacked-specialized-shiftvec|"
                         "production-gmxpacked-shift-virial|"
                         "production-gmxpacked-refresh|"
                         "production-gmxpacked-refresh-rootchild-queue2|"
                         "production-gmxpacked-record-stream-inner-active|"
                         "production-gmxpacked-collect-traversal|"
                         "production-gmxpacked-collect-screen|"
                         "production-gmxpacked-collect-screen-active|"
                         "production-gmxpacked-collect-screen-active-rootchild|"
                         "production-gmxpacked-collect-screen-rootchild-queue|"
                         "production-gmxpacked-collect-screen-rootchild-queue2|"
                         "production-gmxpacked-collect-screen-rootchild-queue2-devicecounter|"
                         "production-gmxpacked-collect-emit|"
                         "production-gmxpacked-collect-coop-traversal|"
                         "production-gmxpacked-collect-coop-screen|"
                         "production-gmxpacked-collect-coop-emit|"
                         "production-gmxpacked-collect-screen-stats|"
                         "production-gmxpacked-collect-coop-screen-stats|"
                         "comb-gmxpacked|comb-gmxpacked-peratom|"
                         "comb-gmxpacked-partial-reduce|"
                         "comb-gmxpacked-partial-reduce-specialized|"
                         "comb-gmxpacked-partial-reduce-threshold4|"
                         "comb-gmxpacked-central-direct-partial-reduce only, got %s\n",
                         args.spongeLjMode.c_str());
            return 1;
        }
        if (args.pairOracle)
        {
            std::fprintf(
                stderr,
                "--pair-oracle requires a SPONGE gmxpacked snapshot: %s\n",
                args.snapshot.c_str());
            return 1;
        }
        SpongeClusteredFullOutputSnapshot fullOutputSnapshot;
        if (nbnxm_microbench::ReadSpongeClusteredFullOutputSnapshot(
                args.snapshot, &fullOutputSnapshot))
        {
            const SpongeForceOnlySnapshot payload =
                ExtractPayloadFromFullOutputSnapshot(fullOutputSnapshot);
            if (args.analyze)
            {
                AnalyzeSpongeSnapshot(payload);
                return 0;
            }
            if (args.spongeLjMode == "fulloutput")
            {
                RunSpongeFullOutput(fullOutputSnapshot, args.warmup,
                                    args.iters, "native-fulloutput");
                return 0;
            }
            if (combGmxPackedReplayMode)
            {
                GromacsPairlistSnapshot converted =
                    ConvertSpongeSnapshotToGromacs(
                        payload, spongeGmxTransform, nullptr,
                        args.exactImaskRadiusScale);
                converted.header.compute_energy =
                    fullOutputSnapshot.header.compute_energy;
                converted.header.compute_virial =
                    fullOutputSnapshot.header.compute_virial;
                const SpongeClusteredFullOutputSnapshot* fullOutputReference =
                    (fullOutputSnapshot.header.compute_energy != 0u &&
                     fullOutputSnapshot.header.compute_virial == 0u)
                        ? nullptr
                        : &fullOutputSnapshot;
                RunSpongeWithGmxPackedLjComb(converted, args.warmup,
                                             args.iters,
                                             combGmxPackedCentralDirectPartialReduceMode
                                                 ? "fulloutput-payload-comb-gmxpacked-central-direct-partial-reduce"
                                                 : (combGmxPackedPartialReduceSpecializedMode
                                                        ? "fulloutput-payload-comb-gmxpacked-partial-reduce-specialized"
                                                        : (combGmxPackedPartialReduceThreshold4Mode
                                                               ? "fulloutput-payload-comb-gmxpacked-partial-reduce-threshold4"
                                                        : (combGmxPackedPartialReduceMode
                                                        ? "fulloutput-payload-comb-gmxpacked-partial-reduce"
                                                        : (combGmxPackedPerAtomMode
                                                               ? "fulloutput-payload-comb-gmxpacked-peratom"
                                                               : "fulloutput-payload-comb-gmxpacked")))),
                                             fullOutputReference,
                                             combGmxPackedPerAtomMode ||
                                                 combGmxPackedAnyPartialReduceMode,
                                             combGmxPackedAnyPartialReduceMode,
                                             combGmxPackedCentralDirectPartialReduceMode,
                                             combGmxPackedPartialReduceSpecializedMode,
                                             combGmxPackedPartialReduceThreshold);
                return 0;
            }
            std::fprintf(stderr,
                         "full-output snapshot currently supports sponge-lj-mode fulloutput|comb-gmxpacked|comb-gmxpacked-peratom|comb-gmxpacked-partial-reduce|comb-gmxpacked-partial-reduce-specialized|comb-gmxpacked-partial-reduce-threshold4|comb-gmxpacked-central-direct-partial-reduce, got %s\n",
                         args.spongeLjMode.c_str());
            return 1;
        }
        SpongeForceOnlySnapshot snapshot;
        if (nbnxm_microbench::ReadSpongeForceOnlySnapshot(args.snapshot,
                                                          &snapshot))
        {
            if (args.analyze)
            {
                AnalyzeSpongeSnapshot(snapshot);
                return 0;
            }
            if (combGmxPackedReplayMode)
            {
                GromacsPairlistSnapshot converted =
                    ConvertSpongeSnapshotToGromacs(
                        snapshot, spongeGmxTransform, nullptr,
                        args.exactImaskRadiusScale);
                if (combGmxPackedPerAtomMode || combGmxPackedAnyPartialReduceMode)
                {
                    converted.header.compute_energy =
                        args.computeEnergy ? 1u : 0u;
                    converted.header.compute_virial =
                        args.computeVirial ? 1u : 0u;
                    if (converted.header.compute_energy == 0u &&
                        converted.header.compute_virial == 0u)
                    {
                        std::fprintf(stderr,
                                     "%s requires --compute-energy and/or --compute-virial\n",
                                     args.spongeLjMode.c_str());
                        return 1;
                    }
                }
                RunSpongeWithGmxPackedLjComb(
                    converted, args.warmup, args.iters,
                    combGmxPackedCentralDirectPartialReduceMode
                        ? "native-comb-gmxpacked-central-direct-partial-reduce"
                        : (combGmxPackedPartialReduceSpecializedMode
                               ? "native-comb-gmxpacked-partial-reduce-specialized"
                               : (combGmxPackedPartialReduceThreshold4Mode
                                      ? "native-comb-gmxpacked-partial-reduce-threshold4"
                               : (combGmxPackedPartialReduceMode
                               ? "native-comb-gmxpacked-partial-reduce"
                               : (combGmxPackedPerAtomMode
                                      ? "native-comb-gmxpacked-peratom"
                                      : "native-comb-gmxpacked")))),
                    nullptr,
                    combGmxPackedPerAtomMode ||
                        combGmxPackedAnyPartialReduceMode,
                    combGmxPackedAnyPartialReduceMode,
                    combGmxPackedCentralDirectPartialReduceMode,
                    combGmxPackedPartialReduceSpecializedMode,
                    combGmxPackedPartialReduceThreshold);
            }
            else
            {
                std::fprintf(stderr,
                             "force-only snapshots currently support sponge-lj-mode comb-gmxpacked|comb-gmxpacked-peratom|comb-gmxpacked-partial-reduce|comb-gmxpacked-partial-reduce-specialized|comb-gmxpacked-partial-reduce-threshold4|comb-gmxpacked-central-direct-partial-reduce only, got %s\n",
                             args.spongeLjMode.c_str());
                return 1;
            }
            return 0;
        }
        GromacsPairlistSnapshot gromacsSnapshot;
        if (!nbnxm_microbench::ReadGromacsPairlistSnapshot(args.snapshot,
                                                           &gromacsSnapshot))
        {
            std::fprintf(stderr, "failed to read SPONGE or GROMACS snapshot: %s\n",
                         args.snapshot.c_str());
            return 1;
        }
        if (args.analyze)
        {
            const SpongeForceOnlySnapshot converted =
                ConvertGromacsSnapshotToSponge(gromacsSnapshot);
            AnalyzeSpongeSnapshot(converted);
            return 0;
        }
        const SpongeForceOnlySnapshot converted =
            ConvertGromacsSnapshotToSponge(gromacsSnapshot);
        if (args.spongeLjMode == "comb-gmxpacked")
        {
            RunSpongeWithGmxPackedLjComb(gromacsSnapshot, args.warmup,
                                         args.iters,
                                         "gmx-adapted-comb-gmxpacked");
        }
        else
        {
            std::fprintf(stderr,
                         "gromacs snapshots currently support sponge-lj-mode comb-gmxpacked only, got %s\n",
                         args.spongeLjMode.c_str());
            return 1;
        }
        return 0;
    }
    if (args.kernel == "gmx")
    {
        SpongeGmxpackedForceOnlySnapshot productionGmxpackedSnapshot;
        if (nbnxm_microbench::ReadSpongeGmxpackedForceOnlySnapshot(
                args.snapshot, &productionGmxpackedSnapshot))
        {
            GromacsPairlistSnapshot converted =
                ConvertSpongeGmxpackedSnapshotToGromacs(
                    productionGmxpackedSnapshot, args.computeEnergy,
                    args.computeVirial);
            if (args.analyze)
            {
                AnalyzeGromacsSnapshot(converted);
                return 0;
            }
            RunGromacsProduction(converted, args.warmup, args.iters,
                                 "production-gmxpacked-converted");
            return 0;
        }
        GromacsPairlistSnapshot snapshot;
        if (nbnxm_microbench::ReadGromacsPairlistSnapshot(args.snapshot,
                                                          &snapshot))
        {
            if (args.analyze)
            {
                AnalyzeGromacsSnapshot(snapshot);
                return 0;
            }
            RunGromacsProduction(snapshot, args.warmup, args.iters, "native");
            return 0;
        }
        SpongeForceOnlySnapshot spongeSnapshot;
        SpongeClusteredFullOutputSnapshot fullOutputSnapshot;
        bool convertedFromFullOutput = false;
        if (nbnxm_microbench::ReadSpongeClusteredFullOutputSnapshot(
                args.snapshot, &fullOutputSnapshot))
        {
            spongeSnapshot = ExtractPayloadFromFullOutputSnapshot(fullOutputSnapshot);
            convertedFromFullOutput = true;
        }
        else
        {
        if (!nbnxm_microbench::ReadSpongeForceOnlySnapshot(args.snapshot,
                                                           &spongeSnapshot))
        {
            std::fprintf(stderr, "failed to read GROMACS or SPONGE snapshot: %s\n",
                         args.snapshot.c_str());
            return 1;
        }
        }
        if (args.analyze)
        {
            GromacsPairlistSnapshot converted =
                ConvertSpongeSnapshotToGromacs(spongeSnapshot,
                                               spongeGmxTransform,
                                               nullptr,
                                               args.exactImaskRadiusScale);
            if (convertedFromFullOutput)
            {
                converted.header.compute_energy =
                    fullOutputSnapshot.header.compute_energy;
                converted.header.compute_virial =
                    fullOutputSnapshot.header.compute_virial;
            }
            AnalyzeGromacsSnapshot(converted);
            return 0;
        }
        GromacsPairlistSnapshot converted =
            ConvertSpongeSnapshotToGromacs(spongeSnapshot,
                                           spongeGmxTransform,
                                           nullptr,
                                           args.exactImaskRadiusScale);
        if (convertedFromFullOutput)
        {
            converted.header.compute_energy =
                fullOutputSnapshot.header.compute_energy;
            converted.header.compute_virial =
                fullOutputSnapshot.header.compute_virial;
        }
        RunGromacsProduction(converted, args.warmup, args.iters,
                             "sponge-adapted");
        return 0;
    }
    std::fprintf(stderr, "unknown kernel kind: %s\n", args.kernel.c_str());
    return 1;
}
