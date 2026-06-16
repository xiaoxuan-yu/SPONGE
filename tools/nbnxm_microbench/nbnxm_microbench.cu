#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "nbnxm_microbench_snapshot.h"

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

struct Vec3
{
    float x;
    float y;
    float z;
};

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

struct VectorLj
{
    Vec3 crd;
    int lj_type;
    float charge;
};

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
    double exactImaskRadiusScale = 1.0)
{
    GromacsPairlistSnapshot converted = {};
    const size_t superClusterCount = snapshot.super_cluster_offsets.size() - 1;
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

    converted.cluster_offsets.resize(
        static_cast<size_t>(converted.header.cluster_numbers), 0);
    converted.cluster_valid_masks.resize(
        static_cast<size_t>(converted.header.cluster_numbers), 0u);
    converted.cluster_local_masks.resize(
        static_cast<size_t>(converted.header.cluster_numbers), 0u);
    converted.sorted_atom_ids.resize(
        static_cast<size_t>(converted.header.total_atom_numbers), -1);
    std::vector<int> denseClusterIndexForOriginalCluster(
        snapshot.cluster_offsets.size(), -1);
    std::vector<int> superClusterForOriginalCluster(
        snapshot.cluster_offsets.size(), -1);
    std::vector<int> localClusterForOriginalCluster(
        snapshot.cluster_offsets.size(), -1);
    converted.sorted_xq.resize(
        static_cast<size_t>(converted.header.total_atom_numbers));
    converted.sorted_lj_type.resize(
        static_cast<size_t>(converted.header.total_atom_numbers), 0);
    converted.sorted_lj_comb.resize(
        static_cast<size_t>(converted.header.total_atom_numbers), {});
    const std::vector<Float2POD> sortedLjComb =
        BuildSortedLjCombFromSpongeSnapshot(snapshot);
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
            converted.cluster_offsets[denseIndex] =
                static_cast<int>(denseIndex * static_cast<size_t>(kClusterSize));
            const unsigned int validMask =
                snapshot.cluster_valid_masks[static_cast<size_t>(clusterI)];
            const unsigned int localMask =
                snapshot.cluster_local_masks[static_cast<size_t>(clusterI)];
            converted.cluster_valid_masks[denseIndex] = validMask;
            converted.cluster_local_masks[denseIndex] = localMask;
            const int srcAtomBase =
                snapshot.cluster_offsets[static_cast<size_t>(clusterI)];
            const int dstAtomBase = converted.cluster_offsets[denseIndex];
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

    for (int shift = 0; shift < 27; ++shift)
    {
        const Vec3 shiftVec = ShiftVectorFromId(shift, MakeMatrix(snapshot.header.cell));
        converted.header.shiftvec[static_cast<size_t>(shift)] = {
            shiftVec.x, shiftVec.y, shiftVec.z, 0.0f};
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

    std::vector<Vec3> clusterCenters(snapshot.cluster_offsets.size(),
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

    std::vector<Vec3> superCenters(superClusterCount, Vec3{0.0f, 0.0f, 0.0f});
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

    for (size_t orderedIdx = 0; orderedIdx < sciOrder.size(); ++orderedIdx)
    {
        const size_t sciIdx = sciOrder[orderedIdx];
        const auto& sci = snapshot.sci[sciIdx];

        std::vector<AggregatedCluster> aggregated;
        std::unordered_map<int, size_t> clusterToIndex;
        const int recordBegin = snapshot.record_offsets[sciIdx];
        const int recordEnd = snapshot.record_offsets[sciIdx + 1];
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

template <bool need_energy, bool need_virial, bool total_output>
__global__ __launch_bounds__(kClusterSize * kSuperClusterClusters,
                             total_output
                                 ? ((need_energy && need_virial) ? 12 : 14)
                                 : 9)
void SpongeGmxPackedLjCombKernel(
    int sci_numbers, const GromacsSciPOD* sci_entries,
    const GromacsCjPackedPOD* cjpacked_entries,
    const GromacsExclPOD* excl_entries,
    const unsigned int* cluster_local_masks, const int* sorted_atom_ids,
    const float4* sorted_xq, const float2* sorted_lj_comb,
    const float4* shiftvec, float cutoff,
    GmxPackedForceStorage<need_energy, need_virial, total_output>* frc_xyz,
    float pme_beta,
    LTMatrix3* atom_virial, float3* shift_force,
    float* atom_energy, float* atom_direct_cf_energy, float* atom_lj_ene)
{
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
    const bool doCalcShift = need_virial && (sci_entry.shift != kCentralShiftId);

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
                AtomicAddVirial(atom_virial + atom_i, reduced_virial);     \
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
                atomicAdd(atom_energy + atom_i,                            \
                          reduced_lj + reduced_coulomb);                   \
                atomicAdd(atom_lj_ene + atom_i, reduced_lj);               \
                atomicAdd(atom_direct_cf_energy + atom_i,                  \
                          reduced_coulomb);                                \
            }                                                              \
        }                                                                  \
    }
    SPONGE_REPLAY_GMXPACKED_I_LOCAL_LIST(SPONGE_REPLAY_GMXPACKED_REDUCE_I)
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
    bool analyze = false;
};

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
        else if (flag == "--analyze")
        {
            args.analyze = true;
        }
        else
        {
            std::fprintf(stderr,
                         "Usage: %s --kernel sponge|gmx --snapshot PATH "
                         "[--sponge-lj-mode fulloutput|comb-gmxpacked] "
                         "[--sponge-gmx-transform baseline] "
                         "[--warmup N] [--iters N] [--analyze]\n",
                         argv[0]);
            std::exit(1);
        }
    }
    if (args.kernel.empty() || args.snapshot.empty())
    {
        std::fprintf(stderr,
                     "Usage: %s --kernel sponge|gmx --snapshot PATH "
                     "[--sponge-lj-mode fulloutput|comb-gmxpacked] "
                     "[--sponge-gmx-transform baseline] "
                     "[--warmup N] [--iters N] [--analyze]\n",
                     argv[0]);
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

struct DiffStats
{
    double max_abs = 0.0;
    double rms = 0.0;
};

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
        for (double diff : diffs)
        {
            stats.max_abs = std::max(stats.max_abs, std::fabs(diff));
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
    const SpongeClusteredFullOutputSnapshot* referenceFullOutputSnapshot = nullptr)
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
        referenceFullOutputSnapshot == nullptr ||
        referenceFullOutputSnapshot->header.total_output != 0u;
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
                    d_cluster_local_masks, nullptr, d_sorted_xq,
                    d_sorted_lj_comb, d_shiftvec, snapshot.header.cutoff,
                    frc_xyz_compact, snapshot.header.pme_beta, nullptr,
                    d_shift_force, d_atom_energy, d_atom_direct_cf_energy,
                    d_atom_lj_energy);
            }
            else
            {
                SpongeGmxPackedLjCombKernel<true, true, false><<<grid, block>>>(
                    sci_numbers, d_sci, d_cjpacked, d_excl,
                    d_cluster_local_masks, d_sorted_atom_ids, d_sorted_xq,
                    d_sorted_lj_comb, d_shiftvec, snapshot.header.cutoff,
                    frc_xyz_regular, snapshot.header.pme_beta, d_atom_virial,
                    nullptr, d_atom_energy, d_atom_direct_cf_energy,
                    d_atom_lj_energy);
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
                    d_cluster_local_masks, nullptr, d_sorted_xq,
                    d_sorted_lj_comb, d_shiftvec, snapshot.header.cutoff,
                    frc_xyz_compact, snapshot.header.pme_beta, nullptr, nullptr,
                    d_atom_energy, d_atom_direct_cf_energy, d_atom_lj_energy);
            }
            else
            {
                SpongeGmxPackedLjCombKernel<true, false, false><<<grid, block>>>(
                    sci_numbers, d_sci, d_cjpacked, d_excl,
                    d_cluster_local_masks, d_sorted_atom_ids, d_sorted_xq,
                    d_sorted_lj_comb, d_shiftvec, snapshot.header.cutoff,
                    frc_xyz_regular, snapshot.header.pme_beta, nullptr, nullptr,
                    d_atom_energy, d_atom_direct_cf_energy, d_atom_lj_energy);
            }
        }
        else if (needVirial)
        {
            if (totalOutput)
            {
                SpongeGmxPackedLjCombKernel<false, true, true><<<grid, block>>>(
                    sci_numbers, d_sci, d_cjpacked, d_excl,
                    d_cluster_local_masks, nullptr, d_sorted_xq,
                    d_sorted_lj_comb, d_shiftvec, snapshot.header.cutoff,
                    frc_xyz_compact, snapshot.header.pme_beta, nullptr,
                    d_shift_force, nullptr, nullptr, nullptr);
            }
            else
            {
                SpongeGmxPackedLjCombKernel<false, true, false><<<grid, block>>>(
                    sci_numbers, d_sci, d_cjpacked, d_excl,
                    d_cluster_local_masks, d_sorted_atom_ids, d_sorted_xq,
                    d_sorted_lj_comb, d_shiftvec, snapshot.header.cutoff,
                    frc_xyz_regular, snapshot.header.pme_beta, d_atom_virial,
                    nullptr, nullptr, nullptr, nullptr);
            }
        }
        else
        {
            SpongeGmxPackedLjCombKernel<false, false, true><<<grid, block>>>(
                sci_numbers, d_sci, d_cjpacked, d_excl, d_cluster_local_masks,
                nullptr, d_sorted_xq, d_sorted_lj_comb, d_shiftvec,
                snapshot.header.cutoff, frc_xyz_regular,
                snapshot.header.pme_beta,
                nullptr, nullptr, nullptr, nullptr, nullptr);
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
            "kernel=sponge_gmxpacked_ljcomb_fulloutput snapshot=%s avg_ms=%.6f iters=%d sci=%d cjpacked=%zu excl=%zu atoms=%d force_max_abs=%.6e force_rms=%.6e virial_max_abs=%.6e virial_rms=%.6e energy_max_abs=%.6e energy_rms=%.6e direct_energy_max_abs=%.6e lj_energy_max_abs=%.6e total_output=%u\n",
            snapshotLabel, total_ms / static_cast<float>(iters), iters,
            sci_numbers, normalizedCjpacked.size(), normalizedExcl.size(),
            total_atom_numbers, force_stats.max_abs, force_stats.rms,
            virial_stats.max_abs, virial_stats.rms, energy_stats.max_abs,
            energy_stats.rms, direct_energy_stats.max_abs,
            lj_energy_stats.max_abs, totalOutput ? 1u : 0u);
    }
    else if (needEnergy || needVirial)
    {
        std::printf("kernel=sponge_gmxpacked_ljcomb_totaloutput snapshot=%s avg_ms=%.6f "
                    "iters=%d sci=%d cjpacked=%zu excl=%zu atoms=%d compute_energy=%u compute_virial=%u "
                    "sanity=%s energy=%.6e direct_energy=%.6e lj_energy=%.6e virial_xx=%.6e virial_yy=%.6e virial_zz=%.6e\n",
                    snapshotLabel, total_ms / static_cast<float>(iters), iters,
                    sci_numbers, normalizedCjpacked.size(), normalizedExcl.size(),
                    total_atom_numbers, snapshot.header.compute_energy,
                    snapshot.header.compute_virial, sane ? "ok" : "bad",
                    host_atom_energy, host_direct_cf_energy, host_lj_energy,
                    host_virial.a11, host_virial.a22, host_virial.a33);
    }
    else
    {
        std::printf("kernel=sponge_gmxpacked_ljcomb snapshot=%s avg_ms=%.6f "
                    "iters=%d sci=%d cjpacked=%zu excl=%zu atoms=%d\n",
                    snapshotLabel, total_ms / static_cast<float>(iters), iters,
                    sci_numbers, normalizedCjpacked.size(), normalizedExcl.size(),
                    total_atom_numbers);
    }

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

} // namespace

int main(int argc, char** argv)
{
    const Arguments args = ParseArguments(argc, argv);
    const SpongeGmxTransformMode spongeGmxTransform =
        ParseSpongeGmxTransform(args.spongeGmxTransform);
    if (args.kernel == "sponge")
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
                return 0;
            }
            if (args.spongeLjMode == "fulloutput")
            {
                RunSpongeFullOutput(fullOutputSnapshot, args.warmup,
                                    args.iters, "native-fulloutput");
                return 0;
            }
            if (args.spongeLjMode == "comb-gmxpacked")
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
                                             "fulloutput-payload-comb-gmxpacked",
                                             fullOutputReference);
                return 0;
            }
            std::fprintf(stderr,
                         "full-output snapshot currently supports sponge-lj-mode fulloutput|comb-gmxpacked, got %s\n",
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
            if (args.spongeLjMode == "comb-gmxpacked")
            {
                const GromacsPairlistSnapshot converted =
                    ConvertSpongeSnapshotToGromacs(
                        snapshot, spongeGmxTransform, nullptr,
                        args.exactImaskRadiusScale);
                RunSpongeWithGmxPackedLjComb(converted, args.warmup,
                                             args.iters,
                                             "native-comb-gmxpacked");
            }
            else
            {
                std::fprintf(stderr,
                             "force-only snapshots currently support sponge-lj-mode comb-gmxpacked only, got %s\n",
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
