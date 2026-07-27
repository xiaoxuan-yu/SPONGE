#include "canonical_pair_oracle.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace nbnxm_microbench
{
namespace
{
constexpr int kClusterSize = 8;
constexpr int kSuperClusterClusters = 8;
constexpr int kWarpSplitCount = 2;
constexpr int kSplitJClusterSize = 4;
constexpr int kJGroupSize = 4;
constexpr int kCentralShiftId = 13;
constexpr int kPairShiftBits = 5;
constexpr uint64_t kPairShiftMask = (1ull << kPairShiftBits) - 1ull;
constexpr int kPairActiveMaskOffset =
    kSuperClusterClusters * kPairShiftBits;
constexpr int kPairActiveMarkerOffset =
    kPairActiveMaskOffset + kWarpSplitCount * kSuperClusterClusters;
constexpr uint64_t kPairActiveMarker =
    1ull << kPairActiveMarkerOffset;

struct Vec3d
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Matrix3d
{
    double a11 = 0.0;
    double a21 = 0.0;
    double a22 = 0.0;
    double a31 = 0.0;
    double a32 = 0.0;
    double a33 = 0.0;
};

struct PairHash
{
    size_t operator()(const CanonicalPair& pair) const
    {
        uint64_t hash = static_cast<uint32_t>(pair.global_i);
        hash ^= static_cast<uint64_t>(
                    static_cast<uint32_t>(pair.global_j))
                << 32;
        hash ^= static_cast<uint64_t>(
                    static_cast<uint32_t>(pair.shift_id))
                * 0x9e3779b97f4a7c15ull;
        hash ^= hash >> 30;
        hash *= 0xbf58476d1ce4e5b9ull;
        hash ^= hash >> 27;
        hash *= 0x94d049bb133111ebull;
        return static_cast<size_t>(hash ^ (hash >> 31));
    }
};

struct LocalPairHash
{
    size_t operator()(uint64_t pair) const
    {
        pair ^= pair >> 30;
        pair *= 0xbf58476d1ce4e5b9ull;
        pair ^= pair >> 27;
        pair *= 0x94d049bb133111ebull;
        return static_cast<size_t>(pair ^ (pair >> 31));
    }
};

Vec3d Multiply(Vec3d value, const Matrix3d& matrix)
{
    return {
        value.x * matrix.a11 + value.y * matrix.a21 +
            value.z * matrix.a31,
        value.y * matrix.a22 + value.z * matrix.a32,
        value.z * matrix.a33,
    };
}

Matrix3d ToMatrix(const LTMatrix3POD& matrix)
{
    return {
        matrix.a11,
        matrix.a21,
        matrix.a22,
        matrix.a31,
        matrix.a32,
        matrix.a33,
    };
}

bool InvertLowerTriangular(const Matrix3d& matrix, Matrix3d* inverse)
{
    constexpr double kMinDiagonal = 1.0e-12;
    if (inverse == nullptr || std::abs(matrix.a11) < kMinDiagonal ||
        std::abs(matrix.a22) < kMinDiagonal ||
        std::abs(matrix.a33) < kMinDiagonal)
    {
        return false;
    }
    inverse->a11 = 1.0 / matrix.a11;
    inverse->a22 = 1.0 / matrix.a22;
    inverse->a33 = 1.0 / matrix.a33;
    inverse->a21 = -matrix.a21 / (matrix.a11 * matrix.a22);
    inverse->a32 = -matrix.a32 / (matrix.a22 * matrix.a33);
    inverse->a31 =
        (matrix.a21 * matrix.a32 - matrix.a22 * matrix.a31) /
        (matrix.a11 * matrix.a22 * matrix.a33);
    return true;
}

std::array<int, 3> ShiftComponents(int shift_id)
{
    return {
        shift_id / 9 - 1,
        (shift_id % 9) / 3 - 1,
        shift_id % 3 - 1,
    };
}

int ShiftId(int sx, int sy, int sz)
{
    if (sx < -1 || sx > 1 || sy < -1 || sy > 1 || sz < -1 || sz > 1)
    {
        return -1;
    }
    return (sx + 1) * 9 + (sy + 1) * 3 + (sz + 1);
}

Vec3d ShiftVector(int shift_id, const Matrix3d& cell)
{
    const auto shift = ShiftComponents(shift_id);
    return Multiply(
        { static_cast<double>(shift[0]), static_cast<double>(shift[1]),
          static_cast<double>(shift[2]) },
        cell);
}

int InverseShiftId(int shift_id)
{
    const auto shift = ShiftComponents(shift_id);
    return ShiftId(-shift[0], -shift[1], -shift[2]);
}

CanonicalPair NormalizePair(int global_i, int global_j, int shift_id)
{
    if (global_i <= global_j)
    {
        return { global_i, global_j, shift_id };
    }
    return { global_j, global_i, InverseShiftId(shift_id) };
}

int GlobalAtomId(const SpongeGmxpackedForceOnlySnapshot& snapshot,
                 int local_atom)
{
    if (local_atom < 0)
    {
        return -1;
    }
    if (!snapshot.atom_local.empty())
    {
        if (static_cast<size_t>(local_atom) >= snapshot.atom_local.size())
        {
            return -1;
        }
        return snapshot.atom_local[static_cast<size_t>(local_atom)];
    }
    return local_atom;
}

uint64_t LocalPairKey(int atom_i, int atom_j)
{
    const uint32_t lo = static_cast<uint32_t>(std::min(atom_i, atom_j));
    const uint32_t hi = static_cast<uint32_t>(std::max(atom_i, atom_j));
    return static_cast<uint64_t>(lo) |
           (static_cast<uint64_t>(hi) << 32);
}

bool BuildExcludedPairs(
    const SpongeGmxpackedForceOnlySnapshot& snapshot,
    std::unordered_set<uint64_t, LocalPairHash>* excluded,
    std::string* failure_reason)
{
    if (excluded == nullptr ||
        snapshot.excluded_list_start.size() !=
            snapshot.excluded_numbers.size())
    {
        if (failure_reason != nullptr)
        {
            *failure_reason = "oracle exclusion metadata is incomplete";
        }
        return false;
    }
    excluded->clear();
    const size_t atom_count = snapshot.excluded_list_start.size();
    size_t expected_begin = 0;
    for (size_t atom_i = 0; atom_i < atom_count; ++atom_i)
    {
        const int begin = snapshot.excluded_list_start[atom_i];
        const int count = snapshot.excluded_numbers[atom_i];
        if (begin < 0 || count < 0 ||
            static_cast<size_t>(begin) != expected_begin ||
            static_cast<size_t>(begin) + static_cast<size_t>(count) >
                snapshot.excluded_list.size())
        {
            if (failure_reason != nullptr)
            {
                *failure_reason =
                    "oracle exclusion CSR range is invalid";
            }
            return false;
        }
        for (int offset = 0; offset < count; ++offset)
        {
            const int atom_j =
                snapshot.excluded_list[static_cast<size_t>(begin + offset)];
            if (atom_j == -1)
            {
                continue;
            }
            if (atom_j < 0 ||
                static_cast<size_t>(atom_j) >=
                    snapshot.atom_local.size())
            {
                if (failure_reason != nullptr)
                {
                    *failure_reason =
                        "oracle exclusion entry is outside the local/ghost "
                        "atom domain";
                }
                return false;
            }
            excluded->insert(
                LocalPairKey(static_cast<int>(atom_i), atom_j));
        }
        expected_begin += static_cast<size_t>(count);
    }
    if (expected_begin != snapshot.excluded_list.size())
    {
        if (failure_reason != nullptr)
        {
            *failure_reason =
                "oracle exclusion CSR does not cover the entry payload";
        }
        return false;
    }
    return true;
}

bool LoadCoordinates(const SpongeGmxpackedForceOnlySnapshot& snapshot,
                     std::vector<Vec3d>* coordinates)
{
    if (coordinates == nullptr)
    {
        return false;
    }
    const size_t total_atoms =
        static_cast<size_t>(snapshot.header.total_atom_numbers);
    if (snapshot.sorted_atom_ids.size() != total_atoms ||
        snapshot.sorted_xq.size() != total_atoms)
    {
        return false;
    }
    coordinates->assign(total_atoms, {});
    std::vector<unsigned char> seen(total_atoms, 0u);
    for (size_t sorted = 0; sorted < total_atoms; ++sorted)
    {
        const int local_atom = snapshot.sorted_atom_ids[sorted];
        if (local_atom < 0 ||
            static_cast<size_t>(local_atom) >= total_atoms)
        {
            return false;
        }
        const Float4POD& value = snapshot.sorted_xq[sorted];
        (*coordinates)[static_cast<size_t>(local_atom)] =
            { value.x, value.y, value.z };
        seen[static_cast<size_t>(local_atom)] = 1u;
    }
    return std::find(seen.begin(), seen.end(), 0u) == seen.end();
}

int PackedPairShiftId(
    const SpongeGmxpackedForceOnlySnapshot& snapshot, size_t sci_index,
    size_t packed_index, int jm, int i_local)
{
    const SpongeGmxpackedSciPOD& sci = snapshot.sci[sci_index];
    const bool sci_shift_safe =
        snapshot.sci_shift_safe_flags.size() != snapshot.sci.size() ||
        snapshot.sci_shift_safe_flags[sci_index] != 0;
    if (sci_shift_safe)
    {
        return sci.shift_id;
    }
    const size_t shift_word =
        packed_index * static_cast<size_t>(kJGroupSize) +
        static_cast<size_t>(jm);
    if (shift_word >= snapshot.pair_shift_bits.size())
    {
        return -1;
    }
    return static_cast<int>(
        (snapshot.pair_shift_bits[shift_word] >>
         (static_cast<uint64_t>(i_local) * kPairShiftBits)) &
        kPairShiftMask);
}

unsigned int PackedPairActiveIMask(
    const SpongeGmxpackedForceOnlySnapshot& snapshot, size_t packed_index,
    int jm, int split)
{
    const size_t shift_word =
        packed_index * static_cast<size_t>(kJGroupSize) +
        static_cast<size_t>(jm);
    if (shift_word >= snapshot.pair_shift_bits.size())
    {
        return 0u;
    }
    const uint64_t bits = snapshot.pair_shift_bits[shift_word];
    if ((bits & kPairActiveMarker) == 0ull)
    {
        return (1u << kSuperClusterClusters) - 1u;
    }
    return static_cast<unsigned int>(
        (bits >>
         (kPairActiveMaskOffset + split * kSuperClusterClusters)) &
        ((1ull << kSuperClusterClusters) - 1ull));
}

double PairDistanceSquared(Vec3d atom_i, Vec3d atom_j, int shift_id,
                           const Matrix3d& cell)
{
    const Vec3d shift = ShiftVector(shift_id, cell);
    const double dx = atom_j.x - atom_i.x - shift.x;
    const double dy = atom_j.y - atom_i.y - shift.y;
    const double dz = atom_j.z - atom_i.z - shift.z;
    return dx * dx + dy * dy + dz * dz;
}

bool DecodePayloadPairs(
    const SpongeGmxpackedForceOnlySnapshot& snapshot,
    const std::vector<Vec3d>& coordinates,
    std::unordered_map<CanonicalPair, size_t, PairHash>* pairs,
    std::unordered_map<CanonicalPair, std::vector<CanonicalPairOccurrence>,
                       PairHash>* occurrences,
    std::string* failure_reason)
{
    if (pairs == nullptr || snapshot.header.cluster_size != kClusterSize ||
        snapshot.header.super_cluster_clusters !=
            kSuperClusterClusters ||
        snapshot.header.warp_split_count != kWarpSplitCount ||
        snapshot.header.j_group_size != kJGroupSize)
    {
        if (failure_reason != nullptr)
        {
            *failure_reason = "unsupported clustered payload shape";
        }
        return false;
    }
    const Matrix3d cell = ToMatrix(snapshot.header.cell);
    const double cutoff_sq =
        static_cast<double>(snapshot.header.cutoff) *
        static_cast<double>(snapshot.header.cutoff);
    const int local_atoms =
        static_cast<int>(snapshot.header.local_atom_numbers);
    const size_t cluster_numbers =
        static_cast<size_t>(snapshot.header.cluster_numbers);
    const size_t super_cluster_numbers =
        static_cast<size_t>(snapshot.header.super_cluster_numbers);
    if (snapshot.cluster_offsets.size() != cluster_numbers ||
        snapshot.cluster_valid_masks.size() != cluster_numbers ||
        snapshot.cluster_local_masks.size() != cluster_numbers ||
        snapshot.super_cluster_offsets.size() !=
            super_cluster_numbers + 1 ||
        snapshot.sci_shift_safe_flags.size() != snapshot.sci.size() ||
        snapshot.pair_shift_bits.size() !=
            snapshot.cjpacked.size() *
                static_cast<size_t>(kJGroupSize))
    {
        if (failure_reason != nullptr)
        {
            *failure_reason =
                "packed shift metadata size is inconsistent";
        }
        return false;
    }
    for (size_t sci_index = 0; sci_index < snapshot.sci.size(); ++sci_index)
    {
        const SpongeGmxpackedSciPOD& sci = snapshot.sci[sci_index];
        if (sci.supercluster_id < 0 ||
            static_cast<uint64_t>(sci.supercluster_id) >=
                snapshot.header.super_cluster_numbers ||
            sci.shift_id < 0 ||
            sci.shift_id >= kCentralShiftId * 2 + 1 ||
            sci.cjpacked_begin < 0 ||
            sci.cjpacked_end < sci.cjpacked_begin ||
            static_cast<size_t>(sci.cjpacked_end) >
                snapshot.cjpacked.size())
        {
            if (failure_reason != nullptr)
            {
                *failure_reason =
                    "SCI metadata is outside the packed payload";
            }
            return false;
        }
        const size_t super_i = static_cast<size_t>(sci.supercluster_id);
        const int cluster_i_start =
            snapshot.super_cluster_offsets[super_i];
        const int cluster_i_end =
            snapshot.super_cluster_offsets[super_i + 1];
        if (cluster_i_start < 0 || cluster_i_end < cluster_i_start ||
            static_cast<size_t>(cluster_i_end) > cluster_numbers ||
            cluster_i_end - cluster_i_start > kSuperClusterClusters)
        {
            if (failure_reason != nullptr)
            {
                *failure_reason =
                    "SCI supercluster range is outside the cluster domain";
            }
            return false;
        }
        const int active_i_clusters = cluster_i_end - cluster_i_start;
        for (int packed_index = sci.cjpacked_begin;
             packed_index < sci.cjpacked_end; ++packed_index)
        {
            const SpongeGmxpackedCjPOD& packed =
                snapshot.cjpacked[static_cast<size_t>(packed_index)];
            for (int split = 0; split < kWarpSplitCount; ++split)
            {
                const unsigned int imask = packed.split[split].imask;
                if (imask == 0u)
                {
                    continue;
                }
                const int exclusion_index =
                    packed.split[split].exclusion_index;
                if (exclusion_index < 0 ||
                    static_cast<size_t>(exclusion_index) >=
                        snapshot.excl.size())
                {
                    if (failure_reason != nullptr)
                    {
                        *failure_reason =
                            "packed exclusion index is outside the pool";
                    }
                    return false;
                }
                for (int split_j_lane = 0;
                     split_j_lane < kSplitJClusterSize; ++split_j_lane)
                {
                    const int j_lane =
                        split * kSplitJClusterSize + split_j_lane;
                    for (int i_lane = 0; i_lane < kClusterSize; ++i_lane)
                    {
                        unsigned int pair_bits = 0xffffffffu;
                        if (exclusion_index != 0)
                        {
                            pair_bits =
                                snapshot.excl[static_cast<size_t>(
                                                  exclusion_index)]
                                    .pair[split_j_lane * kClusterSize +
                                          i_lane];
                        }
                        const unsigned int effective_mask =
                            imask & pair_bits;
                        for (int jm = 0; jm < kJGroupSize; ++jm)
                        {
                            const int cluster_j = packed.cj[jm];
                            if (cluster_j < 0)
                            {
                                continue;
                            }
                            if (static_cast<uint64_t>(cluster_j) >=
                                snapshot.header.cluster_numbers)
                            {
                                if (failure_reason != nullptr)
                                {
                                    *failure_reason =
                                        "active J cluster is outside the "
                                        "cluster domain";
                                }
                                return false;
                            }
                            for (int i_local = 0;
                                 i_local < kSuperClusterClusters; ++i_local)
                            {
                                if (i_local >= active_i_clusters)
                                {
                                    continue;
                                }
                                const unsigned int bit =
                                    1u << (jm * kSuperClusterClusters +
                                           i_local);
                                if ((effective_mask & bit) == 0u)
                                {
                                    continue;
                                }
                                if ((PackedPairActiveIMask(
                                         snapshot,
                                         static_cast<size_t>(packed_index),
                                         jm, split) &
                                     (1u << static_cast<unsigned int>(
                                          i_local))) == 0u)
                                {
                                    continue;
                                }
                                const int cluster_i =
                                    cluster_i_start + i_local;
                                if (cluster_i < 0 ||
                                    static_cast<size_t>(cluster_i) >=
                                        cluster_numbers)
                                {
                                    if (failure_reason != nullptr)
                                    {
                                        *failure_reason =
                                            "active I cluster is outside the "
                                            "cluster domain";
                                    }
                                    return false;
                                }
                                const unsigned int i_lane_bit =
                                    1u << static_cast<unsigned int>(i_lane);
                                if ((snapshot.cluster_valid_masks[
                                         static_cast<size_t>(cluster_i)] &
                                     i_lane_bit) == 0u ||
                                    (snapshot.cluster_local_masks[
                                         static_cast<size_t>(cluster_i)] &
                                     i_lane_bit) == 0u)
                                {
                                    continue;
                                }
                                const unsigned int j_lane_bit =
                                    1u << static_cast<unsigned int>(j_lane);
                                if ((snapshot.cluster_valid_masks[
                                         static_cast<size_t>(cluster_j)] &
                                     j_lane_bit) == 0u)
                                {
                                    continue;
                                }
                                const int sorted_i =
                                    snapshot.cluster_offsets[
                                        static_cast<size_t>(cluster_i)] +
                                    i_lane;
                                const int sorted_j =
                                    snapshot.cluster_offsets[
                                        static_cast<size_t>(cluster_j)] +
                                    j_lane;
                                if (sorted_i < 0 || sorted_j < 0 ||
                                    static_cast<size_t>(sorted_i) >=
                                        snapshot.sorted_atom_ids.size() ||
                                    static_cast<size_t>(sorted_j) >=
                                        snapshot.sorted_atom_ids.size())
                                {
                                    if (failure_reason != nullptr)
                                    {
                                        *failure_reason =
                                            "active packed lane is outside "
                                            "the sorted atom payload";
                                    }
                                    return false;
                                }
                                const int atom_i =
                                    snapshot.sorted_atom_ids[static_cast<size_t>(
                                        sorted_i)];
                                const int atom_j =
                                    snapshot.sorted_atom_ids[static_cast<size_t>(
                                        sorted_j)];
                                if (atom_i < 0 || atom_i >= local_atoms ||
                                    atom_j < 0 ||
                                    static_cast<size_t>(atom_j) >=
                                        coordinates.size() ||
                                    atom_i == atom_j)
                                {
                                    if (failure_reason != nullptr)
                                    {
                                        *failure_reason =
                                            "active packed lane has an "
                                            "invalid local/ghost atom";
                                    }
                                    return false;
                                }
                                const int shift_id = PackedPairShiftId(
                                    snapshot, sci_index,
                                    static_cast<size_t>(packed_index), jm,
                                    i_local);
                                if (shift_id < 0 ||
                                    shift_id >= kCentralShiftId * 2 + 1)
                                {
                                    if (failure_reason != nullptr)
                                    {
                                        *failure_reason =
                                            "packed pair shift is invalid";
                                    }
                                    return false;
                                }
                                const double distance_sq =
                                    PairDistanceSquared(
                                        coordinates[static_cast<size_t>(
                                            atom_i)],
                                        coordinates[static_cast<size_t>(
                                            atom_j)],
                                        shift_id, cell);
                                if (distance_sq >= cutoff_sq)
                                {
                                    continue;
                                }
                                const int global_i =
                                    GlobalAtomId(snapshot, atom_i);
                                const int global_j =
                                    GlobalAtomId(snapshot, atom_j);
                                if (global_i < 0 || global_j < 0 ||
                                    global_i == global_j)
                                {
                                    if (failure_reason != nullptr)
                                    {
                                        *failure_reason =
                                            "active packed lane has an "
                                            "invalid global atom identity";
                                    }
                                    return false;
                                }
                                const CanonicalPair pair = NormalizePair(
                                    global_i, global_j, shift_id);
                                (*pairs)[pair] += 1;
                                if (occurrences != nullptr)
                                {
                                    uint64_t exclusion_hash =
                                        1469598103934665603ull;
                                    for (unsigned int pair_mask :
                                         snapshot.excl[static_cast<size_t>(
                                                           exclusion_index)]
                                             .pair)
                                    {
                                        exclusion_hash ^= pair_mask;
                                        exclusion_hash *= 1099511628211ull;
                                    }
                                    (*occurrences)[pair].push_back(
                                        {pair,
                                         sci_index,
                                         static_cast<size_t>(packed_index),
                                         split,
                                         jm,
                                         i_local,
                                         cluster_i,
                                         cluster_j,
                                         sci.shift_id,
                                         shift_id,
                                         exclusion_index,
                                         imask,
                                         exclusion_hash});
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}

int WrapCell(int value, int extent)
{
    value %= extent;
    return value < 0 ? value + extent : value;
}

int CellIndex(int x, int y, int z, const std::array<int, 3>& dims)
{
    return (z * dims[1] + y) * dims[0] + x;
}

bool BuildOraclePairs(
    const SpongeGmxpackedForceOnlySnapshot& snapshot,
    const std::vector<Vec3d>& coordinates,
    std::unordered_set<CanonicalPair, PairHash>* pairs,
    std::string* failure_reason)
{
    if (pairs == nullptr || snapshot.header.cutoff <= 0.0f)
    {
        if (failure_reason != nullptr)
        {
            *failure_reason = "oracle cutoff is not positive";
        }
        return false;
    }
    const Matrix3d cell = ToMatrix(snapshot.header.cell);
    Matrix3d rcell;
    if (!InvertLowerTriangular(cell, &rcell))
    {
        if (failure_reason != nullptr)
        {
            *failure_reason = "oracle cell matrix is singular";
        }
        return false;
    }
    const double cutoff = snapshot.header.cutoff;
    const std::array<double, 3> reciprocal_norms = {
        std::sqrt(rcell.a11 * rcell.a11 + rcell.a21 * rcell.a21 +
                  rcell.a31 * rcell.a31),
        std::sqrt(rcell.a22 * rcell.a22 + rcell.a32 * rcell.a32),
        std::abs(rcell.a33),
    };
    std::array<int, 3> dims = {};
    for (int axis = 0; axis < 3; ++axis)
    {
        const double fractional_cutoff = cutoff * reciprocal_norms[axis];
        dims[axis] =
            fractional_cutoff > 0.0
                ? std::max(1, static_cast<int>(
                                  std::floor(1.0 / fractional_cutoff)))
                : 1;
    }
    const size_t cell_count =
        static_cast<size_t>(dims[0]) * static_cast<size_t>(dims[1]) *
        static_cast<size_t>(dims[2]);
    std::vector<std::vector<int>> cells(cell_count);
    std::vector<std::array<int, 3>> atom_cells(coordinates.size());
    for (size_t atom = 0; atom < coordinates.size(); ++atom)
    {
        Vec3d fractional = Multiply(coordinates[atom], rcell);
        fractional.x -= std::floor(fractional.x);
        fractional.y -= std::floor(fractional.y);
        fractional.z -= std::floor(fractional.z);
        const std::array<int, 3> cell_xyz = {
            std::min(dims[0] - 1,
                     static_cast<int>(fractional.x * dims[0])),
            std::min(dims[1] - 1,
                     static_cast<int>(fractional.y * dims[1])),
            std::min(dims[2] - 1,
                     static_cast<int>(fractional.z * dims[2])),
        };
        atom_cells[atom] = cell_xyz;
        cells[static_cast<size_t>(
                  CellIndex(cell_xyz[0], cell_xyz[1], cell_xyz[2], dims))]
            .push_back(static_cast<int>(atom));
    }

    std::unordered_set<uint64_t, LocalPairHash> excluded;
    if (!BuildExcludedPairs(snapshot, &excluded, failure_reason))
    {
        return false;
    }
    const int local_atoms =
        static_cast<int>(snapshot.header.local_atom_numbers);
    const double cutoff_sq = cutoff * cutoff;
    for (int atom_i = 0; atom_i < local_atoms; ++atom_i)
    {
        const auto cell_i = atom_cells[static_cast<size_t>(atom_i)];
        std::array<int, 27> neighbor_cells = {};
        int neighbor_cell_count = 0;
        for (int dz = -1; dz <= 1; ++dz)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    const int cell = CellIndex(
                        WrapCell(cell_i[0] + dx, dims[0]),
                        WrapCell(cell_i[1] + dy, dims[1]),
                        WrapCell(cell_i[2] + dz, dims[2]), dims);
                    if (std::find(neighbor_cells.begin(),
                                  neighbor_cells.begin() +
                                      neighbor_cell_count,
                                  cell) ==
                        neighbor_cells.begin() + neighbor_cell_count)
                    {
                        neighbor_cells[neighbor_cell_count++] = cell;
                    }
                }
            }
        }
        for (int cell_offset = 0; cell_offset < neighbor_cell_count;
             ++cell_offset)
        {
            for (int atom_j :
                 cells[static_cast<size_t>(
                     neighbor_cells[cell_offset])])
            {
                if (atom_i == atom_j ||
                    (atom_j < local_atoms && atom_j < atom_i) ||
                    excluded.contains(LocalPairKey(atom_i, atom_j)))
                {
                    continue;
                }
                const int global_i = GlobalAtomId(snapshot, atom_i);
                const int global_j = GlobalAtomId(snapshot, atom_j);
                if (global_i < 0 || global_j < 0 ||
                    global_i == global_j)
                {
                    if (failure_reason != nullptr)
                    {
                        *failure_reason =
                            "oracle atom has an invalid global identity";
                    }
                    return false;
                }
                for (int shift_id = 0;
                     shift_id < kCentralShiftId * 2 + 1; ++shift_id)
                {
                    if (PairDistanceSquared(
                            coordinates[static_cast<size_t>(atom_i)],
                            coordinates[static_cast<size_t>(atom_j)],
                            shift_id, cell) >= cutoff_sq)
                    {
                        continue;
                    }
                    pairs->insert(
                        NormalizePair(global_i, global_j, shift_id));
                }
            }
        }
    }
    return true;
}

template <typename Collection>
std::vector<CanonicalPair> SortedExamples(const Collection& collection,
                                          size_t limit)
{
    std::vector<CanonicalPair> result(collection.begin(), collection.end());
    std::sort(result.begin(), result.end());
    if (result.size() > limit)
    {
        result.resize(limit);
    }
    return result;
}
}  // namespace

CanonicalPairOracleResult CompareCanonicalPairs(
    const SpongeGmxpackedForceOnlySnapshot& snapshot, size_t example_limit)
{
    CanonicalPairOracleResult result;
    const size_t total_atoms =
        static_cast<size_t>(snapshot.header.total_atom_numbers);
    const size_t local_atoms =
        static_cast<size_t>(snapshot.header.local_atom_numbers);
    result.metadata_ready =
        local_atoms <= total_atoms &&
        snapshot.atom_local.size() == total_atoms &&
        snapshot.excluded_list_start.size() == local_atoms &&
        snapshot.excluded_numbers.size() == local_atoms &&
        snapshot.sorted_atom_ids.size() == total_atoms &&
        snapshot.sorted_xq.size() == total_atoms;
    if (!result.metadata_ready)
    {
        result.failure_reason =
            "snapshot lacks independent atom-local/exclusion oracle metadata";
        return result;
    }
    std::unordered_set<int> owned_global_atoms;
    owned_global_atoms.reserve(local_atoms);
    for (size_t atom = 0; atom < snapshot.atom_local.size(); ++atom)
    {
        const int global_atom = snapshot.atom_local[atom];
        if (global_atom < 0 ||
            (atom < local_atoms &&
             !owned_global_atoms.insert(global_atom).second) ||
            (atom >= local_atoms &&
             owned_global_atoms.contains(global_atom)))
        {
            result.failure_reason =
                "snapshot atom-local mapping has an invalid or duplicated "
                "owned global atom";
            return result;
        }
    }

    std::vector<Vec3d> coordinates;
    if (!LoadCoordinates(snapshot, &coordinates))
    {
        result.failure_reason =
            "snapshot sorted coordinates do not form a local atom permutation";
        return result;
    }

    std::unordered_map<CanonicalPair, size_t, PairHash> payload_counts;
    std::unordered_map<CanonicalPair, std::vector<CanonicalPairOccurrence>,
                       PairHash>
        payload_occurrences;
    if (!DecodePayloadPairs(snapshot, coordinates, &payload_counts,
                            &payload_occurrences,
                            &result.failure_reason))
    {
        return result;
    }
    std::unordered_set<CanonicalPair, PairHash> payload_pairs;
    payload_pairs.reserve(payload_counts.size());
    std::vector<CanonicalPair> duplicates;
    for (const auto& [pair, count] : payload_counts)
    {
        payload_pairs.insert(pair);
        if (count > 1)
        {
            result.duplicate_payload_pairs += count - 1;
            duplicates.push_back(pair);
        }
    }

    std::unordered_set<CanonicalPair, PairHash> oracle_pairs;
    if (!BuildOraclePairs(snapshot, coordinates, &oracle_pairs,
                          &result.failure_reason))
    {
        return result;
    }

    std::vector<CanonicalPair> missing;
    std::vector<CanonicalPair> extra;
    for (const CanonicalPair& pair : oracle_pairs)
    {
        if (!payload_pairs.contains(pair))
        {
            missing.push_back(pair);
        }
    }
    for (const CanonicalPair& pair : payload_pairs)
    {
        if (!oracle_pairs.contains(pair))
        {
            extra.push_back(pair);
        }
    }
    result.payload_pair_count = payload_pairs.size();
    result.oracle_pair_count = oracle_pairs.size();
    result.missing_pairs = missing.size();
    result.extra_pairs = extra.size();
    result.first_duplicates = SortedExamples(duplicates, example_limit);
    for (const CanonicalPair& pair : result.first_duplicates)
    {
        const auto occurrence = payload_occurrences.find(pair);
        if (occurrence == payload_occurrences.end())
        {
            continue;
        }
        result.first_duplicate_occurrences.insert(
            result.first_duplicate_occurrences.end(),
            occurrence->second.begin(), occurrence->second.end());
    }
    using SourceKey =
        std::tuple<size_t, size_t, int, int, int, int, int>;
    std::map<SourceKey, CanonicalPairSourceSummary> source_summaries;
    for (const auto& [pair, occurrences] : payload_occurrences)
    {
        const bool duplicate = payload_counts.at(pair) > 1;
        for (const CanonicalPairOccurrence& occurrence : occurrences)
        {
            const SourceKey key = {
                occurrence.sci_index, occurrence.packed_index,
                occurrence.split, occurrence.jm, occurrence.i_local,
                occurrence.cluster_i, occurrence.cluster_j};
            auto [summary, inserted] = source_summaries.try_emplace(
                key,
                CanonicalPairSourceSummary{
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
                    occurrence.exclusion_hash});
            summary->second.accepted_pairs += 1;
            if (duplicate)
            {
                summary->second.duplicate_pairs += 1;
            }
        }
    }
    for (const auto& [key, summary] : source_summaries)
    {
        (void)key;
        if (summary.duplicate_pairs != 0)
        {
            result.duplicate_source_summaries.push_back(summary);
        }
    }
    result.first_missing = SortedExamples(missing, example_limit);
    result.first_extra = SortedExamples(extra, example_limit);
    result.matched = result.duplicate_payload_pairs == 0 &&
                     result.missing_pairs == 0 && result.extra_pairs == 0;
    return result;
}

}  // namespace nbnxm_microbench
