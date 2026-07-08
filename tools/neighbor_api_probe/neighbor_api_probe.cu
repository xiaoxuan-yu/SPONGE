#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <thrust/universal_vector.h>

#include "Lennard_Jones_force/clustered_lj_count_experiments.h"

#include "cstone/cuda/stream_holder.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/focus/source_center.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/traversal/ijloop/common.hpp"
#include "cstone/traversal/ijloop/compressneighbors.cuh"
#include "cstone/traversal/ijloop/gpu_superclusternblist.cuh"
#include "cstone/traversal/groups.hpp"
#include "cstone/tree/csarray.hpp"
#include "cstone/tree/octree.hpp"

int CONTROLLER::MPI_rank = 0;

namespace
{

using Clock = std::chrono::high_resolution_clock;
using KeyType = std::uint64_t;
using SfcKey = cstone::SfcKind<KeyType>;

struct Options
{
    std::string system = "wat160k";
    std::string compression = "none";
    std::string coordinateFile;
    std::size_t n = 0;
    unsigned bucketSize = 64;
    unsigned ncmax = 360;
    unsigned groupSize = cstone::TravConfig::targetSize;
    unsigned warmup = 1;
    unsigned iters = 5;
    unsigned seed = 42;
    float boxLength = 1.0f;
    float h = 0.0f;
    float targetNeighbors = 180.0f;
    float searchExtFactor = 1.0f;
    float spongeCutoffFactor = 2.0f;
    bool symmetric = false;
    bool periodic = true;
    bool compareSpongeInternal = false;
    unsigned spongeRepeat = 1;
    unsigned spongeOnepassCapacityFactor = 32;
    std::string spongeCountMode = "static";
    bool spongePayloadStats = false;
};

template<class T>
void CheckCuda(T code, const char* what)
{
    if (code != cudaSuccess)
    {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(code));
        std::exit(1);
    }
}

std::size_t DefaultSystemSize(const std::string& system)
{
    if (system == "wat160k") return 164544;
    if (system == "wat600k") return 658176;
    if (system == "dna_cou") return 31662;
    throw std::runtime_error("unknown --system: " + system);
}

std::string DefaultCoordinateFile(const std::string& system)
{
    if (system == "wat160k")
        return "benchmarks/performance/wat/SPONGE_water_160k/water_npt_eq.gro";
    if (system == "wat600k")
        return "benchmarks/performance/wat/SPONGE_water_600k_2x2x1/water_npt_eq_2x2x1.gro";
    if (system == "dna_cou")
        return "benchmarks/performance/sinkmeta/statics/dna_cou_sinkmeta/Pmin_coordinate.txt";
    throw std::runtime_error("unknown --system: " + system);
}

void Usage(const char* argv0)
{
    std::fprintf(stderr,
                 "Usage: %s [--system wat160k|wat600k|dna_cou] [--n N]\n"
                 "          [--coordinate-file auto|PATH] [--compression none|nibble|band] [--symmetric 0|1]\n"
                 "          [--bucket-size N] [--ncmax N] [--group-size N]\n"
                 "          [--target-neighbors X | --h X] [--search-ext-factor X]\n"
                 "          [--warmup N] [--iters N] [--seed N] [--periodic 0|1]\n"
                 "          [--compare-sponge-internal 0|1] [--sponge-repeat N]\n"
                 "          [--sponge-cutoff-factor X] [--sponge-onepass-capacity-factor N]\n"
                 "          [--sponge-count-mode static|dynamic|slim|cooperative|coop|both|all]\n"
                 "          [--sponge-payload-stats 0|1]\n",
                 argv0);
}

bool ParseBool(const char* value)
{
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "False") != 0;
}

Options ParseArgs(int argc, char** argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i)
    {
        const std::string key = argv[i];
        const auto needValue = [&]() -> const char*
        {
            if (i + 1 >= argc)
            {
                Usage(argv[0]);
                throw std::runtime_error("missing value for " + key);
            }
            return argv[++i];
        };
        if (key == "--system") opt.system = needValue();
        else if (key == "--compression") opt.compression = needValue();
        else if (key == "--coordinate-file") opt.coordinateFile = needValue();
        else if (key == "--n") opt.n = std::stoull(needValue());
        else if (key == "--bucket-size") opt.bucketSize = std::stoul(needValue());
        else if (key == "--ncmax") opt.ncmax = std::stoul(needValue());
        else if (key == "--group-size") opt.groupSize = std::stoul(needValue());
        else if (key == "--warmup") opt.warmup = std::stoul(needValue());
        else if (key == "--iters") opt.iters = std::stoul(needValue());
        else if (key == "--seed") opt.seed = std::stoul(needValue());
        else if (key == "--box") opt.boxLength = std::stof(needValue());
        else if (key == "--h") opt.h = std::stof(needValue());
        else if (key == "--target-neighbors") opt.targetNeighbors = std::stof(needValue());
        else if (key == "--search-ext-factor") opt.searchExtFactor = std::stof(needValue());
        else if (key == "--sponge-cutoff-factor") opt.spongeCutoffFactor = std::stof(needValue());
        else if (key == "--sponge-repeat") opt.spongeRepeat = std::stoul(needValue());
        else if (key == "--sponge-onepass-capacity-factor")
            opt.spongeOnepassCapacityFactor = std::stoul(needValue());
        else if (key == "--sponge-count-mode") opt.spongeCountMode = needValue();
        else if (key == "--sponge-payload-stats") opt.spongePayloadStats = ParseBool(needValue());
        else if (key == "--symmetric") opt.symmetric = ParseBool(needValue());
        else if (key == "--periodic") opt.periodic = ParseBool(needValue());
        else if (key == "--compare-sponge-internal") opt.compareSpongeInternal = ParseBool(needValue());
        else if (key == "--help" || key == "-h")
        {
            Usage(argv[0]);
            std::exit(0);
        }
        else
        {
            Usage(argv[0]);
            throw std::runtime_error("unknown option: " + key);
        }
    }

    if (opt.coordinateFile == "auto") opt.coordinateFile = DefaultCoordinateFile(opt.system);
    if (opt.n == 0 && opt.coordinateFile.empty()) opt.n = DefaultSystemSize(opt.system);
    if (opt.compression != "none" && opt.compression != "nibble" && opt.compression != "band")
        throw std::runtime_error("invalid --compression: " + opt.compression);
    if (opt.iters == 0) throw std::runtime_error("--iters must be positive");
    if (opt.groupSize == 0) throw std::runtime_error("--group-size must be positive");
    if (opt.spongeRepeat == 0) throw std::runtime_error("--sponge-repeat must be positive");
    if (opt.spongeOnepassCapacityFactor == 0)
        throw std::runtime_error("--sponge-onepass-capacity-factor must be positive");
    if (opt.spongeCountMode != "static" && opt.spongeCountMode != "dynamic" &&
        opt.spongeCountMode != "slim" &&
        opt.spongeCountMode != "cooperative" &&
        opt.spongeCountMode != "coop" && opt.spongeCountMode != "both" &&
        opt.spongeCountMode != "all")
        throw std::runtime_error("invalid --sponge-count-mode: " + opt.spongeCountMode);

    return opt;
}

cstone::Box<float> MakeBox(const Options& opt)
{
    const auto boundary = opt.periodic ? cstone::BoundaryType::periodic : cstone::BoundaryType::open;
    return cstone::Box<float>(0.0f, opt.boxLength, boundary);
}

float ResolveH(const Options& opt, std::size_t n)
{
    if (opt.h > 0.0f) return opt.h;
    const double volume = double(opt.boxLength) * opt.boxLength * opt.boxLength;
    const double radius =
        std::cbrt(double(opt.targetNeighbors) * volume / ((4.0 / 3.0) * M_PI * double(n)));
    return static_cast<float>(0.5 * radius);
}

struct HostCoords
{
    std::vector<float> x, y, z;
    std::vector<KeyType> keys;
};

float WrapUnit(float value)
{
    value -= std::floor(value);
    return value >= 1.0f ? 0.0f : value;
}

HostCoords SortCoordsBySfc(HostCoords coords, const cstone::Box<float>& box)
{
    const std::size_t n = coords.x.size();
    coords.keys.resize(n);

    cstone::computeSfcKeys(coords.x.data(), coords.y.data(), coords.z.data(),
                           reinterpret_cast<SfcKey*>(coords.keys.data()), n, box);

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return coords.keys[a] < coords.keys[b]; });

    HostCoords sorted;
    sorted.x.resize(n);
    sorted.y.resize(n);
    sorted.z.resize(n);
    sorted.keys.resize(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        const std::size_t src = order[i];
        sorted.x[i] = coords.x[src];
        sorted.y[i] = coords.y[src];
        sorted.z[i] = coords.z[src];
        sorted.keys[i] = coords.keys[src];
    }
    return sorted;
}

HostCoords GenerateUniformSfcSortedCoords(const Options& opt, const cstone::Box<float>& box)
{
    HostCoords coords;
    coords.x.resize(opt.n);
    coords.y.resize(opt.n);
    coords.z.resize(opt.n);

    std::mt19937 gen(opt.seed);
    std::uniform_real_distribution<float> dist(0.0f, opt.boxLength);
    for (std::size_t i = 0; i < opt.n; ++i)
    {
        coords.x[i] = dist(gen);
        coords.y[i] = dist(gen);
        coords.z[i] = dist(gen);
    }

    return SortCoordsBySfc(std::move(coords), box);
}

HostCoords LoadGroCoords(const Options& opt, const std::string& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open coordinate file: " + path);

    std::string line;
    std::getline(in, line);
    std::getline(in, line);
    const std::size_t fileAtoms = std::stoull(line);
    const std::size_t n = opt.n > 0 ? std::min(opt.n, fileAtoms) : fileAtoms;
    HostCoords coords;
    coords.x.reserve(n);
    coords.y.reserve(n);
    coords.z.reserve(n);
    std::vector<float> rawX;
    std::vector<float> rawY;
    std::vector<float> rawZ;
    rawX.reserve(n);
    rawY.reserve(n);
    rawZ.reserve(n);

    for (std::size_t i = 0; i < fileAtoms; ++i)
    {
        std::getline(in, line);
        if (i >= n) continue;
        std::istringstream iss(line);
        std::string residue;
        std::string atomName;
        int atomId = 0;
        float x = 0.0f, y = 0.0f, z = 0.0f;
        iss >> residue >> atomName >> atomId >> x >> y >> z;
        if (!iss) throw std::runtime_error("failed to parse .gro atom line in " + path);
        rawX.push_back(x);
        rawY.push_back(y);
        rawZ.push_back(z);
    }

    std::getline(in, line);
    std::istringstream boxLine(line);
    float lx = 0.0f, ly = 0.0f, lz = 0.0f;
    boxLine >> lx >> ly >> lz;
    if (!boxLine || lx <= 0.0f || ly <= 0.0f || lz <= 0.0f)
        throw std::runtime_error("failed to parse .gro box line in " + path);

    coords.x.resize(n);
    coords.y.resize(n);
    coords.z.resize(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        coords.x[i] = WrapUnit(rawX[i] / lx);
        coords.y[i] = WrapUnit(rawY[i] / ly);
        coords.z[i] = WrapUnit(rawZ[i] / lz);
    }
    return coords;
}

HostCoords LoadSpongeCoordinateCoords(const Options& opt, const std::string& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open coordinate file: " + path);

    std::string line;
    std::getline(in, line);
    std::istringstream header(line);
    std::size_t fileAtoms = 0;
    header >> fileAtoms;
    if (!header || fileAtoms == 0) throw std::runtime_error("failed to parse coordinate header in " + path);
    const std::size_t n = opt.n > 0 ? std::min(opt.n, fileAtoms) : fileAtoms;

    HostCoords coords;
    std::vector<float> rawX(n), rawY(n), rawZ(n);
    for (std::size_t i = 0; i < fileAtoms; ++i)
    {
        std::getline(in, line);
        if (i >= n) continue;
        std::istringstream iss(line);
        iss >> rawX[i] >> rawY[i] >> rawZ[i];
        if (!iss) throw std::runtime_error("failed to parse coordinate atom line in " + path);
    }

    std::getline(in, line);
    std::istringstream boxLine(line);
    float lx = 0.0f, ly = 0.0f, lz = 0.0f;
    boxLine >> lx >> ly >> lz;
    if (!boxLine || lx <= 0.0f || ly <= 0.0f || lz <= 0.0f)
        throw std::runtime_error("failed to parse coordinate box line in " + path);

    coords.x.resize(n);
    coords.y.resize(n);
    coords.z.resize(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        coords.x[i] = WrapUnit(rawX[i] / lx);
        coords.y[i] = WrapUnit(rawY[i] / ly);
        coords.z[i] = WrapUnit(rawZ[i] / lz);
    }
    return coords;
}

HostCoords LoadCoordinateFileSfcSortedCoords(const Options& opt, const cstone::Box<float>& box)
{
    HostCoords coords;
    if (opt.coordinateFile.size() >= 4 &&
        opt.coordinateFile.substr(opt.coordinateFile.size() - 4) == ".gro")
    {
        coords = LoadGroCoords(opt, opt.coordinateFile);
    }
    else
    {
        coords = LoadSpongeCoordinateCoords(opt, opt.coordinateFile);
    }
    return SortCoordsBySfc(std::move(coords), box);
}

struct DeviceViews
{
    cstone::OctreeNsView<float, KeyType> nsView;
    cstone::GroupView groupView;
};

struct PreparedData
{
    std::size_t n = 0;
    HostCoords hostCoords;
    cstone::OctreeData<KeyType, cstone::execution::Cpu> octree;
    std::vector<cstone::LocalIndex> layout;
    std::vector<cstone::Vec3<float>> centers;
    std::vector<cstone::Vec3<float>> sizes;

    thrust::universal_vector<float> dX, dY, dZ;
    thrust::universal_vector<KeyType> dCodes;
    thrust::universal_vector<KeyType> dPrefixes;
    thrust::universal_vector<cstone::TreeNodeIndex> dChildOffsets;
    thrust::universal_vector<cstone::TreeNodeIndex> dParents;
    thrust::universal_vector<cstone::TreeNodeIndex> dInternalToLeaf;
    thrust::universal_vector<cstone::TreeNodeIndex> dLeafToInternal;
    thrust::universal_vector<cstone::TreeNodeIndex> dLevelRange;
    thrust::universal_vector<cstone::LocalIndex> dLayout;
    thrust::universal_vector<cstone::Vec3<float>> dCenters;
    thrust::universal_vector<cstone::Vec3<float>> dSizes;
    thrust::universal_vector<cstone::LocalIndex> dGroups;
};

PreparedData PrepareData(const Options& opt, const cstone::Box<float>& box)
{
    HostCoords coords = opt.coordinateFile.empty() ? GenerateUniformSfcSortedCoords(opt, box)
                                                   : LoadCoordinateFileSfcSortedCoords(opt, box);
    auto [csTree, counts] = cstone::computeOctree(std::span<const KeyType>(coords.keys), opt.bucketSize);

    PreparedData data;
    data.hostCoords = std::move(coords);
    data.n = data.hostCoords.x.size();
    data.octree.resize(cstone::nNodes(csTree));
    cstone::updateInternalTree<KeyType>(csTree, data.octree.data());

    data.layout.assign(cstone::nNodes(csTree) + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), data.layout.begin() + 1);

    data.centers.resize(data.octree.numNodes);
    data.sizes.resize(data.octree.numNodes);
    cstone::nodeFpCenters<KeyType>(std::span<const KeyType>(data.octree.prefixes.data(), data.octree.numNodes),
                                   data.centers.data(), data.sizes.data(), box);

    data.dX = thrust::universal_vector<float>(data.hostCoords.x.begin(), data.hostCoords.x.end());
    data.dY = thrust::universal_vector<float>(data.hostCoords.y.begin(), data.hostCoords.y.end());
    data.dZ = thrust::universal_vector<float>(data.hostCoords.z.begin(), data.hostCoords.z.end());
    data.dCodes = thrust::universal_vector<KeyType>(data.hostCoords.keys.begin(), data.hostCoords.keys.end());
    data.dPrefixes = thrust::universal_vector<KeyType>(data.octree.prefixes.begin(), data.octree.prefixes.end());
    data.dChildOffsets = thrust::universal_vector<cstone::TreeNodeIndex>(data.octree.childOffsets.begin(),
                                                                         data.octree.childOffsets.end());
    data.dParents = thrust::universal_vector<cstone::TreeNodeIndex>(data.octree.parents.begin(),
                                                                    data.octree.parents.end());
    data.dInternalToLeaf = thrust::universal_vector<cstone::TreeNodeIndex>(data.octree.internalToLeaf.begin(),
                                                                           data.octree.internalToLeaf.end());
    data.dLeafToInternal = thrust::universal_vector<cstone::TreeNodeIndex>(data.octree.leafToInternal.begin(),
                                                                           data.octree.leafToInternal.end());
    data.dLevelRange = thrust::universal_vector<cstone::TreeNodeIndex>(data.octree.levelRange.begin(),
                                                                       data.octree.levelRange.end());
    data.dLayout = thrust::universal_vector<cstone::LocalIndex>(data.layout.begin(), data.layout.end());
    data.dCenters = thrust::universal_vector<cstone::Vec3<float>>(data.centers.begin(), data.centers.end());
    data.dSizes = thrust::universal_vector<cstone::Vec3<float>>(data.sizes.begin(), data.sizes.end());

    const std::size_t numGroups = (data.n + opt.groupSize - 1) / opt.groupSize;
    data.dGroups.resize(numGroups + 1);
    for (std::size_t i = 0; i <= numGroups; ++i)
        data.dGroups[i] = std::min<std::size_t>(i * opt.groupSize, data.n);

    return data;
}

DeviceViews MakeViews(const Options& opt, PreparedData& data)
{
    DeviceViews views;
    views.nsView = {.numLeafNodes = data.octree.numLeafNodes,
                    .numNodes = data.octree.numNodes,
                    .prefixes = cstone::rawPtr(data.dPrefixes),
                    .childOffsets = cstone::rawPtr(data.dChildOffsets),
                    .parents = cstone::rawPtr(data.dParents),
                    .internalToLeaf = cstone::rawPtr(data.dInternalToLeaf),
                    .leafToInternal = cstone::rawPtr(data.dLeafToInternal),
                    .levelRange = cstone::rawPtr(data.dLevelRange),
                    .leaves = cstone::rawPtr(data.dCodes),
                    .layout = cstone::rawPtr(data.dLayout),
                    .centers = cstone::rawPtr(data.dCenters),
                    .sizes = cstone::rawPtr(data.dSizes),
                    .searchExtFactor = opt.searchExtFactor};
    views.groupView = {.firstBody = 0,
                       .lastBody = static_cast<cstone::LocalIndex>(data.n),
                       .numGroups = static_cast<cstone::LocalIndex>(data.dGroups.size() - 1),
                       .groupStart = cstone::rawPtr(data.dGroups),
                       .groupEnd = cstone::rawPtr(data.dGroups) + 1};
    return views;
}

struct SpongeInternalData
{
    int paddedAtoms = 0;
    int localAtomCount = 0;
    int clusterCount = 0;
    int superclusterCount = 0;
    int candidateSciCount = 0;
    int leafCount = 0;
    int candidateLeafClusterStride = 0;
    int maxLeafClusterSpan = 0;
    int onepassCapacity = 0;
    int candidateLeafCount = 0;
    int collectOverflow = 0;
    float cutoff = 0.0f;
    LTMatrix3 cell;
    LTMatrix3 rcell;

    thrust::universal_vector<VECTOR> crd;
    thrust::universal_vector<int> permutation;
    thrust::universal_vector<int> clusterOffsets;
    thrust::universal_vector<int> leafClusterStarts;
    thrust::universal_vector<int> leafClusterEnds;
    thrust::universal_vector<int> leafAllLocal;
    thrust::universal_vector<int> superClusterOffsets;
    thrust::universal_vector<int> clusterToSupercluster;
    thrust::universal_vector<int> sciSuperclusterIds;
    thrust::universal_vector<unsigned int> clusterValidMasks;
    thrust::universal_vector<unsigned int> clusterLocalMasks;
    thrust::universal_vector<VECTOR> clusterCenters;
    thrust::universal_vector<VECTOR> clusterExtents;
    thrust::universal_vector<VECTOR> superClusterCenters;
    thrust::universal_vector<VECTOR> superClusterSizes;
    thrust::universal_vector<KeyType> nodePrefixes;
    thrust::universal_vector<int> childOffsets;
    thrust::universal_vector<int> parents;
    thrust::universal_vector<int> internalToLeaf;

    thrust::universal_vector<int> collectCounts;
    thrust::universal_vector<LJ_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_PROBE_RECORD> collectRecords;
    thrust::universal_vector<int> collectCursor;
    thrust::universal_vector<int> collectOverflowCounter;

    thrust::universal_vector<int> candidateLeafOffsets;
    thrust::universal_vector<int> candidateLeafIds;
    thrust::universal_vector<int> candidateLeafPrevRunningMaxEnds;
    thrust::universal_vector<unsigned int> candidateLeafReachMasks;

    thrust::universal_vector<int> sciShiftFlags;
    thrust::universal_vector<int> cjpackedGroupCounts;
    thrust::universal_vector<int> exclusionCounts;
    thrust::universal_vector<int> recordStreamSourceRows;
    thrust::universal_vector<int> recordStreamSourceCountsByCandidate;
    thrust::universal_vector<LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT> countSourceFragments;
    thrust::universal_vector<LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT_SLIM> countSlimSourceFragments;
    thrust::universal_vector<int> countSourceFragmentCursor;
    thrust::universal_vector<int> countSourceFragmentOverflowRows;
    thrust::universal_vector<int> dynamicWorkCounter;
};

unsigned int LowMask(int count)
{
    if (count <= 0) return 0u;
    if (count >= 32) return 0xffffffffu;
    return (1u << static_cast<unsigned>(count)) - 1u;
}

VECTOR MakeVector(float x, float y, float z) { return VECTOR{x, y, z}; }

bool ClusterAabbOverlapsShifted(VECTOR centerI, VECTOR extentI, VECTOR centerJ,
                                VECTOR extentJ, float cutoff, VECTOR shiftVec)
{
    const float drX = centerJ.x - (centerI.x + shiftVec.x);
    const float drY = centerJ.y - (centerI.y + shiftVec.y);
    const float drZ = centerJ.z - (centerI.z + shiftVec.z);
    const float gapX = std::max(std::fabs(drX) - (extentI.x + extentJ.x), 0.0f);
    const float gapY = std::max(std::fabs(drY) - (extentI.y + extentJ.y), 0.0f);
    const float gapZ = std::max(std::fabs(drZ) - (extentI.z + extentJ.z), 0.0f);
    return gapX * gapX + gapY * gapY + gapZ * gapZ <= cutoff * cutoff;
}

unsigned int BuildFixedShiftClusterIMask(const SpongeInternalData& sponge,
                                         int clusterIStart, int clusterIEnd,
                                         int clusterJ, int fixedShiftId)
{
    const VECTOR shiftVec = Clustered_Shift_Vector_From_Id(fixedShiftId, sponge.cell);
    const VECTOR centerJ = sponge.clusterCenters[clusterJ];
    const VECTOR extentJ = sponge.clusterExtents[clusterJ];
    unsigned int iMask = 0u;
    for (int clusterI = clusterIStart; clusterI < clusterIEnd; ++clusterI)
    {
        const int iLocal = clusterI - clusterIStart;
        if (sponge.clusterLocalMasks[clusterI] == 0u) continue;
        if (fixedShiftId == kClusteredCentralShiftId &&
            clusterJ >= clusterIStart && clusterJ < clusterIEnd &&
            clusterI > clusterJ)
        {
            continue;
        }
        if (ClusterAabbOverlapsShifted(sponge.clusterCenters[clusterI],
                                       sponge.clusterExtents[clusterI],
                                       centerJ, extentJ, sponge.cutoff,
                                       shiftVec))
        {
            iMask |= (1u << static_cast<unsigned int>(iLocal));
        }
    }
    return iMask;
}

SpongeInternalData PrepareSpongeInternalData(const Options& opt, const PreparedData& data, float h)
{
    SpongeInternalData sponge;
    sponge.cutoff = opt.spongeCutoffFactor * h;
    sponge.localAtomCount = static_cast<int>(data.n);
    sponge.cell = LTMatrix3(1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f);
    sponge.rcell = LTMatrix3(1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f);
    sponge.clusterCount = static_cast<int>((data.n + kClusteredClusterSize - 1) / kClusteredClusterSize);
    sponge.paddedAtoms = sponge.clusterCount * kClusteredClusterSize;
    sponge.superclusterCount =
        (sponge.clusterCount + kClusteredSuperClusterClusters - 1) / kClusteredSuperClusterClusters;
    sponge.candidateSciCount = sponge.superclusterCount * kClusteredShiftCount;
    sponge.leafCount = static_cast<int>(data.octree.numLeafNodes);

    std::vector<VECTOR> crd(static_cast<std::size_t>(sponge.paddedAtoms), MakeVector(0.0f, 0.0f, 0.0f));
    std::vector<int> permutation(static_cast<std::size_t>(sponge.paddedAtoms), -1);
    for (std::size_t i = 0; i < data.n; ++i)
    {
        crd[i] = MakeVector(data.hostCoords.x[i], data.hostCoords.y[i], data.hostCoords.z[i]);
        permutation[i] = static_cast<int>(i);
    }
    sponge.crd = thrust::universal_vector<VECTOR>(crd.begin(), crd.end());
    sponge.permutation = thrust::universal_vector<int>(permutation.begin(), permutation.end());

    std::vector<int> clusterOffsets(static_cast<std::size_t>(sponge.clusterCount + 1), 0);
    std::vector<unsigned int> validMasks(static_cast<std::size_t>(sponge.clusterCount), 0u);
    std::vector<unsigned int> localMasks(static_cast<std::size_t>(sponge.clusterCount), 0u);
    std::vector<VECTOR> clusterCenters(static_cast<std::size_t>(sponge.clusterCount), MakeVector(0.0f, 0.0f, 0.0f));
    std::vector<VECTOR> clusterExtents(static_cast<std::size_t>(sponge.clusterCount), MakeVector(0.0f, 0.0f, 0.0f));
    for (int c = 0; c <= sponge.clusterCount; ++c) clusterOffsets[c] = c * kClusteredClusterSize;
    for (int c = 0; c < sponge.clusterCount; ++c)
    {
        const int begin = c * kClusteredClusterSize;
        const int end = std::min<int>(begin + kClusteredClusterSize, static_cast<int>(data.n));
        const int valid = std::max(0, end - begin);
        validMasks[c] = LowMask(valid);
        localMasks[c] = validMasks[c];
        if (valid == 0) continue;

        float minX = crd[begin].x, maxX = crd[begin].x;
        float minY = crd[begin].y, maxY = crd[begin].y;
        float minZ = crd[begin].z, maxZ = crd[begin].z;
        for (int i = begin + 1; i < end; ++i)
        {
            minX = std::min(minX, crd[i].x);
            maxX = std::max(maxX, crd[i].x);
            minY = std::min(minY, crd[i].y);
            maxY = std::max(maxY, crd[i].y);
            minZ = std::min(minZ, crd[i].z);
            maxZ = std::max(maxZ, crd[i].z);
        }
        clusterCenters[c] = MakeVector(0.5f * (minX + maxX), 0.5f * (minY + maxY), 0.5f * (minZ + maxZ));
        clusterExtents[c] = MakeVector(0.5f * (maxX - minX), 0.5f * (maxY - minY), 0.5f * (maxZ - minZ));
    }
    sponge.clusterOffsets = thrust::universal_vector<int>(clusterOffsets.begin(), clusterOffsets.end());
    sponge.clusterValidMasks = thrust::universal_vector<unsigned int>(validMasks.begin(), validMasks.end());
    sponge.clusterLocalMasks = thrust::universal_vector<unsigned int>(localMasks.begin(), localMasks.end());
    sponge.clusterCenters = thrust::universal_vector<VECTOR>(clusterCenters.begin(), clusterCenters.end());
    sponge.clusterExtents = thrust::universal_vector<VECTOR>(clusterExtents.begin(), clusterExtents.end());

    std::vector<int> superOffsets(static_cast<std::size_t>(sponge.superclusterCount + 1), 0);
    std::vector<int> clusterToSuper(static_cast<std::size_t>(sponge.clusterCount), 0);
    std::vector<int> sciSuperIds(static_cast<std::size_t>(sponge.superclusterCount), 0);
    std::vector<VECTOR> superCenters(static_cast<std::size_t>(sponge.superclusterCount), MakeVector(0.0f, 0.0f, 0.0f));
    std::vector<VECTOR> superSizes(static_cast<std::size_t>(sponge.superclusterCount), MakeVector(0.0f, 0.0f, 0.0f));
    for (int s = 0; s <= sponge.superclusterCount; ++s)
        superOffsets[s] = std::min(s * kClusteredSuperClusterClusters, sponge.clusterCount);
    for (int s = 0; s < sponge.superclusterCount; ++s)
    {
        sciSuperIds[s] = s;
        const int begin = superOffsets[s];
        const int end = superOffsets[s + 1];
        for (int c = begin; c < end; ++c) clusterToSuper[c] = s;
        if (begin == end) continue;

        VECTOR centerSum = MakeVector(0.0f, 0.0f, 0.0f);
        int validClusters = 0;
        for (int c = begin; c < end; ++c)
        {
            if (validMasks[c] == 0u) continue;
            centerSum.x += clusterCenters[c].x;
            centerSum.y += clusterCenters[c].y;
            centerSum.z += clusterCenters[c].z;
            validClusters += 1;
        }
        if (validClusters == 0) continue;
        const float inv = 1.0f / static_cast<float>(validClusters);
        const VECTOR center = MakeVector(centerSum.x * inv, centerSum.y * inv, centerSum.z * inv);
        VECTOR size = MakeVector(0.0f, 0.0f, 0.0f);
        for (int c = begin; c < end; ++c)
        {
            if (validMasks[c] == 0u) continue;
            size.x = std::max(size.x, std::fabs(clusterCenters[c].x - center.x) + clusterExtents[c].x);
            size.y = std::max(size.y, std::fabs(clusterCenters[c].y - center.y) + clusterExtents[c].y);
            size.z = std::max(size.z, std::fabs(clusterCenters[c].z - center.z) + clusterExtents[c].z);
        }
        size.x = std::min(0.5f, size.x + sponge.cutoff);
        size.y = std::min(0.5f, size.y + sponge.cutoff);
        size.z = std::min(0.5f, size.z + sponge.cutoff);
        superCenters[s] = center;
        superSizes[s] = size;
    }
    sponge.superClusterOffsets = thrust::universal_vector<int>(superOffsets.begin(), superOffsets.end());
    sponge.clusterToSupercluster = thrust::universal_vector<int>(clusterToSuper.begin(), clusterToSuper.end());
    sponge.sciSuperclusterIds = thrust::universal_vector<int>(sciSuperIds.begin(), sciSuperIds.end());
    sponge.superClusterCenters = thrust::universal_vector<VECTOR>(superCenters.begin(), superCenters.end());
    sponge.superClusterSizes = thrust::universal_vector<VECTOR>(superSizes.begin(), superSizes.end());

    std::vector<int> leafStarts(static_cast<std::size_t>(sponge.leafCount), 0);
    std::vector<int> leafEnds(static_cast<std::size_t>(sponge.leafCount), 0);
    std::vector<int> leafAllLocal(static_cast<std::size_t>(sponge.leafCount), 1);
    for (int leaf = 0; leaf < sponge.leafCount; ++leaf)
    {
        const int atomBegin = static_cast<int>(data.layout[leaf]);
        const int atomEnd = static_cast<int>(data.layout[leaf + 1]);
        leafStarts[leaf] = atomBegin / kClusteredClusterSize;
        leafEnds[leaf] = (atomEnd + kClusteredClusterSize - 1) / kClusteredClusterSize;
        sponge.maxLeafClusterSpan = std::max(sponge.maxLeafClusterSpan, leafEnds[leaf] - leafStarts[leaf]);
    }
    sponge.candidateLeafClusterStride = std::max(1, sponge.maxLeafClusterSpan);
    sponge.leafClusterStarts = thrust::universal_vector<int>(leafStarts.begin(), leafStarts.end());
    sponge.leafClusterEnds = thrust::universal_vector<int>(leafEnds.begin(), leafEnds.end());
    sponge.leafAllLocal = thrust::universal_vector<int>(leafAllLocal.begin(), leafAllLocal.end());

    sponge.nodePrefixes = thrust::universal_vector<KeyType>(data.octree.prefixes.begin(), data.octree.prefixes.end());
    sponge.childOffsets.assign(data.octree.childOffsets.begin(), data.octree.childOffsets.end());
    sponge.parents.assign(data.octree.parents.begin(), data.octree.parents.end());
    sponge.internalToLeaf.assign(data.octree.internalToLeaf.begin(), data.octree.internalToLeaf.end());

    sponge.onepassCapacity = std::max(1, sponge.candidateSciCount * static_cast<int>(opt.spongeOnepassCapacityFactor));
    sponge.collectCounts.resize(sponge.candidateSciCount);
    sponge.collectRecords.resize(sponge.onepassCapacity);
    sponge.collectCursor.resize(1);
    sponge.collectOverflowCounter.resize(1);
    sponge.sciShiftFlags.resize(sponge.candidateSciCount);
    sponge.cjpackedGroupCounts.resize(sponge.candidateSciCount);
    sponge.exclusionCounts.resize(sponge.candidateSciCount);
    sponge.recordStreamSourceRows.resize(1);
    sponge.recordStreamSourceCountsByCandidate.resize(sponge.candidateSciCount);
    sponge.countSourceFragmentCursor.resize(1);
    sponge.countSourceFragmentOverflowRows.resize(1);
    sponge.dynamicWorkCounter.resize(1);
    return sponge;
}

void PrefetchSpongeInternalData(SpongeInternalData& sponge, cstone::StreamHolder& stream)
{
    int device = 0;
    CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
    const auto prefetch = [&](auto& vector)
    {
        if (!vector.empty())
        {
            CheckCuda(cudaMemPrefetchAsync(
                          cstone::rawPtr(vector),
                          vector.size() * sizeof(typename std::decay_t<decltype(vector)>::value_type),
                          {cudaMemLocationTypeDevice, device}, 0, stream),
                      "cudaMemPrefetchAsync");
        }
    };
    prefetch(sponge.crd);
    prefetch(sponge.permutation);
    prefetch(sponge.clusterOffsets);
    prefetch(sponge.leafClusterStarts);
    prefetch(sponge.leafClusterEnds);
    prefetch(sponge.leafAllLocal);
    prefetch(sponge.superClusterOffsets);
    prefetch(sponge.clusterToSupercluster);
    prefetch(sponge.sciSuperclusterIds);
    prefetch(sponge.clusterValidMasks);
    prefetch(sponge.clusterLocalMasks);
    prefetch(sponge.clusterCenters);
    prefetch(sponge.clusterExtents);
    prefetch(sponge.superClusterCenters);
    prefetch(sponge.superClusterSizes);
    prefetch(sponge.nodePrefixes);
    prefetch(sponge.childOffsets);
    prefetch(sponge.parents);
    prefetch(sponge.internalToLeaf);
    prefetch(sponge.collectCounts);
    prefetch(sponge.collectRecords);
    prefetch(sponge.collectCursor);
    prefetch(sponge.collectOverflowCounter);
    prefetch(sponge.candidateLeafOffsets);
    prefetch(sponge.candidateLeafIds);
    prefetch(sponge.candidateLeafPrevRunningMaxEnds);
    prefetch(sponge.candidateLeafReachMasks);
    prefetch(sponge.sciShiftFlags);
    prefetch(sponge.cjpackedGroupCounts);
    prefetch(sponge.exclusionCounts);
    prefetch(sponge.recordStreamSourceRows);
    prefetch(sponge.recordStreamSourceCountsByCandidate);
    prefetch(sponge.countSourceFragmentCursor);
    prefetch(sponge.countSourceFragmentOverflowRows);
    prefetch(sponge.countSourceFragments);
    prefetch(sponge.countSlimSourceFragments);
    prefetch(sponge.dynamicWorkCounter);
    stream.sync();
}

void RunSpongeCandidateCollect(SpongeInternalData& sponge)
{
    CheckCuda(cudaMemset(cstone::rawPtr(sponge.collectCounts), 0,
                         sizeof(int) * static_cast<std::size_t>(sponge.candidateSciCount)),
              "cudaMemset collectCounts");
    CheckCuda(cudaMemset(cstone::rawPtr(sponge.collectCursor), 0, sizeof(int)), "cudaMemset collectCursor");
    CheckCuda(cudaMemset(cstone::rawPtr(sponge.collectOverflowCounter), 0, sizeof(int)),
              "cudaMemset collectOverflow");
    constexpr int builderBlockSize = 128;
    constexpr int groupsPerBlock = builderBlockSize / kClusteredSuperClusterClusters;
    const int blocks = (sponge.candidateSciCount + groupsPerBlock - 1) / groupsPerBlock;
    Launch_Clustered_Gmxpacked_Candidate_Leaf_Probe(
        ClusteredGmxpackedCandidateLeafProbeMode::Emit, blocks, builderBlockSize, sponge.candidateSciCount,
        cstone::rawPtr(sponge.sciSuperclusterIds), cstone::rawPtr(sponge.superClusterCenters),
        cstone::rawPtr(sponge.superClusterSizes), cstone::rawPtr(sponge.superClusterOffsets),
        cstone::rawPtr(sponge.leafClusterStarts), cstone::rawPtr(sponge.leafClusterEnds),
        cstone::rawPtr(sponge.leafAllLocal), sponge.cell, sponge.cutoff, cstone::rawPtr(sponge.clusterCenters),
        cstone::rawPtr(sponge.clusterExtents), cstone::rawPtr(sponge.clusterValidMasks),
        cstone::rawPtr(sponge.clusterLocalMasks), cstone::rawPtr(sponge.nodePrefixes),
        cstone::rawPtr(sponge.childOffsets), cstone::rawPtr(sponge.parents), cstone::rawPtr(sponge.internalToLeaf),
        nullptr, true, true, true, true, sponge.onepassCapacity, cstone::rawPtr(sponge.collectCounts),
        cstone::rawPtr(sponge.collectRecords), cstone::rawPtr(sponge.collectCursor),
        cstone::rawPtr(sponge.collectOverflowCounter));
}

void BuildSpongeCountInputsFromCollect(SpongeInternalData& sponge)
{
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize collect before scatter");
    const int emitted = sponge.collectCursor[0];
    sponge.collectOverflow = sponge.collectOverflowCounter[0];
    std::vector<int> counts(static_cast<std::size_t>(sponge.candidateSciCount), 0);
    for (int sci = 0; sci < sponge.candidateSciCount; ++sci) counts[sci] = sponge.collectCounts[sci];

    std::vector<int> offsets(static_cast<std::size_t>(sponge.candidateSciCount + 1), 0);
    std::partial_sum(counts.begin(), counts.end(), offsets.begin() + 1);
    sponge.candidateLeafCount = offsets.back();
    std::vector<int> ids(static_cast<std::size_t>(std::max(1, sponge.candidateLeafCount)), -1);
    std::vector<int> prev(static_cast<std::size_t>(std::max(1, sponge.candidateLeafCount)), 0);
    const int recordsToRead = std::min(emitted, sponge.onepassCapacity);
    for (int i = 0; i < recordsToRead; ++i)
    {
        const auto record = sponge.collectRecords[i];
        if (record.sci_id < 0 || record.sci_id >= sponge.candidateSciCount) continue;
        if (record.rank < 0 || record.rank >= counts[record.sci_id]) continue;
        const int index = offsets[record.sci_id] + record.rank;
        if (index < 0 || index >= sponge.candidateLeafCount) continue;
        ids[index] = record.leaf_id;
        prev[index] = record.prev_running_max_end;
    }

    std::vector<unsigned int> reachMasks(
        static_cast<std::size_t>(std::max(1, sponge.candidateLeafCount * sponge.candidateLeafClusterStride)), 0u);
    for (int sci = 0; sci < sponge.candidateSciCount; ++sci)
    {
        const int fixedShiftId = sci % kClusteredShiftCount;
        const int superId = sponge.sciSuperclusterIds[sci / kClusteredShiftCount];
        const int clusterIStart = sponge.superClusterOffsets[superId];
        const int clusterIEnd = sponge.superClusterOffsets[superId + 1];
        const int activeClusters = sponge.superClusterOffsets[superId + 1] - sponge.superClusterOffsets[superId];
        (void)activeClusters;
        for (int index = offsets[sci]; index < offsets[sci + 1]; ++index)
        {
            const int leaf = ids[index];
            if (leaf < 0 || leaf >= sponge.leafCount) continue;
            const int span = sponge.leafClusterEnds[leaf] - sponge.leafClusterStarts[leaf];
            for (int j = 0; j < span; ++j)
            {
                const int clusterJ = sponge.leafClusterStarts[leaf] + j;
                reachMasks[static_cast<std::size_t>(index) *
                               sponge.candidateLeafClusterStride +
                           j] = BuildFixedShiftClusterIMask(
                    sponge, clusterIStart, clusterIEnd, clusterJ, fixedShiftId);
            }
        }
    }

    sponge.candidateLeafOffsets = thrust::universal_vector<int>(offsets.begin(), offsets.end());
    sponge.candidateLeafIds = thrust::universal_vector<int>(ids.begin(), ids.end());
    sponge.candidateLeafPrevRunningMaxEnds = thrust::universal_vector<int>(prev.begin(), prev.end());
    sponge.candidateLeafReachMasks = thrust::universal_vector<unsigned int>(reachMasks.begin(), reachMasks.end());
    sponge.countSourceFragments.resize(static_cast<std::size_t>(std::max(1, sponge.candidateLeafCount * 8)));
    sponge.countSlimSourceFragments.resize(static_cast<std::size_t>(std::max(1, sponge.candidateLeafCount * 8)));
}

void RunSpongeFixedLightCount(SpongeInternalData& sponge, bool dynamic,
                              bool slim, bool cooperative)
{
    CheckCuda(cudaMemset(cstone::rawPtr(sponge.sciShiftFlags), 0,
                         sizeof(int) * static_cast<std::size_t>(sponge.candidateSciCount)),
              "cudaMemset sciShiftFlags");
    CheckCuda(cudaMemset(cstone::rawPtr(sponge.cjpackedGroupCounts), 0,
                         sizeof(int) * static_cast<std::size_t>(sponge.candidateSciCount)),
              "cudaMemset cjpackedGroupCounts");
    CheckCuda(cudaMemset(cstone::rawPtr(sponge.exclusionCounts), 0,
                         sizeof(int) * static_cast<std::size_t>(sponge.candidateSciCount)),
              "cudaMemset exclusionCounts");
    CheckCuda(cudaMemset(cstone::rawPtr(sponge.recordStreamSourceRows), 0, sizeof(int)),
              "cudaMemset recordStreamSourceRows");
    CheckCuda(cudaMemset(cstone::rawPtr(sponge.recordStreamSourceCountsByCandidate), 0,
                         sizeof(int) * static_cast<std::size_t>(sponge.candidateSciCount)),
              "cudaMemset recordStreamSourceCountsByCandidate");
    CheckCuda(cudaMemset(cstone::rawPtr(sponge.countSourceFragmentCursor), 0, sizeof(int)),
              "cudaMemset countSourceFragmentCursor");
    CheckCuda(cudaMemset(cstone::rawPtr(sponge.countSourceFragmentOverflowRows), 0, sizeof(int)),
              "cudaMemset countSourceFragmentOverflowRows");
    if (dynamic)
    {
        CheckCuda(cudaMemset(cstone::rawPtr(sponge.dynamicWorkCounter), 0, sizeof(int)),
                  "cudaMemset dynamicWorkCounter");
    }
    constexpr int builderBlockSize = 128;
    constexpr int warpsPerBlock = builderBlockSize / 32;
    const int staticBlocks = (sponge.candidateSciCount + warpsPerBlock - 1) / warpsPerBlock;
    int blocks = staticBlocks;
    if (cooperative)
    {
        Launch_Clustered_Gmxpacked_Count_Fixed_Light_Dedicated_Cooperative(
            blocks, builderBlockSize, sponge.candidateSciCount,
            kClusteredClusterSize, sponge.localAtomCount, sponge.cutoff,
            sponge.cell, sponge.rcell, cstone::rawPtr(sponge.crd),
            cstone::rawPtr(sponge.permutation),
            cstone::rawPtr(sponge.clusterOffsets),
            cstone::rawPtr(sponge.leafClusterStarts),
            cstone::rawPtr(sponge.leafClusterEnds),
            cstone::rawPtr(sponge.superClusterOffsets),
            cstone::rawPtr(sponge.clusterToSupercluster),
            cstone::rawPtr(sponge.sciSuperclusterIds),
            cstone::rawPtr(sponge.candidateLeafOffsets),
            cstone::rawPtr(sponge.candidateLeafIds),
            sponge.candidateLeafClusterStride,
            cstone::rawPtr(sponge.candidateLeafPrevRunningMaxEnds),
            cstone::rawPtr(sponge.candidateLeafReachMasks),
            cstone::rawPtr(sponge.clusterValidMasks),
            cstone::rawPtr(sponge.clusterLocalMasks),
            cstone::rawPtr(sponge.clusterCenters), nullptr, nullptr, nullptr,
            nullptr, nullptr, sponge.maxLeafClusterSpan,
            cstone::rawPtr(sponge.sciShiftFlags),
            cstone::rawPtr(sponge.cjpackedGroupCounts),
            cstone::rawPtr(sponge.exclusionCounts),
            cstone::rawPtr(sponge.recordStreamSourceRows),
            cstone::rawPtr(sponge.recordStreamSourceCountsByCandidate), true,
            cstone::rawPtr(sponge.countSourceFragments),
            static_cast<int>(sponge.countSourceFragments.size()),
            cstone::rawPtr(sponge.countSourceFragmentCursor),
            cstone::rawPtr(sponge.countSourceFragmentOverflowRows));
    }
    else if (slim)
    {
        Launch_Clustered_Gmxpacked_Count_Fixed_Light_Dedicated_Slim(
            blocks, builderBlockSize, sponge.candidateSciCount, kClusteredClusterSize,
            sponge.localAtomCount, sponge.cutoff, sponge.cell, sponge.rcell, cstone::rawPtr(sponge.crd),
            cstone::rawPtr(sponge.permutation), cstone::rawPtr(sponge.clusterOffsets),
            cstone::rawPtr(sponge.leafClusterStarts), cstone::rawPtr(sponge.leafClusterEnds),
            cstone::rawPtr(sponge.superClusterOffsets), cstone::rawPtr(sponge.clusterToSupercluster),
            cstone::rawPtr(sponge.sciSuperclusterIds), cstone::rawPtr(sponge.candidateLeafOffsets),
            cstone::rawPtr(sponge.candidateLeafIds), sponge.candidateLeafClusterStride,
            cstone::rawPtr(sponge.candidateLeafPrevRunningMaxEnds), cstone::rawPtr(sponge.candidateLeafReachMasks),
            cstone::rawPtr(sponge.clusterValidMasks), cstone::rawPtr(sponge.clusterLocalMasks),
            cstone::rawPtr(sponge.clusterCenters), nullptr, nullptr, nullptr, nullptr, nullptr,
            sponge.maxLeafClusterSpan, cstone::rawPtr(sponge.sciShiftFlags),
            cstone::rawPtr(sponge.cjpackedGroupCounts), cstone::rawPtr(sponge.exclusionCounts),
            cstone::rawPtr(sponge.recordStreamSourceRows),
            cstone::rawPtr(sponge.recordStreamSourceCountsByCandidate), true,
            cstone::rawPtr(sponge.countSlimSourceFragments),
            static_cast<int>(sponge.countSlimSourceFragments.size()),
            cstone::rawPtr(sponge.countSourceFragmentCursor),
            cstone::rawPtr(sponge.countSourceFragmentOverflowRows));
    }
    else if (dynamic)
    {
        int device = 0;
        int smCount = 0;
        CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
        CheckCuda(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device),
                  "cudaDeviceGetAttribute cudaDevAttrMultiProcessorCount");
        blocks = std::min(staticBlocks, std::max(1, smCount * 8));
        Launch_Clustered_Gmxpacked_Count_Fixed_Light_Dedicated_Dynamic(
            blocks, builderBlockSize, sponge.candidateSciCount, kClusteredClusterSize,
            sponge.localAtomCount, sponge.cutoff, sponge.cell, sponge.rcell, cstone::rawPtr(sponge.crd),
            cstone::rawPtr(sponge.permutation), cstone::rawPtr(sponge.clusterOffsets),
            cstone::rawPtr(sponge.leafClusterStarts), cstone::rawPtr(sponge.leafClusterEnds),
            cstone::rawPtr(sponge.superClusterOffsets), cstone::rawPtr(sponge.clusterToSupercluster),
            cstone::rawPtr(sponge.sciSuperclusterIds), cstone::rawPtr(sponge.candidateLeafOffsets),
            cstone::rawPtr(sponge.candidateLeafIds), sponge.candidateLeafClusterStride,
            cstone::rawPtr(sponge.candidateLeafPrevRunningMaxEnds), cstone::rawPtr(sponge.candidateLeafReachMasks),
            cstone::rawPtr(sponge.clusterValidMasks), cstone::rawPtr(sponge.clusterLocalMasks),
            cstone::rawPtr(sponge.clusterCenters), nullptr, nullptr, nullptr, nullptr, nullptr,
            sponge.maxLeafClusterSpan, cstone::rawPtr(sponge.sciShiftFlags),
            cstone::rawPtr(sponge.cjpackedGroupCounts), cstone::rawPtr(sponge.exclusionCounts),
            cstone::rawPtr(sponge.recordStreamSourceRows),
            cstone::rawPtr(sponge.recordStreamSourceCountsByCandidate), true,
            cstone::rawPtr(sponge.countSourceFragments), static_cast<int>(sponge.countSourceFragments.size()),
            cstone::rawPtr(sponge.countSourceFragmentCursor),
            cstone::rawPtr(sponge.countSourceFragmentOverflowRows), cstone::rawPtr(sponge.dynamicWorkCounter));
    }
    else
    {
        Launch_Clustered_Gmxpacked_Count_Fixed_Light_Dedicated(
            blocks, builderBlockSize, sponge.candidateSciCount, kClusteredClusterSize,
            sponge.localAtomCount, sponge.cutoff, sponge.cell, sponge.rcell, cstone::rawPtr(sponge.crd),
            cstone::rawPtr(sponge.permutation), cstone::rawPtr(sponge.clusterOffsets),
            cstone::rawPtr(sponge.leafClusterStarts), cstone::rawPtr(sponge.leafClusterEnds),
            cstone::rawPtr(sponge.superClusterOffsets), cstone::rawPtr(sponge.clusterToSupercluster),
            cstone::rawPtr(sponge.sciSuperclusterIds), cstone::rawPtr(sponge.candidateLeafOffsets),
            cstone::rawPtr(sponge.candidateLeafIds), sponge.candidateLeafClusterStride,
            cstone::rawPtr(sponge.candidateLeafPrevRunningMaxEnds), cstone::rawPtr(sponge.candidateLeafReachMasks),
            cstone::rawPtr(sponge.clusterValidMasks), cstone::rawPtr(sponge.clusterLocalMasks),
            cstone::rawPtr(sponge.clusterCenters), nullptr, nullptr, nullptr, nullptr, nullptr,
            sponge.maxLeafClusterSpan, cstone::rawPtr(sponge.sciShiftFlags),
            cstone::rawPtr(sponge.cjpackedGroupCounts), cstone::rawPtr(sponge.exclusionCounts),
            cstone::rawPtr(sponge.recordStreamSourceRows),
            cstone::rawPtr(sponge.recordStreamSourceCountsByCandidate), true,
            cstone::rawPtr(sponge.countSourceFragments), static_cast<int>(sponge.countSourceFragments.size()),
            cstone::rawPtr(sponge.countSourceFragmentCursor),
            cstone::rawPtr(sponge.countSourceFragmentOverflowRows));
    }
}

void PrintIntDistribution(const char* label, std::vector<int> values)
{
    if (values.empty())
    {
        std::printf("%s count=0\n", label);
        return;
    }
    std::sort(values.begin(), values.end());
    const long long sum =
        std::accumulate(values.begin(), values.end(), 0ll);
    const auto q = [&](double quantile)
    {
        const std::size_t idx = static_cast<std::size_t>(
            std::min<double>(values.size() - 1,
                             std::floor(quantile * (values.size() - 1))));
        return values[idx];
    };
    const int nonzero =
        static_cast<int>(values.end() - std::upper_bound(values.begin(),
                                                         values.end(), 0));
    std::printf(
        "%s count=%zu nonzero=%d mean=%.3f min=%d p50=%d p90=%d p99=%d max=%d\n",
        label, values.size(), nonzero,
        static_cast<double>(sum) / static_cast<double>(values.size()),
        values.front(), q(0.50), q(0.90), q(0.99), values.back());
}

void PrintSpongePayloadStats(SpongeInternalData& sponge)
{
    RunSpongeFixedLightCount(sponge, false, false, false);
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize payload stats count");

    std::vector<int> candidateLeaves;
    candidateLeaves.reserve(static_cast<std::size_t>(sponge.candidateSciCount));
    for (int sci = 0; sci < sponge.candidateSciCount; ++sci)
    {
        candidateLeaves.push_back(sponge.candidateLeafOffsets[sci + 1] -
                                  sponge.candidateLeafOffsets[sci]);
    }

    std::vector<int> leafSpans;
    std::vector<int> dedupSpans;
    std::vector<int> recordReachPopc;
    std::vector<int> recordValidJPopc;
    std::vector<int> recordsPerSci(static_cast<std::size_t>(sponge.candidateSciCount), 0);
    int prepruneRecords = 0;
    int skippedByReach = 0;
    int skippedByValidJ = 0;
    int skippedByHalfshell = 0;
    for (int sci = 0; sci < sponge.candidateSciCount; ++sci)
    {
        const int sciBase = sci / kClusteredShiftCount;
        const int superI = sponge.sciSuperclusterIds[sciBase];
        const int leafBegin = sponge.candidateLeafOffsets[sci];
        const int leafEnd = sponge.candidateLeafOffsets[sci + 1];
        for (int candidateIdx = leafBegin; candidateIdx < leafEnd; ++candidateIdx)
        {
            const int leaf = sponge.candidateLeafIds[candidateIdx];
            if (leaf < 0 || leaf >= sponge.leafCount) continue;
            const int clusterJStart = sponge.leafClusterStarts[leaf];
            const int clusterJEnd = sponge.leafClusterEnds[leaf];
            const int prevRunningMaxEnd =
                sponge.candidateLeafPrevRunningMaxEnds[candidateIdx];
            const int dedupedClusterJStart =
                std::max(clusterJStart, prevRunningMaxEnd);
            leafSpans.push_back(clusterJEnd - clusterJStart);
            dedupSpans.push_back(std::max(0, clusterJEnd - dedupedClusterJStart));
            for (int clusterJ = dedupedClusterJStart; clusterJ < clusterJEnd;
                 ++clusterJ)
            {
                const int maskOffset =
                    candidateIdx * sponge.candidateLeafClusterStride +
                    (clusterJ - clusterJStart);
                unsigned int reachMask = 0u;
                if (maskOffset >= 0 &&
                    maskOffset < static_cast<int>(sponge.candidateLeafReachMasks.size()))
                {
                    reachMask = sponge.candidateLeafReachMasks[maskOffset];
                }
                if (reachMask == 0u)
                {
                    skippedByReach += 1;
                    continue;
                }
                const unsigned int validJ = sponge.clusterValidMasks[clusterJ];
                if (validJ == 0u)
                {
                    skippedByValidJ += 1;
                    continue;
                }
                const unsigned int localJ = sponge.clusterLocalMasks[clusterJ];
                const int superJ = sponge.clusterToSupercluster[clusterJ];
                if (localJ != 0u && superJ < superI)
                {
                    skippedByHalfshell += 1;
                    continue;
                }
                prepruneRecords += 1;
                recordsPerSci[sci] += 1;
                recordReachPopc.push_back(__builtin_popcount(reachMask));
                recordValidJPopc.push_back(__builtin_popcount(validJ));
            }
        }
    }

    const int emitted = std::min<int>(
        sponge.countSourceFragmentCursor[0],
        static_cast<int>(sponge.countSourceFragments.size()));
    std::vector<int> sourceRowsPerSci(static_cast<std::size_t>(sponge.candidateSciCount), 0);
    std::vector<int> fragmentImaskPopc;
    std::vector<int> fragmentValidJPopc;
    int splitRows[kClusteredWarpSplitCount] = {};
    int nonzeroExclusionFragments = 0;
    std::vector<long long> sourceRecordKeys;
    sourceRecordKeys.reserve(static_cast<std::size_t>(emitted));
    for (int i = 0; i < emitted; ++i)
    {
        const auto& fragment = sponge.countSourceFragments[i];
        if (fragment.sci_id >= 0 && fragment.sci_id < sponge.candidateSciCount)
        {
            sourceRowsPerSci[fragment.sci_id] += 1;
        }
        if (fragment.split_id >= 0 && fragment.split_id < kClusteredWarpSplitCount)
        {
            splitRows[fragment.split_id] += 1;
        }
        fragmentImaskPopc.push_back(__builtin_popcount(fragment.imask));
        fragmentValidJPopc.push_back(__builtin_popcount(fragment.valid_mask_j));
        bool hasExclusion = false;
        for (int iLocal = 0; iLocal < kClusteredSuperClusterClusters; ++iLocal)
        {
            hasExclusion = hasExclusion || fragment.exclusion_masks[iLocal] != 0ull;
        }
        if (hasExclusion) nonzeroExclusionFragments += 1;
        sourceRecordKeys.push_back((static_cast<long long>(fragment.sci_id) << 32) ^
                                   static_cast<unsigned int>(fragment.cluster_j));
    }
    std::sort(sourceRecordKeys.begin(), sourceRecordKeys.end());
    std::vector<int> sourceRowsPerRecord;
    for (std::size_t i = 0; i < sourceRecordKeys.size();)
    {
        std::size_t j = i + 1;
        while (j < sourceRecordKeys.size() && sourceRecordKeys[j] == sourceRecordKeys[i])
        {
            ++j;
        }
        sourceRowsPerRecord.push_back(static_cast<int>(j - i));
        i = j;
    }

    std::printf("sponge_payload_stats struct_bytes source_fragment=%zu source_fragment_slim=%zu "
                "record_stream_source=%zu\n",
                sizeof(LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT),
                sizeof(LJ_CLUSTERED_GMXPACKED_COUNT_SOURCE_FRAGMENT_SLIM),
                sizeof(LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE));
    std::printf("sponge_payload_stats preprune_records=%d source_fragments=%d "
                "source_records=%zu skipped_reach=%d skipped_valid_j=%d "
                "skipped_halfshell=%d source_per_preprune=%.6f "
                "nonzero_exclusion_fragments=%d split0_rows=%d split1_rows=%d\n",
                prepruneRecords, emitted, sourceRecordKeys.empty()
                                      ? std::size_t{0}
                                      : sourceRowsPerRecord.size(),
                skippedByReach, skippedByValidJ, skippedByHalfshell,
                prepruneRecords > 0
                    ? static_cast<double>(emitted) /
                          static_cast<double>(prepruneRecords)
                    : 0.0,
                nonzeroExclusionFragments, splitRows[0], splitRows[1]);
    PrintIntDistribution("sponge_payload_candidate_leaves_per_sci", candidateLeaves);
    PrintIntDistribution("sponge_payload_leaf_cluster_span", leafSpans);
    PrintIntDistribution("sponge_payload_dedup_cluster_span", dedupSpans);
    PrintIntDistribution("sponge_payload_preprune_records_per_sci", recordsPerSci);
    PrintIntDistribution("sponge_payload_record_reach_popc", recordReachPopc);
    PrintIntDistribution("sponge_payload_record_valid_j_popc", recordValidJPopc);
    PrintIntDistribution("sponge_payload_source_rows_per_sci", sourceRowsPerSci);
    PrintIntDistribution("sponge_payload_source_rows_per_record", sourceRowsPerRecord);
    PrintIntDistribution("sponge_payload_fragment_imask_popc", fragmentImaskPopc);
    PrintIntDistribution("sponge_payload_fragment_valid_j_popc", fragmentValidJPopc);
}

void PrintStats(const char* label, const std::vector<double>& millis)
{
    const double mean = std::accumulate(millis.begin(), millis.end(), 0.0) / millis.size();
    double var = 0.0;
    for (double value : millis) var += (value - mean) * (value - mean);
    const double sd = millis.size() > 1 ? std::sqrt(var / (millis.size() - 1)) : 0.0;
    std::printf("%s_ms_mean=%.6f %s_ms_sd=%.6f", label, mean, label, sd);
}

void RunSpongeInternalComparison(const Options& opt, const PreparedData& data, float h)
{
    SpongeInternalData sponge = PrepareSpongeInternalData(opt, data, h);
    cstone::StreamHolder stream;
    PrefetchSpongeInternalData(sponge, stream);

    RunSpongeCandidateCollect(sponge);
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize initial candidate collect");
    BuildSpongeCountInputsFromCollect(sponge);
    PrefetchSpongeInternalData(sponge, stream);

    std::printf("sponge_internal_setup clusters=%d superclusters=%d candidate_sci=%d leaves=%d "
                "candidate_leaves=%d stride=%d onepass_capacity=%d collect_overflow=%d cutoff=%.9g\n",
                sponge.clusterCount, sponge.superclusterCount, sponge.candidateSciCount, sponge.leafCount,
                sponge.candidateLeafCount, sponge.candidateLeafClusterStride, sponge.onepassCapacity,
                sponge.collectOverflow, sponge.cutoff);
    if (opt.spongePayloadStats)
    {
        PrintSpongePayloadStats(sponge);
    }
    const bool runStatic = opt.spongeCountMode == "static" ||
                           opt.spongeCountMode == "both" ||
                           opt.spongeCountMode == "all";
    const bool runDynamic = opt.spongeCountMode == "dynamic" ||
                            opt.spongeCountMode == "both" ||
                            opt.spongeCountMode == "all";
    const bool runSlim = opt.spongeCountMode == "slim" ||
                         opt.spongeCountMode == "all";
    const bool runCooperative = opt.spongeCountMode == "cooperative" ||
                                opt.spongeCountMode == "coop" ||
                                opt.spongeCountMode == "all";

    for (unsigned i = 0; i < opt.warmup; ++i)
    {
        RunSpongeCandidateCollect(sponge);
        CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup candidate collect");
        if (runStatic)
        {
            RunSpongeFixedLightCount(sponge, false, false, false);
            CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup fixed light count static");
        }
        if (runDynamic)
        {
            RunSpongeFixedLightCount(sponge, true, false, false);
            CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup fixed light count dynamic");
        }
        if (runSlim)
        {
            RunSpongeFixedLightCount(sponge, false, true, false);
            CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup fixed light count slim");
        }
        if (runCooperative)
        {
            RunSpongeFixedLightCount(sponge, false, false, true);
            CheckCuda(cudaDeviceSynchronize(),
                      "cudaDeviceSynchronize warmup fixed light count cooperative");
        }
    }

    std::vector<double> collectMillis;
    std::vector<double> staticCountMillis;
    std::vector<double> dynamicCountMillis;
    std::vector<double> slimCountMillis;
    std::vector<double> cooperativeCountMillis;
    collectMillis.reserve(opt.iters * opt.spongeRepeat);
    staticCountMillis.reserve(opt.iters * opt.spongeRepeat);
    dynamicCountMillis.reserve(opt.iters * opt.spongeRepeat);
    slimCountMillis.reserve(opt.iters * opt.spongeRepeat);
    cooperativeCountMillis.reserve(opt.iters * opt.spongeRepeat);
    for (unsigned i = 0; i < opt.iters; ++i)
    {
        double collectSum = 0.0;
        double staticCountSum = 0.0;
        double dynamicCountSum = 0.0;
        double slimCountSum = 0.0;
        double cooperativeCountSum = 0.0;
        int staticSourceRows = 0;
        int staticSourceFragments = 0;
        int staticOverflow = 0;
        int dynamicSourceRows = 0;
        int dynamicSourceFragments = 0;
        int dynamicOverflow = 0;
        int slimSourceRows = 0;
        int slimSourceFragments = 0;
        int slimOverflow = 0;
        int cooperativeSourceRows = 0;
        int cooperativeSourceFragments = 0;
        int cooperativeOverflow = 0;
        for (unsigned r = 0; r < opt.spongeRepeat; ++r)
        {
            const auto collectStart = Clock::now();
            RunSpongeCandidateCollect(sponge);
            CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize candidate collect");
            const auto collectEnd = Clock::now();
            const double collectMs = std::chrono::duration<double, std::milli>(collectEnd - collectStart).count();
            collectMillis.push_back(collectMs);
            collectSum += collectMs;

            if (runStatic)
            {
                const auto countStart = Clock::now();
                RunSpongeFixedLightCount(sponge, false, false, false);
                CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize fixed light count static");
                const auto countEnd = Clock::now();
                const double countMs = std::chrono::duration<double, std::milli>(countEnd - countStart).count();
                staticCountMillis.push_back(countMs);
                staticCountSum += countMs;
                staticSourceRows = sponge.recordStreamSourceRows[0];
                staticSourceFragments = sponge.countSourceFragmentCursor[0];
                staticOverflow = sponge.countSourceFragmentOverflowRows[0];
            }

            if (runDynamic)
            {
                const auto countStart = Clock::now();
                RunSpongeFixedLightCount(sponge, true, false, false);
                CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize fixed light count dynamic");
                const auto countEnd = Clock::now();
                const double countMs = std::chrono::duration<double, std::milli>(countEnd - countStart).count();
                dynamicCountMillis.push_back(countMs);
                dynamicCountSum += countMs;
                dynamicSourceRows = sponge.recordStreamSourceRows[0];
                dynamicSourceFragments = sponge.countSourceFragmentCursor[0];
                dynamicOverflow = sponge.countSourceFragmentOverflowRows[0];
            }

            if (runSlim)
            {
                const auto countStart = Clock::now();
                RunSpongeFixedLightCount(sponge, false, true, false);
                CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize fixed light count slim");
                const auto countEnd = Clock::now();
                const double countMs = std::chrono::duration<double, std::milli>(countEnd - countStart).count();
                slimCountMillis.push_back(countMs);
                slimCountSum += countMs;
                slimSourceRows = sponge.recordStreamSourceRows[0];
                slimSourceFragments = sponge.countSourceFragmentCursor[0];
                slimOverflow = sponge.countSourceFragmentOverflowRows[0];
            }

            if (runCooperative)
            {
                const auto countStart = Clock::now();
                RunSpongeFixedLightCount(sponge, false, false, true);
                CheckCuda(cudaDeviceSynchronize(),
                          "cudaDeviceSynchronize fixed light count cooperative");
                const auto countEnd = Clock::now();
                const double countMs =
                    std::chrono::duration<double, std::milli>(countEnd -
                                                              countStart)
                        .count();
                cooperativeCountMillis.push_back(countMs);
                cooperativeCountSum += countMs;
                cooperativeSourceRows = sponge.recordStreamSourceRows[0];
                cooperativeSourceFragments =
                    sponge.countSourceFragmentCursor[0];
                cooperativeOverflow = sponge.countSourceFragmentOverflowRows[0];
            }
        }
        const double collectMean = collectSum / opt.spongeRepeat;
        if (runStatic)
        {
            const double countMean = staticCountSum / opt.spongeRepeat;
            std::printf("sponge_internal iter=%u count_mode=static candidate_collect_ms=%.6f "
                        "fixed_light_count_ms=%.6f total_ms=%.6f source_rows=%d source_fragments=%d "
                        "count_overflow=%d\n",
                        i, collectMean, countMean, collectMean + countMean, staticSourceRows,
                        staticSourceFragments, staticOverflow);
        }
        if (runDynamic)
        {
            const double countMean = dynamicCountSum / opt.spongeRepeat;
            std::printf("sponge_internal iter=%u count_mode=dynamic candidate_collect_ms=%.6f "
                        "fixed_light_count_ms=%.6f total_ms=%.6f source_rows=%d source_fragments=%d "
                        "count_overflow=%d\n",
                        i, collectMean, countMean, collectMean + countMean, dynamicSourceRows,
                        dynamicSourceFragments, dynamicOverflow);
        }
        if (runSlim)
        {
            const double countMean = slimCountSum / opt.spongeRepeat;
            std::printf("sponge_internal iter=%u count_mode=slim candidate_collect_ms=%.6f "
                        "fixed_light_count_ms=%.6f total_ms=%.6f source_rows=%d source_fragments=%d "
                        "count_overflow=%d\n",
                        i, collectMean, countMean, collectMean + countMean, slimSourceRows,
                        slimSourceFragments, slimOverflow);
        }
        if (runCooperative)
        {
            const double countMean = cooperativeCountSum / opt.spongeRepeat;
            std::printf("sponge_internal iter=%u count_mode=cooperative candidate_collect_ms=%.6f "
                        "fixed_light_count_ms=%.6f total_ms=%.6f source_rows=%d source_fragments=%d "
                        "count_overflow=%d\n",
                        i, collectMean, countMean, collectMean + countMean,
                        cooperativeSourceRows, cooperativeSourceFragments,
                        cooperativeOverflow);
        }
    }

    std::printf("sponge_internal_summary count_mode=%s ", opt.spongeCountMode.c_str());
    PrintStats("candidate_collect", collectMillis);
    if (!staticCountMillis.empty())
    {
        std::printf(" ");
        PrintStats("static_fixed_light_count", staticCountMillis);
    }
    if (!dynamicCountMillis.empty())
    {
        std::printf(" ");
        PrintStats("dynamic_fixed_light_count", dynamicCountMillis);
    }
    if (!slimCountMillis.empty())
    {
        std::printf(" ");
        PrintStats("slim_fixed_light_count", slimCountMillis);
    }
    if (!cooperativeCountMillis.empty())
    {
        std::printf(" ");
        PrintStats("cooperative_fixed_light_count", cooperativeCountMillis);
    }
    std::printf("\n");
}

template<class Builder>
void RunProbe(const Options& opt, const Builder& builder)
{
    const cstone::Box<float> box = MakeBox(opt);

    PreparedData data = PrepareData(opt, box);
    const float h = ResolveH(opt, data.n);
    DeviceViews views = MakeViews(opt, data);

    cstone::StreamHolder stream;
    int device = 0;
    CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
    const auto prefetch = [&](auto& vector, const char* name)
    {
        (void)name;
        if (!vector.empty())
        {
            CheckCuda(cudaMemPrefetchAsync(
                          cstone::rawPtr(vector),
                          vector.size() * sizeof(typename std::decay_t<decltype(vector)>::value_type),
                          {cudaMemLocationTypeDevice, device}, 0, stream),
                      "cudaMemPrefetchAsync");
        }
    };
    prefetch(data.dX, "x");
    prefetch(data.dY, "y");
    prefetch(data.dZ, "z");
    prefetch(data.dCodes, "codes");
    prefetch(data.dPrefixes, "prefixes");
    prefetch(data.dChildOffsets, "childOffsets");
    prefetch(data.dParents, "parents");
    prefetch(data.dInternalToLeaf, "internalToLeaf");
    prefetch(data.dLeafToInternal, "leafToInternal");
    prefetch(data.dLevelRange, "levelRange");
    prefetch(data.dLayout, "layout");
    prefetch(data.dCenters, "centers");
    prefetch(data.dSizes, "sizes");
    prefetch(data.dGroups, "groups");
    stream.sync();

    std::printf("neighbor_api_probe system=%s distribution=%s coordinate_file=%s n=%zu bucket=%u group=%u ncmax=%u "
                "compression=%s symmetric=%d periodic=%d h=%.9g search_ext=%.6g leaves=%u nodes=%u\n",
                opt.system.c_str(), opt.coordinateFile.empty() ? "synthetic-uniform" : "real-file",
                opt.coordinateFile.empty() ? "-" : opt.coordinateFile.c_str(), data.n, opt.bucketSize, opt.groupSize,
                opt.ncmax, opt.compression.c_str(),
                opt.symmetric ? 1 : 0, opt.periodic ? 1 : 0, h, opt.searchExtFactor, data.octree.numLeafNodes,
                data.octree.numNodes);

    std::size_t lastBodies = 0;
    std::size_t lastBytes = 0;
    for (unsigned i = 0; i < opt.warmup; ++i)
    {
        auto neighborhood = builder.build(stream.exec(), views.nsView, box, static_cast<cstone::LocalIndex>(data.n),
                                          views.groupView, cstone::rawPtr(data.dX), cstone::rawPtr(data.dY),
                                          cstone::rawPtr(data.dZ), h);
        stream.sync();
        const auto stats = neighborhood.stats();
        lastBodies = stats.numBodies;
        lastBytes = stats.numBytes;
    }

    std::vector<double> millis;
    millis.reserve(opt.iters);
    for (unsigned i = 0; i < opt.iters; ++i)
    {
        const auto start = Clock::now();
        auto neighborhood = builder.build(stream.exec(), views.nsView, box, static_cast<cstone::LocalIndex>(data.n),
                                          views.groupView, cstone::rawPtr(data.dX), cstone::rawPtr(data.dY),
                                          cstone::rawPtr(data.dZ), h);
        stream.sync();
        const auto end = Clock::now();
        const auto stats = neighborhood.stats();
        lastBodies = stats.numBodies;
        lastBytes = stats.numBytes;
        millis.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        std::printf("iter=%u build_ms=%.6f neighbor_bytes=%zu bytes_per_particle=%.3f bodies=%zu\n", i, millis.back(),
                    lastBytes, lastBodies ? double(lastBytes) / lastBodies : 0.0, lastBodies);
    }

    const double mean = std::accumulate(millis.begin(), millis.end(), 0.0) / millis.size();
    double var = 0.0;
    for (double value : millis)
        var += (value - mean) * (value - mean);
    const double sd = millis.size() > 1 ? std::sqrt(var / (millis.size() - 1)) : 0.0;
    std::printf("summary build_ms_mean=%.6f build_ms_sd=%.6f neighbor_bytes=%zu bytes_per_particle=%.3f\n", mean, sd,
                lastBytes, lastBodies ? double(lastBytes) / lastBodies : 0.0);

    if (opt.compareSpongeInternal)
    {
        RunSpongeInternalComparison(opt, data, h);
    }
}

template<bool Symmetric, class Compression>
void RunTyped(const Options& opt)
{
    using Config = cstone::ijloop::gpu_supercluster_nb_list_neighborhood_detail::
        GpuSuperclusterNbListNeighborhoodConfig<8, cstone::GpuConfig::warpSize / 8, 64, Compression, Symmetric>;
    cstone::ijloop::GpuSuperclusterNbListNeighborhoodBuilder<Config> builder{opt.ncmax};
    RunProbe(opt, builder);
}

template<class Compression>
void RunSymmetry(const Options& opt)
{
    if (opt.symmetric)
        RunTyped<true, Compression>(opt);
    else
        RunTyped<false, Compression>(opt);
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Options opt = ParseArgs(argc, argv);
        if (opt.compression == "none")
            RunSymmetry<cstone::DummyWarpCompression<false>>(opt);
        else if (opt.compression == "nibble")
            RunSymmetry<cstone::NibbleWarpCompression<false>>(opt);
        else
            RunSymmetry<cstone::BandEtAlWarpCompression<false>>(opt);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "neighbor_api_probe error: %s\n", e.what());
        return 1;
    }
    return 0;
}
