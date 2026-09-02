#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <tuple>
#include <vector>

#include "control.h"
#include "manybody/edip.h"
#include "manybody/reaxff/hydrogen_bond.h"
#include "manybody/tersoff.h"

// The many-body implementation files contain initialization entry points that
// reference controller state even though this test constructs their state
// directly. Keep those link requirements local to this test executable.
CONTROLLER controller;
unsigned int CONTROLLER::device_optimized_block = 128;
unsigned int CONTROLLER::device_warp = 32;
unsigned int CONTROLLER::device_max_thread = 1024;
int CONTROLLER::force_replica_count = 1;
int CONTROLLER::MPI_rank = 0;
int CONTROLLER::MPI_size = 1;
int CONTROLLER::PP_MPI_size = 1;
int CONTROLLER::PM_MPI_size = 0;
int CONTROLLER::CC_MPI_size = 0;
int CONTROLLER::PP_MPI_rank = 0;
int CONTROLLER::PM_MPI_rank = 0;
int CONTROLLER::CC_MPI_rank = 0;
MPI_Comm CONTROLLER::pp_comm = {};
MPI_Comm CONTROLLER::pm_comm = {};
D_MPI_Comm CONTROLLER::D_MPI_COMM_WORLD = {};
D_MPI_Comm CONTROLLER::d_pp_comm = {};
D_MPI_Comm CONTROLLER::d_pm_comm = {};

bool CONTROLLER::Command_Exist(const char*, const char*) { return false; }
const char* CONTROLLER::Command(const char*, const char*) { return ""; }
bool CONTROLLER::Command_Exist(const char*) { return false; }
const char* CONTROLLER::Command(const char*) { return ""; }

namespace
{
constexpr int kAtomCount = 5;
constexpr int kTypeCount = 2;
constexpr int kExcludedA = 0;
constexpr int kExcludedB = 4;

int failures = 0;

void Check(bool condition, const char* label)
{
    if (!condition)
    {
        std::fprintf(stderr, "FAIL: %s\n", label);
        failures += 1;
    }
}

void Check_Near(double actual, double expected, double tolerance,
                const char* label, int atom)
{
    const double difference = std::fabs(actual - expected);
    if (!(difference <= tolerance))
    {
        std::fprintf(
            stderr,
            "FAIL: %s atom=%d actual=%.9g expected=%.9g diff=%.9g tol=%.9g\n",
            label, atom, actual, expected, difference, tolerance);
        failures += 1;
    }
}

template <typename T>
T* Device_Copy(const std::vector<T>& host)
{
    T* pointer = nullptr;
    if (!host.empty())
    {
#ifdef USE_CPU
        pointer = static_cast<T*>(std::malloc(sizeof(T) * host.size()));
        std::memcpy(pointer, host.data(), sizeof(T) * host.size());
#else
        Device_Malloc_And_Copy_Safely(reinterpret_cast<void**>(&pointer),
                                      const_cast<T*>(host.data()),
                                      sizeof(T) * host.size());
#endif
    }
    return pointer;
}

template <typename T>
T* Device_Allocate(int count)
{
    T* pointer = nullptr;
#ifdef USE_CPU
    pointer =
        static_cast<T*>(std::malloc(sizeof(T) * static_cast<size_t>(count)));
#else
    Device_Malloc_Safely(reinterpret_cast<void**>(&pointer),
                         sizeof(T) * static_cast<size_t>(count));
#endif
    return pointer;
}

template <typename T>
std::vector<T> Device_Read(const T* pointer, int count)
{
    std::vector<T> host(static_cast<size_t>(count));
    if (count > 0)
    {
        deviceMemcpy(host.data(), pointer,
                     sizeof(T) * static_cast<size_t>(count),
                     deviceMemcpyDeviceToHost);
    }
    return host;
}

template <typename T>
void Device_Free(T** pointer)
{
#ifdef USE_CPU
    std::free(*pointer);
    *pointer = nullptr;
#else
    Free_Single_Device_Pointer(reinterpret_cast<void**>(pointer));
#endif
}

LTMatrix3 Diagonal_Matrix(float value)
{
    LTMatrix3 matrix = {};
    matrix.a11 = value;
    matrix.a22 = value;
    matrix.a33 = value;
    return matrix;
}

struct Periodic_Displacement
{
    VECTOR dr = {};
    int sx = 0;
    int sy = 0;
    int sz = 0;
};

Periodic_Displacement Minimum_Image(const VECTOR& ri, const VECTOR& rj,
                                    float box)
{
    Periodic_Displacement result;
    const VECTOR raw = ri - rj;
    const int nx = static_cast<int>(std::floor(raw.x / box + 0.5f));
    const int ny = static_cast<int>(std::floor(raw.y / box + 0.5f));
    const int nz = static_cast<int>(std::floor(raw.z / box + 0.5f));
    result.dr = {raw.x - static_cast<float>(nx) * box,
                 raw.y - static_cast<float>(ny) * box,
                 raw.z - static_cast<float>(nz) * box};
    result.sx = -nx;
    result.sy = -ny;
    result.sz = -nz;
    return result;
}

float Distance(const VECTOR& ri, const VECTOR& rj)
{
    const VECTOR dr = Minimum_Image(ri, rj, 10.0f).dr;
    return std::sqrt(dr * dr);
}

bool Is_Excluded(int atom_i, int atom_j)
{
    return (atom_i == kExcludedA && atom_j == kExcludedB) ||
           (atom_i == kExcludedB && atom_j == kExcludedA);
}

using Rows = std::vector<std::vector<int>>;
using Edip_Triplet = std::tuple<int, int, int, int, int, int, int, int, int>;
using Tersoff_Tuple = std::tuple<int, int, int, int, int, int, int, int, int>;

std::vector<VECTOR> Test_Coordinates()
{
    return {{0.2f, 5.0f, 5.0f},
            {9.2f, 5.0f, 5.0f},
            {9.2f, 6.1f, 5.0f},
            {9.2f, 5.0f, 6.3747735f},
            {9.4f, 5.0f, 5.0f}};
}

std::vector<int> Test_Types() { return {0, 0, 1, 1, 0}; }

struct Clustered_Fixture
{
    CLUSTERED_SPATIAL_VIEW view = {};

    int* sort_permutation = nullptr;
    int* cluster_offsets = nullptr;
    unsigned int* cluster_valid_masks = nullptr;
    unsigned int* cluster_local_masks = nullptr;
    VECTOR* cluster_centers = nullptr;
    VECTOR* cluster_extents = nullptr;
    int* super_cluster_offsets = nullptr;
    CLUSTERED_GMXPACKED_SCI* gmxpacked_sci = nullptr;
    CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked = nullptr;
    CLUSTERED_GMXPACKED_EXCLUSION* gmxpacked_exclusions = nullptr;
    int* gmxpacked_endpoint_incidence_offsets = nullptr;
    CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE*
        gmxpacked_endpoint_incidence_references = nullptr;
    uint64_t* pair_shift_bits = nullptr;

    Clustered_Fixture(const LTMatrix3& rcell)
    {
        sort_permutation = Device_Copy(std::vector<int>{0, 1, 2, 3, 4});
        cluster_offsets = Device_Copy(std::vector<int>{0, 1, 5});
        cluster_valid_masks =
            Device_Copy(std::vector<unsigned int>{0x1u, 0xfu});
        cluster_local_masks =
            Device_Copy(std::vector<unsigned int>{0x1u, 0xfu});
        cluster_centers = Device_Copy(std::vector<VECTOR>{
            {0.2f, 5.0f, 5.0f}, {9.25f, 5.275f, 5.3436934f}});
        cluster_extents = Device_Copy(std::vector<VECTOR>(2));
        super_cluster_offsets = Device_Copy(std::vector<int>{0, 2});

        view.ready = true;
#ifdef USE_CPU
        view.backend = CLUSTERED_SPATIAL_BACKEND::CPU;
        view.readiness_scope = CLUSTERED_SPATIAL_READINESS_SCOPE::HOST_COMPLETE;
#elif defined(USE_CUDA)
        view.backend = CLUSTERED_SPATIAL_BACKEND::CUDA;
        view.readiness_scope =
            CLUSTERED_SPATIAL_READINESS_SCOPE::PRODUCER_STREAM_ORDERED;
#else
        view.backend = CLUSTERED_SPATIAL_BACKEND::HIP;
        view.readiness_scope =
            CLUSTERED_SPATIAL_READINESS_SCOPE::PRODUCER_STREAM_ORDERED;
#endif
        view.provider_incarnation = 41;
        view.lease_epoch = 43;
        view.gmxpacked_payload_generation = 53;
        view.source_generation = 59;
        view.geometry_generation = 61;
        view.cluster_size = kClusteredClusterSize;
        view.super_cluster_clusters = kClusteredSuperClusterClusters;
        view.local_atom_numbers = kAtomCount;
        view.direct_local_atom_numbers = kAtomCount;
        view.ghost_numbers = 0;
        view.total_atom_numbers = kAtomCount;
        view.padded_total_atom_numbers = kAtomCount;
        view.cluster_numbers = 2;
        view.super_cluster_numbers = 1;
        view.cached_cutoff = 7.5f;
        view.rebuild_skin = 0.0f;
        view.sort_permutation = sort_permutation;
        view.cluster_offsets = cluster_offsets;
        view.cluster_valid_masks = cluster_valid_masks;
        view.cluster_local_masks = cluster_local_masks;
        view.cluster_centers = cluster_centers;
        view.cluster_extents = cluster_extents;
        view.super_cluster_offsets = super_cluster_offsets;

        std::vector<CLUSTERED_GMXPACKED_SCI> host_sci(2);
        host_sci[0].supercluster_id = 0;
        host_sci[0].shift_id = 22;
        host_sci[0].cjpacked_begin = 0;
        host_sci[0].cjpacked_end = 1;
        host_sci[1].supercluster_id = 0;
        host_sci[1].shift_id = kClusteredCentralShiftId;
        host_sci[1].cjpacked_begin = 1;
        host_sci[1].cjpacked_end = 2;

        std::vector<CLUSTERED_GMXPACKED_CJ> host_packed(2);
        host_packed[0].cj[0] = 1;
        host_packed[0].split[0].imask = 1u;
        host_packed[0].split[1].imask = 1u;
        host_packed[0].split[0].exclusion_index = 1;
        host_packed[0].split[1].exclusion_index = 1;
        host_packed[1].cj[0] = 1;
        host_packed[1].split[0].imask = 2u;
        host_packed[1].split[1].imask = 2u;
        host_packed[1].split[0].exclusion_index = 2;
        host_packed[1].split[1].exclusion_index = 2;

        std::vector<CLUSTERED_GMXPACKED_EXCLUSION> host_exclusions(3);
        for (auto& exclusion : host_exclusions)
            for (unsigned int& pair : exclusion.pair) pair = 0xffffffffu;
        host_exclusions[1].pair[3 * kClusteredClusterSize] &= ~1u;
        for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
            for (int split_j_lane = 0;
                 split_j_lane < kClusteredSplitJClusterSize; split_j_lane += 1)
                for (int i_lane = 0; i_lane < kClusteredClusterSize;
                     i_lane += 1)
                {
                    const int j_lane =
                        split * kClusteredSplitJClusterSize + split_j_lane;
                    if (j_lane <= i_lane)
                        host_exclusions[2]
                            .pair[split_j_lane * kClusteredClusterSize +
                                  i_lane] &= ~2u;
                }

        std::vector<uint64_t> host_pair_shift_bits(2 * kClusteredJGroupSize,
                                                   0ull);
        Clustered_Set_Pair_Shift_Id(&host_pair_shift_bits[0], 0, 22);
        Clustered_Set_Pair_Active_I_Masks(&host_pair_shift_bits[0], 1u, 1u);
        Clustered_Set_Pair_Shift_Id(&host_pair_shift_bits[kClusteredJGroupSize],
                                    1, kClusteredCentralShiftId);
        Clustered_Set_Pair_Active_I_Masks(
            &host_pair_shift_bits[kClusteredJGroupSize], 2u, 2u);

        gmxpacked_sci = Device_Copy(host_sci);
        gmxpacked_cjpacked = Device_Copy(host_packed);
        gmxpacked_exclusions = Device_Copy(host_exclusions);
        pair_shift_bits = Device_Copy(host_pair_shift_bits);
        view.gmxpacked_sci_numbers = 2;
        view.gmxpacked_cjpacked_numbers = 2;
        view.gmxpacked_exclusion_numbers = 3;
        view.gmxpacked_sci = gmxpacked_sci;
        view.gmxpacked_cjpacked = gmxpacked_cjpacked;
        view.gmxpacked_exclusions = gmxpacked_exclusions;

        CLUSTERED_GMXPACKED_ENDPOINT_INCIDENCE_HOST endpoint_incidence;
        const char* endpoint_failure_reason = nullptr;
        const bool endpoint_ready =
            Clustered_Build_Gmxpacked_Endpoint_Incidence_Host(
                view.provider_incarnation, view.gmxpacked_payload_generation,
                view.cluster_numbers, view.super_cluster_numbers,
                std::vector<int>{0, 2}.data(),
                static_cast<int>(host_sci.size()), host_sci.data(),
                static_cast<int>(host_packed.size()), host_packed.data(),
                &endpoint_incidence, &endpoint_failure_reason);
        Check(endpoint_ready,
              "clustered fixture builds gmxpacked endpoint incidence");
        if (!endpoint_ready && endpoint_failure_reason != nullptr)
            std::fprintf(stderr, "Endpoint incidence rejection: %s\n",
                         endpoint_failure_reason);
        gmxpacked_endpoint_incidence_offsets =
            Device_Copy(endpoint_incidence.offsets);
        gmxpacked_endpoint_incidence_references =
            Device_Copy(endpoint_incidence.references);
        view.gmxpacked_endpoint_incidence_ready = endpoint_ready;
        view.endpoint_incidence_provider_incarnation =
            endpoint_incidence.provider_incarnation;
        view.endpoint_incidence_payload_generation =
            endpoint_incidence.gmxpacked_payload_generation;
        view.endpoint_incidence_sci_numbers = view.gmxpacked_sci_numbers;
        view.endpoint_incidence_cjpacked_numbers =
            view.gmxpacked_cjpacked_numbers;
        view.endpoint_incidence_super_cluster_numbers =
            view.super_cluster_numbers;
        view.endpoint_incidence_reference_numbers =
            static_cast<int>(endpoint_incidence.references.size());
        view.endpoint_incidence_offset_tail =
            endpoint_incidence.offsets.empty()
                ? 0
                : endpoint_incidence.offsets.back();
        view.gmxpacked_endpoint_incidence_offsets =
            gmxpacked_endpoint_incidence_offsets;
        view.gmxpacked_endpoint_incidence_references =
            gmxpacked_endpoint_incidence_references;

        view.pair_shift_metadata_ready = true;
        view.pair_shift_payload_generation = view.gmxpacked_payload_generation;
        view.pair_shift_geometry_generation = view.geometry_generation;
        view.pair_shift_sci_numbers = view.gmxpacked_sci_numbers;
        view.pair_shift_cjpacked_numbers = view.gmxpacked_cjpacked_numbers;
        view.pair_shift_exclusion_numbers = view.gmxpacked_exclusion_numbers;
        view.pair_shift_rcell = rcell;
        view.pair_shift_bits = pair_shift_bits;
    }

    ~Clustered_Fixture()
    {
        Device_Free(&sort_permutation);
        Device_Free(&cluster_offsets);
        Device_Free(&cluster_valid_masks);
        Device_Free(&cluster_local_masks);
        Device_Free(&cluster_centers);
        Device_Free(&cluster_extents);
        Device_Free(&super_cluster_offsets);
        Device_Free(&gmxpacked_sci);
        Device_Free(&gmxpacked_cjpacked);
        Device_Free(&gmxpacked_exclusions);
        Device_Free(&gmxpacked_endpoint_incidence_offsets);
        Device_Free(&gmxpacked_endpoint_incidence_references);
        Device_Free(&pair_shift_bits);
    }
};

Rows Read_Rows(const int* offsets_device, const int* atoms_device,
               int atom_count, int neighbor_count)
{
    const std::vector<int> offsets =
        Device_Read(offsets_device, atom_count + 1);
    const std::vector<int> atoms = Device_Read(atoms_device, neighbor_count);
    Rows rows(static_cast<size_t>(atom_count));
    for (int atom_i = 0; atom_i < atom_count; atom_i += 1)
    {
        Check(offsets[atom_i] <= offsets[atom_i + 1],
              "neighbor offsets are monotonic");
        for (int index = offsets[atom_i]; index < offsets[atom_i + 1];
             index += 1)
            rows[atom_i].push_back(atoms[index]);
        std::sort(rows[atom_i].begin(), rows[atom_i].end());
        Check(std::adjacent_find(rows[atom_i].begin(), rows[atom_i].end()) ==
                  rows[atom_i].end(),
              "clustered relation has no duplicate atom IDs");
    }
    return rows;
}

void Check_Rows(const Rows& actual, const Rows& expected, const char* label)
{
    Check(actual.size() == expected.size(), label);
    if (actual.size() != expected.size()) return;
    for (size_t atom_i = 0; atom_i < actual.size(); atom_i += 1)
    {
        if (actual[atom_i] != expected[atom_i])
        {
            std::fprintf(stderr, "FAIL: %s center=%zu actual=", label, atom_i);
            for (int atom : actual[atom_i]) std::fprintf(stderr, "%d,", atom);
            std::fprintf(stderr, " expected=");
            for (int atom : expected[atom_i]) std::fprintf(stderr, "%d,", atom);
            std::fprintf(stderr, "\n");
            failures += 1;
        }
    }
}

std::set<Edip_Triplet> Edip_Triplets(const Rows& rows,
                                     const std::vector<VECTOR>& coordinates)
{
    std::set<Edip_Triplet> result;
    for (int atom_i = 0; atom_i < kAtomCount; atom_i += 1)
    {
        for (size_t j_index = 0; j_index < rows[atom_i].size(); j_index += 1)
        {
            for (size_t k_index = j_index + 1; k_index < rows[atom_i].size();
                 k_index += 1)
            {
                const int atom_j = rows[atom_i][j_index];
                const int atom_k = rows[atom_i][k_index];
                const auto shift_j = Minimum_Image(coordinates[atom_i],
                                                   coordinates[atom_j], 10.0f);
                const auto shift_k = Minimum_Image(coordinates[atom_i],
                                                   coordinates[atom_k], 10.0f);
                result.emplace(atom_i, atom_j, atom_k, shift_j.sx, shift_j.sy,
                               shift_j.sz, shift_k.sx, shift_k.sy, shift_k.sz);
            }
        }
    }
    return result;
}

enum Edip_Pair_Parameter
{
    edip_alpha = 0,
    edip_c = 1,
    edip_a = 2,
    edip_A = 3,
    edip_B = 4,
    edip_rho = 5,
    edip_beta = 6,
    edip_sigma = 7
};

std::vector<float> Make_Edip_Parameters()
{
    std::vector<float> parameters(4 * 8 + 8 * 9, 0.0f);
    const float cutoffs[2][2] = {{1.0f, 1.8f}, {1.2f, 1.6f}};
    for (int type_i = 0; type_i < kTypeCount; type_i += 1)
    {
        for (int type_j = 0; type_j < kTypeCount; type_j += 1)
        {
            float* pair =
                parameters.data() + 8 * (type_i * kTypeCount + type_j);
            pair[edip_alpha] = 1.1f;
            pair[edip_c] = 0.55f;
            pair[edip_a] = cutoffs[type_i][type_j];
            pair[edip_A] = 1.2f;
            pair[edip_B] = 1.05f;
            pair[edip_rho] = 1.5f;
            pair[edip_beta] = 0.08f;
            pair[edip_sigma] = 0.18f;
        }
    }
    for (int triple = 0; triple < 8; triple += 1)
    {
        float* value = parameters.data() + 4 * 8 + triple * 9;
        value[0] = 0.12f;
        value[1] = 0.45f;
        value[2] = 0.85f;
        value[3] = 1.1f;
        value[4] = 0.09f;
        value[5] = -0.15f;
        value[6] = 0.18f;
        value[7] = 1.05f;
        value[8] = 0.14f;
    }
    return parameters;
}

Rows Edip_Oracle_Rows(const std::vector<VECTOR>& coordinates,
                      const std::vector<int>& types,
                      const std::vector<float>& parameters)
{
    Rows rows(kAtomCount);
    for (int atom_i = 0; atom_i < kAtomCount; atom_i += 1)
    {
        for (int atom_j = 0; atom_j < kAtomCount; atom_j += 1)
        {
            if (atom_i == atom_j || Is_Excluded(atom_i, atom_j)) continue;
            const int pair_index = types[atom_i] * kTypeCount + types[atom_j];
            const float cutoff = parameters[8 * pair_index + edip_a];
            if (Distance(coordinates[atom_i], coordinates[atom_j]) < cutoff)
                rows[atom_i].push_back(atom_j);
        }
        std::sort(rows[atom_i].begin(), rows[atom_i].end());
    }
    return rows;
}

std::vector<double> Edip_Oracle_Z(const Rows& rows,
                                  const std::vector<VECTOR>& coordinates,
                                  const std::vector<int>& types,
                                  const std::vector<float>& parameters)
{
    std::vector<double> z(kAtomCount, 0.0);
    for (int atom_i = 0; atom_i < kAtomCount; atom_i += 1)
    {
        for (int atom_j : rows[atom_i])
        {
            const float* pair =
                parameters.data() +
                8 * (types[atom_i] * kTypeCount + types[atom_j]);
            const double distance =
                Distance(coordinates[atom_i], coordinates[atom_j]);
            if (distance < pair[edip_c])
                z[atom_i] += 1.0;
            else if (distance < pair[edip_a])
            {
                const double scaled =
                    (distance - pair[edip_c]) / (pair[edip_a] - pair[edip_c]);
                z[atom_i] +=
                    std::exp(pair[edip_alpha] / (1.0 - std::pow(scaled, -3.0)));
            }
        }
    }
    return z;
}

double Edip_Reference_Energy(const Rows& rows,
                             const std::vector<VECTOR>& coordinates,
                             const std::vector<int>& types,
                             const std::vector<float>& parameters,
                             const std::vector<double>& z)
{
    double energy = 0.0;
    for (int atom_i = 0; atom_i < kAtomCount; atom_i += 1)
    {
        for (int atom_j : rows[atom_i])
        {
            const float* pair =
                parameters.data() +
                8 * (types[atom_i] * kTypeCount + types[atom_j]);
            const double rij =
                Distance(coordinates[atom_i], coordinates[atom_j]);
            if (atom_j > atom_i && rij < pair[edip_a])
            {
                const double bracket =
                    pair[edip_A] *
                    (2.0 * std::pow(pair[edip_B] / rij, pair[edip_rho]) -
                     std::exp(-pair[edip_beta] * z[atom_i] * z[atom_i]) -
                     std::exp(-pair[edip_beta] * z[atom_j] * z[atom_j]));
                energy +=
                    bracket * std::exp(pair[edip_sigma] / (rij - pair[edip_a]));
            }
        }

        for (size_t j_index = 0; j_index < rows[atom_i].size(); j_index += 1)
        {
            const int atom_j = rows[atom_i][j_index];
            const auto drij =
                Minimum_Image(coordinates[atom_i], coordinates[atom_j], 10.0f)
                    .dr;
            const double rij = std::sqrt(drij * drij);
            const int pair_ij = types[atom_i] * kTypeCount + types[atom_j];
            const double a1 = parameters[8 * pair_ij + edip_a];
            for (size_t k_index = j_index + 1; k_index < rows[atom_i].size();
                 k_index += 1)
            {
                const int atom_k = rows[atom_i][k_index];
                const auto drik = Minimum_Image(coordinates[atom_i],
                                                coordinates[atom_k], 10.0f)
                                      .dr;
                const double rik = std::sqrt(drik * drik);
                const int pair_ik = types[atom_i] * kTypeCount + types[atom_k];
                const double a2 = parameters[8 * pair_ik + edip_a];
                if (!(rij < a1 && rik < a2)) continue;
                const int triple_index = pair_ij * kTypeCount + types[atom_k];
                const float* triple =
                    parameters.data() + 4 * 8 + triple_index * 9;
                const double cosine = (drij * drik) / (rij * rik);
                double angular =
                    triple[5] +
                    triple[6] * (triple[7] * std::exp(-triple[8] * z[atom_i]) -
                                 std::exp(-2.0 * triple[8] * z[atom_i])) +
                    cosine;
                angular *= angular;
                const double q = triple[3] * std::exp(-triple[4] * z[atom_i]);
                double value = triple[2] * (1.0 - std::exp(-q * angular) +
                                            triple[0] * q * angular);
                value *=
                    std::exp(triple[1] / (rij - a1) + triple[1] / (rik - a2));
                energy += value;
            }
        }
    }
    return energy;
}

void Release_Edip(EDIP_INFORMATION* edip)
{
    Device_Free(&edip->d_atom_type);
    Device_Free(&edip->d_parameters);
    Device_Free(&edip->z);
    edip->dE_dz = nullptr;
    Device_Free(&edip->d_clustered_neighbor_counts);
    Device_Free(&edip->d_clustered_neighbor_offsets);
    Device_Free(&edip->d_clustered_neighbor_atoms);
}

void Test_Edip_Oracle(const Clustered_Fixture& fixture,
                      const std::vector<VECTOR>& coordinates,
                      const std::vector<int>& types, const LTMatrix3& cell,
                      const LTMatrix3& rcell)
{
    const std::vector<float> parameters = Make_Edip_Parameters();
    EDIP_INFORMATION edip;
    edip.is_initialized = true;
    edip.atom_numbers = kAtomCount;
    edip.atom_type_numbers = kTypeCount;
    edip.pair_type_numbers = 4;
    edip.triple_type_numbers = 8;
    edip.cut = 1.8f;
    edip.d_atom_type = Device_Copy(types);
    edip.d_parameters = Device_Copy(parameters);
    edip.z = Device_Allocate<float>(2 * kAtomCount);
    edip.dE_dz = edip.z + kAtomCount;

    VECTOR* device_coordinates = Device_Copy(coordinates);
    VECTOR* device_force = Device_Allocate<VECTOR>(kAtomCount);
    deviceMemset(device_force, 0, sizeof(VECTOR) * kAtomCount);
    const char* failure_reason = nullptr;
    const bool accepted = edip.EDIP_Force_Clustered(
        fixture.view, device_coordinates, device_force, cell, rcell, 0, nullptr,
        0, nullptr, &failure_reason);
    Check(accepted, "EDIP accepts the oracle clustered fixture");
    if (!accepted && failure_reason != nullptr)
        std::fprintf(stderr, "EDIP rejection: %s\n", failure_reason);

    const Rows actual_rows = Read_Rows(
        edip.d_clustered_neighbor_offsets, edip.d_clustered_neighbor_atoms,
        kAtomCount, edip.clustered_neighbor_numbers);
    const Rows expected_rows = Edip_Oracle_Rows(coordinates, types, parameters);
    Check_Rows(actual_rows, expected_rows,
               "EDIP canonical directed center-neighbor rows");
    Check(Edip_Triplets(actual_rows, coordinates) ==
              Edip_Triplets(expected_rows, coordinates),
          "EDIP canonical unordered center triplets with image identity");

    const std::vector<float> actual_z = Device_Read(edip.z, kAtomCount);
    const std::vector<float> actual_dE_dz = Device_Read(edip.dE_dz, kAtomCount);
    const std::vector<double> expected_z =
        Edip_Oracle_Z(expected_rows, coordinates, types, parameters);
    for (int atom_i = 0; atom_i < kAtomCount; atom_i += 1)
        Check_Near(actual_z[atom_i], expected_z[atom_i], 2.0e-5,
                   "EDIP z oracle", atom_i);

    constexpr double finite_difference_step = 2.0e-3;
    for (int atom_i = 0; atom_i < kAtomCount; atom_i += 1)
    {
        auto z_plus = expected_z;
        auto z_minus = expected_z;
        z_plus[atom_i] += finite_difference_step;
        z_minus[atom_i] -= finite_difference_step;
        const double derivative =
            (Edip_Reference_Energy(expected_rows, coordinates, types,
                                   parameters, z_plus) -
             Edip_Reference_Energy(expected_rows, coordinates, types,
                                   parameters, z_minus)) /
            (2.0 * finite_difference_step);
        Check_Near(actual_dE_dz[atom_i], derivative, 3.0e-3,
                   "EDIP dE/dz finite-difference oracle", atom_i);
    }

    Check(Distance(coordinates[0], coordinates[1]) == 1.0f &&
              std::find(expected_rows[0].begin(), expected_rows[0].end(), 1) ==
                  expected_rows[0].end(),
          "EDIP strict cutoff rejects the exact-boundary pair");
    Check(std::find(expected_rows[0].begin(), expected_rows[0].end(), 2) !=
                  expected_rows[0].end() &&
              std::find(expected_rows[2].begin(), expected_rows[2].end(), 0) ==
                  expected_rows[2].end(),
          "EDIP oracle exercises asymmetric directed cutoffs");
    Check(std::find(expected_rows[0].begin(), expected_rows[0].end(), 4) ==
              expected_rows[0].end(),
          "EDIP oracle exercises clustered exclusions");

    Device_Free(&device_coordinates);
    Device_Free(&device_force);
    Release_Edip(&edip);
}

enum Tersoff_Parameter
{
    tersoff_m = 0,
    tersoff_gamma,
    tersoff_lam3,
    tersoff_c,
    tersoff_d,
    tersoff_h,
    tersoff_n,
    tersoff_beta,
    tersoff_lam2,
    tersoff_B,
    tersoff_R,
    tersoff_D,
    tersoff_lam1,
    tersoff_A,
    tersoff_c1,
    tersoff_c2,
    tersoff_c3,
    tersoff_c4,
    tersoff_stride
};

struct Tersoff_Parameters
{
    std::vector<float> values;
    std::vector<int> map;
    std::vector<float> center_cutoffs;
};

Tersoff_Parameters Make_Tersoff_Parameters()
{
    Tersoff_Parameters result;
    result.values.resize(8 * tersoff_stride, 0.0f);
    result.map.resize(8);
    const float cutoffs[2][2][2] = {{{1.0f, 1.8f}, {1.2f, 1.6f}},
                                    {{1.55f, 1.45f}, {1.35f, 1.65f}}};
    for (int type_i = 0; type_i < kTypeCount; type_i += 1)
    {
        for (int type_j = 0; type_j < kTypeCount; type_j += 1)
        {
            for (int type_k = 0; type_k < kTypeCount; type_k += 1)
            {
                const int index = type_i * 4 + type_j * 2 + type_k;
                result.map[index] = index;
                float* parameter =
                    result.values.data() + index * tersoff_stride;
                parameter[tersoff_m] = 3.0f;
                parameter[tersoff_gamma] = 1.0f;
                parameter[tersoff_lam3] = 0.0f;
                parameter[tersoff_c] = 1.0f;
                parameter[tersoff_d] = 1.0f;
                parameter[tersoff_h] = -0.25f;
                parameter[tersoff_n] = 1.0f;
                parameter[tersoff_beta] = 0.5f;
                parameter[tersoff_lam2] = 1.1f;
                parameter[tersoff_B] = 1.0f;
                parameter[tersoff_R] = cutoffs[type_i][type_j][type_k] - 0.1f;
                parameter[tersoff_D] = 0.1f;
                parameter[tersoff_lam1] = 1.8f;
                parameter[tersoff_A] = 2.0f;
                parameter[tersoff_c1] = 1.0e16f;
                parameter[tersoff_c2] = 1.0e8f;
                parameter[tersoff_c3] = 1.0e-8f;
                parameter[tersoff_c4] = 1.0e-16f;
            }
        }
    }
    result.center_cutoffs.resize(4, 0.0f);
    for (int type_i = 0; type_i < kTypeCount; type_i += 1)
        for (int type_k = 0; type_k < kTypeCount; type_k += 1)
            for (int type_j = 0; type_j < kTypeCount; type_j += 1)
                result.center_cutoffs[type_i * 2 + type_k] =
                    std::max(result.center_cutoffs[type_i * 2 + type_k],
                             cutoffs[type_i][type_j][type_k]);
    return result;
}

float Tersoff_Cutoff(const Tersoff_Parameters& parameters, int type_i,
                     int type_j, int type_k)
{
    const int map_index = type_i * 4 + type_j * 2 + type_k;
    const int parameter_index = parameters.map[map_index];
    const float* parameter =
        parameters.values.data() + parameter_index * tersoff_stride;
    return parameter[tersoff_R] + parameter[tersoff_D];
}

Rows Tersoff_Oracle_Rows(const std::vector<VECTOR>& coordinates,
                         const std::vector<int>& types,
                         const Tersoff_Parameters& parameters)
{
    Rows rows(kAtomCount);
    for (int atom_i = 0; atom_i < kAtomCount; atom_i += 1)
    {
        for (int atom_j = 0; atom_j < kAtomCount; atom_j += 1)
        {
            if (atom_i == atom_j || Is_Excluded(atom_i, atom_j)) continue;
            const float cutoff =
                parameters
                    .center_cutoffs[types[atom_i] * kTypeCount + types[atom_j]];
            if (Distance(coordinates[atom_i], coordinates[atom_j]) <= cutoff)
                rows[atom_i].push_back(atom_j);
        }
        std::sort(rows[atom_i].begin(), rows[atom_i].end());
    }
    return rows;
}

std::set<Tersoff_Tuple> Tersoff_Tuples(const Rows& rows,
                                       const std::vector<VECTOR>& coordinates,
                                       const std::vector<int>& types,
                                       const Tersoff_Parameters& parameters)
{
    std::set<Tersoff_Tuple> result;
    for (int atom_i = 0; atom_i < kAtomCount; atom_i += 1)
    {
        for (int atom_j : rows[atom_i])
        {
            if (Distance(coordinates[atom_i], coordinates[atom_j]) >
                Tersoff_Cutoff(parameters, types[atom_i], types[atom_j],
                               types[atom_j]))
                continue;
            for (int atom_k : rows[atom_i])
            {
                if (atom_k == atom_j) continue;
                if (Distance(coordinates[atom_i], coordinates[atom_k]) >
                    Tersoff_Cutoff(parameters, types[atom_i], types[atom_j],
                                   types[atom_k]))
                    continue;
                const auto shift_j = Minimum_Image(coordinates[atom_i],
                                                   coordinates[atom_j], 10.0f);
                const auto shift_k = Minimum_Image(coordinates[atom_i],
                                                   coordinates[atom_k], 10.0f);
                result.emplace(atom_i, atom_j, atom_k, shift_j.sx, shift_j.sy,
                               shift_j.sz, shift_k.sx, shift_k.sy, shift_k.sz);
            }
        }
    }
    return result;
}

std::set<Tersoff_Tuple> Tersoff_Oracle_Tuples(
    const std::vector<VECTOR>& coordinates, const std::vector<int>& types,
    const Tersoff_Parameters& parameters)
{
    std::set<Tersoff_Tuple> result;
    for (int atom_i = 0; atom_i < kAtomCount; atom_i += 1)
    {
        for (int atom_j = 0; atom_j < kAtomCount; atom_j += 1)
        {
            if (atom_i == atom_j || Is_Excluded(atom_i, atom_j)) continue;
            if (Distance(coordinates[atom_i], coordinates[atom_j]) >
                Tersoff_Cutoff(parameters, types[atom_i], types[atom_j],
                               types[atom_j]))
                continue;
            for (int atom_k = 0; atom_k < kAtomCount; atom_k += 1)
            {
                if (atom_k == atom_i || atom_k == atom_j ||
                    Is_Excluded(atom_i, atom_k))
                    continue;
                if (Distance(coordinates[atom_i], coordinates[atom_k]) >
                    Tersoff_Cutoff(parameters, types[atom_i], types[atom_j],
                                   types[atom_k]))
                    continue;
                const auto shift_j = Minimum_Image(coordinates[atom_i],
                                                   coordinates[atom_j], 10.0f);
                const auto shift_k = Minimum_Image(coordinates[atom_i],
                                                   coordinates[atom_k], 10.0f);
                result.emplace(atom_i, atom_j, atom_k, shift_j.sx, shift_j.sy,
                               shift_j.sz, shift_k.sx, shift_k.sy, shift_k.sz);
            }
        }
    }
    return result;
}

void Release_Tersoff(TERSOFF_INFORMATION* tersoff)
{
    Device_Free(&tersoff->d_atom_type);
    Device_Free(&tersoff->d_params);
    Device_Free(&tersoff->d_map);
    Device_Free(&tersoff->d_center_cutoffs);
    Device_Free(&tersoff->d_clustered_neighbor_counts);
    Device_Free(&tersoff->d_clustered_neighbor_offsets);
    Device_Free(&tersoff->d_clustered_neighbor_atoms);
}

void Test_Tersoff_Oracle(const Clustered_Fixture& fixture,
                         const std::vector<VECTOR>& coordinates,
                         const std::vector<int>& types, const LTMatrix3& cell,
                         const LTMatrix3& rcell)
{
    const Tersoff_Parameters parameters = Make_Tersoff_Parameters();
    TERSOFF_INFORMATION tersoff;
    tersoff.is_initialized = true;
    tersoff.atom_numbers = kAtomCount;
    tersoff.atom_type_numbers = kTypeCount;
    tersoff.n_unique_params = 8;
    tersoff.cut = *std::max_element(parameters.center_cutoffs.begin(),
                                    parameters.center_cutoffs.end());
    tersoff.d_atom_type = Device_Copy(types);
    tersoff.d_params = Device_Copy(parameters.values);
    tersoff.d_map = Device_Copy(parameters.map);
    tersoff.d_center_cutoffs = Device_Copy(parameters.center_cutoffs);

    VECTOR* device_coordinates = Device_Copy(coordinates);
    VECTOR* device_force = Device_Allocate<VECTOR>(kAtomCount);
    deviceMemset(device_force, 0, sizeof(VECTOR) * kAtomCount);
    const char* failure_reason = nullptr;
    const bool accepted = tersoff.TERSOFF_Force_Clustered(
        fixture.view, device_coordinates, device_force, cell, rcell, 0, nullptr,
        0, nullptr, &failure_reason);
    Check(accepted, "Tersoff accepts the oracle clustered fixture");
    if (!accepted && failure_reason != nullptr)
        std::fprintf(stderr, "Tersoff rejection: %s\n", failure_reason);

    const Rows actual_rows =
        Read_Rows(tersoff.d_clustered_neighbor_offsets,
                  tersoff.d_clustered_neighbor_atoms, kAtomCount,
                  tersoff.clustered_neighbor_numbers);
    const Rows expected_rows =
        Tersoff_Oracle_Rows(coordinates, types, parameters);
    Check_Rows(actual_rows, expected_rows,
               "Tersoff conservative directed center-neighbor rows");
    Check(Tersoff_Tuples(actual_rows, coordinates, types, parameters) ==
              Tersoff_Oracle_Tuples(coordinates, types, parameters),
          "Tersoff canonical directed (i,j,k) tuples with image identity");

    Check(Distance(coordinates[0], coordinates[1]) == 1.0f &&
              std::find(actual_rows[0].begin(), actual_rows[0].end(), 1) !=
                  actual_rows[0].end(),
          "Tersoff relation preserves an exact-boundary directed edge");
    const auto tuples =
        Tersoff_Tuples(actual_rows, coordinates, types, parameters);
    bool has_conservative_non_edge = false;
    if (std::find(actual_rows[0].begin(), actual_rows[0].end(), 3) !=
        actual_rows[0].end())
    {
        has_conservative_non_edge = true;
        for (const auto& tuple : tuples)
            if (std::get<0>(tuple) == 0 && std::get<1>(tuple) == 3)
                has_conservative_non_edge = false;
    }
    Check(has_conservative_non_edge,
          "Tersoff oracle distinguishes conservative relation from exact edge "
          "cutoff");
    Check(std::find(expected_rows[0].begin(), expected_rows[0].end(), 4) ==
              expected_rows[0].end(),
          "Tersoff oracle exercises clustered exclusions");

    Device_Free(&device_coordinates);
    Device_Free(&device_force);
    Release_Tersoff(&tersoff);
}

struct Reaxff_Hb_Result
{
    bool accepted = false;
    float energy = 0.0f;
    std::vector<float> atom_energy;
    std::vector<VECTOR> force;
    float dE_dBO_s = 0.0f;
    const char* failure_reason = nullptr;
};

Reaxff_Hb_Result Run_Reaxff_Hb_Case(const Clustered_Fixture& fixture,
                                    const std::vector<VECTOR>& coordinates,
                                    const LTMatrix3& cell,
                                    const LTMatrix3& rcell)
{
    const std::vector<int> types = {0, 1, 1, 0, 0};
    std::vector<REAXFF_HB_Info> info(kTypeCount * kTypeCount * kTypeCount);
    info[(1 * kTypeCount + 0) * kTypeCount + 1] = {0, 1};
    const std::vector<REAXFF_HB_Entry> entries = {{2.0f, 2.0f, 1.0f, 0.0f}};

    REAXFF_HYDROGEN_BOND hb;
    hb.is_initialized = true;
    hb.atom_numbers = kAtomCount;
    hb.atom_type_numbers = kTypeCount;
    hb.hydrogen_numbers = 1;
    hb.d_atom_type = Device_Copy(types);
    hb.d_is_hydrogen = Device_Copy(std::vector<int>{1, 0, 0, 0, 0});
    hb.d_hydrogen_atoms = Device_Copy(std::vector<int>{0});
    hb.d_clustered_atom_to_sorted = Device_Allocate<int>(kAtomCount);
    hb.d_hb_info = Device_Copy(info);
    hb.d_hb_entries = Device_Copy(entries);
    hb.d_energy_hb_sum = Device_Allocate<float>(1);
    hb.d_dE_dBO_s = Device_Allocate<float>(1);
    hb.d_dE_dBO_pi = Device_Allocate<float>(1);
    hb.d_dE_dBO_pi2 = Device_Allocate<float>(1);
    deviceMemset(hb.d_dE_dBO_s, 0, sizeof(float));
    deviceMemset(hb.d_dE_dBO_pi, 0, sizeof(float));
    deviceMemset(hb.d_dE_dBO_pi2, 0, sizeof(float));

    REAXFF_BOND_ORDER bo;
    bo.d_corrected_bo_s = Device_Copy(std::vector<float>{1.0f});
    bo.d_corrected_bo_pi = Device_Copy(std::vector<float>{0.0f});
    bo.d_corrected_bo_pi2 = Device_Copy(std::vector<float>{0.0f});
    bo.d_bond_count = Device_Copy(std::vector<int>{1, 0, 1, 0, 0});
    bo.d_bond_offset = Device_Copy(std::vector<int>{0, 1, 1, 2, 2, 2});
    bo.d_bond_nbr = Device_Copy(std::vector<int>{2, 0});
    bo.d_bond_idx = Device_Copy(std::vector<int>{0, 0});

    VECTOR* device_coordinates = Device_Copy(coordinates);
    VECTOR* device_force = Device_Allocate<VECTOR>(kAtomCount);
    float* device_atom_energy = Device_Allocate<float>(kAtomCount);
    deviceMemset(device_force, 0, sizeof(VECTOR) * kAtomCount);
    deviceMemset(device_atom_energy, 0, sizeof(float) * kAtomCount);

    Reaxff_Hb_Result result;
    result.accepted = hb.Calculate_HB_Energy_And_Force_Clustered(
        fixture.view, kAtomCount, device_coordinates, device_force, cell, rcell,
        &bo, 1, device_atom_energy, 0, nullptr, &result.failure_reason);
    result.energy = Device_Read(hb.d_energy_hb_sum, 1)[0];
    result.atom_energy = Device_Read(device_atom_energy, kAtomCount);
    result.force = Device_Read(device_force, kAtomCount);
    result.dE_dBO_s = Device_Read(hb.d_dE_dBO_s, 1)[0];

    Device_Free(&device_coordinates);
    Device_Free(&device_force);
    Device_Free(&device_atom_energy);
    Device_Free(&hb.d_atom_type);
    Device_Free(&hb.d_is_hydrogen);
    Device_Free(&hb.d_hydrogen_atoms);
    Device_Free(&hb.d_clustered_atom_to_sorted);
    Device_Free(&hb.d_hb_info);
    Device_Free(&hb.d_hb_entries);
    Device_Free(&hb.d_energy_hb_sum);
    Device_Free(&hb.d_dE_dBO_s);
    Device_Free(&hb.d_dE_dBO_pi);
    Device_Free(&hb.d_dE_dBO_pi2);
    Device_Free(&bo.d_corrected_bo_s);
    Device_Free(&bo.d_corrected_bo_pi);
    Device_Free(&bo.d_corrected_bo_pi2);
    Device_Free(&bo.d_bond_count);
    Device_Free(&bo.d_bond_offset);
    Device_Free(&bo.d_bond_nbr);
    Device_Free(&bo.d_bond_idx);
    return result;
}

void Check_Reaxff_Hb_Result(const Reaxff_Hb_Result& result,
                            double expected_energy, const char* label)
{
    Check(result.accepted, label);
    if (!result.accepted && result.failure_reason != nullptr)
        std::fprintf(stderr, "ReaxFF HB rejection: %s\n",
                     result.failure_reason);
    Check_Near(result.energy, expected_energy, 2.0e-5,
               "ReaxFF HB energy oracle", 0);
    Check_Near(result.atom_energy[0], expected_energy, 2.0e-5,
               "ReaxFF HB hydrogen atom-energy ownership", 0);
    for (int atom = 1; atom < kAtomCount; atom += 1)
        Check_Near(result.atom_energy[atom], 0.0, 2.0e-6,
                   "ReaxFF HB non-hydrogen atom energy", atom);

    VECTOR force_sum = {};
    for (const VECTOR& force : result.force) force_sum = force_sum + force;
    Check_Near(force_sum.x, 0.0, 2.0e-5, "ReaxFF HB force conservation x", 0);
    Check_Near(force_sum.y, 0.0, 2.0e-5, "ReaxFF HB force conservation y", 0);
    Check_Near(force_sum.z, 0.0, 2.0e-5, "ReaxFF HB force conservation z", 0);

    const double expected_derivative =
        expected_energy == 0.0 ? 0.0 : 0.5 * std::exp(-1.0);
    Check_Near(result.dE_dBO_s, expected_derivative, 2.0e-5,
               "ReaxFF HB bond-order derivative", 0);
}

void Test_Reaxff_Hb_Oracle()
{
    constexpr float box = 20.0f;
    const LTMatrix3 cell = Diagonal_Matrix(box);
    const LTMatrix3 rcell = Diagonal_Matrix(1.0f / box);
    Clustered_Fixture fixture(rcell);
    const double expected_energy = 0.5 * (1.0 - std::exp(-1.0));

    const std::vector<VECTOR> cross_periodic = {{0.25f, 1.0f, 1.0f},
                                                {19.25f, 1.0f, 1.0f},
                                                {0.25f, 2.0f, 1.0f},
                                                {10.0f, 10.0f, 10.0f},
                                                {10.0f, 12.0f, 10.0f}};
    const Reaxff_Hb_Result cross_result =
        Run_Reaxff_Hb_Case(fixture, cross_periodic, cell, rcell);
    Check_Reaxff_Hb_Result(cross_result, expected_energy,
                           "ReaxFF HB accepts cross-PBC clustered pair");
    constexpr float difference_step = 1.0e-3f;
    std::vector<VECTOR> donor_plus = cross_periodic;
    std::vector<VECTOR> donor_minus = cross_periodic;
    donor_plus[2].x += difference_step;
    donor_minus[2].x -= difference_step;
    const float energy_plus =
        Run_Reaxff_Hb_Case(fixture, donor_plus, cell, rcell).energy;
    const float energy_minus =
        Run_Reaxff_Hb_Case(fixture, donor_minus, cell, rcell).energy;
    const double donor_force_x =
        -(energy_plus - energy_minus) / (2.0 * difference_step);
    Check_Near(cross_result.force[2].x, donor_force_x, 4.0e-4,
               "ReaxFF HB angular force finite-difference oracle", 2);

    const std::vector<VECTOR> cutoff_inside = {{1.0f, 1.0f, 1.0f},
                                               {8.499f, 1.0f, 1.0f},
                                               {1.0f, 2.0f, 1.0f},
                                               {10.0f, 10.0f, 10.0f},
                                               {10.0f, 12.0f, 10.0f}};
    Check_Reaxff_Hb_Result(
        Run_Reaxff_Hb_Case(fixture, cutoff_inside, cell, rcell),
        expected_energy, "ReaxFF HB includes r_AH=7.5-epsilon");

    const std::vector<VECTOR> cutoff_outside = {{1.0f, 1.0f, 1.0f},
                                                {8.501f, 1.0f, 1.0f},
                                                {1.0f, 2.0f, 1.0f},
                                                {10.0f, 10.0f, 10.0f},
                                                {10.0f, 12.0f, 10.0f}};
    Check_Reaxff_Hb_Result(
        Run_Reaxff_Hb_Case(fixture, cutoff_outside, cell, rcell), 0.0,
        "ReaxFF HB excludes r_AH=7.5+epsilon");
}
}  // namespace

int main()
{
    const std::vector<VECTOR> coordinates = Test_Coordinates();
    const std::vector<int> types = Test_Types();
    const LTMatrix3 cell = Diagonal_Matrix(10.0f);
    const LTMatrix3 rcell = Diagonal_Matrix(0.1f);
    Clustered_Fixture fixture(rcell);

    Test_Edip_Oracle(fixture, coordinates, types, cell, rcell);
    Test_Tersoff_Oracle(fixture, coordinates, types, cell, rcell);
    Test_Reaxff_Hb_Oracle();

    if (failures != 0)
    {
        std::fprintf(stderr, "manybody clustered oracle failures: %d\n",
                     failures);
        return 1;
    }
    std::printf(
        "manybody clustered oracles passed: EDIP relation/triplets/z/dE_dz; "
        "Tersoff relation/directed tuples; ReaxFF HB "
        "PBC/cutoff/energy/dE_dBO\n");
    return 0;
}
