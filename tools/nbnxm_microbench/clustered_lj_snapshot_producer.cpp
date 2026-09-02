#include "clustered_lj_snapshot_producer.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include "Domain_decomposition/Domain_decomposition.h"
#include "Lennard_Jones_force/Lennard_Jones_force.h"
#include "MD_core/MD_core.h"
#include "PM_force/PM_force.h"
#include "nbnxm_microbench_snapshot.h"
#include "neighbor_list/provider/internal.h"

extern MD_INFORMATION md_info;
extern DOMAIN_INFORMATION dd;
extern LENNARD_JONES_INFORMATION lj;
extern Particle_Mesh pm;
extern CONTROLLER controller;

namespace nbnxm_microbench
{
namespace
{

template <typename T>
std::vector<T> CopyDeviceVector(const T* device_ptr, size_t count)
{
    std::vector<T> host(count);
    if (device_ptr != nullptr && count > 0)
    {
        deviceMemcpy(host.data(), device_ptr, sizeof(T) * count,
                     deviceMemcpyDeviceToHost);
    }
    return host;
}

template <typename T>
class DeviceBuffer
{
   public:
    explicit DeviceBuffer(size_t count) : count_(count)
    {
        if (count_ > 0)
        {
            Device_Malloc_Safely(reinterpret_cast<void**>(&data_),
                                 sizeof(T) * count_);
            deviceMemset(data_, 0, sizeof(T) * count_);
        }
    }

    ~DeviceBuffer()
    {
        if (data_ != nullptr)
        {
            deviceFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() { return data_; }

   private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

LTMatrix3POD ToPOD(const LTMatrix3& value)
{
    return {value.a11, value.a21, value.a22, value.a31, value.a32, value.a33};
}

Float4POD ToPOD(const float4& value)
{
    return {value.x, value.y, value.z, value.w};
}

Float2POD ToPOD(const float2& value) { return {value.x, value.y}; }

Float4POD ToPOD(const VECTOR& value)
{
    return {value.x, value.y, value.z, 0.0f};
}

SpongeGmxpackedSciPOD ToPOD(const LJ_CLUSTERED_GMXPACKED_SCI& value)
{
    return {value.supercluster_id, value.shift_id, value.cjpacked_begin,
            value.cjpacked_end};
}

SpongeGmxpackedCjPOD ToPOD(const LJ_CLUSTERED_GMXPACKED_CJ& value)
{
    SpongeGmxpackedCjPOD result = {};
    std::memcpy(result.cj, value.cj, sizeof(result.cj));
    for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
    {
        result.split[split].imask = value.split[split].imask;
        result.split[split].exclusion_index =
            value.split[split].exclusion_index;
    }
    return result;
}

SpongeGmxpackedExclusionPOD ToPOD(const LJ_CLUSTERED_GMXPACKED_EXCLUSION& value)
{
    SpongeGmxpackedExclusionPOD result = {};
    std::memcpy(result.pair, value.pair, sizeof(result.pair));
    return result;
}

template <typename Source, typename Destination, typename Converter>
void CopyConvertedDeviceVector(const Source* device_ptr, size_t count,
                               std::vector<Destination>* destination,
                               Converter converter)
{
    const auto source = CopyDeviceVector(device_ptr, count);
    destination->reserve(source.size());
    for (const Source& value : source)
    {
        destination->push_back(converter(value));
    }
}

void PopulatePairOracleMetadata(const ClusteredNeighborProvider& layout,
                                SpongeGmxpackedForceOnlySnapshot* snapshot)
{
    const auto& domain = ClusteredNeighborProviderInternal::Domain(&layout);
    const auto& spatial = ClusteredNeighborProviderInternal::Spatial(&layout);
    if (domain.local_atom_count <= 0 || domain.atom_local == nullptr ||
        domain.excluded_list_start == nullptr ||
        domain.excluded_numbers == nullptr)
    {
        return;
    }
    snapshot->atom_local = CopyDeviceVector(
        domain.atom_local, static_cast<size_t>(spatial.total_atom_numbers));
    snapshot->excluded_list_start =
        CopyDeviceVector(domain.excluded_list_start,
                         static_cast<size_t>(domain.local_atom_count));
    snapshot->excluded_numbers = CopyDeviceVector(
        domain.excluded_numbers, static_cast<size_t>(domain.local_atom_count));
    const int total_excluded = snapshot->excluded_list_start.empty()
                                   ? 0
                                   : snapshot->excluded_list_start.back() +
                                         snapshot->excluded_numbers.back();
    if (total_excluded > 0 && domain.excluded_list != nullptr)
    {
        snapshot->excluded_list = CopyDeviceVector(
            domain.excluded_list, static_cast<size_t>(total_excluded));
    }
}

SpongeGmxpackedForceOnlySnapshot CaptureForceOnlySnapshot()
{
    const LJClusteredWorkspace* workspace = lj.clustered_workspace;
    if (workspace == nullptr)
    {
        throw std::runtime_error("regular LJ has no clustered workspace");
    }
    const ClusteredNeighborProvider* provider = lj.clustered_neighbor_provider;
    if (provider == nullptr)
    {
        throw std::runtime_error("regular LJ has no clustered provider");
    }
    const ClusteredNeighborProvider& layout = *provider;
    const auto& config = ClusteredNeighborProviderInternal::Config(&layout);
    const auto& domain = ClusteredNeighborProviderInternal::Domain(&layout);
    const auto& spatial = ClusteredNeighborProviderInternal::Spatial(&layout);
    const auto& pair_list =
        ClusteredNeighborProviderInternal::PairList(&layout);
    if (spatial.total_atom_numbers <= 0 ||
        pair_list.gmxpacked_sci_numbers <= 0 ||
        pair_list.gmxpacked_cjpacked_numbers <= 0 ||
        pair_list.gmxpacked_exclusion_numbers <= 0)
    {
        throw std::runtime_error("clustered gmxpacked payload is empty");
    }
    if (pair_list.gmxpacked_sci.data == nullptr ||
        pair_list.gmxpacked_cjpacked.data == nullptr ||
        pair_list.gmxpacked_exclusions.data == nullptr ||
        pair_list.pair_shift_bits.data == nullptr ||
        spatial.clusters.offsets.data == nullptr ||
        spatial.clusters.valid_masks.data == nullptr ||
        spatial.clusters.local_masks.data == nullptr ||
        spatial.superclusters.offsets.data == nullptr ||
        workspace->d_sorted_atom_ids == nullptr ||
        workspace->d_sorted_xq == nullptr ||
        workspace->d_sorted_lj_type == nullptr ||
        workspace->d_sorted_lj_comb == nullptr || lj.d_LJ_AB_packed == nullptr)
    {
        throw std::runtime_error("clustered snapshot fields are not ready");
    }

    SpongeGmxpackedForceOnlySnapshot snapshot = {};
    auto& header = snapshot.header;
    header.file = MakeFileHeader(SnapshotKind::spongeGmxpackedForceOnly);
    header.cluster_size = static_cast<uint32_t>(config.cluster_size);
    header.super_cluster_clusters =
        static_cast<uint32_t>(config.clusters_per_supercluster);
    header.warp_split_count = kClusteredWarpSplitCount;
    header.j_group_size = kClusteredJGroupSize;
    header.force_storage_sorted = 1u;
    header.use_lj_comb = lj.gmxpacked_lj_comb_table_compatible ? 1u : 0u;
    header.lj_type_matrix_stride = 0u;
    header.cluster_numbers = static_cast<uint64_t>(spatial.cluster_numbers);
    header.super_cluster_numbers =
        static_cast<uint64_t>(spatial.super_cluster_numbers);
    header.sci_numbers = static_cast<uint64_t>(pair_list.gmxpacked_sci_numbers);
    header.cjpacked_numbers =
        static_cast<uint64_t>(pair_list.gmxpacked_cjpacked_numbers);
    header.excl_numbers =
        static_cast<uint64_t>(pair_list.gmxpacked_exclusion_numbers);
    header.pair_shift_word_numbers = static_cast<uint64_t>(
        pair_list.gmxpacked_cjpacked_numbers * kClusteredJGroupSize);
    header.total_atom_numbers =
        static_cast<uint64_t>(spatial.total_atom_numbers);
    header.local_atom_numbers = static_cast<uint64_t>(domain.local_atom_count);
    header.lj_param_numbers = static_cast<uint64_t>(lj.pair_type_numbers);
    header.cutoff = lj.cutoff;
    header.pme_beta = pm.beta;
    header.cell = ToPOD(md_info.pbc.cell);

    snapshot.cluster_offsets =
        CopyDeviceVector(spatial.clusters.offsets.data,
                         static_cast<size_t>(spatial.cluster_numbers));
    snapshot.cluster_valid_masks =
        CopyDeviceVector(spatial.clusters.valid_masks.data,
                         static_cast<size_t>(spatial.cluster_numbers));
    snapshot.cluster_local_masks =
        CopyDeviceVector(spatial.clusters.local_masks.data,
                         static_cast<size_t>(spatial.cluster_numbers));
    snapshot.super_cluster_offsets = CopyDeviceVector(
        spatial.superclusters.offsets.data,
        static_cast<size_t>(spatial.super_cluster_numbers + 1));
    CopyConvertedDeviceVector(spatial.clusters.centers.data,
                              static_cast<size_t>(spatial.cluster_numbers),
                              &snapshot.cluster_centers,
                              [](const VECTOR& value) { return ToPOD(value); });
    CopyConvertedDeviceVector(spatial.clusters.extents.data,
                              static_cast<size_t>(spatial.cluster_numbers),
                              &snapshot.cluster_extents,
                              [](const VECTOR& value) { return ToPOD(value); });
    CopyConvertedDeviceVector(
        spatial.superclusters.centers.data,
        static_cast<size_t>(spatial.super_cluster_numbers),
        &snapshot.super_cluster_centers,
        [](const VECTOR& value) { return ToPOD(value); });
    CopyConvertedDeviceVector(
        spatial.superclusters.sizes.data,
        static_cast<size_t>(spatial.super_cluster_numbers),
        &snapshot.super_cluster_sizes,
        [](const VECTOR& value) { return ToPOD(value); });
    CopyConvertedDeviceVector(
        pair_list.gmxpacked_sci.data,
        static_cast<size_t>(pair_list.gmxpacked_sci_numbers), &snapshot.sci,
        [](const LJ_CLUSTERED_GMXPACKED_SCI& value) { return ToPOD(value); });
    CopyConvertedDeviceVector(
        pair_list.gmxpacked_cjpacked.data,
        static_cast<size_t>(pair_list.gmxpacked_cjpacked_numbers),
        &snapshot.cjpacked,
        [](const LJ_CLUSTERED_GMXPACKED_CJ& value) { return ToPOD(value); });
    CopyConvertedDeviceVector(
        pair_list.gmxpacked_exclusions.data,
        static_cast<size_t>(pair_list.gmxpacked_exclusion_numbers),
        &snapshot.excl, [](const LJ_CLUSTERED_GMXPACKED_EXCLUSION& value)
        { return ToPOD(value); });
    snapshot.pair_shift_bits = CopyDeviceVector(
        pair_list.pair_shift_bits.data,
        static_cast<size_t>(pair_list.gmxpacked_cjpacked_numbers *
                            kClusteredJGroupSize));
    if (pair_list.gmxpacked_pair_shift_sci_safe_flags.data != nullptr)
    {
        snapshot.sci_shift_safe_flags = CopyDeviceVector(
            pair_list.gmxpacked_pair_shift_sci_safe_flags.data,
            static_cast<size_t>(pair_list.gmxpacked_sci_numbers));
    }
    else
    {
        snapshot.sci_shift_safe_flags.assign(
            static_cast<size_t>(pair_list.gmxpacked_sci_numbers), 0);
    }

    if (spatial.tree.cornerstone_state != nullptr &&
        spatial.tree.cornerstone_state->octree.numLeafNodes > 0 &&
        spatial.tree.cornerstone_state->octree.numNodes > 0 &&
        spatial.leaves.cluster_starts.data != nullptr &&
        spatial.leaves.cluster_ends.data != nullptr &&
        spatial.candidates.sci_supercluster_ids.data != nullptr &&
        spatial.candidates.leaf_offsets.data != nullptr)
    {
        const auto& octree = spatial.tree.cornerstone_state->octree;
        const size_t leaf_count = static_cast<size_t>(octree.numLeafNodes);
        const size_t node_count = static_cast<size_t>(octree.numNodes);
        const int candidate_sci_count =
            pair_list.gmxpacked_sci_numbers > 0
                ? pair_list.gmxpacked_sci_numbers
                : spatial.candidates.candidate_sci_numbers;
        const int sci_supercluster_id_count =
            (candidate_sci_count + kClusteredShiftCount - 1) /
            kClusteredShiftCount;
        snapshot.leaf_cluster_starts =
            CopyDeviceVector(spatial.leaves.cluster_starts.data, leaf_count);
        snapshot.leaf_cluster_ends =
            CopyDeviceVector(spatial.leaves.cluster_ends.data, leaf_count);
        snapshot.leaf_all_local.assign(leaf_count, 0);
        snapshot.octree_prefixes =
            CopyDeviceVector(octree.prefixes.data(), node_count);
        snapshot.octree_child_offsets =
            CopyDeviceVector(octree.childOffsets.data(), node_count);
        snapshot.octree_parents =
            CopyDeviceVector(octree.parents.data(), octree.parents.size());
        snapshot.octree_internal_to_leaf =
            CopyDeviceVector(octree.internalToLeaf.data(), node_count);
        snapshot.sci_supercluster_ids =
            CopyDeviceVector(spatial.candidates.sci_supercluster_ids.data,
                             static_cast<size_t>(sci_supercluster_id_count));
        snapshot.candidate_leaf_offsets =
            CopyDeviceVector(spatial.candidates.leaf_offsets.data,
                             static_cast<size_t>(candidate_sci_count + 1));
        const int candidate_leaf_count =
            snapshot.candidate_leaf_offsets.empty()
                ? 0
                : snapshot.candidate_leaf_offsets.back();
        if (candidate_leaf_count > 0 &&
            spatial.candidates.leaf_ids.data != nullptr)
        {
            snapshot.candidate_leaf_ids =
                CopyDeviceVector(spatial.candidates.leaf_ids.data,
                                 static_cast<size_t>(candidate_leaf_count));
        }
    }

    snapshot.sorted_atom_ids =
        CopyDeviceVector(workspace->d_sorted_atom_ids,
                         static_cast<size_t>(spatial.total_atom_numbers));
    CopyConvertedDeviceVector(
        workspace->d_sorted_xq, static_cast<size_t>(spatial.total_atom_numbers),
        &snapshot.sorted_xq, [](const float4& value) { return ToPOD(value); });
    snapshot.sorted_lj_type =
        CopyDeviceVector(workspace->d_sorted_lj_type,
                         static_cast<size_t>(spatial.total_atom_numbers));
    CopyConvertedDeviceVector(workspace->d_sorted_lj_comb,
                              static_cast<size_t>(spatial.total_atom_numbers),
                              &snapshot.sorted_lj_comb,
                              [](const float2& value) { return ToPOD(value); });
    CopyConvertedDeviceVector(
        lj.d_LJ_AB_packed, static_cast<size_t>(lj.pair_type_numbers),
        &snapshot.lj_ab, [](const float2& value) { return ToPOD(value); });
    PopulatePairOracleMetadata(layout, &snapshot);
    return snapshot;
}

void RefreshRegularLJPayload()
{
    const size_t atom_count =
        static_cast<size_t>(dd.atom_numbers + dd.ghost_numbers);
    DeviceBuffer<VECTOR> force(atom_count);
    md_info.MD_Reset_Atom_Energy_And_Virial_And_Force();
    pm.Get_Atoms(&controller, md_info.crd, md_info.d_charge, dd.atom_numbers,
                 dd.crd, dd.d_charge, dd.atom_local, false, false, true, false);
    dd.Reset_Force_and_Virial(&md_info);
    dd.Update_Ghost(&controller);
    lj.LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(
        md_info.atom_numbers, dd.atom_numbers, dd.ghost_numbers, dd.crd,
        dd.d_charge, force.data(), md_info.pbc.cell, md_info.pbc.rcell, pm.beta,
        0, nullptr, 0, nullptr, nullptr);
}

void WriteForceOnlySnapshot(const std::filesystem::path& path,
                            const SpongeGmxpackedForceOnlySnapshot& snapshot)
{
    if (!WriteSpongeGmxpackedForceOnlySnapshot(path.string(), snapshot))
    {
        throw std::runtime_error("failed to write snapshot: " + path.string());
    }
}

void WriteFullOutputSnapshot(const std::filesystem::path& path)
{
    const ClusteredNeighborProvider* provider = lj.clustered_neighbor_provider;
    if (provider == nullptr ||
        provider->TotalAtomNumbers() != md_info.atom_numbers)
    {
        throw std::runtime_error(
            "full-output snapshot producer currently requires one PP rank");
    }
    const size_t atom_count = static_cast<size_t>(md_info.atom_numbers);
    DeviceBuffer<VECTOR> force(atom_count);
    DeviceBuffer<float> atom_energy(atom_count);
    DeviceBuffer<LTMatrix3> atom_virial(atom_count);
    DeviceBuffer<float> direct_energy(atom_count);

    lj.LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(
        md_info.atom_numbers, dd.atom_numbers, dd.ghost_numbers, dd.crd,
        dd.d_charge, force.data(), md_info.pbc.cell, md_info.pbc.rcell, pm.beta,
        1, atom_energy.data(), 1, atom_virial.data(), direct_energy.data());

    SpongeGmxpackedFullOutputSnapshot snapshot = {};
    snapshot.header.file =
        MakeFileHeader(SnapshotKind::spongeGmxpackedFullOutput);
    snapshot.header.compute_energy = 1u;
    snapshot.header.compute_virial = 1u;
    snapshot.header.force_soa = 0u;
    snapshot.header.total_output = 0u;
    snapshot.payload = CaptureForceOnlySnapshot();

    const auto host_force = CopyDeviceVector(force.data(), atom_count);
    snapshot.reference_force.reserve(atom_count);
    for (const VECTOR& value : host_force)
    {
        snapshot.reference_force.push_back(ToPOD(value));
    }
    snapshot.reference_atom_energy =
        CopyDeviceVector(atom_energy.data(), atom_count);
    snapshot.reference_direct_cf_energy =
        CopyDeviceVector(direct_energy.data(), atom_count);
    snapshot.reference_lj_energy =
        CopyDeviceVector(lj.d_LJ_energy_atom, atom_count);
    const auto host_virial = CopyDeviceVector(atom_virial.data(), atom_count);
    snapshot.reference_atom_virial.reserve(atom_count);
    for (const LTMatrix3& value : host_virial)
    {
        snapshot.reference_atom_virial.push_back(ToPOD(value));
    }
    snapshot.header.force_reference_numbers = atom_count;
    snapshot.header.energy_reference_numbers = atom_count;
    snapshot.header.direct_energy_reference_numbers = atom_count;
    snapshot.header.lj_energy_reference_numbers = atom_count;
    snapshot.header.virial_reference_numbers = atom_count;

    if (!WriteSpongeGmxpackedFullOutputSnapshot(path.string(), snapshot))
    {
        throw std::runtime_error("failed to write snapshot: " + path.string());
    }
}

}  // namespace

void WriteCurrentClusteredLJSnapshots(const std::string& prefix,
                                      bool write_full_output)
{
    const std::filesystem::path prefix_path(prefix);
    if (!prefix_path.parent_path().empty())
    {
        std::filesystem::create_directories(prefix_path.parent_path());
    }
    RefreshRegularLJPayload();
    auto payload = CaptureForceOnlySnapshot();
    WriteForceOnlySnapshot(
        prefix_path.string() + ".sponge_gmxpacked_forceonly.bin", payload);
    if (write_full_output)
    {
        WriteFullOutputSnapshot(prefix_path.string() +
                                ".sponge_fulloutput.bin");
    }
}

}  // namespace nbnxm_microbench
