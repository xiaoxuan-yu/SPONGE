#include "internal.h"

#include "../provider/internal.h"
#include "../provider/lifecycle.h"
#include "../provider/runtime.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include "../../MD_core/MD_core.h"
#include "../../third_party/cornerstone_octree/include/cstone/sfc/box.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/sfc/common.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/sfc/sfc.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/traversal/boxoverlap.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/tree/csarray.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/tree/octree.hpp"

#ifndef USE_CPU
#include "../../third_party/cornerstone_octree/include/cstone/cuda/device_vector.h"
#include "../../third_party/cornerstone_octree/include/cstone/execution.hpp"
#include "../../third_party/cornerstone_octree/include/cstone/primitives/primitives_gpu.h"
#include "../../third_party/cornerstone_octree/include/cstone/tree/octree_gpu.h"
#include "../../third_party/cornerstone_octree/include/cstone/tree/update_gpu.cuh"
#endif

extern MD_INFORMATION md_info;

namespace
{

using CornerstoneKey = uint64_t;
using CornerstoneNodeIndex = cstone::TreeNodeIndex;
constexpr int kClusteredMaxSuperClusterClusters =
    kClusteredSuperClusterClusters;
using clustered_neighbor_runtime::Reserve_Device_Buffer;
using cstone::rawPtr;
#ifndef USE_CPU
using clustered_neighbor_runtime::Bind_Clustered_Working_Device;
#endif

#include "primitives/geometry.cuh"
#include "detail/pair_mask_primitives.inc.cuh"
#include "detail/geometry_kernels.inc.cuh"

#ifndef USE_CPU
static void CheckCudaStatus(cudaError_t error, const char* tag)
{
    if (error == cudaSuccess)
    {
        return;
    }
    int current_device = -1;
    cudaGetDevice(&current_device);
    fprintf(stderr, "clustered cuda failure: tag=%s device=%d error=%s\n", tag,
            current_device, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
}

static void Sort_Cornerstone_Keys_On_Device(ClusteredNeighborProvider* layout,
                                            int count, uint64_t* d_keys,
                                            int* d_values)
{
    if (count <= 1 || d_keys == NULL || d_values == NULL || layout == NULL)
    {
        return;
    }
    Bind_Clustered_Working_Device(&ClusteredNeighborProviderInternal::WorkingDevice(layout));
    (void)cudaGetLastError();
    const uint64_t temp_storage_bytes =
        cstone::sortByKeyTempStorage<uint64_t, int>(
            static_cast<uint64_t>(count));
    clustered_neighbor_runtime::Reserve_Raw_Device_Workspace(
        sizeof(uint64_t) * static_cast<size_t>(count),
        &ClusteredNeighborProviderInternal::Workspace(layout).sort_keys);
    clustered_neighbor_runtime::Reserve_Raw_Device_Workspace(
        sizeof(int) * static_cast<size_t>(count),
        &ClusteredNeighborProviderInternal::Workspace(layout).sort_values);
    clustered_neighbor_runtime::Reserve_Raw_Device_Workspace(temp_storage_bytes,
                                                       &ClusteredNeighborProviderInternal::Workspace(layout).sort);

    cstone::sortByKey<uint64_t, int>(
        cstone::execution::gpuDefaultStream, d_keys, d_keys + count, d_values,
        reinterpret_cast<uint64_t*>(ClusteredNeighborProviderInternal::Workspace(layout).sort_keys.data),
        reinterpret_cast<int*>(ClusteredNeighborProviderInternal::Workspace(layout).sort_values.data),
        ClusteredNeighborProviderInternal::Workspace(layout).sort.data, temp_storage_bytes);

    const cudaError_t post_sort_err = cudaGetLastError();
    const cudaError_t sync_err = cudaDeviceSynchronize();
    CheckCudaStatus(post_sort_err,
                                "clustered-cstone-sort-post-launch");
    CheckCudaStatus(sync_err, "clustered-cstone-sort-sync");
}
#endif

static bool Prepare_Atom_To_Molecule_Metadata(ClusteredNeighborProvider* layout)
{
    if (layout == NULL || ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers <= 0 ||
        md_info.atom_numbers <= 0 || md_info.nb.h_excluded_list_start == NULL ||
        md_info.nb.h_excluded_numbers == NULL ||
        (md_info.nb.excluded_atom_numbers > 0 &&
         md_info.nb.h_excluded_list == NULL))
    {
        return false;
    }

    // Exclusions, rather than residue/molecule input, are the authoritative
    // relation for pruning exclusion-mask work. Residue metadata can be absent
    // or intentionally finer than the exclusion graph, so using it here can
    // incorrectly discard real excluded pairs.
    std::vector<int> parent((size_t)md_info.atom_numbers);
    for (int atom = 0; atom < md_info.atom_numbers; atom += 1)
    {
        parent[(size_t)atom] = atom;
    }
    const auto find_root = [&parent](int atom)
    {
        int root = atom;
        while (parent[(size_t)root] != root)
        {
            root = parent[(size_t)root];
        }
        while (parent[(size_t)atom] != atom)
        {
            const int next = parent[(size_t)atom];
            parent[(size_t)atom] = root;
            atom = next;
        }
        return root;
    };
    for (int atom_i = 0; atom_i < md_info.atom_numbers; atom_i += 1)
    {
        const int exclude_start = md_info.nb.h_excluded_list_start[atom_i];
        const int exclude_count = md_info.nb.h_excluded_numbers[atom_i];
        for (int exclude_offset = 0; exclude_offset < exclude_count;
             exclude_offset += 1)
        {
            const int atom_j =
                md_info.nb.h_excluded_list[exclude_start + exclude_offset];
            if (atom_j < 0 || atom_j >= md_info.atom_numbers)
            {
                continue;
            }
            const int root_i = find_root(atom_i);
            const int root_j = find_root(atom_j);
            if (root_i != root_j)
            {
                parent[(size_t)root_j] = root_i;
            }
        }
    }
    std::vector<int> host_global_atom_to_molecule((size_t)md_info.atom_numbers);
    for (int atom = 0; atom < md_info.atom_numbers; atom += 1)
    {
        host_global_atom_to_molecule[(size_t)atom] = find_root(atom);
    }

    Reserve_Device_Buffer(md_info.atom_numbers,
                          &ClusteredNeighborProviderInternal::Spatial(layout).molecules.global_atom_to_molecule);
    deviceMemcpy(ClusteredNeighborProviderInternal::Spatial(layout).molecules.global_atom_to_molecule.data,
                 host_global_atom_to_molecule.data(),
                 sizeof(int) * md_info.atom_numbers, deviceMemcpyHostToDevice);

    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers,
                          &ClusteredNeighborProviderInternal::Spatial(layout).molecules.local_atom_to_molecule);
    Launch_Device_Kernel(Build_Local_Atom_To_Molecule_Map,
                         (ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers,
                         md_info.atom_numbers, ClusteredNeighborProviderInternal::Domain(layout).atom_local,
                         ClusteredNeighborProviderInternal::Spatial(layout).molecules.global_atom_to_molecule.data,
                         ClusteredNeighborProviderInternal::Spatial(layout).molecules.local_atom_to_molecule.data);
    return true;
}

static void Reset_Cornerstone_Root(ClusteredTreeState* state,
                                   int total_atom_numbers)
{
    std::vector<CornerstoneKey> root_leaves = {
        0ull, cstone::nodeRange<CornerstoneKey>(0)};
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

static void Build_Cornerstone_Tree(ClusteredNeighborProvider* layout)
{
    if (ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers <= 0)
    {
        return;
    }

    if (ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state != NULL)
    {
        delete ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state;
        ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state = NULL;
    }
    ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state = new ClusteredTreeState();
    auto* state = ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state;

    Reset_Cornerstone_Root(state, ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers);

#ifndef USE_CPU
    const auto* sorted_keys = ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys.data;
    const std::span<const CornerstoneKey> key_span(
        sorted_keys, static_cast<size_t>(ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers));
    bool converged = false;
    for (int iter = 0; iter < 64 && !converged; iter += 1)
    {
        converged = cstone::updateOctreeGpu<CornerstoneKey>(
            cstone::execution::gpuDefaultStream, key_span,
            static_cast<unsigned>(ClusteredNeighborProviderInternal::Config(layout).cornerstone_leaf_size),
            state->leaves, state->leaf_counts, state->tmp_leaves,
            state->work_array);
    }
    state->octree.resize(
        static_cast<CornerstoneNodeIndex>(cstone::nNodes(state->leaves)));
    cstone::buildOctreeGpu(cstone::execution::gpuDefaultStream,
                           rawPtr(state->leaves), state->octree.data());
#else
    std::vector<CornerstoneKey> host_keys(
        static_cast<size_t>(ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers));
    deviceMemcpy(host_keys.data(), ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys.data,
                 sizeof(CornerstoneKey) * ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers,
                 deviceMemcpyDeviceToHost);
    const std::span<const CornerstoneKey> key_span(host_keys.data(),
                                                   host_keys.size());
    bool converged = false;
    for (int iter = 0; iter < 64 && !converged; iter += 1)
    {
        converged = cstone::updateOctree<CornerstoneKey>(
            key_span,
            static_cast<unsigned>(ClusteredNeighborProviderInternal::Config(layout).cornerstone_leaf_size),
            state->leaves, state->leaf_counts);
    }
    state->octree.resize(
        static_cast<CornerstoneNodeIndex>(cstone::nNodes(state->leaves)));
    cstone::updateInternalTree<CornerstoneKey>(
        std::span<const CornerstoneKey>(state->leaves.data(),
                                        state->leaves.size()),
        state->octree.data());
#endif
}

}  // namespace

namespace clustered_neighbor_builder_internal
{

BuildPreparationResult PrepareBuildState(ClusteredNeighborProvider* provider,
                                         LTMatrix3 reciprocal_cell,
                                         float cutoff)
{
    const float minimum_box_face_height =
        clustered_neighbor_runtime::Clustered_Minimum_Box_Face_Height(
            reciprocal_cell);
    auto& rebuild_skin =
        ClusteredNeighborProviderInternal::EffectiveRebuildSkin(provider);
    if (minimum_box_face_height > 0.0f &&
        cutoff + rebuild_skin >= 0.5f * minimum_box_face_height)
    {
        // Dynamic image dedup makes the current payload unique, but a wide
        // small-box outer horizon is not yet stable across rebuilds. Keep the
        // production horizon at the independently qualified global skin.
        const float safe_skin =
            fmaxf(0.0f, fminf(rebuild_skin, md_info.nb.skin));
        if (safe_skin < rebuild_skin)
        {
            if (ClusteredNeighborProviderInternal::Controller(provider) != NULL)
            {
                ClusteredNeighborProviderInternal::Controller(provider)->printf(
                    "    clustered rebuild skin reduced from %.2f to %.2f "
                    "for a %.2f Angstrom minimum box height\n",
                    rebuild_skin, safe_skin, minimum_box_face_height);
            }
            rebuild_skin = safe_skin;
            ClusteredNeighborProviderInternal::RebuildDirty(provider) = true;
            ClusteredNeighborProviderInternal::Spatial(provider).ready = false;
            Invalidate_Gmxpacked_Incremental_Source_Cache_State(provider);
        }
    }
#ifndef USE_CPU
    if (ClusteredNeighborProviderInternal::Controller(provider) != NULL)
    {
        ClusteredNeighborProviderInternal::WorkingDevice(provider) =
            ClusteredNeighborProviderInternal::Controller(provider)
                ->working_device;
    }
    clustered_neighbor_runtime::Bind_Clustered_Working_Device(
        &ClusteredNeighborProviderInternal::WorkingDevice(provider));
#endif
    auto& spatial = ClusteredNeighborProviderInternal::Spatial(provider);
    const auto& domain = ClusteredNeighborProviderInternal::Domain(provider);
    spatial.total_atom_numbers =
        domain.local_atom_count + domain.ghost_atom_count;
    spatial.padded_total_atom_numbers = spatial.total_atom_numbers;
    if (spatial.total_atom_numbers > 0)
    {
        return BuildPreparationResult::ready;
    }

    spatial.cluster_numbers = 0;
    spatial.padded_total_atom_numbers = 0;
    spatial.super_cluster_numbers = 0;
    spatial.local_cluster_numbers = 0;
    spatial.leaf_numbers = 0;
    spatial.candidates.candidate_sci_numbers = 0;
    spatial.candidates.sci_numbers = 0;
    Reset_Gmxpacked_Payload(provider);
    spatial.candidates.leaf_numbers = 0;
    spatial.ready = false;
    ClusteredNeighborProviderInternal::CachedCutoff(provider) = -1.0f;
    Invalidate_Gmxpacked_Incremental_Source_Cache_State(provider);
    return BuildPreparationResult::empty;
}

SpatialGeometryBuildResult BuildSpatialGeometry(
    ClusteredNeighborProvider* layout, const VECTOR* crd,
    const VECTOR* builder_geometry_crd, LTMatrix3 cell, LTMatrix3 rcell,
    float build_cutoff)
{
    const bool has_atom_to_molecule = Prepare_Atom_To_Molecule_Metadata(layout);
    const int* atom_to_molecule =
        has_atom_to_molecule
            ? ClusteredNeighborProviderInternal::Spatial(layout).molecules.local_atom_to_molecule.data
            : NULL;

    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers,
                          &ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys);
    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers,
                          &ClusteredNeighborProviderInternal::Spatial(layout).ordering.sort_permutation);

    Launch_Device_Kernel(Build_Cornerstone_Sort_Keys,
                         (ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers, crd, rcell,
                         ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys.data,
                         ClusteredNeighborProviderInternal::Spatial(layout).ordering.sort_permutation.data);

#ifndef USE_CPU
    Sort_Cornerstone_Keys_On_Device(
        layout, ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers,
        ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys.data,
        ClusteredNeighborProviderInternal::Spatial(layout).ordering.sort_permutation.data);
#else
    std::vector<uint64_t> h_keys(
        static_cast<size_t>(ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers));
    std::vector<int> h_perm(
        static_cast<size_t>(ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers));
    deviceMemcpy(h_keys.data(), ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys.data,
                 sizeof(uint64_t) * ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_perm.data(), ClusteredNeighborProviderInternal::Spatial(layout).ordering.sort_permutation.data,
                 sizeof(int) * ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers,
                 deviceMemcpyDeviceToHost);
    std::vector<int> order(
        static_cast<size_t>(ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers));
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
    std::vector<uint64_t> h_keys_sorted(
        static_cast<size_t>(ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers));
    std::vector<int> h_perm_sorted(
        static_cast<size_t>(ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers));
    for (int i = 0; i < ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers; i += 1)
    {
        h_keys_sorted[static_cast<size_t>(i)] =
            h_keys[static_cast<size_t>(order[static_cast<size_t>(i)])];
        h_perm_sorted[static_cast<size_t>(i)] =
            h_perm[static_cast<size_t>(order[static_cast<size_t>(i)])];
    }
    deviceMemcpy(ClusteredNeighborProviderInternal::Workspace(layout).stable_sort_keys.data, h_keys_sorted.data(),
                 sizeof(uint64_t) * ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(ClusteredNeighborProviderInternal::Spatial(layout).ordering.sort_permutation.data,
                 h_perm_sorted.data(),
                 sizeof(int) * ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers,
                 deviceMemcpyHostToDevice);
#endif

    Build_Cornerstone_Tree(layout);
    const int leaf_numbers =
        ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->octree.numLeafNodes;
    ClusteredNeighborProviderInternal::Spatial(layout).leaf_numbers = leaf_numbers;
    if (leaf_numbers <= 0)
    {
        ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers = 0;
        ClusteredNeighborProviderInternal::Spatial(layout).padded_total_atom_numbers = 0;
        ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers = 0;
        return {};
    }

    Reserve_Device_Buffer(leaf_numbers + 1,
                          &ClusteredNeighborProviderInternal::Spatial(layout).leaves.atom_offsets);
    Launch_Device_Kernel(
        Copy_UInt_To_Int,
        (leaf_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, leaf_numbers,
        cstone::rawPtr(ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->leaf_counts),
        ClusteredNeighborProviderInternal::Spatial(layout).leaves.atom_offsets.data);
    ExclusiveScanCounts(layout, leaf_numbers,
                          ClusteredNeighborProviderInternal::Spatial(layout).leaves.atom_offsets.data,
                          ClusteredNeighborProviderInternal::Spatial(layout).leaves.atom_offsets.data);

    int leaf_atom_total = 0;
    deviceMemcpy(&leaf_atom_total,
                 ClusteredNeighborProviderInternal::Spatial(layout).leaves.atom_offsets.data + leaf_numbers,
                 sizeof(int), deviceMemcpyDeviceToHost);
    const int raw_cluster_numbers =
        IntMax(0, (leaf_atom_total + ClusteredNeighborProviderInternal::Config(layout).cluster_size - 1) /
                      ClusteredNeighborProviderInternal::Config(layout).cluster_size);
    ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers = raw_cluster_numbers;
    ClusteredNeighborProviderInternal::Spatial(layout).padded_total_atom_numbers =
        ClusteredNeighborProviderInternal::Spatial(layout).total_atom_numbers;
    if (ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers <= 0)
    {
        ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers = 0;
        ClusteredNeighborProviderInternal::Spatial(layout).padded_total_atom_numbers = 0;
        return {};
    }

    Reserve_Device_Buffer(leaf_numbers, &ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_starts);
    Reserve_Device_Buffer(leaf_numbers, &ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_ends);
    Launch_Device_Kernel(Build_Leaf_Cluster_Ranges,
                         (leaf_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, leaf_numbers,
                         ClusteredNeighborProviderInternal::Config(layout).cluster_size,
                         ClusteredNeighborProviderInternal::Spatial(layout).leaves.atom_offsets.data,
                         ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_starts.data,
                         ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_ends.data);
    // Piggyback the periodic-image extent bound on the existing leaf-span
    // reduction and host copy, avoiding another kernel or synchronization.
    {
        Reserve_Device_Buffer(2, &ClusteredNeighborProviderInternal::Workspace(layout).leaf_cluster_span_max);
        int zero[2] = {0, 0};
        deviceMemcpy(ClusteredNeighborProviderInternal::Workspace(layout).leaf_cluster_span_max.data, zero,
                     sizeof(zero), deviceMemcpyHostToDevice);
        Launch_Device_Kernel(
            Reduce_Max_Leaf_Cluster_Span,
            (leaf_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, leaf_numbers,
            ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_starts.data,
            ClusteredNeighborProviderInternal::Spatial(layout).leaves.cluster_ends.data,
            cstone::rawPtr(ClusteredNeighborProviderInternal::Spatial(layout).tree.cornerstone_state->leaves),
            ClusteredNeighborProviderInternal::Workspace(layout).leaf_cluster_span_max.data,
            ClusteredNeighborProviderInternal::Workspace(layout).leaf_cluster_span_max.data + 1);
        int reduction_results[2] = {0, 0};
        deviceMemcpy(reduction_results,
                     ClusteredNeighborProviderInternal::Workspace(layout).leaf_cluster_span_max.data,
                     sizeof(reduction_results), deviceMemcpyDeviceToHost);
        ClusteredNeighborProviderInternal::Spatial(layout).leaves.max_cluster_span = reduction_results[0];
        if (ClusteredNeighborProviderInternal::Spatial(layout).leaves.max_cluster_span < 1)
        {
            ClusteredNeighborProviderInternal::Spatial(layout).leaves.max_cluster_span = 1;
        }
        union
        {
            int bits;
            float value;
        } max_fractional_extent = {reduction_results[1]};
        ClusteredNeighborProviderInternal::Spatial(layout).leaves.periodic_image_max_fractional_extent_bound =
            fmaxf(0.0f, max_fractional_extent.value);
    }

    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers + 1,
                          &ClusteredNeighborProviderInternal::Spatial(layout).clusters.offsets);
    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers,
                          &ClusteredNeighborProviderInternal::Spatial(layout).clusters.valid_masks);
    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers,
                          &ClusteredNeighborProviderInternal::Spatial(layout).clusters.local_masks);
    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers,
                          &ClusteredNeighborProviderInternal::Spatial(layout).clusters.centers);
    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers,
                          &ClusteredNeighborProviderInternal::Spatial(layout).clusters.extents);
    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers,
                          &ClusteredNeighborProviderInternal::Spatial(layout).clusters.fractional_centers);
    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers,
                          &ClusteredNeighborProviderInternal::Spatial(layout).clusters.fractional_extents);
    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers,
                          &ClusteredNeighborProviderInternal::Spatial(layout).clusters.radii);

    Launch_Device_Kernel(
        Build_Global_Cluster_Metadata,
        (ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers,
        leaf_atom_total, ClusteredNeighborProviderInternal::Domain(layout).direct_local_atom_count,
        ClusteredNeighborProviderInternal::Config(layout).cluster_size,
        ClusteredNeighborProviderInternal::Spatial(layout).ordering.sort_permutation.data, builder_geometry_crd,
        cell, rcell, ClusteredNeighborProviderInternal::Spatial(layout).clusters.offsets.data,
        ClusteredNeighborProviderInternal::Spatial(layout).clusters.valid_masks.data,
        ClusteredNeighborProviderInternal::Spatial(layout).clusters.local_masks.data,
        ClusteredNeighborProviderInternal::Spatial(layout).clusters.centers.data,
        ClusteredNeighborProviderInternal::Spatial(layout).clusters.extents.data,
        ClusteredNeighborProviderInternal::Spatial(layout).clusters.fractional_centers.data,
        ClusteredNeighborProviderInternal::Spatial(layout).clusters.fractional_extents.data,
        ClusteredNeighborProviderInternal::Spatial(layout).clusters.radii.data);

    if (atom_to_molecule != NULL)
    {
        Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers,
                              &ClusteredNeighborProviderInternal::Spatial(layout).molecules.cluster_signatures);
        Reserve_Device_Buffer(
            ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers * ClusteredNeighborProviderInternal::Config(layout).cluster_size,
            &ClusteredNeighborProviderInternal::Spatial(layout).molecules.cluster_ids);
        Launch_Device_Kernel(
            Build_Cluster_Molecule_Metadata,
            (ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers + CONTROLLER::device_max_thread -
             1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL,
            ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers, ClusteredNeighborProviderInternal::Config(layout).cluster_size,
            ClusteredNeighborProviderInternal::Spatial(layout).ordering.sort_permutation.data,
            ClusteredNeighborProviderInternal::Spatial(layout).clusters.offsets.data,
            ClusteredNeighborProviderInternal::Spatial(layout).clusters.valid_masks.data, atom_to_molecule,
            ClusteredNeighborProviderInternal::Spatial(layout).molecules.cluster_signatures.data,
            ClusteredNeighborProviderInternal::Spatial(layout).molecules.cluster_ids.data);
    }

    ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers =
        (ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers +
         ClusteredNeighborProviderInternal::Config(layout).clusters_per_supercluster - 1) /
        ClusteredNeighborProviderInternal::Config(layout).clusters_per_supercluster;
    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers + 1,
                          &ClusteredNeighborProviderInternal::Spatial(layout).superclusters.offsets);
    Launch_Device_Kernel(Build_Fixed_Group_Offsets,
                         (ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers + 1 +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers + 1,
                         ClusteredNeighborProviderInternal::Config(layout).clusters_per_supercluster,
                         ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers,
                         ClusteredNeighborProviderInternal::Spatial(layout).superclusters.offsets.data);

    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers,
                          &ClusteredNeighborProviderInternal::Spatial(layout).superclusters.has_local);
    Reserve_Device_Buffer(
        ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers,
        &ClusteredNeighborProviderInternal::Spatial(layout).superclusters.cluster_to_supercluster);
    Launch_Device_Kernel(
        Build_Cluster_To_Supercluster,
        (ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers + CONTROLLER::device_max_thread -
         1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL,
        ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers,
        ClusteredNeighborProviderInternal::Spatial(layout).superclusters.offsets.data,
        ClusteredNeighborProviderInternal::Spatial(layout).superclusters.cluster_to_supercluster.data);
    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers,
                          &ClusteredNeighborProviderInternal::Spatial(layout).superclusters.centers);
    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers,
                          &ClusteredNeighborProviderInternal::Spatial(layout).superclusters.sizes);
    Launch_Device_Kernel(Build_Supercluster_Metadata,
                         (ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers,
                         ClusteredNeighborProviderInternal::Config(layout).clusters_per_supercluster, build_cutoff,
                         rcell, ClusteredNeighborProviderInternal::Spatial(layout).superclusters.offsets.data,
                         ClusteredNeighborProviderInternal::Spatial(layout).clusters.valid_masks.data,
                         ClusteredNeighborProviderInternal::Spatial(layout).clusters.local_masks.data,
                         ClusteredNeighborProviderInternal::Spatial(layout).clusters.centers.data,
                         ClusteredNeighborProviderInternal::Spatial(layout).clusters.extents.data,
                         ClusteredNeighborProviderInternal::Spatial(layout).superclusters.has_local.data,
                         ClusteredNeighborProviderInternal::Spatial(layout).superclusters.centers.data,
                         ClusteredNeighborProviderInternal::Spatial(layout).superclusters.sizes.data);

    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers,
                          &ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_counts);
    Reserve_Device_Buffer(ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers + 1,
                          &ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_offsets);
    Launch_Device_Kernel(Build_Local_Supercluster_Flags,
                         (ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL,
                         ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers,
                         ClusteredNeighborProviderInternal::Spatial(layout).superclusters.has_local.data,
                         ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_counts.data);
    ClusteredNeighborProviderInternal::Spatial(layout).candidates.sci_numbers =
        ExclusiveScanCounts(layout, ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers,
                              ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_counts.data,
                              ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_offsets.data);
    if (ClusteredNeighborProviderInternal::Spatial(layout).candidates.sci_numbers <= 0)
    {
        ClusteredNeighborProviderInternal::Spatial(layout).candidates.leaf_numbers = 0;
        return {};
    }

    return {clustered_neighbor_builder_internal::BuildPreparationResult::ready};
}

}  // namespace clustered_neighbor_builder_internal
