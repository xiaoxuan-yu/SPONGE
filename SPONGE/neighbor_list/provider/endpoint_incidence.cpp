#include "endpoint_incidence.h"

#include <limits>

#include "../builder/internal.h"
#include "../contract/view.h"
#include "internal.h"
#include "runtime.h"

#ifndef USE_CPU
namespace
{

static __global__ void Build_Gmxpacked_Endpoint_Incidence_Records(
    const int sci_numbers, const int cluster_numbers,
    const int super_cluster_numbers, const int* super_cluster_offsets,
    const int* cluster_to_supercluster,
    const CLUSTERED_GMXPACKED_SCI* sci_entries, const int cjpacked_numbers,
    const CLUSTERED_GMXPACKED_CJ* cjpacked_entries, uint64_t* incidence_keys,
    CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE* incidence_references,
    int* reference_counts, int* error_flag)
{
    SIMPLE_DEVICE_FOR(sci_id, sci_numbers)
    {
        const CLUSTERED_GMXPACKED_SCI sci = sci_entries[sci_id];
        if (sci.supercluster_id < 0 ||
            sci.supercluster_id >= super_cluster_numbers ||
            sci.cjpacked_begin < 0 || sci.cjpacked_begin > sci.cjpacked_end ||
            sci.cjpacked_end > cjpacked_numbers)
        {
            atomicExch(error_flag, 1);
            return;
        }
        const int i_cluster_begin = super_cluster_offsets[sci.supercluster_id];
        const int i_cluster_end =
            super_cluster_offsets[sci.supercluster_id + 1];
        const int i_cluster_count = i_cluster_end - i_cluster_begin;
        if (i_cluster_begin < 0 || i_cluster_end > cluster_numbers ||
            i_cluster_count <= 0 ||
            i_cluster_count > kClusteredSuperClusterClusters)
        {
            atomicExch(error_flag, 1);
            return;
        }
        const unsigned int valid_i_cluster_mask =
            (1u << static_cast<unsigned int>(i_cluster_count)) - 1u;
        for (int cjpacked_id = sci.cjpacked_begin;
             cjpacked_id < sci.cjpacked_end; cjpacked_id += 1)
        {
            const CLUSTERED_GMXPACKED_CJ packed =
                cjpacked_entries[cjpacked_id];
            const unsigned int combined_imask =
                packed.split[0].imask | packed.split[1].imask;
            for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
            {
                const unsigned int i_cluster_mask =
                    (combined_imask >>
                     static_cast<unsigned int>(
                         jm * kClusteredSuperClusterClusters)) &
                    ((1u << kClusteredSuperClusterClusters) - 1u);
                if (i_cluster_mask == 0u)
                {
                    continue;
                }
                if ((i_cluster_mask & ~valid_i_cluster_mask) != 0u)
                {
                    atomicExch(error_flag, 1);
                    continue;
                }
                const int cluster_j = packed.cj[jm];
                if (cluster_j < 0 || cluster_j >= cluster_numbers)
                {
                    atomicExch(error_flag, 1);
                    continue;
                }
                const int j_supercluster = cluster_to_supercluster[cluster_j];
                if (j_supercluster < 0 ||
                    j_supercluster >= super_cluster_numbers)
                {
                    atomicExch(error_flag, 1);
                    continue;
                }

                CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE native = {};
                native.sci_id = sci_id;
                native.cjpacked_id = cjpacked_id;
                native.i_cluster_mask = i_cluster_mask;
                native.jm = static_cast<unsigned char>(jm);
                native.orientation = CLUSTERED_ENDPOINT_ORIENTATION::NATIVE_I;
                const int native_slot =
                    (cjpacked_id * kClusteredJGroupSize + jm) * 2;
                const uint64_t native_key =
                    (static_cast<uint64_t>(sci.supercluster_id) << 32) |
                    static_cast<uint32_t>(native_slot);
                const unsigned long long native_previous = atomicCAS(
                    reinterpret_cast<unsigned long long*>(incidence_keys +
                                                          native_slot),
                    ~0ull, static_cast<unsigned long long>(native_key));
                if (native_previous != ~0ull)
                {
                    atomicExch(error_flag, 1);
                    continue;
                }
                incidence_references[native_slot] = native;
                atomicAdd(reference_counts + sci.supercluster_id, 1);

                CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE transposed = native;
                transposed.orientation =
                    CLUSTERED_ENDPOINT_ORIENTATION::TRANSPOSED_J;
                const int transposed_slot = native_slot + 1;
                const uint64_t transposed_key =
                    (static_cast<uint64_t>(j_supercluster) << 32) |
                    static_cast<uint32_t>(transposed_slot);
                const unsigned long long transposed_previous = atomicCAS(
                    reinterpret_cast<unsigned long long*>(incidence_keys +
                                                          transposed_slot),
                    ~0ull, static_cast<unsigned long long>(transposed_key));
                if (transposed_previous != ~0ull)
                {
                    atomicExch(error_flag, 1);
                    continue;
                }
                incidence_references[transposed_slot] = transposed;
                atomicAdd(reference_counts + j_supercluster, 1);
            }
        }
    }
}

}  // namespace
#endif

void Invalidate_Gmxpacked_Endpoint_Incidence(ClusteredNeighborProvider* layout)
{
    if (layout == NULL)
    {
        return;
    }
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_ready = false;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_provider_incarnation = -1;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_payload_generation = -1;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_sci_numbers = 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_cjpacked_numbers = 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_super_cluster_numbers = 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_reference_numbers = 0;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_offset_tail = 0;
}

void Build_Gmxpacked_Endpoint_Incidence_Metadata(ClusteredNeighborProvider* layout)
{
#ifdef USE_CPU
    if (layout == NULL)
    {
        return;
    }
    const bool exact_generation_ready =
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_ready &&
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_provider_incarnation ==
            ClusteredNeighborProviderInternal::ProviderIncarnation(layout) &&
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_payload_generation ==
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_compact_payload_generation &&
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_sci_numbers ==
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers &&
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_cjpacked_numbers ==
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked_numbers &&
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_super_cluster_numbers ==
            ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers &&
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_offset_tail ==
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_reference_numbers;
    if (exact_generation_ready)
    {
        return;
    }
    Invalidate_Gmxpacked_Endpoint_Incidence(layout);
    CLUSTERED_GMXPACKED_ENDPOINT_INCIDENCE_HOST incidence;
    const char* failure_reason = NULL;
    if (!Clustered_Build_Gmxpacked_Endpoint_Incidence_Host(
            ClusteredNeighborProviderInternal::ProviderIncarnation(layout),
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_compact_payload_generation,
            ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers,
            ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers,
            ClusteredNeighborProviderInternal::Spatial(layout).superclusters.offsets.data,
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers,
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci.data,
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked_numbers,
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked.data, &incidence,
            &failure_reason))
    {
        return;
    }
    const int reference_numbers = static_cast<int>(incidence.references.size());
    if (reference_numbers <= 0 ||
        incidence.offsets.size() !=
            static_cast<size_t>(ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers + 1) ||
        incidence.offsets.back() != reference_numbers)
    {
        return;
    }
    clustered_neighbor_runtime::Reserve_Device_Buffer(
        ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers + 1,
        &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_offsets);
    clustered_neighbor_runtime::Reserve_Device_Buffer(
        reference_numbers,
        &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_references);
    deviceMemcpy(ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_offsets.data,
                 incidence.offsets.data(),
                 sizeof(int) * static_cast<size_t>(
                                   ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers + 1),
                 deviceMemcpyHostToDevice);
    deviceMemcpy(ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_references.data,
                 incidence.references.data(),
                 sizeof(CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE) *
                     static_cast<size_t>(reference_numbers),
                 deviceMemcpyHostToDevice);
#else
    if (layout == NULL)
    {
        return;
    }
    const bool exact_generation_ready =
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_ready &&
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_provider_incarnation ==
            ClusteredNeighborProviderInternal::ProviderIncarnation(layout) &&
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_payload_generation ==
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_compact_payload_generation &&
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_sci_numbers ==
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers &&
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_cjpacked_numbers ==
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked_numbers &&
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_super_cluster_numbers ==
            ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers &&
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_offset_tail ==
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_reference_numbers;
    if (exact_generation_ready)
    {
        return;
    }
    Invalidate_Gmxpacked_Endpoint_Incidence(layout);
    if (ClusteredNeighborProviderInternal::ProviderIncarnation(layout) <= 0 ||
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_compact_payload_generation < 0 ||
        ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers <= 0 ||
        ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers <= 0 ||
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers <= 0 ||
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked_numbers <= 0 ||
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked_numbers >
            std::numeric_limits<int>::max() / (2 * kClusteredJGroupSize) ||
        ClusteredNeighborProviderInternal::Spatial(layout).superclusters.offsets.data == NULL ||
        ClusteredNeighborProviderInternal::Spatial(layout).superclusters.cluster_to_supercluster.data == NULL ||
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci.data == NULL ||
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked.data == NULL)
    {
        return;
    }

    const int max_reference_numbers =
        2 * kClusteredJGroupSize * ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked_numbers;
    clustered_neighbor_runtime::Reserve_Device_Buffer(
        ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers + 1,
        &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_offsets);
    clustered_neighbor_runtime::Reserve_Device_Buffer(
        max_reference_numbers,
        &ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_references);
    clustered_neighbor_runtime::Reserve_Device_Buffer(
        max_reference_numbers, &ClusteredNeighborProviderInternal::Workspace(layout).endpoint_incidence_keys);
    clustered_neighbor_runtime::Reserve_Device_Buffer(
        1, &ClusteredNeighborProviderInternal::Workspace(layout).endpoint_incidence_error);
    clustered_neighbor_runtime::Reserve_Device_Buffer(
        ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers + 1,
        &ClusteredNeighborProviderInternal::Workspace(layout).endpoint_incidence_counts);

    deviceMemset(ClusteredNeighborProviderInternal::Workspace(layout).endpoint_incidence_keys.data, 0xff,
                 sizeof(uint64_t) * static_cast<size_t>(max_reference_numbers));
    deviceMemset(ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_references.data,
                 0xff,
                 sizeof(CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE) *
                     static_cast<size_t>(max_reference_numbers));
    deviceMemset(ClusteredNeighborProviderInternal::Workspace(layout).endpoint_incidence_error.data, 0,
                 sizeof(int));
    deviceMemset(ClusteredNeighborProviderInternal::Workspace(layout).endpoint_incidence_counts.data, 0,
                 sizeof(int) * static_cast<size_t>(
                                   ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers + 1));

    Launch_Device_Kernel(
        Build_Gmxpacked_Endpoint_Incidence_Records,
        (ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers +
         CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL,
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers,
        ClusteredNeighborProviderInternal::Spatial(layout).cluster_numbers, ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers,
        ClusteredNeighborProviderInternal::Spatial(layout).superclusters.offsets.data,
        ClusteredNeighborProviderInternal::Spatial(layout).superclusters.cluster_to_supercluster.data,
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci.data,
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked_numbers,
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked.data,
        ClusteredNeighborProviderInternal::Workspace(layout).endpoint_incidence_keys.data,
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_references.data,
        ClusteredNeighborProviderInternal::Workspace(layout).endpoint_incidence_counts.data,
        ClusteredNeighborProviderInternal::Workspace(layout).endpoint_incidence_error.data);

    int build_error = 0;
    deviceMemcpy(&build_error, ClusteredNeighborProviderInternal::Workspace(layout).endpoint_incidence_error.data,
                 sizeof(int), deviceMemcpyDeviceToHost);
    if (build_error != 0)
    {
        return;
    }

    const int reference_numbers =
        clustered_neighbor_builder_internal::ExclusiveScanCounts(
            layout, ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers,
            ClusteredNeighborProviderInternal::Workspace(layout).endpoint_incidence_counts.data,
            ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_offsets.data);
    if (reference_numbers <= 0 || reference_numbers > max_reference_numbers)
    {
        return;
    }
    clustered_neighbor_builder_internal::StableSortU64EndpointReferences(
        layout, max_reference_numbers,
        ClusteredNeighborProviderInternal::Workspace(layout).endpoint_incidence_keys.data,
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_references.data);
#endif

    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_provider_incarnation =
        ClusteredNeighborProviderInternal::ProviderIncarnation(layout);
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_payload_generation =
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_compact_payload_generation;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_sci_numbers =
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_sci_numbers;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_cjpacked_numbers =
        ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_cjpacked_numbers;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_super_cluster_numbers =
        ClusteredNeighborProviderInternal::Spatial(layout).super_cluster_numbers;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_reference_numbers =
        reference_numbers;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_offset_tail =
        reference_numbers;
    ClusteredNeighborProviderInternal::PairList(layout).gmxpacked_endpoint_incidence_ready = true;
}
