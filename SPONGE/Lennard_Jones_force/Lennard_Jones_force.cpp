#include "Lennard_Jones_force.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "../Domain_decomposition/Domain_decomposition.h"
// [DIAGNOSTIC DUMP ONLY] used exclusively by clustered microbench
// snapshot capture / export functions below; no production code path
// depends on nbnxm_microbench_snapshot.h types.
#include "../../tools/nbnxm_microbench/nbnxm_microbench_snapshot.h"
#include "../xponge/load/native/lj.hpp"
#include "../xponge/xponge.h"
// #include "assert.h"

namespace
{

struct OrderedResiduePoint
{
    int residue_index = 0;
    int atom_start = 0;
    int atom_count = 0;
    VECTOR wrapped = {0.0f, 0.0f, 0.0f};
    VECTOR normalized = {0.0f, 0.0f, 0.0f};
    uint64_t point_hilbert = 0;
};

struct CornerstoneLeaf
{
    std::vector<int> residues;
    VECTOR min_bound = {0.0f, 0.0f, 0.0f};
    VECTOR max_bound = {1.0f, 1.0f, 1.0f};
    uint64_t leaf_hilbert = 0;
};

static VECTOR Wrap_To_Box_Fractional(VECTOR crd, LTMatrix3 rcell,
                                     VECTOR box_length)
{
    VECTOR frac = crd * rcell;
    frac.x -= floorf(frac.x);
    frac.y -= floorf(frac.y);
    frac.z -= floorf(frac.z);
    return wiseproduct(frac, box_length);
}

static __host__ __device__ __forceinline__ int
Clustered_Gmxpacked_Get_LJ_Type_MinMax(const int a, const int b)
{
    const int hi = a > b ? a : b;
    const int lo = a > b ? b : a;
    return (hi * (hi + 1) >> 1) + lo;
}

static bool Clustered_Gmxpacked_Lj_Comb_Table_Compatible(
    const float* lj_a, const float* lj_b, int atom_type_numbers)
{
    if (lj_a == NULL || lj_b == NULL || atom_type_numbers <= 0)
    {
        return false;
    }
    constexpr float rel_tol = 1.0e-4f;
    constexpr float abs_tol = 1.0e-4f;
    for (int i = 0; i < atom_type_numbers; i += 1)
    {
        const int self_i = Get_LJ_Type(i, i);
        if (lj_a[self_i] < 0.0f || lj_b[self_i] < 0.0f)
        {
            return false;
        }
        for (int j = 0; j < atom_type_numbers; j += 1)
        {
            const int self_j = Get_LJ_Type(j, j);
            const int pair = Get_LJ_Type(i, j);
            if (lj_a[self_j] < 0.0f || lj_b[self_j] < 0.0f ||
                lj_a[pair] < 0.0f || lj_b[pair] < 0.0f)
            {
                return false;
            }
            const float expected_a =
                sqrtf(fmaxf(lj_a[self_i], 0.0f)) *
                sqrtf(fmaxf(lj_a[self_j], 0.0f));
            const float expected_b =
                sqrtf(fmaxf(lj_b[self_i], 0.0f)) *
                sqrtf(fmaxf(lj_b[self_j], 0.0f));
            const float scale_a = fmaxf(1.0f, fabsf(lj_a[pair]));
            const float scale_b = fmaxf(1.0f, fabsf(lj_b[pair]));
            if (fabsf(expected_a - lj_a[pair]) >
                    abs_tol + rel_tol * scale_a ||
                fabsf(expected_b - lj_b[pair]) >
                    abs_tol + rel_tol * scale_b)
            {
                return false;
            }
        }
    }
    return true;
}

static bool Clustered_Layout_Has_Primary_Gmxpacked_Payload(
    const LJ_CLUSTER_LAYOUT& layout)
{
    return layout.gmxpacked_sci_numbers > 0 &&
           layout.gmxpacked_cjpacked_numbers > 0 &&
           layout.gmxpacked_exclusion_numbers > 0 &&
           layout.d_gmxpacked_sci != NULL &&
           layout.d_gmxpacked_cjpacked != NULL &&
           layout.d_gmxpacked_exclusions != NULL;
}

static const char* Clustered_Microbench_Dump_Prefix()
{
    const char* prefix = std::getenv("SPONGE_CLUSTERED_DUMP_MICROBENCH");
    return (prefix != NULL && prefix[0] != '\0') ? prefix : NULL;
}

template <typename T>
static std::vector<T> Copy_Device_Vector_To_Host(const T* device_ptr,
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

static nbnxm_microbench::LTMatrix3POD To_Microbench_Matrix_POD(
    const LTMatrix3& cell)
{
    return {cell.a11, cell.a21, cell.a22, cell.a31, cell.a32, cell.a33};
}

static nbnxm_microbench::Float4POD To_Microbench_Float4_POD(
    const float4& value)
{
    return {value.x, value.y, value.z, value.w};
}

static nbnxm_microbench::Float2POD To_Microbench_Float2_POD(
    const float2& value)
{
    return {value.x, value.y};
}

static nbnxm_microbench::Float4POD To_Microbench_Force_POD(
    const VECTOR& value)
{
    return {value.x, value.y, value.z, 0.0f};
}

static nbnxm_microbench::SpongeSciPOD To_Microbench_Sci_POD(
    const LJ_CLUSTERED_SCI& sci)
{
    return {sci.supercluster_id, sci.shift_id, sci.cjpacked_begin,
            sci.cjpacked_end};
}

static nbnxm_microbench::SpongeWarpJRecordPOD To_Microbench_Record_POD(
    const LJ_CLUSTERED_WARP_J_RECORD& record)
{
    nbnxm_microbench::SpongeWarpJRecordPOD pod = {};
    pod.cluster_j = record.cluster_j;
    pod.sorted_j_base = record.sorted_j_base;
    pod.pair_shift_index = record.pair_shift_index;
    pod.valid_mask = record.valid_mask;
    pod.imask = record.imask;
    pod.local_mask = record.local_mask;
    pod.j_lane_base = record.j_lane_base;
    std::memcpy(pod.pair_excl, record.pair_excl, sizeof(pod.pair_excl));
    return pod;
}

static nbnxm_microbench::SpongeGmxpackedSciPOD To_Microbench_Gmxpacked_Sci_POD(
    const LJ_CLUSTERED_GMXPACKED_SCI& sci)
{
    return {sci.supercluster_id, sci.shift_id, sci.cjpacked_begin,
            sci.cjpacked_end};
}

static nbnxm_microbench::SpongeGmxpackedCjPOD To_Microbench_Gmxpacked_Cj_POD(
    const LJ_CLUSTERED_GMXPACKED_CJ& cj)
{
    nbnxm_microbench::SpongeGmxpackedCjPOD pod = {};
    std::memcpy(pod.cj, cj.cj, sizeof(pod.cj));
    for (int split = 0; split < kClusteredWarpSplitCount; split += 1)
    {
        pod.split[split].imask = cj.split[split].imask;
        pod.split[split].exclusion_index = cj.split[split].exclusion_index;
    }
    return pod;
}

static nbnxm_microbench::SpongeGmxpackedExclusionPOD
To_Microbench_Gmxpacked_Exclusion_POD(
    const LJ_CLUSTERED_GMXPACKED_EXCLUSION& exclusion)
{
    nbnxm_microbench::SpongeGmxpackedExclusionPOD pod = {};
    std::memcpy(pod.pair, exclusion.pair, sizeof(pod.pair));
    return pod;
}

static void Populate_Microbench_Pair_Oracle_Metadata(
    const LJ_CLUSTER_LAYOUT& layout,
    nbnxm_microbench::SpongeGmxpackedForceOnlySnapshot* snapshot)
{
    if (snapshot == NULL || layout.local_atom_numbers <= 0 ||
        layout.d_atom_local == NULL ||
        layout.d_excluded_list_start == NULL ||
        layout.d_excluded_numbers == NULL)
    {
        return;
    }
    snapshot->atom_local = Copy_Device_Vector_To_Host(
        layout.d_atom_local,
        static_cast<size_t>(layout.total_atom_numbers));
    snapshot->excluded_list_start = Copy_Device_Vector_To_Host(
        layout.d_excluded_list_start,
        static_cast<size_t>(layout.local_atom_numbers));
    snapshot->excluded_numbers = Copy_Device_Vector_To_Host(
        layout.d_excluded_numbers,
        static_cast<size_t>(layout.local_atom_numbers));
    const int total_excluded =
        snapshot->excluded_list_start.empty() ||
                snapshot->excluded_numbers.empty()
            ? 0
            : snapshot->excluded_list_start.back() +
                  snapshot->excluded_numbers.back();
    if (total_excluded > 0 && layout.d_excluded_list != NULL)
    {
        snapshot->excluded_list = Copy_Device_Vector_To_Host(
            layout.d_excluded_list, static_cast<size_t>(total_excluded));
    }
}

static void Maybe_Dump_Clustered_Microbench_Diagnostic_Snapshot(
    const LJ_CLUSTERED_DIRECT_CACHE* clustered_direct_cache,
    const float2* d_LJ_AB_packed, size_t lj_param_numbers, float cutoff,
    float pme_beta, const LTMatrix3& cell, bool use_lj_comb_kernel,
    int lj_type_matrix_stride = 0)
{
    const char* dump_prefix = Clustered_Microbench_Dump_Prefix();
    static bool dumped = false;
    if (dump_prefix == NULL || dumped ||
        clustered_direct_cache == NULL ||
        clustered_direct_cache->layout.total_atom_numbers <= 0)
    {
        return;
    }

    const auto& layout = clustered_direct_cache->layout;
    // The microbench force-only schema is native-shaped. Keep this as a
    // diagnostic compatibility view of the primary compact payload state.
    if (layout.sci_numbers <= 0 || layout.forceonly_warp_record_numbers <= 0 ||
        !Clustered_Layout_Has_Primary_Gmxpacked_Payload(layout) ||
        layout.d_forceonly_warp_record_offsets == NULL ||
        layout.d_forceonly_warp_j_records == NULL ||
        clustered_direct_cache->d_sorted_xq == NULL ||
        clustered_direct_cache->d_sorted_lj_type == NULL ||
        clustered_direct_cache->d_sorted_atom_ids == NULL ||
        d_LJ_AB_packed == NULL)
    {
        return;
    }

    nbnxm_microbench::SpongeForceOnlySnapshot snapshot = {};
    snapshot.header.file = nbnxm_microbench::MakeFileHeader(
        nbnxm_microbench::SnapshotKind::spongeForceOnly);
    snapshot.header.cluster_size = static_cast<uint32_t>(layout.cluster_size);
    snapshot.header.super_cluster_clusters =
        static_cast<uint32_t>(layout.super_cluster_clusters);
    snapshot.header.warp_split_count = kClusteredWarpSplitCount;
    snapshot.header.cluster_numbers =
        static_cast<uint64_t>(layout.cluster_numbers);
    snapshot.header.super_cluster_numbers =
        static_cast<uint64_t>(layout.super_cluster_numbers);
    snapshot.header.sci_numbers = static_cast<uint64_t>(layout.sci_numbers);
    snapshot.header.record_numbers =
        static_cast<uint64_t>(layout.forceonly_warp_record_numbers);
    snapshot.header.pair_shift_word_numbers =
        static_cast<uint64_t>(layout.cjpacked_numbers * kClusteredJGroupSize);
    snapshot.header.total_atom_numbers =
        static_cast<uint64_t>(layout.total_atom_numbers);
    snapshot.header.local_atom_numbers =
        static_cast<uint64_t>(layout.local_atom_numbers);
    snapshot.header.lj_param_numbers =
        static_cast<uint64_t>(lj_param_numbers);
    snapshot.header.cutoff = cutoff;
    snapshot.header.pme_beta = pme_beta;
    snapshot.header.cell = To_Microbench_Matrix_POD(cell);

    snapshot.cluster_offsets =
        Copy_Device_Vector_To_Host(layout.d_cluster_offsets,
                                   static_cast<size_t>(layout.cluster_numbers));
    snapshot.cluster_valid_masks = Copy_Device_Vector_To_Host(
        layout.d_cluster_valid_masks,
        static_cast<size_t>(layout.cluster_numbers));
    snapshot.cluster_local_masks = Copy_Device_Vector_To_Host(
        layout.d_cluster_local_masks,
        static_cast<size_t>(layout.cluster_numbers));
    snapshot.super_cluster_offsets = Copy_Device_Vector_To_Host(
        layout.d_super_cluster_offsets,
        static_cast<size_t>(layout.super_cluster_numbers + 1));

    const auto host_sci = Copy_Device_Vector_To_Host(
        layout.d_nbnxm_sci, static_cast<size_t>(layout.sci_numbers));
    snapshot.sci.reserve(host_sci.size());
    for (const LJ_CLUSTERED_SCI& sci : host_sci)
    {
        snapshot.sci.push_back(To_Microbench_Sci_POD(sci));
    }

    snapshot.record_offsets = Copy_Device_Vector_To_Host(
        layout.d_forceonly_warp_record_offsets,
        static_cast<size_t>(layout.sci_numbers + 1));
    const auto host_records = Copy_Device_Vector_To_Host(
        layout.d_forceonly_warp_j_records,
        static_cast<size_t>(layout.forceonly_warp_record_numbers));
    snapshot.records.reserve(host_records.size());
    for (const LJ_CLUSTERED_WARP_J_RECORD& record : host_records)
    {
        snapshot.records.push_back(To_Microbench_Record_POD(record));
    }

    snapshot.pair_shift_bits = Copy_Device_Vector_To_Host(
        layout.d_pair_shift_bits,
        static_cast<size_t>(layout.cjpacked_numbers * kClusteredJGroupSize));
    snapshot.sorted_atom_ids = Copy_Device_Vector_To_Host(
        clustered_direct_cache->d_sorted_atom_ids,
        static_cast<size_t>(layout.total_atom_numbers));
    const auto host_sorted_xq = Copy_Device_Vector_To_Host(
        clustered_direct_cache->d_sorted_xq,
        static_cast<size_t>(layout.total_atom_numbers));
    snapshot.sorted_xq.reserve(host_sorted_xq.size());
    for (const float4& value : host_sorted_xq)
    {
        snapshot.sorted_xq.push_back(To_Microbench_Float4_POD(value));
    }
    snapshot.sorted_lj_type = Copy_Device_Vector_To_Host(
        clustered_direct_cache->d_sorted_lj_type,
        static_cast<size_t>(layout.total_atom_numbers));
    const auto host_lj_ab =
        Copy_Device_Vector_To_Host(d_LJ_AB_packed, lj_param_numbers);
    snapshot.lj_ab.reserve(host_lj_ab.size());
    for (const float2& value : host_lj_ab)
    {
        snapshot.lj_ab.push_back(To_Microbench_Float2_POD(value));
    }
    const fs::path prefix_path(dump_prefix);
    if (!prefix_path.parent_path().empty())
    {
        fs::create_directories(prefix_path.parent_path());
    }
    const fs::path snapshot_path =
        prefix_path.string() + ".sponge_forceonly.bin";
    if (!nbnxm_microbench::WriteSpongeForceOnlySnapshot(
            snapshot_path.string(), snapshot))
    {
        fprintf(stderr,
                "[clustered microbench dump] failed to write %s\n",
                snapshot_path.string().c_str());
        return;
    }
    fprintf(stderr,
            "[clustered microbench dump] wrote %s sci=%d records=%d atoms=%d\n",
            snapshot_path.string().c_str(), layout.sci_numbers,
            layout.forceonly_warp_record_numbers, layout.total_atom_numbers);

    if (layout.gmxpacked_sci_numbers > 0 &&
        layout.gmxpacked_cjpacked_numbers > 0 &&
        layout.gmxpacked_exclusion_numbers > 0 &&
        layout.d_gmxpacked_sci != NULL &&
        layout.d_gmxpacked_cjpacked != NULL &&
        layout.d_gmxpacked_exclusions != NULL &&
        layout.d_gmxpacked_pair_shift_sci_safe_flags != NULL &&
        layout.d_pair_shift_bits != NULL &&
        clustered_direct_cache->d_sorted_lj_comb != NULL)
    {
        nbnxm_microbench::SpongeGmxpackedForceOnlySnapshot gmxpacked_snapshot =
            {};
        gmxpacked_snapshot.header.file =
            nbnxm_microbench::MakeFileHeader(
                nbnxm_microbench::SnapshotKind::spongeGmxpackedForceOnly);
        gmxpacked_snapshot.header.cluster_size =
            static_cast<uint32_t>(layout.cluster_size);
        gmxpacked_snapshot.header.super_cluster_clusters =
            static_cast<uint32_t>(layout.super_cluster_clusters);
        gmxpacked_snapshot.header.warp_split_count =
            kClusteredWarpSplitCount;
        gmxpacked_snapshot.header.j_group_size = kClusteredJGroupSize;
        gmxpacked_snapshot.header.force_storage_sorted = 1u;
        gmxpacked_snapshot.header.use_lj_comb =
            use_lj_comb_kernel ? 1u : 0u;
        gmxpacked_snapshot.header.lj_type_matrix_stride =
            static_cast<uint32_t>(lj_type_matrix_stride > 0
                                      ? lj_type_matrix_stride
                                      : 0);
        gmxpacked_snapshot.header.cluster_numbers =
            static_cast<uint64_t>(layout.cluster_numbers);
        gmxpacked_snapshot.header.super_cluster_numbers =
            static_cast<uint64_t>(layout.super_cluster_numbers);
        gmxpacked_snapshot.header.sci_numbers =
            static_cast<uint64_t>(layout.gmxpacked_sci_numbers);
        gmxpacked_snapshot.header.cjpacked_numbers =
            static_cast<uint64_t>(layout.gmxpacked_cjpacked_numbers);
        gmxpacked_snapshot.header.excl_numbers =
            static_cast<uint64_t>(layout.gmxpacked_exclusion_numbers);
        gmxpacked_snapshot.header.pair_shift_word_numbers =
            static_cast<uint64_t>(layout.gmxpacked_cjpacked_numbers *
                                  kClusteredJGroupSize);
        gmxpacked_snapshot.header.total_atom_numbers =
            static_cast<uint64_t>(layout.total_atom_numbers);
        gmxpacked_snapshot.header.local_atom_numbers =
            static_cast<uint64_t>(layout.local_atom_numbers);
        gmxpacked_snapshot.header.lj_param_numbers =
            static_cast<uint64_t>(lj_param_numbers);
        gmxpacked_snapshot.header.cutoff = cutoff;
        gmxpacked_snapshot.header.pme_beta = pme_beta;
        gmxpacked_snapshot.header.cell = To_Microbench_Matrix_POD(cell);

        gmxpacked_snapshot.cluster_offsets =
            Copy_Device_Vector_To_Host(
                layout.d_cluster_offsets,
                static_cast<size_t>(layout.cluster_numbers));
        gmxpacked_snapshot.cluster_valid_masks = Copy_Device_Vector_To_Host(
            layout.d_cluster_valid_masks,
            static_cast<size_t>(layout.cluster_numbers));
        gmxpacked_snapshot.cluster_local_masks = Copy_Device_Vector_To_Host(
            layout.d_cluster_local_masks,
            static_cast<size_t>(layout.cluster_numbers));
        gmxpacked_snapshot.super_cluster_offsets =
            Copy_Device_Vector_To_Host(
                layout.d_super_cluster_offsets,
                static_cast<size_t>(layout.super_cluster_numbers + 1));

        const auto host_gmxpacked_sci = Copy_Device_Vector_To_Host(
            layout.d_gmxpacked_sci,
            static_cast<size_t>(layout.gmxpacked_sci_numbers));
        gmxpacked_snapshot.sci.reserve(host_gmxpacked_sci.size());
        for (const LJ_CLUSTERED_GMXPACKED_SCI& sci : host_gmxpacked_sci)
        {
            gmxpacked_snapshot.sci.push_back(
                To_Microbench_Gmxpacked_Sci_POD(sci));
        }

        const auto host_gmxpacked_cj = Copy_Device_Vector_To_Host(
            layout.d_gmxpacked_cjpacked,
            static_cast<size_t>(layout.gmxpacked_cjpacked_numbers));
        gmxpacked_snapshot.cjpacked.reserve(host_gmxpacked_cj.size());
        for (const LJ_CLUSTERED_GMXPACKED_CJ& cj : host_gmxpacked_cj)
        {
            gmxpacked_snapshot.cjpacked.push_back(
                To_Microbench_Gmxpacked_Cj_POD(cj));
        }

        const auto host_gmxpacked_exclusions = Copy_Device_Vector_To_Host(
            layout.d_gmxpacked_exclusions,
            static_cast<size_t>(layout.gmxpacked_exclusion_numbers));
        gmxpacked_snapshot.excl.reserve(host_gmxpacked_exclusions.size());
        for (const LJ_CLUSTERED_GMXPACKED_EXCLUSION& exclusion :
             host_gmxpacked_exclusions)
        {
            gmxpacked_snapshot.excl.push_back(
                To_Microbench_Gmxpacked_Exclusion_POD(exclusion));
        }

        gmxpacked_snapshot.pair_shift_bits = Copy_Device_Vector_To_Host(
            layout.d_pair_shift_bits,
            static_cast<size_t>(layout.gmxpacked_cjpacked_numbers *
                                kClusteredJGroupSize));
        gmxpacked_snapshot.sci_shift_safe_flags =
            Copy_Device_Vector_To_Host(
                layout.d_gmxpacked_pair_shift_sci_safe_flags,
                static_cast<size_t>(layout.gmxpacked_sci_numbers));
        gmxpacked_snapshot.sorted_atom_ids = snapshot.sorted_atom_ids;
        gmxpacked_snapshot.sorted_xq = snapshot.sorted_xq;
        gmxpacked_snapshot.sorted_lj_type = snapshot.sorted_lj_type;
        const auto host_sorted_lj_comb = Copy_Device_Vector_To_Host(
            clustered_direct_cache->d_sorted_lj_comb,
            static_cast<size_t>(layout.total_atom_numbers));
        gmxpacked_snapshot.sorted_lj_comb.reserve(host_sorted_lj_comb.size());
        for (const float2& value : host_sorted_lj_comb)
        {
            gmxpacked_snapshot.sorted_lj_comb.push_back(
                To_Microbench_Float2_POD(value));
        }
        gmxpacked_snapshot.lj_ab = snapshot.lj_ab;
        Populate_Microbench_Pair_Oracle_Metadata(layout,
                                                 &gmxpacked_snapshot);

        const fs::path gmxpacked_snapshot_path =
            prefix_path.string() + ".sponge_gmxpacked_forceonly.bin";
        if (!nbnxm_microbench::WriteSpongeGmxpackedForceOnlySnapshot(
                gmxpacked_snapshot_path.string(), gmxpacked_snapshot))
        {
            fprintf(stderr,
                    "[clustered microbench dump] failed to write %s\n",
                    gmxpacked_snapshot_path.string().c_str());
            return;
        }
        fprintf(stderr,
                "[clustered microbench dump] wrote %s sci=%d cjpacked=%d "
                "excl=%d atoms=%d\n",
                gmxpacked_snapshot_path.string().c_str(),
                layout.gmxpacked_sci_numbers, layout.gmxpacked_cjpacked_numbers,
                layout.gmxpacked_exclusion_numbers, layout.total_atom_numbers);
    }
    dumped = true;
}

static bool Maybe_Dump_Clustered_Gmxpacked_Microbench_Diagnostic_Snapshot(
    const LJ_CLUSTERED_DIRECT_CACHE* clustered_direct_cache,
    const float2* d_LJ_AB_packed, size_t lj_param_numbers, float cutoff,
    float pme_beta, const LTMatrix3& cell, bool use_lj_comb_kernel,
    int lj_type_matrix_stride = 0,
    nbnxm_microbench::SpongeGmxpackedForceOnlySnapshot* captured_snapshot =
        nullptr)
{
    const char* dump_prefix = Clustered_Microbench_Dump_Prefix();
    static bool dumped = false;
    if (dump_prefix == NULL ||
        (dumped && captured_snapshot == nullptr) ||
        clustered_direct_cache == NULL ||
        clustered_direct_cache->layout.total_atom_numbers <= 0)
    {
        return false;
    }
    const auto& layout = clustered_direct_cache->layout;
    if (layout.gmxpacked_sci_numbers <= 0 ||
        layout.gmxpacked_cjpacked_numbers <= 0 ||
        layout.gmxpacked_exclusion_numbers <= 0)
    {
        fprintf(stderr,
                "[clustered microbench dump] gmxpacked payload is empty "
                "(sci=%d cjpacked=%d exclusions=%d)\n",
                layout.gmxpacked_sci_numbers,
                layout.gmxpacked_cjpacked_numbers,
                layout.gmxpacked_exclusion_numbers);
        return false;
    }
    if (layout.d_gmxpacked_sci == NULL ||
        layout.d_gmxpacked_cjpacked == NULL ||
        layout.d_gmxpacked_exclusions == NULL)
    {
        fprintf(stderr,
                "[clustered microbench dump] gmxpacked payload pointers are "
                "not ready\n");
        return false;
    }
    if (layout.d_pair_shift_bits == NULL ||
        layout.d_cluster_offsets == NULL ||
        layout.d_cluster_valid_masks == NULL ||
        layout.d_cluster_local_masks == NULL ||
        layout.d_super_cluster_offsets == NULL)
    {
        fprintf(stderr,
                "[clustered microbench dump] clustered structural metadata "
                "is not ready\n");
        return false;
    }
    if (clustered_direct_cache->d_sorted_atom_ids == NULL ||
        clustered_direct_cache->d_sorted_xq == NULL ||
        clustered_direct_cache->d_sorted_lj_type == NULL ||
        clustered_direct_cache->d_sorted_lj_comb == NULL ||
        d_LJ_AB_packed == NULL)
    {
        fprintf(stderr,
                "[clustered microbench dump] sorted LJ replay fields are "
                "not ready\n");
        return false;
    }

    nbnxm_microbench::SpongeGmxpackedForceOnlySnapshot snapshot = {};
    snapshot.header.file = nbnxm_microbench::MakeFileHeader(
        nbnxm_microbench::SnapshotKind::spongeGmxpackedForceOnly);
    snapshot.header.cluster_size = static_cast<uint32_t>(layout.cluster_size);
    snapshot.header.super_cluster_clusters =
        static_cast<uint32_t>(layout.super_cluster_clusters);
    snapshot.header.warp_split_count = kClusteredWarpSplitCount;
    snapshot.header.j_group_size = kClusteredJGroupSize;
    snapshot.header.force_storage_sorted = 1u;
    snapshot.header.use_lj_comb = use_lj_comb_kernel ? 1u : 0u;
    snapshot.header.lj_type_matrix_stride =
        static_cast<uint32_t>(lj_type_matrix_stride > 0
                                  ? lj_type_matrix_stride
                                  : 0);
    snapshot.header.cluster_numbers =
        static_cast<uint64_t>(layout.cluster_numbers);
    snapshot.header.super_cluster_numbers =
        static_cast<uint64_t>(layout.super_cluster_numbers);
    snapshot.header.sci_numbers =
        static_cast<uint64_t>(layout.gmxpacked_sci_numbers);
    snapshot.header.cjpacked_numbers =
        static_cast<uint64_t>(layout.gmxpacked_cjpacked_numbers);
    snapshot.header.excl_numbers =
        static_cast<uint64_t>(layout.gmxpacked_exclusion_numbers);
    snapshot.header.pair_shift_word_numbers =
        static_cast<uint64_t>(layout.gmxpacked_cjpacked_numbers *
                              kClusteredJGroupSize);
    snapshot.header.total_atom_numbers =
        static_cast<uint64_t>(layout.total_atom_numbers);
    snapshot.header.local_atom_numbers =
        static_cast<uint64_t>(layout.local_atom_numbers);
    snapshot.header.lj_param_numbers = static_cast<uint64_t>(lj_param_numbers);
    snapshot.header.cutoff = cutoff;
    snapshot.header.pme_beta = pme_beta;
    snapshot.header.cell = To_Microbench_Matrix_POD(cell);

    snapshot.cluster_offsets =
        Copy_Device_Vector_To_Host(layout.d_cluster_offsets,
                                   static_cast<size_t>(layout.cluster_numbers));
    snapshot.cluster_valid_masks = Copy_Device_Vector_To_Host(
        layout.d_cluster_valid_masks,
        static_cast<size_t>(layout.cluster_numbers));
    snapshot.cluster_local_masks = Copy_Device_Vector_To_Host(
        layout.d_cluster_local_masks,
        static_cast<size_t>(layout.cluster_numbers));
    snapshot.super_cluster_offsets = Copy_Device_Vector_To_Host(
        layout.d_super_cluster_offsets,
        static_cast<size_t>(layout.super_cluster_numbers + 1));
    const auto host_cluster_centers = Copy_Device_Vector_To_Host(
        layout.d_cluster_centers, static_cast<size_t>(layout.cluster_numbers));
    snapshot.cluster_centers.reserve(host_cluster_centers.size());
    for (const VECTOR& value : host_cluster_centers)
    {
        snapshot.cluster_centers.push_back(To_Microbench_Force_POD(value));
    }
    const auto host_cluster_extents = Copy_Device_Vector_To_Host(
        layout.d_cluster_extents, static_cast<size_t>(layout.cluster_numbers));
    snapshot.cluster_extents.reserve(host_cluster_extents.size());
    for (const VECTOR& value : host_cluster_extents)
    {
        snapshot.cluster_extents.push_back(To_Microbench_Force_POD(value));
    }
    const auto host_super_cluster_centers = Copy_Device_Vector_To_Host(
        layout.d_super_cluster_centers,
        static_cast<size_t>(layout.super_cluster_numbers));
    snapshot.super_cluster_centers.reserve(host_super_cluster_centers.size());
    for (const VECTOR& value : host_super_cluster_centers)
    {
        snapshot.super_cluster_centers.push_back(
            To_Microbench_Force_POD(value));
    }
    const auto host_super_cluster_sizes = Copy_Device_Vector_To_Host(
        layout.d_super_cluster_sizes,
        static_cast<size_t>(layout.super_cluster_numbers));
    snapshot.super_cluster_sizes.reserve(host_super_cluster_sizes.size());
    for (const VECTOR& value : host_super_cluster_sizes)
    {
        snapshot.super_cluster_sizes.push_back(To_Microbench_Force_POD(value));
    }

    const auto host_sci = Copy_Device_Vector_To_Host(
        layout.d_gmxpacked_sci,
        static_cast<size_t>(layout.gmxpacked_sci_numbers));
    snapshot.sci.reserve(host_sci.size());
    for (const LJ_CLUSTERED_GMXPACKED_SCI& sci : host_sci)
    {
        snapshot.sci.push_back(To_Microbench_Gmxpacked_Sci_POD(sci));
    }

    const auto host_cj = Copy_Device_Vector_To_Host(
        layout.d_gmxpacked_cjpacked,
        static_cast<size_t>(layout.gmxpacked_cjpacked_numbers));
    snapshot.cjpacked.reserve(host_cj.size());
    for (const LJ_CLUSTERED_GMXPACKED_CJ& cj : host_cj)
    {
        snapshot.cjpacked.push_back(To_Microbench_Gmxpacked_Cj_POD(cj));
    }

    const auto host_exclusions = Copy_Device_Vector_To_Host(
        layout.d_gmxpacked_exclusions,
        static_cast<size_t>(layout.gmxpacked_exclusion_numbers));
    snapshot.excl.reserve(host_exclusions.size());
    for (const LJ_CLUSTERED_GMXPACKED_EXCLUSION& exclusion : host_exclusions)
    {
        snapshot.excl.push_back(
            To_Microbench_Gmxpacked_Exclusion_POD(exclusion));
    }

    snapshot.pair_shift_bits = Copy_Device_Vector_To_Host(
        layout.d_pair_shift_bits,
        static_cast<size_t>(layout.gmxpacked_cjpacked_numbers *
                            kClusteredJGroupSize));
    if (layout.d_gmxpacked_pair_shift_sci_safe_flags != NULL)
    {
        snapshot.sci_shift_safe_flags = Copy_Device_Vector_To_Host(
            layout.d_gmxpacked_pair_shift_sci_safe_flags,
            static_cast<size_t>(layout.gmxpacked_sci_numbers));
    }
    else
    {
        snapshot.sci_shift_safe_flags.assign(
            static_cast<size_t>(layout.gmxpacked_sci_numbers), 0);
    }
    if (layout.cornerstone_state != NULL &&
        layout.cornerstone_state->octree.numLeafNodes > 0 &&
        layout.cornerstone_state->octree.numNodes > 0 &&
        layout.d_leaf_cluster_starts != NULL &&
        layout.d_leaf_cluster_ends != NULL &&
        layout.d_leaf_all_local != NULL &&
        layout.d_sci_supercluster_ids != NULL &&
        layout.d_sci_candidate_leaf_offsets != NULL)
    {
        const auto& octree = layout.cornerstone_state->octree;
        const size_t leaf_numbers =
            static_cast<size_t>(octree.numLeafNodes);
        const size_t node_numbers = static_cast<size_t>(octree.numNodes);
        const size_t parent_numbers = octree.parents.size();
        const int candidate_sci_numbers =
            layout.gmxpacked_sci_numbers > 0
                ? layout.gmxpacked_sci_numbers
                : layout.candidate_sci_numbers;
        const bool sparse_shift_candidates =
            layout.d_candidate_shift_ids != NULL;
        const int sci_supercluster_id_numbers =
            sparse_shift_candidates
                ? candidate_sci_numbers
                : (candidate_sci_numbers + kClusteredShiftCount - 1) /
                      kClusteredShiftCount;

        snapshot.leaf_cluster_starts = Copy_Device_Vector_To_Host(
            layout.d_leaf_cluster_starts, leaf_numbers);
        snapshot.leaf_cluster_ends = Copy_Device_Vector_To_Host(
            layout.d_leaf_cluster_ends, leaf_numbers);
        snapshot.leaf_all_local = Copy_Device_Vector_To_Host(
            layout.d_leaf_all_local, leaf_numbers);
        snapshot.octree_prefixes = Copy_Device_Vector_To_Host(
            octree.prefixes.data(), node_numbers);
        snapshot.octree_child_offsets = Copy_Device_Vector_To_Host(
            octree.childOffsets.data(), node_numbers);
        snapshot.octree_parents = Copy_Device_Vector_To_Host(
            octree.parents.data(), parent_numbers);
        snapshot.octree_internal_to_leaf = Copy_Device_Vector_To_Host(
            octree.internalToLeaf.data(), node_numbers);
        snapshot.sci_supercluster_ids = Copy_Device_Vector_To_Host(
            layout.d_sci_supercluster_ids,
            static_cast<size_t>(sci_supercluster_id_numbers));
        if (sparse_shift_candidates)
        {
            snapshot.candidate_shift_ids = Copy_Device_Vector_To_Host(
                layout.d_candidate_shift_ids,
                static_cast<size_t>(candidate_sci_numbers));
        }
        snapshot.candidate_leaf_offsets = Copy_Device_Vector_To_Host(
            layout.d_sci_candidate_leaf_offsets,
            static_cast<size_t>(candidate_sci_numbers + 1));
        const int candidate_leaf_numbers =
            snapshot.candidate_leaf_offsets.empty()
                ? 0
                : snapshot.candidate_leaf_offsets.back();
        if (candidate_leaf_numbers > 0 && layout.d_sci_candidate_leaf_ids != NULL)
        {
            snapshot.candidate_leaf_ids = Copy_Device_Vector_To_Host(
                layout.d_sci_candidate_leaf_ids,
                static_cast<size_t>(candidate_leaf_numbers));
        }
        if (candidate_leaf_numbers > 0 &&
            layout.d_sci_candidate_leaf_prev_running_max_ends != NULL)
        {
            snapshot.candidate_leaf_prev_running_max_ends =
                Copy_Device_Vector_To_Host(
                    layout.d_sci_candidate_leaf_prev_running_max_ends,
                    static_cast<size_t>(candidate_leaf_numbers));
        }
    }
    snapshot.sorted_atom_ids = Copy_Device_Vector_To_Host(
        clustered_direct_cache->d_sorted_atom_ids,
        static_cast<size_t>(layout.total_atom_numbers));
    const auto host_sorted_xq = Copy_Device_Vector_To_Host(
        clustered_direct_cache->d_sorted_xq,
        static_cast<size_t>(layout.total_atom_numbers));
    snapshot.sorted_xq.reserve(host_sorted_xq.size());
    for (const float4& value : host_sorted_xq)
    {
        snapshot.sorted_xq.push_back(To_Microbench_Float4_POD(value));
    }
    snapshot.sorted_lj_type = Copy_Device_Vector_To_Host(
        clustered_direct_cache->d_sorted_lj_type,
        static_cast<size_t>(layout.total_atom_numbers));
    const auto host_sorted_lj_comb = Copy_Device_Vector_To_Host(
        clustered_direct_cache->d_sorted_lj_comb,
        static_cast<size_t>(layout.total_atom_numbers));
    snapshot.sorted_lj_comb.reserve(host_sorted_lj_comb.size());
    for (const float2& value : host_sorted_lj_comb)
    {
        snapshot.sorted_lj_comb.push_back(To_Microbench_Float2_POD(value));
    }
    const auto host_lj_ab =
        Copy_Device_Vector_To_Host(d_LJ_AB_packed, lj_param_numbers);
    snapshot.lj_ab.reserve(host_lj_ab.size());
    for (const float2& value : host_lj_ab)
    {
        snapshot.lj_ab.push_back(To_Microbench_Float2_POD(value));
    }
    Populate_Microbench_Pair_Oracle_Metadata(layout, &snapshot);

    if (!dumped)
    {
        const fs::path prefix_path(dump_prefix);
        if (!prefix_path.parent_path().empty())
        {
            fs::create_directories(prefix_path.parent_path());
        }
        const fs::path snapshot_path =
            prefix_path.string() + ".sponge_gmxpacked_forceonly.bin";
        if (!nbnxm_microbench::WriteSpongeGmxpackedForceOnlySnapshot(
                snapshot_path.string(), snapshot))
        {
            fprintf(stderr,
                    "[clustered microbench dump] failed to write %s\n",
                    snapshot_path.string().c_str());
            return false;
        }
        fprintf(stderr,
                "[clustered microbench dump] wrote %s sci=%d cjpacked=%d "
                "excl=%d atoms=%d\n",
                snapshot_path.string().c_str(),
                layout.gmxpacked_sci_numbers,
                layout.gmxpacked_cjpacked_numbers,
                layout.gmxpacked_exclusion_numbers,
                layout.total_atom_numbers);
        dumped = true;
    }
    if (captured_snapshot != nullptr)
    {
        *captured_snapshot = std::move(snapshot);
    }
    return true;
}

static bool Capture_Clustered_Microbench_Full_Output_Diagnostic_View(
    nbnxm_microbench::SpongeGmxpackedForceOnlySnapshot payload,
    nbnxm_microbench::SpongeGmxpackedFullOutputSnapshot* snapshot)
{
    if (snapshot == NULL ||
        !nbnxm_microbench::IsValidFileHeader(
            payload.header.file,
            nbnxm_microbench::SnapshotKind::spongeGmxpackedForceOnly) ||
        payload.header.total_atom_numbers == 0u ||
        payload.header.sci_numbers == 0u ||
        payload.header.cjpacked_numbers == 0u ||
        payload.sci.empty() || payload.cjpacked.empty() ||
        payload.excl.empty() || payload.sorted_xq.empty() ||
        payload.sorted_atom_ids.empty() || payload.sorted_lj_type.empty() ||
        payload.lj_ab.empty())
    {
        return false;
    }

    snapshot->header.file = nbnxm_microbench::MakeFileHeader(
        nbnxm_microbench::SnapshotKind::spongeGmxpackedFullOutput);
    snapshot->payload = std::move(payload);
    return true;
}

static void Finalize_Clustered_Microbench_Full_Output_Snapshot(
    nbnxm_microbench::SpongeGmxpackedFullOutputSnapshot* snapshot,
    const std::vector<VECTOR>& force_before, const VECTOR* frc_after,
    const std::vector<float>& atom_energy_before, const float* atom_energy_after,
    const std::vector<LTMatrix3>& atom_virial_before,
    const LTMatrix3* atom_virial_after, const float* atom_direct_cf_energy,
    const float* atom_lj_energy)
{
    if (snapshot == NULL)
    {
        return;
    }

    const size_t total_atom_numbers =
        static_cast<size_t>(snapshot->payload.header.total_atom_numbers);
    const bool need_energy = snapshot->header.compute_energy != 0u;
    const bool need_virial = snapshot->header.compute_virial != 0u;
    const bool total_output = snapshot->header.total_output != 0u;
    const size_t scalar_output_numbers = total_output ? 1u : total_atom_numbers;

    const auto force_after =
        Copy_Device_Vector_To_Host(frc_after, total_atom_numbers);
    snapshot->reference_force.clear();
    snapshot->reference_force.reserve(total_atom_numbers);
    for (size_t i = 0; i < total_atom_numbers; i += 1)
    {
        const VECTOR delta = force_after[i] - force_before[i];
        snapshot->reference_force.push_back(To_Microbench_Force_POD(delta));
    }
    snapshot->header.force_reference_numbers =
        static_cast<uint64_t>(snapshot->reference_force.size());

    snapshot->reference_atom_energy.clear();
    snapshot->reference_direct_cf_energy.clear();
    snapshot->reference_lj_energy.clear();
    snapshot->reference_atom_virial.clear();

    if (need_energy)
    {
        const auto host_atom_energy =
            Copy_Device_Vector_To_Host(atom_energy_after, scalar_output_numbers);
        const auto host_direct_cf_energy = Copy_Device_Vector_To_Host(
            atom_direct_cf_energy, scalar_output_numbers);
        const auto host_lj_energy =
            Copy_Device_Vector_To_Host(atom_lj_energy, scalar_output_numbers);
        snapshot->reference_atom_energy.reserve(scalar_output_numbers);
        snapshot->reference_direct_cf_energy.reserve(scalar_output_numbers);
        snapshot->reference_lj_energy.reserve(scalar_output_numbers);
        for (size_t i = 0; i < scalar_output_numbers; i += 1)
        {
            snapshot->reference_atom_energy.push_back(host_atom_energy[i] -
                                                      atom_energy_before[i]);
            snapshot->reference_direct_cf_energy.push_back(
                host_direct_cf_energy[i]);
            snapshot->reference_lj_energy.push_back(host_lj_energy[i]);
        }
        snapshot->header.energy_reference_numbers =
            static_cast<uint64_t>(snapshot->reference_atom_energy.size());
        snapshot->header.direct_energy_reference_numbers =
            static_cast<uint64_t>(
                snapshot->reference_direct_cf_energy.size());
        snapshot->header.lj_energy_reference_numbers =
            static_cast<uint64_t>(snapshot->reference_lj_energy.size());
    }

    if (need_virial)
    {
        const auto host_atom_virial =
            Copy_Device_Vector_To_Host(atom_virial_after, scalar_output_numbers);
        snapshot->reference_atom_virial.reserve(scalar_output_numbers);
        for (size_t i = 0; i < scalar_output_numbers; i += 1)
        {
            snapshot->reference_atom_virial.push_back(To_Microbench_Matrix_POD(
                host_atom_virial[i] - atom_virial_before[i]));
        }
        snapshot->header.virial_reference_numbers =
            static_cast<uint64_t>(snapshot->reference_atom_virial.size());
    }
}

static void Maybe_Write_Clustered_Microbench_Full_Output_Snapshot(
    const nbnxm_microbench::SpongeGmxpackedFullOutputSnapshot& snapshot)
{
    const char* dump_prefix = Clustered_Microbench_Dump_Prefix();
    static bool dumped = false;
    if (dump_prefix == NULL || dumped)
    {
        return;
    }

    const fs::path prefix_path(dump_prefix);
    if (!prefix_path.parent_path().empty())
    {
        fs::create_directories(prefix_path.parent_path());
    }
    const fs::path snapshot_path =
        prefix_path.string() + ".sponge_fulloutput.bin";
    if (!nbnxm_microbench::WriteSpongeGmxpackedFullOutputSnapshot(
            snapshot_path.string(), snapshot))
    {
        fprintf(stderr,
                "[clustered microbench dump] failed to write %s\n",
                snapshot_path.string().c_str());
        return;
    }
    fprintf(stderr,
            "[clustered microbench dump] wrote %s sci=%llu cjpacked=%llu "
            "excl=%llu atoms=%llu energy=%u virial=%u total_output=%u\n",
            snapshot_path.string().c_str(),
            static_cast<unsigned long long>(
                snapshot.payload.header.sci_numbers),
            static_cast<unsigned long long>(
                snapshot.payload.header.cjpacked_numbers),
            static_cast<unsigned long long>(
                snapshot.payload.header.excl_numbers),
            static_cast<unsigned long long>(
                snapshot.payload.header.total_atom_numbers),
            snapshot.header.compute_energy, snapshot.header.compute_virial,
            snapshot.header.total_output);
    dumped = true;
}

static uint32_t Quantize_Unit_Coordinate(float value, int bits)
{
    if (bits <= 0)
    {
        return 0;
    }
    const uint32_t grid = 1u << bits;
    float clamped = std::max(0.0f, std::min(0.99999994f, value));
    uint32_t coord = static_cast<uint32_t>(clamped * grid);
    if (coord >= grid)
    {
        coord = grid - 1;
    }
    return coord;
}

static void Hilbert_Axes_To_Transpose(std::array<uint32_t, 3>* coords,
                                      int bits)
{
    if (bits <= 0)
    {
        return;
    }
    uint32_t Q = 1u << (bits - 1);
    while (Q > 1)
    {
        const uint32_t P = Q - 1;
        for (int dim = 0; dim < 3; dim += 1)
        {
            if (((*coords)[dim] & Q) != 0)
            {
                (*coords)[0] ^= P;
            }
            else
            {
                const uint32_t t = ((*coords)[0] ^ (*coords)[dim]) & P;
                (*coords)[0] ^= t;
                (*coords)[dim] ^= t;
            }
        }
        Q >>= 1;
    }

    for (int dim = 1; dim < 3; dim += 1)
    {
        (*coords)[dim] ^= (*coords)[dim - 1];
    }

    uint32_t t = 0;
    Q = 1u << (bits - 1);
    while (Q > 1)
    {
        if (((*coords)[2] & Q) != 0)
        {
            t ^= Q - 1;
        }
        Q >>= 1;
    }
    for (int dim = 0; dim < 3; dim += 1)
    {
        (*coords)[dim] ^= t;
    }
}

static uint64_t Hilbert_Index_3D(uint32_t x, uint32_t y, uint32_t z, int bits)
{
    std::array<uint32_t, 3> coords = {x, y, z};
    Hilbert_Axes_To_Transpose(&coords, bits);

    uint64_t index = 0;
    for (int bit = bits - 1; bit >= 0; bit -= 1)
    {
        for (int dim = 0; dim < 3; dim += 1)
        {
            index = (index << 1) | ((coords[dim] >> bit) & 1u);
        }
    }
    return index;
}

static uint64_t Hilbert_Index_3D(VECTOR normalized, int bits)
{
    return Hilbert_Index_3D(Quantize_Unit_Coordinate(normalized.x, bits),
                            Quantize_Unit_Coordinate(normalized.y, bits),
                            Quantize_Unit_Coordinate(normalized.z, bits), bits);
}

#ifdef USE_GPU
static __device__ __forceinline__ unsigned int Clustered_Subgroup_Mask(
    int lane, int subgroup_width)
{
    const unsigned int subgroup =
        static_cast<unsigned int>(lane / subgroup_width);
    const unsigned int width_mask =
        (1u << static_cast<unsigned int>(subgroup_width)) - 1u;
    return width_mask << (subgroup * static_cast<unsigned int>(subgroup_width));
}

static __device__ __forceinline__ VECTOR Reduce_Clustered_Subgroup_Vector(
    VECTOR value, int lane, int subgroup_width)
{
    const unsigned int subgroup_mask =
        Clustered_Subgroup_Mask(lane, subgroup_width);
    for (int delta = subgroup_width >> 1; delta > 0; delta >>= 1)
    {
        value.x +=
            deviceShflDown(subgroup_mask, value.x, delta, subgroup_width);
        value.y +=
            deviceShflDown(subgroup_mask, value.y, delta, subgroup_width);
        value.z +=
            deviceShflDown(subgroup_mask, value.z, delta, subgroup_width);
    }
    return value;
}

static __device__ __forceinline__
    void Reduce_Clustered_Subgroup_Vector_Components(float& x, float& y,
                                                     float& z, int lane,
                                                     int subgroup_width)
{
    const unsigned int subgroup_mask =
        Clustered_Subgroup_Mask(lane, subgroup_width);
    for (int delta = subgroup_width >> 1; delta > 0; delta >>= 1)
    {
        x += deviceShflDown(subgroup_mask, x, delta, subgroup_width);
        y += deviceShflDown(subgroup_mask, y, delta, subgroup_width);
        z += deviceShflDown(subgroup_mask, z, delta, subgroup_width);
    }
}

static __device__ __forceinline__ VECTOR Reduce_Clustered_Warp_Vector_Over_J(
    VECTOR value, int subgroup_width)
{
    for (int delta = warpSize >> 1; delta >= subgroup_width; delta >>= 1)
    {
        value.x += deviceShflDown(FULL_MASK, value.x, delta, warpSize);
        value.y += deviceShflDown(FULL_MASK, value.y, delta, warpSize);
        value.z += deviceShflDown(FULL_MASK, value.z, delta, warpSize);
    }
    return value;
}

static __device__ __forceinline__
    void Reduce_Clustered_Warp_Vector_Over_J_Components(float& x, float& y,
                                                        float& z,
                                                        int subgroup_width)
{
    for (int delta = warpSize >> 1; delta >= subgroup_width; delta >>= 1)
    {
        x += deviceShflDown(FULL_MASK, x, delta, warpSize);
        y += deviceShflDown(FULL_MASK, y, delta, warpSize);
        z += deviceShflDown(FULL_MASK, z, delta, warpSize);
    }
}

template <bool enabled, int size>
struct Clustered_Energy_Buffer
{
    float unused;

    __device__ __forceinline__ float& operator[](int)
    {
        return unused;
    }

    __device__ __forceinline__ const float& operator[](int) const
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
        for (int i = 0; i < size; i += 1)
        {
            values[i] = 0.0f;
        }
    }

    __device__ __forceinline__ float& operator[](int idx)
    {
        return values[idx];
    }

    __device__ __forceinline__ const float& operator[](int idx) const
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
        const bool store_energy, const bool store_virial)
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
            for (int i = 0; i < size; i += 1)
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
        const bool store_energy, const bool store_virial)
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

template <typename T>
static __device__ __forceinline__ T Broadcast_Clustered_Subgroup_Value(
    T value, int lane, int subgroup_width)
{
    const unsigned int subgroup_mask =
        Clustered_Subgroup_Mask(lane, subgroup_width);
    const int subgroup_leader = lane - lane % subgroup_width;
    return deviceShfl(subgroup_mask, value, subgroup_leader, warpSize);
}

static __device__ __forceinline__ float4 Broadcast_Clustered_Subgroup_Float4(
    float4 value, int lane, int subgroup_width)
{
    value.x = Broadcast_Clustered_Subgroup_Value(value.x, lane, subgroup_width);
    value.y = Broadcast_Clustered_Subgroup_Value(value.y, lane, subgroup_width);
    value.z = Broadcast_Clustered_Subgroup_Value(value.z, lane, subgroup_width);
    value.w = Broadcast_Clustered_Subgroup_Value(value.w, lane, subgroup_width);
    return value;
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

static __device__ __forceinline__ uint64_t Broadcast_Clustered_Subgroup_U64(
    uint64_t value, int lane, int subgroup_width)
{
    unsigned int lo = static_cast<unsigned int>(value);
    unsigned int hi = static_cast<unsigned int>(value >> 32);
    lo = Broadcast_Clustered_Subgroup_Value(lo, lane, subgroup_width);
    hi = Broadcast_Clustered_Subgroup_Value(hi, lane, subgroup_width);
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
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

static __device__ __forceinline__ float
Reduce_Clustered_Subgroup_Vector_To_Component(float x, float y, float z,
                                               int component_lane, int lane,
                                               int subgroup_width)
{
    const unsigned int subgroup_mask =
        Clustered_Subgroup_Mask(lane, subgroup_width);
    const int subgroup_leader = lane - lane % subgroup_width;
    Reduce_Clustered_Subgroup_Vector_Components(x, y, z, lane,
                                                subgroup_width);
    x = deviceShfl(subgroup_mask, x, subgroup_leader, warpSize);
    y = deviceShfl(subgroup_mask, y, subgroup_leader, warpSize);
    z = deviceShfl(subgroup_mask, z, subgroup_leader, warpSize);
    if (component_lane == 0)
    {
        return x;
    }
    if (component_lane == 1)
    {
        return y;
    }
    return z;
}

static __device__ __forceinline__ float
Reduce_Clustered_Warp_I_To_Component(float x, float y, float z, int i_lane,
                                      int component_lane, int subgroup_width)
{
    Reduce_Clustered_Warp_Vector_Over_J_Components(x, y, z, subgroup_width);
    x = deviceShfl(FULL_MASK, x, i_lane, warpSize);
    y = deviceShfl(FULL_MASK, y, i_lane, warpSize);
    z = deviceShfl(FULL_MASK, z, i_lane, warpSize);
    if (component_lane == 0)
    {
        return x;
    }
    if (component_lane == 1)
    {
        return y;
    }
    return z;
}

static __device__ __forceinline__ void Clustered_Atomic_Add_Force_Component(
    VECTOR* frc, int atom_index, int component, float value)
{
    float* frc_component = reinterpret_cast<float*>(frc + atom_index);
    atomicAdd(frc_component + component, value);
}

static __global__ void Scatter_Sorted_Clustered_Force(
    const int total_atom_numbers, const int* sorted_atom_ids,
    const VECTOR* sorted_frc, VECTOR* frc)
{
    SIMPLE_DEVICE_FOR(sorted_i, total_atom_numbers)
    {
        const int atom_i = sorted_atom_ids[sorted_i];
        frc[atom_i] = frc[atom_i] + sorted_frc[sorted_i];
    }
}

static __device__ __forceinline__ float Reduce_Clustered_Warp_Float_Over_J(
    float value, int subgroup_width)
{
    for (int delta = warpSize >> 1; delta >= subgroup_width; delta >>= 1)
    {
        value += deviceShflDown(FULL_MASK, value, delta, warpSize);
    }
    return value;
}

static __device__ __forceinline__ LTMatrix3 Reduce_Clustered_Warp_Virial_Over_J(
    LTMatrix3 value, int subgroup_width)
{
    for (int delta = warpSize >> 1; delta >= subgroup_width; delta >>= 1)
    {
        value.a11 += deviceShflDown(FULL_MASK, value.a11, delta, warpSize);
        value.a21 += deviceShflDown(FULL_MASK, value.a21, delta, warpSize);
        value.a22 += deviceShflDown(FULL_MASK, value.a22, delta, warpSize);
        value.a31 += deviceShflDown(FULL_MASK, value.a31, delta, warpSize);
        value.a32 += deviceShflDown(FULL_MASK, value.a32, delta, warpSize);
        value.a33 += deviceShflDown(FULL_MASK, value.a33, delta, warpSize);
    }
    return value;
}

static __device__ __forceinline__ float Reduce_Clustered_Warp_Float_All(
    float value)
{
    for (int delta = warpSize >> 1; delta > 0; delta >>= 1)
    {
        value += deviceShflDown(FULL_MASK, value, delta, warpSize);
    }
    return value;
}

static __device__ __forceinline__ LTMatrix3 Reduce_Clustered_Warp_Virial_All(
    LTMatrix3 value)
{
    for (int delta = warpSize >> 1; delta > 0; delta >>= 1)
    {
        value.a11 += deviceShflDown(FULL_MASK, value.a11, delta, warpSize);
        value.a21 += deviceShflDown(FULL_MASK, value.a21, delta, warpSize);
        value.a22 += deviceShflDown(FULL_MASK, value.a22, delta, warpSize);
        value.a31 += deviceShflDown(FULL_MASK, value.a31, delta, warpSize);
        value.a32 += deviceShflDown(FULL_MASK, value.a32, delta, warpSize);
        value.a33 += deviceShflDown(FULL_MASK, value.a33, delta, warpSize);
    }
    return value;
}

static __device__ __forceinline__ float4 Pack_Clustered_Virial_Lo(
    LTMatrix3 value)
{
    return {value.a11, value.a21, value.a22, value.a31};
}

static __device__ __forceinline__ float2 Pack_Clustered_Virial_Hi(
    LTMatrix3 value)
{
    return {value.a32, value.a33};
}

static __device__ __forceinline__ LTMatrix3 Unpack_Clustered_Virial(
    float4 lo, float2 hi)
{
    return {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y};
}
#endif

static void Build_Cornerstone_Leaves(
    const std::vector<OrderedResiduePoint>& points,
    const std::vector<int>& residue_indices, int depth, int max_depth,
    int leaf_size, VECTOR min_bound, VECTOR max_bound,
    std::vector<CornerstoneLeaf>* leaves)
{
    if (residue_indices.empty())
    {
        return;
    }
    if ((int)residue_indices.size() <= leaf_size || depth >= max_depth)
    {
        leaves->push_back(
            {residue_indices, min_bound, max_bound, static_cast<uint64_t>(0)});
        return;
    }

    const VECTOR mid = 0.5f * (min_bound + max_bound);
    std::array<std::vector<int>, 8> children;
    for (int residue_index : residue_indices)
    {
        const VECTOR& p = points[residue_index].normalized;
        int octant = 0;
        if (p.x >= mid.x)
        {
            octant |= 1;
        }
        if (p.y >= mid.y)
        {
            octant |= 2;
        }
        if (p.z >= mid.z)
        {
            octant |= 4;
        }
        children[octant].push_back(residue_index);
    }

    int non_empty_children = 0;
    for (const auto& child : children)
    {
        non_empty_children += !child.empty();
    }
    if (non_empty_children <= 1)
    {
        leaves->push_back(
            {residue_indices, min_bound, max_bound, static_cast<uint64_t>(0)});
        return;
    }

    for (int octant = 0; octant < 8; octant += 1)
    {
        if (children[octant].empty())
        {
            continue;
        }
        VECTOR child_min = min_bound;
        VECTOR child_max = max_bound;
        if ((octant & 1) != 0)
        {
            child_min.x = mid.x;
        }
        else
        {
            child_max.x = mid.x;
        }
        if ((octant & 2) != 0)
        {
            child_min.y = mid.y;
        }
        else
        {
            child_max.y = mid.y;
        }
        if ((octant & 4) != 0)
        {
            child_min.z = mid.z;
        }
        else
        {
            child_max.z = mid.z;
        }
        Build_Cornerstone_Leaves(points, children[octant], depth + 1,
                                 max_depth, leaf_size, child_min, child_max,
                                 leaves);
    }
}

}  // namespace

// 由LJ坐标和转化系数求距离
__global__ void Copy_LJ_Type_To_New_Crd(const int atom_numbers,
                                        VECTOR_LJ* new_crd, const int* LJ_type)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        new_crd[atom_i].LJ_type = LJ_type[atom_i];
    }
}

__global__ void Copy_Crd_And_Charge_To_New_Crd(const int atom_numbers,
                                               const VECTOR* crd,
                                               VECTOR_LJ* new_crd,
                                               const float* charge)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        new_crd[atom_i].crd = crd[atom_i];
        new_crd[atom_i].charge = charge[atom_i];
    }
}

__global__ void Copy_Crd_To_New_Crd(const int atom_numbers, const VECTOR* crd,
                                    VECTOR_LJ* new_crd)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        new_crd[atom_i].crd = crd[atom_i];
    }
}

static __global__ void Gather_Sorted_LJ_Crd(const int atom_numbers,
                                            const int* permutation,
                                            const VECTOR_LJ* src,
                                            VECTOR_LJ* dest)
{
    SIMPLE_DEVICE_FOR(sorted_i, atom_numbers)
    {
        dest[sorted_i] = src[permutation[sorted_i]];
    }
}

static __global__ void Gather_Sorted_LJ_Packed(const int atom_numbers,
                                               const int* permutation,
                                               const VECTOR_LJ* src,
                                               int* sorted_atom_ids,
                                               float4* sorted_xq,
                                               int* sorted_lj_type)
{
    SIMPLE_DEVICE_FOR(sorted_i, atom_numbers)
    {
        const int atom_i = permutation[sorted_i];
        const VECTOR_LJ atom = src[atom_i];
        sorted_atom_ids[sorted_i] = atom_i;
        sorted_xq[sorted_i] = {atom.crd.x, atom.crd.y, atom.crd.z, atom.charge};
        sorted_lj_type[sorted_i] = atom.LJ_type;
    }
}

static __global__ void device_add(float* variable, const float adder)
{
    variable[0] += adder;
}

static __host__ __device__ __forceinline__ VECTOR_LJ Make_Packed_LJ_Atom(
    const float4 xq, const int lj_type)
{
    VECTOR_LJ atom = {};
    atom.crd = {xq.x, xq.y, xq.z};
    atom.LJ_type = lj_type;
    atom.charge = xq.w;
    return atom;
}

static __host__ __device__ __forceinline__ VECTOR
Get_Clustered_Shifted_Displacement(const VECTOR_LJ r2, const VECTOR_LJ r1,
                                   const VECTOR shift_vec)
{
    return (r2.crd - r1.crd) - shift_vec;
}

static __device__ __forceinline__ float
Get_Clustered_LJ_Force_Abs(const float inv_r2, const float inv_r6,
                           const float A, const float B)
{
    return (B - A * inv_r6) * inv_r6 * inv_r2;
}

static __device__ __forceinline__ float
Get_Clustered_LJ_Energy(const float inv_r6, const float A, const float B)
{
    return (0.083333333f * A * inv_r6 - 0.166666667f * B) * inv_r6;
}

static __device__ __forceinline__ float Get_Clustered_Direct_Coulomb_Energy(
    const float charge_product, const float inv_r, const float beta_dr)
{
    return charge_product * erfcf(beta_dr) * inv_r;
}

static __device__ __forceinline__ float Get_Clustered_Direct_Coulomb_Force_Abs(
    const float charge_product, const float inv_r, const float inv_r2,
    const float beta_dr)
{
    return charge_product * inv_r * inv_r2 *
           (beta_dr * TWO_DIVIDED_BY_SQRT_PI * expf(-beta_dr * beta_dr) +
            erfcf(beta_dr));
}

static __host__ __device__ __forceinline__ float
Clustered_PME_Corr_F(const float z2)
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

static __device__ __forceinline__ float
Get_Clustered_Direct_Coulomb_Force_Abs_PME_Corr(
    const float charge_product, const float inv_r, const float inv_r2,
    const float beta2_r2, const float beta3)
{
    return charge_product *
           (inv_r * inv_r2 + Clustered_PME_Corr_F(beta2_r2) * beta3);
}

template <bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb>
static __global__ void Lennard_Jones_And_Direct_Coulomb_Device(
    const int local_atom_numbers, const int solvent_numbers,
    const ATOM_GROUP* nl, const VECTOR_LJ* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* LJ_type_A, const float* LJ_type_B,
    const float cutoff, VECTOR* frc, const float pme_beta, float* atom_energy,
    LTMatrix3* atom_virial, float* atom_direct_cf_energy, float* atom_LJ_ene,
    const bool store_energy, const bool store_virial)
{
#ifdef USE_GPU
    int atom_i = 0 + blockDim.y * blockIdx.x + threadIdx.y;
    if (atom_i < local_atom_numbers - solvent_numbers)
#else
#pragma omp parallel for schedule(dynamic)
    for (int atom_i = 0; atom_i < local_atom_numbers - solvent_numbers;
         atom_i++)
#endif
    {
        VECTOR frc_record = {0.0f, 0.0f, 0.0f};
        LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float energy_lj = 0.0f;
        float energy_coulomb = 0.0f;
        float energy_total = 0.0f;
        ATOM_GROUP nl_i = nl[atom_i];
        VECTOR_LJ r1 = crd[atom_i];
#ifdef USE_GPU
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
#else
        for (int j = 0; j < nl_i.atom_numbers; j += 1)
#endif
        {
            int atom_j = nl_i.atom_serial[j];
            float ij_factor = atom_j < local_atom_numbers ? 1.0f : 0.5f;
            VECTOR_LJ r2 = crd[atom_j];
            VECTOR dr = Get_Periodic_Displacement(r2, r1, cell, rcell);
            float dr_abs = norm3df(dr.x, dr.y, dr.z);
            if (dr_abs < cutoff)
            {
                int atom_pair_LJ_type = Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                float A = LJ_type_A[atom_pair_LJ_type];
                float B = LJ_type_B[atom_pair_LJ_type];
                if (need_force)
                {
                    float frc_abs = Get_LJ_Force(r1, r2, dr_abs, A, B);
                    if (need_coulomb)
                    {
                        float frc_cf_abs =
                            Get_Direct_Coulomb_Force(r1, r2, dr_abs, pme_beta);
                        frc_abs = frc_abs - frc_cf_abs;
                    }
                    VECTOR frc_lin = frc_abs * dr;
                    frc_record = frc_record + frc_lin;
                    if (atom_j < local_atom_numbers)
                    {
                        atomicAdd(frc + atom_j, -frc_lin);
                    }
                    if (need_virial)
                    {
                        virial = virial - ij_factor * Get_Virial_From_Force_Dis(
                                                          frc_lin, dr);
                    }
                }
                if (need_energy && store_energy)
                {
                    energy_lj +=
                        ij_factor * Get_LJ_Energy(r1, r2, dr_abs, A, B);
                    if (need_coulomb)
                    {
                        energy_coulomb +=
                            ij_factor *
                            Get_Direct_Coulomb_Energy(r1, r2, dr_abs, pme_beta);
                    }
                }
            }
        }
        energy_total = energy_lj + energy_coulomb;
        if (need_force)
        {
            Warp_Sum_To(frc + atom_i, frc_record, warpSize);
        }
        if (need_energy && store_energy)
        {
            Warp_Sum_To(atom_energy + atom_i, energy_total, warpSize);
            Warp_Sum_To(atom_LJ_ene + atom_i, energy_lj, warpSize);
            if (need_coulomb)
                Warp_Sum_To(atom_direct_cf_energy + atom_i, energy_coulomb,
                            warpSize);
        }
        if (need_virial && store_virial)
        {
            Warp_Sum_To(atom_virial + atom_i, virial, warpSize);
        }
    }
}

#ifdef USE_CPU
template <bool full_output>
static void Cpu_Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb(
    const int sci_numbers, const int cluster_size,
    const int super_cluster_clusters, const int local_atom_numbers,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const int* super_cluster_offsets, const LJ_CLUSTERED_SCI* sci_entries,
    const LJ_CLUSTERED_CJ_PACKED* cj_packed_entries,
    const unsigned long long* exclusion_mask_pool, const int* sorted_atom_ids,
    const float4* sorted_xq, const int* sorted_lj_type, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* LJ_type_A, const float* LJ_type_B,
    const float cutoff, VECTOR* frc, const float pme_beta, float* atom_energy,
    LTMatrix3* atom_virial, float* atom_direct_cf_energy, float* atom_LJ_ene,
    const bool store_energy, const bool store_virial)
{
    constexpr int max_cluster_size = kClusteredClusterSize;
    const float cutoff_sq = cutoff * cutoff;

#pragma omp parallel for schedule(dynamic)
    for (int sci = 0; sci < sci_numbers; sci += 1)
    {
        const LJ_CLUSTERED_SCI sci_entry = sci_entries[sci];
        const int super_i = sci_entry.supercluster_id;
        const int cluster_i_start = super_cluster_offsets[super_i];
        const int cluster_i_end = super_cluster_offsets[super_i + 1];
        const bool sci_is_central =
            sci_entry.shift_id == kClusteredCentralShiftId;

        for (int cluster_i = cluster_i_start; cluster_i < cluster_i_end;
             cluster_i += 1)
        {
            const unsigned int valid_mask_i = cluster_valid_masks[cluster_i];
            const unsigned int local_mask_i = cluster_local_masks[cluster_i];
            const int i_local = cluster_i - cluster_i_start;
            for (int lane_i = 0; lane_i < cluster_size; lane_i += 1)
            {
                if ((valid_mask_i & (1u << lane_i)) == 0u ||
                    (local_mask_i & (1u << lane_i)) == 0u)
                {
                    continue;
                }

                const int sorted_atom_i =
                    cluster_offsets[cluster_i] + lane_i;
                const int atom_i = sorted_atom_ids[sorted_atom_i];
                const VECTOR_LJ r1 = Make_Packed_LJ_Atom(
                    sorted_xq[sorted_atom_i], sorted_lj_type[sorted_atom_i]);
                VECTOR frc_i = {0.0f, 0.0f, 0.0f};
                float energy_lj = 0.0f;
                float energy_coulomb = 0.0f;
                LTMatrix3 virial = {0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f};

                for (int packed_idx = sci_entry.cjpacked_begin;
                     packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
                {
                    const LJ_CLUSTERED_CJ_PACKED& packed =
                        cj_packed_entries[packed_idx];
                    for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
                    {
                        const int cluster_j = packed.cj[jm];
                        if (cluster_j < 0)
                        {
                            continue;
                        }
                        const unsigned int imask =
                            Clustered_Jm_Imask(packed.imei[0], jm) |
                            Clustered_Jm_Imask(packed.imei[1], jm);
                        if ((imask & (1u << i_local)) == 0u)
                        {
                            continue;
                        }

                        const unsigned int valid_mask_j =
                            cluster_valid_masks[cluster_j];
                        const int exclusion_index =
                            Clustered_First_Exclusion_Index(
                                packed, jm, i_local);
                        const unsigned long long exclusion_mask =
                            exclusion_index >= 0
                                ? exclusion_mask_pool[exclusion_index]
                                : 0ull;
                        const VECTOR pair_shift_vec =
                            Clustered_Shift_Vector_From_Id(
                                sci_entry.shift_id, cell);
                        VECTOR frc_j[max_cluster_size] = {};

                        for (int lane_j = 0; lane_j < cluster_size;
                             lane_j += 1)
                        {
                            if ((valid_mask_j & (1u << lane_j)) == 0u)
                            {
                                continue;
                            }
                            const int sorted_atom_j =
                                cluster_offsets[cluster_j] + lane_j;
                            const int atom_j = sorted_atom_ids[sorted_atom_j];
                            if (sci_is_central && cluster_i == cluster_j &&
                                atom_j < local_atom_numbers &&
                                lane_j <= lane_i)
                            {
                                continue;
                            }
                            if ((exclusion_mask &
                                 (1ull << (lane_i * cluster_size + lane_j))) !=
                                0ull)
                            {
                                continue;
                            }

                            const VECTOR_LJ r2 = Make_Packed_LJ_Atom(
                                sorted_xq[sorted_atom_j],
                                sorted_lj_type[sorted_atom_j]);
                            const VECTOR dr =
                                Get_Clustered_Shifted_Displacement(
                                    r2, r1, pair_shift_vec);
                            const float dr2 = dr * dr;
                            if (dr2 >= cutoff_sq || dr2 == 0.0f)
                            {
                                continue;
                            }

                            const float dr_abs = sqrtf(dr2);
                            const int atom_pair_LJ_type =
                                Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                            const float A = LJ_type_A[atom_pair_LJ_type];
                            const float B = LJ_type_B[atom_pair_LJ_type];
                            const float ij_factor =
                                atom_j < local_atom_numbers ? 1.0f : 0.5f;
                            float frc_abs =
                                Get_LJ_Force(r1, r2, dr_abs, A, B);
                            frc_abs -= Get_Direct_Coulomb_Force(
                                r1, r2, dr_abs, pme_beta);
                            const VECTOR frc_lin = frc_abs * dr;
                            frc_i = frc_i + frc_lin;
                            if (atom_j < local_atom_numbers)
                            {
                                frc_j[lane_j] = frc_j[lane_j] - frc_lin;
                            }

                            if constexpr (full_output)
                            {
                                virial =
                                    virial -
                                    ij_factor *
                                        Get_Virial_From_Force_Dis(frc_lin, dr);
                                energy_lj +=
                                    ij_factor *
                                    Get_LJ_Energy(r1, r2, dr_abs, A, B);
                                energy_coulomb +=
                                    ij_factor *
                                    Get_Direct_Coulomb_Energy(
                                        r1, r2, dr_abs, pme_beta);
                            }
                        }

                        for (int lane_j = 0; lane_j < cluster_size;
                             lane_j += 1)
                        {
                            const int sorted_atom_j =
                                cluster_offsets[cluster_j] + lane_j;
                            const int atom_j = sorted_atom_ids[sorted_atom_j];
                            if ((valid_mask_j & (1u << lane_j)) != 0u &&
                                atom_j < local_atom_numbers)
                            {
                                atomicAdd(frc + atom_j, frc_j[lane_j]);
                            }
                        }
                    }
                }

                atomicAdd(frc + atom_i, frc_i);
                if constexpr (full_output)
                {
                    if (store_energy)
                    {
                        atomicAdd(atom_energy + atom_i,
                                  energy_lj + energy_coulomb);
                        atomicAdd(atom_LJ_ene + atom_i, energy_lj);
                        atomicAdd(atom_direct_cf_energy + atom_i,
                                  energy_coulomb);
                    }
                    if (store_virial)
                    {
                        atomicAdd(atom_virial + atom_i, virial);
                    }
                }
            }
        }
    }
}
#endif

#ifndef USE_CPU
template <bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb>
static __global__ void Clustered_Lennard_Jones_And_Direct_Coulomb_Device(
    const int sci_numbers, const int cluster_size,
    const int super_cluster_clusters, const int local_atom_numbers,
    const int* cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const int* super_cluster_offsets, const int* sci_supercluster_ids,
    const int* sci_offsets, const int* cjpacked_cluster_ids,
    const unsigned int* cjpacked_imasks,
    const int* cjpacked_exclusion_indices,
    const unsigned long long* exclusion_mask_pool,
    const int* sorted_atom_ids, const float4* sorted_xq,
    const int* sorted_lj_type,
    const LTMatrix3 cell, const LTMatrix3 rcell, const float* LJ_type_A,
    const float* LJ_type_B, const float cutoff, VECTOR* frc,
    const float pme_beta, float* atom_energy, LTMatrix3* atom_virial,
    float* atom_direct_cf_energy, float* atom_LJ_ene)
{
    constexpr int max_cluster_size = 8;
    constexpr int max_super_cluster_atoms = 64;
    constexpr int max_block_warps = 2;
#ifdef USE_GPU
    const int sci = blockIdx.x;
    const int tid = threadIdx.x;
    if (sci < sci_numbers &&
        tid < super_cluster_clusters * cluster_size)
#else
#pragma omp parallel for schedule(dynamic)
    for (int sci = 0; sci < sci_numbers; sci += 1)
#endif
    {
#ifndef USE_GPU
        const int super_i = sci_supercluster_ids[sci];
        const int cluster_i_start = super_cluster_offsets[super_i];
        const int cluster_i_end = super_cluster_offsets[super_i + 1];
        for (int cluster_i = cluster_i_start; cluster_i < cluster_i_end;
             cluster_i += 1)
        {
            const unsigned int valid_mask_i = cluster_valid_masks[cluster_i];
            const unsigned int local_mask_i = cluster_local_masks[cluster_i];
            const int i_local = cluster_i - cluster_i_start;
            const float cutoff_sq = cutoff * cutoff;
            for (int lane_i = 0; lane_i < cluster_size; lane_i += 1)
            {
                if ((valid_mask_i & (1u << lane_i)) == 0u ||
                    (local_mask_i & (1u << lane_i)) == 0u)
                {
                    continue;
                }
                const int start_i = cluster_offsets[cluster_i];
                const int sorted_atom_i = start_i + lane_i;
                const int atom_i = sorted_atom_ids[sorted_atom_i];
                const VECTOR_LJ r1 = Make_Packed_LJ_Atom(
                    sorted_xq[sorted_atom_i], sorted_lj_type[sorted_atom_i]);
                VECTOR frc_i = {0.0f, 0.0f, 0.0f};
                float energy_lj = 0.0f;
                float energy_coulomb = 0.0f;
                LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

                for (int cj = sci_offsets[sci]; cj < sci_offsets[sci + 1];
                     cj += 1)
                {
                    const unsigned int imask = cjpacked_imasks[cj];
                    if (imask == 0u)
                    {
                        continue;
                    }
                    if ((imask & (1u << i_local)) == 0u)
                    {
                        continue;
                    }
                    const int cluster_j = cjpacked_cluster_ids[cj];
                    const unsigned int valid_mask_j =
                        cluster_valid_masks[cluster_j];
                    const int exclusion_index =
                        cjpacked_exclusion_indices[cj * super_cluster_clusters +
                                                   i_local];
                    const unsigned long long exclusion_mask =
                        exclusion_index >= 0 ? exclusion_mask_pool[exclusion_index]
                                             : 0ull;
                    VECTOR frc_j[max_cluster_size];
                    for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
                    {
                        frc_j[lane_j] = {0.0f, 0.0f, 0.0f};
                    }
                    for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
                    {
                        if ((valid_mask_j & (1u << lane_j)) == 0u)
                        {
                            continue;
                        }
                        const int sorted_atom_j =
                            cluster_offsets[cluster_j] + lane_j;
                        const int atom_j = sorted_atom_ids[sorted_atom_j];
                        if (cluster_i == cluster_j && atom_j < local_atom_numbers &&
                            lane_j <= lane_i)
                        {
                            continue;
                        }
                        if ((exclusion_mask &
                             (1ull << (lane_i * cluster_size + lane_j))) != 0ull)
                        {
                            continue;
                        }
                        const VECTOR_LJ r2 = Make_Packed_LJ_Atom(
                            sorted_xq[sorted_atom_j],
                            sorted_lj_type[sorted_atom_j]);
                        const VECTOR dr =
                            Get_Periodic_Displacement(r2, r1, cell, rcell);
                        const float dr2 = dr * dr;
                        if (dr2 >= cutoff_sq || dr2 == 0.0f)
                        {
                            continue;
                        }
                        const float dr_abs = sqrtf(dr2);
                        const int atom_pair_LJ_type =
                            Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                        const float A = LJ_type_A[atom_pair_LJ_type];
                        const float B = LJ_type_B[atom_pair_LJ_type];
                        const float ij_factor =
                            atom_j < local_atom_numbers ? 1.0f : 0.5f;
                        if (need_force)
                        {
                            float frc_abs = Get_LJ_Force(r1, r2, dr_abs, A, B);
                            if (need_coulomb)
                            {
                                frc_abs -= Get_Direct_Coulomb_Force(
                                    r1, r2, dr_abs, pme_beta);
                            }
                            const VECTOR frc_lin = frc_abs * dr;
                            frc_i = frc_i + frc_lin;
                            if (atom_j < local_atom_numbers)
                            {
                                frc_j[lane_j] = frc_j[lane_j] - frc_lin;
                            }
                            if (need_virial)
                            {
                                virial = virial -
                                         ij_factor *
                                             Get_Virial_From_Force_Dis(frc_lin, dr);
                            }
                        }
                        if (need_energy)
                        {
                            energy_lj +=
                                ij_factor * Get_LJ_Energy(r1, r2, dr_abs, A, B);
                            if (need_coulomb)
                            {
                                energy_coulomb +=
                                    ij_factor * Get_Direct_Coulomb_Energy(
                                                    r1, r2, dr_abs, pme_beta);
                            }
                        }
                    }
                    if (need_force)
                    {
                        for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
                        {
                            if ((valid_mask_j & (1u << lane_j)) == 0u)
                            {
                                continue;
                            }
                            const int sorted_atom_j =
                                cluster_offsets[cluster_j] + lane_j;
                            const int atom_j = sorted_atom_ids[sorted_atom_j];
                            if (atom_j < local_atom_numbers)
                            {
                                atomicAdd(frc + atom_j, frc_j[lane_j]);
                            }
                        }
                    }
                }
                if (need_energy)
                {
                    atomicAdd(atom_energy + atom_i, energy_lj + energy_coulomb);
                    atomicAdd(atom_LJ_ene + atom_i, energy_lj);
                    if (need_coulomb)
                    {
                        atomicAdd(atom_direct_cf_energy + atom_i,
                                  energy_coulomb);
                    }
                }
                if (need_force)
                {
                    atomicAdd(frc + atom_i, frc_i);
                }
                if (need_virial)
                {
                    atomicAdd(atom_virial + atom_i, virial);
                }
            }
        }
#else
        __shared__ float4 shared_i_xq[max_super_cluster_atoms];
        __shared__ int shared_i_lj_type[max_super_cluster_atoms];
        __shared__ int shared_i_atom_ids[max_super_cluster_atoms];
        __shared__ float4 shared_j_xq[max_cluster_size];
        __shared__ int shared_j_lj_type[max_cluster_size];
        __shared__ int shared_j_atom_ids[max_cluster_size];
        __shared__ int shared_j_local_flags[max_cluster_size];
        __shared__ unsigned int shared_j_valid_mask;
        __shared__ VECTOR warp_j_force[max_block_warps][max_cluster_size];

        const int super_i = sci_supercluster_ids[sci];
        const int cluster_i_start = super_cluster_offsets[super_i];
        const int cluster_i_end = super_cluster_offsets[super_i + 1];
        const int i_cluster_local = tid / cluster_size;
        const int i_lane = tid % cluster_size;
        const int active_cluster_count = cluster_i_end - cluster_i_start;
        bool active_i = false;
        int cluster_i = -1;
        int atom_i = -1;
        VECTOR frc_i = {0.0f, 0.0f, 0.0f};
        float energy_lj = 0.0f;
        float energy_coulomb = 0.0f;
        LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float4 r1_xq = {0.0f, 0.0f, 0.0f, 0.0f};
        int r1_lj_type = 0;
        const float cutoff_sq = cutoff * cutoff;

        if (i_cluster_local < active_cluster_count)
        {
            cluster_i = cluster_i_start + i_cluster_local;
            if ((cluster_valid_masks[cluster_i] & (1u << i_lane)) != 0u)
            {
                const int sorted_atom_i = cluster_offsets[cluster_i] + i_lane;
                shared_i_xq[tid] = sorted_xq[sorted_atom_i];
                shared_i_lj_type[tid] = sorted_lj_type[sorted_atom_i];
                shared_i_atom_ids[tid] = sorted_atom_ids[sorted_atom_i];
                if ((cluster_local_masks[cluster_i] & (1u << i_lane)) != 0u)
                {
                    active_i = true;
                    atom_i = shared_i_atom_ids[tid];
                    r1_xq = shared_i_xq[tid];
                    r1_lj_type = shared_i_lj_type[tid];
                }
            }
        }
        __syncthreads();

        const int lane = tid & (warpSize - 1);
        const int warp_id = tid / warpSize;
        const int warp_count =
            (super_cluster_clusters * cluster_size + warpSize - 1) / warpSize;

        for (int cj = sci_offsets[sci]; cj < sci_offsets[sci + 1]; cj += 1)
        {
            const unsigned int imask = cjpacked_imasks[cj];
            if (imask == 0u)
            {
                continue;
            }
            const int cluster_j = cjpacked_cluster_ids[cj];
            const unsigned int valid_mask_j = cluster_valid_masks[cluster_j];
            if (tid == 0)
            {
                shared_j_valid_mask = valid_mask_j;
            }
            if (tid < cluster_size)
            {
                if ((valid_mask_j & (1u << tid)) != 0u)
                {
                    const int sorted_atom_j = cluster_offsets[cluster_j] + tid;
                    shared_j_xq[tid] = sorted_xq[sorted_atom_j];
                    shared_j_lj_type[tid] = sorted_lj_type[sorted_atom_j];
                    shared_j_atom_ids[tid] = sorted_atom_ids[sorted_atom_j];
                    shared_j_local_flags[tid] =
                        shared_j_atom_ids[tid] < local_atom_numbers ? 1 : 0;
                }
                else
                {
                    shared_j_atom_ids[tid] = -1;
                    shared_j_local_flags[tid] = 0;
                }
            }
            __syncthreads();
            const bool tile_active =
                active_i && ((imask & (1u << i_cluster_local)) != 0u);
            unsigned long long exclusion_mask = 0ull;
            VECTOR_LJ r1 = {};
            if (active_i)
            {
                r1 = Make_Packed_LJ_Atom(r1_xq, r1_lj_type);
            }
            if (tile_active)
            {
                const int exclusion_index =
                    cjpacked_exclusion_indices[cj * super_cluster_clusters +
                                               i_cluster_local];
                exclusion_mask =
                    exclusion_index >= 0 ? exclusion_mask_pool[exclusion_index]
                                         : 0ull;
            }

            for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
            {
                VECTOR j_force_local = {0.0f, 0.0f, 0.0f};
                if (tile_active && (shared_j_valid_mask & (1u << lane_j)) != 0u)
                {
                    const int atom_j = shared_j_atom_ids[lane_j];
                    if (!(cluster_i == cluster_j &&
                          atom_j < local_atom_numbers && lane_j <= i_lane) &&
                        (exclusion_mask &
                         (1ull << (i_lane * cluster_size + lane_j))) == 0ull)
                    {
                        const VECTOR_LJ r2 = Make_Packed_LJ_Atom(
                            shared_j_xq[lane_j], shared_j_lj_type[lane_j]);
                        const VECTOR dr =
                            Get_Periodic_Displacement(r2, r1, cell, rcell);
                        const float dr2 = dr * dr;
                        if (dr2 < cutoff_sq && dr2 != 0.0f)
                        {
                            const float dr_abs = sqrtf(dr2);
                            const int atom_pair_LJ_type =
                                Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                            const float A = LJ_type_A[atom_pair_LJ_type];
                            const float B = LJ_type_B[atom_pair_LJ_type];
                            const float ij_factor =
                                atom_j < local_atom_numbers ? 1.0f : 0.5f;
                            if (need_force)
                            {
                                float frc_abs =
                                    Get_LJ_Force(r1, r2, dr_abs, A, B);
                                if (need_coulomb)
                                {
                                    frc_abs -= Get_Direct_Coulomb_Force(
                                        r1, r2, dr_abs, pme_beta);
                                }
                                const VECTOR frc_lin = frc_abs * dr;
                                frc_i = frc_i + frc_lin;
                                if (shared_j_local_flags[lane_j] != 0)
                                {
                                    j_force_local = j_force_local - frc_lin;
                                }
                                if (need_virial)
                                {
                                    virial = virial -
                                             ij_factor *
                                                 Get_Virial_From_Force_Dis(
                                                     frc_lin, dr);
                                }
                            }
                            if (need_energy)
                            {
                                energy_lj += ij_factor *
                                             Get_LJ_Energy(r1, r2, dr_abs, A, B);
                                if (need_coulomb)
                                {
                                    energy_coulomb +=
                                        ij_factor *
                                        Get_Direct_Coulomb_Energy(
                                            r1, r2, dr_abs, pme_beta);
                                }
                            }
                        }
                    }
                }
                if (need_force)
                {
                    VECTOR reduced = j_force_local;
                    for (int delta = warpSize >> 1; delta > 0; delta >>= 1)
                    {
                        reduced.x +=
                            deviceShflDown(FULL_MASK, reduced.x, delta, warpSize);
                        reduced.y +=
                            deviceShflDown(FULL_MASK, reduced.y, delta, warpSize);
                        reduced.z +=
                            deviceShflDown(FULL_MASK, reduced.z, delta, warpSize);
                    }
                    if (lane == 0)
                    {
                        warp_j_force[warp_id][lane_j] = reduced;
                    }
                }
            }
            if (need_force)
            {
                __syncthreads();
                if (tid < cluster_size &&
                    (shared_j_valid_mask & (1u << tid)) != 0u &&
                    shared_j_local_flags[tid] != 0)
                {
                    VECTOR total = {0.0f, 0.0f, 0.0f};
                    for (int warp_i = 0; warp_i < warp_count; warp_i += 1)
                    {
                        total = total + warp_j_force[warp_i][tid];
                    }
                    atomicAdd(frc + shared_j_atom_ids[tid], total);
                }
                __syncthreads();
            }
        }

        if (active_i)
        {
            if (need_force)
            {
                atomicAdd(frc + atom_i, frc_i);
            }
            if (need_energy)
            {
                atomicAdd(atom_energy + atom_i, energy_lj + energy_coulomb);
                atomicAdd(atom_LJ_ene + atom_i, energy_lj);
                if (need_coulomb)
                {
                    atomicAdd(atom_direct_cf_energy + atom_i, energy_coulomb);
                }
            }
            if (need_virial)
            {
                atomicAdd(atom_virial + atom_i, virial);
            }
        }
#endif
    }
}

#include "clustered_lj_warp_record_kernel.cuh"
#endif

void LENNARD_JONES_INFORMATION::LJ_Malloc()
{
    Malloc_Safely((void**)&h_atom_LJ_type, sizeof(int) * atom_numbers);
    Malloc_Safely((void**)&h_LJ_A, sizeof(float) * pair_type_numbers);
    Malloc_Safely((void**)&h_LJ_B, sizeof(float) * pair_type_numbers);
    Malloc_Safely((void**)&h_LJ_energy_atom, sizeof(float) * atom_numbers);
}

static __global__ void Total_C6_Get(int atom_numbers, int* atom_lj_type,
                                    float* d_lj_b, float* d_factor)
{
    int j;
    double temp_sum = 0;
    int x, y;
    int itype, jtype, atom_pair_LJ_type;
#ifdef USE_GPU
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers;
         i += gridDim.x * blockDim.x)
#else
#pragma omp parallel for firstprivate( \
        j, x, y, itype, jtype, atom_pair_LJ_type) reduction(+ : temp_sum)
    for (int i = 0; i < atom_numbers; i++)
#endif
    {
        itype = atom_lj_type[i];
        double temp_small_sum = 0;
#ifdef USE_GPU
        for (j = blockIdx.y * blockDim.y + threadIdx.y; j < atom_numbers;
             j += gridDim.y * blockDim.y)
#else
        for (j = 0; j < atom_numbers; j++)
#endif
        {
            jtype = atom_lj_type[j];
            y = (jtype - itype);
            x = y >> 31;
            y = (y ^ x) - x;
            x = jtype + itype;
            jtype = (x + y) >> 1;
            x = (x - y) >> 1;
            atom_pair_LJ_type = (jtype * (jtype + 1) >> 1) + x;
            temp_small_sum += d_lj_b[atom_pair_LJ_type];
        }
        temp_sum += temp_small_sum;
    }
    atomicAdd(d_factor, temp_sum);
}

void LENNARD_JONES_INFORMATION::Maybe_Apply_Ordered_Layout(
    CONTROLLER* controller, DOMAIN_INFORMATION* domain, LTMatrix3 cell,
    LTMatrix3 rcell, VECTOR box_length)
{
    (void)cell;
    if (!is_initialized || !use_ordered_layout || ordered_layout_applied ||
        domain == NULL)
    {
        return;
    }
    if (CONTROLLER::PP_MPI_size != 1 || domain->ghost_numbers != 0)
    {
        controller->printf(
            "    Skip LJ ordered layout: only single-rank local domains "
            "without ghosts are supported in this experiment.\n");
        ordered_layout_applied = 1;
        return;
    }
    if (domain->res_numbers < ordered_layout_min_residue_numbers ||
        domain->atom_numbers <= 0)
    {
        controller->printf(
            "    Skip LJ ordered layout: residue count %d is below threshold "
            "%d.\n",
            domain->res_numbers, ordered_layout_min_residue_numbers);
        ordered_layout_applied = 1;
        return;
    }
    if (box_length.x <= 0.0f || box_length.y <= 0.0f || box_length.z <= 0.0f)
    {
        controller->printf(
            "    Skip LJ ordered layout: invalid box lengths (%f, %f, %f).\n",
            box_length.x, box_length.y, box_length.z);
        ordered_layout_applied = 1;
        return;
    }

    std::vector<int> h_atom_local(domain->atom_numbers);
    std::vector<VECTOR> h_crd(domain->atom_numbers);
    std::vector<VECTOR> h_vel(domain->atom_numbers);
    std::vector<float> h_mass(domain->atom_numbers);
    std::vector<float> h_mass_inverse(domain->atom_numbers);
    std::vector<float> h_charge(domain->atom_numbers);
    std::vector<int> h_res_start(domain->res_numbers);
    std::vector<int> h_res_len(domain->res_numbers);

    deviceMemcpy(h_atom_local.data(), domain->atom_local,
                 sizeof(int) * domain->atom_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_crd.data(), domain->crd, sizeof(VECTOR) * domain->atom_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_vel.data(), domain->vel, sizeof(VECTOR) * domain->atom_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_mass.data(), domain->d_mass,
                 sizeof(float) * domain->atom_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_mass_inverse.data(), domain->d_mass_inverse,
                 sizeof(float) * domain->atom_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_charge.data(), domain->d_charge,
                 sizeof(float) * domain->atom_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_res_start.data(), domain->res_start,
                 sizeof(int) * domain->res_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_res_len.data(), domain->res_len,
                 sizeof(int) * domain->res_numbers,
                 deviceMemcpyDeviceToHost);

    std::vector<OrderedResiduePoint> points((size_t)domain->res_numbers);
    std::vector<int> residue_indices((size_t)domain->res_numbers);
    std::iota(residue_indices.begin(), residue_indices.end(), 0);
    for (int residue = 0; residue < domain->res_numbers; residue += 1)
    {
        OrderedResiduePoint point;
        point.residue_index = residue;
        point.atom_start = h_res_start[residue];
        point.atom_count = h_res_len[residue];
        point.wrapped = Wrap_To_Box_Fractional(h_crd[point.atom_start], rcell,
                                              box_length);
        point.normalized = {point.wrapped.x / box_length.x,
                            point.wrapped.y / box_length.y,
                            point.wrapped.z / box_length.z};
        point.point_hilbert =
            Hilbert_Index_3D(point.normalized, ordered_layout_max_depth);
        points[(size_t)residue] = point;
    }

    std::vector<CornerstoneLeaf> leaves;
    Build_Cornerstone_Leaves(points, residue_indices, 0, ordered_layout_max_depth,
                             ordered_layout_leaf_size, {0.0f, 0.0f, 0.0f},
                             {1.0f, 1.0f, 1.0f}, &leaves);
    for (auto& leaf : leaves)
    {
        const VECTOR center = 0.5f * (leaf.min_bound + leaf.max_bound);
        leaf.leaf_hilbert = Hilbert_Index_3D(center, ordered_layout_max_depth);
        std::stable_sort(
            leaf.residues.begin(), leaf.residues.end(),
            [&](int lhs, int rhs)
            {
                const uint64_t key_l = points[(size_t)lhs].point_hilbert;
                const uint64_t key_r = points[(size_t)rhs].point_hilbert;
                if (key_l != key_r)
                {
                    return key_l < key_r;
                }
                return lhs < rhs;
            });
    }
    std::stable_sort(
        leaves.begin(), leaves.end(),
        [](const CornerstoneLeaf& lhs, const CornerstoneLeaf& rhs)
        {
            if (lhs.leaf_hilbert != rhs.leaf_hilbert)
            {
                return lhs.leaf_hilbert < rhs.leaf_hilbert;
            }
            if (lhs.residues.empty() || rhs.residues.empty())
            {
                return lhs.residues.size() < rhs.residues.size();
            }
            return lhs.residues.front() < rhs.residues.front();
        });

    std::vector<int> residue_order;
    residue_order.reserve((size_t)domain->res_numbers);
    for (const auto& leaf : leaves)
    {
        residue_order.insert(residue_order.end(), leaf.residues.begin(),
                             leaf.residues.end());
    }
    if ((int)residue_order.size() != domain->res_numbers)
    {
        controller->printf(
            "    Skip LJ ordered layout: octree produced inconsistent residue "
            "count.\n");
        ordered_layout_applied = 1;
        return;
    }

    bool changed = false;
    for (int residue = 0; residue < domain->res_numbers; residue += 1)
    {
        if (residue_order[(size_t)residue] != residue)
        {
            changed = true;
            break;
        }
    }
    if (!changed)
    {
        controller->printf(
            "    LJ ordered layout leaves the current residue ordering "
            "unchanged.\n");
        ordered_layout_applied = 1;
        return;
    }

    std::vector<int> new_atom_local((size_t)domain->atom_numbers);
    std::vector<VECTOR> new_crd((size_t)domain->atom_numbers);
    std::vector<VECTOR> new_vel((size_t)domain->atom_numbers);
    std::vector<float> new_mass((size_t)domain->atom_numbers);
    std::vector<float> new_mass_inverse((size_t)domain->atom_numbers);
    std::vector<float> new_charge((size_t)domain->atom_numbers);
    std::vector<int> new_res_start((size_t)domain->res_numbers);
    std::vector<int> new_res_len((size_t)domain->res_numbers);
    std::vector<int> new_atom_local_id((size_t)domain->max_atom_numbers, -1);

    int write_atom = 0;
    for (int residue = 0; residue < domain->res_numbers; residue += 1)
    {
        const OrderedResiduePoint& point =
            points[(size_t)residue_order[(size_t)residue]];
        new_res_start[(size_t)residue] = write_atom;
        new_res_len[(size_t)residue] = point.atom_count;
        for (int atom = 0; atom < point.atom_count; atom += 1)
        {
            const int source = point.atom_start + atom;
            const int global_atom = h_atom_local[(size_t)source];
            new_atom_local[(size_t)write_atom] = global_atom;
            new_crd[(size_t)write_atom] = h_crd[(size_t)source];
            new_vel[(size_t)write_atom] = h_vel[(size_t)source];
            new_mass[(size_t)write_atom] = h_mass[(size_t)source];
            new_mass_inverse[(size_t)write_atom] =
                h_mass_inverse[(size_t)source];
            new_charge[(size_t)write_atom] = h_charge[(size_t)source];
            if (global_atom >= 0 &&
                global_atom < static_cast<int>(new_atom_local_id.size()))
            {
                new_atom_local_id[(size_t)global_atom] = write_atom;
            }
            write_atom += 1;
        }
    }

    deviceMemcpy(domain->atom_local, new_atom_local.data(),
                 sizeof(int) * domain->atom_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(domain->crd, new_crd.data(),
                 sizeof(VECTOR) * domain->atom_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(domain->vel, new_vel.data(),
                 sizeof(VECTOR) * domain->atom_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(domain->d_mass, new_mass.data(),
                 sizeof(float) * domain->atom_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(domain->d_mass_inverse, new_mass_inverse.data(),
                 sizeof(float) * domain->atom_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(domain->d_charge, new_charge.data(),
                 sizeof(float) * domain->atom_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(domain->res_start, new_res_start.data(),
                 sizeof(int) * domain->res_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(domain->res_len, new_res_len.data(),
                 sizeof(int) * domain->res_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(domain->atom_local_id, new_atom_local_id.data(),
                 sizeof(int) * domain->max_atom_numbers,
                 deviceMemcpyHostToDevice);

    ordered_layout_applied = 1;
    controller->printf(
        "    Applied LJ ordered layout with %d residues, %zu cornerstone "
        "leaves, depth=%d, leaf_size=%d.\n",
        domain->res_numbers, leaves.size(), ordered_layout_max_depth,
        ordered_layout_leaf_size);
}

void LENNARD_JONES_INFORMATION::Initial(CONTROLLER* controller, float cutoff,
                                        const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "LJ");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    controller->printf("START INITIALIZING LENNADR JONES INFORMATION:\n");
    const auto& lj = Xponge::system.classical_force_field.lj;
    Xponge::LennardJones local_lj;
    const Xponge::LennardJones* lj_to_use = NULL;
    if (module_name == NULL)
    {
        lj_to_use = &lj;
    }
    else if (controller->Command_Exist(this->module_name, "in_file"))
    {
        Xponge::Native_Load_LJ(&local_lj, controller, 0, this->module_name);
        lj_to_use = &local_lj;
    }
    if (lj_to_use != NULL)
    {
        atom_numbers = static_cast<int>(lj_to_use->atom_type.size());
        atom_type_numbers = lj_to_use->atom_type_numbers;
    }
    if (atom_numbers > 0)
    {
        controller->printf("    atom_numbers is %d\n", atom_numbers);
        controller->printf("    atom_LJ_type_number is %d\n",
                           atom_type_numbers);
        pair_type_numbers = atom_type_numbers * (atom_type_numbers + 1) / 2;
        LJ_Malloc();

        for (int i = 0; i < pair_type_numbers; i++)
        {
            h_LJ_A[i] = lj_to_use->pair_A[i];
            h_LJ_B[i] = lj_to_use->pair_B[i];
        }
        for (int i = 0; i < atom_numbers; i++)
        {
            h_atom_LJ_type[i] = lj_to_use->atom_type[i];
        }
        gmxpacked_lj_comb_table_compatible =
            Clustered_Gmxpacked_Lj_Comb_Table_Compatible(
                h_LJ_A, h_LJ_B, atom_type_numbers);
        Parameter_Host_To_Device();
        is_initialized = 1;
    }
    if (is_initialized)
    {
        this->cutoff = cutoff;
        use_ordered_layout = false;
        ordered_layout_applied = 0;
        ordered_layout_max_depth = 6;
        ordered_layout_leaf_size = 32;
        ordered_layout_min_residue_numbers = 256;
        if (controller->Command_Exist(this->module_name, "ordered_layout"))
        {
            use_ordered_layout = controller->Get_Bool(
                this->module_name, "ordered_layout",
                "LENNARD_JONES_INFORMATION::Initial");
        }
        if (controller->Command_Exist(this->module_name,
                                      "ordered_layout_max_depth"))
        {
            controller->Check_Int(this->module_name,
                                  "ordered_layout_max_depth",
                                  "LENNARD_JONES_INFORMATION::Initial");
            ordered_layout_max_depth = atoi(controller->Command(
                this->module_name, "ordered_layout_max_depth"));
        }
        if (controller->Command_Exist(this->module_name,
                                      "ordered_layout_leaf_size"))
        {
            controller->Check_Int(this->module_name,
                                  "ordered_layout_leaf_size",
                                  "LENNARD_JONES_INFORMATION::Initial");
            ordered_layout_leaf_size = atoi(controller->Command(
                this->module_name, "ordered_layout_leaf_size"));
        }
        if (controller->Command_Exist(this->module_name,
                                      "ordered_layout_min_residue_numbers"))
        {
            controller->Check_Int(this->module_name,
                                  "ordered_layout_min_residue_numbers",
                                  "LENNARD_JONES_INFORMATION::Initial");
            ordered_layout_min_residue_numbers = atoi(controller->Command(
                this->module_name,
                "ordered_layout_min_residue_numbers"));
        }
        ordered_layout_max_depth = std::max(1, std::min(ordered_layout_max_depth,
                                                        21));
        ordered_layout_leaf_size = std::max(1, ordered_layout_leaf_size);
        ordered_layout_min_residue_numbers =
            std::max(1, ordered_layout_min_residue_numbers);
        controller->printf("    ordered_layout: %s\n",
                           use_ordered_layout ? "true" : "false");
        if (use_ordered_layout)
        {
            controller->printf(
                "        cornerstone octree depth=%d leaf_size=%d "
                "min_residues=%d\n",
                ordered_layout_max_depth, ordered_layout_leaf_size,
                ordered_layout_min_residue_numbers);
        }
        clustered_direct_cache = Acquire_Shared_LJ_Clustered_Direct_Cache(
            controller, this->module_name, use_ordered_layout);
        clustered_direct_requested =
            clustered_direct_cache != NULL &&
            clustered_direct_cache->Use_Clustered_Direct();
        Device_Malloc_Safely((void**)&crd_with_LJ_parameters,
                             sizeof(VECTOR_LJ) * atom_numbers);
        Launch_Device_Kernel(
            Copy_LJ_Type_To_New_Crd,
            (this->atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, atom_numbers,
            crd_with_LJ_parameters, d_atom_LJ_type);
        controller->printf("    Start initializing long range LJ correction\n");
        long_range_factor = 0;

        Device_Malloc_And_Copy_Safely((void**)&d_long_range_factor,
                                      &long_range_factor, sizeof(float));
        deviceMemset(d_long_range_factor, 0, sizeof(float));

        dim3 gridSize = {(atom_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         1};
        dim3 blockSize = {
            CONTROLLER::device_warp,
            CONTROLLER::device_max_thread / CONTROLLER::device_warp};
        Launch_Device_Kernel(Total_C6_Get, gridSize, blockSize, 0, NULL,
                             atom_numbers, d_atom_LJ_type, d_LJ_B,
                             d_long_range_factor);

        deviceMemcpy(&long_range_factor, d_long_range_factor, sizeof(float),
                     deviceMemcpyDeviceToHost);
        printf("        Total C6 factor is %e\n", long_range_factor);

        long_range_factor *=
            -2.0f / 3.0f * CONSTANT_Pi / cutoff / cutoff / cutoff / 6.0f;
        controller->printf("        long range correction factor is: %e\n",
                           long_range_factor);
        controller->printf("    End initializing long range LJ correction\n");
    }
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller->Step_Print_Initial("LJ_short", "%.2f");
        controller->Step_Print_Initial("LJ_long", "%.2f");
        controller->Step_Print_Initial("LJ", "%.2f");
        is_controller_printf_initialized = 1;
        controller->printf("    structure last modify date is %d\n",
                           last_modify_date);
    }
    controller->printf("END INITIALIZING LENNADR JONES INFORMATION\n\n");
}

static __global__ void get_local_device(int* atom_local, int local_atom_numbers,
                                        int ghost_numbers, int* d_atom_LJ_type,
                                        VECTOR_LJ* crd_with_LJ_parameters_local)
{
    SIMPLE_DEVICE_FOR(i, local_atom_numbers + ghost_numbers)
    {
        int atom_i = atom_local[i];
        crd_with_LJ_parameters_local[i].LJ_type = d_atom_LJ_type[atom_i];
    }
}

void LENNARD_JONES_INFORMATION::Get_Local(int* atom_local,
                                          int local_atom_numbers,
                                          int ghost_numbers)
{
    if (!is_initialized) return;
    this->local_atom_numbers = local_atom_numbers;
    this->ghost_numbers = ghost_numbers;
    Launch_Device_Kernel(get_local_device,
                         (local_atom_numbers + ghost_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, atom_local,
                         local_atom_numbers, ghost_numbers, d_atom_LJ_type,
                         crd_with_LJ_parameters_local);
}

void LENNARD_JONES_INFORMATION::Refresh_Clustered_Metadata(
    int solvent_numbers, const int* d_atom_local,
    const int* d_excluded_list_start,
    const int* d_excluded_list,
    const int* d_excluded_numbers)
{
    if (!is_initialized) return;
    if (clustered_direct_cache != NULL)
    {
        const int capped_solvent_numbers =
            solvent_numbers > 0 ? solvent_numbers : 0;
        const int direct_local_atom_numbers =
            local_atom_numbers > capped_solvent_numbers
                ? (local_atom_numbers - capped_solvent_numbers)
                : 0;
        clustered_direct_cache->Refresh_Metadata(
            local_atom_numbers, direct_local_atom_numbers, ghost_numbers,
            d_atom_local, d_excluded_list_start,
            d_excluded_list, d_excluded_numbers);
    }
}

static __global__ void Long_Range_Virial_Correction(LTMatrix3* d_virial,
                                                    const float factor)
{
    d_virial[0].a11 += factor;
    d_virial[0].a22 += factor;
    d_virial[0].a33 += factor;
}

void LENNARD_JONES_INFORMATION::Long_Range_Correction(int need_pressure,
                                                      LTMatrix3* d_virial,
                                                      int need_potential,
                                                      float* d_potential,
                                                      const float volume)
{
    if (is_initialized && CONTROLLER::PP_MPI_rank == 0)
    {
        if (need_pressure)
        {
            Launch_Device_Kernel(Long_Range_Virial_Correction, 1, 1, 0, 0,
                                 d_virial, 2 * long_range_factor / volume);
        }
        if (need_potential)
        {
            Launch_Device_Kernel(device_add, 1, 1, 0, 0, d_potential,
                                 long_range_factor / volume);

            h_LJ_long_energy = long_range_factor / volume;
        }
    }
}

void LENNARD_JONES_INFORMATION::Parameter_Host_To_Device()
{
    std::vector<float2> h_LJ_AB_packed((size_t)pair_type_numbers);
    std::vector<float2> h_LJ_AB_matrix(
        (size_t)atom_type_numbers * (size_t)atom_type_numbers);
    for (int i = 0; i < pair_type_numbers; i += 1)
    {
        h_LJ_AB_packed[(size_t)i] = {h_LJ_A[i], h_LJ_B[i]};
    }
    for (int i = 0; i < atom_type_numbers; i += 1)
    {
        for (int j = 0; j < atom_type_numbers; j += 1)
        {
            const int pair_type = Get_LJ_Type(i, j);
            h_LJ_AB_matrix[(size_t)i * (size_t)atom_type_numbers + (size_t)j] =
                h_LJ_AB_packed[(size_t)pair_type];
        }
    }
    Device_Malloc_And_Copy_Safely((void**)&d_atom_LJ_type, h_atom_LJ_type,
                                  sizeof(int) * atom_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_A, h_LJ_A,
                                  sizeof(float) * pair_type_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_B, h_LJ_B,
                                  sizeof(float) * pair_type_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_AB_packed,
                                  h_LJ_AB_packed.data(),
                                  sizeof(float2) * pair_type_numbers);
    Device_Malloc_And_Copy_Safely(
        (void**)&d_LJ_AB_matrix, h_LJ_AB_matrix.data(),
        sizeof(float2) * (size_t)atom_type_numbers * (size_t)atom_type_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_energy_sum, h_LJ_energy_atom,
                                  sizeof(float));
    Device_Malloc_Safely((void**)&d_LJ_energy_atom,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&crd_with_LJ_parameters_local,
                         sizeof(VECTOR_LJ) * atom_numbers);
}

void LENNARD_JONES_INFORMATION::LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(
    const int atom_numbers, const int local_atom_numbers,
    const int solvent_numbers, const int ghost_numbers, const VECTOR* crd,
    const float* charge, VECTOR* frc, const LTMatrix3 cell,
    const LTMatrix3 rcell, const ATOM_GROUP* nl, const float pme_beta,
    const int need_atom_energy, float* atom_energy, const int need_virial,
    LTMatrix3* atom_virial, float* atom_direct_pme_energy)
{
    if (is_initialized)
    {
        const bool use_clustered_direct = Use_Clustered_Direct();
        const bool want_full_output_snapshot =
            Clustered_Microbench_Dump_Prefix() != NULL &&
            (need_atom_energy || need_virial) && use_clustered_direct;
        bool have_full_output_snapshot = false;
        nbnxm_microbench::SpongeGmxpackedFullOutputSnapshot
            full_output_snapshot = {};
        std::vector<VECTOR> full_output_force_before;
        std::vector<float> full_output_atom_energy_before;
        std::vector<LTMatrix3> full_output_atom_virial_before;
        CLUSTERED_SPATIAL_VIEW clustered_view = {};
        if (use_clustered_direct)
        {
            if (d_LJ_AB_packed == NULL)
            {
                throw std::runtime_error(
                    "clustered regular LJ requires packed LJ parameters");
            }
#ifdef USE_CPU
            clustered_direct_cache->Build(crd, cell, rcell, cutoff,
                                          need_virial != 0, false, false,
                                          false, false);
#else
            clustered_direct_cache->Build(crd, cell, rcell, cutoff,
                                          need_virial != 0,
                                          false,
                                          true,
                                          false,
                                          true);
#endif
        }
        if (use_clustered_direct &&
            clustered_direct_cache->layout.total_atom_numbers > 0)
        {
            clustered_direct_cache->Gather_Plain(
                crd, charge, crd_with_LJ_parameters_local, cell, rcell,
                d_LJ_AB_packed);
            const char* view_failure_reason = nullptr;
            if (!Make_Clustered_Spatial_View_From_LJ_Cache(
                    clustered_direct_cache, &clustered_view,
                    &view_failure_reason))
            {
                throw std::runtime_error(
                    view_failure_reason != nullptr
                        ? view_failure_reason
                        : "clustered regular LJ spatial view is unavailable");
            }
            CLUSTERED_SPATIAL_VIEW_REQUIREMENTS requirements;
            requirements.local_atom_numbers = local_atom_numbers;
            requirements.ghost_numbers = ghost_numbers;
            requirements.cutoff = cutoff;
            requirements.provider_incarnation =
                clustered_direct_cache->layout.provider_incarnation;
            requirements.lease_epoch =
                clustered_direct_cache->layout.spatial_view_lease_epoch;
            requirements.native_payload_generation =
                clustered_direct_cache->layout.native_payload_generation;
            requirements.gmxpacked_payload_generation =
                clustered_direct_cache->layout
                    .gmxpacked_compact_payload_generation;
            requirements.geometry_generation =
                clustered_direct_cache->layout.geometry_generation;
            requirements.require_all_local_atoms = true;
#if defined(USE_CUDA)
            requirements.require_backend = true;
            requirements.backend = CLUSTERED_SPATIAL_BACKEND::CUDA;
            requirements.require_same_producer_stream = true;
            requirements.consumer_stream = nullptr;
            requirements.require_gmxpacked_payload = true;
            requirements.require_pair_shift_metadata = true;
            requirements.require_pair_shift_rcell = true;
            requirements.pair_shift_rcell = rcell;
#elif defined(USE_HIP)
            requirements.require_backend = true;
            requirements.backend = CLUSTERED_SPATIAL_BACKEND::HIP;
            requirements.require_same_producer_stream = true;
            requirements.consumer_stream = nullptr;
            requirements.require_gmxpacked_payload = true;
            requirements.require_pair_shift_metadata = true;
            requirements.require_pair_shift_rcell = true;
            requirements.pair_shift_rcell = rcell;
#else
            requirements.require_backend = true;
            requirements.backend = CLUSTERED_SPATIAL_BACKEND::CPU;
            requirements.require_native_payload = true;
#endif
            if (!Clustered_Validate_Spatial_View(
                    clustered_view, requirements, &view_failure_reason))
            {
                throw std::runtime_error(
                    view_failure_reason != nullptr
                        ? view_failure_reason
                        : "clustered regular LJ spatial view is invalid");
            }
            const bool dump_use_gmxpacked_lj_comb_kernel =
                gmxpacked_lj_comb_table_compatible;
            Compare_Gmxpacked_Record_Stream_Focus_Pair_Forces(
                clustered_direct_cache, d_LJ_AB_packed,
                static_cast<size_t>(pair_type_numbers), cutoff, pme_beta, cell,
                rcell);
            if (want_full_output_snapshot)
            {
                nbnxm_microbench::SpongeGmxpackedForceOnlySnapshot
                    full_output_payload = {};
                const bool have_full_output_payload =
                    Maybe_Dump_Clustered_Gmxpacked_Microbench_Diagnostic_Snapshot(
                        clustered_direct_cache, d_LJ_AB_packed,
                        static_cast<size_t>(pair_type_numbers), cutoff,
                        pme_beta, cell,
                        dump_use_gmxpacked_lj_comb_kernel, 0,
                        &full_output_payload);
                have_full_output_snapshot = have_full_output_payload &&
                    Capture_Clustered_Microbench_Full_Output_Diagnostic_View(
                        std::move(full_output_payload),
                        &full_output_snapshot);
            }
            else
            {
                Maybe_Dump_Clustered_Gmxpacked_Microbench_Diagnostic_Snapshot(
                    clustered_direct_cache, d_LJ_AB_packed,
                    static_cast<size_t>(pair_type_numbers), cutoff, pme_beta,
                    cell, dump_use_gmxpacked_lj_comb_kernel);
                Maybe_Dump_Clustered_Microbench_Diagnostic_Snapshot(
                    clustered_direct_cache, d_LJ_AB_packed,
                    static_cast<size_t>(pair_type_numbers), cutoff, pme_beta,
                    cell, dump_use_gmxpacked_lj_comb_kernel);
            }
        }
        else if (!use_clustered_direct)
        {
            Launch_Device_Kernel(
                Copy_Crd_And_Charge_To_New_Crd,
                (this->atom_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL,
                this->local_atom_numbers + this->ghost_numbers, crd,
                crd_with_LJ_parameters_local, charge);
        }
        if (need_atom_energy)
        {
            deviceMemset(atom_direct_pme_energy, 0,
                         sizeof(float) * this->atom_numbers);
            deviceMemset(d_LJ_energy_atom, 0,
                         sizeof(float) * this->atom_numbers);
        }

        if (atom_numbers == 0 || local_atom_numbers == 0) return;

        if (use_clustered_direct)
        {
            auto& clustered_layout = clustered_direct_cache->layout;
            const bool clustered_gather_ready =
                clustered_direct_cache->Coordinate_Gather_Ready_For_Current_Step();
            if (!clustered_gather_ready)
                return;
#ifdef USE_CPU
            if (clustered_view.sci_numbers <= 0 ||
                clustered_view.cjpacked_numbers <= 0 ||
                clustered_view.sci == NULL ||
                clustered_view.cjpacked == NULL)
            {
                return;
            }
            auto cpu_f =
                Cpu_Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb<false>;
            if (need_atom_energy || need_virial)
            {
                cpu_f =
                    Cpu_Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb<true>;
            }
            if (clustered_direct_cache->direct_kernel_time_recorder != NULL)
            {
                clustered_direct_cache->direct_kernel_time_recorder->Start();
            }
            cpu_f(
                clustered_view.sci_numbers, clustered_view.cluster_size,
                clustered_view.super_cluster_clusters, local_atom_numbers,
                clustered_view.cluster_offsets,
                clustered_view.cluster_valid_masks,
                clustered_view.cluster_local_masks,
                clustered_view.cluster_centers,
                clustered_view.super_cluster_offsets,
                clustered_view.sci,
                clustered_view.cjpacked,
                clustered_view.exclusion_mask_pool,
                clustered_direct_cache->d_sorted_atom_ids,
                clustered_direct_cache->d_sorted_xq,
                clustered_direct_cache->d_sorted_lj_type, cell, rcell,
                d_LJ_A, d_LJ_B, cutoff, frc, pme_beta, atom_energy,
                atom_virial, atom_direct_pme_energy, d_LJ_energy_atom,
                need_atom_energy != 0, need_virial != 0);
            if (clustered_direct_cache->direct_kernel_time_recorder != NULL)
            {
                clustered_direct_cache->direct_kernel_time_recorder->Stop();
            }
            return;
#endif
#ifndef USE_CPU
            dim3 blockSize = {
                static_cast<unsigned int>(clustered_view.cluster_size),
                static_cast<unsigned int>(clustered_view.cluster_size), 1u};
            VECTOR* clustered_force_target = frc;
#ifndef USE_CPU
            const bool requested_gmxpacked_lj_comb_kernel =
                true;
            const bool requested_gmxpacked_fast_kernel =
                true;
            const bool requested_gmxpacked_assume_sci_shift =
                false;
            const bool requested_gmxpacked_sci_shift_split =
                true;
            const bool requested_gmxpacked_sci_shift_runtime =
                false;
            const bool use_gmxpacked_lj_comb_kernel =
                requested_gmxpacked_lj_comb_kernel &&
                gmxpacked_lj_comb_table_compatible;
            static bool warned_gmxpacked_lj_comb_incompatible = false;
            if (requested_gmxpacked_lj_comb_kernel &&
                !gmxpacked_lj_comb_table_compatible &&
                !warned_gmxpacked_lj_comb_incompatible)
            {
                fprintf(stderr,
                        "[clustered gmxpacked lj comb] requested but LJ pair "
                        "table is not compatible with geometric comb; using "
                        "AB-table parameter path\n");
                fflush(stderr);
                warned_gmxpacked_lj_comb_incompatible = true;
            }
            const bool has_sorted_force_scratch =
                clustered_direct_cache->d_sorted_frc != NULL;
            const bool has_gmxpacked_payload =
                clustered_view.gmxpacked_sci_numbers > 0 &&
                clustered_view.gmxpacked_cjpacked_numbers > 0 &&
                clustered_view.gmxpacked_exclusion_numbers > 0 &&
                clustered_view.gmxpacked_sci != NULL &&
                clustered_view.gmxpacked_cjpacked != NULL &&
                clustered_view.gmxpacked_exclusions != NULL &&
                clustered_direct_cache->d_sorted_atom_ids != NULL &&
                clustered_direct_cache->d_sorted_xq != NULL &&
                clustered_direct_cache->d_sorted_lj_type != NULL &&
                (!use_gmxpacked_lj_comb_kernel ||
                 clustered_direct_cache->d_sorted_lj_comb != NULL) &&
                d_LJ_AB_packed != NULL;
            const bool gmxpacked_needs_compact_force_scratch =
                need_atom_energy || need_virial;
            const bool use_gmxpacked_direct =
                has_gmxpacked_payload &&
                (!gmxpacked_needs_compact_force_scratch ||
                 has_sorted_force_scratch);
            const bool gmxpacked_fast_layout_compatible =
                clustered_view.cluster_size == kClusteredClusterSize &&
                clustered_view.super_cluster_clusters ==
                    kClusteredSuperClusterClusters &&
                clustered_view.cluster_numbers > 0;
            const bool use_gmxpacked_fast_kernel =
                use_gmxpacked_direct && requested_gmxpacked_fast_kernel &&
                gmxpacked_fast_layout_compatible;
            const bool gmxpacked_fast_full_local_dense_compatible =
                use_gmxpacked_fast_kernel &&
                clustered_view.ghost_numbers == 0 &&
                clustered_view.local_atom_numbers ==
                    clustered_view.total_atom_numbers &&
                clustered_view.direct_local_atom_numbers ==
                    clustered_view.total_atom_numbers &&
                clustered_view.padded_total_atom_numbers ==
                    clustered_view.cluster_numbers * kClusteredClusterSize &&
                clustered_view.cluster_numbers %
                        kClusteredSuperClusterClusters ==
                    0;
            const bool use_gmxpacked_sci_shift_only =
                requested_gmxpacked_assume_sci_shift &&
                gmxpacked_fast_full_local_dense_compatible &&
                clustered_layout.gmxpacked_pair_shift_sci_only_compatible;
            const bool use_gmxpacked_sci_shift_split =
                requested_gmxpacked_sci_shift_split &&
                use_gmxpacked_fast_kernel &&
                clustered_view.gmxpacked_sci_shift_safe_flags != NULL;
            const bool use_gmxpacked_sci_shift_split_skip_empty =
                use_gmxpacked_sci_shift_split;
            const bool gmxpacked_sci_shift_split_counts_valid =
                use_gmxpacked_sci_shift_split_skip_empty &&
                clustered_layout.gmxpacked_pair_shift_sci_safe_counts_ready &&
                clustered_layout.gmxpacked_pair_shift_safe_sci_numbers >= 0 &&
                clustered_layout.gmxpacked_pair_shift_unsafe_sci_numbers >= 0 &&
                clustered_layout.gmxpacked_pair_shift_safe_sci_numbers +
                        clustered_layout.gmxpacked_pair_shift_unsafe_sci_numbers ==
                    clustered_view.gmxpacked_sci_numbers;
            const bool gmxpacked_sci_shift_split_has_safe =
                !gmxpacked_sci_shift_split_counts_valid ||
                clustered_layout.gmxpacked_pair_shift_safe_sci_numbers > 0;
            const bool gmxpacked_sci_shift_split_has_unsafe =
                !gmxpacked_sci_shift_split_counts_valid ||
                clustered_layout.gmxpacked_pair_shift_unsafe_sci_numbers > 0;
            const bool use_gmxpacked_sci_shift_runtime =
                requested_gmxpacked_sci_shift_runtime &&
                use_gmxpacked_fast_kernel &&
                clustered_view.gmxpacked_sci_shift_safe_flags != NULL;
            static bool warned_gmxpacked_sci_shift_unsafe = false;
            if (requested_gmxpacked_assume_sci_shift &&
                gmxpacked_fast_full_local_dense_compatible &&
                !use_gmxpacked_sci_shift_split &&
                !clustered_layout.gmxpacked_pair_shift_sci_only_compatible &&
                !warned_gmxpacked_sci_shift_unsafe)
            {
                fprintf(stderr,
                        "[clustered gmxpacked sci-shift] requested but pair "
                        "shift metadata is not sci-uniform; falling back to "
                        "pair-shift metadata\n");
                fflush(stderr);
                warned_gmxpacked_sci_shift_unsafe = true;
            }
            const uint64_t* gmxpacked_pair_shift_bits =
                use_gmxpacked_sci_shift_only ? NULL
                                             : clustered_view.pair_shift_bits;
            static bool warned_gmxpacked_fast_unavailable = false;
            if (use_gmxpacked_direct && requested_gmxpacked_fast_kernel &&
                !use_gmxpacked_fast_kernel &&
                !warned_gmxpacked_fast_unavailable)
            {
                fprintf(stderr,
                        "[clustered gmxpacked fast] requested but requires "
                        "dense %dx%d gmxpacked layout "
                        "(cluster_size=%d super_cluster_clusters=%d "
                        "lj_comb=%d); falling back to regular gmxpacked "
                        "kernel\n",
                        kClusteredClusterSize, kClusteredSuperClusterClusters,
                        clustered_view.cluster_size,
                        clustered_view.super_cluster_clusters,
                        use_gmxpacked_lj_comb_kernel ? 1 : 0);
                fflush(stderr);
                warned_gmxpacked_fast_unavailable = true;
            }
            const bool use_gmxpacked_compact_force_scratch =
                use_gmxpacked_direct && gmxpacked_needs_compact_force_scratch;
            const float2* gmxpacked_LJ_AB_table = d_LJ_AB_packed;
            const bool use_sorted_force_scratch =
                use_gmxpacked_compact_force_scratch;
            const int clustered_force_scratch_slot_numbers =
                std::max(clustered_view.total_atom_numbers,
                         clustered_view.padded_total_atom_numbers);
            if (have_full_output_snapshot)
            {
                const size_t total_atom_numbers_snapshot = static_cast<size_t>(
                    clustered_view.total_atom_numbers);
                const size_t scalar_output_numbers = total_atom_numbers_snapshot;
                full_output_snapshot.header.compute_energy =
                    need_atom_energy ? 1u : 0u;
                full_output_snapshot.header.compute_virial =
                    need_virial ? 1u : 0u;
                full_output_snapshot.header.force_soa = 0u;
                full_output_snapshot.header.total_output = 0u;
                full_output_force_before = Copy_Device_Vector_To_Host(
                    frc, total_atom_numbers_snapshot);
                if (need_atom_energy)
                {
                    full_output_atom_energy_before = Copy_Device_Vector_To_Host(
                        atom_energy, scalar_output_numbers);
                }
                if (need_virial)
                {
                    full_output_atom_virial_before = Copy_Device_Vector_To_Host(
                        atom_virial, scalar_output_numbers);
                }
            }
            if (use_sorted_force_scratch)
            {
                if (use_gmxpacked_compact_force_scratch &&
                    clustered_direct_cache
                            ->gmxpacked_force_scratch_memset_time_recorder !=
                        NULL)
                {
                    clustered_direct_cache
                        ->gmxpacked_force_scratch_memset_time_recorder
                        ->Start();
                }
                deviceMemset(clustered_direct_cache->d_sorted_frc, 0,
                             sizeof(VECTOR) *
                                 clustered_force_scratch_slot_numbers);
                if (use_gmxpacked_compact_force_scratch &&
                    clustered_direct_cache
                            ->gmxpacked_force_scratch_memset_time_recorder !=
                        NULL)
                {
                    clustered_direct_cache
                        ->gmxpacked_force_scratch_memset_time_recorder
                        ->Stop();
                }
                clustered_force_target = clustered_direct_cache->d_sorted_frc;
            }
#else
            const bool use_gmxpacked_direct = false;
            const bool use_gmxpacked_lj_comb_kernel = false;
            const bool use_gmxpacked_fast_kernel = false;
            const bool gmxpacked_fast_full_local_dense_compatible = false;
            const bool use_gmxpacked_sci_shift_only = false;
            const bool use_gmxpacked_sci_shift_split = false;
            const bool use_gmxpacked_sci_shift_runtime = false;
            const bool use_gmxpacked_compact_force_scratch = false;
            const float2* gmxpacked_LJ_AB_table = d_LJ_AB_packed;
            const uint64_t* gmxpacked_pair_shift_bits = NULL;
            const bool use_sorted_force_scratch = false;
#endif
            if (!use_gmxpacked_direct)
            {
                throw std::runtime_error(
                    "clustered regular LJ gmxpacked payload is unavailable");
            }
            if (clustered_direct_cache->direct_kernel_time_recorder != NULL)
            {
                clustered_direct_cache->direct_kernel_time_recorder->Start();
            }
            if (use_gmxpacked_direct)
            {
                dim3 gmxpackedGridSize = {
                    static_cast<unsigned int>(
                        clustered_view.gmxpacked_sci_numbers),
                    1u, 1u};
                if (clustered_direct_cache->gmxpacked_kernel_launch_time_recorder !=
                    NULL)
                {
                    clustered_direct_cache->gmxpacked_kernel_launch_time_recorder
                        ->Start();
                }
                if (gmxpacked_fast_full_local_dense_compatible)
                {
                    if (use_gmxpacked_sci_shift_runtime)
                    {
                        const int* sci_shift_flags =
                            clustered_view.gmxpacked_sci_shift_safe_flags;
#define CLUSTERED_GMXPACKED_RUNTIME_FULL_DENSE_KERNEL(NEED_ENERGY, NEED_VIRIAL, COMPACT_FORCE) \
    (use_gmxpacked_lj_comb_kernel                                                                                                   \
	         ? Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<                                           \
	               NEED_ENERGY, NEED_VIRIAL, false, COMPACT_FORCE, true, true, true, false, VECTOR, true>                               \
	         : Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<                                           \
	               NEED_ENERGY, NEED_VIRIAL, false, COMPACT_FORCE, false, true, true, false, VECTOR, true>)
                            auto gmxpacked_runtime_f =
                                CLUSTERED_GMXPACKED_RUNTIME_FULL_DENSE_KERNEL(
                                    false, false, false);
                            if (need_atom_energy || need_virial)
                            {
                                gmxpacked_runtime_f =
                                    CLUSTERED_GMXPACKED_RUNTIME_FULL_DENSE_KERNEL(
                                        true, true, true);
                            }
#undef CLUSTERED_GMXPACKED_RUNTIME_FULL_DENSE_KERNEL
                            Launch_Device_Kernel(
                                gmxpacked_runtime_f, gmxpackedGridSize,
                                blockSize, 0, NULL,
                                clustered_view.gmxpacked_sci_numbers,
                                clustered_view.cluster_size,
                                clustered_view.super_cluster_clusters,
                                clustered_view.cluster_numbers,
                                clustered_view.cluster_offsets,
                                clustered_view.cluster_valid_masks,
                                clustered_view.cluster_local_masks,
                                clustered_view.super_cluster_offsets,
                                clustered_view.gmxpacked_sci,
                                clustered_view.gmxpacked_cjpacked,
                                clustered_view.gmxpacked_exclusions,
                                clustered_view.pair_shift_bits,
                                sci_shift_flags, 0,
                                clustered_direct_cache->d_sorted_atom_ids,
                                clustered_direct_cache->d_sorted_xq,
                                clustered_direct_cache->d_sorted_lj_type,
                                clustered_direct_cache->d_sorted_lj_comb, cell,
                                gmxpacked_LJ_AB_table, cutoff,
                                clustered_force_target,
                                pme_beta, atom_energy, atom_virial,
                                atom_direct_pme_energy, d_LJ_energy_atom,
                                need_atom_energy != 0, need_virial != 0);
                    }
                    else if (use_gmxpacked_sci_shift_split)
                    {
                        const int* sci_shift_flags =
                            clustered_view.gmxpacked_sci_shift_safe_flags;
                        const int* fast_sci_shift_flags =
                            gmxpacked_sci_shift_split_counts_valid &&
                                    !gmxpacked_sci_shift_split_has_unsafe
                                ? NULL
                                : sci_shift_flags;
                        constexpr unsigned int kAbForceOnlySciWorkParts = 4u;
                        constexpr unsigned int kAbFullOutputSciWorkParts = 4u;
                        const bool use_ab_force_only_partition =
                            !use_gmxpacked_lj_comb_kernel &&
                            !need_atom_energy && !need_virial;
                        const bool use_ab_full_output_partition =
                            !use_gmxpacked_lj_comb_kernel &&
                            (need_atom_energy || need_virial);
                        const unsigned int gmxpacked_sci_work_parts =
                            use_ab_force_only_partition
                                ? kAbForceOnlySciWorkParts
                                : (use_ab_full_output_partition
                                       ? kAbFullOutputSciWorkParts
                                       : 1u);
                        const dim3 gmxpacked_sci_shift_split_grid_size = {
                            static_cast<unsigned int>(
                                clustered_view.gmxpacked_sci_numbers) *
                                gmxpacked_sci_work_parts,
                            1u, 1u};
#define CLUSTERED_GMXPACKED_SPLIT_FULL_DENSE_KERNEL(NEED_ENERGY, NEED_VIRIAL, COMPACT_FORCE, SCI_SHIFT_ONLY) \
    (use_gmxpacked_lj_comb_kernel                                                                                                         \
	         ? Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<                                                  \
	               NEED_ENERGY, NEED_VIRIAL, false, COMPACT_FORCE, true, true, true, SCI_SHIFT_ONLY>                                         \
	         : Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<                                                  \
	               NEED_ENERGY, NEED_VIRIAL, false, COMPACT_FORCE, false, true, true, SCI_SHIFT_ONLY>)
                            auto gmxpacked_fast_f =
                                CLUSTERED_GMXPACKED_SPLIT_FULL_DENSE_KERNEL(
                                    false, false, false, true);
                            auto gmxpacked_slow_f =
                                CLUSTERED_GMXPACKED_SPLIT_FULL_DENSE_KERNEL(
                                    false, false, false, false);
                            if (use_ab_force_only_partition)
                            {
                                gmxpacked_fast_f =
                                    Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                                        false, false, false, false, false,
                                        true, true, true, VECTOR, false, 4,
                                        true>;
                                gmxpacked_slow_f =
                                    Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                                        false, false, false, false, false,
                                        true, true, false, VECTOR, false, 4,
                                        true>;
                            }
                            else if (use_ab_full_output_partition)
                            {
                                gmxpacked_fast_f =
                                    Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                                        true, true, false, true, false, true,
                                        true, true, VECTOR, false, 4, false>;
                                gmxpacked_slow_f =
                                    Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                                        true, true, false, true, false, true,
                                        true, false, VECTOR, false, 4, false>;
                            }
                            else if (need_atom_energy || need_virial)
                            {
                                gmxpacked_fast_f =
                                    CLUSTERED_GMXPACKED_SPLIT_FULL_DENSE_KERNEL(
                                        true, true, true, true);
                                gmxpacked_slow_f =
                                    CLUSTERED_GMXPACKED_SPLIT_FULL_DENSE_KERNEL(
                                        true, true, true, false);
                            }
#undef CLUSTERED_GMXPACKED_SPLIT_FULL_DENSE_KERNEL
                            if (gmxpacked_sci_shift_split_has_safe)
                            {
                                Launch_Device_Kernel(
                                    gmxpacked_fast_f,
                                    gmxpacked_sci_shift_split_grid_size,
                                    blockSize, 0, NULL,
                                    clustered_view.gmxpacked_sci_numbers,
                                    clustered_view.cluster_size,
                                    clustered_view.super_cluster_clusters,
                                    clustered_view.cluster_numbers,
                                    clustered_view.cluster_offsets,
                                    clustered_view.cluster_valid_masks,
                                    clustered_view.cluster_local_masks,
                                    clustered_view.super_cluster_offsets,
                                    clustered_view.gmxpacked_sci,
                                    clustered_view.gmxpacked_cjpacked,
                                    clustered_view.gmxpacked_exclusions,
                                    NULL, fast_sci_shift_flags, 1,
                                    clustered_direct_cache->d_sorted_atom_ids,
                                    clustered_direct_cache->d_sorted_xq,
                                    clustered_direct_cache->d_sorted_lj_type,
                                    clustered_direct_cache->d_sorted_lj_comb,
                                    cell, gmxpacked_LJ_AB_table, cutoff,
                                    clustered_force_target, pme_beta,
                                    atom_energy, atom_virial,
                                    atom_direct_pme_energy, d_LJ_energy_atom,
                                    need_atom_energy != 0, need_virial != 0);
                            }
                            if (gmxpacked_sci_shift_split_has_unsafe)
                            {
                                Launch_Device_Kernel(
                                    gmxpacked_slow_f,
                                    gmxpacked_sci_shift_split_grid_size,
                                    blockSize, 0, NULL,
                                    clustered_view.gmxpacked_sci_numbers,
                                    clustered_view.cluster_size,
                                    clustered_view.super_cluster_clusters,
                                    clustered_view.cluster_numbers,
                                    clustered_view.cluster_offsets,
                                    clustered_view.cluster_valid_masks,
                                    clustered_view.cluster_local_masks,
                                    clustered_view.super_cluster_offsets,
                                    clustered_view.gmxpacked_sci,
                                    clustered_view.gmxpacked_cjpacked,
                                    clustered_view.gmxpacked_exclusions,
                                    clustered_view.pair_shift_bits,
                                    sci_shift_flags, 0,
                                    clustered_direct_cache->d_sorted_atom_ids,
                                    clustered_direct_cache->d_sorted_xq,
                                    clustered_direct_cache->d_sorted_lj_type,
                                    clustered_direct_cache->d_sorted_lj_comb,
                                    cell, gmxpacked_LJ_AB_table, cutoff,
                                    clustered_force_target, pme_beta,
                                    atom_energy, atom_virial,
                                    atom_direct_pme_energy, d_LJ_energy_atom,
                                    need_atom_energy != 0, need_virial != 0);
                            }
                    }
                    else
                    {
#define CLUSTERED_GMXPACKED_FULL_DENSE_KERNEL(NEED_ENERGY, NEED_VIRIAL, COMPACT_FORCE) \
    (use_gmxpacked_lj_comb_kernel                                                                                                         \
         ? (use_gmxpacked_sci_shift_only                                                                                                  \
                ? Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<                                          \
	                      NEED_ENERGY, NEED_VIRIAL, false, COMPACT_FORCE, true, true, true, true>                                             \
	                : Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<                                          \
	                      NEED_ENERGY, NEED_VIRIAL, false, COMPACT_FORCE, true, true, true, false>)                                           \
	         : (use_gmxpacked_sci_shift_only                                                                                                  \
	                ? Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<                                          \
	                      NEED_ENERGY, NEED_VIRIAL, false, COMPACT_FORCE, false, true, true, true>                                             \
	                : Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<                                          \
	                      NEED_ENERGY, NEED_VIRIAL, false, COMPACT_FORCE, false, true, true, false>))
                    auto gmxpacked_f =
                        CLUSTERED_GMXPACKED_FULL_DENSE_KERNEL(
                            false, false, false);
                    if (need_atom_energy || need_virial)
                    {
                        gmxpacked_f =
                            CLUSTERED_GMXPACKED_FULL_DENSE_KERNEL(true, true,
                                                                   true);
                    }
#undef CLUSTERED_GMXPACKED_FULL_DENSE_KERNEL
                    Launch_Device_Kernel(
                        gmxpacked_f, gmxpackedGridSize, blockSize, 0, NULL,
                        clustered_view.gmxpacked_sci_numbers,
                        clustered_view.cluster_size,
                        clustered_view.super_cluster_clusters,
                        clustered_view.cluster_numbers,
                        clustered_view.cluster_offsets,
                        clustered_view.cluster_valid_masks,
                        clustered_view.cluster_local_masks,
                        clustered_view.super_cluster_offsets,
                        clustered_view.gmxpacked_sci,
                        clustered_view.gmxpacked_cjpacked,
                        clustered_view.gmxpacked_exclusions,
                        gmxpacked_pair_shift_bits, NULL, 0,
                        clustered_direct_cache->d_sorted_atom_ids,
                        clustered_direct_cache->d_sorted_xq,
                        clustered_direct_cache->d_sorted_lj_type,
                        clustered_direct_cache->d_sorted_lj_comb, cell,
                        gmxpacked_LJ_AB_table, cutoff, clustered_force_target,
                        pme_beta, atom_energy, atom_virial,
                        atom_direct_pme_energy, d_LJ_energy_atom,
                        need_atom_energy != 0, need_virial != 0);
                    }
                }
                else if (use_gmxpacked_fast_kernel)
                {
                    constexpr unsigned int kAbSciWorkParts = 4u;
                    const dim3 gmxpacked_dense_offset_grid_size = {
                        static_cast<unsigned int>(
                            clustered_view.gmxpacked_sci_numbers) *
                            (use_gmxpacked_lj_comb_kernel
                                 ? 1u
                                 : kAbSciWorkParts),
                        1u, 1u};
                    if (use_gmxpacked_sci_shift_split)
                    {
                        const int* sci_shift_flags =
                            clustered_view.gmxpacked_sci_shift_safe_flags;
                        const int* fast_sci_shift_flags =
                            gmxpacked_sci_shift_split_counts_valid &&
                                    !gmxpacked_sci_shift_split_has_unsafe
                                ? NULL
                                : sci_shift_flags;
#define CLUSTERED_GMXPACKED_SPLIT_DENSE_OFFSET_KERNEL(NEED_ENERGY, NEED_VIRIAL, TOTAL_OUTPUT, COMPACT_FORCE, SCI_SHIFT_ONLY) \
	    (use_gmxpacked_lj_comb_kernel                                                                                            \
	         ? Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<                                   \
	               NEED_ENERGY, NEED_VIRIAL, TOTAL_OUTPUT, COMPACT_FORCE, true, true, false, SCI_SHIFT_ONLY>                     \
	         : Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<                                   \
	               NEED_ENERGY, NEED_VIRIAL, TOTAL_OUTPUT, COMPACT_FORCE, false, true, false, SCI_SHIFT_ONLY, VECTOR, false, 4>)
                        auto gmxpacked_fast_f =
                            CLUSTERED_GMXPACKED_SPLIT_DENSE_OFFSET_KERNEL(
                                false, false, false, false, true);
                        auto gmxpacked_slow_f =
                            CLUSTERED_GMXPACKED_SPLIT_DENSE_OFFSET_KERNEL(
                                false, false, false, false, false);
                        if (need_atom_energy || need_virial)
                        {
                            gmxpacked_fast_f =
                                CLUSTERED_GMXPACKED_SPLIT_DENSE_OFFSET_KERNEL(
                                    true, true, false, true, true);
                            gmxpacked_slow_f =
                                CLUSTERED_GMXPACKED_SPLIT_DENSE_OFFSET_KERNEL(
                                    true, true, false, true, false);
                        }
#undef CLUSTERED_GMXPACKED_SPLIT_DENSE_OFFSET_KERNEL
                        if (gmxpacked_sci_shift_split_has_safe)
                        {
                            Launch_Device_Kernel(
                                gmxpacked_fast_f,
                                gmxpacked_dense_offset_grid_size, blockSize,
                                0, NULL,
                                clustered_view.gmxpacked_sci_numbers,
                                clustered_view.cluster_size,
                                clustered_view.super_cluster_clusters,
                                clustered_view.cluster_numbers,
                                clustered_view.cluster_offsets,
                                clustered_view.cluster_valid_masks,
                                clustered_view.cluster_local_masks,
                                clustered_view.super_cluster_offsets,
                                clustered_view.gmxpacked_sci,
                                clustered_view.gmxpacked_cjpacked,
                                clustered_view.gmxpacked_exclusions,
                                NULL, fast_sci_shift_flags, 1,
                                clustered_direct_cache->d_sorted_atom_ids,
                                clustered_direct_cache->d_sorted_xq,
                                clustered_direct_cache->d_sorted_lj_type,
                                clustered_direct_cache->d_sorted_lj_comb, cell,
                                gmxpacked_LJ_AB_table, cutoff,
                                clustered_force_target,
                                pme_beta, atom_energy, atom_virial,
                                atom_direct_pme_energy, d_LJ_energy_atom,
                                need_atom_energy != 0, need_virial != 0);
                        }
                        if (gmxpacked_sci_shift_split_has_unsafe)
                        {
                            Launch_Device_Kernel(
                                gmxpacked_slow_f,
                                gmxpacked_dense_offset_grid_size, blockSize,
                                0, NULL,
                                clustered_view.gmxpacked_sci_numbers,
                                clustered_view.cluster_size,
                                clustered_view.super_cluster_clusters,
                                clustered_view.cluster_numbers,
                                clustered_view.cluster_offsets,
                                clustered_view.cluster_valid_masks,
                                clustered_view.cluster_local_masks,
                                clustered_view.super_cluster_offsets,
                                clustered_view.gmxpacked_sci,
                                clustered_view.gmxpacked_cjpacked,
                                clustered_view.gmxpacked_exclusions,
                                clustered_view.pair_shift_bits,
                                sci_shift_flags, 0,
                                clustered_direct_cache->d_sorted_atom_ids,
                                clustered_direct_cache->d_sorted_xq,
                                clustered_direct_cache->d_sorted_lj_type,
                                clustered_direct_cache->d_sorted_lj_comb, cell,
                                gmxpacked_LJ_AB_table, cutoff,
                                clustered_force_target,
                                pme_beta, atom_energy, atom_virial,
                                atom_direct_pme_energy, d_LJ_energy_atom,
                                need_atom_energy != 0, need_virial != 0);
                        }
                    }
                    else
                    {
#define CLUSTERED_GMXPACKED_DENSE_OFFSET_KERNEL(NEED_ENERGY, NEED_VIRIAL, TOTAL_OUTPUT, COMPACT_FORCE) \
	    (use_gmxpacked_lj_comb_kernel                                                                       \
	         ? Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<              \
	               NEED_ENERGY, NEED_VIRIAL, TOTAL_OUTPUT, COMPACT_FORCE, true, true, false, false>         \
	         : Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<              \
	               NEED_ENERGY, NEED_VIRIAL, TOTAL_OUTPUT, COMPACT_FORCE, false, true, false, false, VECTOR, false, 4>)
                    auto gmxpacked_f =
                        CLUSTERED_GMXPACKED_DENSE_OFFSET_KERNEL(
                            false, false, false, false);
                    if (need_atom_energy || need_virial)
                    {
                        gmxpacked_f =
                            CLUSTERED_GMXPACKED_DENSE_OFFSET_KERNEL(
                                true, true, false, true);
                    }
#undef CLUSTERED_GMXPACKED_DENSE_OFFSET_KERNEL
                    Launch_Device_Kernel(
                        gmxpacked_f, gmxpacked_dense_offset_grid_size,
                        blockSize, 0, NULL,
                        clustered_view.gmxpacked_sci_numbers,
                        clustered_view.cluster_size,
                        clustered_view.super_cluster_clusters,
                        clustered_view.cluster_numbers,
                        clustered_view.cluster_offsets,
                        clustered_view.cluster_valid_masks,
                        clustered_view.cluster_local_masks,
                        clustered_view.super_cluster_offsets,
                        clustered_view.gmxpacked_sci,
                        clustered_view.gmxpacked_cjpacked,
                        clustered_view.gmxpacked_exclusions,
                        gmxpacked_pair_shift_bits, NULL, 0,
                        clustered_direct_cache->d_sorted_atom_ids,
                        clustered_direct_cache->d_sorted_xq,
                        clustered_direct_cache->d_sorted_lj_type,
                        clustered_direct_cache->d_sorted_lj_comb, cell,
                        gmxpacked_LJ_AB_table, cutoff, clustered_force_target,
                        pme_beta, atom_energy, atom_virial,
                        atom_direct_pme_energy, d_LJ_energy_atom,
                        need_atom_energy != 0, need_virial != 0);
                    }
                }
                else if (use_gmxpacked_lj_comb_kernel)
                {
                    auto gmxpacked_f =
                        Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                            false, false, false, false, true, false, false,
                            false>;
                    if (need_atom_energy || need_virial)
                    {
                        gmxpacked_f =
                            Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                                true, true, false, true, true, false, false,
                                false>;
                    }
                    Launch_Device_Kernel(
                        gmxpacked_f, gmxpackedGridSize, blockSize, 0, NULL,
                        clustered_view.gmxpacked_sci_numbers,
                        clustered_view.cluster_size,
                        clustered_view.super_cluster_clusters,
                        clustered_view.cluster_numbers,
                        clustered_view.cluster_offsets,
                        clustered_view.cluster_valid_masks,
                        clustered_view.cluster_local_masks,
                        clustered_view.super_cluster_offsets,
                        clustered_view.gmxpacked_sci,
                        clustered_view.gmxpacked_cjpacked,
                        clustered_view.gmxpacked_exclusions,
                        gmxpacked_pair_shift_bits, NULL, 0,
                        clustered_direct_cache->d_sorted_atom_ids,
                        clustered_direct_cache->d_sorted_xq,
                        clustered_direct_cache->d_sorted_lj_type,
                        clustered_direct_cache->d_sorted_lj_comb, cell,
                        gmxpacked_LJ_AB_table, cutoff, clustered_force_target,
                        pme_beta, atom_energy, atom_virial,
                        atom_direct_pme_energy, d_LJ_energy_atom,
                        need_atom_energy != 0, need_virial != 0);
                }
                else
                {
                    auto gmxpacked_f =
                        Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                            false, false, false, false, false, false, false,
                            false>;
                    if (need_atom_energy || need_virial)
                    {
                        gmxpacked_f =
                            Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                                true, true, false, true, false, false, false,
                                false>;
                    }
                    Launch_Device_Kernel(
                        gmxpacked_f, gmxpackedGridSize, blockSize, 0, NULL,
                        clustered_view.gmxpacked_sci_numbers,
                        clustered_view.cluster_size,
                        clustered_view.super_cluster_clusters,
                        clustered_view.cluster_numbers,
                        clustered_view.cluster_offsets,
                        clustered_view.cluster_valid_masks,
                        clustered_view.cluster_local_masks,
                        clustered_view.super_cluster_offsets,
                        clustered_view.gmxpacked_sci,
                        clustered_view.gmxpacked_cjpacked,
                        clustered_view.gmxpacked_exclusions,
                        gmxpacked_pair_shift_bits, NULL, 0,
                        clustered_direct_cache->d_sorted_atom_ids,
                        clustered_direct_cache->d_sorted_xq,
                        clustered_direct_cache->d_sorted_lj_type, NULL, cell,
                        gmxpacked_LJ_AB_table, cutoff, clustered_force_target,
                        pme_beta, atom_energy, atom_virial,
                        atom_direct_pme_energy, d_LJ_energy_atom,
                        need_atom_energy != 0, need_virial != 0);
                }
                if (clustered_direct_cache->gmxpacked_kernel_launch_time_recorder !=
                    NULL)
                {
                    clustered_direct_cache->gmxpacked_kernel_launch_time_recorder
                        ->Stop();
                }
            }
#ifndef USE_CPU
            if (use_sorted_force_scratch)
            {
                if (use_gmxpacked_compact_force_scratch &&
                    clustered_direct_cache
                            ->gmxpacked_sorted_force_scatter_time_recorder !=
                        NULL)
                {
                    clustered_direct_cache
                        ->gmxpacked_sorted_force_scatter_time_recorder->Start();
                }
                Launch_Device_Kernel(
                    Scatter_Sorted_Clustered_Force,
                    (clustered_view.total_atom_numbers +
                     CONTROLLER::device_max_thread - 1) /
                        CONTROLLER::device_max_thread,
                    CONTROLLER::device_max_thread, 0, NULL,
                    clustered_view.total_atom_numbers,
                    clustered_direct_cache->d_sorted_atom_ids,
                    clustered_direct_cache->d_sorted_frc, frc);
                if (use_gmxpacked_compact_force_scratch &&
                    clustered_direct_cache
                            ->gmxpacked_sorted_force_scatter_time_recorder !=
                        NULL)
                {
                    clustered_direct_cache
                        ->gmxpacked_sorted_force_scatter_time_recorder->Stop();
                }
            }
#endif
            if (have_full_output_snapshot)
            {
                if (clustered_direct_cache
                        ->gmxpacked_full_output_snapshot_time_recorder != NULL)
                {
                    clustered_direct_cache
                        ->gmxpacked_full_output_snapshot_time_recorder->Start();
                }
                Finalize_Clustered_Microbench_Full_Output_Snapshot(
                    &full_output_snapshot, full_output_force_before, frc,
                    full_output_atom_energy_before, atom_energy,
                    full_output_atom_virial_before, atom_virial,
                    atom_direct_pme_energy, d_LJ_energy_atom);
                Maybe_Write_Clustered_Microbench_Full_Output_Snapshot(
                    full_output_snapshot);
                if (clustered_direct_cache
                        ->gmxpacked_full_output_snapshot_time_recorder != NULL)
                {
                    clustered_direct_cache
                        ->gmxpacked_full_output_snapshot_time_recorder->Stop();
                }
            }
            if (clustered_direct_cache->direct_kernel_time_recorder != NULL)
            {
                clustered_direct_cache->direct_kernel_time_recorder->Stop();
            }
#endif
        }
        else
        {
            dim3 blockSize = {
                CONTROLLER::device_warp,
                CONTROLLER::device_max_thread / CONTROLLER::device_warp};
            dim3 gridSize = (atom_numbers + blockSize.y - 1) / blockSize.y;
            auto f =
                Lennard_Jones_And_Direct_Coulomb_Device<true, false, false,
                                                        true>;
            if (need_atom_energy || need_virial)
            {
                f = Lennard_Jones_And_Direct_Coulomb_Device<
                    true, true, true, true>;
            }
            Launch_Device_Kernel(
                f, gridSize, blockSize, 0, NULL, local_atom_numbers,
                solvent_numbers, nl, crd_with_LJ_parameters_local, cell, rcell,
                d_LJ_A, d_LJ_B, cutoff, frc, pme_beta, atom_energy,
                atom_virial, atom_direct_pme_energy, d_LJ_energy_atom,
                need_atom_energy != 0, need_virial != 0);
        }
    }
}

void LENNARD_JONES_INFORMATION::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized || CONTROLLER::MPI_rank >= CONTROLLER::PP_MPI_size)
        return;
    Sum_Of_List(d_LJ_energy_atom, d_LJ_energy_sum, atom_numbers);
    deviceMemcpy(&h_LJ_energy_sum, d_LJ_energy_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &h_LJ_energy_sum, 1, MPI_FLOAT, MPI_SUM,
                  CONTROLLER::pp_comm);
#endif
    controller->Step_Print("LJ_short", h_LJ_energy_sum);
    controller->Step_Print("LJ_long", h_LJ_long_energy);
    controller->Step_Print("LJ", h_LJ_energy_sum + h_LJ_long_energy, true);
}
