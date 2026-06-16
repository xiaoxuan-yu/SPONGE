#include "Lennard_Jones_force.h"

#include <array>
#include <cstdlib>
#include <cstdint>

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

// ---------------------------------------------------------------------------
// Clustered direct dispatch env-var policy (T11 final, 2026-06-13)
// ---------------------------------------------------------------------------
// Production flags:
//   SPONGE_CLUSTERED_DUMP_MICROBENCH=<prefix>  diagnostic/debug snapshot path;
//                                               builds compact payload and
//                                               writes .sponge_fulloutput.bin
//   SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1     explicit opt-in for gmxpacked
//                                               production dispatch (requires
//                                               host-converted compact payload)
//   SPONGE_CLUSTERED_GMXPACKED_FALLBACK_NATIVE=1  forces native clustered
//                                                 dispatch; overrides opt-in
//
// Experimental / development-only flags (not production defaults):
//   SPONGE_CLUSTERED_USE_WARP_RECORD_FULL        selects warp-record virial
//                                                kernel variant (native path)
//   SPONGE_CLUSTERED_USE_WARP_RECORD_TOTAL_ONLY  selects total-output variant
//                                                of the warp-record kernel
//   SPONGE_CLUSTERED_USE_GROUPED_VIRIAL          selects grouped-clustered
//                                                virial kernel variant
//
// Default production path when none of the experimental flags are set:
//   base native clustered direct kernel
//   (Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Device)
// ---------------------------------------------------------------------------

static bool Clustered_Gmxpacked_Fallback_Native_Enabled()
{
    const char* enabled =
        std::getenv("SPONGE_CLUSTERED_GMXPACKED_FALLBACK_NATIVE");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

static bool Clustered_Gmxpacked_Direct_Opt_In_Enabled()
{
    const char* enabled =
        std::getenv("SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

// [EXPERIMENTAL] selects total-output warp-record variant, not a production default
static bool Clustered_Use_Warp_Record_Total_Output_Enabled()
{
    const char* enabled =
        std::getenv("SPONGE_CLUSTERED_USE_WARP_RECORD_TOTAL_ONLY");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
}

// [EXPERIMENTAL] selects grouped-clustered virial kernel, not a production default
static bool Clustered_Use_Grouped_Virial_Enabled()
{
    const char* enabled =
        std::getenv("SPONGE_CLUSTERED_USE_GROUPED_VIRIAL");
    return enabled != NULL && enabled[0] != '\0' && enabled[0] != '0';
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

static void Maybe_Dump_Clustered_Microbench_Diagnostic_Snapshot(
    const LJ_CLUSTERED_DIRECT_CACHE* clustered_direct_cache,
    const float2* d_LJ_AB_packed, size_t lj_param_numbers, float cutoff,
    float pme_beta, const LTMatrix3& cell)
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
    dumped = true;
}

static bool Capture_Clustered_Microbench_Full_Output_Diagnostic_View(
    const LJ_CLUSTERED_DIRECT_CACHE* clustered_direct_cache,
    const float2* d_LJ_AB_packed, size_t lj_param_numbers, float cutoff,
    float pme_beta, const LTMatrix3& cell,
    nbnxm_microbench::SpongeClusteredFullOutputSnapshot* snapshot)
{
    if (clustered_direct_cache == NULL || snapshot == NULL)
    {
        return false;
    }

    const auto& layout = clustered_direct_cache->layout;
    // Full-output snapshots intentionally serialize a diagnostic compatibility
    // view. Runtime gmxpacked dispatch consumes the compact payload directly.
    if (layout.total_atom_numbers <= 0 || layout.sci_numbers <= 0 ||
        layout.forceonly_warp_record_numbers <= 0 ||
        !Clustered_Layout_Has_Primary_Gmxpacked_Payload(layout) ||
        layout.d_forceonly_warp_record_offsets == NULL ||
        layout.d_forceonly_warp_j_records == NULL ||
        clustered_direct_cache->d_sorted_xq == NULL ||
        clustered_direct_cache->d_sorted_lj_type == NULL ||
        clustered_direct_cache->d_sorted_atom_ids == NULL ||
        d_LJ_AB_packed == NULL)
    {
        return false;
    }

    snapshot->header.file = nbnxm_microbench::MakeFileHeader(
        nbnxm_microbench::SnapshotKind::spongeClusteredFullOutput);
    snapshot->header.cluster_size = static_cast<uint32_t>(layout.cluster_size);
    snapshot->header.super_cluster_clusters =
        static_cast<uint32_t>(layout.super_cluster_clusters);
    snapshot->header.warp_split_count = kClusteredWarpSplitCount;
    snapshot->header.cluster_numbers =
        static_cast<uint64_t>(layout.cluster_numbers);
    snapshot->header.super_cluster_numbers =
        static_cast<uint64_t>(layout.super_cluster_numbers);
    snapshot->header.sci_numbers = static_cast<uint64_t>(layout.sci_numbers);
    snapshot->header.record_numbers =
        static_cast<uint64_t>(layout.forceonly_warp_record_numbers);
    snapshot->header.pair_shift_word_numbers =
        static_cast<uint64_t>(layout.cjpacked_numbers * kClusteredJGroupSize);
    snapshot->header.total_atom_numbers =
        static_cast<uint64_t>(layout.total_atom_numbers);
    snapshot->header.local_atom_numbers =
        static_cast<uint64_t>(layout.local_atom_numbers);
    snapshot->header.lj_param_numbers =
        static_cast<uint64_t>(lj_param_numbers);
    snapshot->header.cutoff = cutoff;
    snapshot->header.pme_beta = pme_beta;
    snapshot->header.cell = To_Microbench_Matrix_POD(cell);

    snapshot->cluster_offsets = Copy_Device_Vector_To_Host(
        layout.d_cluster_offsets, static_cast<size_t>(layout.cluster_numbers));
    snapshot->cluster_valid_masks = Copy_Device_Vector_To_Host(
        layout.d_cluster_valid_masks,
        static_cast<size_t>(layout.cluster_numbers));
    snapshot->cluster_local_masks = Copy_Device_Vector_To_Host(
        layout.d_cluster_local_masks,
        static_cast<size_t>(layout.cluster_numbers));
    snapshot->super_cluster_offsets = Copy_Device_Vector_To_Host(
        layout.d_super_cluster_offsets,
        static_cast<size_t>(layout.super_cluster_numbers + 1));

    const auto host_sci = Copy_Device_Vector_To_Host(
        layout.d_nbnxm_sci, static_cast<size_t>(layout.sci_numbers));
    snapshot->sci.clear();
    snapshot->sci.reserve(host_sci.size());
    for (const LJ_CLUSTERED_SCI& sci : host_sci)
    {
        snapshot->sci.push_back(To_Microbench_Sci_POD(sci));
    }

    snapshot->record_offsets = Copy_Device_Vector_To_Host(
        layout.d_forceonly_warp_record_offsets,
        static_cast<size_t>(layout.sci_numbers + 1));
    const auto host_records = Copy_Device_Vector_To_Host(
        layout.d_forceonly_warp_j_records,
        static_cast<size_t>(layout.forceonly_warp_record_numbers));
    snapshot->records.clear();
    snapshot->records.reserve(host_records.size());
    for (const LJ_CLUSTERED_WARP_J_RECORD& record : host_records)
    {
        snapshot->records.push_back(To_Microbench_Record_POD(record));
    }

    snapshot->pair_shift_bits = Copy_Device_Vector_To_Host(
        layout.d_pair_shift_bits,
        static_cast<size_t>(layout.cjpacked_numbers * kClusteredJGroupSize));
    snapshot->sorted_atom_ids = Copy_Device_Vector_To_Host(
        clustered_direct_cache->d_sorted_atom_ids,
        static_cast<size_t>(layout.total_atom_numbers));
    const auto host_sorted_xq = Copy_Device_Vector_To_Host(
        clustered_direct_cache->d_sorted_xq,
        static_cast<size_t>(layout.total_atom_numbers));
    snapshot->sorted_xq.clear();
    snapshot->sorted_xq.reserve(host_sorted_xq.size());
    for (const float4& value : host_sorted_xq)
    {
        snapshot->sorted_xq.push_back(To_Microbench_Float4_POD(value));
    }
    snapshot->sorted_lj_type = Copy_Device_Vector_To_Host(
        clustered_direct_cache->d_sorted_lj_type,
        static_cast<size_t>(layout.total_atom_numbers));
    const auto host_lj_ab =
        Copy_Device_Vector_To_Host(d_LJ_AB_packed, lj_param_numbers);
    snapshot->lj_ab.clear();
    snapshot->lj_ab.reserve(host_lj_ab.size());
    for (const float2& value : host_lj_ab)
    {
        snapshot->lj_ab.push_back(To_Microbench_Float2_POD(value));
    }
    return true;
}

static void Finalize_Clustered_Microbench_Full_Output_Snapshot(
    nbnxm_microbench::SpongeClusteredFullOutputSnapshot* snapshot,
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
        static_cast<size_t>(snapshot->header.total_atom_numbers);
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
    const nbnxm_microbench::SpongeClusteredFullOutputSnapshot& snapshot)
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
    if (!nbnxm_microbench::WriteSpongeClusteredFullOutputSnapshot(
            snapshot_path.string(), snapshot))
    {
        fprintf(stderr,
                "[clustered microbench dump] failed to write %s\n",
                snapshot_path.string().c_str());
        return;
    }
    fprintf(stderr,
            "[clustered microbench dump] wrote %s sci=%llu records=%llu atoms=%llu energy=%u virial=%u total_output=%u\n",
            snapshot_path.string().c_str(),
            static_cast<unsigned long long>(snapshot.header.sci_numbers),
            static_cast<unsigned long long>(snapshot.header.record_numbers),
            static_cast<unsigned long long>(snapshot.header.total_atom_numbers),
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
};

template <int size>
struct Clustered_Energy_Buffer<true, size>
{
    float values[size] = {};

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

    __device__ __forceinline__ Clustered_Full_Record_Output_Buffer()
    {
        for (int i = 0; i < size; i += 1)
        {
            virial[i] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
};

template <bool need_energy, int size>
struct Clustered_Full_Record_Output_Buffer<true, need_energy, size>
{
    float energy_lj_total = 0.0f;
    float energy_coulomb_total = 0.0f;
    LTMatrix3 virial_total = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
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

static __global__ void Scatter_Sorted_Clustered_Force_SoA(
    const int total_atom_numbers, const int* sorted_atom_ids,
    const float* sorted_frc_x, const float* sorted_frc_y,
    const float* sorted_frc_z, VECTOR* frc)
{
    SIMPLE_DEVICE_FOR(sorted_i, total_atom_numbers)
    {
        const int atom_i = sorted_atom_ids[sorted_i];
        frc[atom_i] = frc[atom_i] +
                      VECTOR{sorted_frc_x[sorted_i], sorted_frc_y[sorted_i],
                             sorted_frc_z[sorted_i]};
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
Wrap_Clustered_Center_Fractional(const VECTOR center, const LTMatrix3 rcell)
{
    VECTOR frac = center * rcell;
    frac.x -= floorf(frac.x);
    frac.y -= floorf(frac.y);
    frac.z -= floorf(frac.z);
    return frac;
}

static __host__ __device__ __forceinline__ int Encode_Clustered_Pair_Shift_Id(
    int sx, int sy, int sz)
{
    sx = sx < -1 ? -1 : (sx > 1 ? 1 : sx);
    sy = sy < -1 ? -1 : (sy > 1 ? 1 : sy);
    sz = sz < -1 ? -1 : (sz > 1 ? 1 : sz);
    return (sx + 1) * 9 + (sy + 1) * 3 + (sz + 1);
}

static __host__ __device__ __forceinline__ int
Determine_Clustered_Pair_Shift_Id(const VECTOR center_i, const VECTOR center_j,
                                  const LTMatrix3 rcell)
{
    const VECTOR frac_i = Wrap_Clustered_Center_Fractional(center_i, rcell);
    const VECTOR frac_j = Wrap_Clustered_Center_Fractional(center_j, rcell);
    const VECTOR dfrac = frac_j - frac_i;
    return Encode_Clustered_Pair_Shift_Id(
        static_cast<int>(floorf(dfrac.x + 0.5f)),
        static_cast<int>(floorf(dfrac.y + 0.5f)),
        static_cast<int>(floorf(dfrac.z + 0.5f)));
}

static __host__ __device__ __forceinline__ VECTOR
Get_Clustered_Pair_Shift_Vector(const VECTOR center_i, const VECTOR center_j,
                                const LTMatrix3 cell, const LTMatrix3 rcell)
{
    return Clustered_Shift_Vector_From_Id(
        Determine_Clustered_Pair_Shift_Id(center_i, center_j, rcell), cell);
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
    LTMatrix3* atom_virial, float* atom_direct_cf_energy, float* atom_LJ_ene)
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
                if (need_energy)
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
        if (need_energy)
        {
            Warp_Sum_To(atom_energy + atom_i, energy_total, warpSize);
            Warp_Sum_To(atom_LJ_ene + atom_i, energy_lj, warpSize);
            if (need_coulomb)
                Warp_Sum_To(atom_direct_cf_energy + atom_i, energy_coulomb,
                            warpSize);
        }
        if (need_virial)
        {
            Warp_Sum_To(atom_virial + atom_i, virial, warpSize);
        }
    }
}

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

template <bool need_energy, bool need_virial, bool total_output,
          bool compact_force_storage>
static __global__ __launch_bounds__(kClusteredClusterSize *
                                         kClusteredSuperClusterClusters,
                                     total_output ? 12 : 13)
    void
Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device(
    const int sci_numbers, const int cluster_size,
    const int super_cluster_clusters, const int* cluster_offsets,
    const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const int* super_cluster_offsets,
    const LJ_CLUSTERED_GMXPACKED_SCI* sci_entries,
    const LJ_CLUSTERED_GMXPACKED_CJ* cjpacked_entries,
    const LJ_CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    const uint64_t* pair_shift_bits, const int* sorted_atom_ids,
    const float4* sorted_xq, const int* sorted_lj_type, const LTMatrix3 cell,
    const float2* LJ_type_AB_packed, const float cutoff, VECTOR* frc,
    const float pme_beta, float* atom_energy, LTMatrix3* atom_virial,
    float* atom_direct_cf_energy, float* atom_LJ_ene)
{
    static_assert(total_output || (!need_energy && !need_virial),
                  "gmxpacked energy/virial dispatch must use total output");
    constexpr int max_super_cluster_atoms =
        kClusteredClusterSize * kClusteredSuperClusterClusters;
    constexpr int max_block_warps = 2;
#ifdef USE_GPU
    const int sci = blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (sci >= sci_numbers ||
        tid >= super_cluster_clusters * cluster_size)
    {
        return;
    }
#else
    (void)sci_numbers;
    (void)cluster_size;
    (void)super_cluster_clusters;
    (void)cluster_offsets;
    (void)cluster_valid_masks;
    (void)cluster_local_masks;
    (void)super_cluster_offsets;
    (void)sci_entries;
    (void)cjpacked_entries;
    (void)exclusion_entries;
    (void)pair_shift_bits;
    (void)sorted_atom_ids;
    (void)sorted_xq;
    (void)sorted_lj_type;
    (void)cell;
    (void)LJ_type_AB_packed;
    (void)cutoff;
    (void)frc;
    (void)pme_beta;
    (void)atom_energy;
    (void)atom_virial;
    (void)atom_direct_cf_energy;
    (void)atom_LJ_ene;
    return;
#endif

    const int i_lane = threadIdx.x;
    const int j_lane = threadIdx.y;
    const int lane = tid & (warpSize - 1);
    const int split = tid / warpSize;
    const int split_j_lane = j_lane - split * kClusteredSplitJClusterSize;
    const int i_slot = j_lane * cluster_size + i_lane;

    const LJ_CLUSTERED_GMXPACKED_SCI sci_entry = sci_entries[sci];
    const int super_i = sci_entry.supercluster_id;
    const int cluster_i_start = super_cluster_offsets[super_i];
    const int cluster_i_end = super_cluster_offsets[super_i + 1];
    const int active_cluster_count = cluster_i_end - cluster_i_start;
    const float cutoff_sq = cutoff * cutoff;
    const float beta2 = pme_beta * pme_beta;
    const float beta3 = beta2 * pme_beta;
    constexpr float min_distance_sq = 3.82e-07f;
    const VECTOR sci_shift =
        Clustered_Shift_Vector_From_Id(sci_entry.shift_id, cell);

#define CLUSTERED_GMXPACKED_I_LOCAL_LIST(OP) \
    OP(0)                                    \
    OP(1)                                    \
    OP(2)                                    \
    OP(3)                                    \
    OP(4)                                    \
    OP(5)                                    \
    OP(6)                                    \
    OP(7)

#define CLUSTERED_GMXPACKED_JM_LIST(OP) \
    OP(0)                               \
    OP(1)                               \
    OP(2)                               \
    OP(3)

    __shared__ float4 shared_i_xq[max_super_cluster_atoms];
    __shared__ int shared_i_lj_type[max_super_cluster_atoms];
    __shared__ int shared_i_sorted_ids[max_super_cluster_atoms];
    __shared__ unsigned int shared_i_valid_masks[kClusteredSuperClusterClusters];
    __shared__ unsigned int shared_i_local_masks[kClusteredSuperClusterClusters];
    __shared__ float4 shared_total_virial_lo[max_block_warps];
    __shared__ float2 shared_total_virial_hi[max_block_warps];
    __shared__ float shared_total_energy_lj[max_block_warps];
    __shared__ float shared_total_energy_coulomb[max_block_warps];

#define CLUSTERED_GMXPACKED_DECLARE_FCI(I) \
    float fci_x_##I = 0.0f;               \
    float fci_y_##I = 0.0f;               \
    float fci_z_##I = 0.0f;
    CLUSTERED_GMXPACKED_I_LOCAL_LIST(CLUSTERED_GMXPACKED_DECLARE_FCI)
#undef CLUSTERED_GMXPACKED_DECLARE_FCI

    if (j_lane == 0)
    {
        if (i_lane < active_cluster_count)
        {
            const int cluster_i = cluster_i_start + i_lane;
            shared_i_valid_masks[i_lane] = cluster_valid_masks[cluster_i];
            shared_i_local_masks[i_lane] = cluster_local_masks[cluster_i];
        }
        else if (i_lane < kClusteredSuperClusterClusters)
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
            shared_i_xq[i_slot] = Clustered_Load_ReadOnly(sorted_xq + sorted_i);
            shared_i_lj_type[i_slot] =
                Clustered_Load_ReadOnly(sorted_lj_type + sorted_i);
            shared_i_sorted_ids[i_slot] = sorted_i;
        }
        else
        {
            shared_i_sorted_ids[i_slot] = -1;
        }
    }
    __syncthreads();

    Clustered_Full_Record_Output_Buffer<total_output, need_energy,
                                        kClusteredSuperClusterClusters>
        output_buf;
    const unsigned int i_lane_mask = 1u << static_cast<unsigned int>(i_lane);
    unsigned int active_i_mask = 0u;
#define CLUSTERED_GMXPACKED_CACHE_ACTIVE_I(I)                              \
    if ((I) < active_cluster_count)                                         \
    {                                                                       \
        const unsigned int valid_mask_i = shared_i_valid_masks[I];          \
        const unsigned int local_mask_i = shared_i_local_masks[I];          \
        if ((valid_mask_i & i_lane_mask) != 0u &&                           \
            (local_mask_i & i_lane_mask) != 0u)                             \
        {                                                                   \
            active_i_mask |= (1u << static_cast<unsigned int>(I));          \
        }                                                                   \
    }
    CLUSTERED_GMXPACKED_I_LOCAL_LIST(CLUSTERED_GMXPACKED_CACHE_ACTIVE_I)
#undef CLUSTERED_GMXPACKED_CACHE_ACTIVE_I

    for (int packed_idx = sci_entry.cjpacked_begin;
         packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
    {
        const LJ_CLUSTERED_GMXPACKED_CJ* packed = cjpacked_entries + packed_idx;
        const unsigned int imask = packed->split[split].imask;
        if (imask == 0u)
        {
            continue;
        }
        const int exclusion_index = packed->split[split].exclusion_index;
        unsigned int pair_bits = 0xffffffffu;
        if (exclusion_index != 0 && exclusion_entries != NULL)
        {
            pair_bits = exclusion_entries[exclusion_index]
                            .pair[split_j_lane * cluster_size + i_lane];
        }
        const unsigned int effective_mask = imask & pair_bits;
#define CLUSTERED_GMXPACKED_COMPUTE_I(I)                                    \
    {                                                                       \
        const unsigned int packed_bit =                                      \
            base_mask << static_cast<unsigned int>(I);                      \
        if ((effective_mask & packed_bit) != 0u &&                          \
            (active_i_mask & (1u << static_cast<unsigned int>(I))) != 0u)   \
        {                                                                   \
            const float4 r1_xq =                                            \
                shared_i_xq[(I) * cluster_size + i_lane];                   \
            const VECTOR pair_shift =                                       \
                pair_shift_bits != NULL                                     \
                    ? Clustered_Shift_Vector_From_Id(                       \
                          Clustered_Get_Pair_Shift_Id(shift_bits, I), cell) \
                    : sci_shift;                                            \
            const float dx = shifted_j_x - r1_xq.x - pair_shift.x;          \
            const float dy = shifted_j_y - r1_xq.y - pair_shift.y;          \
            const float dz = shifted_j_z - r1_xq.z - pair_shift.z;          \
            const float dr2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));          \
            if (dr2 < cutoff_sq && dr2 != 0.0f)                             \
            {                                                               \
                const float r2 = fmaxf(dr2, min_distance_sq);               \
                const float inv_r = rsqrtf(r2);                             \
                const float inv_r2 = inv_r * inv_r;                         \
                const float inv_r6 = inv_r2 * inv_r2 * inv_r2;              \
                const float beta_dr = pme_beta * (r2 * inv_r);              \
                const float charge_product = r1_xq.w * qj;                 \
                const int r1_lj_type =                                      \
                    shared_i_lj_type[(I) * cluster_size + i_lane];          \
                const int atom_pair_LJ_type =                               \
                    Get_LJ_Type(r1_lj_type, r2_lj_type);                    \
                const float2 AB = Clustered_Load_ReadOnly(                  \
                    LJ_type_AB_packed + atom_pair_LJ_type);                 \
                float frc_abs =                                             \
                    Get_Clustered_LJ_Force_Abs(inv_r2, inv_r6, AB.x, AB.y); \
                frc_abs -=                                                  \
                    Get_Clustered_Direct_Coulomb_Force_Abs_PME_Corr(        \
                        charge_product, inv_r, inv_r2, beta2 * r2, beta3);  \
                const float fij_x = frc_abs * dx;                           \
                const float fij_y = frc_abs * dy;                           \
                const float fij_z = frc_abs * dz;                           \
                fci_x_##I += fij_x;                                         \
                fci_y_##I += fij_y;                                         \
                fci_z_##I += fij_z;                                         \
                if (j_is_local)                                             \
                {                                                           \
                    fcj_x -= fij_x;                                         \
                    fcj_y -= fij_y;                                         \
                    fcj_z -= fij_z;                                         \
                }                                                           \
                if constexpr (need_virial)                                  \
                {                                                           \
                    output_buf.virial_total = output_buf.virial_total -     \
                        ij_factor * Get_Virial_From_Force_Dis(              \
                                        VECTOR{fij_x, fij_y, fij_z},         \
                                        VECTOR{dx, dy, dz});                \
                }                                                           \
                if constexpr (need_energy)                                  \
                {                                                           \
                    const float pair_energy_lj =                            \
                        ij_factor *                                         \
                        Get_Clustered_LJ_Energy(inv_r6, AB.x, AB.y);        \
                    const float pair_energy_coulomb =                       \
                        ij_factor * Get_Clustered_Direct_Coulomb_Energy(    \
                                        charge_product, inv_r, beta_dr);     \
                    output_buf.energy_lj_total += pair_energy_lj;           \
                    output_buf.energy_coulomb_total += pair_energy_coulomb; \
                }                                                           \
            }                                                               \
        }                                                                   \
    }

#define CLUSTERED_GMXPACKED_PROCESS_JM(JM)                                  \
    {                                                                       \
        constexpr unsigned int base_mask =                                  \
            1u << ((JM) * kClusteredSuperClusterClusters);                  \
        constexpr unsigned int jm_mask =                                    \
            ((1u << kClusteredSuperClusterClusters) - 1u)                   \
            << ((JM) * kClusteredSuperClusterClusters);                     \
        if ((imask & jm_mask) != 0u)                                        \
        {                                                                   \
            const int cluster_j = packed->cj[JM];                           \
            if (cluster_j >= 0)                                             \
            {                                                               \
                const uint64_t shift_bits =                                 \
                    pair_shift_bits != NULL                                 \
                        ? pair_shift_bits[packed_idx * kClusteredJGroupSize + (JM)] \
                        : 0ull;                                             \
                const unsigned int valid_mask_j = cluster_valid_masks[cluster_j]; \
                if ((valid_mask_j &                                         \
                     (1u << static_cast<unsigned int>(j_lane))) != 0u)      \
                {                                                           \
                    const int sorted_j = cluster_offsets[cluster_j] + j_lane; \
                    const float4 r2_xq =                                    \
                        Clustered_Load_ReadOnly(sorted_xq + sorted_j);      \
                    const int r2_lj_type =                                  \
                        Clustered_Load_ReadOnly(sorted_lj_type + sorted_j); \
                    const float shifted_j_x = r2_xq.x;                      \
                    const float shifted_j_y = r2_xq.y;                      \
                    const float shifted_j_z = r2_xq.z;                      \
                    const float qj = r2_xq.w;                               \
                    const bool j_is_local =                                 \
                        (cluster_local_masks[cluster_j] &                   \
                         (1u << static_cast<unsigned int>(j_lane))) != 0u;  \
                    const float ij_factor = j_is_local ? 1.0f : 0.5f;       \
                    float fcj_x = 0.0f;                                     \
                    float fcj_y = 0.0f;                                     \
                    float fcj_z = 0.0f;                                     \
                    CLUSTERED_GMXPACKED_I_LOCAL_LIST(                       \
                        CLUSTERED_GMXPACKED_COMPUTE_I)                      \
                    if (j_is_local)                                         \
                    {                                                       \
                        const float fcj_component =                         \
                            Reduce_Clustered_Subgroup_Vector_To_Component(  \
                                fcj_x, fcj_y, fcj_z, i_lane, lane,          \
                                cluster_size);                              \
                        if (i_lane < 3)                                     \
                        {                                                   \
                            const int force_index =                         \
                                compact_force_storage                       \
                                    ? sorted_j                              \
                                    : sorted_atom_ids[sorted_j];            \
                            Clustered_Atomic_Add_Force_Component(           \
                                frc, force_index, i_lane, fcj_component);   \
                        }                                                   \
                    }                                                       \
                }                                                           \
            }                                                               \
        }                                                                   \
    }
        CLUSTERED_GMXPACKED_JM_LIST(CLUSTERED_GMXPACKED_PROCESS_JM)
#undef CLUSTERED_GMXPACKED_PROCESS_JM
#undef CLUSTERED_GMXPACKED_COMPUTE_I
    }

#define CLUSTERED_GMXPACKED_REDUCE_I(I)                                     \
    if ((I) < active_cluster_count)                                         \
    {                                                                       \
        const bool active_i =                                               \
            (active_i_mask & (1u << static_cast<unsigned int>(I))) != 0u;   \
        float reduced_x = active_i ? fci_x_##I : 0.0f;                      \
        float reduced_y = active_i ? fci_y_##I : 0.0f;                      \
        float reduced_z = active_i ? fci_z_##I : 0.0f;                      \
        const float reduced_component =                                     \
            Reduce_Clustered_Warp_I_To_Component(                           \
                reduced_x, reduced_y, reduced_z, i_lane, split_j_lane,      \
                cluster_size);                                              \
        if (active_i && split_j_lane < 3)                                   \
        {                                                                   \
            const int sorted_i =                                            \
                shared_i_sorted_ids[(I) * cluster_size + i_lane];           \
            const int force_index =                                         \
                compact_force_storage ? sorted_i : sorted_atom_ids[sorted_i]; \
            Clustered_Atomic_Add_Force_Component(                           \
                frc, force_index, split_j_lane, reduced_component);         \
        }                                                                   \
    }
    CLUSTERED_GMXPACKED_I_LOCAL_LIST(CLUSTERED_GMXPACKED_REDUCE_I)
#undef CLUSTERED_GMXPACKED_REDUCE_I

    if constexpr (total_output)
    {
        LTMatrix3 reduced_total_virial = LTMatrix3(0.0f);
        if constexpr (need_virial)
        {
            reduced_total_virial =
                Reduce_Clustered_Warp_Virial_All(output_buf.virial_total);
        }
        float reduced_total_energy_lj = 0.0f;
        float reduced_total_energy_coulomb = 0.0f;
        if constexpr (need_energy)
        {
            reduced_total_energy_lj =
                Reduce_Clustered_Warp_Float_All(output_buf.energy_lj_total);
            reduced_total_energy_coulomb = Reduce_Clustered_Warp_Float_All(
                output_buf.energy_coulomb_total);
        }
        if (lane == 0)
        {
            if constexpr (need_virial)
            {
                shared_total_virial_lo[split] =
                    Pack_Clustered_Virial_Lo(reduced_total_virial);
                shared_total_virial_hi[split] =
                    Pack_Clustered_Virial_Hi(reduced_total_virial);
            }
            if constexpr (need_energy)
            {
                shared_total_energy_lj[split] = reduced_total_energy_lj;
                shared_total_energy_coulomb[split] =
                    reduced_total_energy_coulomb;
            }
        }
        __syncthreads();
        if (tid == 0)
        {
            if constexpr (need_virial)
            {
                LTMatrix3 block_total_virial = Unpack_Clustered_Virial(
                    shared_total_virial_lo[0], shared_total_virial_hi[0]);
                for (int warp = 1; warp < max_block_warps; warp += 1)
                {
                    block_total_virial =
                        block_total_virial +
                        Unpack_Clustered_Virial(shared_total_virial_lo[warp],
                                                shared_total_virial_hi[warp]);
                }
                atomicAdd(atom_virial, block_total_virial);
            }
            if constexpr (need_energy)
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
                atomicAdd(atom_LJ_ene, block_total_energy_lj);
                atomicAdd(atom_direct_cf_energy, block_total_energy_coulomb);
            }
        }
    }

#undef CLUSTERED_GMXPACKED_JM_LIST
#undef CLUSTERED_GMXPACKED_I_LOCAL_LIST
}

template <bool need_energy, bool force_soa, bool total_output = false>
static __global__ __launch_bounds__(kClusteredClusterSize *
                                         kClusteredSuperClusterClusters,
                                     13)
    void
Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Virial_Warp_Record_Device(
    const int sci_numbers, const int cluster_size,
    const int super_cluster_clusters, const int local_atom_numbers,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const int* super_cluster_offsets,
    const LJ_CLUSTERED_SCI* sci_entries,
    const int* compact_record_offsets,
    const LJ_CLUSTERED_WARP_J_RECORD* compact_warp_j_records,
    const uint64_t* pair_shift_bits,
    const unsigned long long* exclusion_mask_pool,
    const int* sorted_atom_ids, const float4* sorted_xq,
    const int* sorted_lj_type, const LTMatrix3 cell, const float* LJ_type_A,
    const float* LJ_type_B, const float2* LJ_type_AB_packed,
    const float cutoff, VECTOR* frc, float* frc_x,
    float* frc_y, float* frc_z, const float pme_beta, float* atom_energy,
    LTMatrix3* atom_virial, float* atom_direct_cf_energy, float* atom_LJ_ene)
{
    constexpr int max_cluster_size = kClusteredClusterSize;
    constexpr int max_super_cluster_atoms =
        kClusteredClusterSize * kClusteredSuperClusterClusters;
    constexpr int max_block_warps = 2;
    constexpr bool need_atom_i_ids = !force_soa || !total_output;
#ifdef USE_GPU
    const int sci = blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (sci >= sci_numbers ||
        tid >= super_cluster_clusters * cluster_size)
    {
        return;
    }
#else
    (void)sci_numbers;
    (void)cluster_size;
    (void)super_cluster_clusters;
    (void)local_atom_numbers;
    (void)cluster_offsets;
    (void)cluster_valid_masks;
    (void)cluster_local_masks;
    (void)super_cluster_offsets;
    (void)sci_entries;
    (void)compact_record_offsets;
    (void)compact_warp_j_records;
    (void)pair_shift_bits;
    (void)exclusion_mask_pool;
    (void)sorted_atom_ids;
    (void)sorted_xq;
    (void)sorted_lj_type;
    (void)cell;
    (void)LJ_type_A;
    (void)LJ_type_B;
    (void)LJ_type_AB_packed;
    (void)cutoff;
    (void)frc;
    (void)frc_x;
    (void)frc_y;
    (void)frc_z;
    (void)pme_beta;
    (void)atom_energy;
    (void)atom_virial;
    (void)atom_direct_cf_energy;
    (void)atom_LJ_ene;
    return;
#endif

    const LJ_CLUSTERED_SCI sci_entry = sci_entries[sci];
    const int super_i = sci_entry.supercluster_id;
    const int cluster_i_start = super_cluster_offsets[super_i];
    const int cluster_i_end = super_cluster_offsets[super_i + 1];
    const int record_begin = compact_record_offsets[sci];
    const int record_end = compact_record_offsets[sci + 1];
    const bool sci_is_central =
        sci_entry.shift_id == kClusteredCentralShiftId;
    const float cutoff_sq = cutoff * cutoff;
    const float beta2 = pme_beta * pme_beta;
    const float beta3 = beta2 * pme_beta;

    __shared__ float4 shared_i_xq[max_super_cluster_atoms];
    __shared__ int shared_i_lj_type[max_super_cluster_atoms];
    __shared__ int shared_i_atom_ids[max_super_cluster_atoms];
    __shared__ int shared_i_sorted_ids[max_super_cluster_atoms];
    __shared__ unsigned int shared_i_valid_masks[kClusteredSuperClusterClusters];
    __shared__ unsigned int shared_i_local_masks[kClusteredSuperClusterClusters];
    __shared__ float4 warp1_i_force[kClusteredSuperClusterClusters]
                                    [max_cluster_size];
    __shared__ float warp1_i_energy_lj[kClusteredSuperClusterClusters]
                                      [max_cluster_size];
    __shared__ float warp1_i_energy_coulomb
        [kClusteredSuperClusterClusters][max_cluster_size];
    __shared__ float4 warp1_i_virial_lo[kClusteredSuperClusterClusters]
                                       [max_cluster_size];
    __shared__ float2 warp1_i_virial_hi[kClusteredSuperClusterClusters]
                                       [max_cluster_size];
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
    const int warp_j_base = warp_id * kClusteredSplitJClusterSize;
    const int warp_j_local = j_lane - warp_j_base;

#define CLUSTERED_STEADY_I_LOCAL_LIST(OP) \
    OP(0)                                 \
    OP(1)                                 \
    OP(2)                                 \
    OP(3)                                 \
    OP(4)                                 \
    OP(5)                                 \
    OP(6)                                 \
    OP(7)

#define CLUSTERED_DECLARE_FCI_BUF(I) \
    float fci_x_##I = 0.0f;         \
    float fci_y_##I = 0.0f;         \
    float fci_z_##I = 0.0f;
    CLUSTERED_STEADY_I_LOCAL_LIST(CLUSTERED_DECLARE_FCI_BUF)
#undef CLUSTERED_DECLARE_FCI_BUF
    Clustered_Full_Record_Output_Buffer<total_output, need_energy,
                                        kClusteredSuperClusterClusters>
        output_buf;

    if (j_lane == 0)
    {
        if (i_lane < active_cluster_count)
        {
            const int cluster_i = cluster_i_start + i_lane;
            shared_i_valid_masks[i_lane] = cluster_valid_masks[cluster_i];
            shared_i_local_masks[i_lane] = cluster_local_masks[cluster_i];
        }
        else if (i_lane < kClusteredSuperClusterClusters)
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
            if constexpr (need_atom_i_ids)
            {
                shared_i_atom_ids[i_slot] = sorted_atom_ids[sorted_atom_i];
            }
            shared_i_sorted_ids[i_slot] = sorted_atom_i;
        }
        else
        {
            if constexpr (need_atom_i_ids)
            {
                shared_i_atom_ids[i_slot] = -1;
            }
            shared_i_sorted_ids[i_slot] = -1;
        }
    }
    __syncthreads();
    const unsigned int i_lane_mask =
        1u << static_cast<unsigned int>(i_lane);
    unsigned int active_i_mask = 0u;
#define CLUSTERED_CACHE_ACTIVE_I(I)                                        \
    if ((I) < active_cluster_count)                                        \
    {                                                                      \
        const unsigned int valid_mask_i = shared_i_valid_masks[I];         \
        const unsigned int local_mask_i = shared_i_local_masks[I];         \
        if ((valid_mask_i & i_lane_mask) != 0u &&                          \
            (local_mask_i & i_lane_mask) != 0u)                            \
        {                                                                  \
            active_i_mask |= (1u << static_cast<unsigned int>(I));         \
        }                                                                  \
    }
    CLUSTERED_STEADY_I_LOCAL_LIST(CLUSTERED_CACHE_ACTIVE_I)
#undef CLUSTERED_CACHE_ACTIVE_I
#define CLUSTERED_DECLARE_I_LJ_TYPE_CACHE(I) int cached_i_lj_type_##I = 0;
    CLUSTERED_STEADY_I_LOCAL_LIST(CLUSTERED_DECLARE_I_LJ_TYPE_CACHE)
#undef CLUSTERED_DECLARE_I_LJ_TYPE_CACHE
#define CLUSTERED_CACHE_I_LJ_TYPE(I)                                      \
    if ((I) < active_cluster_count &&                                     \
        (active_i_mask & (1u << static_cast<unsigned int>(I))) != 0u)    \
    {                                                                     \
        cached_i_lj_type_##I = shared_i_lj_type[(I) * cluster_size +      \
                                                 i_lane];                  \
    }
    CLUSTERED_STEADY_I_LOCAL_LIST(CLUSTERED_CACHE_I_LJ_TYPE)
#undef CLUSTERED_CACHE_I_LJ_TYPE

    for (int record_idx = record_begin + warp_id; record_idx < record_end;
         record_idx += max_block_warps)
    {
        const LJ_CLUSTERED_WARP_J_RECORD* record =
            compact_warp_j_records + record_idx;
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
        cluster_j = deviceShfl(FULL_MASK, cluster_j, 0, warpSize);
        pair_shift_index = deviceShfl(FULL_MASK, pair_shift_index, 0, warpSize);
        valid_mask_j = deviceShfl(FULL_MASK, valid_mask_j, 0, warpSize);
        imask = deviceShfl(FULL_MASK, imask, 0, warpSize);
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
        shift_bits = Broadcast_Clustered_Warp_U64(shift_bits, 0);
        local_mask_j = deviceShfl(FULL_MASK, local_mask_j, 0, warpSize);
        j_lane_base = deviceShfl(FULL_MASK, j_lane_base, 0, warpSize);

        if (cluster_j < 0 ||
            (valid_mask_j &
             (1u << static_cast<unsigned int>(warp_j_local))) == 0u)
        {
            continue;
        }

        int sorted_j = -1;
        int atom_j = -1;
        int absolute_j_lane = -1;
        float4 r2_xq = {0.0f, 0.0f, 0.0f, 0.0f};
        int r2_lj_type = 0;
        if (i_lane == 0)
        {
            sorted_j = record->sorted_j_base + warp_j_local;
            if constexpr (!force_soa)
            {
                atom_j = sorted_atom_ids[sorted_j];
            }
            absolute_j_lane =
                static_cast<int>(j_lane_base) + warp_j_local;
            r2_xq = sorted_xq[sorted_j];
            r2_lj_type = sorted_lj_type[sorted_j];
        }
        sorted_j =
            Broadcast_Clustered_Subgroup_Value(sorted_j, lane, cluster_size);
        if constexpr (!force_soa)
        {
            atom_j = Broadcast_Clustered_Subgroup_Value(atom_j, lane,
                                                        cluster_size);
        }
        absolute_j_lane = Broadcast_Clustered_Subgroup_Value(
            absolute_j_lane, lane, cluster_size);
        r2_xq = Broadcast_Clustered_Subgroup_Float4(r2_xq, lane, cluster_size);
        r2_lj_type = Broadcast_Clustered_Subgroup_Value(r2_lj_type, lane,
                                                        cluster_size);
        const int atom_j_is_local =
            (local_mask_j & (1u << static_cast<unsigned int>(warp_j_local))) !=
                    0u
                ? 1
                : 0;
        const VECTOR_LJ r2 = Make_Packed_LJ_Atom(r2_xq, r2_lj_type);
        const unsigned char pair_excl_mask =
            record->pair_excl[warp_j_local * cluster_size + i_lane];
        float fcj_x = 0.0f;
        float fcj_y = 0.0f;
        float fcj_z = 0.0f;

#define CLUSTERED_ACCUMULATE_I_LOCAL(I)                                     \
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
                const VECTOR_LJ r1 =                                        \
                    Make_Packed_LJ_Atom(r1_xq, r1_lj_type);                 \
                const VECTOR pair_shift =                                   \
                    Clustered_Shift_Vector_From_Id(                         \
                        Clustered_Get_Pair_Shift_Id(shift_bits, I),         \
                        cell);                                              \
                const VECTOR dr =                                           \
                    Get_Clustered_Shifted_Displacement(r2, r1,              \
                                                       pair_shift);         \
                const float dr2 = dr * dr;                                  \
                if (dr2 < cutoff_sq && dr2 != 0.0f)                         \
                {                                                           \
                    const float inv_r = rsqrtf(dr2);                        \
                    const float inv_r2 = inv_r * inv_r;                     \
                    const float inv_r6 = inv_r2 * inv_r2 * inv_r2;          \
                    const float beta_dr = pme_beta * (dr2 * inv_r);        \
                    const float charge_product = r1.charge * r2.charge;     \
                    const int atom_pair_LJ_type =                           \
                        Get_LJ_Type(r1.LJ_type, r2.LJ_type);                \
                    const float2 AB =                                       \
                        LJ_type_AB_packed[atom_pair_LJ_type];               \
                    const float A = AB.x;                                   \
                    const float B = AB.y;                                   \
                    const float ij_factor =                                 \
                        atom_j_is_local != 0 ? 1.0f : 0.5f;                 \
                    float frc_abs =                                         \
                        Get_Clustered_LJ_Force_Abs(inv_r2, inv_r6, A, B);   \
                    frc_abs -=                                               \
                        Get_Clustered_Direct_Coulomb_Force_Abs_PME_Corr(    \
                            charge_product, inv_r, inv_r2, beta2 * dr2,     \
                            beta3);                                          \
                    const float frc_x = frc_abs * dr.x;                     \
                    const float frc_y = frc_abs * dr.y;                     \
                    const float frc_z = frc_abs * dr.z;                     \
                    fci_x_##I += frc_x;                                     \
                    fci_y_##I += frc_y;                                     \
                    fci_z_##I += frc_z;                                     \
                    if (atom_j_is_local != 0)                               \
                    {                                                       \
                        fcj_x -= frc_x;                                     \
                        fcj_y -= frc_y;                                     \
                        fcj_z -= frc_z;                                     \
                    }                                                       \
                    const LTMatrix3 pair_virial =                           \
                        -ij_factor *                                        \
                        Get_Virial_From_Force_Dis(                          \
                            VECTOR{frc_x, frc_y, frc_z}, dr);               \
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
                            ij_factor *                                     \
                            Get_Clustered_LJ_Energy(inv_r6, A, B);          \
                        const float pair_energy_coulomb =                   \
                            ij_factor *                                     \
                            Get_Clustered_Direct_Coulomb_Energy(            \
                                charge_product, inv_r, beta_dr);            \
                        if constexpr (total_output)                         \
                        {                                                   \
                            output_buf.energy_lj_total +=                   \
                                pair_energy_lj;                             \
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
        CLUSTERED_STEADY_I_LOCAL_LIST(CLUSTERED_ACCUMULATE_I_LOCAL)
#undef CLUSTERED_ACCUMULATE_I_LOCAL

        if (atom_j_is_local != 0)
        {
            Reduce_Clustered_Subgroup_Vector_Components(fcj_x, fcj_y, fcj_z,
                                                        lane, cluster_size);
            if (i_lane == 0)
            {
                if constexpr (force_soa)
                {
                    atomicAdd(frc_x + sorted_j, fcj_x);
                    atomicAdd(frc_y + sorted_j, fcj_y);
                    atomicAdd(frc_z + sorted_j, fcj_z);
                }
                else
                {
                    atomicAdd(frc + atom_j, VECTOR{fcj_x, fcj_y, fcj_z});
                }
            }
        }
    }

#define CLUSTERED_REDUCE_I_LOCAL(I)                                        \
    if ((I) < active_cluster_count)                                        \
    {                                                                      \
        const bool active_i =                                              \
            (active_i_mask & (1u << static_cast<unsigned int>(I))) != 0u;  \
        if constexpr (!total_output)                                       \
        {                                                                  \
            LTMatrix3 reduced_virial =                                     \
                active_i ? output_buf.virial[I] : LTMatrix3(0.0f);         \
            reduced_virial = Reduce_Clustered_Warp_Virial_Over_J(          \
                reduced_virial, cluster_size);                             \
            if (lane < cluster_size)                                       \
            {                                                              \
                if (warp_id == 0)                                          \
                {                                                          \
                    output_buf.virial[I] = reduced_virial;                 \
                }                                                          \
                else                                                       \
                {                                                          \
                    warp1_i_virial_lo[I][lane] =                           \
                        Pack_Clustered_Virial_Lo(reduced_virial);          \
                    warp1_i_virial_hi[I][lane] =                           \
                        Pack_Clustered_Virial_Hi(reduced_virial);          \
                }                                                          \
            }                                                              \
        }                                                                  \
        float reduced_x = active_i ? fci_x_##I : 0.0f;                     \
        float reduced_y = active_i ? fci_y_##I : 0.0f;                     \
        float reduced_z = active_i ? fci_z_##I : 0.0f;                     \
        Reduce_Clustered_Warp_Vector_Over_J_Components(                    \
            reduced_x, reduced_y, reduced_z, cluster_size);                \
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
                warp1_i_force[I][lane] = {reduced_x, reduced_y,            \
                                          reduced_z, 0.0f};                \
            }                                                              \
        }                                                                  \
        if constexpr (need_energy)                                         \
        {                                                                  \
            if constexpr (!total_output)                                   \
            {                                                              \
                float reduced_lj = active_i ? output_buf.energy_lj[I]      \
                                            : 0.0f;                        \
                float reduced_coulomb =                                    \
                    active_i ? output_buf.energy_coulomb[I] : 0.0f;        \
                reduced_lj = Reduce_Clustered_Warp_Float_Over_J(           \
                    reduced_lj, cluster_size);                             \
                reduced_coulomb = Reduce_Clustered_Warp_Float_Over_J(      \
                    reduced_coulomb, cluster_size);                        \
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
    CLUSTERED_STEADY_I_LOCAL_LIST(CLUSTERED_REDUCE_I_LOCAL)
#undef CLUSTERED_REDUCE_I_LOCAL
    __syncthreads();

    if (warp_id == 0 && j_lane == 0)
    {
#define CLUSTERED_WRITEBACK_I_LOCAL(I)                                       \
        if ((I) < active_cluster_count)                                      \
        {                                                                    \
            const bool active_i =                                            \
                (active_i_mask &                                             \
                 (1u << static_cast<unsigned int>(I))) != 0u;                \
            if (active_i)                                                    \
            {                                                                \
                int atom_i = -1;                                             \
                if constexpr (need_atom_i_ids)                               \
                {                                                            \
                    atom_i = shared_i_atom_ids[(I) * cluster_size +          \
                                               i_lane];                      \
                }                                                            \
                const int sorted_i =                                         \
                    shared_i_sorted_ids[(I) * cluster_size + i_lane];        \
                const float4 warp1_force = warp1_i_force[I][i_lane];         \
                if constexpr (force_soa)                                     \
                {                                                            \
                    atomicAdd(frc_x + sorted_i, fci_x_##I + warp1_force.x);  \
                    atomicAdd(frc_y + sorted_i, fci_y_##I + warp1_force.y);  \
                    atomicAdd(frc_z + sorted_i, fci_z_##I + warp1_force.z);  \
                }                                                            \
                else                                                         \
                {                                                            \
                    atomicAdd(frc + atom_i,                                  \
                              VECTOR{fci_x_##I + warp1_force.x,              \
                                     fci_y_##I + warp1_force.y,              \
                                     fci_z_##I + warp1_force.z});            \
                }                                                            \
                if constexpr (!total_output)                                 \
                {                                                            \
                    if constexpr (need_energy)                               \
                    {                                                        \
                        const float total_energy_lj =                        \
                            output_buf.energy_lj[I] +                        \
                            warp1_i_energy_lj[I][i_lane];                    \
                        const float total_energy_coulomb =                   \
                            output_buf.energy_coulomb[I] +                   \
                            warp1_i_energy_coulomb[I][i_lane];               \
                        atomicAdd(atom_energy + atom_i,                      \
                                  total_energy_lj + total_energy_coulomb);   \
                        atomicAdd(atom_LJ_ene + atom_i, total_energy_lj);    \
                        atomicAdd(atom_direct_cf_energy + atom_i,            \
                                  total_energy_coulomb);                     \
                    }                                                        \
                    atomicAdd(                                               \
                        atom_virial + atom_i,                                \
                        output_buf.virial[I] +                               \
                            Unpack_Clustered_Virial(                         \
                                warp1_i_virial_lo[I][i_lane],                \
                                warp1_i_virial_hi[I][i_lane]));              \
                }                                                            \
            }                                                                \
        }
        CLUSTERED_STEADY_I_LOCAL_LIST(CLUSTERED_WRITEBACK_I_LOCAL)
#undef CLUSTERED_WRITEBACK_I_LOCAL
    }

    if constexpr (total_output)
    {
        LTMatrix3 reduced_total_virial =
            Reduce_Clustered_Warp_Virial_All(output_buf.virial_total);
        float reduced_total_energy_lj = 0.0f;
        float reduced_total_energy_coulomb = 0.0f;
        if constexpr (need_energy)
        {
            reduced_total_energy_lj =
                Reduce_Clustered_Warp_Float_All(output_buf.energy_lj_total);
            reduced_total_energy_coulomb = Reduce_Clustered_Warp_Float_All(
                output_buf.energy_coulomb_total);
        }
        if (lane == 0)
        {
            shared_total_virial_lo[warp_id] =
                Pack_Clustered_Virial_Lo(reduced_total_virial);
            shared_total_virial_hi[warp_id] =
                Pack_Clustered_Virial_Hi(reduced_total_virial);
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
            LTMatrix3 block_total_virial = Unpack_Clustered_Virial(
                shared_total_virial_lo[0], shared_total_virial_hi[0]);
            for (int warp = 1; warp < max_block_warps; warp += 1)
            {
                block_total_virial =
                    block_total_virial +
                    Unpack_Clustered_Virial(shared_total_virial_lo[warp],
                                            shared_total_virial_hi[warp]);
            }
            atomicAdd(atom_virial, block_total_virial);
            if constexpr (need_energy)
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
                atomicAdd(atom_LJ_ene, block_total_energy_lj);
                atomicAdd(atom_direct_cf_energy, block_total_energy_coulomb);
            }
        }
    }

#undef CLUSTERED_STEADY_I_LOCAL_LIST
}

template <bool need_energy>
static __global__ void
Nbnxm_Grouped_Lennard_Jones_And_Direct_Coulomb_Virial_Warp_Record_Total_Device(
    const int candidate_sci_numbers, const int cluster_size,
    const int super_cluster_clusters, const int local_atom_numbers,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const int* super_cluster_offsets, const int* sci_supercluster_ids,
    const int* grouped_sci_offsets, const int* grouped_sci_ids,
    const LJ_CLUSTERED_SCI* sci_entries,
    const int* compact_record_offsets,
    const LJ_CLUSTERED_WARP_J_RECORD* compact_warp_j_records,
    const uint64_t* pair_shift_bits, const float4* sorted_xq,
    const int* sorted_lj_type, const LTMatrix3 cell, const float* LJ_type_A,
    const float* LJ_type_B, const float cutoff, float* frc_x, float* frc_y,
    float* frc_z, const float pme_beta, float* atom_energy,
    LTMatrix3* atom_virial, float* atom_direct_cf_energy, float* atom_LJ_ene)
{
    constexpr int max_cluster_size = kClusteredClusterSize;
    constexpr int max_super_cluster_atoms =
        kClusteredClusterSize * kClusteredSuperClusterClusters;
    constexpr int max_block_warps = 2;
#ifdef USE_GPU
    const int candidate_sci = blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (candidate_sci >= candidate_sci_numbers ||
        tid >= super_cluster_clusters * cluster_size)
    {
        return;
    }
#else
    (void)candidate_sci_numbers;
    (void)cluster_size;
    (void)super_cluster_clusters;
    (void)local_atom_numbers;
    (void)cluster_offsets;
    (void)cluster_valid_masks;
    (void)cluster_local_masks;
    (void)super_cluster_offsets;
    (void)sci_supercluster_ids;
    (void)grouped_sci_offsets;
    (void)grouped_sci_ids;
    (void)sci_entries;
    (void)compact_record_offsets;
    (void)compact_warp_j_records;
    (void)pair_shift_bits;
    (void)sorted_xq;
    (void)sorted_lj_type;
    (void)cell;
    (void)LJ_type_A;
    (void)LJ_type_B;
    (void)cutoff;
    (void)frc_x;
    (void)frc_y;
    (void)frc_z;
    (void)pme_beta;
    (void)atom_energy;
    (void)atom_virial;
    (void)atom_direct_cf_energy;
    (void)atom_LJ_ene;
    return;
#endif

    const int super_i = sci_supercluster_ids[candidate_sci];
    const int grouped_sci_begin = grouped_sci_offsets[super_i];
    const int grouped_sci_end = grouped_sci_offsets[super_i + 1];
    if (grouped_sci_begin >= grouped_sci_end)
    {
        return;
    }

    const int cluster_i_start = super_cluster_offsets[super_i];
    const int cluster_i_end = super_cluster_offsets[super_i + 1];
    const float cutoff_sq = cutoff * cutoff;

    __shared__ float4 shared_i_xq[max_super_cluster_atoms];
    __shared__ int shared_i_lj_type[max_super_cluster_atoms];
    __shared__ int shared_i_sorted_ids[max_super_cluster_atoms];
    __shared__ unsigned int shared_i_valid_masks[kClusteredSuperClusterClusters];
    __shared__ unsigned int shared_i_local_masks[kClusteredSuperClusterClusters];
    __shared__ int shared_i_cluster_ids[kClusteredSuperClusterClusters];
    __shared__ float4 warp1_i_force[kClusteredSuperClusterClusters]
                                    [max_cluster_size];
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
    const int warp_j_base = warp_id * kClusteredSplitJClusterSize;
    const int warp_j_local = j_lane - warp_j_base;

#define CLUSTERED_STEADY_I_LOCAL_LIST(OP) \
    OP(0)                                 \
    OP(1)                                 \
    OP(2)                                 \
    OP(3)                                 \
    OP(4)                                 \
    OP(5)                                 \
    OP(6)                                 \
    OP(7)

#define CLUSTERED_DECLARE_FCI_BUF(I) \
    float fci_x_##I = 0.0f;         \
    float fci_y_##I = 0.0f;         \
    float fci_z_##I = 0.0f;
    CLUSTERED_STEADY_I_LOCAL_LIST(CLUSTERED_DECLARE_FCI_BUF)
#undef CLUSTERED_DECLARE_FCI_BUF
    Clustered_Full_Record_Output_Buffer<true, need_energy,
                                        kClusteredSuperClusterClusters>
        output_buf;

    if (j_lane == 0)
    {
        if (i_lane < active_cluster_count)
        {
            const int cluster_i = cluster_i_start + i_lane;
            shared_i_valid_masks[i_lane] = cluster_valid_masks[cluster_i];
            shared_i_local_masks[i_lane] = cluster_local_masks[cluster_i];
            shared_i_cluster_ids[i_lane] = cluster_i;
        }
        else if (i_lane < kClusteredSuperClusterClusters)
        {
            shared_i_valid_masks[i_lane] = 0u;
            shared_i_local_masks[i_lane] = 0u;
            shared_i_cluster_ids[i_lane] = -1;
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
            shared_i_sorted_ids[i_slot] = sorted_atom_i;
        }
        else
        {
            shared_i_sorted_ids[i_slot] = -1;
        }
    }
    __syncthreads();

    for (int grouped_sci = grouped_sci_begin; grouped_sci < grouped_sci_end;
         grouped_sci += 1)
    {
        const int sci = grouped_sci_ids[grouped_sci];
        const LJ_CLUSTERED_SCI sci_entry = sci_entries[sci];
        const int record_begin = compact_record_offsets[sci];
        const int record_end = compact_record_offsets[sci + 1];
        const bool sci_is_central =
            sci_entry.shift_id == kClusteredCentralShiftId;

        for (int record_idx = record_begin + warp_id; record_idx < record_end;
             record_idx += max_block_warps)
        {
            const LJ_CLUSTERED_WARP_J_RECORD* record =
                compact_warp_j_records + record_idx;
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
            cluster_j = deviceShfl(FULL_MASK, cluster_j, 0, warpSize);
            pair_shift_index =
                deviceShfl(FULL_MASK, pair_shift_index, 0, warpSize);
            valid_mask_j = deviceShfl(FULL_MASK, valid_mask_j, 0, warpSize);
            imask = deviceShfl(FULL_MASK, imask, 0, warpSize);

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
            shift_bits = Broadcast_Clustered_Warp_U64(shift_bits, 0);
            local_mask_j = deviceShfl(FULL_MASK, local_mask_j, 0, warpSize);
            j_lane_base = deviceShfl(FULL_MASK, j_lane_base, 0, warpSize);

            if (cluster_j < 0 ||
                (valid_mask_j &
                 (1u << static_cast<unsigned int>(warp_j_local))) == 0u)
            {
                continue;
            }

            int sorted_j = -1;
            int absolute_j_lane = -1;
            float4 r2_xq = {0.0f, 0.0f, 0.0f, 0.0f};
            int r2_lj_type = 0;
            if (i_lane == 0)
            {
                sorted_j = record->sorted_j_base + warp_j_local;
                absolute_j_lane =
                    static_cast<int>(j_lane_base) + warp_j_local;
                r2_xq = sorted_xq[sorted_j];
                r2_lj_type = sorted_lj_type[sorted_j];
            }
            sorted_j = Broadcast_Clustered_Subgroup_Value(sorted_j, lane,
                                                          cluster_size);
            absolute_j_lane = Broadcast_Clustered_Subgroup_Value(
                absolute_j_lane, lane, cluster_size);
            r2_xq =
                Broadcast_Clustered_Subgroup_Float4(r2_xq, lane, cluster_size);
            r2_lj_type = Broadcast_Clustered_Subgroup_Value(r2_lj_type, lane,
                                                            cluster_size);
            const int atom_j_is_local =
                (local_mask_j &
                 (1u << static_cast<unsigned int>(warp_j_local))) != 0u
                    ? 1
                    : 0;
            const VECTOR_LJ r2 = Make_Packed_LJ_Atom(r2_xq, r2_lj_type);
            const unsigned char pair_excl_mask =
                record->pair_excl[warp_j_local * cluster_size + i_lane];
            float fcj_x = 0.0f;
            float fcj_y = 0.0f;
            float fcj_z = 0.0f;

#define CLUSTERED_ACCUMULATE_I_LOCAL(I)                                     \
            if ((I) < active_cluster_count &&                               \
                (imask & (1u << static_cast<unsigned int>(I))) != 0u)       \
            {                                                               \
                const unsigned int valid_mask_i = shared_i_valid_masks[I];  \
                const unsigned int local_mask_i = shared_i_local_masks[I];  \
                if ((valid_mask_i & (1u << i_lane)) != 0u &&                \
                    (local_mask_i & (1u << i_lane)) != 0u)                  \
                {                                                           \
                    const int cluster_i = shared_i_cluster_ids[I];          \
                    if (!(sci_is_central && cluster_i == cluster_j &&       \
                          atom_j_is_local != 0 &&                           \
                          absolute_j_lane <= i_lane) &&                     \
                        (pair_excl_mask &                                   \
                         (1u << static_cast<unsigned int>(I))) == 0u)       \
                    {                                                       \
                        const float4 r1_xq =                                \
                            shared_i_xq[(I) * cluster_size + i_lane];       \
                        const int r1_lj_type =                              \
                            shared_i_lj_type[(I) * cluster_size + i_lane];  \
                        const VECTOR_LJ r1 =                                \
                            Make_Packed_LJ_Atom(r1_xq, r1_lj_type);         \
                        const VECTOR pair_shift =                           \
                            Clustered_Shift_Vector_From_Id(                 \
                                Clustered_Get_Pair_Shift_Id(shift_bits, I), \
                                cell);                                      \
                        const VECTOR dr =                                   \
                            Get_Clustered_Shifted_Displacement(             \
                                r2, r1, pair_shift);                        \
                        const float dr2 = dr * dr;                          \
                        if (dr2 < cutoff_sq && dr2 != 0.0f)                 \
                        {                                                   \
                            const float inv_r = rsqrtf(dr2);                \
                            const float inv_r2 = inv_r * inv_r;             \
                            const float inv_r6 = inv_r2 * inv_r2 * inv_r2;  \
                            const float beta_dr = pme_beta * (dr2 * inv_r); \
                            const float charge_product =                    \
                                r1.charge * r2.charge;                      \
                            const int atom_pair_LJ_type =                   \
                                Get_LJ_Type(r1.LJ_type, r2.LJ_type);        \
                            const float A = LJ_type_A[atom_pair_LJ_type];   \
                            const float B = LJ_type_B[atom_pair_LJ_type];   \
                            const float ij_factor =                         \
                                atom_j_is_local != 0 ? 1.0f : 0.5f;        \
                            const float frc_abs =                           \
                                Get_Clustered_LJ_Force_Abs(inv_r2, inv_r6,  \
                                                            A, B) -         \
                                Get_Clustered_Direct_Coulomb_Force_Abs(     \
                                    charge_product, inv_r, inv_r2, beta_dr); \
                            const float frc_x = frc_abs * dr.x;             \
                            const float frc_y = frc_abs * dr.y;             \
                            const float frc_z = frc_abs * dr.z;             \
                            fci_x_##I += frc_x;                             \
                            fci_y_##I += frc_y;                             \
                            fci_z_##I += frc_z;                             \
                            if (atom_j_is_local != 0)                       \
                            {                                               \
                                fcj_x -= frc_x;                             \
                                fcj_y -= frc_y;                             \
                                fcj_z -= frc_z;                             \
                            }                                               \
                            output_buf.virial_total =                       \
                                output_buf.virial_total -                   \
                                ij_factor *                                 \
                                Get_Virial_From_Force_Dis(                  \
                                    VECTOR{frc_x, frc_y, frc_z}, dr);       \
                            if constexpr (need_energy)                      \
                            {                                               \
                                output_buf.energy_lj_total +=               \
                                    ij_factor *                             \
                                    Get_Clustered_LJ_Energy(inv_r6, A, B);  \
                                output_buf.energy_coulomb_total +=          \
                                    ij_factor *                             \
                                    Get_Clustered_Direct_Coulomb_Energy(    \
                                        charge_product, inv_r, beta_dr);    \
                            }                                               \
                        }                                                   \
                    }                                                       \
                }                                                           \
            }
            CLUSTERED_STEADY_I_LOCAL_LIST(CLUSTERED_ACCUMULATE_I_LOCAL)
#undef CLUSTERED_ACCUMULATE_I_LOCAL

            if (atom_j_is_local != 0)
            {
                Reduce_Clustered_Subgroup_Vector_Components(fcj_x, fcj_y, fcj_z,
                                                            lane, cluster_size);
                if (i_lane == 0)
                {
                    atomicAdd(frc_x + sorted_j, fcj_x);
                    atomicAdd(frc_y + sorted_j, fcj_y);
                    atomicAdd(frc_z + sorted_j, fcj_z);
                }
            }
        }
    }

#define CLUSTERED_REDUCE_I_LOCAL(I)                                        \
    if ((I) < active_cluster_count)                                        \
    {                                                                      \
        const unsigned int valid_mask_i = shared_i_valid_masks[I];         \
        const unsigned int local_mask_i = shared_i_local_masks[I];         \
        const bool active_i =                                              \
            (valid_mask_i & (1u << i_lane)) != 0u &&                       \
            (local_mask_i & (1u << i_lane)) != 0u;                         \
        float reduced_x = active_i ? fci_x_##I : 0.0f;                     \
        float reduced_y = active_i ? fci_y_##I : 0.0f;                     \
        float reduced_z = active_i ? fci_z_##I : 0.0f;                     \
        Reduce_Clustered_Warp_Vector_Over_J_Components(                    \
            reduced_x, reduced_y, reduced_z, cluster_size);                \
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
                warp1_i_force[I][lane] = {reduced_x, reduced_y,            \
                                          reduced_z, 0.0f};                \
            }                                                              \
        }                                                                  \
    }
    CLUSTERED_STEADY_I_LOCAL_LIST(CLUSTERED_REDUCE_I_LOCAL)
#undef CLUSTERED_REDUCE_I_LOCAL
    __syncthreads();

    if (warp_id == 0 && j_lane == 0)
    {
#define CLUSTERED_WRITEBACK_I_LOCAL(I)                                      \
        if ((I) < active_cluster_count)                                     \
        {                                                                   \
            const unsigned int valid_mask_i = shared_i_valid_masks[I];      \
            const unsigned int local_mask_i = shared_i_local_masks[I];      \
            const bool active_i =                                            \
                (valid_mask_i & (1u << i_lane)) != 0u &&                    \
                (local_mask_i & (1u << i_lane)) != 0u;                      \
            if (active_i)                                                   \
            {                                                               \
                const int sorted_i =                                        \
                    shared_i_sorted_ids[(I) * cluster_size + i_lane];       \
                const float4 warp1_force = warp1_i_force[I][i_lane];        \
                atomicAdd(frc_x + sorted_i, fci_x_##I + warp1_force.x);     \
                atomicAdd(frc_y + sorted_i, fci_y_##I + warp1_force.y);     \
                atomicAdd(frc_z + sorted_i, fci_z_##I + warp1_force.z);     \
            }                                                               \
        }
        CLUSTERED_STEADY_I_LOCAL_LIST(CLUSTERED_WRITEBACK_I_LOCAL)
#undef CLUSTERED_WRITEBACK_I_LOCAL
    }

    LTMatrix3 reduced_total_virial =
        Reduce_Clustered_Warp_Virial_All(output_buf.virial_total);
    float reduced_total_energy_lj = 0.0f;
    float reduced_total_energy_coulomb = 0.0f;
    if constexpr (need_energy)
    {
        reduced_total_energy_lj =
            Reduce_Clustered_Warp_Float_All(output_buf.energy_lj_total);
        reduced_total_energy_coulomb =
            Reduce_Clustered_Warp_Float_All(output_buf.energy_coulomb_total);
    }
    if (lane == 0)
    {
        shared_total_virial_lo[warp_id] =
            Pack_Clustered_Virial_Lo(reduced_total_virial);
        shared_total_virial_hi[warp_id] =
            Pack_Clustered_Virial_Hi(reduced_total_virial);
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
        LTMatrix3 block_total_virial = Unpack_Clustered_Virial(
            shared_total_virial_lo[0], shared_total_virial_hi[0]);
        for (int warp = 1; warp < max_block_warps; warp += 1)
        {
            block_total_virial =
                block_total_virial +
                Unpack_Clustered_Virial(shared_total_virial_lo[warp],
                                        shared_total_virial_hi[warp]);
        }
        atomicAdd(atom_virial, block_total_virial);
        if constexpr (need_energy)
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
            atomicAdd(atom_LJ_ene, block_total_energy_lj);
            atomicAdd(atom_direct_cf_energy, block_total_energy_coulomb);
        }
    }

#undef CLUSTERED_STEADY_I_LOCAL_LIST
}

template <bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb>
static __global__ void Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Device(
    const int sci_numbers, const int cluster_size,
    const int super_cluster_clusters, const int local_atom_numbers,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const int* super_cluster_offsets,
    const LJ_CLUSTERED_SCI* sci_entries,
    const LJ_CLUSTERED_CJ_PACKED* cj_packed_entries,
    const uint64_t* pair_shift_bits,
    const unsigned long long* exclusion_mask_pool, const int* sorted_atom_ids,
    const float4* sorted_xq, const int* sorted_lj_type, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* LJ_type_A, const float* LJ_type_B,
    const float cutoff, VECTOR* frc, const float pme_beta, float* atom_energy,
    LTMatrix3* atom_virial, float* atom_direct_cf_energy, float* atom_LJ_ene)
{
    constexpr int max_cluster_size = kClusteredClusterSize;
    constexpr int max_super_cluster_atoms =
        kClusteredClusterSize * kClusteredSuperClusterClusters;
    constexpr int max_block_warps = 2;
#ifdef USE_GPU
    const int sci = blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (sci >= sci_numbers ||
        tid >= super_cluster_clusters * cluster_size)
    {
        return;
    }
#else
#pragma omp parallel for schedule(dynamic)
    for (int sci = 0; sci < sci_numbers; sci += 1)
#endif
    {
        const LJ_CLUSTERED_SCI sci_entry = sci_entries[sci];
        const int super_i = sci_entry.supercluster_id;
        const int cluster_i_start = super_cluster_offsets[super_i];
        const int cluster_i_end = super_cluster_offsets[super_i + 1];
        const bool sci_is_central =
            sci_entry.shift_id == kClusteredCentralShiftId;
        const float cutoff_sq = cutoff * cutoff;

#ifndef USE_GPU
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
                const int sorted_atom_i = cluster_offsets[cluster_i] + lane_i;
                const int atom_i = sorted_atom_ids[sorted_atom_i];
                VECTOR_LJ r1 = Make_Packed_LJ_Atom(sorted_xq[sorted_atom_i],
                                                   sorted_lj_type[sorted_atom_i]);
                VECTOR frc_i = {0.0f, 0.0f, 0.0f};
                float energy_lj = 0.0f;
                float energy_coulomb = 0.0f;
                LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

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
                            Clustered_First_Exclusion_Index(packed, jm, i_local);
                        const unsigned long long exclusion_mask =
                            exclusion_index >= 0
                                ? exclusion_mask_pool[exclusion_index]
                                : 0ull;
                        const VECTOR pair_shift_vec =
                            Get_Clustered_Pair_Shift_Vector(
                                cluster_centers[cluster_i],
                                cluster_centers[cluster_j], cell, rcell);
                        VECTOR frc_j[max_cluster_size] = {};

                        for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
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
                                if (atom_j < local_atom_numbers)
                                {
                                    frc_j[lane_j] = frc_j[lane_j] - frc_lin;
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
                                energy_lj +=
                                    ij_factor *
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
                        if (need_force)
                        {
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
                }

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
                        atomicAdd(atom_direct_cf_energy + atom_i,
                                  energy_coulomb);
                    }
                }
                if (need_virial)
                {
                    atomicAdd(atom_virial + atom_i, virial);
                }
            }
        }
#else
        if constexpr (need_virial)
        {
            __shared__ float4 shared_i_xq[max_super_cluster_atoms];
            __shared__ int shared_i_lj_type[max_super_cluster_atoms];
            __shared__ int shared_i_atom_ids[max_super_cluster_atoms];
            __shared__ int shared_i_sorted_ids[max_super_cluster_atoms];
            __shared__ unsigned int shared_i_valid_masks[kClusteredSuperClusterClusters];
            __shared__ unsigned int shared_i_local_masks[kClusteredSuperClusterClusters];
            __shared__ int shared_i_cluster_ids[kClusteredSuperClusterClusters];
            __shared__ float4 warp1_i_force[kClusteredSuperClusterClusters]
                                            [max_cluster_size];
            __shared__ float warp1_i_energy_lj[kClusteredSuperClusterClusters]
                                              [max_cluster_size];
            __shared__ float warp1_i_energy_coulomb
                [kClusteredSuperClusterClusters][max_cluster_size];
            __shared__ float4 warp1_i_virial_lo
                [kClusteredSuperClusterClusters][max_cluster_size];
            __shared__ float2 warp1_i_virial_hi
                [kClusteredSuperClusterClusters][max_cluster_size];
            __shared__ unsigned long long
                shared_exclusion_masks[max_block_warps]
                                      [kClusteredSuperClusterClusters];
            __shared__ unsigned int shared_j_valid_masks[max_block_warps];
            __shared__ float4 shared_j_xq[max_block_warps]
                                         [kClusteredSplitJClusterSize];
            __shared__ int shared_j_lj_type[max_block_warps]
                                           [kClusteredSplitJClusterSize];
            __shared__ int shared_j_atom_ids[max_block_warps]
                                            [kClusteredSplitJClusterSize];
            __shared__ int shared_j_sorted_ids[max_block_warps]
                                              [kClusteredSplitJClusterSize];
            __shared__ int shared_j_local_flags[max_block_warps]
                                               [kClusteredSplitJClusterSize];

            const int i_lane = threadIdx.x;
            const int j_lane = threadIdx.y;
            const int lane = tid & (warpSize - 1);
            const int warp_id = tid / warpSize;
            const int active_cluster_count = cluster_i_end - cluster_i_start;
            const int i_slot = j_lane * cluster_size + i_lane;
            const int warp_j_base = warp_id * kClusteredSplitJClusterSize;
            const int warp_j_local = j_lane - warp_j_base;

#define CLUSTERED_STEADY_I_LOCAL_LIST(OP) \
    OP(0)                                 \
    OP(1)                                 \
    OP(2)                                 \
    OP(3)                                 \
    OP(4)                                 \
    OP(5)                                 \
    OP(6)                                 \
    OP(7)

#define CLUSTERED_DECLARE_FCI_BUF(I) \
    float fci_x_##I = 0.0f;         \
    float fci_y_##I = 0.0f;         \
    float fci_z_##I = 0.0f;
            CLUSTERED_STEADY_I_LOCAL_LIST(CLUSTERED_DECLARE_FCI_BUF)
#undef CLUSTERED_DECLARE_FCI_BUF
            float energy_lj_buf[kClusteredSuperClusterClusters] = {};
            float energy_coulomb_buf[kClusteredSuperClusterClusters] = {};
            LTMatrix3 virial_buf[kClusteredSuperClusterClusters];
            for (int i_local = 0; i_local < kClusteredSuperClusterClusters;
                 i_local += 1)
            {
                virial_buf[i_local] = {0.0f, 0.0f, 0.0f,
                                       0.0f, 0.0f, 0.0f};
            }

            if (j_lane == 0)
            {
                if (i_lane < active_cluster_count)
                {
                    const int cluster_i = cluster_i_start + i_lane;
                    shared_i_valid_masks[i_lane] = cluster_valid_masks[cluster_i];
                    shared_i_local_masks[i_lane] = cluster_local_masks[cluster_i];
                    shared_i_cluster_ids[i_lane] = cluster_i;
                }
                else if (i_lane < kClusteredSuperClusterClusters)
                {
                    shared_i_valid_masks[i_lane] = 0u;
                    shared_i_local_masks[i_lane] = 0u;
                    shared_i_cluster_ids[i_lane] = -1;
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

            for (int packed_idx = sci_entry.cjpacked_begin;
                 packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
            {
                const LJ_CLUSTERED_CJ_PACKED* packed =
                    cj_packed_entries + packed_idx;
                const LJ_CLUSTERED_IMEI* imei = packed->imei + warp_id;
                for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
                {
                    const int cluster_j = packed->cj[jm];
                    const unsigned int imask =
                        cluster_j >= 0 ? Clustered_Jm_Imask(*imei, jm) : 0u;
                    const uint64_t shift_bits =
                        pair_shift_bits[packed_idx * kClusteredJGroupSize + jm];
                    if (j_lane == warp_j_base)
                    {
                        const unsigned int valid_mask_j =
                            cluster_j >= 0 ? cluster_valid_masks[cluster_j] : 0u;
                        if (i_lane == 0)
                        {
                            shared_j_valid_masks[warp_id] = valid_mask_j;
                        }
                        if (i_lane < kClusteredSplitJClusterSize)
                        {
                            const int j_local = warp_j_base + i_lane;
                            if ((valid_mask_j & (1u << j_local)) != 0u)
                            {
                                const int sorted_atom_j =
                                    cluster_offsets[cluster_j] + j_local;
                                const int atom_j =
                                    sorted_atom_ids[sorted_atom_j];
                                shared_j_atom_ids[warp_id][i_lane] = atom_j;
                                shared_j_sorted_ids[warp_id][i_lane] =
                                    sorted_atom_j;
                                shared_j_local_flags[warp_id][i_lane] =
                                    atom_j < local_atom_numbers ? 1 : 0;
                                shared_j_xq[warp_id][i_lane] =
                                    sorted_xq[sorted_atom_j];
                                shared_j_lj_type[warp_id][i_lane] =
                                    sorted_lj_type[sorted_atom_j];
                            }
                            else
                            {
                                shared_j_atom_ids[warp_id][i_lane] = -1;
                                shared_j_sorted_ids[warp_id][i_lane] = -1;
                                shared_j_local_flags[warp_id][i_lane] = 0;
                                shared_j_xq[warp_id][i_lane] = {
                                    0.0f, 0.0f, 0.0f, 0.0f};
                                shared_j_lj_type[warp_id][i_lane] = 0;
                            }
                        }
                        if (i_lane < active_cluster_count && cluster_j >= 0 &&
                            (imask & (1u << i_lane)) != 0u)
                        {
                            const int exclusion_index =
                                Clustered_Exclusion_Index(*imei, jm, i_lane);
                            shared_exclusion_masks[warp_id][i_lane] =
                                exclusion_index >= 0
                                    ? exclusion_mask_pool[exclusion_index]
                                    : 0ull;
                        }
                        else if (i_lane < kClusteredSuperClusterClusters)
                        {
                            shared_exclusion_masks[warp_id][i_lane] = 0ull;
                        }
                    }
                    __syncwarp(FULL_MASK);

                    if (cluster_j >= 0 && imask != 0u)
                    {
                        const unsigned int valid_mask_j =
                            shared_j_valid_masks[warp_id];
                        if ((valid_mask_j & (1u << j_lane)) != 0u)
                        {
                            const int atom_j =
                                shared_j_atom_ids[warp_id][warp_j_local];
                            const int sorted_atom_j =
                                shared_j_sorted_ids[warp_id][warp_j_local];
                            const int atom_j_is_local =
                                shared_j_local_flags[warp_id][warp_j_local];
                            const float4 r2_xq =
                                shared_j_xq[warp_id][warp_j_local];
                            const int r2_lj_type =
                                shared_j_lj_type[warp_id][warp_j_local];
                            const VECTOR_LJ r2 =
                                Make_Packed_LJ_Atom(r2_xq, r2_lj_type);
                            float fcj_x = 0.0f;
                            float fcj_y = 0.0f;
                            float fcj_z = 0.0f;
                            for (int i_local = 0;
                                 i_local < active_cluster_count; i_local += 1)
                            {
                                if ((imask & (1u << i_local)) == 0u)
                                {
                                    continue;
                                }
                                const unsigned int valid_mask_i =
                                    shared_i_valid_masks[i_local];
                                const unsigned int local_mask_i =
                                    shared_i_local_masks[i_local];
                                if ((valid_mask_i & (1u << i_lane)) == 0u ||
                                    (local_mask_i & (1u << i_lane)) == 0u)
                                {
                                    continue;
                                }
                                const int cluster_i =
                                    shared_i_cluster_ids[i_local];
                                const unsigned long long exclusion_mask =
                                    shared_exclusion_masks[warp_id][i_local];
                                if (sci_is_central && cluster_i == cluster_j &&
                                    atom_j < local_atom_numbers &&
                                    j_lane <= i_lane)
                                {
                                    continue;
                                }
                                if ((exclusion_mask &
                                     (1ull << (i_lane * cluster_size + j_lane))) !=
                                    0ull)
                                {
                                    continue;
                                }
                                const float4 r1_xq =
                                    shared_i_xq[i_local * cluster_size + i_lane];
                                const int r1_lj_type =
                                    shared_i_lj_type[i_local * cluster_size + i_lane];
                                const VECTOR_LJ r1 =
                                    Make_Packed_LJ_Atom(r1_xq, r1_lj_type);
                                const VECTOR pair_shift =
                                    Clustered_Shift_Vector_From_Id(
                                        Clustered_Get_Pair_Shift_Id(
                                            shift_bits, i_local),
                                        cell);
                                const VECTOR dr =
                                    Get_Clustered_Shifted_Displacement(
                                        r2, r1, pair_shift);
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
                                    atom_j_is_local != 0 ? 1.0f : 0.5f;

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
                                    switch (i_local)
                                    {
#define CLUSTERED_ACCUMULATE_FORCE_CASE(I)                  \
                                    case I:                \
                                        fci_x_##I += frc_lin.x; \
                                        fci_y_##I += frc_lin.y; \
                                        fci_z_##I += frc_lin.z; \
                                        break;
                                        CLUSTERED_STEADY_I_LOCAL_LIST(
                                            CLUSTERED_ACCUMULATE_FORCE_CASE)
#undef CLUSTERED_ACCUMULATE_FORCE_CASE
                                    default:
                                        break;
                                    }
                                    if (atom_j_is_local != 0)
                                    {
                                        fcj_x -= frc_lin.x;
                                        fcj_y -= frc_lin.y;
                                        fcj_z -= frc_lin.z;
                                    }
                                    virial_buf[i_local] =
                                        virial_buf[i_local] -
                                        ij_factor *
                                            Get_Virial_From_Force_Dis(
                                                frc_lin, dr);
                                }
                                if (need_energy)
                                {
                                    energy_lj_buf[i_local] +=
                                        ij_factor *
                                        Get_LJ_Energy(r1, r2, dr_abs, A, B);
                                    if (need_coulomb)
                                    {
                                        energy_coulomb_buf[i_local] +=
                                            ij_factor *
                                            Get_Direct_Coulomb_Energy(
                                                r1, r2, dr_abs, pme_beta);
                                    }
                                }
                            }

                            if (need_force && atom_j_is_local != 0)
                            {
                                Reduce_Clustered_Subgroup_Vector_Components(
                                    fcj_x, fcj_y, fcj_z, lane, cluster_size);
                                if (i_lane == 0)
                                {
                                    atomicAdd(
                                        frc + sorted_atom_j,
                                        VECTOR{fcj_x, fcj_y, fcj_z});
                                }
                            }
                        }
                    }
                }
            }
            for (int i_local = 0; i_local < active_cluster_count; i_local += 1)
            {
                const unsigned int valid_mask_i = shared_i_valid_masks[i_local];
                const unsigned int local_mask_i = shared_i_local_masks[i_local];
                const bool active_i =
                    (valid_mask_i & (1u << i_lane)) != 0u &&
                    (local_mask_i & (1u << i_lane)) != 0u;
                LTMatrix3 reduced = active_i ? virial_buf[i_local]
                                             : LTMatrix3(0.0f);
                reduced =
                    Reduce_Clustered_Warp_Virial_Over_J(reduced, cluster_size);
                if (lane < cluster_size)
                {
                    if (warp_id == 0)
                    {
                        virial_buf[i_local] = reduced;
                    }
                    else
                    {
                        warp1_i_virial_lo[i_local][lane] =
                            Pack_Clustered_Virial_Lo(reduced);
                        warp1_i_virial_hi[i_local][lane] =
                            Pack_Clustered_Virial_Hi(reduced);
                    }
                }
                if (need_force)
                {
                    float reduced_x = 0.0f;
                    float reduced_y = 0.0f;
                    float reduced_z = 0.0f;
                    if (active_i)
                    {
                        switch (i_local)
                        {
#define CLUSTERED_LOAD_FORCE_CASE(I) \
                        case I:      \
                            reduced_x = fci_x_##I; \
                            reduced_y = fci_y_##I; \
                            reduced_z = fci_z_##I; \
                            break;
                            CLUSTERED_STEADY_I_LOCAL_LIST(
                                CLUSTERED_LOAD_FORCE_CASE)
#undef CLUSTERED_LOAD_FORCE_CASE
                        default:
                            break;
                        }
                    }
                    Reduce_Clustered_Warp_Vector_Over_J_Components(
                        reduced_x, reduced_y, reduced_z, cluster_size);
                    if (lane < cluster_size)
                    {
                        if (warp_id == 0)
                        {
                            switch (i_local)
                            {
#define CLUSTERED_STORE_FORCE_CASE(I) \
                            case I:   \
                                fci_x_##I = reduced_x; \
                                fci_y_##I = reduced_y; \
                                fci_z_##I = reduced_z; \
                                break;
                                CLUSTERED_STEADY_I_LOCAL_LIST(
                                    CLUSTERED_STORE_FORCE_CASE)
#undef CLUSTERED_STORE_FORCE_CASE
                            default:
                                break;
                            }
                        }
                        else
                        {
                            warp1_i_force[i_local][lane] = {
                                reduced_x, reduced_y, reduced_z, 0.0f};
                        }
                    }
                }
                if (need_energy)
                {
                    float reduced_lj = active_i ? energy_lj_buf[i_local] : 0.0f;
                    float reduced_coulomb =
                        active_i ? energy_coulomb_buf[i_local] : 0.0f;
                    reduced_lj = Reduce_Clustered_Warp_Float_Over_J(
                        reduced_lj, cluster_size);
                    reduced_coulomb = Reduce_Clustered_Warp_Float_Over_J(
                        reduced_coulomb, cluster_size);
                    if (lane < cluster_size)
                    {
                        if (warp_id == 0)
                        {
                            energy_lj_buf[i_local] = reduced_lj;
                            energy_coulomb_buf[i_local] = reduced_coulomb;
                        }
                        else
                        {
                            warp1_i_energy_lj[i_local][lane] = reduced_lj;
                            warp1_i_energy_coulomb[i_local][lane] =
                                reduced_coulomb;
                        }
                    }
                }
            }
            __syncthreads();

            if (warp_id == 0 && j_lane == 0)
            {
                for (int i_local = 0; i_local < active_cluster_count;
                     i_local += 1)
                {
                    const unsigned int valid_mask_i =
                        shared_i_valid_masks[i_local];
                    const unsigned int local_mask_i =
                        shared_i_local_masks[i_local];
                    const bool active_i =
                        (valid_mask_i & (1u << i_lane)) != 0u &&
                        (local_mask_i & (1u << i_lane)) != 0u;
                    if (!active_i)
                    {
                        continue;
                    }

                    const int atom_i =
                        shared_i_atom_ids[i_local * cluster_size + i_lane];
                    const int sorted_atom_i =
                        shared_i_sorted_ids[i_local * cluster_size + i_lane];
                    if (need_force)
                    {
                        float accumulated_x = 0.0f;
                        float accumulated_y = 0.0f;
                        float accumulated_z = 0.0f;
                        switch (i_local)
                        {
#define CLUSTERED_WRITEBACK_FORCE_CASE(I)   \
                        case I:             \
                            accumulated_x = fci_x_##I; \
                            accumulated_y = fci_y_##I; \
                            accumulated_z = fci_z_##I; \
                            break;
                            CLUSTERED_STEADY_I_LOCAL_LIST(
                                CLUSTERED_WRITEBACK_FORCE_CASE)
#undef CLUSTERED_WRITEBACK_FORCE_CASE
                        default:
                            break;
                        }
                        const float4 warp1_force = warp1_i_force[i_local][i_lane];
                        atomicAdd(
                            frc + sorted_atom_i,
                            VECTOR{accumulated_x + warp1_force.x,
                                   accumulated_y + warp1_force.y,
                                   accumulated_z + warp1_force.z});
                    }
                    if (need_energy)
                    {
                        const float total_energy_lj =
                            energy_lj_buf[i_local] +
                            warp1_i_energy_lj[i_local][i_lane];
                        const float total_energy_coulomb =
                            energy_coulomb_buf[i_local] +
                            warp1_i_energy_coulomb[i_local][i_lane];
                        atomicAdd(atom_energy + atom_i,
                                  total_energy_lj + total_energy_coulomb);
                        atomicAdd(atom_LJ_ene + atom_i, total_energy_lj);
                        if (need_coulomb)
                        {
                            atomicAdd(atom_direct_cf_energy + atom_i,
                                      total_energy_coulomb);
                        }
                    }
                    atomicAdd(atom_virial + atom_i,
                              virial_buf[i_local] +
                                  Unpack_Clustered_Virial(
                                      warp1_i_virial_lo[i_local][i_lane],
                                      warp1_i_virial_hi[i_local][i_lane]));
                }
            }
#undef CLUSTERED_STEADY_I_LOCAL_LIST
        }
        else
        {
            __shared__ float4 shared_i_xq[max_super_cluster_atoms];
            __shared__ int shared_i_lj_type[max_super_cluster_atoms];
            __shared__ int shared_i_atom_ids[max_super_cluster_atoms];
            __shared__ unsigned int shared_i_valid_masks[kClusteredSuperClusterClusters];
            __shared__ unsigned int shared_i_local_masks[kClusteredSuperClusterClusters];
            __shared__ int shared_i_cluster_ids[kClusteredSuperClusterClusters];
            __shared__ float4 warp1_i_force[kClusteredSuperClusterClusters]
                                            [max_cluster_size];
            __shared__ float warp1_i_energy_lj[kClusteredSuperClusterClusters]
                                              [max_cluster_size];
            __shared__ float warp1_i_energy_coulomb
                [kClusteredSuperClusterClusters][max_cluster_size];
            __shared__ float4 warp1_i_virial_lo
                [kClusteredSuperClusterClusters][max_cluster_size];
            __shared__ float2 warp1_i_virial_hi
                [kClusteredSuperClusterClusters][max_cluster_size];
            __shared__ unsigned long long
                shared_exclusion_masks[max_block_warps]
                                      [kClusteredSuperClusterClusters];
            __shared__ unsigned int shared_j_valid_masks[max_block_warps];
            __shared__ float4 shared_j_xq[max_block_warps]
                                         [kClusteredSplitJClusterSize];
            __shared__ int shared_j_lj_type[max_block_warps]
                                           [kClusteredSplitJClusterSize];
            __shared__ int shared_j_atom_ids[max_block_warps]
                                            [kClusteredSplitJClusterSize];
            __shared__ int shared_j_local_flags[max_block_warps]
                                               [kClusteredSplitJClusterSize];

            const int i_lane = threadIdx.x;
            const int j_lane = threadIdx.y;
            const int lane = tid & (warpSize - 1);
            const int warp_id = tid / warpSize;
            const int active_cluster_count = cluster_i_end - cluster_i_start;
            const int i_slot = j_lane * cluster_size + i_lane;
            const int warp_j_base = warp_id * kClusteredSplitJClusterSize;
            const int warp_j_local = j_lane - warp_j_base;

#define CLUSTERED_STEADY_I_LOCAL_LIST(OP) \
    OP(0)                                 \
    OP(1)                                 \
    OP(2)                                 \
    OP(3)                                 \
    OP(4)                                 \
    OP(5)                                 \
    OP(6)                                 \
    OP(7)

#define CLUSTERED_DECLARE_FCI_BUF(I) \
    float fci_x_##I = 0.0f;         \
    float fci_y_##I = 0.0f;         \
    float fci_z_##I = 0.0f;
            CLUSTERED_STEADY_I_LOCAL_LIST(CLUSTERED_DECLARE_FCI_BUF)
#undef CLUSTERED_DECLARE_FCI_BUF
            Clustered_Energy_Buffer<need_energy, kClusteredSuperClusterClusters>
                energy_lj_buf;
            Clustered_Energy_Buffer<need_energy, kClusteredSuperClusterClusters>
                energy_coulomb_buf;

            if (j_lane == 0)
            {
                if (i_lane < active_cluster_count)
                {
                    const int cluster_i = cluster_i_start + i_lane;
                    shared_i_valid_masks[i_lane] = cluster_valid_masks[cluster_i];
                    shared_i_local_masks[i_lane] = cluster_local_masks[cluster_i];
                    shared_i_cluster_ids[i_lane] = cluster_i;
                }
                else if (i_lane < kClusteredSuperClusterClusters)
                {
                    shared_i_valid_masks[i_lane] = 0u;
                    shared_i_local_masks[i_lane] = 0u;
                    shared_i_cluster_ids[i_lane] = -1;
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
                }
                else
                {
                    shared_i_atom_ids[i_slot] = -1;
                }
            }
            __syncthreads();

            for (int packed_idx = sci_entry.cjpacked_begin;
                 packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
            {
                const LJ_CLUSTERED_CJ_PACKED* packed =
                    cj_packed_entries + packed_idx;
                const LJ_CLUSTERED_IMEI* imei = packed->imei + warp_id;
                for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
                {
                    const int cluster_j = packed->cj[jm];
                    const unsigned int imask =
                        cluster_j >= 0 ? Clustered_Jm_Imask(*imei, jm) : 0u;
                    const uint64_t shift_bits =
                        pair_shift_bits[packed_idx * kClusteredJGroupSize + jm];
                    if (j_lane == warp_j_base)
                    {
                        const unsigned int valid_mask_j =
                            cluster_j >= 0 ? cluster_valid_masks[cluster_j] : 0u;
                        if (i_lane == 0)
                        {
                            shared_j_valid_masks[warp_id] = valid_mask_j;
                        }
                        if (i_lane < kClusteredSplitJClusterSize)
                        {
                            const int j_local = warp_j_base + i_lane;
                            if ((valid_mask_j & (1u << j_local)) != 0u)
                            {
                                const int sorted_atom_j =
                                    cluster_offsets[cluster_j] + j_local;
                                const int atom_j =
                                    sorted_atom_ids[sorted_atom_j];
                                shared_j_atom_ids[warp_id][i_lane] = atom_j;
                                shared_j_local_flags[warp_id][i_lane] =
                                    atom_j < local_atom_numbers ? 1 : 0;
                                shared_j_xq[warp_id][i_lane] =
                                    sorted_xq[sorted_atom_j];
                                shared_j_lj_type[warp_id][i_lane] =
                                    sorted_lj_type[sorted_atom_j];
                            }
                            else
                            {
                                shared_j_atom_ids[warp_id][i_lane] = -1;
                                shared_j_local_flags[warp_id][i_lane] = 0;
                                shared_j_xq[warp_id][i_lane] = {
                                    0.0f, 0.0f, 0.0f, 0.0f};
                                shared_j_lj_type[warp_id][i_lane] = 0;
                            }
                        }
                        if (i_lane < active_cluster_count && cluster_j >= 0 &&
                            (imask & (1u << i_lane)) != 0u)
                        {
                            const int exclusion_index =
                                Clustered_Exclusion_Index(*imei, jm, i_lane);
                            shared_exclusion_masks[warp_id][i_lane] =
                                exclusion_index >= 0
                                    ? exclusion_mask_pool[exclusion_index]
                                    : 0ull;
                        }
                        else if (i_lane < kClusteredSuperClusterClusters)
                        {
                            shared_exclusion_masks[warp_id][i_lane] = 0ull;
                        }
                    }
                    __syncwarp(FULL_MASK);

                    if (cluster_j >= 0 && imask != 0u)
                    {
                        const unsigned int valid_mask_j =
                            shared_j_valid_masks[warp_id];
                        if ((valid_mask_j & (1u << j_lane)) != 0u)
                        {
                            const int atom_j =
                                shared_j_atom_ids[warp_id][warp_j_local];
                            const int atom_j_is_local =
                                shared_j_local_flags[warp_id][warp_j_local];
                            const float4 r2_xq =
                                shared_j_xq[warp_id][warp_j_local];
                            const int r2_lj_type =
                                shared_j_lj_type[warp_id][warp_j_local];
                            const VECTOR_LJ r2 =
                                Make_Packed_LJ_Atom(r2_xq, r2_lj_type);
                            float fcj_x = 0.0f;
                            float fcj_y = 0.0f;
                            float fcj_z = 0.0f;
#define CLUSTERED_ACCUMULATE_I_LOCAL(I)                                   \
                            if ((I) < active_cluster_count &&              \
                                (imask & (1u << (I))) != 0u)              \
                            {                                              \
                                const unsigned int valid_mask_i =          \
                                    shared_i_valid_masks[I];               \
                                const unsigned int local_mask_i =          \
                                    shared_i_local_masks[I];               \
                                if ((valid_mask_i & (1u << i_lane)) != 0u && \
                                    (local_mask_i & (1u << i_lane)) != 0u) \
                                {                                          \
                                    const int cluster_i =                  \
                                        shared_i_cluster_ids[I];           \
                                const unsigned long long exclusion_mask =  \
                                    shared_exclusion_masks[warp_id][I];    \
                                if (!(sci_is_central &&                    \
                                      cluster_i == cluster_j &&            \
                                      atom_j < local_atom_numbers &&       \
                                      j_lane <= i_lane) &&                \
                                    (exclusion_mask &                      \
                                     (1ull <<                             \
                                      (i_lane * cluster_size +            \
                                       j_lane))) == 0ull)                 \
                                {                                          \
                                    const float4 r1_xq =                   \
                                        shared_i_xq[(I) * cluster_size +   \
                                                    i_lane];               \
                                    const int r1_lj_type =                 \
                                        shared_i_lj_type[(I) * cluster_size + \
                                                         i_lane];          \
                                    const VECTOR_LJ r1 =                   \
                                        Make_Packed_LJ_Atom(r1_xq,         \
                                                            r1_lj_type);   \
                                    const VECTOR pair_shift =              \
                                        Clustered_Shift_Vector_From_Id(     \
                                            Clustered_Get_Pair_Shift_Id(    \
                                                shift_bits, I), cell);     \
                                    const VECTOR dr =                      \
                                        Get_Clustered_Shifted_Displacement( \
                                            r2, r1,                        \
                                            pair_shift);                   \
                                    const float dr2 = dr * dr;             \
                                    if (dr2 < cutoff_sq && dr2 != 0.0f)    \
                                    {                                      \
                                        const float dr_abs = sqrtf(dr2);   \
                                        const int atom_pair_LJ_type =      \
                                            Get_LJ_Type(r1.LJ_type,        \
                                                        r2.LJ_type);       \
                                        const float A =                    \
                                            LJ_type_A[atom_pair_LJ_type];  \
                                        const float B =                    \
                                            LJ_type_B[atom_pair_LJ_type];  \
                                        const float ij_factor =            \
                                            atom_j_is_local != 0 ? 1.0f    \
                                                                 : 0.5f;   \
                                        if (need_force)                    \
                                        {                                  \
                                            float frc_abs =                \
                                                Get_LJ_Force(r1, r2,       \
                                                             dr_abs, A, B); \
                                            if (need_coulomb)              \
                                            {                              \
                                                frc_abs -=                 \
                                                    Get_Direct_Coulomb_Force( \
                                                        r1, r2, dr_abs,    \
                                                        pme_beta);         \
                                            }                              \
                                            const float frc_x =            \
                                                frc_abs * dr.x;            \
                                            const float frc_y =            \
                                                frc_abs * dr.y;            \
                                            const float frc_z =            \
                                                frc_abs * dr.z;            \
                                            fci_x_##I += frc_x;            \
                                            fci_y_##I += frc_y;            \
                                            fci_z_##I += frc_z;            \
                                            if (atom_j_is_local != 0)      \
                                            {                              \
                                                fcj_x -= frc_x;            \
                                                fcj_y -= frc_y;            \
                                                fcj_z -= frc_z;            \
                                            }                              \
                                        }                                  \
                                        if constexpr (need_energy)         \
                                        {                                  \
                                            energy_lj_buf[I] +=            \
                                                ij_factor *                \
                                                Get_LJ_Energy(r1, r2,      \
                                                              dr_abs, A,   \
                                                              B);          \
                                            if (need_coulomb)              \
                                            {                              \
                                                energy_coulomb_buf[I] +=   \
                                                    ij_factor *            \
                                                    Get_Direct_Coulomb_Energy( \
                                                        r1, r2, dr_abs,    \
                                                        pme_beta);         \
                                            }                              \
                                        }                                  \
                                    }                                      \
                                }                                          \
                                }                                          \
                            }
                            CLUSTERED_STEADY_I_LOCAL_LIST(
                                CLUSTERED_ACCUMULATE_I_LOCAL)
#undef CLUSTERED_ACCUMULATE_I_LOCAL

                            if (need_force && atom_j_is_local != 0)
                            {
                                Reduce_Clustered_Subgroup_Vector_Components(
                                    fcj_x, fcj_y, fcj_z, lane, cluster_size);
                                if (i_lane == 0)
                                {
                                    atomicAdd(
                                        frc + atom_j,
                                        VECTOR{fcj_x, fcj_y, fcj_z});
                                }
                            }
                        }
                    }
                    __syncwarp(FULL_MASK);
                }
            }

#define CLUSTERED_REDUCE_I_LOCAL(I)                                      \
            if ((I) < active_cluster_count)                               \
            {                                                              \
                const unsigned int valid_mask_i = shared_i_valid_masks[I]; \
                const unsigned int local_mask_i = shared_i_local_masks[I]; \
                const bool active_i =                                       \
                    (valid_mask_i & (1u << i_lane)) != 0u &&               \
                    (local_mask_i & (1u << i_lane)) != 0u;                 \
                                                                           \
                if (need_force)                                            \
                {                                                           \
                    float reduced_x = active_i ? fci_x_##I : 0.0f;        \
                    float reduced_y = active_i ? fci_y_##I : 0.0f;        \
                    float reduced_z = active_i ? fci_z_##I : 0.0f;        \
                    Reduce_Clustered_Warp_Vector_Over_J_Components(        \
                        reduced_x, reduced_y, reduced_z, cluster_size);    \
                    if (lane < cluster_size)                               \
                    {                                                       \
                        if (warp_id == 0)                                   \
                        {                                                   \
                            fci_x_##I = reduced_x;                         \
                            fci_y_##I = reduced_y;                         \
                            fci_z_##I = reduced_z;                         \
                        }                                                   \
                        else                                                \
                        {                                                   \
                            warp1_i_force[I][lane] = {                     \
                                reduced_x, reduced_y, reduced_z, 0.0f};   \
                        }                                                   \
                    }                                                       \
                }                                                           \
                if constexpr (need_energy)                                 \
                {                                                           \
                    float reduced_lj = active_i ? energy_lj_buf[I] : 0.0f; \
                    float reduced_coulomb =                                \
                        active_i ? energy_coulomb_buf[I] : 0.0f;           \
                    reduced_lj = Reduce_Clustered_Warp_Float_Over_J(       \
                        reduced_lj, cluster_size);                         \
                    reduced_coulomb = Reduce_Clustered_Warp_Float_Over_J(  \
                        reduced_coulomb, cluster_size);                    \
                    if (lane < cluster_size)                               \
                    {                                                       \
                        if (warp_id == 0)                                  \
                        {                                                   \
                            energy_lj_buf[I] = reduced_lj;                 \
                            energy_coulomb_buf[I] = reduced_coulomb;       \
                        }                                                   \
                        else                                                \
                        {                                                   \
                            warp1_i_energy_lj[I][lane] = reduced_lj;       \
                            warp1_i_energy_coulomb[I][lane] =              \
                                reduced_coulomb;                           \
                        }                                                   \
                    }                                                       \
                }                                                           \
            }
            CLUSTERED_STEADY_I_LOCAL_LIST(CLUSTERED_REDUCE_I_LOCAL)
#undef CLUSTERED_REDUCE_I_LOCAL
            __syncthreads();

            if (warp_id == 0 && j_lane == 0)
            {
 #define CLUSTERED_WRITEBACK_I_LOCAL(I)                                     \
                if ((I) < active_cluster_count)                             \
                {                                                            \
                    const unsigned int valid_mask_i =                       \
                        shared_i_valid_masks[I];                            \
                    const unsigned int local_mask_i =                       \
                        shared_i_local_masks[I];                            \
                    const bool active_i =                                    \
                        (valid_mask_i & (1u << i_lane)) != 0u &&            \
                        (local_mask_i & (1u << i_lane)) != 0u;              \
                    if (active_i)                                           \
                    {                                                        \
                        const int atom_i =                                  \
                            shared_i_atom_ids[(I) * cluster_size + i_lane]; \
                        if (need_force)                                     \
                        {                                                    \
                            const float4 warp1_force =                      \
                                warp1_i_force[I][i_lane];                   \
                            atomicAdd(                                      \
                                frc + atom_i,                               \
                                VECTOR{fci_x_##I + warp1_force.x,           \
                                       fci_y_##I + warp1_force.y,           \
                                       fci_z_##I + warp1_force.z});         \
                        }                                                    \
                        if constexpr (need_energy)                          \
                        {                                                    \
                            const float total_energy_lj =                   \
                                energy_lj_buf[I] +                          \
                                warp1_i_energy_lj[I][i_lane];               \
                            const float total_energy_coulomb =              \
                                energy_coulomb_buf[I] +                     \
                                warp1_i_energy_coulomb[I][i_lane];          \
                            atomicAdd(atom_energy + atom_i,                 \
                                      total_energy_lj +                     \
                                          total_energy_coulomb);            \
                            atomicAdd(atom_LJ_ene + atom_i, total_energy_lj); \
                            if (need_coulomb)                               \
                            {                                                \
                                atomicAdd(atom_direct_cf_energy + atom_i,   \
                                          total_energy_coulomb);            \
                            }                                                \
                        }                                                    \
                    }                                                        \
                }
                CLUSTERED_STEADY_I_LOCAL_LIST(CLUSTERED_WRITEBACK_I_LOCAL)
#undef CLUSTERED_WRITEBACK_I_LOCAL
            }

#undef CLUSTERED_STEADY_I_LOCAL_LIST
        }
#endif
    }
}

template <bool need_energy, bool need_coulomb, bool force_soa>
static __global__ void
Nbnxm_Grouped_Lennard_Jones_And_Direct_Coulomb_Virial_Device(
    const int candidate_sci_numbers, const int cluster_size,
    const int super_cluster_clusters, const int local_atom_numbers,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks,
    const int* super_cluster_offsets, const int* sci_supercluster_ids,
    const int* grouped_sci_offsets, const int* grouped_sci_ids,
    const LJ_CLUSTERED_SCI* sci_entries,
    const LJ_CLUSTERED_CJ_PACKED* cj_packed_entries,
    const uint64_t* pair_shift_bits,
    const unsigned long long* exclusion_mask_pool, const int* sorted_atom_ids,
    const float4* sorted_xq, const int* sorted_lj_type, const LTMatrix3 cell,
    const float* LJ_type_A, const float* LJ_type_B, const float cutoff,
    VECTOR* frc, float* frc_x, float* frc_y, float* frc_z,
    const float pme_beta, float* atom_energy, LTMatrix3* atom_virial,
    float* atom_direct_cf_energy, float* atom_LJ_ene)
{
    constexpr int max_cluster_size = kClusteredClusterSize;
    constexpr int max_super_cluster_atoms =
        kClusteredClusterSize * kClusteredSuperClusterClusters;
    constexpr int max_block_warps = 2;
#ifdef USE_GPU
    const int candidate_sci = blockIdx.x;
    const int tid = threadIdx.x;
    if (candidate_sci >= candidate_sci_numbers ||
        tid >= super_cluster_clusters * cluster_size)
    {
        return;
    }
#else
    (void)candidate_sci_numbers;
    (void)cluster_size;
    (void)super_cluster_clusters;
    (void)local_atom_numbers;
    (void)cluster_offsets;
    (void)cluster_valid_masks;
    (void)cluster_local_masks;
    (void)super_cluster_offsets;
    (void)sci_supercluster_ids;
    (void)grouped_sci_offsets;
    (void)grouped_sci_ids;
    (void)sci_entries;
    (void)cj_packed_entries;
    (void)pair_shift_bits;
    (void)exclusion_mask_pool;
    (void)sorted_atom_ids;
    (void)sorted_xq;
    (void)sorted_lj_type;
    (void)cell;
    (void)LJ_type_A;
    (void)LJ_type_B;
    (void)cutoff;
    (void)frc;
    (void)frc_x;
    (void)frc_y;
    (void)frc_z;
    (void)pme_beta;
    (void)atom_energy;
    (void)atom_virial;
    (void)atom_direct_cf_energy;
    (void)atom_LJ_ene;
    return;
#endif

    __shared__ float4 shared_i_xq[max_super_cluster_atoms];
    __shared__ int shared_i_lj_type[max_super_cluster_atoms];
    __shared__ int shared_i_atom_ids[max_super_cluster_atoms];
    __shared__ int shared_i_sorted_ids[max_super_cluster_atoms];
    __shared__ float4 shared_j_xq[max_cluster_size];
    __shared__ int shared_j_lj_type[max_cluster_size];
    __shared__ int shared_j_atom_ids[max_cluster_size];
    __shared__ int shared_j_sorted_ids[max_cluster_size];
    __shared__ int shared_j_local_flags[max_cluster_size];
    __shared__ unsigned int shared_j_valid_mask;
    __shared__ float4 warp_j_force[max_block_warps][max_cluster_size];

    const int super_i = sci_supercluster_ids[candidate_sci];
    const int grouped_sci_begin = grouped_sci_offsets[super_i];
    const int grouped_sci_end = grouped_sci_offsets[super_i + 1];
    if (grouped_sci_begin >= grouped_sci_end)
    {
        return;
    }

    const int cluster_i_start = super_cluster_offsets[super_i];
    const int cluster_i_end = super_cluster_offsets[super_i + 1];
    const int i_cluster_local = tid / cluster_size;
    const int i_lane = tid % cluster_size;
    const int active_cluster_count = cluster_i_end - cluster_i_start;
    const int lane = tid & (warpSize - 1);
    const int warp_id = tid / warpSize;
    const int warp_count =
        (super_cluster_clusters * cluster_size + warpSize - 1) / warpSize;
    const float cutoff_sq = cutoff * cutoff;

    bool active_i = false;
    int cluster_i = -1;
    int atom_i = -1;
    VECTOR frc_i = {0.0f, 0.0f, 0.0f};
    float energy_lj = 0.0f;
    float energy_coulomb = 0.0f;
    LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float4 r1_xq = {0.0f, 0.0f, 0.0f, 0.0f};
    int r1_lj_type = 0;

    if (i_cluster_local < active_cluster_count)
    {
        cluster_i = cluster_i_start + i_cluster_local;
        if ((cluster_valid_masks[cluster_i] & (1u << i_lane)) != 0u)
        {
            const int sorted_atom_i = cluster_offsets[cluster_i] + i_lane;
            shared_i_xq[tid] = sorted_xq[sorted_atom_i];
            shared_i_lj_type[tid] = sorted_lj_type[sorted_atom_i];
            shared_i_atom_ids[tid] = sorted_atom_ids[sorted_atom_i];
            shared_i_sorted_ids[tid] = sorted_atom_i;
            if ((cluster_local_masks[cluster_i] & (1u << i_lane)) != 0u)
            {
                active_i = true;
                atom_i = shared_i_atom_ids[tid];
                r1_xq = shared_i_xq[tid];
                r1_lj_type = shared_i_lj_type[tid];
            }
        }
        else
        {
            shared_i_atom_ids[tid] = -1;
            shared_i_sorted_ids[tid] = -1;
        }
    }
    __syncthreads();

    for (int grouped_sci = grouped_sci_begin; grouped_sci < grouped_sci_end;
         grouped_sci += 1)
    {
        const LJ_CLUSTERED_SCI sci_entry =
            sci_entries[grouped_sci_ids[grouped_sci]];
        const bool sci_is_central =
            sci_entry.shift_id == kClusteredCentralShiftId;
        for (int packed_idx = sci_entry.cjpacked_begin;
             packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
        {
            const LJ_CLUSTERED_CJ_PACKED* packed =
                cj_packed_entries + packed_idx;
            for (int jm = 0; jm < kClusteredJGroupSize; jm += 1)
            {
                const int cluster_j = packed->cj[jm];
                if (cluster_j < 0)
                {
                    continue;
                }
                const unsigned int imask =
                    Clustered_Jm_Imask(packed->imei[0], jm) |
                    Clustered_Jm_Imask(packed->imei[1], jm);
                if (imask == 0u)
                {
                    continue;
                }

                const unsigned int valid_mask_j =
                    cluster_valid_masks[cluster_j];
                if (tid == 0)
                {
                    shared_j_valid_mask = valid_mask_j;
                }
                if (tid < cluster_size)
                {
                    if ((valid_mask_j & (1u << tid)) != 0u)
                    {
                        const int sorted_atom_j =
                            cluster_offsets[cluster_j] + tid;
                        shared_j_xq[tid] = sorted_xq[sorted_atom_j];
                        shared_j_lj_type[tid] = sorted_lj_type[sorted_atom_j];
                        shared_j_atom_ids[tid] = sorted_atom_ids[sorted_atom_j];
                        shared_j_sorted_ids[tid] = sorted_atom_j;
                        shared_j_local_flags[tid] =
                            shared_j_atom_ids[tid] < local_atom_numbers ? 1 : 0;
                    }
                    else
                    {
                        shared_j_atom_ids[tid] = -1;
                        shared_j_sorted_ids[tid] = -1;
                        shared_j_local_flags[tid] = 0;
                    }
                }
                __syncthreads();

                const bool tile_active =
                    active_i &&
                    ((imask & (1u << static_cast<unsigned int>(i_cluster_local))) !=
                     0u);
                unsigned long long exclusion_mask = 0ull;
                VECTOR_LJ r1 = {};
                VECTOR pair_shift = {0.0f, 0.0f, 0.0f};
                if (active_i)
                {
                    r1 = Make_Packed_LJ_Atom(r1_xq, r1_lj_type);
                }
                if (tile_active)
                {
                    const int exclusion_index =
                        Clustered_First_Exclusion_Index(*packed, jm,
                                                        i_cluster_local);
                    exclusion_mask =
                        exclusion_index >= 0
                            ? exclusion_mask_pool[exclusion_index]
                            : 0ull;
                    pair_shift = Clustered_Shift_Vector_From_Id(
                        Clustered_Get_Pair_Shift_Id(
                            pair_shift_bits[packed_idx * kClusteredJGroupSize +
                                            jm],
                            i_cluster_local),
                        cell);
                }

                for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
                {
                    VECTOR j_force_local = {0.0f, 0.0f, 0.0f};
                    if (tile_active &&
                        (shared_j_valid_mask & (1u << lane_j)) != 0u)
                    {
                        const int atom_j = shared_j_atom_ids[lane_j];
                        if (!(sci_is_central && cluster_i == cluster_j &&
                              atom_j < local_atom_numbers &&
                              lane_j <= i_lane) &&
                            (exclusion_mask &
                             (1ull << (i_lane * cluster_size + lane_j))) == 0ull)
                        {
                            const VECTOR_LJ r2 = Make_Packed_LJ_Atom(
                                shared_j_xq[lane_j], shared_j_lj_type[lane_j]);
                            const VECTOR dr =
                                Get_Clustered_Shifted_Displacement(
                                    r2, r1, pair_shift);
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
                                virial = virial -
                                         ij_factor *
                                             Get_Virial_From_Force_Dis(
                                                 frc_lin, dr);
                                if constexpr (need_energy)
                                {
                                    energy_lj +=
                                        ij_factor *
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
                        warp_j_force[warp_id][lane_j] = {
                            reduced.x, reduced.y, reduced.z, 0.0f};
                    }
                }

                __syncthreads();
                if (tid < cluster_size &&
                    (shared_j_valid_mask & (1u << tid)) != 0u &&
                    shared_j_local_flags[tid] != 0)
                {
                    VECTOR total = {0.0f, 0.0f, 0.0f};
                    for (int warp_i = 0; warp_i < warp_count; warp_i += 1)
                    {
                        const float4 cached = warp_j_force[warp_i][tid];
                        total = total +
                                VECTOR{cached.x, cached.y, cached.z};
                    }
                    if constexpr (force_soa)
                    {
                        const int sorted_j = shared_j_sorted_ids[tid];
                        atomicAdd(frc_x + sorted_j, total.x);
                        atomicAdd(frc_y + sorted_j, total.y);
                        atomicAdd(frc_z + sorted_j, total.z);
                    }
                    else
                    {
                        atomicAdd(frc + shared_j_sorted_ids[tid], total);
                    }
                }
                // Only warp 0 consumes and then overwrites the shared j-tile
                // metadata, so the next tile's load barrier is sufficient to
                // order reuse of these shared buffers.
            }
        }
    }

    if (active_i)
    {
        if constexpr (force_soa)
        {
            const int sorted_i = shared_i_sorted_ids[tid];
            atomicAdd(frc_x + sorted_i, frc_i.x);
            atomicAdd(frc_y + sorted_i, frc_i.y);
            atomicAdd(frc_z + sorted_i, frc_i.z);
        }
        else
        {
            atomicAdd(frc + shared_i_sorted_ids[tid], frc_i);
        }
        if constexpr (need_energy)
        {
            atomicAdd(atom_energy + atom_i, energy_lj + energy_coulomb);
            atomicAdd(atom_LJ_ene + atom_i, energy_lj);
            if (need_coulomb)
            {
                atomicAdd(atom_direct_cf_energy + atom_i, energy_coulomb);
            }
        }
        atomicAdd(atom_virial + atom_i, virial);
    }
}

// Reference GPU path for validating NBNXM metadata without the warp-split
// execution scheme. This keeps the clustered payload unchanged while aligning
// pair traversal with the scalar logic.
template <bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb>
static __global__ void Reference_Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Device(
    const int sci_numbers, const int cluster_size,
    const int super_cluster_clusters, const int local_atom_numbers,
    const int* cluster_offsets, const unsigned int* cluster_valid_masks,
    const unsigned int* cluster_local_masks, const VECTOR* cluster_centers,
    const int* super_cluster_offsets,
    const LJ_CLUSTERED_SCI* sci_entries,
    const LJ_CLUSTERED_CJ_PACKED* cj_packed_entries,
    const unsigned long long* exclusion_mask_pool, const int* sorted_atom_ids,
    const float4* sorted_xq, const int* sorted_lj_type, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* LJ_type_A, const float* LJ_type_B,
    const float cutoff, VECTOR* frc, const float pme_beta, float* atom_energy,
    LTMatrix3* atom_virial, float* atom_direct_cf_energy, float* atom_LJ_ene)
{
    constexpr int max_cluster_size = kClusteredClusterSize;
#ifdef USE_GPU
    const int sci = blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (sci >= sci_numbers ||
        tid >= super_cluster_clusters * cluster_size)
    {
        return;
    }
#else
#pragma omp parallel for schedule(dynamic)
    for (int sci = 0; sci < sci_numbers; sci += 1)
#endif
    {
        const LJ_CLUSTERED_SCI sci_entry = sci_entries[sci];
        const int super_i = sci_entry.supercluster_id;
        const int cluster_i_start = super_cluster_offsets[super_i];
        const int cluster_i_end = super_cluster_offsets[super_i + 1];
        const VECTOR shift_vec =
            Clustered_Shift_Vector_From_Id(sci_entry.shift_id, cell);
        const float cutoff_sq = cutoff * cutoff;
        (void)rcell;

#ifndef USE_GPU
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
                const int sorted_atom_i = cluster_offsets[cluster_i] + lane_i;
                const int atom_i = sorted_atom_ids[sorted_atom_i];
                VECTOR_LJ r1 = Make_Packed_LJ_Atom(sorted_xq[sorted_atom_i],
                                                   sorted_lj_type[sorted_atom_i]);
                r1.crd = r1.crd + shift_vec;
                VECTOR frc_i = {0.0f, 0.0f, 0.0f};
                float energy_lj = 0.0f;
                float energy_coulomb = 0.0f;
                LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

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
                            Clustered_First_Exclusion_Index(packed, jm, i_local);
                        const unsigned long long exclusion_mask =
                            exclusion_index >= 0
                                ? exclusion_mask_pool[exclusion_index]
                                : 0ull;
                        VECTOR frc_j[max_cluster_size] = {};

                        for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
                        {
                            if ((valid_mask_j & (1u << lane_j)) == 0u)
                            {
                                continue;
                            }
                            const int sorted_atom_j =
                                cluster_offsets[cluster_j] + lane_j;
                            const int atom_j = sorted_atom_ids[sorted_atom_j];
                            if (sci_entry.shift_id == kClusteredCentralShiftId &&
                                cluster_i == cluster_j &&
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
                                float frc_abs =
                                    Get_LJ_Force(r1, r2, dr_abs, A, B);
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
                                                 Get_Virial_From_Force_Dis(
                                                     frc_lin, dr);
                                }
                            }
                            if (need_energy)
                            {
                                energy_lj +=
                                    ij_factor *
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
                        if (need_force)
                        {
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
                                if (atom_j < local_atom_numbers)
                                {
                                    atomicAdd(frc + atom_j, frc_j[lane_j]);
                                }
                            }
                        }
                    }
                }

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
                        atomicAdd(atom_direct_cf_energy + atom_i,
                                  energy_coulomb);
                    }
                }
                if (need_virial)
                {
                    atomicAdd(atom_virial + atom_i, virial);
                }
            }
        }
#else
        const int active_cluster_count = cluster_i_end - cluster_i_start;
        const int i_local = tid / cluster_size;
        const int lane_i = tid % cluster_size;
        if (i_local >= active_cluster_count)
        {
            return;
        }
        const int cluster_i = cluster_i_start + i_local;
        const unsigned int valid_mask_i = cluster_valid_masks[cluster_i];
        const unsigned int local_mask_i = cluster_local_masks[cluster_i];
        if ((valid_mask_i & (1u << lane_i)) == 0u ||
            (local_mask_i & (1u << lane_i)) == 0u)
        {
            return;
        }

        const int sorted_atom_i = cluster_offsets[cluster_i] + lane_i;
        const int atom_i = sorted_atom_ids[sorted_atom_i];
        VECTOR_LJ r1 = Make_Packed_LJ_Atom(sorted_xq[sorted_atom_i],
                                           sorted_lj_type[sorted_atom_i]);
        r1.crd = r1.crd + shift_vec;
        VECTOR frc_i = {0.0f, 0.0f, 0.0f};
        float energy_lj = 0.0f;
        float energy_coulomb = 0.0f;
        LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        for (int packed_idx = sci_entry.cjpacked_begin;
             packed_idx < sci_entry.cjpacked_end; packed_idx += 1)
        {
            const LJ_CLUSTERED_CJ_PACKED packed =
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
                    Clustered_First_Exclusion_Index(packed, jm, i_local);
                const unsigned long long exclusion_mask =
                    exclusion_index >= 0 ? exclusion_mask_pool[exclusion_index]
                                         : 0ull;
                VECTOR frc_j[max_cluster_size] = {};

                for (int lane_j = 0; lane_j < cluster_size; lane_j += 1)
                {
                    if ((valid_mask_j & (1u << lane_j)) == 0u)
                    {
                        continue;
                    }
                    const int sorted_atom_j =
                        cluster_offsets[cluster_j] + lane_j;
                    const int atom_j = sorted_atom_ids[sorted_atom_j];
                    if (sci_entry.shift_id == kClusteredCentralShiftId &&
                        cluster_i == cluster_j &&
                        atom_j < local_atom_numbers && lane_j <= lane_i)
                    {
                        continue;
                    }
                    if ((exclusion_mask &
                         (1ull << (lane_i * cluster_size + lane_j))) != 0ull)
                    {
                        continue;
                    }
                    const VECTOR_LJ r2 = Make_Packed_LJ_Atom(
                        sorted_xq[sorted_atom_j], sorted_lj_type[sorted_atom_j]);
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
                                ij_factor *
                                Get_Direct_Coulomb_Energy(
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
        }

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
#endif
    }
}

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
    for (int i = 0; i < pair_type_numbers; i += 1)
    {
        h_LJ_AB_packed[(size_t)i] = {h_LJ_A[i], h_LJ_B[i]};
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
        nbnxm_microbench::SpongeClusteredFullOutputSnapshot
            full_output_snapshot = {};
        std::vector<VECTOR> full_output_force_before;
        std::vector<float> full_output_atom_energy_before;
        std::vector<LTMatrix3> full_output_atom_virial_before;
        if (use_clustered_direct)
        {
            const bool need_gmxpacked_payload =
                Clustered_Microbench_Dump_Prefix() != NULL ||
                (Clustered_Gmxpacked_Direct_Opt_In_Enabled() &&
                 !Clustered_Gmxpacked_Fallback_Native_Enabled());
            // [EXPERIMENTAL] see env-var policy comment at top of section
            const char* full_warp_record_env =
                std::getenv("SPONGE_CLUSTERED_USE_WARP_RECORD_FULL");
            const bool full_warp_record_env_set =
                full_warp_record_env != NULL &&
                full_warp_record_env[0] != '\0';
            const bool force_full_warp_record =
                full_warp_record_env_set && full_warp_record_env[0] != '0';
            const bool auto_prefer_full_warp_record = false;
            const bool prefer_full_warp_record_build =
                need_virial &&
                (force_full_warp_record || auto_prefer_full_warp_record);
            const bool intend_gmxpacked_direct =
                Clustered_Gmxpacked_Direct_Opt_In_Enabled() &&
                !Clustered_Gmxpacked_Fallback_Native_Enabled();
            const bool need_aux_clustered_metadata =
                !intend_gmxpacked_direct && need_virial &&
                (prefer_full_warp_record_build ||
                 Clustered_Use_Warp_Record_Total_Output_Enabled() ||
                 Clustered_Use_Grouped_Virial_Enabled());
            clustered_direct_cache->Build(crd, cell, rcell, cutoff,
                                          need_virial != 0,
                                          prefer_full_warp_record_build,
                                          need_gmxpacked_payload,
                                          need_aux_clustered_metadata,
                                          intend_gmxpacked_direct);
        }
        if (use_clustered_direct &&
            clustered_direct_cache->layout.total_atom_numbers > 0)
        {
            clustered_direct_cache->Gather_Plain(
                crd, charge, crd_with_LJ_parameters_local, cell, rcell);
            if (want_full_output_snapshot)
            {
                have_full_output_snapshot =
                    Capture_Clustered_Microbench_Full_Output_Diagnostic_View(
                        clustered_direct_cache, d_LJ_AB_packed,
                        static_cast<size_t>(pair_type_numbers), cutoff,
                        pme_beta, cell, &full_output_snapshot);
            }
            else
            {
                Maybe_Dump_Clustered_Microbench_Diagnostic_Snapshot(
                    clustered_direct_cache, d_LJ_AB_packed,
                    static_cast<size_t>(pair_type_numbers), cutoff, pme_beta,
                    cell);
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
            if (clustered_layout.cjpacked_numbers == 0 ||
                clustered_layout.sci_numbers == 0)
                return;
            dim3 blockSize = {
                static_cast<unsigned int>(clustered_layout.cluster_size),
                static_cast<unsigned int>(clustered_layout.cluster_size), 1u};
            dim3 gridSize = {
                static_cast<unsigned int>(clustered_layout.sci_numbers), 1u, 1u};
            auto f = Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Device<
                true, false, false, true>;
            if (!need_atom_energy && !need_virial)
            {
                f = Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Device<
                    true, false, false, true>;
            }
            else if (need_atom_energy && !need_virial)
            {
                f = Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Device<
                    true, true, false, true>;
            }
            else if (!need_atom_energy && need_virial)
            {
                f = Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Device<
                    true, false, true, true>;
            }
            else
            {
                f = Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Device<
                    true, true, true, true>;
            }
            if (clustered_direct_cache->direct_kernel_time_recorder != NULL)
            {
                clustered_direct_cache->direct_kernel_time_recorder->Start();
            }
            VECTOR* clustered_force_target = frc;
#ifndef USE_CPU
            // [EXPERIMENTAL] see env-var policy comment at top of section
            const char* full_warp_record_env =
                std::getenv("SPONGE_CLUSTERED_USE_WARP_RECORD_FULL");
            const bool full_warp_record_env_set =
                full_warp_record_env != NULL &&
                full_warp_record_env[0] != '\0';
            const bool force_full_warp_record =
                full_warp_record_env_set && full_warp_record_env[0] != '0';
            const bool native_gmxpacked_fallback =
                Clustered_Gmxpacked_Fallback_Native_Enabled();
            const bool gmxpacked_direct_opt_in =
                Clustered_Gmxpacked_Direct_Opt_In_Enabled();
            const bool has_sorted_force_scratch =
                clustered_direct_cache->d_sorted_frc != NULL;
            const bool has_sorted_force_soa_scratch =
                clustered_direct_cache->d_sorted_frc_x != NULL &&
                clustered_direct_cache->d_sorted_frc_y != NULL &&
                clustered_direct_cache->d_sorted_frc_z != NULL;
            const bool has_gmxpacked_payload =
                Clustered_Layout_Has_Primary_Gmxpacked_Payload(
                    clustered_layout) &&
                clustered_direct_cache->d_sorted_atom_ids != NULL &&
                clustered_direct_cache->d_sorted_xq != NULL &&
                clustered_direct_cache->d_sorted_lj_type != NULL &&
                d_LJ_AB_packed != NULL;
            const bool gmxpacked_needs_compact_force_scratch =
                need_atom_energy || need_virial;
            const bool use_gmxpacked_direct =
                gmxpacked_direct_opt_in && !native_gmxpacked_fallback &&
                has_gmxpacked_payload &&
                (!gmxpacked_needs_compact_force_scratch ||
                 has_sorted_force_scratch);
            const bool use_gmxpacked_compact_force_scratch =
                use_gmxpacked_direct && gmxpacked_needs_compact_force_scratch;
            const bool use_total_output_warp_record =
                !use_gmxpacked_direct && !native_gmxpacked_fallback &&
                need_atom_energy && need_virial &&
                Clustered_Use_Warp_Record_Total_Output_Enabled();
            const bool auto_use_full_warp_record = false;
            const bool use_full_warp_record =
                !use_gmxpacked_direct && !native_gmxpacked_fallback &&
                need_virial &&
                (force_full_warp_record || auto_use_full_warp_record) &&
                clustered_layout.forceonly_warp_record_numbers > 0 &&
                clustered_layout.d_forceonly_warp_record_offsets != NULL &&
                clustered_layout.d_forceonly_warp_j_records != NULL;
            const bool use_grouped_clustered_kernel =
                !use_gmxpacked_direct && !native_gmxpacked_fallback &&
                Clustered_Use_Grouped_Virial_Enabled() &&
                !use_full_warp_record && need_virial &&
                clustered_layout.grouped_sci_ready;
            const bool use_sorted_force_soa_scratch =
                !use_gmxpacked_direct &&
                (use_grouped_clustered_kernel || use_full_warp_record) &&
                has_sorted_force_soa_scratch;
            const bool use_sorted_force_scratch =
                use_gmxpacked_compact_force_scratch ||
                (!use_gmxpacked_direct && need_virial &&
                 !use_sorted_force_soa_scratch && has_sorted_force_scratch);
            const bool use_total_output_clustered =
                use_gmxpacked_compact_force_scratch ||
                use_total_output_warp_record;
            if (have_full_output_snapshot)
            {
                const size_t total_atom_numbers_snapshot = static_cast<size_t>(
                    clustered_layout.total_atom_numbers);
                const size_t scalar_output_numbers =
                    use_total_output_clustered ? 1u
                                               : total_atom_numbers_snapshot;
                full_output_snapshot.header.compute_energy =
                    need_atom_energy ? 1u : 0u;
                full_output_snapshot.header.compute_virial =
                    need_virial ? 1u : 0u;
                full_output_snapshot.header.force_soa =
                    use_sorted_force_soa_scratch ? 1u : 0u;
                full_output_snapshot.header.total_output =
                    use_total_output_clustered ? 1u : 0u;
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
            if (use_sorted_force_soa_scratch)
            {
                deviceMemset(clustered_direct_cache->d_sorted_frc_x, 0,
                             sizeof(float) *
                                 clustered_layout.total_atom_numbers);
                deviceMemset(clustered_direct_cache->d_sorted_frc_y, 0,
                             sizeof(float) *
                                 clustered_layout.total_atom_numbers);
                deviceMemset(clustered_direct_cache->d_sorted_frc_z, 0,
                             sizeof(float) *
                                 clustered_layout.total_atom_numbers);
            }
            else if (use_sorted_force_scratch)
            {
                deviceMemset(clustered_direct_cache->d_sorted_frc, 0,
                             sizeof(VECTOR) *
                                 clustered_layout.total_atom_numbers);
                clustered_force_target = clustered_direct_cache->d_sorted_frc;
            }
#else
            const bool use_gmxpacked_direct = false;
            const bool use_full_warp_record = false;
            const bool use_grouped_clustered_kernel = false;
            const bool use_sorted_force_soa_scratch = false;
            const bool use_sorted_force_scratch = false;
            const bool use_total_output_warp_record = false;
#endif
            if (use_gmxpacked_direct)
            {
                dim3 gmxpackedGridSize = {
                    static_cast<unsigned int>(
                        clustered_layout.gmxpacked_sci_numbers),
                    1u, 1u};
                auto gmxpacked_f =
                    Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                        false, false, false, false>;
                if (need_atom_energy && need_virial)
                {
                    gmxpacked_f =
                        Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                            true, true, true, true>;
                }
                else if (need_atom_energy)
                {
                    gmxpacked_f =
                        Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                            true, false, true, true>;
                }
                else if (need_virial)
                {
                    gmxpacked_f =
                        Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<
                            false, true, true, true>;
                }
                Launch_Device_Kernel(
                    gmxpacked_f, gmxpackedGridSize, blockSize, 0, NULL,
                    clustered_layout.gmxpacked_sci_numbers,
                    clustered_layout.cluster_size,
                    clustered_layout.super_cluster_clusters,
                    clustered_layout.d_cluster_offsets,
                    clustered_layout.d_cluster_valid_masks,
                    clustered_layout.d_cluster_local_masks,
                    clustered_layout.d_super_cluster_offsets,
                    clustered_layout.d_gmxpacked_sci,
                    clustered_layout.d_gmxpacked_cjpacked,
                    clustered_layout.d_gmxpacked_exclusions,
                    clustered_layout.d_pair_shift_bits,
                    clustered_direct_cache->d_sorted_atom_ids,
                    clustered_direct_cache->d_sorted_xq,
                    clustered_direct_cache->d_sorted_lj_type, cell,
                    d_LJ_AB_packed, cutoff, clustered_force_target, pme_beta,
                    atom_energy, atom_virial, atom_direct_pme_energy,
                    d_LJ_energy_atom);
            }
            else if (use_grouped_clustered_kernel)
            {
                const int grouped_block_size =
                    clustered_layout.cluster_size *
                    clustered_layout.super_cluster_clusters;
                const int grouped_grid_size =
                    clustered_layout.candidate_sci_numbers;
                if (need_atom_energy)
                {
                    auto grouped_f_aos =
                        Nbnxm_Grouped_Lennard_Jones_And_Direct_Coulomb_Virial_Device<
                            true, true, false>;
                    auto grouped_f_soa =
                        Nbnxm_Grouped_Lennard_Jones_And_Direct_Coulomb_Virial_Device<
                            true, true, true>;
                    if (use_sorted_force_soa_scratch)
                    {
                        Launch_Device_Kernel(
                            grouped_f_soa,
                            grouped_grid_size, grouped_block_size, 0, NULL,
                            clustered_layout.candidate_sci_numbers,
                            clustered_layout.cluster_size,
                            clustered_layout.super_cluster_clusters,
                            local_atom_numbers,
                            clustered_layout.d_cluster_offsets,
                            clustered_layout.d_cluster_valid_masks,
                            clustered_layout.d_cluster_local_masks,
                            clustered_layout.d_super_cluster_offsets,
                            clustered_layout.d_sci_supercluster_ids,
                            clustered_layout.d_grouped_sci_offsets,
                            clustered_layout.d_grouped_sci_ids,
                            clustered_layout.d_nbnxm_sci,
                            clustered_layout.d_nbnxm_cjpacked,
                            clustered_layout.d_pair_shift_bits,
                            clustered_layout.d_exclusion_mask_pool,
                            clustered_direct_cache->d_sorted_atom_ids,
                            clustered_direct_cache->d_sorted_xq,
                            clustered_direct_cache->d_sorted_lj_type, cell,
                            d_LJ_A, d_LJ_B, cutoff, NULL,
                            clustered_direct_cache->d_sorted_frc_x,
                            clustered_direct_cache->d_sorted_frc_y,
                            clustered_direct_cache->d_sorted_frc_z, pme_beta,
                            atom_energy, atom_virial, atom_direct_pme_energy,
                            d_LJ_energy_atom);
                    }
                    else
                    {
                        Launch_Device_Kernel(
                            grouped_f_aos,
                            grouped_grid_size, grouped_block_size, 0, NULL,
                            clustered_layout.candidate_sci_numbers,
                            clustered_layout.cluster_size,
                            clustered_layout.super_cluster_clusters,
                            local_atom_numbers,
                            clustered_layout.d_cluster_offsets,
                            clustered_layout.d_cluster_valid_masks,
                            clustered_layout.d_cluster_local_masks,
                            clustered_layout.d_super_cluster_offsets,
                            clustered_layout.d_sci_supercluster_ids,
                            clustered_layout.d_grouped_sci_offsets,
                            clustered_layout.d_grouped_sci_ids,
                            clustered_layout.d_nbnxm_sci,
                            clustered_layout.d_nbnxm_cjpacked,
                            clustered_layout.d_pair_shift_bits,
                            clustered_layout.d_exclusion_mask_pool,
                            clustered_direct_cache->d_sorted_atom_ids,
                            clustered_direct_cache->d_sorted_xq,
                            clustered_direct_cache->d_sorted_lj_type, cell,
                            d_LJ_A, d_LJ_B, cutoff, clustered_force_target,
                            NULL, NULL, NULL, pme_beta, atom_energy,
                            atom_virial, atom_direct_pme_energy,
                            d_LJ_energy_atom);
                    }
                }
                else
                {
                    auto grouped_f_aos =
                        Nbnxm_Grouped_Lennard_Jones_And_Direct_Coulomb_Virial_Device<
                            false, true, false>;
                    auto grouped_f_soa =
                        Nbnxm_Grouped_Lennard_Jones_And_Direct_Coulomb_Virial_Device<
                            false, true, true>;
                    if (use_sorted_force_soa_scratch)
                    {
                        Launch_Device_Kernel(
                            grouped_f_soa,
                            grouped_grid_size, grouped_block_size, 0, NULL,
                            clustered_layout.candidate_sci_numbers,
                            clustered_layout.cluster_size,
                            clustered_layout.super_cluster_clusters,
                            local_atom_numbers,
                            clustered_layout.d_cluster_offsets,
                            clustered_layout.d_cluster_valid_masks,
                            clustered_layout.d_cluster_local_masks,
                            clustered_layout.d_super_cluster_offsets,
                            clustered_layout.d_sci_supercluster_ids,
                            clustered_layout.d_grouped_sci_offsets,
                            clustered_layout.d_grouped_sci_ids,
                            clustered_layout.d_nbnxm_sci,
                            clustered_layout.d_nbnxm_cjpacked,
                            clustered_layout.d_pair_shift_bits,
                            clustered_layout.d_exclusion_mask_pool,
                            clustered_direct_cache->d_sorted_atom_ids,
                            clustered_direct_cache->d_sorted_xq,
                            clustered_direct_cache->d_sorted_lj_type, cell,
                            d_LJ_A, d_LJ_B, cutoff, NULL,
                            clustered_direct_cache->d_sorted_frc_x,
                            clustered_direct_cache->d_sorted_frc_y,
                            clustered_direct_cache->d_sorted_frc_z, pme_beta,
                            atom_energy, atom_virial, atom_direct_pme_energy,
                            d_LJ_energy_atom);
                    }
                    else
                    {
                        Launch_Device_Kernel(
                            grouped_f_aos,
                            grouped_grid_size, grouped_block_size, 0, NULL,
                            clustered_layout.candidate_sci_numbers,
                            clustered_layout.cluster_size,
                            clustered_layout.super_cluster_clusters,
                            local_atom_numbers,
                            clustered_layout.d_cluster_offsets,
                            clustered_layout.d_cluster_valid_masks,
                            clustered_layout.d_cluster_local_masks,
                            clustered_layout.d_super_cluster_offsets,
                            clustered_layout.d_sci_supercluster_ids,
                            clustered_layout.d_grouped_sci_offsets,
                            clustered_layout.d_grouped_sci_ids,
                            clustered_layout.d_nbnxm_sci,
                            clustered_layout.d_nbnxm_cjpacked,
                            clustered_layout.d_pair_shift_bits,
                            clustered_layout.d_exclusion_mask_pool,
                            clustered_direct_cache->d_sorted_atom_ids,
                            clustered_direct_cache->d_sorted_xq,
                            clustered_direct_cache->d_sorted_lj_type, cell,
                            d_LJ_A, d_LJ_B, cutoff, clustered_force_target,
                            NULL, NULL, NULL, pme_beta, atom_energy,
                            atom_virial, atom_direct_pme_energy,
                            d_LJ_energy_atom);
                    }
                }
            }
            else if (use_full_warp_record)
            {
                auto full_record_f_aos =
                    Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Virial_Warp_Record_Device<
                        true, false, false>;
                auto full_record_f_soa =
                    Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Virial_Warp_Record_Device<
                        true, true, false>;
                auto full_record_total_f_aos =
                    Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Virial_Warp_Record_Device<
                        true, false, true>;
                auto full_record_total_f_soa =
                    Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Virial_Warp_Record_Device<
                        true, true, true>;
                auto virial_record_f_aos =
                    Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Virial_Warp_Record_Device<
                        false, false, false>;
                auto virial_record_f_soa =
                    Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Virial_Warp_Record_Device<
                        false, true, false>;
                if (need_atom_energy)
                {
                    if (use_sorted_force_soa_scratch)
                    {
                        if (use_total_output_warp_record)
                        {
                            Launch_Device_Kernel(
                                full_record_total_f_soa,
                                gridSize, blockSize, 0, NULL,
                                clustered_layout.sci_numbers,
                                clustered_layout.cluster_size,
                                clustered_layout.super_cluster_clusters,
                                local_atom_numbers,
                                clustered_layout.d_cluster_offsets,
                                clustered_layout.d_cluster_valid_masks,
                                clustered_layout.d_cluster_local_masks,
                                clustered_layout.d_super_cluster_offsets,
                                clustered_layout.d_nbnxm_sci,
                                clustered_layout.d_forceonly_warp_record_offsets,
                                clustered_layout.d_forceonly_warp_j_records,
                                clustered_layout.d_pair_shift_bits,
                                clustered_layout.d_exclusion_mask_pool,
                                clustered_direct_cache->d_sorted_atom_ids,
                                clustered_direct_cache->d_sorted_xq,
                                clustered_direct_cache->d_sorted_lj_type, cell,
                                d_LJ_A, d_LJ_B, d_LJ_AB_packed, cutoff, NULL,
                                clustered_direct_cache->d_sorted_frc_x,
                                clustered_direct_cache->d_sorted_frc_y,
                                clustered_direct_cache->d_sorted_frc_z, pme_beta,
                                atom_energy, atom_virial,
                                atom_direct_pme_energy, d_LJ_energy_atom);
                        }
                        else
                        {
                            Launch_Device_Kernel(
                                full_record_f_soa,
                                gridSize, blockSize, 0, NULL,
                                clustered_layout.sci_numbers,
                                clustered_layout.cluster_size,
                                clustered_layout.super_cluster_clusters,
                                local_atom_numbers,
                                clustered_layout.d_cluster_offsets,
                                clustered_layout.d_cluster_valid_masks,
                                clustered_layout.d_cluster_local_masks,
                                clustered_layout.d_super_cluster_offsets,
                                clustered_layout.d_nbnxm_sci,
                                clustered_layout.d_forceonly_warp_record_offsets,
                                clustered_layout.d_forceonly_warp_j_records,
                                clustered_layout.d_pair_shift_bits,
                                clustered_layout.d_exclusion_mask_pool,
                                clustered_direct_cache->d_sorted_atom_ids,
                                clustered_direct_cache->d_sorted_xq,
                                clustered_direct_cache->d_sorted_lj_type, cell,
                                d_LJ_A, d_LJ_B, d_LJ_AB_packed, cutoff, NULL,
                                clustered_direct_cache->d_sorted_frc_x,
                                clustered_direct_cache->d_sorted_frc_y,
                                clustered_direct_cache->d_sorted_frc_z, pme_beta,
                                atom_energy, atom_virial,
                                atom_direct_pme_energy, d_LJ_energy_atom);
                        }
                    }
                    else
                    {
                        if (use_total_output_warp_record)
                        {
                            Launch_Device_Kernel(
                                full_record_total_f_aos,
                                gridSize, blockSize, 0, NULL,
                                clustered_layout.sci_numbers,
                                clustered_layout.cluster_size,
                                clustered_layout.super_cluster_clusters,
                                local_atom_numbers,
                                clustered_layout.d_cluster_offsets,
                                clustered_layout.d_cluster_valid_masks,
                                clustered_layout.d_cluster_local_masks,
                                clustered_layout.d_super_cluster_offsets,
                                clustered_layout.d_nbnxm_sci,
                                clustered_layout.d_forceonly_warp_record_offsets,
                                clustered_layout.d_forceonly_warp_j_records,
                                clustered_layout.d_pair_shift_bits,
                                clustered_layout.d_exclusion_mask_pool,
                                clustered_direct_cache->d_sorted_atom_ids,
                                clustered_direct_cache->d_sorted_xq,
                                clustered_direct_cache->d_sorted_lj_type, cell,
                                d_LJ_A, d_LJ_B, d_LJ_AB_packed, cutoff,
                                clustered_force_target,
                                NULL, NULL, NULL, pme_beta, atom_energy,
                                atom_virial, atom_direct_pme_energy,
                                d_LJ_energy_atom);
                        }
                        else
                        {
                            Launch_Device_Kernel(
                                full_record_f_aos,
                                gridSize, blockSize, 0, NULL,
                                clustered_layout.sci_numbers,
                                clustered_layout.cluster_size,
                                clustered_layout.super_cluster_clusters,
                                local_atom_numbers,
                                clustered_layout.d_cluster_offsets,
                                clustered_layout.d_cluster_valid_masks,
                                clustered_layout.d_cluster_local_masks,
                                clustered_layout.d_super_cluster_offsets,
                                clustered_layout.d_nbnxm_sci,
                                clustered_layout.d_forceonly_warp_record_offsets,
                                clustered_layout.d_forceonly_warp_j_records,
                                clustered_layout.d_pair_shift_bits,
                                clustered_layout.d_exclusion_mask_pool,
                                clustered_direct_cache->d_sorted_atom_ids,
                                clustered_direct_cache->d_sorted_xq,
                                clustered_direct_cache->d_sorted_lj_type, cell,
                                d_LJ_A, d_LJ_B, d_LJ_AB_packed, cutoff,
                                clustered_force_target,
                                NULL, NULL, NULL, pme_beta, atom_energy,
                                atom_virial, atom_direct_pme_energy,
                                d_LJ_energy_atom);
                        }
                    }
                }
                else
                {
                    if (use_sorted_force_soa_scratch)
                    {
                        Launch_Device_Kernel(
                            virial_record_f_soa, gridSize, blockSize, 0, NULL,
                            clustered_layout.sci_numbers,
                            clustered_layout.cluster_size,
                            clustered_layout.super_cluster_clusters,
                            local_atom_numbers,
                            clustered_layout.d_cluster_offsets,
                            clustered_layout.d_cluster_valid_masks,
                            clustered_layout.d_cluster_local_masks,
                            clustered_layout.d_super_cluster_offsets,
                            clustered_layout.d_nbnxm_sci,
                            clustered_layout.d_forceonly_warp_record_offsets,
                            clustered_layout.d_forceonly_warp_j_records,
                            clustered_layout.d_pair_shift_bits,
                            clustered_layout.d_exclusion_mask_pool,
                            clustered_direct_cache->d_sorted_atom_ids,
                            clustered_direct_cache->d_sorted_xq,
                            clustered_direct_cache->d_sorted_lj_type, cell,
                            d_LJ_A, d_LJ_B, d_LJ_AB_packed, cutoff, NULL,
                            clustered_direct_cache->d_sorted_frc_x,
                            clustered_direct_cache->d_sorted_frc_y,
                            clustered_direct_cache->d_sorted_frc_z, pme_beta,
                            atom_energy, atom_virial, atom_direct_pme_energy,
                            d_LJ_energy_atom);
                    }
                    else
                    {
                        Launch_Device_Kernel(
                            virial_record_f_aos, gridSize, blockSize, 0, NULL,
                            clustered_layout.sci_numbers,
                            clustered_layout.cluster_size,
                            clustered_layout.super_cluster_clusters,
                            local_atom_numbers,
                            clustered_layout.d_cluster_offsets,
                            clustered_layout.d_cluster_valid_masks,
                            clustered_layout.d_cluster_local_masks,
                            clustered_layout.d_super_cluster_offsets,
                            clustered_layout.d_nbnxm_sci,
                            clustered_layout.d_forceonly_warp_record_offsets,
                            clustered_layout.d_forceonly_warp_j_records,
                            clustered_layout.d_pair_shift_bits,
                            clustered_layout.d_exclusion_mask_pool,
                            clustered_direct_cache->d_sorted_atom_ids,
                            clustered_direct_cache->d_sorted_xq,
                            clustered_direct_cache->d_sorted_lj_type, cell,
                            d_LJ_A, d_LJ_B, d_LJ_AB_packed, cutoff,
                            clustered_force_target,
                            NULL, NULL, NULL, pme_beta, atom_energy,
                            atom_virial, atom_direct_pme_energy,
                            d_LJ_energy_atom);
                    }
                }
            }
            else
            {
                Launch_Device_Kernel(
                    f, gridSize, blockSize, 0, NULL,
                    clustered_layout.sci_numbers,
                    clustered_layout.cluster_size,
                    clustered_layout.super_cluster_clusters,
                    local_atom_numbers, clustered_layout.d_cluster_offsets,
                    clustered_layout.d_cluster_valid_masks,
                    clustered_layout.d_cluster_local_masks,
                    clustered_layout.d_cluster_centers,
                    clustered_layout.d_super_cluster_offsets,
                    clustered_layout.d_nbnxm_sci,
                    clustered_layout.d_nbnxm_cjpacked,
                    clustered_layout.d_pair_shift_bits,
                    clustered_layout.d_exclusion_mask_pool,
                    clustered_direct_cache->d_sorted_atom_ids,
                    clustered_direct_cache->d_sorted_xq,
                    clustered_direct_cache->d_sorted_lj_type, cell, rcell,
                    d_LJ_A, d_LJ_B, cutoff, clustered_force_target, pme_beta,
                    atom_energy,
                    atom_virial, atom_direct_pme_energy, d_LJ_energy_atom);
            }
#ifndef USE_CPU
            if (use_sorted_force_soa_scratch)
            {
                Launch_Device_Kernel(
                    Scatter_Sorted_Clustered_Force_SoA,
                    (clustered_layout.total_atom_numbers +
                     CONTROLLER::device_max_thread - 1) /
                        CONTROLLER::device_max_thread,
                    CONTROLLER::device_max_thread, 0, NULL,
                    clustered_layout.total_atom_numbers,
                    clustered_direct_cache->d_sorted_atom_ids,
                    clustered_direct_cache->d_sorted_frc_x,
                    clustered_direct_cache->d_sorted_frc_y,
                    clustered_direct_cache->d_sorted_frc_z, frc);
            }
            else if (use_sorted_force_scratch)
            {
                Launch_Device_Kernel(
                    Scatter_Sorted_Clustered_Force,
                    (clustered_layout.total_atom_numbers +
                     CONTROLLER::device_max_thread - 1) /
                        CONTROLLER::device_max_thread,
                    CONTROLLER::device_max_thread, 0, NULL,
                    clustered_layout.total_atom_numbers,
                    clustered_direct_cache->d_sorted_atom_ids,
                    clustered_direct_cache->d_sorted_frc, frc);
            }
#endif
            if (have_full_output_snapshot)
            {
                Finalize_Clustered_Microbench_Full_Output_Snapshot(
                    &full_output_snapshot, full_output_force_before, frc,
                    full_output_atom_energy_before, atom_energy,
                    full_output_atom_virial_before, atom_virial,
                    atom_direct_pme_energy, d_LJ_energy_atom);
                Maybe_Write_Clustered_Microbench_Full_Output_Snapshot(
                    full_output_snapshot);
            }
            if (clustered_direct_cache->direct_kernel_time_recorder != NULL)
            {
                clustered_direct_cache->direct_kernel_time_recorder->Stop();
            }
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
            if (!need_atom_energy && !need_virial)
            {
                f = Lennard_Jones_And_Direct_Coulomb_Device<
                    true, false, false, true>;
            }
            else if (need_atom_energy && !need_virial)
            {
                f = Lennard_Jones_And_Direct_Coulomb_Device<
                    true, true, false, true>;
            }
            else if (!need_atom_energy && need_virial)
            {
                f = Lennard_Jones_And_Direct_Coulomb_Device<
                    true, false, true, true>;
            }
            else
            {
                f = Lennard_Jones_And_Direct_Coulomb_Device<
                    true, true, true, true>;
            }
            Launch_Device_Kernel(
                f, gridSize, blockSize, 0, NULL, local_atom_numbers,
                solvent_numbers, nl, crd_with_LJ_parameters_local, cell, rcell,
                d_LJ_A, d_LJ_B, cutoff, frc, pme_beta, atom_energy,
                atom_virial, atom_direct_pme_energy, d_LJ_energy_atom);
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
