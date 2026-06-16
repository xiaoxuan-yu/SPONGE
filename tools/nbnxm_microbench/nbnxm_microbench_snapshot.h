#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

namespace nbnxm_microbench
{

constexpr char kSnapshotMagic[8] = { 'N', 'B', 'N', 'X', 'M', 'B', 'E', 'N' };
constexpr uint32_t kSnapshotVersion = 2u;

enum class SnapshotKind : uint32_t
{
    spongeForceOnly = 1u,
    gromacsPairlist = 2u,
    spongeClusteredFullOutput = 3u,
};

struct SnapshotFileHeader
{
    char magic[8] = {};
    uint32_t version = kSnapshotVersion;
    uint32_t kind = 0u;
};

struct Float2POD
{
    float x = 0.0f;
    float y = 0.0f;
};

struct Float4POD
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

struct LTMatrix3POD
{
    float a11 = 0.0f;
    float a21 = 0.0f;
    float a22 = 0.0f;
    float a31 = 0.0f;
    float a32 = 0.0f;
    float a33 = 0.0f;
};

struct SpongeSciPOD
{
    int supercluster_id = 0;
    int shift_id = 13;
    int cjpacked_begin = 0;
    int cjpacked_end = 0;
};

struct SpongeWarpJRecordPOD
{
    int cluster_j = -1;
    int sorted_j_base = 0;
    int pair_shift_index = -1;
    unsigned char valid_mask = 0u;
    unsigned char imask = 0u;
    unsigned char local_mask = 0u;
    unsigned char j_lane_base = 0u;
    unsigned char pair_excl[32] = {};
};

struct GromacsSciPOD
{
    int sci = 0;
    int shift = 13;
    int cjPackedBegin = 0;
    int cjPackedEnd = 0;
};

struct GromacsImeEiPOD
{
    unsigned int imask = 0u;
    int excl_ind = 0;
};

struct GromacsCjPackedPOD
{
    int cj[4] = { -1, -1, -1, -1 };
    GromacsImeEiPOD imei[2] = {};
};

struct GromacsExclPOD
{
    unsigned int pair[32] = {};
};

struct SpongeForceOnlySnapshotHeader
{
    SnapshotFileHeader file = {};
    uint32_t cluster_size = 8u;
    uint32_t super_cluster_clusters = 8u;
    uint32_t warp_split_count = 2u;
    uint32_t reserved0 = 0u;
    uint64_t cluster_numbers = 0u;
    uint64_t super_cluster_numbers = 0u;
    uint64_t sci_numbers = 0u;
    uint64_t record_numbers = 0u;
    uint64_t pair_shift_word_numbers = 0u;
    uint64_t total_atom_numbers = 0u;
    uint64_t local_atom_numbers = 0u;
    uint64_t lj_param_numbers = 0u;
    float cutoff = 0.0f;
    float pme_beta = 0.0f;
    LTMatrix3POD cell = {};
};

struct SpongeClusteredFullOutputSnapshotHeader
{
    SnapshotFileHeader file = {};
    uint32_t cluster_size = 8u;
    uint32_t super_cluster_clusters = 8u;
    uint32_t warp_split_count = 2u;
    uint32_t compute_energy = 0u;
    uint32_t compute_virial = 0u;
    uint32_t force_soa = 1u;
    uint32_t total_output = 0u;
    uint32_t reserved0 = 0u;
    uint64_t cluster_numbers = 0u;
    uint64_t super_cluster_numbers = 0u;
    uint64_t sci_numbers = 0u;
    uint64_t record_numbers = 0u;
    uint64_t pair_shift_word_numbers = 0u;
    uint64_t total_atom_numbers = 0u;
    uint64_t local_atom_numbers = 0u;
    uint64_t lj_param_numbers = 0u;
    uint64_t force_reference_numbers = 0u;
    uint64_t energy_reference_numbers = 0u;
    uint64_t virial_reference_numbers = 0u;
    uint64_t direct_energy_reference_numbers = 0u;
    uint64_t lj_energy_reference_numbers = 0u;
    float cutoff = 0.0f;
    float pme_beta = 0.0f;
    LTMatrix3POD cell = {};
};

struct GromacsPairlistSnapshotHeader
{
    SnapshotFileHeader file = {};
    uint32_t cluster_size = 8u;
    uint32_t super_cluster_clusters = 8u;
    uint32_t cluster_pair_split = 2u;
    uint32_t j_group_size = 4u;
    uint32_t elec_type = 0u;
    uint32_t vdw_type = 0u;
    uint32_t num_types = 0u;
    uint32_t num_threads_z = 1u;
    uint32_t compute_energy = 0u;
    uint32_t compute_virial = 0u;
    uint32_t use_prune_kernel = 0u;
    uint32_t reserved0 = 0u;
    uint64_t cluster_numbers = 0u;
    uint64_t sci_numbers = 0u;
    uint64_t cjpacked_numbers = 0u;
    uint64_t excl_numbers = 0u;
    uint64_t total_atom_numbers = 0u;
    uint64_t local_atom_numbers = 0u;
    uint64_t lj_param_numbers = 0u;
    float cutoff = 0.0f;
    float pme_beta = 0.0f;
    float epsfac = 1.0f;
    LTMatrix3POD cell = {};
    std::array<Float4POD, 27> shiftvec = {};
};

struct SpongeForceOnlySnapshot
{
    SpongeForceOnlySnapshotHeader header = {};
    std::vector<int> cluster_offsets;
    std::vector<unsigned int> cluster_valid_masks;
    std::vector<unsigned int> cluster_local_masks;
    std::vector<int> super_cluster_offsets;
    std::vector<SpongeSciPOD> sci;
    std::vector<int> record_offsets;
    std::vector<SpongeWarpJRecordPOD> records;
    std::vector<uint64_t> pair_shift_bits;
    std::vector<int> sorted_atom_ids;
    std::vector<Float4POD> sorted_xq;
    std::vector<int> sorted_lj_type;
    std::vector<Float2POD> lj_ab;
};

struct SpongeClusteredFullOutputSnapshot
{
    SpongeClusteredFullOutputSnapshotHeader header = {};
    std::vector<int> cluster_offsets;
    std::vector<unsigned int> cluster_valid_masks;
    std::vector<unsigned int> cluster_local_masks;
    std::vector<int> super_cluster_offsets;
    std::vector<SpongeSciPOD> sci;
    std::vector<int> record_offsets;
    std::vector<SpongeWarpJRecordPOD> records;
    std::vector<uint64_t> pair_shift_bits;
    std::vector<int> sorted_atom_ids;
    std::vector<Float4POD> sorted_xq;
    std::vector<int> sorted_lj_type;
    std::vector<Float2POD> lj_ab;
    std::vector<Float4POD> reference_force;
    std::vector<float> reference_atom_energy;
    std::vector<LTMatrix3POD> reference_atom_virial;
    std::vector<float> reference_direct_cf_energy;
    std::vector<float> reference_lj_energy;
};

struct GromacsPairlistSnapshot
{
    GromacsPairlistSnapshotHeader header = {};
    std::vector<int> cluster_offsets;
    std::vector<unsigned int> cluster_valid_masks;
    std::vector<unsigned int> cluster_local_masks;
    std::vector<GromacsSciPOD> sci;
    std::vector<GromacsCjPackedPOD> cjpacked;
    std::vector<GromacsExclPOD> excl;
    std::vector<int> sorted_atom_ids;
    std::vector<Float4POD> sorted_xq;
    std::vector<int> sorted_lj_type;
    std::vector<Float2POD> sorted_lj_comb;
    std::vector<Float2POD> lj_ab;
};

template <typename T>
constexpr bool IsTriviallySerializable()
{
    return std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;
}

template <typename T>
inline bool WriteBinary(std::ofstream* out, const T& value)
{
    static_assert(IsTriviallySerializable<T>(), "binary write requires POD");
    out->write(reinterpret_cast<const char*>(&value), sizeof(T));
    return out->good();
}

template <typename T>
inline bool WriteVector(std::ofstream* out, const std::vector<T>& values)
{
    static_assert(IsTriviallySerializable<T>(), "binary write requires POD");
    if (!values.empty())
    {
        out->write(reinterpret_cast<const char*>(values.data()),
                   static_cast<std::streamsize>(sizeof(T) * values.size()));
    }
    return out->good();
}

template <typename T>
inline bool ReadBinary(std::ifstream* in, T* value)
{
    static_assert(IsTriviallySerializable<T>(), "binary read requires POD");
    in->read(reinterpret_cast<char*>(value), sizeof(T));
    return in->good();
}

template <typename T>
inline bool ReadVector(std::ifstream* in, std::vector<T>* values, size_t count)
{
    static_assert(IsTriviallySerializable<T>(), "binary read requires POD");
    values->resize(count);
    if (count > 0)
    {
        in->read(reinterpret_cast<char*>(values->data()),
                 static_cast<std::streamsize>(sizeof(T) * count));
    }
    return in->good();
}

inline SnapshotFileHeader MakeFileHeader(SnapshotKind kind)
{
    SnapshotFileHeader header = {};
    std::memcpy(header.magic, kSnapshotMagic, sizeof(kSnapshotMagic));
    header.version = kSnapshotVersion;
    header.kind = static_cast<uint32_t>(kind);
    return header;
}

inline bool IsValidFileHeader(const SnapshotFileHeader& header,
                              SnapshotKind expected_kind)
{
    return std::memcmp(header.magic, kSnapshotMagic, sizeof(kSnapshotMagic)) == 0 &&
           header.version == kSnapshotVersion &&
           header.kind == static_cast<uint32_t>(expected_kind);
}

inline bool WriteSpongeForceOnlySnapshot(const std::string& path,
                                         const SpongeForceOnlySnapshot& snapshot)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.good())
    {
        return false;
    }
    if (!WriteBinary(&out, snapshot.header) ||
        !WriteVector(&out, snapshot.cluster_offsets) ||
        !WriteVector(&out, snapshot.cluster_valid_masks) ||
        !WriteVector(&out, snapshot.cluster_local_masks) ||
        !WriteVector(&out, snapshot.super_cluster_offsets) ||
        !WriteVector(&out, snapshot.sci) ||
        !WriteVector(&out, snapshot.record_offsets) ||
        !WriteVector(&out, snapshot.records) ||
        !WriteVector(&out, snapshot.pair_shift_bits) ||
        !WriteVector(&out, snapshot.sorted_atom_ids) ||
        !WriteVector(&out, snapshot.sorted_xq) ||
        !WriteVector(&out, snapshot.sorted_lj_type) ||
        !WriteVector(&out, snapshot.lj_ab))
    {
        return false;
    }
    return out.good();
}

inline bool ReadSpongeForceOnlySnapshot(const std::string& path,
                                        SpongeForceOnlySnapshot* snapshot)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.good() || !ReadBinary(&in, &snapshot->header) ||
        !IsValidFileHeader(snapshot->header.file,
                           SnapshotKind::spongeForceOnly))
    {
        return false;
    }
    const auto cluster_numbers =
        static_cast<size_t>(snapshot->header.cluster_numbers);
    const auto super_cluster_numbers =
        static_cast<size_t>(snapshot->header.super_cluster_numbers);
    const auto sci_numbers = static_cast<size_t>(snapshot->header.sci_numbers);
    const auto record_numbers =
        static_cast<size_t>(snapshot->header.record_numbers);
    const auto pair_shift_word_numbers =
        static_cast<size_t>(snapshot->header.pair_shift_word_numbers);
    const auto total_atom_numbers =
        static_cast<size_t>(snapshot->header.total_atom_numbers);
    const auto lj_param_numbers =
        static_cast<size_t>(snapshot->header.lj_param_numbers);
    return ReadVector(&in, &snapshot->cluster_offsets, cluster_numbers) &&
           ReadVector(&in, &snapshot->cluster_valid_masks, cluster_numbers) &&
           ReadVector(&in, &snapshot->cluster_local_masks, cluster_numbers) &&
           ReadVector(&in, &snapshot->super_cluster_offsets,
                      super_cluster_numbers + 1) &&
           ReadVector(&in, &snapshot->sci, sci_numbers) &&
           ReadVector(&in, &snapshot->record_offsets, sci_numbers + 1) &&
           ReadVector(&in, &snapshot->records, record_numbers) &&
           ReadVector(&in, &snapshot->pair_shift_bits, pair_shift_word_numbers) &&
           ReadVector(&in, &snapshot->sorted_atom_ids, total_atom_numbers) &&
           ReadVector(&in, &snapshot->sorted_xq, total_atom_numbers) &&
           ReadVector(&in, &snapshot->sorted_lj_type, total_atom_numbers) &&
           ReadVector(&in, &snapshot->lj_ab, lj_param_numbers);
}

inline bool WriteGromacsPairlistSnapshot(const std::string& path,
                                         const GromacsPairlistSnapshot& snapshot)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.good())
    {
        return false;
    }
    if (!WriteBinary(&out, snapshot.header) ||
        !WriteVector(&out, snapshot.cluster_offsets) ||
        !WriteVector(&out, snapshot.cluster_valid_masks) ||
        !WriteVector(&out, snapshot.cluster_local_masks) ||
        !WriteVector(&out, snapshot.sci) ||
        !WriteVector(&out, snapshot.cjpacked) ||
        !WriteVector(&out, snapshot.excl) ||
        !WriteVector(&out, snapshot.sorted_xq) ||
        !WriteVector(&out, snapshot.sorted_lj_type) ||
        !WriteVector(&out, snapshot.sorted_lj_comb) ||
        !WriteVector(&out, snapshot.lj_ab))
    {
        return false;
    }
    return out.good();
}

inline bool ReadGromacsPairlistSnapshot(const std::string& path,
                                        GromacsPairlistSnapshot* snapshot)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.good() || !ReadBinary(&in, &snapshot->header) ||
        !IsValidFileHeader(snapshot->header.file,
                           SnapshotKind::gromacsPairlist))
    {
        return false;
    }
    const auto cluster_numbers =
        static_cast<size_t>(snapshot->header.cluster_numbers);
    const auto sci_numbers = static_cast<size_t>(snapshot->header.sci_numbers);
    const auto cjpacked_numbers =
        static_cast<size_t>(snapshot->header.cjpacked_numbers);
    const auto excl_numbers =
        static_cast<size_t>(snapshot->header.excl_numbers);
    const auto total_atom_numbers =
        static_cast<size_t>(snapshot->header.total_atom_numbers);
    const auto lj_param_numbers =
        static_cast<size_t>(snapshot->header.lj_param_numbers);
    return ReadVector(&in, &snapshot->cluster_offsets, cluster_numbers) &&
           ReadVector(&in, &snapshot->cluster_valid_masks, cluster_numbers) &&
           ReadVector(&in, &snapshot->cluster_local_masks, cluster_numbers) &&
           ReadVector(&in, &snapshot->sci, sci_numbers) &&
           ReadVector(&in, &snapshot->cjpacked, cjpacked_numbers) &&
           ReadVector(&in, &snapshot->excl, excl_numbers) &&
           ReadVector(&in, &snapshot->sorted_xq, total_atom_numbers) &&
           ReadVector(&in, &snapshot->sorted_lj_type, total_atom_numbers) &&
           ReadVector(&in, &snapshot->sorted_lj_comb, total_atom_numbers) &&
           ReadVector(&in, &snapshot->lj_ab, lj_param_numbers);
}

inline bool WriteSpongeClusteredFullOutputSnapshot(
    const std::string& path,
    const SpongeClusteredFullOutputSnapshot& snapshot)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.good())
    {
        return false;
    }
    if (!WriteBinary(&out, snapshot.header) ||
        !WriteVector(&out, snapshot.cluster_offsets) ||
        !WriteVector(&out, snapshot.cluster_valid_masks) ||
        !WriteVector(&out, snapshot.cluster_local_masks) ||
        !WriteVector(&out, snapshot.super_cluster_offsets) ||
        !WriteVector(&out, snapshot.sci) ||
        !WriteVector(&out, snapshot.record_offsets) ||
        !WriteVector(&out, snapshot.records) ||
        !WriteVector(&out, snapshot.pair_shift_bits) ||
        !WriteVector(&out, snapshot.sorted_atom_ids) ||
        !WriteVector(&out, snapshot.sorted_xq) ||
        !WriteVector(&out, snapshot.sorted_lj_type) ||
        !WriteVector(&out, snapshot.lj_ab) ||
        !WriteVector(&out, snapshot.reference_force) ||
        !WriteVector(&out, snapshot.reference_atom_energy) ||
        !WriteVector(&out, snapshot.reference_atom_virial) ||
        !WriteVector(&out, snapshot.reference_direct_cf_energy) ||
        !WriteVector(&out, snapshot.reference_lj_energy))
    {
        return false;
    }
    return out.good();
}

inline bool ReadSpongeClusteredFullOutputSnapshot(
    const std::string& path,
    SpongeClusteredFullOutputSnapshot* snapshot)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.good() || !ReadBinary(&in, &snapshot->header) ||
        !IsValidFileHeader(snapshot->header.file,
                           SnapshotKind::spongeClusteredFullOutput))
    {
        return false;
    }
    const auto cluster_numbers =
        static_cast<size_t>(snapshot->header.cluster_numbers);
    const auto super_cluster_numbers =
        static_cast<size_t>(snapshot->header.super_cluster_numbers);
    const auto sci_numbers = static_cast<size_t>(snapshot->header.sci_numbers);
    const auto record_numbers =
        static_cast<size_t>(snapshot->header.record_numbers);
    const auto pair_shift_word_numbers =
        static_cast<size_t>(snapshot->header.pair_shift_word_numbers);
    const auto total_atom_numbers =
        static_cast<size_t>(snapshot->header.total_atom_numbers);
    const auto lj_param_numbers =
        static_cast<size_t>(snapshot->header.lj_param_numbers);
    const auto force_reference_numbers =
        static_cast<size_t>(snapshot->header.force_reference_numbers);
    const auto energy_reference_numbers =
        static_cast<size_t>(snapshot->header.energy_reference_numbers);
    const auto virial_reference_numbers =
        static_cast<size_t>(snapshot->header.virial_reference_numbers);
    const auto direct_energy_reference_numbers =
        static_cast<size_t>(snapshot->header.direct_energy_reference_numbers);
    const auto lj_energy_reference_numbers =
        static_cast<size_t>(snapshot->header.lj_energy_reference_numbers);
    return ReadVector(&in, &snapshot->cluster_offsets, cluster_numbers) &&
           ReadVector(&in, &snapshot->cluster_valid_masks, cluster_numbers) &&
           ReadVector(&in, &snapshot->cluster_local_masks, cluster_numbers) &&
           ReadVector(&in, &snapshot->super_cluster_offsets,
                      super_cluster_numbers + 1) &&
           ReadVector(&in, &snapshot->sci, sci_numbers) &&
           ReadVector(&in, &snapshot->record_offsets, sci_numbers + 1) &&
           ReadVector(&in, &snapshot->records, record_numbers) &&
           ReadVector(&in, &snapshot->pair_shift_bits, pair_shift_word_numbers) &&
           ReadVector(&in, &snapshot->sorted_atom_ids, total_atom_numbers) &&
           ReadVector(&in, &snapshot->sorted_xq, total_atom_numbers) &&
           ReadVector(&in, &snapshot->sorted_lj_type, total_atom_numbers) &&
           ReadVector(&in, &snapshot->lj_ab, lj_param_numbers) &&
           ReadVector(&in, &snapshot->reference_force, force_reference_numbers) &&
           ReadVector(&in, &snapshot->reference_atom_energy,
                      energy_reference_numbers) &&
           ReadVector(&in, &snapshot->reference_atom_virial,
                      virial_reference_numbers) &&
           ReadVector(&in, &snapshot->reference_direct_cf_energy,
                      direct_energy_reference_numbers) &&
           ReadVector(&in, &snapshot->reference_lj_energy,
                      lj_energy_reference_numbers);
}

} // namespace nbnxm_microbench
