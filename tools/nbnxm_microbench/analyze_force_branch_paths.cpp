#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Header {
    char magic[8];
    uint32_t version;
    uint32_t kind;
    uint32_t cluster_size;
    uint32_t super_cluster_clusters;
    uint32_t warp_split_count;
    uint32_t j_group_size;
    uint32_t force_storage_sorted;
    uint32_t use_lj_comb;
    uint32_t lj_type_matrix_stride;
    uint32_t reserved1;
    uint64_t cluster_numbers;
    uint64_t super_cluster_numbers;
    uint64_t sci_numbers;
    uint64_t cjpacked_numbers;
    uint64_t excl_numbers;
    uint64_t pair_shift_word_numbers;
    uint64_t total_atom_numbers;
    uint64_t local_atom_numbers;
    uint64_t lj_param_numbers;
    float cutoff;
    float pme_beta;
    float cell[6];
};

struct Sci {
    int supercluster_id;
    int shift_id;
    int cjpacked_begin;
    int cjpacked_end;
};

struct CjPacked {
    int cj[4];
    uint32_t imask0;
    int excl0;
    uint32_t imask1;
    int excl1;
};

struct Float4 {
    float x, y, z, w;
};

struct Vec3 {
    float x, y, z;
};

struct Counts {
    uint64_t dynamic_inactive = 0;
    uint64_t dynamic_pass = 0;
    uint64_t dynamic_fail = 0;
    uint64_t site_A0 = 0;
    uint64_t site_A1_C1 = 0;
    uint64_t site_A1_C0 = 0;
    uint64_t site_A1_Cmix = 0;
    uint64_t site_Amix_C1 = 0;
    uint64_t site_Amix_C0 = 0;
    uint64_t site_Amix_Cmix = 0;
    uint64_t process_jm = 0;
    uint64_t invalid_j = 0;
    uint64_t oob = 0;
    uint64_t split_excl_zero = 0;
    uint64_t split_excl_nonzero = 0;
    uint64_t unsafe_sci = 0;
};

template <typename T>
void read_exact(std::ifstream& in, T* dst, size_t count)
{
    in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(sizeof(T) * count));
    if (!in) {
        throw std::runtime_error("short read");
    }
}

template <typename T>
void skip_exact(std::ifstream& in, size_t count)
{
    in.seekg(static_cast<std::streamoff>(sizeof(T) * count), std::ios::cur);
    if (!in) {
        throw std::runtime_error("short seek");
    }
}

Vec3 shift_from_id(int shift_id, const float cell[6])
{
    const int fx = shift_id / 9 - 1;
    const int fy = (shift_id % 9) / 3 - 1;
    const int fz = shift_id % 3 - 1;
    return {fx * cell[0] + fy * cell[1] + fz * cell[3],
            fy * cell[2] + fz * cell[4],
            fz * cell[5]};
}

double pct(uint64_t n, uint64_t d)
{
    return d ? 100.0 * static_cast<double>(n) / static_cast<double>(d) : 0.0;
}

void add_site(Counts& c, int active_count, int pass_count)
{
    if (active_count == 0) {
        ++c.site_A0;
    } else if (active_count == 32) {
        if (pass_count == 0) {
            ++c.site_A1_C0;
        } else if (pass_count == active_count) {
            ++c.site_A1_C1;
        } else {
            ++c.site_A1_Cmix;
        }
    } else {
        if (pass_count == 0) {
            ++c.site_Amix_C0;
        } else if (pass_count == active_count) {
            ++c.site_Amix_C1;
        } else {
            ++c.site_Amix_Cmix;
        }
    }
}

void analyze(const std::string& name, const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }

    Header h{};
    read_exact(in, &h, 1);
    skip_exact<int32_t>(in, h.cluster_numbers);
    skip_exact<uint32_t>(in, h.cluster_numbers);
    skip_exact<uint32_t>(in, h.cluster_numbers);
    skip_exact<int32_t>(in, h.super_cluster_numbers + 1);

    std::vector<Sci> sci(h.sci_numbers);
    std::vector<CjPacked> cjpacked(h.cjpacked_numbers);
    std::vector<std::array<uint32_t, 32>> excl(h.excl_numbers);
    read_exact(in, sci.data(), sci.size());
    read_exact(in, cjpacked.data(), cjpacked.size());
    read_exact(in, excl.data(), excl.size());
    skip_exact<uint64_t>(in, h.pair_shift_word_numbers);
    std::vector<int32_t> sci_safe(h.sci_numbers);
    read_exact(in, sci_safe.data(), sci_safe.size());
    skip_exact<int32_t>(in, h.total_atom_numbers);
    std::vector<Float4> xq(h.total_atom_numbers);
    read_exact(in, xq.data(), xq.size());

    Counts c{};
    for (int safe : sci_safe) {
        if (safe == 0) {
            ++c.unsafe_sci;
        }
    }

    const float cutoff_sq = h.cutoff * h.cutoff;
    for (const Sci& s : sci) {
        const Vec3 sh = shift_from_id(s.shift_id, h.cell);
        const int cluster_i_start = s.supercluster_id * 8;
        for (int pidx = s.cjpacked_begin; pidx < s.cjpacked_end; ++pidx) {
            const CjPacked& p = cjpacked[pidx];
            for (int split = 0; split < 2; ++split) {
                const uint32_t imask = split == 0 ? p.imask0 : p.imask1;
                const int exclusion_index = split == 0 ? p.excl0 : p.excl1;
                if (imask == 0) {
                    continue;
                }
                if (exclusion_index != 0) {
                    ++c.split_excl_nonzero;
                } else {
                    ++c.split_excl_zero;
                }
                for (int jm = 0; jm < 4; ++jm) {
                    const uint32_t jm_mask = 0xffu << (jm * 8);
                    if ((imask & jm_mask) == 0) {
                        continue;
                    }
                    const int cluster_j = p.cj[jm];
                    if (cluster_j < 0) {
                        ++c.invalid_j;
                        continue;
                    }
                    ++c.process_jm;
                    for (int I = 0; I < 8; ++I) {
                        int active_count = 0;
                        int pass_count = 0;
                        const uint32_t bit = 1u << (jm * 8 + I);
                        for (int split_j_lane = 0; split_j_lane < 4; ++split_j_lane) {
                            const int j_lane = split * 4 + split_j_lane;
                            const uint64_t sorted_j = static_cast<uint64_t>(cluster_j) * 8 + j_lane;
                            for (int i_lane = 0; i_lane < 8; ++i_lane) {
                                const uint64_t sorted_i =
                                    static_cast<uint64_t>(cluster_i_start + I) * 8 + i_lane;
                                if (sorted_i >= xq.size() || sorted_j >= xq.size()) {
                                    ++c.oob;
                                    continue;
                                }
                                uint32_t pair_bits = 0xffffffffu;
                                if (exclusion_index != 0) {
                                    pair_bits = (exclusion_index >= 0 &&
                                                 static_cast<uint64_t>(exclusion_index) < excl.size())
                                                    ? excl[exclusion_index][split_j_lane * 8 + i_lane]
                                                    : 0u;
                                }
                                const bool active = ((imask & pair_bits) & bit) != 0;
                                if (!active) {
                                    ++c.dynamic_inactive;
                                    continue;
                                }
                                ++active_count;
                                const Float4& xi = xq[sorted_i];
                                const Float4& xj = xq[sorted_j];
                                const float dx = xj.x - (xi.x + sh.x);
                                const float dy = xj.y - (xi.y + sh.y);
                                const float dz = xj.z - (xi.z + sh.z);
                                const bool pass = dx * dx + dy * dy + dz * dz < cutoff_sq;
                                if (pass) {
                                    ++pass_count;
                                    ++c.dynamic_pass;
                                } else {
                                    ++c.dynamic_fail;
                                }
                            }
                        }
                        add_site(c, active_count, pass_count);
                    }
                }
            }
        }
    }

    const uint64_t sites = c.site_A0 + c.site_A1_C1 + c.site_A1_C0 + c.site_A1_Cmix +
                           c.site_Amix_C1 + c.site_Amix_C0 + c.site_Amix_Cmix;
    const uint64_t dynamic = c.dynamic_inactive + c.dynamic_pass + c.dynamic_fail;
    const uint64_t active = c.dynamic_pass + c.dynamic_fail;

    std::cout << "== " << name << " ==\n";
    std::cout << "magic=" << std::string(h.magic, h.magic + 8)
              << " version=" << h.version << " kind=" << h.kind
              << " clusters=" << h.cluster_numbers << " sci=" << h.sci_numbers
              << " cjpacked=" << h.cjpacked_numbers << " excl=" << h.excl_numbers
              << " atoms=" << h.total_atom_numbers << " cutoff=" << h.cutoff
              << " unsafe_sci=" << c.unsafe_sci << " invalid_j=" << c.invalid_j
              << " oob=" << c.oob << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "dynamic_lanes total=" << dynamic
              << " inactive=" << c.dynamic_inactive << " (" << pct(c.dynamic_inactive, dynamic)
              << "%) active_pass=" << c.dynamic_pass << " (" << pct(c.dynamic_pass, dynamic)
              << "%) active_fail=" << c.dynamic_fail << " (" << pct(c.dynamic_fail, dynamic)
              << "%)\n";
    std::cout << "active_lanes pass=" << c.dynamic_pass << " (" << pct(c.dynamic_pass, active)
              << "%) fail=" << c.dynamic_fail << " (" << pct(c.dynamic_fail, active) << "%)\n";
    std::cout << "warp_sites total=" << sites << " process_jm=" << c.process_jm << "\n";
    auto print_site = [&](const char* label, uint64_t v) {
        std::cout << "  " << label << "=" << v << " (" << pct(v, sites) << "%)\n";
    };
    print_site("A0", c.site_A0);
    print_site("A1_C1", c.site_A1_C1);
    print_site("A1_C0", c.site_A1_C0);
    print_site("A1_Cmix", c.site_A1_Cmix);
    print_site("Amix_C1", c.site_Amix_C1);
    print_site("Amix_C0", c.site_Amix_C0);
    print_site("Amix_Cmix", c.site_Amix_Cmix);
    std::cout << "split_exclusion_indices zero=" << c.split_excl_zero
              << " nonzero=" << c.split_excl_nonzero << "\n\n";
}

}  // namespace

int main(int argc, char** argv)
{
    if ((argc - 1) % 2 != 0 || argc == 1) {
        std::cerr << "usage: " << argv[0] << " NAME SNAPSHOT [NAME SNAPSHOT...]\n";
        return 2;
    }
    try {
        for (int i = 1; i < argc; i += 2) {
            analyze(argv[i], argv[i + 1]);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
