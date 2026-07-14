#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <highfive/highfive.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "h5_input_matrix_fixture.hpp"
#include "utils/h5md/h5md_writer.hpp"

namespace
{
constexpr int kSkipReturnCode = 77;

struct PreparedCase
{
    std::filesystem::path root;
    std::filesystem::path mdin;
    std::filesystem::path mdout;
    std::filesystem::path mdinfo;
    std::filesystem::path h5_restart;
};

std::string Read_Text(const std::filesystem::path& path)
{
    std::ifstream in(path.c_str());
    if (!in.good())
    {
        throw TestFailure("failed to read " + path.string());
    }
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

void Write_Text(const std::filesystem::path& path, const std::string& text)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path.c_str());
    out << text;
    if (!out.good())
    {
        throw TestFailure("failed to write " + path.string());
    }
}

void Require_Contains(const std::string& text, const std::string& needle)
{
    if (text.find(needle) == std::string::npos)
    {
        throw TestFailure("expected text to contain: " + needle);
    }
}

void Replace_All(std::string* text, const std::string& from,
                 const std::string& to)
{
    std::size_t pos = 0;
    while ((pos = text->find(from, pos)) != std::string::npos)
    {
        text->replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string Remove_Key_Lines(const std::string& mdin,
                             const std::vector<std::string>& keys)
{
    std::istringstream input(mdin);
    std::ostringstream output;
    std::string line;
    while (std::getline(input, line))
    {
        bool remove = false;
        const auto stripped = line.substr(0, line.find('#'));
        for (const auto& key : keys)
        {
            std::size_t pos = 0;
            while (pos < stripped.size() &&
                   std::isspace(static_cast<unsigned char>(stripped[pos])))
            {
                ++pos;
            }
            if (stripped.compare(pos, key.size(), key) != 0)
            {
                continue;
            }
            pos += key.size();
            while (pos < stripped.size() &&
                   std::isspace(static_cast<unsigned char>(stripped[pos])))
            {
                ++pos;
            }
            if (pos < stripped.size() && stripped[pos] == '=')
            {
                remove = true;
                break;
            }
        }
        if (!remove)
        {
            output << line << "\n";
        }
    }
    return output.str();
}

void Copy_Directory_Contents(const std::filesystem::path& source,
                             const std::filesystem::path& destination)
{
    SpongeH5InputMatrix::Require_Path_Exists(source);
    std::filesystem::create_directories(destination);
    for (const auto& entry : std::filesystem::directory_iterator(source))
    {
        std::filesystem::copy(
            entry.path(), destination / entry.path().filename(),
            std::filesystem::copy_options::recursive |
                std::filesystem::copy_options::overwrite_existing);
    }
}

void Delete_H5_Object_If_Exists(const std::filesystem::path& h5_path,
                                const char* object_path)
{
    HighFive::File file(h5_path.string(), HighFive::File::ReadWrite);
    if (file.exist(object_path))
    {
        H5Ldelete(file.getId(), object_path, H5P_DEFAULT);
    }
}

void Write_H5_String_Overwrite(const std::filesystem::path& h5_path,
                               const std::string& object_path,
                               const std::string& value)
{
    HighFive::File file(h5_path.string(), HighFive::File::ReadWrite);
    if (file.exist(object_path))
    {
        H5Ldelete(file.getId(), object_path.c_str(), H5P_DEFAULT);
    }
    auto dataset = file.createDataSet<std::string>(
        object_path, HighFive::DataSpace::From(value));
    dataset.write(value);
}

void Enable_Meta_In_Restart_Protocol_Sidecar(
    const std::filesystem::path& restart_h5_path)
{
    const std::string cv_text =
        "print\n"
        "{\n"
        "    CV = distance\n"
        "}\n"
        "distance\n"
        "{\n"
        "    CV_type = distance\n"
        "    atom = 0 1\n"
        "}\n"
        "meta\n"
        "{\n"
        "    CV = distance\n"
        "    CV_period = 0\n"
        "    CV_sigma = 0.5\n"
        "    CV_minimal = 0\n"
        "    CV_maximum = 10\n"
        "    CV_grid = 4\n"
        "    edge_in_file = sumhill.log\n"
        "    potential_update_interval = 1000\n"
        "}\n";
    Write_H5_String_Overwrite(
        restart_h5_path,
        SpongeH5MD::Restart_Protocol_Sidecar_Path("cv_in_file"), cv_text);
}

PreparedCase Prepare_Restart_Load_Case(
    const std::filesystem::path& temp_root, const std::string& name,
    const std::filesystem::path& source_dir, const std::string& load_policy,
    const bool nvt_with_nhc, const bool remove_metadynamics_restart_state,
    const bool remove_sits_restart_state, const bool enable_sits,
    const bool enable_meta)
{
    PreparedCase prepared;
    prepared.root = temp_root / name;
    Copy_Directory_Contents(source_dir, prepared.root);
    std::filesystem::create_directories(prepared.root / "out");

    if (remove_metadynamics_restart_state)
    {
        Delete_H5_Object_If_Exists(prepared.root / "restart.spgr.h5",
                                   SpongeH5MD::path::restart_meta);
    }
    if (remove_sits_restart_state)
    {
        Delete_H5_Object_If_Exists(prepared.root / "restart.spgr.h5",
                                   SpongeH5MD::path::restart_sits);
    }
    if (enable_meta)
    {
        Enable_Meta_In_Restart_Protocol_Sidecar(prepared.root /
                                                "restart.spgr.h5");
    }

    std::string mdin = Read_Text(prepared.root / "mdin.bundled.spg.toml");
    Replace_All(&mdin, "input_h5_restart_load = \"full\"",
                "input_h5_restart_load = \"" + load_policy + "\"");
    Replace_All(&mdin, "output_h5_trajectory_path = \"prod.spg.h5md\"",
                "output_h5_trajectory_path = \"out/traj.spg.h5md\"");
    Replace_All(&mdin, "output_h5_restart_path = \"prod.spgr.h5\"",
                "output_h5_restart_path = \"out/restart.spgr.h5\"");
    Replace_All(&mdin, "output_h5_observable_path = \"prod.obs.spg.h5md\"",
                "output_h5_observable_path = \"out/obs.spg.h5md\"");
    Replace_All(&mdin, "mdout = \"mdout.txt\"", "mdout = \"out/mdout.txt\"");
    Replace_All(&mdin, "mdinfo = \"mdinfo.txt\"",
                "mdinfo = \"out/mdinfo.txt\"");

    if (nvt_with_nhc)
    {
        Replace_All(&mdin, "mode = \"rerun\"", "mode = \"nvt\"");
        const std::string step_limit = enable_sits ? "0" : "1";
        Replace_All(&mdin, "step_limit = 10",
                    "step_limit = " + step_limit +
                        "\n"
                        "dt = 0\n"
                        "thermostat = \"nose_hoover_chain\"\n"
                        "target_temperature = 300\n"
                        "write_mdout_interval = 1\n"
                        "write_trajectory_interval = 0\n"
                        "write_restart_file_interval = 1");
        mdin = Remove_Key_Lines(mdin, {"input_h5_trajectory_path",
                                       "input_h5_trajectory_particle_stream"});
        mdin += "\n[nose_hoover_chain]\nlength = 2\n";
    }
    else
    {
        Replace_All(&mdin, "step_limit = 10", "step_limit = 0");
    }
    if (enable_sits)
    {
        mdin +=
            "\n[SITS]\n"
            "mode = \"production\"\n"
            "k_numbers = 2\n"
            "T = \"300/310\"\n";
    }

    prepared.mdin = prepared.root / "mdin.restart_load.spg.toml";
    prepared.mdout = prepared.root / "out" / "mdout.txt";
    prepared.mdinfo = prepared.root / "out" / "mdinfo.txt";
    prepared.h5_restart = prepared.root / "out" / "restart.spgr.h5";
    Write_Text(prepared.mdin, mdin);
    return prepared;
}

std::string Shell_Quote(const std::filesystem::path& path)
{
    std::string text = path.string();
    std::string quoted = "'";
    for (const char c : text)
    {
        if (c == '\'')
        {
            quoted += "'\\''";
        }
        else
        {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

std::filesystem::path Run_SPONGE(const std::filesystem::path& executable,
                                 const PreparedCase& test_case)
{
    const auto log_path = test_case.root / "sponge.stdout.txt";
    const std::string command = Shell_Quote(executable) + " -mdin " +
                                Shell_Quote(test_case.mdin) + " > " +
                                Shell_Quote(log_path) + " 2>&1";
    const int ret = std::system(command.c_str());
    if (ret != 0)
    {
        throw TestFailure("SPONGE restart-load smoke failed for " +
                          test_case.root.filename().string() + "\n" +
                          Read_Text(log_path));
    }
    SpongeH5InputMatrix::Require_Path_Exists(test_case.mdout);
    SpongeH5InputMatrix::Require_Path_Exists(test_case.mdinfo);
    return log_path;
}

void Run_SPONGE_Expect_Failure(const std::filesystem::path& executable,
                               const PreparedCase& test_case,
                               const std::string& expected_error)
{
    const auto log_path = test_case.root / "sponge.stdout.txt";
    const std::string command = Shell_Quote(executable) + " -mdin " +
                                Shell_Quote(test_case.mdin) + " > " +
                                Shell_Quote(log_path) + " 2>&1";
    const int ret = std::system(command.c_str());
    REQUIRE_TRUE(ret != 0);
    Require_Contains(Read_Text(log_path), expected_error);
}

void Require_Restart_Contains(const std::filesystem::path& h5_path,
                              const std::string& object_path)
{
    SpongeH5InputMatrix::Require_Path_Exists(h5_path);
    HighFive::File file(h5_path.string(), HighFive::File::ReadOnly);
    REQUIRE_TRUE(file.exist(object_path));
}

void Require_Runtime_Smoke_Enabled()
{
    const char* enabled = std::getenv("SPONGE_H5_ENABLE_RUNTIME_SMOKE");
    if (enabled == nullptr || std::string(enabled) != "1")
    {
        std::cerr << "Skipping restart-load runtime closure smoke; set "
                     "SPONGE_H5_ENABLE_RUNTIME_SMOKE=1 in a runnable SPONGE "
                     "CPU/GPU environment to enable it.\n";
        std::exit(kSkipReturnCode);
    }
}

void Run_Restart_Load_Runtime_Closure(
    const std::filesystem::path& sponge_executable)
{
    const auto temp_root =
        SpongeH5Test::Unique_Temp_Path("h5_restart_load_runtime_closure");
    std::filesystem::create_directories(temp_root);

    const auto full = SpongeH5InputMatrix::Full_Contract_Rerun_Path();
    const auto pure_source = full / "bundled_input" / "bundle";
    const auto sidecar_source =
        full / "bundled_input_with_legacy_sidecar" / "bundle";

    const auto protocol_sits = Prepare_Restart_Load_Case(
        temp_root, "restart_load_protocol_sidecar_sits_only", sidecar_source,
        "protocol", false, true, false, true, false);
    const auto protocol_sits_log = Run_SPONGE(sponge_executable, protocol_sits);
    SpongeH5InputMatrix::Require_Path_Exists(protocol_sits.root /
                                             ".sponge_h5_restart_protocol" /
                                             "SITS_nk_in_file.txt");
    Require_Contains(Read_Text(protocol_sits_log), "START INITIALIZING SITS");
    Require_Contains(Read_Text(protocol_sits_log),
                     "Read Nk from .sponge_h5_restart_protocol/"
                     "SITS_nk_in_file.txt");

    const auto dynamic_nhc = Prepare_Restart_Load_Case(
        temp_root, "restart_load_dynamic_supported_nhc_nvt", pure_source,
        "dynamic", true, false, false, false, false);
    Run_SPONGE(sponge_executable, dynamic_nhc);
    Require_Restart_Contains(dynamic_nhc.h5_restart,
                             SpongeH5MD::path::restart_nhc);

    const auto full_supported = Prepare_Restart_Load_Case(
        temp_root, "restart_load_full_supported_nhc_sits", sidecar_source,
        "full", true, true, false, true, false);
    const auto full_supported_log =
        Run_SPONGE(sponge_executable, full_supported);
    Require_Contains(Read_Text(full_supported_log),
                     "START INITIALIZING NOSE HOOVER CHAIN");
    Require_Contains(Read_Text(full_supported_log), "START INITIALIZING SITS");
    Require_Contains(Read_Text(full_supported_log),
                     "Read Nk from .sponge_h5_restart_protocol/"
                     "SITS_nk_in_file.txt");

    const auto protocol_meta = Prepare_Restart_Load_Case(
        temp_root, "restart_load_protocol_sidecar_meta_initialized",
        sidecar_source, "protocol", false, false, true, false, true);
    const auto protocol_meta_log = Run_SPONGE(sponge_executable, protocol_meta);
    Require_Contains(Read_Text(protocol_meta_log),
                     "START INITIALIZING 1D-META");
    SpongeH5InputMatrix::Require_Path_Exists(protocol_meta.root / "myhill.log");
    SpongeH5InputMatrix::Require_Path_Exists(protocol_meta.root /
                                             "Meta_Potential.txt");

    const auto dynamic_rerun_reject = Prepare_Restart_Load_Case(
        temp_root, "restart_load_dynamic_rerun_without_nhc_rejects",
        pure_source, "dynamic", false, false, false, false, false);
    Run_SPONGE_Expect_Failure(
        sponge_executable, dynamic_rerun_reject,
        "Restart contains Nose-Hoover chain state, but the "
        "nose_hoover_chain thermostat is not initialized");

    const auto protocol_meta_reject = Prepare_Restart_Load_Case(
        temp_root, "restart_load_protocol_meta_without_module_rejects",
        sidecar_source, "protocol", false, false, false, false, false);
    Run_SPONGE_Expect_Failure(
        sponge_executable, protocol_meta_reject,
        "Restart contains metadynamics state, but the meta module is not "
        "initialized");

    const auto pure_protocol_custom_force = Prepare_Restart_Load_Case(
        temp_root, "restart_load_protocol_pure_bundled_custom_force_native",
        pure_source, "protocol", false, true, false, true, false);
    const auto pure_protocol_custom_force_log =
        Run_SPONGE(sponge_executable, pure_protocol_custom_force);
    SpongeH5InputMatrix::Require_Path_Exists(pure_protocol_custom_force.root /
                                             ".sponge_h5_native_custom_force" /
                                             "pairwise_force.txt");
    SpongeH5InputMatrix::Require_Path_Exists(pure_protocol_custom_force.root /
                                             ".sponge_h5_native_custom_force" /
                                             "custom_pair.txt");
    SpongeH5InputMatrix::Require_Path_Exists(pure_protocol_custom_force.root /
                                             ".sponge_h5_native_custom_force" /
                                             "listed_forces.txt");
    SpongeH5InputMatrix::Require_Path_Exists(pure_protocol_custom_force.root /
                                             ".sponge_h5_native_custom_force" /
                                             "custom_bond.txt");
    Require_Contains(Read_Text(pure_protocol_custom_force_log),
                     "START INITIALIZING PAIRWISE FORCE");
    Require_Contains(Read_Text(pure_protocol_custom_force_log),
                     "START INITIALIZING LISTED FORCES");
    Require_Contains(Read_Text(pure_protocol_custom_force_log),
                     "START INITIALIZING SITS");

    std::filesystem::remove_all(temp_root);
}
}  // namespace

int main(int argc, char** argv)
{
    try
    {
        Require_Runtime_Smoke_Enabled();
        REQUIRE_TRUE(argc >= 2);
        Run_Restart_Load_Runtime_Closure(argv[1]);
    }
    catch (const std::exception& err)
    {
        std::cerr << err.what() << "\n";
        return 1;
    }
    return 0;
}
