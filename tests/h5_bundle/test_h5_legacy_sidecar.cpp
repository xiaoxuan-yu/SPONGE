#include <chrono>
#include <filesystem>
#include <fstream>
#include <highfive/highfive.hpp>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils/h5md/h5_legacy_sidecar.hpp"

struct TestFailure : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

#define REQUIRE_TRUE(expr)                                    \
    do                                                        \
    {                                                         \
        if (!(expr))                                          \
        {                                                     \
            std::ostringstream require_message;               \
            require_message << __FILE__ << ":" << __LINE__    \
                            << " requirement failed: " #expr; \
            throw TestFailure(require_message.str());         \
        }                                                     \
    } while (false)

#define REQUIRE_EQ(lhs, rhs)                                          \
    do                                                                \
    {                                                                 \
        const auto require_lhs = (lhs);                               \
        const auto require_rhs = (rhs);                               \
        if (!(require_lhs == require_rhs))                            \
        {                                                             \
            std::ostringstream require_message;                       \
            require_message << __FILE__ << ":" << __LINE__            \
                            << " equality failed: " #lhs " == " #rhs; \
            throw TestFailure(require_message.str());                 \
        }                                                             \
    } while (false)

class FakeController
{
   public:
    bool Command_Exist(const char* key)
    {
        return key != nullptr && commands_.count(key) != 0;
    }

    const char* Command(const char* key)
    {
        const auto iter = commands_.find(key == nullptr ? "" : key);
        if (iter == commands_.end())
        {
            return "";
        }
        return iter->second.c_str();
    }

    void Set_Command(const char* key, const char* value, int check = 1)
    {
        commands_[key == nullptr ? "" : key] = value == nullptr ? "" : value;
        checks_[key == nullptr ? "" : key] = check;
    }

    int Check_Value(const std::string& key) const
    {
        const auto iter = checks_.find(key);
        return iter == checks_.end() ? -1 : iter->second;
    }

   private:
    std::map<std::string, std::string> commands_;
    std::map<std::string, int> checks_;
};

static std::filesystem::path Unique_Temp_Dir(const std::string& name)
{
    const auto stamp =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           ("sponge_h5_legacy_sidecar_" + std::to_string(stamp) + "_" + name);
}

static void Ensure_Group(HighFive::File& file, const std::string& group_path)
{
    if (group_path.empty() || group_path == "/")
    {
        return;
    }
    std::string current;
    std::size_t begin = 1;
    while (begin < group_path.size())
    {
        const std::size_t slash = group_path.find('/', begin);
        const std::string component = group_path.substr(
            begin,
            slash == std::string::npos ? std::string::npos : slash - begin);
        current += "/" + component;
        if (!file.exist(current))
        {
            file.createGroup(current);
        }
        if (slash == std::string::npos)
        {
            break;
        }
        begin = slash + 1;
    }
}

static void Ensure_Parent_Group(HighFive::File& file,
                                const std::string& dataset_path)
{
    const std::size_t slash = dataset_path.find_last_of('/');
    if (slash == std::string::npos || slash == 0)
    {
        return;
    }
    Ensure_Group(file, dataset_path.substr(0, slash));
}

static void Write_String_Vector(HighFive::File& file,
                                const std::string& dataset_path,
                                const std::vector<std::string>& values)
{
    Ensure_Parent_Group(file, dataset_path);
    auto dataset = file.createDataSet<std::string>(
        dataset_path, HighFive::DataSpace({values.size()}));
    dataset.write(values);
}

static void Write_Sidecar_File(const std::filesystem::path& path,
                               const std::vector<std::string>& keys,
                               const std::vector<std::string>& sidecar_paths)
{
    HighFive::File file(path.string(), HighFive::File::Overwrite);
    Write_String_Vector(file, SpongeH5MD::path::legacy_sidecar_keys, keys);
    Write_String_Vector(file, SpongeH5MD::path::legacy_sidecar_paths,
                        sidecar_paths);
}

static void Test_Reads_And_Resolves_Relative_Sidecar_Paths()
{
    const auto dir = Unique_Temp_Dir("read");
    std::filesystem::create_directories(dir / "containers");
    const auto container = dir / "containers" / "topology.spgt.h5";

    Write_Sidecar_File(container, {"mass_in_file", "charge_in_file"},
                       {"sidecars/mass.txt", (dir / "charge.txt").string()});

    std::vector<SpongeH5MD::LegacySidecarBinding> sidecars;
    std::string error;
    REQUIRE_TRUE(SpongeH5MD::Read_Legacy_Sidecars_From_H5(container.string(),
                                                          &sidecars, &error));
    REQUIRE_EQ(sidecars.size(), static_cast<std::size_t>(2));
    REQUIRE_EQ(sidecars[0].key, std::string("mass_in_file"));
    REQUIRE_EQ(sidecars[0].path, (dir / "containers" / "sidecars" / "mass.txt")
                                     .lexically_normal()
                                     .string());
    REQUIRE_EQ(sidecars[1].path,
               (dir / "charge.txt").lexically_normal().string());

    std::filesystem::remove_all(dir);
}

static void Test_Injects_Allowed_Sidecar_Command()
{
    FakeController controller;
    std::vector<SpongeH5MD::LegacySidecarBinding> sidecars = {
        {"mass_in_file", "/tmp/mass.txt"},
    };
    std::string error;

    REQUIRE_TRUE(SpongeH5MD::Inject_Legacy_Sidecar_Commands(
        &controller, sidecars, SpongeH5MD::H5_Topology_Sidecar_Command_Keys(),
        "topology", &error));
    REQUIRE_TRUE(controller.Command_Exist("mass_in_file"));
    REQUIRE_EQ(std::string(controller.Command("mass_in_file")),
               std::string("/tmp/mass.txt"));
    REQUIRE_EQ(controller.Check_Value("mass_in_file"), 0);
}

static void Test_Rejects_Unsupported_Sidecar_Command()
{
    FakeController controller;
    std::vector<SpongeH5MD::LegacySidecarBinding> sidecars = {
        {"coordinate_in_file", "/tmp/coordinate.txt"},
    };
    std::string error;

    REQUIRE_TRUE(!SpongeH5MD::Inject_Legacy_Sidecar_Commands(
        &controller, sidecars, SpongeH5MD::H5_Topology_Sidecar_Command_Keys(),
        "topology", &error));
    REQUIRE_TRUE(error.find("unsupported H5 legacy sidecar key") !=
                 std::string::npos);
}

static void Test_Rejects_Conflicting_Sidecar_Command()
{
    FakeController controller;
    controller.Set_Command("mass_in_file", "/tmp/legacy_mass.txt", 1);
    std::vector<SpongeH5MD::LegacySidecarBinding> sidecars = {
        {"mass_in_file", "/tmp/h5_mass.txt"},
    };
    std::string error;

    REQUIRE_TRUE(!SpongeH5MD::Inject_Legacy_Sidecar_Commands(
        &controller, sidecars, SpongeH5MD::H5_Topology_Sidecar_Command_Keys(),
        "topology", &error));
    REQUIRE_TRUE(error.find("conflicts with existing command") !=
                 std::string::npos);
}

static void Test_Accepts_Relative_Existing_Path_For_Same_Sidecar()
{
    const auto dir = Unique_Temp_Dir("same_path");
    std::filesystem::create_directories(dir / "legacy_sidecars" /
                                        "qc_type_in_file");
    const auto sidecar_path =
        dir / "legacy_sidecars" / "qc_type_in_file" / "qc_type.txt";
    {
        std::ofstream out(sidecar_path);
        out << "1\n0 H\n";
    }

    const auto previous_cwd = std::filesystem::current_path();
    std::filesystem::current_path(dir);
    try
    {
        FakeController controller;
        controller.Set_Command("qc_type_in_file",
                               "legacy_sidecars/qc_type_in_file/qc_type.txt",
                               1);
        std::vector<SpongeH5MD::LegacySidecarBinding> sidecars = {
            {"qc_type_in_file", sidecar_path.string()},
        };
        std::string error;

        REQUIRE_TRUE(SpongeH5MD::Inject_Legacy_Sidecar_Commands(
            &controller, sidecars,
            SpongeH5MD::H5_Topology_Sidecar_Command_Keys(), "topology",
            &error));
        REQUIRE_EQ(std::string(controller.Command("qc_type_in_file")),
                   std::string("legacy_sidecars/qc_type_in_file/qc_type.txt"));
    }
    catch (...)
    {
        std::filesystem::current_path(previous_cwd);
        std::filesystem::remove_all(dir);
        throw;
    }
    std::filesystem::current_path(previous_cwd);
    std::filesystem::remove_all(dir);
}

int main()
{
    try
    {
        Test_Reads_And_Resolves_Relative_Sidecar_Paths();
        Test_Injects_Allowed_Sidecar_Command();
        Test_Rejects_Unsupported_Sidecar_Command();
        Test_Rejects_Conflicting_Sidecar_Command();
        Test_Accepts_Relative_Existing_Path_For_Same_Sidecar();
    }
    catch (const std::exception& err)
    {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    return 0;
}
