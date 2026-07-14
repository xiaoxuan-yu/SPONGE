#include <array>
#include <filesystem>
#include <highfive/highfive.hpp>
#include <string>
#include <vector>

#include "h5_bundle_test_common.hpp"
#include "utils/h5md/highfive_backend.hpp"
#include "utils/h5md/input_validation.hpp"
#include "utils/h5md/restart_h5_writer.hpp"
#include "utils/h5md/trajectory_h5_writer.hpp"

using namespace SpongeH5Test;
using namespace SpongeH5MD;

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

template <typename T>
static void Write_Scalar(HighFive::File& file, const std::string& path,
                         const T& value)
{
    Ensure_Parent_Group(file, path);
    auto dataset =
        file.createDataSet<T>(path, HighFive::DataSpace::From(value));
    dataset.write(value);
}

static void Write_Float_Vector(HighFive::File& file, const std::string& path,
                               const std::vector<float>& values)
{
    Ensure_Parent_Group(file, path);
    auto dataset =
        file.createDataSet<float>(path, HighFive::DataSpace::From(values));
    dataset.write(values);
}

static void Write_Topology_Metadata(const std::filesystem::path& path,
                                    std::int64_t atom_count)
{
    HighFive::File file(path.string(), HighFive::File::Overwrite);
    Write_Scalar(file, "/schema/version", std::string("1"));
    Write_Scalar(file, "/topology/atom_count", atom_count);
    Write_Scalar(file, "/topology/atom_order_hash", std::string("atoms"));
    Write_Scalar(file, "/topology/topology_hash", std::string("top"));
    Write_Scalar(file, "/topology/forcefield_hash", std::string("ff"));
}

static void Write_Protocol_Metadata(const std::filesystem::path& path,
                                    const std::string& topology_hash)
{
    HighFive::File file(path.string(), HighFive::File::Overwrite);
    Write_Scalar(file, "/schema/version", std::string("1"));
    Write_Scalar(file, "/protocol/topology_compatibility/topology_hash",
                 topology_hash);
    Write_Scalar(file, "/identity/content_hash", std::string("protocol"));
}

static SpongeH5OutputPlan::ResolvedOutputPlan Make_Restart_Output_Plan(
    const std::filesystem::path& restart_path)
{
    SpongeH5OutputPlan::ResolvedOutputPlan plan;
    plan.restart.enabled = true;
    plan.restart.path = restart_path.string();
    return plan;
}

static SpongeH5OutputPlan::ResolvedOutputPlan Make_Trajectory_Output_Plan(
    const std::filesystem::path& trajectory_path)
{
    SpongeH5OutputPlan::ResolvedOutputPlan plan;
    plan.trajectory.enabled = true;
    plan.trajectory.path = trajectory_path.string();
    plan.trajectory.vds = false;
    return plan;
}

static void Write_Restart_File(const std::filesystem::path& path,
                               std::size_t atom_count,
                               bool include_dynamic_state,
                               bool include_sits_state = false,
                               bool include_metad_state = false,
                               bool include_protocol_sidecar_state = false)
{
    std::vector<float> position(atom_count * 3, 0.0f);
    std::vector<float> velocity(atom_count * 3, 0.0f);
    for (std::size_t i = 0; i < position.size(); ++i)
    {
        position[i] = static_cast<float>(i + 1);
        velocity[i] = static_cast<float>(i) * 0.1f;
    }
    const std::array<float, 9> box = {
        10.0f, 0.0f, 0.0f, 0.0f, 20.0f, 0.0f, 0.0f, 0.0f, 30.0f,
    };
    const std::vector<float> nhc_state = {0.25f, 0.50f};
    const std::vector<float> sits_nk_state = {1.0f, 2.0f};

    HighFiveBackend backend;
    RestartH5Writer writer(&backend);
    REQUIRE_TRUE(writer.Open(Make_Restart_Output_Plan(path), "1"));
    REQUIRE_TRUE(writer.Define_Structural_State(atom_count, true));
    REQUIRE_TRUE(writer.Write_Structural_State(10, 0.02, position.data(),
                                               box.data(), velocity.data()));
    if (include_dynamic_state)
    {
        REQUIRE_TRUE(writer.Write_Nose_Hoover_Chain_State(nhc_state.data(), 1));
    }
    if (include_sits_state)
    {
        REQUIRE_TRUE(writer.Write_Sits_State("SITS", "nk", sits_nk_state.data(),
                                             sits_nk_state.size()));
    }
    if (include_metad_state)
    {
        REQUIRE_TRUE(writer.Write_Metad_State_Text("meta", "hills", "HILLS\n"));
        REQUIRE_TRUE(
            writer.Write_Metad_State_Text("meta", "history", "HISTORY\n"));
    }
    if (include_protocol_sidecar_state)
    {
        REQUIRE_TRUE(
            writer.Write_Protocol_Sidecar_Text("cv_in_file", "CV_PAYLOAD\n"));
    }
    REQUIRE_TRUE(writer.Finalize());
    REQUIRE_TRUE(writer.Close());
}

static void Add_Unsupported_Dynamic_State(const std::filesystem::path& path)
{
    HighFive::File file(path.string(), HighFive::File::ReadWrite);
    Write_Scalar(file, SpongeH5MD::Restart_Rng_State_Path("middle_langevin"),
                 std::string("unsupported:philox_device_state"));
}

static void Add_Unsupported_Protocol_State(const std::filesystem::path& path)
{
    HighFive::File file(path.string(), HighFive::File::ReadWrite);
    Write_Float_Vector(
        file,
        std::string(SpongeH5MD::path::restart_bias) + "/unsupported/state",
        {1.0f});
}

static void Write_Trajectory_File(const std::filesystem::path& path,
                                  std::size_t atom_count)
{
    std::vector<float> position(atom_count * 3, 0.0f);
    for (std::size_t i = 0; i < position.size(); ++i)
    {
        position[i] = static_cast<float>(i + 1);
    }
    const std::array<float, 9> box = {
        10.0f, 0.0f, 0.0f, 0.0f, 20.0f, 0.0f, 0.0f, 0.0f, 30.0f,
    };

    HighFiveBackend backend;
    TrajectoryH5Writer writer(&backend);
    REQUIRE_TRUE(
        writer.Open_Single_File(Make_Trajectory_Output_Plan(path), "1"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(atom_count, false, false));
    REQUIRE_TRUE(
        writer.Append_Particle_Frame(10, 0.02, position.data(), box.data()));
    REQUIRE_TRUE(writer.Finalize());
    REQUIRE_TRUE(writer.Close());
}

static SpongeH5InputPlan::ResolvedInputPlan Make_Input_Plan(
    const std::filesystem::path& topology_path,
    const std::filesystem::path& protocol_path,
    const std::filesystem::path& restart_path)
{
    SpongeH5InputPlan::ResolvedInputPlan plan;
    plan.any_h5_input_enabled = true;
    plan.legacy_input_allowed = false;
    plan.topology.enabled = true;
    plan.topology.path = topology_path.string();
    plan.protocol.enabled = true;
    plan.protocol.path = protocol_path.string();
    plan.restart.binding.enabled = true;
    plan.restart.binding.path = restart_path.string();
    return plan;
}

static void Require_Valid(
    const SpongeH5InputValidation::ValidationResult& result)
{
    if (!result.valid)
    {
        throw TestFailure(result.error_message);
    }
}

static void Require_Invalid_Contains(
    const SpongeH5InputValidation::ValidationResult& result,
    const std::string& needle)
{
    REQUIRE_TRUE(!result.valid);
    REQUIRE_TRUE(result.error_message.find(needle) != std::string::npos);
}

static void Test_Validates_Structural_Bundle_Metadata()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_valid");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false);

    Require_Valid(SpongeH5InputValidation::Validate_Resolved_Input_Plan(
        Make_Input_Plan(topology, protocol, restart)));

    std::filesystem::remove_all(dir);
}

static void Test_Validates_Controller_Input_Bindings()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_controller");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";
    const auto trajectory = dir / "prod.spg.h5md";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, true, true);
    Write_Trajectory_File(trajectory, 2);

    CONTROLLER controller;
    controller.Set("mode", "rerun");
    controller.Set("input_h5_topology_path", topology.string());
    controller.Set("input_h5_protocol_path", protocol.string());
    controller.Set("input_h5_restart_path", restart.string());
    controller.Set("input_h5_restart_load", "full");
    controller.Set("input_h5_trajectory_path", trajectory.string());
    controller.Set("input_h5_trajectory_particle_stream", "all");

    Require_Valid(
        SpongeH5InputValidation::Validate_Input_Bindings(&controller));

    std::filesystem::remove_all(dir);
}

static void Test_Controller_Input_Bindings_Propagate_Resolver_Errors()
{
    {
        CONTROLLER controller;
        controller.Set("mode", "nve");
        controller.Set("input_h5_protocol_path", "protocol.spgp.h5");
        controller.Set("input_h5_restart_path", "restart.spgr.h5");

        Require_Invalid_Contains(
            SpongeH5InputValidation::Validate_Input_Bindings(&controller),
            "input_h5_topology_path");
    }

    {
        CONTROLLER controller;
        controller.Set("mode", "nve");
        controller.Set("input_h5_topology_path", "topology.spgt.h5");
        controller.Set("input_h5_protocol_path", "protocol.spgp.h5");
        controller.Set("input_h5_restart_path", "restart.spgr.h5");
        controller.Set("input_h5_restart_load", "custom");

        Require_Invalid_Contains(
            SpongeH5InputValidation::Validate_Input_Bindings(&controller),
            "input_h5_restart_load = custom is reserved");
    }
}

static void Test_Rejects_Restart_Atom_Count_Mismatch()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_restart_mismatch");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 3);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false);

    Require_Invalid_Contains(
        SpongeH5InputValidation::Validate_Resolved_Input_Plan(
            Make_Input_Plan(topology, protocol, restart)),
        "restart atom_count");

    std::filesystem::remove_all(dir);
}

static void Test_Validates_H5md_Rerun_Trajectory_Metadata()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_trajectory");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";
    const auto trajectory = dir / "prod.spg.h5md";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false);
    Write_Trajectory_File(trajectory, 2);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.trajectory.binding.enabled = true;
    plan.trajectory.binding.path = trajectory.string();
    Require_Valid(SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan));

    std::filesystem::remove_all(dir);
}

static void Test_Rejects_Missing_Requested_Trajectory_Stream()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_stream");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";
    const auto trajectory = dir / "prod.spg.h5md";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false);
    Write_Trajectory_File(trajectory, 2);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.trajectory.binding.enabled = true;
    plan.trajectory.binding.path = trajectory.string();
    plan.trajectory.particle_stream = "solute";
    Require_Invalid_Contains(
        SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan),
        "position");

    std::filesystem::remove_all(dir);
}

static void Test_Validates_Dynamic_Load_When_Dynamic_State_Is_Present()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_dynamic");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, true);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.restart.load_policy =
        SpongeH5InputContract::RestartLoadPolicy::dynamic;
    Require_Valid(SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan));

    std::filesystem::remove_all(dir);
}

static void Test_Rejects_Dynamic_Load_When_Dynamic_State_Is_Absent()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_dynamic_absent");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.restart.load_policy =
        SpongeH5InputContract::RestartLoadPolicy::dynamic;
    Require_Invalid_Contains(
        SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan),
        "dynamic state is absent");

    std::filesystem::remove_all(dir);
}

static void Test_Rejects_Dynamic_Load_When_Only_Unsupported_State_Is_Present()
{
    const auto dir =
        Unique_Temp_Path("h5_input_validation_dynamic_unsupported");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false);
    Add_Unsupported_Dynamic_State(restart);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.restart.load_policy =
        SpongeH5InputContract::RestartLoadPolicy::dynamic;
    Require_Invalid_Contains(
        SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan),
        "unsupported payloads");

    std::filesystem::remove_all(dir);
}

static void Test_Validates_Protocol_Load_When_Sits_State_Is_Present()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_protocol");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false, true);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.restart.load_policy =
        SpongeH5InputContract::RestartLoadPolicy::protocol;
    Require_Valid(SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan));

    std::filesystem::remove_all(dir);
}

static void Test_Validates_Protocol_Load_When_Metad_State_Is_Present()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_protocol_metad");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false, false, true);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.restart.load_policy =
        SpongeH5InputContract::RestartLoadPolicy::protocol;
    Require_Valid(SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan));

    std::filesystem::remove_all(dir);
}

static void Test_Validates_Protocol_Load_When_Sidecar_State_Is_Present()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_protocol_sidecar");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false, false, false, true);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.restart.load_policy =
        SpongeH5InputContract::RestartLoadPolicy::protocol;
    Require_Valid(SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan));

    std::filesystem::remove_all(dir);
}

static void Test_Validates_Full_Load_When_Supported_States_Are_Present()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_full");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, true, true);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.restart.load_policy = SpongeH5InputContract::RestartLoadPolicy::full;
    Require_Valid(SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan));

    std::filesystem::remove_all(dir);
}

static void Test_Rejects_Protocol_Load_When_Only_Unsupported_State_Is_Present()
{
    const auto dir =
        Unique_Temp_Path("h5_input_validation_protocol_unsupported");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false);
    Add_Unsupported_Protocol_State(restart);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.restart.load_policy =
        SpongeH5InputContract::RestartLoadPolicy::protocol;
    Require_Invalid_Contains(
        SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan),
        "no currently supported payload");

    std::filesystem::remove_all(dir);
}

static void Test_Rejects_Protocol_Load_When_Protocol_State_Is_Absent()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_protocol_absent");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.restart.load_policy =
        SpongeH5InputContract::RestartLoadPolicy::protocol;
    Require_Invalid_Contains(
        SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan),
        "protocol state is absent");

    std::filesystem::remove_all(dir);
}

int main()
{
    return Run_Test(
        []
        {
            Test_Validates_Structural_Bundle_Metadata();
            Test_Validates_Controller_Input_Bindings();
            Test_Controller_Input_Bindings_Propagate_Resolver_Errors();
            Test_Rejects_Restart_Atom_Count_Mismatch();
            Test_Validates_H5md_Rerun_Trajectory_Metadata();
            Test_Rejects_Missing_Requested_Trajectory_Stream();
            Test_Validates_Dynamic_Load_When_Dynamic_State_Is_Present();
            Test_Rejects_Dynamic_Load_When_Dynamic_State_Is_Absent();
            Test_Rejects_Dynamic_Load_When_Only_Unsupported_State_Is_Present();
            Test_Validates_Protocol_Load_When_Sits_State_Is_Present();
            Test_Validates_Protocol_Load_When_Metad_State_Is_Present();
            Test_Validates_Protocol_Load_When_Sidecar_State_Is_Present();
            Test_Validates_Full_Load_When_Supported_States_Are_Present();
            Test_Rejects_Protocol_Load_When_Only_Unsupported_State_Is_Present();
            Test_Rejects_Protocol_Load_When_Protocol_State_Is_Absent();
        });
}
