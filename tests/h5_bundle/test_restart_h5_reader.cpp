#include <array>
#include <cmath>
#include <filesystem>
#include <highfive/highfive.hpp>
#include <string>
#include <vector>

#include "h5_bundle_test_common.hpp"
#include "utils/h5md/highfive_backend.hpp"
#include "utils/h5md/restart_h5_reader.hpp"
#include "utils/h5md/restart_h5_writer.hpp"

using namespace SpongeH5Test;
using namespace SpongeH5MD;

static SpongeH5OutputPlan::ResolvedOutputPlan Make_Restart_Plan(
    const std::filesystem::path& restart_path)
{
    SpongeH5OutputPlan::ResolvedOutputPlan plan;
    plan.restart.enabled = true;
    plan.restart.path = restart_path.string();
    return plan;
}

static void Require_Float_Vector_Close(const std::vector<float>& actual,
                                       const std::vector<float>& expected)
{
    REQUIRE_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        REQUIRE_TRUE(std::fabs(actual[i] - expected[i]) < 1.0e-6f);
    }
}

static void Require_Box_Close(const std::array<float, 9>& actual,
                              const std::array<float, 9>& expected)
{
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        REQUIRE_TRUE(std::fabs(actual[i] - expected[i]) < 1.0e-6f);
    }
}

static void Require_Writer(bool ok, const RestartH5Writer& writer,
                           const char* operation)
{
    if (!ok)
    {
        throw TestFailure(std::string(operation) + ": " + writer.Last_Error());
    }
}

static void Require_Reader(bool ok, const RestartH5Reader& reader,
                           const char* operation)
{
    if (!ok)
    {
        throw TestFailure(std::string(operation) + ": " + reader.Last_Error());
    }
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

static void Add_Float_Vector_Dataset(const std::filesystem::path& file_path,
                                     const std::string& dataset_path,
                                     const std::vector<float>& values)
{
    HighFive::File file(file_path.string(), HighFive::File::ReadWrite);
    Ensure_Parent_Group(file, dataset_path);
    auto dataset = file.createDataSet<float>(dataset_path,
                                             HighFive::DataSpace::From(values));
    dataset.write(values);
}

static void Write_Restart_File(const std::filesystem::path& file_path,
                               bool include_velocity)
{
    const std::vector<float> position = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
    };
    const std::vector<float> velocity = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f,
    };
    const std::array<float, 9> box = {
        10.0f, 0.0f, 0.0f, 0.0f, 20.0f, 0.0f, 0.0f, 0.0f, 30.0f,
    };

    HighFiveBackend backend;
    RestartH5Writer writer(&backend);
    Require_Writer(writer.Open(Make_Restart_Plan(file_path), "1"), writer,
                   "open restart writer");
    Require_Writer(writer.Define_Structural_State(3, include_velocity), writer,
                   "define structural restart state");
    Require_Writer(writer.Write_Structural_State(
                       42, 0.084, position.data(), box.data(),
                       include_velocity ? velocity.data() : nullptr),
                   writer, "write structural restart state");
    Require_Writer(writer.Finalize(), writer, "finalize restart writer");
    Require_Writer(writer.Close(), writer, "close restart writer");
}

static void Write_Restart_File_With_Module_State(
    const std::filesystem::path& file_path)
{
    const std::vector<float> position = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
    };
    const std::vector<float> velocity = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
    };
    const std::array<float, 9> box = {
        10.0f, 0.0f, 0.0f, 0.0f, 20.0f, 0.0f, 0.0f, 0.0f, 30.0f,
    };
    const std::vector<float> nhc_state = {
        0.25f,
        0.50f,
        0.75f,
        1.00f,
    };
    const std::vector<float> sits_weights = {2.0f, 3.0f, 5.0f};

    HighFiveBackend backend;
    RestartH5Writer writer(&backend);
    Require_Writer(writer.Open(Make_Restart_Plan(file_path), "1"), writer,
                   "open restart writer with module state");
    Require_Writer(writer.Define_Structural_State(2, true), writer,
                   "define structural restart state with module state");
    Require_Writer(writer.Write_Structural_State(7, 0.014, position.data(),
                                                 box.data(), velocity.data()),
                   writer, "write structural restart state with module state");
    Require_Writer(writer.Write_Nose_Hoover_Chain_State(nhc_state.data(), 2),
                   writer, "write NHC restart state");
    Require_Writer(writer.Write_Integrator_State_Text("mode", "npt"), writer,
                   "write integrator restart mode");
    Require_Writer(
        writer.Write_Rng_State_Text("bussi_thermostat", "12345 67890"), writer,
        "write Bussi RNG restart state");
    Require_Writer(
        writer.Write_Thermostat_State_Text("bussi_thermostat", "rng_engine",
                                           "std::default_random_engine"),
        writer, "write Bussi RNG engine restart state");
    const float bussi_lambda[1] = {0.95f};
    Require_Writer(writer.Write_Thermostat_State_Float(
                       "bussi_thermostat", "lambda", bussi_lambda, 1),
                   writer, "write Bussi lambda restart state");
    const float pressure_g[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Require_Writer(
        writer.Write_Rng_State_Text("pressure_based_barostat", "24680 13579"),
        writer, "write pressure barostat RNG restart state");
    Require_Writer(writer.Write_Barostat_State_Float("pressure_based_barostat",
                                                     "g", pressure_g, 6),
                   writer, "write pressure barostat g restart state");
    Require_Writer(
        writer.Write_Sits_State("sits_bias", "weights", sits_weights.data(),
                                sits_weights.size()),
        writer, "write SITS restart state");
    Require_Writer(
        writer.Write_Metad_State_Text("metad_bias", "hills", "HILLS_PAYLOAD"),
        writer, "write metad hills restart state");
    Require_Writer(writer.Write_Metad_State_Text("metad_bias", "history",
                                                 "HISTORY_PAYLOAD"),
                   writer, "write metad history restart state");
    Require_Writer(
        writer.Write_Protocol_Sidecar_Text("cv_in_file", "CV_PAYLOAD"), writer,
        "write protocol sidecar restart state");
    Require_Writer(writer.Finalize(), writer,
                   "finalize restart writer with module state");
    Require_Writer(writer.Close(), writer,
                   "close restart writer with module state");
}

static void Test_Restart_Reader_Round_Trips_Structural_State()
{
    const auto dir = Unique_Temp_Path("restart_reader");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "prod.spgr.h5";

    Write_Restart_File(file_path, true);

    RestartH5Reader reader;
    Require_Reader(reader.Open(file_path.string()), reader,
                   "open restart reader");

    SpongeH5InputMetadata::RestartMetadata metadata;
    Require_Reader(reader.Read_Metadata(&metadata), reader,
                   "read restart metadata");
    REQUIRE_EQ(metadata.schema_version, std::string("1"));
    REQUIRE_EQ(metadata.atom_count, static_cast<std::int64_t>(3));
    REQUIRE_TRUE(metadata.has_structural_state);
    REQUIRE_TRUE(metadata.has_velocity);
    REQUIRE_TRUE(!metadata.has_protocol_state);

    RestartStructuralState state;
    Require_Reader(reader.Read_Structural_State(&state), reader,
                   "read restart structural state");
    REQUIRE_EQ(state.step, static_cast<std::int64_t>(42));
    REQUIRE_TRUE(std::fabs(state.time - 0.084) < 1.0e-12);
    REQUIRE_EQ(state.atom_count, static_cast<std::size_t>(3));
    REQUIRE_TRUE(state.has_velocity);

    Require_Float_Vector_Close(
        state.position_xyz,
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    Require_Float_Vector_Close(
        state.velocity_xyz,
        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f});
    Require_Box_Close(state.box_edges, {10.0f, 0.0f, 0.0f, 0.0f, 20.0f, 0.0f,
                                        0.0f, 0.0f, 30.0f});

    std::filesystem::remove_all(dir);
}

static void Test_Restart_Reader_Allows_No_Velocity_State()
{
    const auto dir = Unique_Temp_Path("restart_reader_no_velocity");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "prod.spgr.h5";

    Write_Restart_File(file_path, false);

    RestartH5Reader reader;
    Require_Reader(reader.Open(file_path.string()), reader,
                   "open restart reader without velocity");

    SpongeH5InputMetadata::RestartMetadata metadata;
    Require_Reader(reader.Read_Metadata(&metadata), reader,
                   "read restart metadata without velocity");
    REQUIRE_TRUE(metadata.has_structural_state);
    REQUIRE_TRUE(!metadata.has_velocity);

    RestartStructuralState state;
    Require_Reader(reader.Read_Structural_State(&state), reader,
                   "read restart structural state without velocity");
    REQUIRE_TRUE(!state.has_velocity);
    REQUIRE_TRUE(state.velocity_xyz.empty());

    std::filesystem::remove_all(dir);
}

static void Test_Restart_Reader_Reports_Unsupported_Dynamic_State_As_Metadata()
{
    const auto dir = Unique_Temp_Path("restart_reader_unsupported_dynamic");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "prod.spgr.h5";

    Write_Restart_File(file_path, true);
    {
        HighFive::File file(file_path.string(), HighFive::File::ReadWrite);
        Ensure_Parent_Group(
            file, SpongeH5MD::Restart_Rng_State_Path("middle_langevin"));
        auto dataset = file.createDataSet<std::string>(
            SpongeH5MD::Restart_Rng_State_Path("middle_langevin"),
            HighFive::DataSpace::From(std::string("unsupported")));
        dataset.write(std::string("unsupported"));
    }

    RestartH5Reader reader;
    Require_Reader(reader.Open(file_path.string()), reader,
                   "open restart reader with unsupported dynamic state");

    SpongeH5InputMetadata::RestartMetadata metadata;
    Require_Reader(reader.Read_Metadata(&metadata), reader,
                   "read restart metadata with unsupported dynamic state");
    REQUIRE_TRUE(metadata.has_dynamic_state);

    RestartDynamicState dynamic_state;
    Require_Reader(reader.Read_Dynamic_State(&dynamic_state), reader,
                   "read unsupported dynamic state");
    REQUIRE_TRUE(!dynamic_state.has_nose_hoover_chain);

    std::filesystem::remove_all(dir);
}

static void Test_Restart_Reader_Round_Trips_Dynamic_And_Protocol_State()
{
    const auto dir = Unique_Temp_Path("restart_reader_module_state");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "prod.spgr.h5";

    Write_Restart_File_With_Module_State(file_path);

    RestartH5Reader reader;
    Require_Reader(reader.Open(file_path.string()), reader,
                   "open restart reader with module state");

    SpongeH5InputMetadata::RestartMetadata metadata;
    Require_Reader(reader.Read_Metadata(&metadata), reader,
                   "read restart metadata with module state");
    REQUIRE_TRUE(metadata.has_dynamic_state);
    REQUIRE_TRUE(metadata.has_protocol_state);

    RestartDynamicState dynamic_state;
    Require_Reader(reader.Read_Dynamic_State(&dynamic_state), reader,
                   "read restart dynamic state");
    REQUIRE_TRUE(dynamic_state.has_nose_hoover_chain);
    REQUIRE_EQ(dynamic_state.nose_hoover_chain_pair_count,
               static_cast<std::size_t>(2));
    Require_Float_Vector_Close(
        dynamic_state.nose_hoover_chain_coordinate_velocity_pairs,
        {0.25f, 0.50f, 0.75f, 1.00f});
    REQUIRE_EQ(dynamic_state.integrator_state_text["mode"], std::string("npt"));
    REQUIRE_EQ(dynamic_state.rng_state_text["bussi_thermostat"],
               std::string("12345 67890"));
    REQUIRE_EQ(
        dynamic_state.thermostat_text_states["bussi_thermostat"]["rng_engine"],
        std::string("std::default_random_engine"));
    Require_Float_Vector_Close(
        dynamic_state.thermostat_float_states["bussi_thermostat"]["lambda"],
        {0.95f});
    REQUIRE_EQ(dynamic_state.rng_state_text["pressure_based_barostat"],
               std::string("24680 13579"));
    Require_Float_Vector_Close(
        dynamic_state.barostat_float_states["pressure_based_barostat"]["g"],
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    RestartProtocolState protocol_state;
    Require_Reader(reader.Read_Protocol_State(&protocol_state), reader,
                   "read restart protocol state");
    REQUIRE_EQ(protocol_state.sits_states.size(), static_cast<std::size_t>(1));
    REQUIRE_EQ(protocol_state.sits_states[0].module_name,
               std::string("sits_bias"));
    REQUIRE_EQ(protocol_state.sits_states[0].float_states.count("weights"),
               static_cast<std::size_t>(1));
    Require_Float_Vector_Close(
        protocol_state.sits_states[0].float_states["weights"],
        {2.0f, 3.0f, 5.0f});

    REQUIRE_EQ(protocol_state.metadynamics_states.size(),
               static_cast<std::size_t>(1));
    REQUIRE_EQ(protocol_state.metadynamics_states[0].name,
               std::string("metad_bias"));
    REQUIRE_EQ(protocol_state.metadynamics_states[0].text_states["hills"],
               std::string("HILLS_PAYLOAD"));
    REQUIRE_EQ(protocol_state.metadynamics_states[0].text_states["history"],
               std::string("HISTORY_PAYLOAD"));
    REQUIRE_EQ(protocol_state.sidecar_text_states.size(),
               static_cast<std::size_t>(1));
    REQUIRE_EQ(protocol_state.sidecar_text_states[0].key,
               std::string("cv_in_file"));
    REQUIRE_EQ(protocol_state.sidecar_text_states[0].text,
               std::string("CV_PAYLOAD"));

    std::filesystem::remove_all(dir);
}

static void Test_Restart_Reader_Reports_Open_Failure()
{
    RestartH5Reader reader;
    REQUIRE_TRUE(!reader.Open("/tmp/sponge_restart_reader_missing.spgr.h5"));
    REQUIRE_TRUE(reader.Last_Error().find("failed to open restart H5 file") !=
                 std::string::npos);
}

int main()
{
    return Run_Test(
        []
        {
            Test_Restart_Reader_Round_Trips_Structural_State();
            Test_Restart_Reader_Allows_No_Velocity_State();
            Test_Restart_Reader_Reports_Unsupported_Dynamic_State_As_Metadata();
            Test_Restart_Reader_Round_Trips_Dynamic_And_Protocol_State();
            Test_Restart_Reader_Reports_Open_Failure();
        });
}
