#include <cmath>
#include <stdexcept>
#include <string>

#include "barostat/pressure_based_barostat.h"
#include "thermostat/Bussi_thermostat.h"

int CONTROLLER::MPI_rank = 0;

static void Require(bool value)
{
    if (!value)
    {
        throw std::runtime_error("requirement failed");
    }
}

static void Test_Bussi_H5_Restart_State_Round_Trips()
{
    BUSSI_THERMOSTAT_INFORMATION source;
    source.is_initialized = 1;
    source.lambda = 0.87f;
    source.e.seed(12345);

    SpongeH5MD::RestartDynamicState state;
    std::string error;
    Require(source.Export_H5_Restart_State(&state, &error));
    Require(state.rng_state_text.count("bussi_thermostat") == 1);
    Require(
        state.thermostat_float_states["bussi_thermostat"]["lambda"].size() ==
        1);

    BUSSI_THERMOSTAT_INFORMATION target;
    target.is_initialized = 1;
    target.lambda = 1.0f;
    target.e.seed(67890);
    Require(target.Apply_H5_Restart_State(state, &error));
    Require(std::fabs(target.lambda - 0.87f) < 1.0e-6f);
    Require(target.e() == source.e());
}

static void Test_Bussi_H5_Restart_Requires_Lambda()
{
    BUSSI_THERMOSTAT_INFORMATION source;
    source.is_initialized = 1;
    source.lambda = 0.87f;
    source.e.seed(12345);

    SpongeH5MD::RestartDynamicState state;
    std::string error;
    Require(source.Export_H5_Restart_State(&state, &error));
    state.thermostat_float_states["bussi_thermostat"].erase("lambda");

    BUSSI_THERMOSTAT_INFORMATION target;
    target.is_initialized = 1;
    Require(!target.Apply_H5_Restart_State(state, &error));
    Require(error.find("lambda") != std::string::npos);
}

static void Test_Pressure_Barostat_H5_Restart_State_Round_Trips()
{
    PRESSURE_BASED_BAROSTAT_INFORMATION source;
    source.is_initialized = 1;
    source.g.a11 = 1.0f;
    source.g.a21 = 2.0f;
    source.g.a22 = 3.0f;
    source.g.a31 = 4.0f;
    source.g.a32 = 5.0f;
    source.g.a33 = 6.0f;
    source.generator.seed(24680);

    SpongeH5MD::RestartDynamicState state;
    std::string error;
    Require(source.Export_H5_Restart_State(&state, &error));
    Require(state.rng_state_text.count("pressure_based_barostat") == 1);
    Require(
        state.barostat_float_states["pressure_based_barostat"]["g"].size() ==
        6);

    PRESSURE_BASED_BAROSTAT_INFORMATION target;
    target.is_initialized = 1;
    target.generator.seed(13579);
    Require(target.Apply_H5_Restart_State(state, &error));
    Require(std::fabs(target.g.a11 - 1.0f) < 1.0e-6f);
    Require(std::fabs(target.g.a21 - 2.0f) < 1.0e-6f);
    Require(std::fabs(target.g.a22 - 3.0f) < 1.0e-6f);
    Require(std::fabs(target.g.a31 - 4.0f) < 1.0e-6f);
    Require(std::fabs(target.g.a32 - 5.0f) < 1.0e-6f);
    Require(std::fabs(target.g.a33 - 6.0f) < 1.0e-6f);
    Require(target.generator() == source.generator());
}

static void Test_Pressure_Barostat_H5_Restart_Requires_Rng()
{
    PRESSURE_BASED_BAROSTAT_INFORMATION source;
    source.is_initialized = true;
    source.g = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    source.generator.seed(24680);

    SpongeH5MD::RestartDynamicState state;
    std::string error;
    Require(source.Export_H5_Restart_State(&state, &error));
    state.rng_state_text.erase("pressure_based_barostat");

    PRESSURE_BASED_BAROSTAT_INFORMATION target;
    target.is_initialized = true;
    Require(!target.Apply_H5_Restart_State(state, &error));
    Require(error.find("RNG state") != std::string::npos);
}

int main()
{
    Test_Bussi_H5_Restart_State_Round_Trips();
    Test_Bussi_H5_Restart_Requires_Lambda();
    Test_Pressure_Barostat_H5_Restart_State_Round_Trips();
    Test_Pressure_Barostat_H5_Restart_Requires_Rng();
    return 0;
}
