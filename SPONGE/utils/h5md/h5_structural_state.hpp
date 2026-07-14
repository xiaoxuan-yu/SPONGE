#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace SpongeH5MD
{
struct RestartStructuralState
{
    std::int64_t step = 0;
    double time = 0.0;
    std::size_t atom_count = 0;
    std::vector<float> position_xyz;
    std::vector<float> velocity_xyz;
    std::array<float, 9> box_edges{};
    bool has_velocity = false;
};

struct RestartDynamicState
{
    bool has_nose_hoover_chain = false;
    std::size_t nose_hoover_chain_pair_count = 0;
    std::vector<float> nose_hoover_chain_coordinate_velocity_pairs;
    std::map<std::string, std::string> rng_state_text;
    std::map<std::string, std::string> integrator_state_text;
    std::map<std::string, std::map<std::string, std::string>>
        thermostat_text_states;
    std::map<std::string, std::map<std::string, std::vector<float>>>
        thermostat_float_states;
    std::map<std::string, std::map<std::string, std::string>>
        barostat_text_states;
    std::map<std::string, std::map<std::string, std::vector<float>>>
        barostat_float_states;
};

struct RestartSitsState
{
    std::string module_name;
    std::map<std::string, std::vector<float>> float_states;
};

struct RestartMetadynamicsState
{
    std::string name;
    std::map<std::string, std::string> text_states;
};

struct RestartProtocolSidecarTextState
{
    std::string key;
    std::string text;
};

struct RestartProtocolState
{
    std::vector<RestartSitsState> sits_states;
    std::vector<RestartMetadynamicsState> metadynamics_states;
    std::vector<RestartProtocolSidecarTextState> sidecar_text_states;
};
}  // namespace SpongeH5MD
