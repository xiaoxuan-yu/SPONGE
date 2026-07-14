#pragma once

#include <string>

#include "utils/control/h5_input_contract.hpp"
#include "utils/h5md/h5_input_metadata.hpp"
#include "utils/h5md/h5_legacy_sidecar_contract.hpp"
#include "utils/h5md/input_plan.hpp"
#include "utils/h5md/protocol_h5_reader.hpp"
#include "utils/h5md/restart_h5_reader.hpp"
#include "utils/h5md/topology_h5_reader.hpp"
#include "utils/h5md/trajectory_h5_reader.hpp"

namespace SpongeH5InputValidation
{
struct ValidationResult
{
    bool valid = true;
    std::string error_message;
};

inline ValidationResult Valid() { return {}; }

inline ValidationResult Invalid(const std::string& message)
{
    ValidationResult result;
    result.valid = false;
    result.error_message = message;
    return result;
}

inline ValidationResult With_Key(const char* key, const std::string& message)
{
    return Invalid(std::string(key == nullptr ? "input_h5" : key) + ": " +
                   message);
}

inline ValidationResult From_Compatibility(
    const SpongeH5InputMetadata::CompatibilityResult& compatibility,
    const char* key)
{
    if (compatibility.compatible)
    {
        return Valid();
    }
    return With_Key(key, compatibility.error_message);
}

inline bool Requests_Dynamic_State(
    const SpongeH5InputContract::RestartLoadPolicy policy)
{
    return policy == SpongeH5InputContract::RestartLoadPolicy::dynamic ||
           policy == SpongeH5InputContract::RestartLoadPolicy::full;
}

inline bool Requests_Protocol_State(
    const SpongeH5InputContract::RestartLoadPolicy policy)
{
    return policy == SpongeH5InputContract::RestartLoadPolicy::protocol ||
           policy == SpongeH5InputContract::RestartLoadPolicy::full;
}

inline bool Has_Supported_Dynamic_State(
    const SpongeH5MD::RestartDynamicState& state)
{
    const bool has_integrator_state =
        state.integrator_state_text.count("mode") != 0 &&
        state.integrator_state_text.count("step") != 0 &&
        state.integrator_state_text.count("time") != 0 &&
        !state.integrator_state_text.at("mode").empty() &&
        !state.integrator_state_text.at("step").empty() &&
        !state.integrator_state_text.at("time").empty();
    const auto bussi_state =
        state.thermostat_float_states.find("bussi_thermostat");
    const bool has_bussi_state =
        state.rng_state_text.count("bussi_thermostat") != 0 &&
        bussi_state != state.thermostat_float_states.end() &&
        bussi_state->second.count("lambda") != 0 &&
        bussi_state->second.at("lambda").size() == 1;
    const auto pressure_state =
        state.barostat_float_states.find("pressure_based_barostat");
    const bool has_pressure_state =
        state.rng_state_text.count("pressure_based_barostat") != 0 &&
        pressure_state != state.barostat_float_states.end() &&
        pressure_state->second.count("g") != 0 &&
        pressure_state->second.at("g").size() == 6;
    return state.has_nose_hoover_chain || has_integrator_state ||
           has_bussi_state || has_pressure_state;
}

inline std::string Unsupported_Dynamic_State_Reason(
    const SpongeH5MD::RestartDynamicState& state)
{
    if (state.rng_state_text.count("middle_langevin") != 0)
    {
        return "middle_langevin: Middle Langevin Philox RNG state cannot be "
               "restored";
    }
    if (state.rng_state_text.count("andersen") != 0)
    {
        return "andersen: Andersen thermostat Philox RNG state cannot be "
               "restored";
    }
    if (state.rng_state_text.count("monte_carlo_barostat") != 0)
    {
        return "monte_carlo_barostat: Monte Carlo barostat C rand state cannot "
               "be restored";
    }
    for (const auto& module : state.rng_state_text)
    {
        if (module.first != "bussi_thermostat" &&
            module.first != "pressure_based_barostat")
        {
            return "unsupported RNG dynamic state module: " + module.first;
        }
    }
    for (const auto& module : state.thermostat_text_states)
    {
        if (module.first != "bussi_thermostat")
        {
            return "unsupported thermostat dynamic state module: " +
                   module.first;
        }
    }
    for (const auto& module : state.thermostat_float_states)
    {
        if (module.first != "bussi_thermostat")
        {
            return "unsupported thermostat dynamic state module: " +
                   module.first;
        }
    }
    for (const auto& module : state.barostat_text_states)
    {
        if (module.first != "pressure_based_barostat" &&
            module.first != "monte_carlo_barostat")
        {
            return "unsupported barostat dynamic state module: " + module.first;
        }
        if (module.first == "monte_carlo_barostat")
        {
            return "monte_carlo_barostat: Monte Carlo barostat C rand state "
                   "cannot be restored";
        }
    }
    for (const auto& module : state.barostat_float_states)
    {
        if (module.first != "pressure_based_barostat" &&
            module.first != "monte_carlo_barostat")
        {
            return "unsupported barostat dynamic state module: " + module.first;
        }
        if (module.first == "monte_carlo_barostat")
        {
            return "monte_carlo_barostat: Monte Carlo barostat C rand state "
                   "cannot be restored";
        }
    }
    return "";
}

inline bool Has_Unsupported_Dynamic_State(
    const SpongeH5MD::RestartDynamicState& state)
{
    return !Unsupported_Dynamic_State_Reason(state).empty();
}

inline bool Has_Supported_Metadynamics_Text_State(
    const SpongeH5MD::RestartMetadynamicsState& state)
{
    if (state.name != "meta")
    {
        return false;
    }
    return state.text_states.count("hills") != 0 ||
           state.text_states.count("history") != 0 ||
           state.text_states.count("edge") != 0 ||
           state.text_states.count("potential_export") != 0 ||
           state.text_states.count("direct_export") != 0;
}

inline bool Has_Supported_Protocol_State(
    const SpongeH5MD::RestartProtocolState& state)
{
    for (const auto& sits_state : state.sits_states)
    {
        if (sits_state.float_states.count("nk") != 0)
        {
            return true;
        }
    }
    for (const auto& metadynamics_state : state.metadynamics_states)
    {
        if (Has_Supported_Metadynamics_Text_State(metadynamics_state))
        {
            return true;
        }
    }
    for (const auto& sidecar_state : state.sidecar_text_states)
    {
        if (SpongeH5MD::Command_Key_Allowed(
                SpongeH5MD::H5_Protocol_Sidecar_Command_Keys(),
                sidecar_state.key))
        {
            return true;
        }
    }
    return false;
}

inline ValidationResult Validate_Resolved_Input_Plan(
    const SpongeH5InputPlan::ResolvedInputPlan& plan)
{
    if (!plan.valid)
    {
        return Invalid(plan.error_message);
    }
    if (!plan.any_h5_input_enabled)
    {
        return Valid();
    }

    SpongeH5MD::TopologyH5Reader topology_reader;
    if (!topology_reader.Open(plan.topology.path))
    {
        return With_Key(SpongeH5InputContract::kTopologyPathKey,
                        topology_reader.Last_Error());
    }
    SpongeH5InputMetadata::TopologyMetadata topology;
    if (!topology_reader.Read_Metadata(&topology))
    {
        return With_Key(SpongeH5InputContract::kTopologyPathKey,
                        topology_reader.Last_Error());
    }
    auto validation = From_Compatibility(
        SpongeH5InputMetadata::Check_Topology_Metadata(topology),
        SpongeH5InputContract::kTopologyPathKey);
    if (!validation.valid)
    {
        return validation;
    }

    SpongeH5MD::ProtocolH5Reader protocol_reader;
    if (!protocol_reader.Open(plan.protocol.path))
    {
        return With_Key(SpongeH5InputContract::kProtocolPathKey,
                        protocol_reader.Last_Error());
    }
    SpongeH5InputMetadata::ProtocolMetadata protocol;
    if (!protocol_reader.Read_Metadata(&protocol))
    {
        return With_Key(SpongeH5InputContract::kProtocolPathKey,
                        protocol_reader.Last_Error());
    }
    validation = From_Compatibility(
        SpongeH5InputMetadata::Check_Protocol_Against_Topology(protocol,
                                                               topology),
        SpongeH5InputContract::kProtocolPathKey);
    if (!validation.valid)
    {
        return validation;
    }

    if (plan.restart.binding.enabled)
    {
        SpongeH5MD::RestartH5Reader restart_reader;
        if (!restart_reader.Open(plan.restart.binding.path))
        {
            return With_Key(SpongeH5InputContract::kRestartPathKey,
                            restart_reader.Last_Error());
        }
        SpongeH5InputMetadata::RestartMetadata restart;
        if (!restart_reader.Read_Metadata(&restart))
        {
            return With_Key(SpongeH5InputContract::kRestartPathKey,
                            restart_reader.Last_Error());
        }
        validation = From_Compatibility(
            SpongeH5InputMetadata::Check_Restart_Against_Topology(restart,
                                                                  topology),
            SpongeH5InputContract::kRestartPathKey);
        if (!validation.valid)
        {
            return validation;
        }
        if (Requests_Protocol_State(plan.restart.load_policy))
        {
            validation = From_Compatibility(
                SpongeH5InputMetadata::Check_Protocol_State_Against_Protocol(
                    restart, protocol),
                SpongeH5InputContract::kRestartPathKey);
            if (!validation.valid)
            {
                return validation;
            }
        }
        if (Requests_Dynamic_State(plan.restart.load_policy) &&
            !restart.has_dynamic_state)
        {
            return With_Key(
                SpongeH5InputContract::kRestartLoadKey,
                "requested dynamic state is absent from restart.spgr.h5");
        }
        if (Requests_Dynamic_State(plan.restart.load_policy))
        {
            SpongeH5MD::RestartDynamicState dynamic_state;
            if (!restart_reader.Read_Dynamic_State(&dynamic_state))
            {
                return With_Key(SpongeH5InputContract::kRestartPathKey,
                                restart_reader.Last_Error());
            }
            const std::string unsupported_reason =
                Unsupported_Dynamic_State_Reason(dynamic_state);
            if (!unsupported_reason.empty())
            {
                return With_Key(
                    SpongeH5InputContract::kRestartLoadKey,
                    "requested dynamic state contains unsupported payloads: " +
                        unsupported_reason);
            }
            if (!Has_Supported_Dynamic_State(dynamic_state))
            {
                return With_Key(
                    SpongeH5InputContract::kRestartLoadKey,
                    "requested dynamic state has no currently supported "
                    "payload; supported payloads are integrator metadata, "
                    "Bussi thermostat RNG, pressure-based barostat state, "
                    "and Nose-Hoover chain state");
            }
        }
        if (Requests_Protocol_State(plan.restart.load_policy) &&
            !restart.has_protocol_state)
        {
            return With_Key(
                SpongeH5InputContract::kRestartLoadKey,
                "requested protocol state is absent from restart.spgr.h5");
        }
        if (Requests_Protocol_State(plan.restart.load_policy))
        {
            SpongeH5MD::RestartProtocolState protocol_state;
            if (!restart_reader.Read_Protocol_State(&protocol_state))
            {
                return With_Key(SpongeH5InputContract::kRestartPathKey,
                                restart_reader.Last_Error());
            }
            if (!Has_Supported_Protocol_State(protocol_state))
            {
                return With_Key(
                    SpongeH5InputContract::kRestartLoadKey,
                    "requested protocol state has no currently supported "
                    "payload; supported payloads are SITS nk and meta text "
                    "state");
            }
        }
    }

    if (plan.trajectory.binding.enabled)
    {
        SpongeH5MD::TrajectoryH5Reader trajectory_reader;
        if (!trajectory_reader.Open(plan.trajectory.binding.path,
                                    plan.trajectory.particle_stream))
        {
            return With_Key(SpongeH5InputContract::kTrajectoryPathKey,
                            trajectory_reader.Last_Error());
        }
        SpongeH5InputMetadata::TrajectoryMetadata trajectory;
        if (!trajectory_reader.Read_Metadata(&trajectory))
        {
            return With_Key(SpongeH5InputContract::kTrajectoryPathKey,
                            trajectory_reader.Last_Error());
        }
        trajectory.particle_stream = plan.trajectory.particle_stream;
        validation = From_Compatibility(
            SpongeH5InputMetadata::Check_Trajectory_Against_Topology(trajectory,
                                                                     topology),
            SpongeH5InputContract::kTrajectoryPathKey);
        if (!validation.valid)
        {
            return validation;
        }
    }

    return Valid();
}

template <typename ControllerType>
inline ValidationResult Validate_Input_Bindings(ControllerType* controller)
{
    return Validate_Resolved_Input_Plan(
        SpongeH5InputPlan::Resolve_Input_Plan(controller));
}
}  // namespace SpongeH5InputValidation
