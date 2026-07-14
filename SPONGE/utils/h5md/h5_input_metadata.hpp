#pragma once

#include <cstdint>
#include <string>

namespace SpongeH5InputMetadata
{
struct CompatibilityResult
{
    bool compatible = true;
    std::string error_message;
};

struct TopologyMetadata
{
    std::string schema_version;
    std::int64_t atom_count = 0;
    std::string atom_ordering_hash;
    std::string topology_hash;
    std::string force_field_hash;
};

struct ProtocolMetadata
{
    std::string schema_version;
    std::string topology_hash;
    std::string protocol_hash;
    bool has_protocol_owned_state = false;
};

struct RestartMetadata
{
    std::string schema_version;
    std::int64_t atom_count = 0;
    std::string atom_ordering_hash;
    std::string producer_topology_hash;
    std::string producer_protocol_hash;
    bool has_structural_state = false;
    bool has_velocity = false;
    bool has_dynamic_state = false;
    bool has_protocol_state = false;
};

struct TrajectoryMetadata
{
    std::string schema_version;
    std::string particle_stream = "all";
    std::int64_t atom_count = 0;
    std::int64_t frame_count = 0;
    std::string atom_ordering_hash;
    bool has_position = false;
    bool has_box = false;
    bool has_velocity = false;
    bool has_force = false;
    bool has_vds_manifest = false;
};

inline CompatibilityResult Compatible() { return {}; }

inline CompatibilityResult Incompatible(const std::string& message)
{
    CompatibilityResult result;
    result.compatible = false;
    result.error_message = message;
    return result;
}

inline bool Both_Set(const std::string& lhs, const std::string& rhs)
{
    return !lhs.empty() && !rhs.empty();
}

inline CompatibilityResult Check_Topology_Metadata(
    const TopologyMetadata& topology)
{
    if (topology.atom_count <= 0)
    {
        return Incompatible("topology atom_count must be positive");
    }
    if (topology.topology_hash.empty())
    {
        return Incompatible("topology_hash is required");
    }
    return Compatible();
}

inline CompatibilityResult Check_Restart_Against_Topology(
    const RestartMetadata& restart, const TopologyMetadata& topology)
{
    const auto topology_check = Check_Topology_Metadata(topology);
    if (!topology_check.compatible)
    {
        return topology_check;
    }
    if (!restart.has_structural_state)
    {
        return Incompatible("restart structural state is required");
    }
    if (restart.atom_count != topology.atom_count)
    {
        return Incompatible("restart atom_count does not match topology");
    }
    if (Both_Set(restart.atom_ordering_hash, topology.atom_ordering_hash) &&
        restart.atom_ordering_hash != topology.atom_ordering_hash)
    {
        return Incompatible(
            "restart atom_ordering_hash does not match topology");
    }
    if (Both_Set(restart.producer_topology_hash, topology.topology_hash) &&
        restart.producer_topology_hash != topology.topology_hash)
    {
        return Incompatible("restart topology hash does not match topology");
    }
    return Compatible();
}

inline CompatibilityResult Check_Trajectory_Against_Topology(
    const TrajectoryMetadata& trajectory, const TopologyMetadata& topology)
{
    const auto topology_check = Check_Topology_Metadata(topology);
    if (!topology_check.compatible)
    {
        return topology_check;
    }
    if (!trajectory.has_position)
    {
        return Incompatible("trajectory position dataset is required");
    }
    if (!trajectory.has_box)
    {
        return Incompatible("trajectory box dataset is required");
    }
    if (trajectory.atom_count != topology.atom_count)
    {
        return Incompatible("trajectory atom_count does not match topology");
    }
    if (trajectory.frame_count <= 0)
    {
        return Incompatible("trajectory frame_count must be positive");
    }
    if (Both_Set(trajectory.atom_ordering_hash, topology.atom_ordering_hash) &&
        trajectory.atom_ordering_hash != topology.atom_ordering_hash)
    {
        return Incompatible(
            "trajectory atom_ordering_hash does not match topology");
    }
    return Compatible();
}

inline CompatibilityResult Check_Protocol_Against_Topology(
    const ProtocolMetadata& protocol, const TopologyMetadata& topology)
{
    const auto topology_check = Check_Topology_Metadata(topology);
    if (!topology_check.compatible)
    {
        return topology_check;
    }
    if (Both_Set(protocol.topology_hash, topology.topology_hash) &&
        protocol.topology_hash != topology.topology_hash)
    {
        return Incompatible("protocol topology hash does not match topology");
    }
    return Compatible();
}

inline CompatibilityResult Check_Protocol_State_Against_Protocol(
    const RestartMetadata& restart, const ProtocolMetadata& protocol)
{
    if (!restart.has_protocol_state)
    {
        return Compatible();
    }
    if (Both_Set(restart.producer_protocol_hash, protocol.protocol_hash) &&
        restart.producer_protocol_hash != protocol.protocol_hash)
    {
        return Incompatible("restart protocol state does not match protocol");
    }
    return Compatible();
}

}  // namespace SpongeH5InputMetadata
