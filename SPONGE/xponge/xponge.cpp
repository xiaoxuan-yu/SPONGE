#include "xponge.h"

#include "load/amber.hpp"
#include "load/gromacs.hpp"
#include "load/native.hpp"
#include "utils/h5md/topology_h5_reader.hpp"

namespace
{
bool Is_Supported_Topology_Schema_Version(const std::string& version)
{
    return version == "0" || version == "1" ||
           version == "xponge.legacy_to_bundle.v1";
}

void Validate_H5_Topology_Schema_Version(CONTROLLER* controller)
{
    constexpr const char* input_key = "input_h5_topology_path";
    if (!controller->Command_Exist(input_key))
    {
        return;
    }

    SpongeH5MD::TopologyH5Reader reader;
    if (!reader.Open(controller->Command(input_key)))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat,
            "Xponge::Validate_H5_Topology_Schema_Version",
            reader.Last_Error().c_str());
    }
    SpongeH5InputMetadata::TopologyMetadata metadata;
    if (!reader.Read_Metadata(&metadata))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat,
            "Xponge::Validate_H5_Topology_Schema_Version",
            reader.Last_Error().c_str());
    }
    if (!Is_Supported_Topology_Schema_Version(metadata.schema_version))
    {
        const std::string message = std::string("Reason:\n\t") + input_key +
                                    " has unsupported /schema/version '" +
                                    metadata.schema_version +
                                    "'\n\tSupported versions: 0, 1, "
                                    "xponge.legacy_to_bundle.v1\n";
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "Xponge::Validate_H5_Topology_Schema_Version", message.c_str());
    }
}
}  // namespace

void Xponge::System::Load_Inputs(CONTROLLER* controller)
{
    Validate_H5_Topology_Schema_Version(controller);
    if (controller->Command_Exist("gromacs_top") ||
        controller->Command_Exist("gromacs_gro"))
    {
        Load_Gromacs_Inputs(this, controller);
    }
    else if (controller->Command_Exist("amber_parm7") ||
             controller->Command_Exist("amber_rst7"))
    {
        Load_Amber_Inputs(this, controller);
    }
    else
    {
        Load_Native_Inputs(this, controller);
    }
}
