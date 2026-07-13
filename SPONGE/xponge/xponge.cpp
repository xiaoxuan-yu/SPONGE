#include "xponge.h"

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

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

std::vector<int> Read_H5_Residue_Atom_Numbers(CONTROLLER* controller,
                                              std::size_t atom_count)
{
    constexpr const char* input_key = "input_h5_topology_path";
    constexpr const char* dataset_path = "/atoms/residue_index";
    constexpr const char* offset_path = "/residues/atom_offset";
    if (!controller->Command_Exist(input_key))
    {
        return {};
    }

    try
    {
        HighFive::File file(controller->Command(input_key),
                            HighFive::File::ReadOnly);
        if (!file.exist(dataset_path))
        {
            return {};
        }
        if (controller->Command_Exist("residue_in_file"))
        {
            const std::string message =
                std::string("Reason:\n\tnative residue mapping ") +
                dataset_path +
                " conflicts with residue_in_file; select exactly one residue "
                "input owner\n";
            controller->Throw_SPONGE_Error(
                spongeErrorConflictingCommand,
                "Xponge::Read_H5_Residue_Atom_Numbers", message.c_str());
        }

        HighFive::DataSet dataset = file.getDataSet(dataset_path);
        const auto dimensions = dataset.getSpace().getDimensions();
        if (dimensions.size() != 1)
        {
            throw std::runtime_error(
                "/atoms/residue_index must be a one-dimensional dataset");
        }
        if (dimensions[0] != atom_count)
        {
            std::ostringstream message;
            message << "/atoms/residue_index length " << dimensions[0]
                    << " does not match atom count " << atom_count;
            throw std::runtime_error(message.str());
        }

        std::vector<std::int64_t> residue_indices;
        dataset.read(residue_indices);
        if (residue_indices.empty() || residue_indices.front() != 0)
        {
            throw std::runtime_error(
                "/atoms/residue_index must start with residue index 0");
        }

        std::vector<int> residue_atom_numbers;
        residue_atom_numbers.push_back(1);
        for (std::size_t atom = 1; atom < residue_indices.size(); ++atom)
        {
            const std::int64_t previous = residue_indices[atom - 1];
            const std::int64_t current = residue_indices[atom];
            if (current == previous)
            {
                ++residue_atom_numbers.back();
            }
            else if (current == previous + 1)
            {
                residue_atom_numbers.push_back(1);
            }
            else
            {
                std::ostringstream message;
                message << "/atoms/residue_index must contain contiguous, "
                           "nondecreasing labels; atom "
                        << atom << " changes from " << previous << " to "
                        << current;
                throw std::runtime_error(message.str());
            }
        }

        if (file.exist(offset_path))
        {
            HighFive::DataSet offsets_dataset = file.getDataSet(offset_path);
            const auto offset_dimensions =
                offsets_dataset.getSpace().getDimensions();
            if (offset_dimensions.size() != 1)
            {
                throw std::runtime_error(
                    "/residues/atom_offset must be a one-dimensional dataset");
            }
            std::vector<std::int64_t> atom_offsets;
            offsets_dataset.read(atom_offsets);
            std::vector<std::int64_t> expected_offsets = {0};
            expected_offsets.reserve(residue_atom_numbers.size() + 1);
            for (const int residue_atom_number : residue_atom_numbers)
            {
                expected_offsets.push_back(expected_offsets.back() +
                                           residue_atom_number);
            }
            if (atom_offsets != expected_offsets)
            {
                throw std::runtime_error(
                    "/atoms/residue_index and /residues/atom_offset describe "
                    "different residue mappings");
            }
        }
        return residue_atom_numbers;
    }
    catch (const std::exception& error)
    {
        const std::string message =
            std::string("Reason:\n\tfailed to read typed residue membership ") +
            "from " + dataset_path + ": " + error.what() + "\n";
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "Xponge::Read_H5_Residue_Atom_Numbers",
                                       message.c_str());
    }
    return {};
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
        const auto residue_atom_numbers =
            Read_H5_Residue_Atom_Numbers(controller, this->atoms.mass.size());
        if (!residue_atom_numbers.empty())
        {
            this->residues.atom_numbers = residue_atom_numbers;
        }
    }
}
