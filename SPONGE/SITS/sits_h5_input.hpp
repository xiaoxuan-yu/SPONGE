#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <highfive/highfive.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../control.h"

namespace SpongeH5MD
{
namespace SitsH5InputDetail
{
inline std::vector<std::string> Read_String_Vector(HighFive::File* file,
                                                   const std::string& path)
{
    HighFive::DataSet dataset = file->getDataSet(path);
    const auto dimensions = dataset.getSpace().getDimensions();
    if (dimensions.size() != 1)
    {
        throw std::runtime_error(path + " must be one-dimensional");
    }
    std::vector<std::string> values;
    dataset.read(values);
    return values;
}

inline std::vector<std::int64_t> Read_Int64_Vector(HighFive::File* file,
                                                   const std::string& path)
{
    HighFive::DataSet dataset = file->getDataSet(path);
    const auto dimensions = dataset.getSpace().getDimensions();
    if (dimensions.size() != 1)
    {
        throw std::runtime_error(path + " must be one-dimensional");
    }
    std::vector<std::int64_t> values;
    dataset.read(values);
    return values;
}

inline std::vector<int> Read_Int_Vector(HighFive::File* file,
                                        const std::string& path)
{
    HighFive::DataSet dataset = file->getDataSet(path);
    const auto dimensions = dataset.getSpace().getDimensions();
    if (dimensions.size() != 1)
    {
        throw std::runtime_error(path + " must be one-dimensional");
    }
    std::vector<int> values;
    dataset.read(values);
    return values;
}

inline void Validate_Config_Token(const std::string& value,
                                  const std::string& path, bool is_key)
{
    if (value.empty() || value.find_first_of("\r\n{}") != std::string::npos ||
        (is_key && value.find('=') != std::string::npos))
    {
        throw std::runtime_error(path + " contains an invalid config token");
    }
}

inline bool Has_Explicit_Module_Config(CONTROLLER* controller,
                                       const std::string& module_name)
{
    const std::string prefix = module_name + "_";
    for (const auto& command : controller->commands)
    {
        if (command.first.compare(0, prefix.size(), prefix) == 0)
        {
            return true;
        }
    }
    return false;
}

inline std::filesystem::path Materialization_Path(const char* file_name)
{
    return std::filesystem::absolute(
               std::filesystem::path(".sponge_h5_native_protocol") / file_name)
        .lexically_normal();
}

inline void Materialize_Config(CONTROLLER* controller, HighFive::File* file,
                               const std::string& module_name)
{
    constexpr const char* root = "/sits/config";
    if (!file->exist(root) ||
        Has_Explicit_Module_Config(controller, module_name))
    {
        return;
    }

    long long declared_section_count = 0;
    file->getDataSet(std::string(root) + "/section/count")
        .read(declared_section_count);
    const auto section_names =
        Read_String_Vector(file, std::string(root) + "/section/name");
    const auto key_offsets =
        Read_Int64_Vector(file, std::string(root) + "/section/key_offset");
    const auto keys = Read_String_Vector(file, std::string(root) + "/key");
    const auto values = Read_String_Vector(file, std::string(root) + "/value");

    if (declared_section_count != 1 || section_names.size() != 1 ||
        section_names.front() != module_name)
    {
        throw std::runtime_error(
            "/sits/config must contain exactly the active SITS section");
    }
    if (key_offsets.size() != 2 || key_offsets.front() != 0 ||
        key_offsets.back() < 0 ||
        static_cast<std::size_t>(key_offsets.back()) != keys.size() ||
        keys.size() != values.size())
    {
        throw std::runtime_error(
            "/sits/config section offsets and key/value lengths are "
            "inconsistent");
    }
    Validate_Config_Token(section_names.front(), "/sits/config/section/name",
                          true);
    std::set<std::string> unique_keys;
    for (std::size_t item = 0; item < keys.size(); ++item)
    {
        Validate_Config_Token(keys[item], "/sits/config/key", true);
        Validate_Config_Token(values[item], "/sits/config/value", false);
        if (!unique_keys.insert(keys[item]).second)
        {
            throw std::runtime_error(
                "/sits/config/key contains a duplicate key");
        }
    }
    // A legacy SITS section without mode is a valid inactive configuration.
    // Materialize its remaining keys so the runtime preserves that state.
    if (file->exist("/sits/atom_indices") &&
        (unique_keys.count("atom_in_file") ||
         unique_keys.count("atom_numbers")))
    {
        throw std::runtime_error(
            "/sits/config atom ownership conflicts with /sits/atom_indices");
    }

    const auto output_path = Materialization_Path("sits.txt");
    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream out(output_path);
    if (!out)
    {
        throw std::runtime_error("failed to create materialized SITS config " +
                                 output_path.string());
    }
    out << module_name << "\n{\n";
    for (std::size_t item = 0; item < keys.size(); ++item)
    {
        out << "    " << keys[item] << " = " << values[item] << '\n';
    }
    out << "}\n";
    out.close();
    if (!out)
    {
        throw std::runtime_error("failed to write materialized SITS config " +
                                 output_path.string());
    }

    for (std::size_t item = 0; item < keys.size(); ++item)
    {
        controller->Set_Command(keys[item].c_str(), values[item].c_str(), 0,
                                module_name.c_str());
    }
}

inline void Materialize_Atom_Indices(CONTROLLER* controller,
                                     HighFive::File* file,
                                     const std::string& module_name,
                                     int atom_numbers)
{
    constexpr const char* path = "/sits/atom_indices";
    if (!file->exist(path) ||
        controller->Command_Exist(module_name.c_str(), "atom_in_file") ||
        controller->Command_Exist(module_name.c_str(), "atom_numbers"))
    {
        return;
    }

    const auto atom_indices = Read_Int_Vector(file, path);
    if (atom_indices.empty())
    {
        throw std::runtime_error("/sits/atom_indices must not be empty");
    }
    std::set<int> unique_indices;
    for (std::size_t row = 0; row < atom_indices.size(); ++row)
    {
        const int atom = atom_indices[row];
        if (atom < 0 || atom >= atom_numbers)
        {
            std::ostringstream out;
            out << path << " row " << row << " has out-of-range atom " << atom;
            throw std::runtime_error(out.str());
        }
        if (!unique_indices.insert(atom).second)
        {
            std::ostringstream out;
            out << path << " row " << row << " duplicates atom " << atom;
            throw std::runtime_error(out.str());
        }
    }

    const auto output_path = Materialization_Path("sits_atom.txt");
    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream out(output_path);
    if (!out)
    {
        throw std::runtime_error(
            "failed to create materialized SITS atom list " +
            output_path.string());
    }
    for (int atom : atom_indices)
    {
        out << atom << '\n';
    }
    out.close();
    if (!out)
    {
        throw std::runtime_error(
            "failed to write materialized SITS atom list " +
            output_path.string());
    }
    controller->Set_Command("atom_in_file", output_path.string().c_str(), 0,
                            module_name.c_str());
}
}  // namespace SitsH5InputDetail

inline void Materialize_H5_SITS_Input(CONTROLLER* controller,
                                      const char* module_name, int atom_numbers)
{
    constexpr const char* input_key = "input_h5_protocol_path";
    if (!controller->Command_Exist(input_key))
    {
        return;
    }

    try
    {
        HighFive::File file(controller->Command(input_key),
                            HighFive::File::ReadOnly);
        if (!file.exist("/sits"))
        {
            return;
        }
        SitsH5InputDetail::Materialize_Config(controller, &file, module_name);
        SitsH5InputDetail::Materialize_Atom_Indices(controller, &file,
                                                    module_name, atom_numbers);
    }
    catch (const std::exception& error)
    {
        const std::string message =
            std::string("Reason:\n\tfailed to materialize typed SITS input: ") +
            error.what() + "\n";
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "Materialize_H5_SITS_Input",
                                       message.c_str());
    }
}
}  // namespace SpongeH5MD
