#pragma once

#include <hdf5.h>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <highfive/highfive.hpp>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace SpongeH5MD
{
class NativeRestraintH5Materializer
{
   public:
    bool Open_Protocol(const std::string& file_path)
    {
        last_error_.clear();
        try
        {
            protocol_.reset(
                new HighFive::File(file_path, HighFive::File::ReadOnly));
            return true;
        }
        catch (const std::exception& error)
        {
            return Fail(std::string("failed to open protocol H5 file: ") +
                        error.what());
        }
    }

    bool Open_Restart(const std::string& file_path)
    {
        try
        {
            restart_.reset(
                new HighFive::File(file_path, HighFive::File::ReadOnly));
            return true;
        }
        catch (const std::exception& error)
        {
            return Fail(std::string("failed to open restart H5 file: ") +
                        error.what());
        }
    }

    bool Has_Positional_Restraint() const
    {
        return Protocol_Exists("/restraint/default/atom_indices");
    }

    bool Materialize(const std::filesystem::path& output_dir,
                     std::size_t atom_count, bool* has_weight,
                     bool* has_reference)
    {
        if (has_weight == nullptr || has_reference == nullptr)
        {
            return Fail("native restraint presence output pointer is null");
        }
        *has_weight = false;
        *has_reference = false;
        if (protocol_ == nullptr)
        {
            return Fail(
                "native restraint protocol H5 materializer is not open");
        }
        if (atom_count == 0)
        {
            return Fail("native restraint requires a positive atom count");
        }

        try
        {
            const std::string atom_path = "/restraint/default/atom_indices";
            const auto atom_dimensions = Dimensions(*protocol_, atom_path);
            if (atom_dimensions.size() != 1 || atom_dimensions[0] == 0)
            {
                return Fail(atom_path +
                            " must have shape [n] with n greater than zero");
            }
            const auto atom_indices =
                Read_Flat<int>(*protocol_, atom_path, atom_dimensions,
                               "restraint atom indices");
            for (std::size_t row = 0; row < atom_indices.size(); ++row)
            {
                if (atom_indices[row] < 0 ||
                    static_cast<std::size_t>(atom_indices[row]) >= atom_count)
                {
                    std::ostringstream message;
                    message << atom_path << " contains out-of-range atom "
                            << atom_indices[row] << " at row " << row;
                    return Fail(message.str());
                }
            }

            std::vector<float> weight;
            const std::string weight_path = "/restraint/default/weight";
            if (Protocol_Exists(weight_path))
            {
                weight = Read_Flat<float>(
                    *protocol_, weight_path,
                    {atom_indices.size(), static_cast<std::size_t>(3)},
                    "restraint weight");
                Validate_Finite(weight, weight_path);
                *has_weight = true;
            }

            std::vector<float> reference;
            const std::string reference_path =
                "/parameters/restart/references/restraint/default/coordinate";
            if (Restart_Exists(reference_path))
            {
                reference =
                    Read_Flat<float>(*restart_, reference_path,
                                     {atom_count, static_cast<std::size_t>(3)},
                                     "restraint reference coordinate");
                Validate_Finite(reference, reference_path);
                *has_reference = true;
            }

            std::filesystem::create_directories(output_dir);
            if (!Write_Atom_Indices(output_dir / "restrain_atom_id.txt",
                                    atom_indices))
            {
                return false;
            }
            if (*has_weight &&
                !Write_XYZ(output_dir / "restrain_weight.txt", weight, false))
            {
                return false;
            }
            if (*has_reference &&
                !Write_XYZ(output_dir / "restrain_coordinate.txt", reference,
                           true))
            {
                return false;
            }
            return true;
        }
        catch (const std::exception& error)
        {
            return Fail(
                std::string("failed to materialize native restraint: ") +
                error.what());
        }
    }

    const std::string& Last_Error() const { return last_error_; }

   private:
    bool Protocol_Exists(const std::string& path) const
    {
        return protocol_ != nullptr && protocol_->exist(path);
    }

    bool Restart_Exists(const std::string& path) const
    {
        return restart_ != nullptr && restart_->exist(path);
    }

    static std::vector<std::size_t> Dimensions(HighFive::File& file,
                                               const std::string& path)
    {
        if (!file.exist(path))
        {
            throw std::runtime_error("dataset is missing: " + path);
        }
        return file.getDataSet(path).getSpace().getDimensions();
    }

    template <typename T>
    static std::vector<T> Read_Flat(HighFive::File& file,
                                    const std::string& path,
                                    const std::vector<std::size_t>& expected,
                                    const std::string& label)
    {
        const auto dimensions = Dimensions(file, path);
        if (dimensions != expected)
        {
            std::ostringstream message;
            message << label << " dataset " << path << " must have shape [";
            for (std::size_t index = 0; index < expected.size(); ++index)
            {
                if (index != 0) message << ',';
                message << expected[index];
            }
            message << ']';
            throw std::runtime_error(message.str());
        }
        std::size_t count = 1;
        for (std::size_t value : dimensions) count *= value;
        std::vector<T> values(count);
        HighFive::DataSet dataset = file.getDataSet(path);
        if (H5Dread(dataset.getId(), Native_H5_Type<T>(), H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, values.data()) < 0)
        {
            throw std::runtime_error(label + " failed to read dataset " + path);
        }
        return values;
    }

    bool Write_Atom_Indices(const std::filesystem::path& path,
                            const std::vector<int>& atom_indices)
    {
        std::ofstream output(path);
        if (!output)
        {
            return Fail("failed to create materialized restraint atom IDs: " +
                        path.string());
        }
        for (int value : atom_indices) output << value << '\n';
        output.close();
        if (!output)
        {
            return Fail("failed to write materialized restraint atom IDs: " +
                        path.string());
        }
        return true;
    }

    bool Write_XYZ(const std::filesystem::path& path,
                   const std::vector<float>& values, bool write_count)
    {
        std::ofstream output(path);
        if (!output)
        {
            return Fail("failed to create materialized restraint values: " +
                        path.string());
        }
        output << std::setprecision(9);
        if (write_count) output << values.size() / 3 << '\n';
        for (std::size_t row = 0; row < values.size() / 3; ++row)
        {
            output << values[3 * row] << ' ' << values[3 * row + 1] << ' '
                   << values[3 * row + 2] << '\n';
        }
        output.close();
        if (!output)
        {
            return Fail("failed to write materialized restraint values: " +
                        path.string());
        }
        return true;
    }

    static void Validate_Finite(const std::vector<float>& values,
                                const std::string& path)
    {
        for (float value : values)
        {
            if (!std::isfinite(value))
            {
                throw std::runtime_error(path + " contains a non-finite value");
            }
        }
    }

    template <typename T>
    static hid_t Native_H5_Type();

    bool Fail(const std::string& message)
    {
        last_error_ = message;
        return false;
    }

    std::unique_ptr<HighFive::File> protocol_;
    std::unique_ptr<HighFive::File> restart_;
    std::string last_error_;
};

template <>
inline hid_t NativeRestraintH5Materializer::Native_H5_Type<int>()
{
    return H5T_NATIVE_INT;
}

template <>
inline hid_t NativeRestraintH5Materializer::Native_H5_Type<float>()
{
    return H5T_NATIVE_FLOAT;
}
}  // namespace SpongeH5MD
