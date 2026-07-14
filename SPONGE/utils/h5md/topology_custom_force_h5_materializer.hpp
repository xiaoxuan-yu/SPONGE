#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <highfive/highfive.hpp>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace SpongeH5MD
{
class TopologyCustomForceH5Materializer
{
   public:
    bool Open(const std::string& file_path)
    {
        last_error_.clear();
        try
        {
            file_.reset(
                new HighFive::File(file_path, HighFive::File::ReadOnly));
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to open topology H5 file: ") +
                        err.what());
        }
    }

    bool Has_Pairwise() const
    {
        return Exists("/forcefield/custom_force/pairwise/name") &&
               Exists("/forcefield/custom_force/pairwise/data");
    }

    bool Has_Listed() const
    {
        return Exists("/forcefield/custom_force/listed/name") &&
               Exists("/forcefield/custom_force/listed/data");
    }

    bool Materialize_Pairwise(const std::filesystem::path& descriptor_path,
                              const std::filesystem::path& data_path)
    {
        if (!Ensure_File()) return false;
        try
        {
            const std::string root = "/forcefield/custom_force/pairwise";
            const auto name = Read_String(root + "/name");
            const auto potential = Read_String(root + "/potential");
            const auto parameter_types =
                Read_String_Vector(root + "/parameters/type");
            const auto parameter_names =
                Read_String_Vector(root + "/parameters/name");
            const bool with_ele = Read_Bool(root + "/with_ele");
            const std::string data_root = root + "/data/" + name;
            const auto parameter_values =
                Read_Float_Matrix(data_root + "/parameter/value");
            const auto atom_type = Read_Int_Vector(data_root + "/atom_type");
            const auto atom_count = Read_Int_Scalar(data_root + "/atom_count");
            const auto type_count = Read_Int_Scalar(data_root + "/type_count");
            const auto pair_count = Read_Int_Scalar(data_root + "/pair_count");

            if (parameter_types.size() != parameter_names.size())
            {
                return Fail(
                    "custom pairwise parameter type/name count mismatch");
            }
            if (atom_count != static_cast<int>(atom_type.size()))
            {
                return Fail(
                    "custom pairwise atom_count does not match atom_type");
            }
            if (pair_count * static_cast<int>(parameter_types.size()) !=
                static_cast<int>(parameter_values.size()))
            {
                return Fail("custom pairwise parameter matrix size mismatch");
            }

            Write_Pairwise_Descriptor(descriptor_path, name, potential,
                                      parameter_types, parameter_names,
                                      with_ele);
            Write_Pairwise_Data(data_path, atom_count, type_count,
                                parameter_values, atom_type,
                                parameter_types.size());
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to materialize custom pairwise "
                                    "force from native H5 payload: ") +
                        err.what());
        }
    }

    bool Materialize_Listed(const std::filesystem::path& descriptor_path,
                            const std::filesystem::path& data_path)
    {
        if (!Ensure_File()) return false;
        try
        {
            const std::string root = "/forcefield/custom_force/listed";
            const auto names = Read_String_Vector(root + "/name");
            const auto potentials = Read_String_Vector(root + "/potential");
            const auto connected_atoms =
                Read_String_Vector(root + "/connected_atoms");
            const auto constrain_distance =
                Read_String_Vector(root + "/constrain_distance");
            const auto parameter_types =
                Read_String_Vector(root + "/parameters/type");
            const auto parameter_names =
                Read_String_Vector(root + "/parameters/name");
            if (names.empty())
            {
                return Fail("custom listed force name list is empty");
            }
            if (potentials.size() != names.size() ||
                connected_atoms.size() != names.size() ||
                constrain_distance.size() != names.size())
            {
                return Fail("custom listed descriptor array size mismatch");
            }
            if (parameter_types.size() != parameter_names.size())
            {
                return Fail("custom listed parameter type/name count mismatch");
            }

            Write_Listed_Descriptor(descriptor_path, names, potentials,
                                    parameter_types, parameter_names,
                                    connected_atoms, constrain_distance);
            Write_Listed_Data(data_path, root, names, parameter_types.size());
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to materialize custom listed "
                                    "force from native H5 payload: ") +
                        err.what());
        }
    }

    std::string Last_Error() const { return last_error_; }

   private:
    static std::string Number_String(const float value)
    {
        std::ostringstream out;
        out << std::setprecision(9) << value;
        return out.str();
    }

    bool Ensure_File()
    {
        if (file_ == nullptr)
        {
            return Fail("topology H5 custom force materializer is not open");
        }
        return true;
    }

    bool Exists(const std::string& object_path) const
    {
        return file_ != nullptr && file_->exist(object_path);
    }

    std::vector<std::size_t> Dimensions(const std::string& dataset_path)
    {
        if (!Exists(dataset_path))
        {
            throw std::runtime_error("dataset is missing: " + dataset_path);
        }
        return file_->getDataSet(dataset_path).getSpace().getDimensions();
    }

    std::string Read_String(const std::string& dataset_path)
    {
        std::string value;
        file_->getDataSet(dataset_path).read(value);
        return value;
    }

    std::vector<std::string> Read_String_Vector(const std::string& dataset_path)
    {
        std::vector<std::string> values;
        file_->getDataSet(dataset_path).read(values);
        return values;
    }

    int Read_Int_Scalar(const std::string& dataset_path)
    {
        int value = 0;
        file_->getDataSet(dataset_path).read(value);
        return value;
    }

    bool Read_Bool(const std::string& dataset_path)
    {
        signed char value = 0;
        const auto dataset = file_->getDataSet(dataset_path);
        if (H5Dread(dataset.getId(), H5T_NATIVE_SCHAR, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, &value) < 0)
        {
            throw std::runtime_error("failed to read enum bool dataset: " +
                                     dataset_path);
        }
        return value != 0;
    }

    std::vector<int> Read_Int_Vector(const std::string& dataset_path)
    {
        const auto dims = Dimensions(dataset_path);
        std::size_t count = 1;
        for (const auto dim : dims) count *= dim;
        std::vector<int> values(count);
        file_->getDataSet(dataset_path).read(values);
        return values;
    }

    std::vector<float> Read_Float_Matrix(const std::string& dataset_path)
    {
        const auto dims = Dimensions(dataset_path);
        if (dims.size() != 2)
        {
            throw std::runtime_error("dataset must be a matrix: " +
                                     dataset_path);
        }
        std::vector<float> values(dims[0] * dims[1]);
        const auto dataset = file_->getDataSet(dataset_path);
        const hsize_t h_dims[2] = {static_cast<hsize_t>(dims[0]),
                                   static_cast<hsize_t>(dims[1])};
        hid_t mem_space = H5Screate_simple(2, h_dims, nullptr);
        if (mem_space < 0)
        {
            throw std::runtime_error("failed to create memory dataspace for " +
                                     dataset_path);
        }
        const herr_t read_rc =
            H5Dread(dataset.getId(), H5T_NATIVE_FLOAT, mem_space, H5S_ALL,
                    H5P_DEFAULT, values.data());
        H5Sclose(mem_space);
        if (read_rc < 0)
        {
            throw std::runtime_error("failed to read float matrix: " +
                                     dataset_path);
        }
        return values;
    }

    void Write_Pairwise_Descriptor(
        const std::filesystem::path& path, const std::string& name,
        const std::string& potential,
        const std::vector<std::string>& parameter_types,
        const std::vector<std::string>& parameter_names, const bool with_ele)
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path);
        out << "[[[ " << name << " ]]]\n"
            << "[[ potential ]]\n"
            << potential << "\n"
            << "[[ parameters ]]\n";
        for (std::size_t i = 0; i < parameter_names.size(); ++i)
        {
            if (i != 0) out << ", ";
            out << parameter_types[i] << ' ' << parameter_names[i];
        }
        out << "\n[[ with_ele ]]\n"
            << (with_ele ? "true" : "false") << "\n"
            << "[[ end ]]\n";
    }

    void Write_Pairwise_Data(const std::filesystem::path& path,
                             const int atom_count, const int type_count,
                             const std::vector<float>& parameter_values,
                             const std::vector<int>& atom_type,
                             const std::size_t parameter_count)
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path);
        out << atom_count << ' ' << type_count << "\n";
        for (std::size_t row = 0;
             row < parameter_values.size() / parameter_count; ++row)
        {
            for (std::size_t col = 0; col < parameter_count; ++col)
            {
                if (col != 0) out << ' ';
                out << Number_String(
                    parameter_values[row * parameter_count + col]);
            }
            out << "\n";
        }
        for (const int value : atom_type)
        {
            out << value << "\n";
        }
    }

    void Write_Listed_Descriptor(
        const std::filesystem::path& path,
        const std::vector<std::string>& names,
        const std::vector<std::string>& potentials,
        const std::vector<std::string>& parameter_types,
        const std::vector<std::string>& parameter_names,
        const std::vector<std::string>& connected_atoms,
        const std::vector<std::string>& constrain_distance)
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path);
        for (std::size_t force = 0; force < names.size(); ++force)
        {
            out << "[[[ " << names[force] << " ]]]\n"
                << "[[ potential ]]\n"
                << potentials[force] << "\n"
                << "[[ parameters ]]\n";
            for (std::size_t i = 0; i < parameter_names.size(); ++i)
            {
                if (i != 0) out << ", ";
                out << parameter_types[i] << ' ' << parameter_names[i];
            }
            out << "\n";
            if (!connected_atoms[force].empty())
            {
                out << "[[ connected_atoms ]]\n"
                    << connected_atoms[force] << "\n";
            }
            if (!constrain_distance[force].empty())
            {
                out << "[[ constrain_distance ]]\n"
                    << constrain_distance[force] << "\n";
            }
            out << "[[ end ]]\n";
        }
    }

    void Write_Listed_Data(const std::filesystem::path& path,
                           const std::string& root,
                           const std::vector<std::string>& names,
                           const std::size_t parameter_count)
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path);
        for (const auto& name : names)
        {
            const std::string data_root = root + "/data/" + name;
            const int item_count = Read_Int_Scalar(data_root + "/item_count");
            const auto values =
                Read_Float_Matrix(data_root + "/parameter/value");
            if (values.size() !=
                static_cast<std::size_t>(item_count) * parameter_count)
            {
                throw std::runtime_error(
                    "custom listed data size mismatch for " + name);
            }
            out << item_count << "\n";
            for (int item = 0; item < item_count; ++item)
            {
                for (std::size_t param = 0; param < parameter_count; ++param)
                {
                    if (param != 0) out << ' ';
                    out << Number_String(
                        values[item * parameter_count + param]);
                }
                out << "\n";
            }
        }
    }

    bool Fail(const std::string& message)
    {
        last_error_ = message;
        return false;
    }

    std::unique_ptr<HighFive::File> file_;
    std::string last_error_;
};

template <typename ControllerType>
inline bool Materialize_Native_Custom_Force_Text_Inputs_From_H5(
    ControllerType* controller, const std::string& topology_h5_path,
    const std::filesystem::path& output_dir, std::string* error_message)
{
    auto fail = [error_message](const std::string& message)
    {
        if (error_message != nullptr)
        {
            *error_message = message;
        }
        return false;
    };
    if (controller == nullptr)
    {
        return fail("controller pointer is null");
    }

    TopologyCustomForceH5Materializer materializer;
    if (!materializer.Open(topology_h5_path))
    {
        return fail(materializer.Last_Error());
    }

    if (materializer.Has_Pairwise() &&
        controller->commands.count("custom_pair_in_file") == 0)
    {
        const auto descriptor_path =
            std::filesystem::absolute(output_dir / "pairwise_force.txt")
                .lexically_normal();
        const auto data_path =
            std::filesystem::absolute(output_dir / "custom_pair.txt")
                .lexically_normal();
        if (!materializer.Materialize_Pairwise(descriptor_path, data_path))
        {
            return fail(materializer.Last_Error());
        }
        if (!controller->Command_Exist("pairwise_force", "in_file"))
        {
            controller->Set_Command("pairwise_force_in_file",
                                    descriptor_path.string().c_str(), 0);
        }
        controller->Set_Command("custom_pair_in_file",
                                data_path.string().c_str(), 0);
    }

    if (materializer.Has_Listed() &&
        controller->commands.count("custom_bond_in_file") == 0)
    {
        const auto descriptor_path =
            std::filesystem::absolute(output_dir / "listed_forces.txt")
                .lexically_normal();
        const auto data_path =
            std::filesystem::absolute(output_dir / "custom_bond.txt")
                .lexically_normal();
        if (!materializer.Materialize_Listed(descriptor_path, data_path))
        {
            return fail(materializer.Last_Error());
        }
        if (!controller->Command_Exist("listed_forces", "in_file"))
        {
            controller->Set_Command("listed_forces_in_file",
                                    descriptor_path.string().c_str(), 0);
        }
        controller->Set_Command("custom_bond_in_file",
                                data_path.string().c_str(), 0);
    }
    return true;
}
}  // namespace SpongeH5MD
