#pragma once

#include <hdf5.h>

#include <cstdint>
#include <highfive/highfive.hpp>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "utils/h5md/h5_input_metadata.hpp"
#include "utils/h5md/h5_structural_state.hpp"
#include "utils/h5md/h5md_writer.hpp"

namespace SpongeH5MD
{
class RestartH5Reader
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
            return Fail(std::string("failed to open restart H5 file: ") +
                        err.what());
        }
    }

    bool Read_Metadata(SpongeH5InputMetadata::RestartMetadata* metadata)
    {
        if (metadata == nullptr)
        {
            return Fail("restart metadata output pointer is null");
        }
        if (!Ensure_File()) return false;

        SpongeH5InputMetadata::RestartMetadata result;
        try
        {
            if (Exists(path::sponge_schema_version))
            {
                result.schema_version =
                    Read_String(path::sponge_schema_version);
            }

            result.has_structural_state = Has_Structural_State();
            result.has_velocity = Exists(path::velocity_value);
            if (Exists(path::position_value))
            {
                const auto dims = Dimensions(path::position_value);
                if (dims.size() == 3)
                {
                    result.atom_count = static_cast<std::int64_t>(dims[1]);
                }
            }
            result.has_dynamic_state =
                Group_Has_Dataset(path::restart_rng_state) ||
                Group_Has_Dataset(path::restart_integrator_state) ||
                Group_Has_Dataset(path::restart_thermostat) ||
                Group_Has_Dataset(path::restart_barostat);
            result.has_protocol_state =
                Group_Has_Dataset(path::restart_bias) ||
                Group_Has_Dataset(path::restart_protocol_sidecars);

            *metadata = result;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to read restart metadata: ") +
                        err.what());
        }
    }

    bool Read_Structural_State(RestartStructuralState* state)
    {
        if (state == nullptr)
        {
            return Fail("restart structural state output pointer is null");
        }
        if (!Ensure_File()) return false;

        try
        {
            const auto position_dims = Require_Dimensions(
                path::position_value, {1, 0, 3}, "restart position");
            const std::size_t atom_count = position_dims[1];
            if (atom_count == 0)
            {
                return Fail("restart position atom dimension must be positive");
            }

            Require_Dimensions(path::particles_all_step, {1}, "restart step");
            Require_Dimensions(path::particles_all_time, {1}, "restart time");
            Require_Dimensions(path::box_edges_value, {1, 3, 3},
                               "restart box edges");

            RestartStructuralState result;
            result.atom_count = atom_count;
            result.step = Read_Required_Single<std::int64_t>(
                path::particles_all_step, "restart step");
            result.time = Read_Required_Single<double>(path::particles_all_time,
                                                       "restart time");
            result.position_xyz = Read_Required_Vector<float>(
                path::position_value, atom_count * 3, "restart position");

            const auto box = Read_Required_Vector<float>(
                path::box_edges_value, 9, "restart box edges");
            for (std::size_t i = 0; i < result.box_edges.size(); ++i)
            {
                result.box_edges[i] = box[i];
            }

            if (Exists(path::velocity_value))
            {
                Require_Dimensions(path::velocity_value, {1, atom_count, 3},
                                   "restart velocity");
                result.velocity_xyz = Read_Required_Vector<float>(
                    path::velocity_value, atom_count * 3, "restart velocity");
                result.has_velocity = true;
            }

            *state = result;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(
                std::string("failed to read restart structural state: ") +
                err.what());
        }
    }

    bool Read_Dynamic_State(RestartDynamicState* state)
    {
        if (state == nullptr)
        {
            return Fail("restart dynamic state output pointer is null");
        }
        if (!Ensure_File()) return false;

        try
        {
            RestartDynamicState result;
            if (Exists(path::restart_nhc))
            {
                const auto dims =
                    Require_Dimensions(path::restart_nhc, {0, 2},
                                       "restart Nose-Hoover chain state");
                result.has_nose_hoover_chain = true;
                result.nose_hoover_chain_pair_count = dims[0];
                result.nose_hoover_chain_coordinate_velocity_pairs =
                    Read_Required_Vector<float>(
                        path::restart_nhc, dims[0] * dims[1],
                        "restart Nose-Hoover chain state");
            }
            if (Exists(path::restart_rng_state))
            {
                for (const auto& module_name :
                     List_Group_Children(path::restart_rng_state,
                                         HighFive::ObjectType::Dataset))
                {
                    result.rng_state_text[module_name] =
                        Read_String(Restart_Rng_State_Path(module_name));
                }
            }
            if (Exists(path::restart_integrator_state))
            {
                for (const auto& key :
                     List_Group_Children(path::restart_integrator_state,
                                         HighFive::ObjectType::Dataset))
                {
                    result.integrator_state_text[key] =
                        Read_String(Restart_Integrator_State_Path(key));
                }
            }
            if (Exists(path::restart_thermostat))
            {
                for (const auto& module_name : List_Group_Children(
                         path::restart_thermostat, HighFive::ObjectType::Group))
                {
                    Read_Named_Dynamic_State_Group(
                        Restart_Thermostat_State_Root(module_name),
                        &result.thermostat_text_states[module_name],
                        &result.thermostat_float_states[module_name]);
                }
            }
            if (Exists(path::restart_barostat))
            {
                for (const auto& module_name : List_Group_Children(
                         path::restart_barostat, HighFive::ObjectType::Group))
                {
                    Read_Named_Dynamic_State_Group(
                        Restart_Barostat_State_Root(module_name),
                        &result.barostat_text_states[module_name],
                        &result.barostat_float_states[module_name]);
                }
            }

            *state = result;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to read restart dynamic state: ") +
                        err.what());
        }
    }

    bool Read_Protocol_State(RestartProtocolState* state)
    {
        if (state == nullptr)
        {
            return Fail("restart protocol state output pointer is null");
        }
        if (!Ensure_File()) return false;

        try
        {
            RestartProtocolState result;
            if (Exists(path::restart_sits))
            {
                for (const auto& module_name : List_Group_Children(
                         path::restart_sits, HighFive::ObjectType::Group))
                {
                    RestartSitsState module;
                    module.module_name = module_name;
                    const std::string module_path =
                        Restart_Sits_State_Root(module_name);
                    for (const auto& state_name : List_Group_Children(
                             module_path, HighFive::ObjectType::Dataset))
                    {
                        const std::string dataset_path =
                            Restart_Sits_State_Path(module_name, state_name);
                        module.float_states[state_name] = Read_Float_Dataset(
                            dataset_path, "restart SITS state");
                    }
                    result.sits_states.push_back(module);
                }
            }

            if (Exists(path::restart_meta))
            {
                for (const auto& metad_name : List_Group_Children(
                         path::restart_meta, HighFive::ObjectType::Group))
                {
                    RestartMetadynamicsState metadynamics;
                    metadynamics.name = metad_name;
                    const std::string metad_path =
                        Restart_Metad_State_Root(metad_name);
                    for (const auto& component_name : List_Group_Children(
                             metad_path, HighFive::ObjectType::Dataset))
                    {
                        metadynamics.text_states[component_name] =
                            Read_String(Restart_Metad_State_Path(
                                metad_name, component_name));
                    }
                    result.metadynamics_states.push_back(metadynamics);
                }
            }

            if (Exists(path::restart_protocol_sidecars))
            {
                for (const auto& key :
                     List_Group_Children(path::restart_protocol_sidecars,
                                         HighFive::ObjectType::Dataset))
                {
                    RestartProtocolSidecarTextState sidecar;
                    sidecar.key = key;
                    sidecar.text =
                        Read_String(Restart_Protocol_Sidecar_Path(key));
                    result.sidecar_text_states.push_back(sidecar);
                }
            }

            *state = result;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to read restart protocol state: ") +
                        err.what());
        }
    }

    std::string Last_Error() const { return last_error_; }

   private:
    bool Ensure_File()
    {
        if (file_ == nullptr)
        {
            return Fail("restart H5 reader is not open");
        }
        return true;
    }

    bool Exists(const std::string& object_path) const
    {
        return file_ != nullptr && file_->exist(object_path);
    }

    bool Has_Structural_State() const
    {
        return Exists(path::particles_all_step) &&
               Exists(path::particles_all_time) &&
               Exists(path::position_value) && Exists(path::box_edges_value);
    }

    bool Group_Has_Dataset(const std::string& group_path)
    {
        if (!Exists(group_path))
        {
            return false;
        }
        const auto group = file_->getGroup(group_path);
        for (const auto& object_name : group.listObjectNames())
        {
            const std::string child_path = group_path + "/" + object_name;
            const auto type = file_->getObjectType(child_path);
            if (type == HighFive::ObjectType::Dataset)
            {
                return true;
            }
            if (type == HighFive::ObjectType::Group &&
                Group_Has_Dataset(child_path))
            {
                return true;
            }
        }
        return false;
    }

    std::vector<std::string> List_Group_Children(
        const std::string& group_path, HighFive::ObjectType object_type)
    {
        std::vector<std::string> names;
        if (!Exists(group_path))
        {
            return names;
        }
        const auto group = file_->getGroup(group_path);
        for (const auto& object_name : group.listObjectNames())
        {
            const std::string child_path = group_path + "/" + object_name;
            if (file_->getObjectType(child_path) == object_type)
            {
                names.push_back(object_name);
            }
        }
        return names;
    }

    std::string Read_String(const std::string& dataset_path)
    {
        std::string value;
        file_->getDataSet(dataset_path).read(value);
        return value;
    }

    std::vector<std::size_t> Dimensions(const std::string& dataset_path)
    {
        return file_->getDataSet(dataset_path).getSpace().getDimensions();
    }

    std::vector<std::size_t> Require_Dimensions(
        const std::string& dataset_path,
        const std::vector<std::size_t>& expected, const std::string& label)
    {
        if (!Exists(dataset_path))
        {
            throw std::runtime_error(label +
                                     " dataset is missing: " + dataset_path);
        }
        const auto dims = Dimensions(dataset_path);
        if (dims.size() != expected.size())
        {
            std::ostringstream out;
            out << label << " rank mismatch at " << dataset_path
                << ": expected " << expected.size() << ", got " << dims.size();
            throw std::runtime_error(out.str());
        }
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            if (expected[i] != 0 && dims[i] != expected[i])
            {
                std::ostringstream out;
                out << label << " shape mismatch at " << dataset_path
                    << ": dimension " << i << " expected " << expected[i]
                    << ", got " << dims[i];
                throw std::runtime_error(out.str());
            }
        }
        return dims;
    }

    template <typename T>
    std::vector<T> Read_Required_Vector(const std::string& dataset_path,
                                        std::size_t expected_size,
                                        const std::string& label)
    {
        const auto dims = Dimensions(dataset_path);
        const std::vector<std::size_t> offsets(dims.size(), 0);
        std::vector<T> values =
            Read_Required_Selection<T>(dataset_path, offsets, dims, label);
        if (values.size() != expected_size)
        {
            std::ostringstream out;
            out << label << " value count mismatch at " << dataset_path
                << ": expected " << expected_size << ", got " << values.size();
            throw std::runtime_error(out.str());
        }
        return values;
    }

    std::vector<float> Read_Float_Dataset(const std::string& dataset_path,
                                          const std::string& label)
    {
        if (!Exists(dataset_path))
        {
            throw std::runtime_error(label +
                                     " dataset is missing: " + dataset_path);
        }
        const auto dims = Dimensions(dataset_path);
        const std::vector<std::size_t> offsets(dims.size(), 0);
        return Read_Required_Selection<float>(dataset_path, offsets, dims,
                                              label);
    }

    void Read_Named_Dynamic_State_Group(
        const std::string& group_path,
        std::map<std::string, std::string>* text_states,
        std::map<std::string, std::vector<float>>* float_states)
    {
        for (const auto& state_name :
             List_Group_Children(group_path, HighFive::ObjectType::Dataset))
        {
            const std::string dataset_path = group_path + "/" + state_name;
            if (Dynamic_State_Name_Is_Float(state_name))
            {
                (*float_states)[state_name] =
                    Read_Float_Dataset(dataset_path, "restart dynamic state");
            }
            else
            {
                (*text_states)[state_name] = Read_String(dataset_path);
            }
        }
    }

    static bool Dynamic_State_Name_Is_Float(const std::string& state_name)
    {
        return state_name == "g" || state_name == "lambda" ||
               state_name == "delta_box_length_max" ||
               state_name == "total_count" || state_name == "accept_count" ||
               state_name == "accept_rate";
    }

    template <typename T>
    std::vector<T> Read_Required_Selection(
        const std::string& dataset_path,
        const std::vector<std::size_t>& offsets,
        const std::vector<std::size_t>& counts, const std::string& label)
    {
        if (offsets.size() != counts.size())
        {
            throw std::runtime_error(label + " selection rank mismatch at " +
                                     dataset_path);
        }
        const std::size_t value_count = Product(counts);
        std::vector<T> values(value_count);
        HighFive::DataSet dataset = file_->getDataSet(dataset_path);
        const std::vector<hsize_t> h_offsets = To_HSize(offsets);
        const std::vector<hsize_t> h_counts = To_HSize(counts);
        hid_t file_space = H5Dget_space(dataset.getId());
        if (file_space < 0)
        {
            throw std::runtime_error(label + " failed to get dataspace at " +
                                     dataset_path);
        }
        const herr_t select_rc =
            H5Sselect_hyperslab(file_space, H5S_SELECT_SET, h_offsets.data(),
                                nullptr, h_counts.data(), nullptr);
        if (select_rc < 0)
        {
            H5Sclose(file_space);
            throw std::runtime_error(label + " failed to select hyperslab at " +
                                     dataset_path);
        }
        hid_t mem_space = H5Screate_simple(static_cast<int>(h_counts.size()),
                                           h_counts.data(), nullptr);
        if (mem_space < 0)
        {
            H5Sclose(file_space);
            throw std::runtime_error(label +
                                     " failed to create memory dataspace at " +
                                     dataset_path);
        }
        const herr_t read_rc =
            H5Dread(dataset.getId(), Native_H5_Type<T>(), mem_space, file_space,
                    H5P_DEFAULT, values.data());
        H5Sclose(mem_space);
        H5Sclose(file_space);
        if (read_rc < 0)
        {
            throw std::runtime_error(label + " failed to read hyperslab at " +
                                     dataset_path);
        }
        return values;
    }

    template <typename T>
    T Read_Required_Single(const std::string& dataset_path,
                           const std::string& label)
    {
        const auto values = Read_Required_Vector<T>(dataset_path, 1, label);
        return values[0];
    }

    static std::vector<hsize_t> To_HSize(const std::vector<std::size_t>& values)
    {
        std::vector<hsize_t> converted;
        converted.reserve(values.size());
        for (const std::size_t value : values)
        {
            converted.push_back(static_cast<hsize_t>(value));
        }
        return converted;
    }

    static std::size_t Product(const std::vector<std::size_t>& values)
    {
        return std::accumulate(
            values.begin(), values.end(), static_cast<std::size_t>(1),
            [](std::size_t lhs, std::size_t rhs) { return lhs * rhs; });
    }

    template <typename T>
    static hid_t Native_H5_Type()
    {
        if constexpr (std::is_same<T, float>::value)
        {
            return H5T_NATIVE_FLOAT;
        }
        else if constexpr (std::is_same<T, double>::value)
        {
            return H5T_NATIVE_DOUBLE;
        }
        else if constexpr (std::is_same<T, std::int64_t>::value)
        {
            return H5T_NATIVE_INT64;
        }
        else
        {
            static_assert(std::is_same<T, float>::value ||
                              std::is_same<T, double>::value ||
                              std::is_same<T, std::int64_t>::value,
                          "unsupported HDF5 numeric read type");
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
}  // namespace SpongeH5MD
