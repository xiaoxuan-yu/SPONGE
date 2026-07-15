#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "utils/h5md/h5md_writer.hpp"
#include "utils/h5md/output_plan.hpp"

namespace SpongeH5MD
{
class RestartH5Writer
{
   public:
    explicit RestartH5Writer(WriterBackend* backend) : writer_(backend) {}

    bool Open(const SpongeH5OutputPlan::ResolvedOutputPlan& plan,
              const std::string& schema_version = kCanonicalSchemaVersion)
    {
        if (!plan.restart.enabled)
        {
            last_error_ = "RestartH5Writer requires enabled restart plan";
            return false;
        }
        WriterOptions options;
        options.path = plan.restart.path;
        options.schema_name = "sponge.restart.h5";
        options.schema_version = schema_version;
        options.observable_only = false;
        if (!writer_.Open(options))
        {
            last_error_ = writer_.Last_Error();
            return false;
        }
        return Ensure_Base_Layout();
    }

    bool Ensure_Base_Layout()
    {
        if (!writer_.Ensure_Group(path::run)) return false;
        if (!writer_.Ensure_Group(path::particles_all)) return false;
        if (!writer_.Ensure_Group(path::particles_all_position)) return false;
        if (!writer_.Ensure_Group(path::particles_all_velocity)) return false;
        if (!writer_.Ensure_Group(path::particles_all_box)) return false;
        if (!writer_.Ensure_Group(path::particles_all_box_edges)) return false;
        if (!writer_.Ensure_Group(path::parameters_restart)) return false;
        if (!writer_.Ensure_Group(path::restart_rng_state)) return false;
        if (!writer_.Ensure_Group(path::restart_integrator_state)) return false;
        if (!writer_.Ensure_Group(path::restart_thermostat)) return false;
        if (!writer_.Ensure_Group(path::restart_barostat)) return false;
        if (!writer_.Ensure_Group(path::restart_protocol_sidecars))
            return false;
        if (!writer_.Ensure_Group(path::restart_bias)) return false;
        if (!writer_.Ensure_Group(path::restart_sits)) return false;
        if (!writer_.Ensure_Group(path::restart_meta)) return false;
        return true;
    }

    bool Define_Structural_State(const std::size_t atom_count,
                                 bool include_velocity)
    {
        atom_count_ = atom_count;
        if (!writer_.Create_Dataset({path::particles_all_step,
                                     DataType::int64,
                                     {{0}, {1}, {1}},
                                     true}))
        {
            return false;
        }
        if (!writer_.Create_Dataset({path::particles_all_time,
                                     DataType::float64,
                                     {{0}, {1}, {1}},
                                     true}))
        {
            return false;
        }
        if (!writer_.Set_String_Attribute(path::particles_all_time, "unit",
                                          "ps"))
        {
            return false;
        }
        if (!writer_.Create_Dataset(
                {path::position_value,
                 DataType::float32,
                 {{0, atom_count, 3}, {1, atom_count, 3}, {1, atom_count, 3}},
                 true}))
        {
            return false;
        }
        if (!writer_.Set_String_Attribute(path::position_value, "unit",
                                          "Angstrom"))
        {
            return false;
        }
        if (!writer_.Create_Hard_Link(path::particles_all_step,
                                      path::position_step))
        {
            return false;
        }
        if (!writer_.Create_Hard_Link(path::particles_all_time,
                                      path::position_time))
        {
            return false;
        }
        if (!writer_.Create_Dataset({path::box_edges_value,
                                     DataType::float32,
                                     {{0, 3, 3}, {1, 3, 3}, {1, 3, 3}},
                                     true}))
        {
            return false;
        }
        if (!writer_.Set_String_Attribute(path::box_edges_value, "unit",
                                          "Angstrom"))
        {
            return false;
        }
        if (!writer_.Create_Hard_Link(path::particles_all_step,
                                      path::box_edges_step))
        {
            return false;
        }
        if (!writer_.Create_Hard_Link(path::particles_all_time,
                                      path::box_edges_time))
        {
            return false;
        }
        if (include_velocity)
        {
            if (!writer_.Create_Dataset({path::velocity_value,
                                         DataType::float32,
                                         {{0, atom_count, 3},
                                          {1, atom_count, 3},
                                          {1, atom_count, 3}},
                                         true}))
            {
                return false;
            }
            if (!writer_.Set_String_Attribute(path::velocity_value, "unit",
                                              "Angstrom ps-1"))
            {
                return false;
            }
            if (!writer_.Create_Hard_Link(path::particles_all_step,
                                          path::velocity_step))
            {
                return false;
            }
            if (!writer_.Create_Hard_Link(path::particles_all_time,
                                          path::velocity_time))
            {
                return false;
            }
        }
        include_velocity_ = include_velocity;
        return true;
    }

    bool Write_Structural_State(const int64_t step, const double time,
                                const float* position_xyz,
                                const float* box_edges_3x3,
                                const float* velocity_xyz = nullptr)
    {
        if (state_written_)
        {
            last_error_ = "restart H5 already contains one structural state";
            return Mark_Failed();
        }
        if (!writer_.Append_Int64(path::particles_all_step, &step, 1))
        {
            return Mark_Failed();
        }
        if (!writer_.Append_Float64(path::particles_all_time, &time, 1))
        {
            return Mark_Failed();
        }
        if (!writer_.Append_Float32(path::position_value, position_xyz,
                                    atom_count_ * 3))
        {
            return Mark_Failed();
        }
        if (!writer_.Append_Float32(path::box_edges_value, box_edges_3x3, 9))
        {
            return Mark_Failed();
        }
        if (include_velocity_ && velocity_xyz != nullptr)
        {
            if (!writer_.Append_Float32(path::velocity_value, velocity_xyz,
                                        atom_count_ * 3))
            {
                return Mark_Failed();
            }
        }
        state_written_ = true;
        if (!Write_Run_Metadata(step, time) ||
            !writer_.Write_Output_Completion(1, step, time))
        {
            return Mark_Failed();
        }
        return true;
    }

    bool Write_Run_Metadata(const int64_t step, const double time,
                            const std::string& state_type = "restart")
    {
        if (!writer_.Create_Dataset({path::run_current_step,
                                     DataType::int64,
                                     {{0}, {1}, {1}},
                                     true}))
        {
            return false;
        }
        if (!writer_.Create_Dataset({path::run_current_time,
                                     DataType::float64,
                                     {{0}, {1}, {1}},
                                     true}))
        {
            return false;
        }
        if (!writer_.Append_Int64(path::run_current_step, &step, 1))
        {
            return false;
        }
        if (!writer_.Append_Float64(path::run_current_time, &time, 1))
        {
            return false;
        }
        return writer_.Write_String(path::run_state_type, state_type);
    }

    bool Write_Nose_Hoover_Chain_State(const float* coordinate_velocity_pairs,
                                       std::size_t pair_count)
    {
        if (!writer_.Create_Dataset({path::restart_nhc,
                                     DataType::float32,
                                     {{0, 2}, {pair_count, 2}, {pair_count, 2}},
                                     true}))
        {
            return false;
        }
        return writer_.Append_Float32(
            path::restart_nhc, coordinate_velocity_pairs, pair_count * 2);
    }

    bool Write_Rng_State_Text(const std::string& module_name,
                              const std::string& value)
    {
        if (!Validate_State_Component_Name(module_name, "rng state module"))
        {
            return Mark_Failed();
        }
        if (!writer_.Ensure_Group(path::restart_rng_state)) return false;
        return writer_.Write_String(Restart_Rng_State_Path(module_name), value);
    }

    bool Write_Integrator_State_Text(const std::string& key,
                                     const std::string& value)
    {
        if (!Validate_State_Component_Name(key, "integrator state key"))
        {
            return Mark_Failed();
        }
        if (!writer_.Ensure_Group(path::restart_integrator_state)) return false;
        return writer_.Write_String(Restart_Integrator_State_Path(key), value);
    }

    bool Write_Thermostat_State_Text(const std::string& module_name,
                                     const std::string& state_name,
                                     const std::string& value)
    {
        if (!Validate_State_Component_Name(module_name, "thermostat module") ||
            !Validate_State_Component_Name(state_name, "thermostat state"))
        {
            return Mark_Failed();
        }
        if (!writer_.Ensure_Group(Restart_Thermostat_State_Root(module_name)))
        {
            return false;
        }
        return writer_.Write_String(
            Restart_Thermostat_State_Path(module_name, state_name), value);
    }

    bool Write_Thermostat_State_Float(const std::string& module_name,
                                      const std::string& state_name,
                                      const float* values, std::size_t count)
    {
        if (!Validate_State_Component_Name(module_name, "thermostat module") ||
            !Validate_State_Component_Name(state_name, "thermostat state"))
        {
            return Mark_Failed();
        }
        if (!writer_.Ensure_Group(Restart_Thermostat_State_Root(module_name)))
        {
            return false;
        }
        const std::string dataset_path =
            Restart_Thermostat_State_Path(module_name, state_name);
        if (!writer_.Create_Dataset({dataset_path,
                                     DataType::float32,
                                     {{0}, {count}, {count}},
                                     true}))
        {
            return false;
        }
        return writer_.Append_Float32(dataset_path, values, count);
    }

    bool Write_Barostat_State_Text(const std::string& module_name,
                                   const std::string& state_name,
                                   const std::string& value)
    {
        if (!Validate_State_Component_Name(module_name, "barostat module") ||
            !Validate_State_Component_Name(state_name, "barostat state"))
        {
            return Mark_Failed();
        }
        if (!writer_.Ensure_Group(Restart_Barostat_State_Root(module_name)))
        {
            return false;
        }
        return writer_.Write_String(
            Restart_Barostat_State_Path(module_name, state_name), value);
    }

    bool Write_Barostat_State_Float(const std::string& module_name,
                                    const std::string& state_name,
                                    const float* values, std::size_t count)
    {
        if (!Validate_State_Component_Name(module_name, "barostat module") ||
            !Validate_State_Component_Name(state_name, "barostat state"))
        {
            return Mark_Failed();
        }
        if (!writer_.Ensure_Group(Restart_Barostat_State_Root(module_name)))
        {
            return false;
        }
        const std::string dataset_path =
            Restart_Barostat_State_Path(module_name, state_name);
        if (!writer_.Create_Dataset({dataset_path,
                                     DataType::float32,
                                     {{0}, {count}, {count}},
                                     true}))
        {
            return false;
        }
        return writer_.Append_Float32(dataset_path, values, count);
    }

    bool Write_Sits_State(const std::string& module_name,
                          const std::string& state_name, const float* values,
                          std::size_t count)
    {
        const std::string module_path = Restart_Sits_State_Root(module_name);
        const std::string dataset_path =
            Restart_Sits_State_Path(module_name, state_name);
        if (!writer_.Ensure_Group(module_path)) return false;
        if (!writer_.Create_Dataset({dataset_path,
                                     DataType::float32,
                                     {{0}, {count}, {count}},
                                     true}))
        {
            return false;
        }
        return writer_.Append_Float32(dataset_path, values, count);
    }

    bool Write_Metad_State_Text(const std::string& name,
                                const std::string& component,
                                const std::string& value)
    {
        const std::string meta_path = Restart_Metad_State_Root(name);
        if (!writer_.Ensure_Group(meta_path)) return false;
        return writer_.Write_String(Restart_Metad_State_Path(name, component),
                                    value);
    }

    bool Write_Protocol_Sidecar_Text(const std::string& key,
                                     const std::string& value)
    {
        if (key.empty() || key.find('/') != std::string::npos)
        {
            last_error_ =
                "protocol sidecar restart key must be non-empty and must not "
                "contain '/'";
            return Mark_Failed();
        }
        if (!writer_.Ensure_Group(path::restart_protocol_sidecars))
            return false;
        return writer_.Write_String(Restart_Protocol_Sidecar_Path(key), value);
    }

    bool Write_Legacy_Sidecar_Paths(const std::vector<std::string>& keys,
                                    const std::vector<std::string>& paths)
    {
        if (!writer_.Ensure_Group(path::sponge_files)) return false;
        if (!writer_.Ensure_Group(path::legacy_sidecars)) return false;
        return writer_.Write_String_Array(path::legacy_sidecar_keys, keys) &&
               writer_.Write_String_Array(path::legacy_sidecar_paths, paths);
    }

    bool Finalize() { return writer_.Finalize(); }
    bool Flush() { return writer_.Flush(); }
    bool Close() { return writer_.Close(); }

    bool State_Written() const { return state_written_; }

    std::string Last_Error() const
    {
        if (!last_error_.empty()) return last_error_;
        return writer_.Last_Error();
    }

   private:
    bool Validate_State_Component_Name(const std::string& name,
                                       const char* label)
    {
        if (name.empty() || name.find('/') != std::string::npos)
        {
            last_error_ = std::string(label) +
                          " must be non-empty and must not contain '/'";
            return false;
        }
        return true;
    }

    bool Mark_Failed()
    {
        const std::string reason = Last_Error();
        writer_.Mark_Failed(reason);
        return false;
    }

    H5MDWriter writer_;
    std::size_t atom_count_ = 0;
    bool include_velocity_ = false;
    bool state_written_ = false;
    std::string last_error_;
};
}  // namespace SpongeH5MD
