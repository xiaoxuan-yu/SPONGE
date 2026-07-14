#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "utils/h5md/h5md_writer.hpp"
#include "utils/h5md/module_h5_mappings.hpp"
#include "utils/h5md/output_plan.hpp"

namespace SpongeH5MD
{
class TrajectoryH5Writer
{
   public:
    explicit TrajectoryH5Writer(WriterBackend* backend) : writer_(backend) {}

    bool Open_Single_File(const SpongeH5OutputPlan::ResolvedOutputPlan& plan,
                          const std::string& schema_version = "0")
    {
        if (!plan.trajectory.enabled || plan.trajectory.vds)
        {
            last_error_ =
                "TrajectoryH5Writer requires enabled non-VDS trajectory plan";
            return false;
        }
        WriterOptions options;
        options.path = plan.trajectory.path;
        options.schema_name = "sponge.output.h5md";
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
        if (!writer_.Ensure_Group(path::particles_all)) return false;
        if (!writer_.Ensure_Group(path::particles_all_position)) return false;
        if (!writer_.Ensure_Group(path::particles_all_velocity)) return false;
        if (!writer_.Ensure_Group(path::particles_all_force)) return false;
        if (!writer_.Ensure_Group(path::particles_all_box)) return false;
        if (!writer_.Ensure_Group(path::particles_all_box_edges)) return false;
        if (!writer_.Ensure_Group(path::observables_all)) return false;
        if (!writer_.Ensure_Group(path::sponge_mdout)) return false;
        if (!writer_.Ensure_Group(path::mdout_columns)) return false;
        if (!writer_.Ensure_Group(path::sponge_log)) return false;
        return true;
    }

    bool Define_Particle_Datasets(const std::size_t atom_count,
                                  bool include_velocity, bool include_force)
    {
        atom_count_ = atom_count;
        if (!Create_Frame_Index_Dataset(path::particles_all_step,
                                        DataType::int64))
        {
            return false;
        }
        if (!Create_Frame_Index_Dataset(path::particles_all_time,
                                        DataType::float64))
        {
            return false;
        }
        if (!Create_Vector3_Frame_Dataset(path::position_value, atom_count))
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
        if (!Create_Box_Frame_Dataset(path::box_edges_value))
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
            if (!Create_Vector3_Frame_Dataset(path::velocity_value, atom_count))
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
        if (include_force)
        {
            if (!Create_Vector3_Frame_Dataset(path::force_value, atom_count))
            {
                return false;
            }
            if (!writer_.Create_Hard_Link(path::particles_all_step,
                                          path::force_step))
            {
                return false;
            }
            if (!writer_.Create_Hard_Link(path::particles_all_time,
                                          path::force_time))
            {
                return false;
            }
        }
        return true;
    }

    bool Define_Observable_Stream(
        const std::vector<std::string>& hdf5_names,
        const std::vector<std::string>& original_names)
    {
        observable_names_ = hdf5_names;
        if (!writer_.Create_Dataset({path::observables_all_step,
                                     DataType::int64,
                                     {{0}, {0}, {0}},
                                     true}))
        {
            return false;
        }
        if (!writer_.Create_Dataset({path::observables_all_time,
                                     DataType::float64,
                                     {{0}, {0}, {0}},
                                     true}))
        {
            return false;
        }
        for (const std::string& name : hdf5_names)
        {
            const std::string group = Observable_Root(name);
            if (!writer_.Ensure_Group(group)) return false;
            if (!writer_.Create_Dataset({Observable_Value_Path(name),
                                         DataType::float64,
                                         {{0}, {0}, {0}},
                                         true}))
            {
                return false;
            }
            if (!writer_.Create_Hard_Link(path::observables_all_step,
                                          Observable_Step_Path(name)))
            {
                return false;
            }
            if (!writer_.Create_Hard_Link(path::observables_all_time,
                                          Observable_Time_Path(name)))
            {
                return false;
            }
        }
        if (!writer_.Write_String_Array(path::mdout_columns_original_name,
                                        original_names))
        {
            return false;
        }
        if (!writer_.Write_String_Array(path::mdout_columns_hdf5_name,
                                        hdf5_names))
        {
            return false;
        }
        return true;
    }

    bool Append_Particle_Frame(const int64_t step, const double time,
                               const float* position_xyz,
                               const float* box_edges_3x3,
                               const float* velocity_xyz = nullptr,
                               const float* force_xyz = nullptr)
    {
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
        if (velocity_xyz != nullptr &&
            !writer_.Append_Float32(path::velocity_value, velocity_xyz,
                                    atom_count_ * 3))
        {
            return Mark_Failed();
        }
        if (force_xyz != nullptr &&
            !writer_.Append_Float32(path::force_value, force_xyz,
                                    atom_count_ * 3))
        {
            return Mark_Failed();
        }
        ++particle_frame_count_;
        if (!writer_.Write_Output_Completion(
                static_cast<int64_t>(particle_frame_count_), step, time))
        {
            return Mark_Failed();
        }
        return true;
    }

    bool Append_Observable_Frame(
        const int64_t step, const double time,
        const std::map<std::string, double>& values_by_hdf5_name)
    {
        if (!writer_.Append_Int64(path::observables_all_step, &step, 1))
        {
            return false;
        }
        if (!writer_.Append_Float64(path::observables_all_time, &time, 1))
        {
            return false;
        }
        for (const std::string& name : observable_names_)
        {
            const auto iter = values_by_hdf5_name.find(name);
            if (iter == values_by_hdf5_name.end())
            {
                last_error_ = "observable value is missing: " + name;
                return false;
            }
            const double value = iter->second;
            if (!writer_.Append_Float64(Observable_Value_Path(name), &value, 1))
            {
                return false;
            }
        }
        ++observable_frame_count_;
        return true;
    }

    bool Ensure_Nose_Hoover_Chain_Observables(std::size_t chain_length)
    {
        ModuleH5MappingWriter module_writer(&writer_);
        return module_writer.Ensure_Nose_Hoover_Chain_Observables(chain_length);
    }

    bool Append_Nose_Hoover_Chain_Frame(const int64_t step, const double time,
                                        const float* coordinates,
                                        const float* velocities,
                                        std::size_t chain_length)
    {
        ModuleH5MappingWriter module_writer(&writer_);
        return module_writer.Append_Nose_Hoover_Chain_Frame(
            step, time, coordinates, velocities, chain_length);
    }

    bool Ensure_Sits_Nk_Observable(const std::string& module_name,
                                   std::size_t k_count)
    {
        ModuleH5MappingWriter module_writer(&writer_);
        return module_writer.Ensure_Sits_Nk_Observable(module_name, k_count);
    }

    bool Append_Sits_Nk_Frame(const int64_t step, const double time,
                              const std::string& module_name,
                              const float* values, std::size_t k_count)
    {
        ModuleH5MappingWriter module_writer(&writer_);
        return module_writer.Append_Sits_Nk_Frame(step, time, module_name,
                                                  values, k_count);
    }

    bool Ensure_Metadynamics_Scalars()
    {
        ModuleH5MappingWriter module_writer(&writer_);
        return module_writer.Ensure_Metadynamics_Scalars();
    }

    bool Append_Metadynamics_Scalar_Frame(const int64_t step, const double time,
                                          double meta, double rbias, double rct)
    {
        ModuleH5MappingWriter module_writer(&writer_);
        return module_writer.Append_Metadynamics_Scalar_Frame(step, time, meta,
                                                              rbias, rct);
    }

    bool Write_Metadynamics_Diagnostic(const std::string& name,
                                       const std::string& component,
                                       const std::string& text)
    {
        ModuleH5MappingWriter module_writer(&writer_);
        return module_writer.Write_Metadynamics_Diagnostic(name, component,
                                                           text);
    }

    bool Write_Qc_Scf_Output(const std::string& text)
    {
        ModuleH5MappingWriter module_writer(&writer_);
        return module_writer.Write_Qc_Scf_Output(text);
    }

    bool Ensure_Qc_Observables(bool include_spin_square)
    {
        ModuleH5MappingWriter module_writer(&writer_);
        return module_writer.Ensure_Qc_Observables(include_spin_square);
    }

    bool Append_Qc_Frame(const int64_t step, const double time, double energy,
                         const double* spin_square = nullptr)
    {
        ModuleH5MappingWriter module_writer(&writer_);
        return module_writer.Append_Qc_Frame(step, time, energy, spin_square);
    }

    bool Ensure_Reaxff_Energy_Terms(const std::vector<std::string>& terms)
    {
        reaxff_terms_ = terms;
        ModuleH5MappingWriter module_writer(&writer_);
        return module_writer.Ensure_Reaxff_Energy_Terms(reaxff_terms_);
    }

    bool Append_Reaxff_Frame(
        const int64_t step, const double time,
        const std::map<std::string, double>& values_by_term)
    {
        ModuleH5MappingWriter module_writer(&writer_);
        return module_writer.Append_Reaxff_Frame(step, time, reaxff_terms_,
                                                 values_by_term);
    }

    bool Write_Mdinfo_Text(const std::string& text)
    {
        return writer_.Write_String(path::mdinfo_text, text);
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

    std::size_t Particle_Frame_Count() const { return particle_frame_count_; }
    std::size_t Observable_Frame_Count() const
    {
        return observable_frame_count_;
    }

    std::string Last_Error() const
    {
        if (!last_error_.empty()) return last_error_;
        return writer_.Last_Error();
    }

   private:
    bool Create_Frame_Index_Dataset(const std::string& dataset_path,
                                    DataType type)
    {
        return writer_.Create_Dataset(
            {dataset_path, type, {{0}, {0}, {0}}, true});
    }

    bool Create_Vector3_Frame_Dataset(const std::string& dataset_path,
                                      const std::size_t atom_count)
    {
        return writer_.Create_Dataset(
            {dataset_path,
             DataType::float32,
             {{0, atom_count, 3}, {0, 0, 0}, {0, atom_count, 3}},
             true});
    }

    bool Create_Box_Frame_Dataset(const std::string& dataset_path)
    {
        return writer_.Create_Dataset({dataset_path,
                                       DataType::float32,
                                       {{0, 3, 3}, {0, 0, 0}, {0, 3, 3}},
                                       true});
    }

    bool Mark_Failed()
    {
        const std::string reason = Last_Error();
        writer_.Mark_Failed(reason);
        return false;
    }

    H5MDWriter writer_;
    std::size_t atom_count_ = 0;
    std::size_t particle_frame_count_ = 0;
    std::size_t observable_frame_count_ = 0;
    std::vector<std::string> observable_names_;
    std::vector<std::string> reaxff_terms_;
    std::string last_error_;
};
}  // namespace SpongeH5MD
