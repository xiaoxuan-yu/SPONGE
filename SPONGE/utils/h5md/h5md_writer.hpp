#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace SpongeH5MD
{
#ifndef SPONGE_VERSION_STR
#define SPONGE_VERSION_STR "unknown"
#endif

inline constexpr const char* kCanonicalSchemaVersion =
    "xponge.legacy_to_bundle.v1";
inline constexpr const char* kSpongeWriterVersion = SPONGE_VERSION_STR;

enum class FileStatus
{
    closed,
    open,
    closing,
    finalized,
    failed
};

enum class DataType
{
    int64,
    float32,
    float64,
    string
};

struct DatasetShape
{
    std::vector<std::size_t> dims;
    std::vector<std::size_t> max_dims;
    std::vector<std::size_t> chunk_dims;
};

struct DatasetSpec
{
    std::string path;
    DataType type = DataType::float64;
    DatasetShape shape;
    bool appendable = true;
};

struct VirtualDatasetSource
{
    std::string file_path;
    std::string dataset_path;
    std::vector<std::size_t> source_dims;
    std::vector<std::size_t> virtual_start;
};

struct WriterOptions
{
    std::string path;
    std::string schema_name = "sponge.output.h5md";
    std::string schema_version = kCanonicalSchemaVersion;
    bool observable_only = false;
    bool swmr_compatible = false;
};

class WriterBackend
{
   public:
    virtual ~WriterBackend() = default;

    virtual bool Open(const WriterOptions& options) = 0;
    virtual bool Start_Swmr_Write() { return false; }
    virtual bool Flush() = 0;
    virtual bool Close() = 0;
    virtual bool Finalize() = 0;

    virtual bool Ensure_Group(const std::string& path) = 0;
    virtual bool Create_Dataset(const DatasetSpec& spec) = 0;
    virtual bool Create_Virtual_Dataset(
        const DatasetSpec& spec,
        const std::vector<VirtualDatasetSource>& sources) = 0;
    virtual bool Create_Hard_Link(const std::string& target,
                                  const std::string& link_path) = 0;

    virtual bool Append_Int64(const std::string& path, const int64_t* data,
                              std::size_t count) = 0;
    virtual bool Append_Float32(const std::string& path, const float* data,
                                std::size_t count) = 0;
    virtual bool Append_Float64(const std::string& path, const double* data,
                                std::size_t count) = 0;
    virtual bool Write_String(const std::string& path,
                              const std::string& value) = 0;
    virtual bool Write_String_Array(const std::string& path,
                                    const std::vector<std::string>& values) = 0;
    virtual bool Set_String_Attribute(const std::string& object_path,
                                      const std::string& name,
                                      const std::string& value) = 0;
    virtual bool Set_Status(FileStatus status) = 0;

    virtual FileStatus Status() const = 0;
    virtual std::string Last_Error() const = 0;
};

class WriterBackendFactory
{
   public:
    virtual ~WriterBackendFactory() = default;
    virtual std::unique_ptr<WriterBackend> Create_Backend() = 0;
};

class H5MDWriter
{
   public:
    explicit H5MDWriter(WriterBackend* backend) : backend_(backend) {}

    bool Is_Attached() const { return backend_ != nullptr; }

    bool Open(const WriterOptions& options)
    {
        if (backend_ == nullptr)
        {
            return false;
        }
        options_ = options;
        if (!backend_->Open(options))
        {
            return false;
        }
        return Initialize_Common_Layout();
    }

    bool Flush() { return backend_ != nullptr && backend_->Flush(); }
    bool Start_Swmr_Write()
    {
        return backend_ != nullptr && backend_->Start_Swmr_Write();
    }
    bool Close() { return backend_ != nullptr && backend_->Close(); }
    bool Finalize() { return backend_ != nullptr && backend_->Finalize(); }

    bool Ensure_Group(const std::string& path)
    {
        return backend_ != nullptr && backend_->Ensure_Group(path);
    }

    bool Create_Dataset(const DatasetSpec& spec)
    {
        return backend_ != nullptr && backend_->Create_Dataset(spec);
    }

    bool Create_Virtual_Dataset(
        const DatasetSpec& spec,
        const std::vector<VirtualDatasetSource>& sources)
    {
        return backend_ != nullptr &&
               backend_->Create_Virtual_Dataset(spec, sources);
    }

    bool Create_Hard_Link(const std::string& target,
                          const std::string& link_path)
    {
        return backend_ != nullptr &&
               backend_->Create_Hard_Link(target, link_path);
    }

    bool Append_Int64(const std::string& path, const int64_t* data,
                      std::size_t count)
    {
        return backend_ != nullptr && backend_->Append_Int64(path, data, count);
    }

    bool Append_Float32(const std::string& path, const float* data,
                        std::size_t count)
    {
        return backend_ != nullptr &&
               backend_->Append_Float32(path, data, count);
    }

    bool Append_Float64(const std::string& path, const double* data,
                        std::size_t count)
    {
        return backend_ != nullptr &&
               backend_->Append_Float64(path, data, count);
    }

    bool Write_String(const std::string& path, const std::string& value)
    {
        return backend_ != nullptr && backend_->Write_String(path, value);
    }

    bool Write_String_Array(const std::string& path,
                            const std::vector<std::string>& values)
    {
        return backend_ != nullptr &&
               backend_->Write_String_Array(path, values);
    }

    bool Set_String_Attribute(const std::string& object_path,
                              const std::string& name,
                              const std::string& value)
    {
        return backend_ != nullptr &&
               backend_->Set_String_Attribute(object_path, name, value);
    }

    bool Write_Topology_Compatibility(const std::string& topology_hash,
                                      const std::string& atom_order_hash)
    {
        if (backend_ == nullptr)
        {
            return false;
        }
        constexpr const char* compatibility_root =
            "/parameters/sponge/topology_compatibility";
        constexpr const char* topology_hash_path =
            "/parameters/sponge/topology_compatibility/topology_hash";
        constexpr const char* atom_order_hash_path =
            "/parameters/sponge/topology_compatibility/atom_order_hash";
        if (!Ensure_Group(compatibility_root)) return false;
        if (!topology_hash.empty() &&
            !Write_String(topology_hash_path, topology_hash))
        {
            return false;
        }
        return atom_order_hash.empty() ||
               Write_String(atom_order_hash_path, atom_order_hash);
    }

    bool Set_Status(FileStatus status)
    {
        return backend_ != nullptr && backend_->Set_Status(status);
    }

    bool Mark_Failed(const std::string& reason)
    {
        if (backend_ == nullptr)
        {
            return false;
        }
        return backend_->Set_Status(FileStatus::failed) &&
               backend_->Write_String(kOutputError, reason);
    }

    bool Write_Output_Completion(int64_t frame_count, int64_t step, double time)
    {
        if (backend_ == nullptr)
        {
            return false;
        }
        if (!Create_Dataset(
                {kOutputFrameCount, DataType::int64, {{0}, {0}, {1}}, true}))
        {
            return false;
        }
        if (!Create_Dataset({kOutputLastCompleteStep,
                             DataType::int64,
                             {{0}, {0}, {1}},
                             true}))
        {
            return false;
        }
        if (!Create_Dataset({kOutputLastCompleteTime,
                             DataType::float64,
                             {{0}, {0}, {1}},
                             true}))
        {
            return false;
        }
        return Append_Int64(kOutputFrameCount, &frame_count, 1) &&
               Append_Int64(kOutputLastCompleteStep, &step, 1) &&
               Append_Float64(kOutputLastCompleteTime, &time, 1);
    }

    FileStatus Status() const
    {
        if (backend_ == nullptr)
        {
            return FileStatus::closed;
        }
        return backend_->Status();
    }

    std::string Last_Error() const
    {
        if (backend_ == nullptr)
        {
            return "H5MD writer backend is not attached";
        }
        return backend_->Last_Error();
    }

   private:
    bool Initialize_Common_Layout()
    {
        if (!Ensure_Group("/h5md")) return false;
        if (!Ensure_Group("/h5md/creator")) return false;
        if (!Set_String_Attribute("/h5md/creator", "name", "SPONGE"))
        {
            return false;
        }
        if (!Set_String_Attribute("/h5md/creator", "version",
                                  kSpongeWriterVersion))
        {
            return false;
        }
        if (!options_.observable_only && !Ensure_Group("/particles"))
        {
            return false;
        }
        if (!Ensure_Group("/observables")) return false;
        if (!Ensure_Group("/parameters")) return false;
        if (!Ensure_Group("/parameters/sponge")) return false;
        if (!Ensure_Group("/parameters/sponge/schema")) return false;
        if (!Ensure_Group("/parameters/sponge/output")) return false;
        if (!Write_String("/parameters/sponge/schema/name",
                          options_.schema_name))
        {
            return false;
        }
        if (!Write_String("/parameters/sponge/schema/version",
                          options_.schema_version))
        {
            return false;
        }
        return Set_Status(FileStatus::open) && Write_String(kOutputError, "") &&
               Write_Output_Completion(0, -1, 0.0);
    }

    WriterBackend* backend_ = nullptr;
    WriterOptions options_;
    static constexpr const char* kOutputFrameCount =
        "/parameters/sponge/output/frame_count";
    static constexpr const char* kOutputLastCompleteStep =
        "/parameters/sponge/output/last_complete_step";
    static constexpr const char* kOutputLastCompleteTime =
        "/parameters/sponge/output/last_complete_time";
    static constexpr const char* kOutputError =
        "/parameters/sponge/output/error";
};

namespace path
{
static constexpr const char* h5md = "/h5md";
static constexpr const char* particles = "/particles";
static constexpr const char* observables = "/observables";
static constexpr const char* parameters = "/parameters";
static constexpr const char* sponge = "/parameters/sponge";
static constexpr const char* sponge_schema = "/parameters/sponge/schema";
static constexpr const char* sponge_schema_name =
    "/parameters/sponge/schema/name";
static constexpr const char* sponge_schema_version =
    "/parameters/sponge/schema/version";
static constexpr const char* sponge_output = "/parameters/sponge/output";
static constexpr const char* sponge_mdout = "/parameters/sponge/mdout";
static constexpr const char* sponge_log = "/parameters/sponge/log";
static constexpr const char* sponge_files = "/parameters/sponge/files";
static constexpr const char* sponge_provenance =
    "/parameters/sponge/provenance";
static constexpr const char* sponge_topology_compatibility =
    "/parameters/sponge/topology_compatibility";
static constexpr const char* sponge_topology_hash =
    "/parameters/sponge/topology_compatibility/topology_hash";
static constexpr const char* sponge_atom_order_hash =
    "/parameters/sponge/topology_compatibility/atom_order_hash";
static constexpr const char* output_status = "/parameters/sponge/output/status";
static constexpr const char* output_frame_count =
    "/parameters/sponge/output/frame_count";
static constexpr const char* output_last_complete_step =
    "/parameters/sponge/output/last_complete_step";
static constexpr const char* output_last_complete_time =
    "/parameters/sponge/output/last_complete_time";
static constexpr const char* output_error = "/parameters/sponge/output/error";
static constexpr const char* output_trajectory_chunk_size =
    "/parameters/sponge/output/trajectory_chunk_size";
static constexpr const char* output_vds_status =
    "/parameters/sponge/output/vds_status";
static constexpr const char* output_repair_policy =
    "/parameters/sponge/output/repair_policy";
static constexpr const char* output_repair_status =
    "/parameters/sponge/output/repair_status";
static constexpr const char* output_repaired_shard_count =
    "/parameters/sponge/output/repaired_shard_count";
static constexpr const char* shard_status = "/parameters/sponge/shard/status";
static constexpr const char* shard_frame_start =
    "/parameters/sponge/shard/frame_start";
static constexpr const char* shard_frame_count =
    "/parameters/sponge/shard/frame_count";
static constexpr const char* shard_last_complete_step =
    "/parameters/sponge/shard/last_complete_step";
static constexpr const char* shard_last_complete_time =
    "/parameters/sponge/shard/last_complete_time";
static constexpr const char* shard_manifest =
    "/parameters/sponge/output/shard_manifest";
static constexpr const char* shard_manifest_index =
    "/parameters/sponge/output/shard_manifest/index";
static constexpr const char* shard_manifest_path =
    "/parameters/sponge/output/shard_manifest/path";
static constexpr const char* shard_manifest_frame_start =
    "/parameters/sponge/output/shard_manifest/frame_start";
static constexpr const char* shard_manifest_frame_count =
    "/parameters/sponge/output/shard_manifest/frame_count";
static constexpr const char* shard_manifest_step_start =
    "/parameters/sponge/output/shard_manifest/step_start";
static constexpr const char* shard_manifest_step_end =
    "/parameters/sponge/output/shard_manifest/step_end";
static constexpr const char* shard_manifest_time_start =
    "/parameters/sponge/output/shard_manifest/time_start";
static constexpr const char* shard_manifest_time_end =
    "/parameters/sponge/output/shard_manifest/time_end";
static constexpr const char* shard_manifest_status =
    "/parameters/sponge/output/shard_manifest/status";
static constexpr const char* mdout_columns = "/parameters/sponge/mdout/columns";
static constexpr const char* mdout_columns_original_name =
    "/parameters/sponge/mdout/columns/original_name";
static constexpr const char* mdout_columns_hdf5_name =
    "/parameters/sponge/mdout/columns/hdf5_name";
static constexpr const char* mdinfo_text = "/parameters/sponge/log/mdinfo_text";
static constexpr const char* legacy_sidecars =
    "/parameters/sponge/files/legacy_sidecars";
static constexpr const char* legacy_sidecar_keys =
    "/parameters/sponge/files/legacy_sidecars/key";
static constexpr const char* legacy_sidecar_paths =
    "/parameters/sponge/files/legacy_sidecars/path";
static constexpr const char* particles_all = "/particles/all";
static constexpr const char* particles_all_position = "/particles/all/position";
static constexpr const char* particles_all_velocity = "/particles/all/velocity";
static constexpr const char* particles_all_force = "/particles/all/force";
static constexpr const char* particles_all_box = "/particles/all/box";
static constexpr const char* particles_all_box_edges =
    "/particles/all/box/edges";
static constexpr const char* particles_all_step = "/particles/all/step";
static constexpr const char* particles_all_time = "/particles/all/time";
static constexpr const char* position_value = "/particles/all/position/value";
static constexpr const char* position_step = "/particles/all/position/step";
static constexpr const char* position_time = "/particles/all/position/time";
static constexpr const char* velocity_value = "/particles/all/velocity/value";
static constexpr const char* velocity_step = "/particles/all/velocity/step";
static constexpr const char* velocity_time = "/particles/all/velocity/time";
static constexpr const char* force_value = "/particles/all/force/value";
static constexpr const char* force_step = "/particles/all/force/step";
static constexpr const char* force_time = "/particles/all/force/time";
static constexpr const char* box_edges_value = "/particles/all/box/edges/value";
static constexpr const char* box_edges_step = "/particles/all/box/edges/step";
static constexpr const char* box_edges_time = "/particles/all/box/edges/time";
static constexpr const char* observables_all = "/observables/all";
static constexpr const char* observables_all_step = "/observables/all/step";
static constexpr const char* observables_all_time = "/observables/all/time";
static constexpr const char* run = "/run";
static constexpr const char* run_current_step = "/run/current_step";
static constexpr const char* run_current_time = "/run/current_time";
static constexpr const char* run_state_type = "/run/state_type";
static constexpr const char* parameters_restart = "/parameters/restart";
static constexpr const char* restart_rng_state =
    "/parameters/restart/rng_state";
static constexpr const char* restart_integrator_state =
    "/parameters/restart/integrator_state";
static constexpr const char* restart_thermostat =
    "/parameters/restart/thermostat";
static constexpr const char* restart_nhc =
    "/parameters/restart/thermostat/nose_hoover_chain";
static constexpr const char* restart_barostat = "/parameters/restart/barostat";
static constexpr const char* restart_protocol_sidecars =
    "/parameters/restart/protocol_sidecars";
static constexpr const char* restart_bias = "/parameters/restart/bias";
static constexpr const char* restart_sits = "/parameters/restart/bias/sits";
static constexpr const char* restart_meta = "/parameters/restart/bias/meta";
}  // namespace path

inline std::string Restart_Sits_State_Root(const std::string& module_name)
{
    return std::string(path::restart_sits) + "/" + module_name;
}

inline std::string Restart_Sits_State_Path(const std::string& module_name,
                                           const std::string& state_name)
{
    return Restart_Sits_State_Root(module_name) + "/" + state_name;
}

inline std::string Restart_Metad_State_Root(const std::string& name)
{
    return std::string(path::restart_meta) + "/" + name;
}

inline std::string Restart_Metad_State_Path(const std::string& name,
                                            const std::string& component)
{
    return Restart_Metad_State_Root(name) + "/" + component;
}

inline std::string Restart_Protocol_Sidecar_Path(const std::string& key)
{
    return std::string(path::restart_protocol_sidecars) + "/" + key;
}

inline std::string Restart_Rng_State_Path(const std::string& module_name)
{
    return std::string(path::restart_rng_state) + "/" + module_name;
}

inline std::string Restart_Integrator_State_Path(const std::string& key)
{
    return std::string(path::restart_integrator_state) + "/" + key;
}

inline std::string Restart_Thermostat_State_Root(const std::string& module_name)
{
    return std::string(path::restart_thermostat) + "/" + module_name;
}

inline std::string Restart_Thermostat_State_Path(const std::string& module_name,
                                                 const std::string& state_name)
{
    return Restart_Thermostat_State_Root(module_name) + "/" + state_name;
}

inline std::string Restart_Barostat_State_Root(const std::string& module_name)
{
    return std::string(path::restart_barostat) + "/" + module_name;
}

inline std::string Restart_Barostat_State_Path(const std::string& module_name,
                                               const std::string& state_name)
{
    return Restart_Barostat_State_Root(module_name) + "/" + state_name;
}

inline std::string Observable_Root(const std::string& name)
{
    return std::string(path::observables_all) + "/" + name;
}

inline std::string Observable_Value_Path(const std::string& name)
{
    return Observable_Root(name) + "/value";
}

inline std::string Observable_Step_Path(const std::string& name)
{
    return Observable_Root(name) + "/step";
}

inline std::string Observable_Time_Path(const std::string& name)
{
    return Observable_Root(name) + "/time";
}

inline std::string Sponge_Provenance_Path(const std::string& name)
{
    return std::string(path::sponge_provenance) + "/" + name;
}
}  // namespace SpongeH5MD
