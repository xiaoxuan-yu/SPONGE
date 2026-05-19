#include "config.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../third_party/toml/toml_decode.hpp"

namespace fs = std::filesystem;

namespace sponge::manager
{

struct ManagerTomlSection
{
    std::optional<std::string> name;
    std::optional<int> block_steps;
    std::optional<std::string> log_path;
    std::optional<std::string> exchange_log_path;
    std::optional<int> epochs;
    std::optional<bool> emit_output;
    std::optional<std::string> remd_mode;
    std::optional<int> exchange_round;
    std::optional<std::string> transport;
};

struct ExchangeTomlSection
{
    std::optional<bool> enabled;
    std::optional<std::string> mode;
    std::optional<std::string> pairing;
    std::optional<int> start_round;
};

struct WorkerTomlSection
{
    std::optional<std::string> name;
    std::optional<std::vector<std::string>> args;
    std::optional<std::string> working_directory;
    std::optional<std::string> executable_path;
};

struct WorkerDefaultsTomlSection
{
    std::optional<std::string> executable;
    std::optional<std::string> executable_path;
    std::optional<std::string> mdin;
    std::optional<std::vector<std::string>> args;
    std::optional<std::string> working_directory_root;
    std::optional<bool> emit_output;
    std::optional<sponge::toml_decode::table> inputs;
};

struct ScheduleTomlSection
{
    std::optional<int> schedule_id;
    std::optional<std::string> name;
    std::optional<std::string> label;
    std::optional<std::string> working_directory;
    std::optional<sponge::toml_decode::table> inputs;
    WorkerTomlSection worker;
};

struct ManagerTomlRoot
{
    ManagerTomlSection manager;
    ExchangeTomlSection exchange;
    WorkerDefaultsTomlSection worker_defaults;
    std::optional<sponge::toml_decode::node> schedules;
};

namespace
{

std::string ResolveMaybeRelativePath(const fs::path& base,
                                     const std::string& raw)
{
    if (raw.empty())
    {
        return raw;
    }
    const fs::path path(raw);
    if (path.is_absolute())
    {
        return fs::absolute(path).lexically_normal().string();
    }
    return fs::absolute(base / path).lexically_normal().string();
}

void ValidateRemdMode(const std::string& remd_mode)
{
    if (remd_mode.empty())
    {
        return;
    }
    if (remd_mode != "tremd" && remd_mode != "hremd" && remd_mode != "htremd" &&
        remd_mode != "rest2")
    {
        throw std::runtime_error("unsupported remd mode in manager config: " +
                                 remd_mode);
    }
}

void ValidateTransport(const std::string& transport)
{
    if (transport.empty())
    {
        return;
    }
    if (transport != "tcp" && transport != "file" && transport != "shm")
    {
        throw std::runtime_error(
            "manager.transport must be tcp, file, or shm when set");
    }
}

ScheduleInputValue DecodeScheduleInputValue(
    const std::string& key, const sponge::toml_decode::node& node)
{
    if (const auto* value = node.as_integer())
    {
        return static_cast<std::int64_t>(*value);
    }
    if (const auto* value = node.as_floating())
    {
        return *value;
    }
    if (const auto* value = node.as_bool())
    {
        return *value;
    }
    if (const auto* value = node.as_string())
    {
        return *value;
    }
    throw std::runtime_error(
        "schedules.inputs." + key +
        " must be a scalar string, integer, float, or bool");
}

ScheduleInputs DecodeScheduleInputs(const sponge::toml_decode::table& input)
{
    ScheduleInputs out;
    for (const auto& item : input)
    {
        out.values[item.first] =
            DecodeScheduleInputValue(item.first, item.second);
    }
    return out;
}

void MergeScheduleInputs(ScheduleInputs* target, const ScheduleInputs& overlay)
{
    if (target == nullptr)
    {
        return;
    }
    for (const auto& item : overlay.values)
    {
        target->values[item.first] = item.second;
    }
}

bool ArgsContainKey(const std::vector<std::string>& args,
                    const std::string& key)
{
    return std::find(args.begin(), args.end(), key) != args.end();
}

std::vector<std::string> BuildDefaultWorkerArgs(
    const fs::path& config_dir,
    const WorkerDefaultsTomlSection& worker_defaults)
{
    std::vector<std::string> args =
        worker_defaults.args.value_or(std::vector<std::string>{});
    if (!worker_defaults.mdin.has_value())
    {
        return args;
    }
    if (ArgsContainKey(args, "-mdin"))
    {
        throw std::runtime_error(
            "worker_defaults.mdin cannot be combined with "
            "worker_defaults.args containing -mdin");
    }
    std::vector<std::string> with_mdin;
    with_mdin.reserve(args.size() + 2);
    with_mdin.push_back("-mdin");
    with_mdin.push_back(
        ResolveMaybeRelativePath(config_dir, *worker_defaults.mdin));
    with_mdin.insert(with_mdin.end(), args.begin(), args.end());
    return with_mdin;
}

std::string ResolveWorkingDirectory(const fs::path& config_dir,
                                    const std::optional<std::string>& root,
                                    const std::string& raw)
{
    const fs::path path(raw);
    if (path.is_absolute())
    {
        return fs::absolute(path).lexically_normal().string();
    }
    if (root.has_value())
    {
        const fs::path root_path = ResolveMaybeRelativePath(config_dir, *root);
        return fs::absolute(root_path / path).lexically_normal().string();
    }
    return ResolveMaybeRelativePath(config_dir, raw);
}

std::string ExecutableName(const std::string& base)
{
#ifdef _WIN32
    return base + ".exe";
#else
    return base;
#endif
}

std::optional<fs::path> FindExecutableOnPath(const std::string& name)
{
    const char* path_env = std::getenv("PATH");
    if (path_env == nullptr)
    {
        return std::nullopt;
    }
#ifdef _WIN32
    constexpr char kPathSeparator = ';';
#else
    constexpr char kPathSeparator = ':';
#endif
    std::string current;
    const std::string paths(path_env);
    for (const char c : paths)
    {
        if (c == kPathSeparator)
        {
            if (!current.empty())
            {
                const fs::path candidate = fs::path(current) / name;
                if (fs::exists(candidate))
                {
                    return fs::absolute(candidate).lexically_normal();
                }
            }
            current.clear();
            continue;
        }
        current.push_back(c);
    }
    if (!current.empty())
    {
        const fs::path candidate = fs::path(current) / name;
        if (fs::exists(candidate))
        {
            return fs::absolute(candidate).lexically_normal();
        }
    }
    return std::nullopt;
}

std::string ResolveDefaultSpongeExecutable(const fs::path& manager_exe)
{
    const std::string sponge_name = ExecutableName("SPONGE");
    const fs::path sibling = manager_exe.parent_path() / sponge_name;
    if (fs::exists(sibling))
    {
        return fs::absolute(sibling).lexically_normal().string();
    }
    const auto from_path = FindExecutableOnPath(sponge_name);
    if (from_path.has_value())
    {
        return from_path->string();
    }
    throw std::runtime_error(
        "unable to find SPONGE executable. Set worker_defaults.executable or "
        "place SPONGE next to SPONGE_MANAGER/in PATH");
}

std::vector<int> DecodeScheduleIds(const sponge::toml_decode::node& node)
{
    const auto* items = node.as_array();
    if (items == nullptr)
    {
        throw std::runtime_error("schedules.ids must be an integer array");
    }
    std::vector<int> ids;
    ids.reserve(items->size());
    for (const auto& item : *items)
    {
        if (const auto* value = item.as_integer())
        {
            ids.push_back(static_cast<int>(*value));
            continue;
        }
        throw std::runtime_error("schedules.ids must contain only integers");
    }
    return ids;
}

std::vector<ScheduleTomlSection> ExpandBulkSchedules(
    const sponge::toml_decode::table& table)
{
    const auto id_node = table.find("ids");
    if (id_node == table.end())
    {
        throw std::runtime_error("bulk [schedules] requires ids");
    }
    const auto ids = DecodeScheduleIds(id_node->second);
    std::vector<ScheduleTomlSection> schedules(ids.size());
    for (std::size_t i = 0; i < ids.size(); i++)
    {
        schedules[i].schedule_id = ids[i];
    }

    const auto inputs_node = table.find("inputs");
    if (inputs_node == table.end())
    {
        return schedules;
    }
    const auto* inputs_table = inputs_node->second.as_table();
    if (inputs_table == nullptr)
    {
        throw std::runtime_error("schedules.inputs must be a table");
    }
    for (const auto& item : *inputs_table)
    {
        if (const auto* values = item.second.as_array())
        {
            if (values->size() != ids.size())
            {
                throw std::runtime_error(
                    "schedules.inputs." + item.first +
                    " array length must match schedules.ids length");
            }
            for (std::size_t i = 0; i < values->size(); i++)
            {
                if (!schedules[i].inputs.has_value())
                {
                    schedules[i].inputs = sponge::toml_decode::table{};
                }
                (*schedules[i].inputs)[item.first] = (*values)[i];
            }
            continue;
        }
        for (auto& schedule : schedules)
        {
            if (!schedule.inputs.has_value())
            {
                schedule.inputs = sponge::toml_decode::table{};
            }
            (*schedule.inputs)[item.first] = item.second;
        }
    }
    return schedules;
}

std::vector<ScheduleTomlSection> DecodeSchedules(
    const std::optional<sponge::toml_decode::node>& schedules_node)
{
    if (!schedules_node.has_value())
    {
        throw std::runtime_error(
            "manager config must define schedules or [[schedules]]");
    }
    if (const auto* schedules_array = schedules_node->as_array())
    {
        std::vector<ScheduleTomlSection> schedules;
        schedules.reserve(schedules_array->size());
        for (const auto& item : *schedules_array)
        {
            schedules.push_back(
                sponge::toml_decode::decode_node<ScheduleTomlSection>(item));
        }
        return schedules;
    }
    if (const auto* schedules_table = schedules_node->as_table())
    {
        return ExpandBulkSchedules(*schedules_table);
    }
    throw std::runtime_error("schedules must be a table or array of tables");
}

void ValidateExchangeInputs(const std::string& remd_mode,
                            const std::vector<ScheduleConfig>& schedules)
{
    if (remd_mode.empty())
    {
        return;
    }
    for (const auto& schedule : schedules)
    {
        if ((remd_mode == "tremd" || remd_mode == "hremd" ||
             remd_mode == "htremd" || remd_mode == "rest2") &&
            !schedule.inputs.FindDouble("target_temperature").has_value())
        {
            throw std::runtime_error(
                "schedules.inputs.target_temperature is required for " +
                remd_mode);
        }
        if ((remd_mode == "hremd" || remd_mode == "htremd") &&
            !schedule.inputs.FindInt("hamiltonian_id").has_value())
        {
            throw std::runtime_error(
                "schedules.inputs.hamiltonian_id is required for " + remd_mode);
        }
        if (remd_mode == "rest2" &&
            !schedule.inputs.FindDouble("REST2_lambda_m").has_value())
        {
            throw std::runtime_error(
                "schedules.inputs.REST2_lambda_m is required for rest2");
        }
    }
}

void ValidateScheduleIds(const std::vector<ScheduleConfig>& schedules)
{
    std::unordered_map<int, std::size_t> owners;
    for (std::size_t i = 0; i < schedules.size(); i++)
    {
        const int schedule_id = schedules[i].schedule_id;
        const auto [it, inserted] = owners.emplace(schedule_id, i);
        if (!inserted)
        {
            throw std::runtime_error(
                "duplicate schedule_id in manager config: " +
                std::to_string(schedule_id));
        }
    }
}

std::string ScheduleOutputPrefix(const ScheduleConfig& schedule)
{
    const std::string base_prefix =
        schedule.inputs.FindString("default_out_file_prefix").value_or("mdout");
    return base_prefix + "_" + std::to_string(schedule.schedule_id);
}

void RegisterOutputPath(std::unordered_map<std::string, int>* owners,
                        const fs::path& path, int schedule_id)
{
    const std::string normalized =
        fs::absolute(path).lexically_normal().string();
    const auto [it, inserted] = owners->emplace(normalized, schedule_id);
    if (!inserted && it->second != schedule_id)
    {
        throw std::runtime_error("output path conflict: schedules " +
                                 std::to_string(it->second) + " and " +
                                 std::to_string(schedule_id) + " both write " +
                                 normalized);
    }
}

void ValidateOutputPathConflicts(const std::vector<ScheduleConfig>& schedules,
                                 const std::string& exchange_log_path)
{
    std::unordered_map<std::string, int> owners;
    for (const auto& schedule : schedules)
    {
        const fs::path directory(schedule.worker.working_directory);
        const fs::path prefix_path = directory / ScheduleOutputPrefix(schedule);
        RegisterOutputPath(&owners, prefix_path, schedule.schedule_id);
        RegisterOutputPath(&owners, prefix_path.string() + ".info",
                           schedule.schedule_id);
        RegisterOutputPath(&owners, prefix_path.string() + ".out",
                           schedule.schedule_id);
        RegisterOutputPath(&owners, prefix_path.string() + ".dat",
                           schedule.schedule_id);
        RegisterOutputPath(&owners, prefix_path.string() + ".box",
                           schedule.schedule_id);
    }

    if (!exchange_log_path.empty())
    {
        const std::string normalized_log =
            fs::absolute(exchange_log_path).lexically_normal().string();
        if (owners.find(normalized_log) != owners.end())
        {
            throw std::runtime_error(
                "manager log path conflicts with worker output path: " +
                normalized_log);
        }
    }
}

}  // namespace

}  // namespace sponge::manager

SPONGE_TOML_DECODE_REFLECT(
    sponge::manager::ManagerTomlSection,
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ManagerTomlSection, name),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ManagerTomlSection, block_steps),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ManagerTomlSection, log_path),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ManagerTomlSection,
                              exchange_log_path),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ManagerTomlSection, epochs),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ManagerTomlSection, emit_output),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ManagerTomlSection, remd_mode),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ManagerTomlSection,
                              exchange_round),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ManagerTomlSection, transport))

SPONGE_TOML_DECODE_REFLECT(
    sponge::manager::ExchangeTomlSection,
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ExchangeTomlSection, enabled),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ExchangeTomlSection, mode),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ExchangeTomlSection, pairing),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ExchangeTomlSection,
                              start_round))

SPONGE_TOML_DECODE_REFLECT(
    sponge::manager::WorkerTomlSection,
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::WorkerTomlSection, name),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::WorkerTomlSection, args),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::WorkerTomlSection,
                              working_directory),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::WorkerTomlSection,
                              executable_path))

SPONGE_TOML_DECODE_REFLECT(
    sponge::manager::WorkerDefaultsTomlSection,
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::WorkerDefaultsTomlSection,
                              executable),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::WorkerDefaultsTomlSection,
                              executable_path),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::WorkerDefaultsTomlSection, mdin),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::WorkerDefaultsTomlSection, args),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::WorkerDefaultsTomlSection,
                              working_directory_root),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::WorkerDefaultsTomlSection,
                              emit_output),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::WorkerDefaultsTomlSection,
                              inputs))

SPONGE_TOML_DECODE_REFLECT(
    sponge::manager::ScheduleTomlSection,
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ScheduleTomlSection,
                              schedule_id),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ScheduleTomlSection, name),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ScheduleTomlSection, label),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ScheduleTomlSection,
                              working_directory),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ScheduleTomlSection, inputs),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ScheduleTomlSection, worker))

SPONGE_TOML_DECODE_REFLECT(
    sponge::manager::ManagerTomlRoot,
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ManagerTomlRoot, manager),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ManagerTomlRoot, exchange),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ManagerTomlRoot,
                              worker_defaults),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::ManagerTomlRoot, schedules))

namespace sponge::manager
{

ManagerExecutionConfig LoadManagerExecutionConfigFromToml(
    const std::string& config_path, const std::string& manager_executable_path)
{
    const fs::path path = fs::absolute(config_path).lexically_normal();
    const fs::path config_dir = path.parent_path();
    const fs::path manager_exe =
        fs::absolute(manager_executable_path).lexically_normal();

    const auto parsed =
        sponge::toml_decode::parse_file<ManagerTomlRoot>(path.string());

    ManagerExecutionConfig out;
    out.manager.block_steps = parsed.manager.block_steps.value_or(1000);
    out.epochs = parsed.manager.epochs.value_or(1);
    out.emit_output = parsed.worker_defaults.emit_output.value_or(
        parsed.manager.emit_output.value_or(true));
    out.remd_mode =
        parsed.exchange.mode.value_or(parsed.manager.remd_mode.value_or(""));
    const bool exchange_enabled =
        parsed.exchange.enabled.value_or(!out.remd_mode.empty());
    if (!exchange_enabled)
    {
        out.remd_mode.clear();
    }
    out.exchange_round = parsed.exchange.start_round.value_or(
        parsed.manager.exchange_round.value_or(0));

    if (out.manager.block_steps <= 0)
    {
        throw std::runtime_error("manager.block_steps must be > 0");
    }
    if (out.epochs <= 0)
    {
        throw std::runtime_error("manager.epochs must be > 0");
    }
    ValidateRemdMode(out.remd_mode);
    const std::string manager_transport = parsed.manager.transport.value_or("");
    ValidateTransport(manager_transport);

    const auto log_path = parsed.manager.log_path.has_value()
                              ? parsed.manager.log_path
                              : parsed.manager.exchange_log_path;
    if (log_path.has_value())
    {
        out.manager.exchange_log_path =
            ResolveMaybeRelativePath(config_dir, *log_path);
    }
    else
    {
        out.manager.exchange_log_path =
            (config_dir / "manager_exchange.log").string();
    }

    const auto parsed_schedules = DecodeSchedules(parsed.schedules);
    if (parsed_schedules.empty())
    {
        throw std::runtime_error(
            "manager config must define at least one schedule");
    }

    out.manager.schedules.reserve(parsed_schedules.size());
    const auto default_args =
        BuildDefaultWorkerArgs(config_dir, parsed.worker_defaults);
    ScheduleInputs default_inputs;
    if (parsed.worker_defaults.inputs.has_value())
    {
        default_inputs = DecodeScheduleInputs(*parsed.worker_defaults.inputs);
    }
    const auto default_executable =
        parsed.worker_defaults.executable.has_value()
            ? parsed.worker_defaults.executable
            : parsed.worker_defaults.executable_path;
    for (std::size_t i = 0; i < parsed_schedules.size(); i++)
    {
        const auto& schedule_in = parsed_schedules[i];
        ScheduleConfig schedule;
        schedule.schedule_id =
            schedule_in.schedule_id.value_or(static_cast<int>(i));
        schedule.name = schedule_in.label.value_or(schedule_in.name.value_or(
            "schedule_" + std::to_string(schedule.schedule_id)));
        schedule.inputs = default_inputs;
        if (schedule_in.inputs.has_value())
        {
            MergeScheduleInputs(&schedule.inputs,
                                DecodeScheduleInputs(*schedule_in.inputs));
        }

        schedule.worker.name = schedule_in.worker.name.value_or(
            "worker_" + std::to_string(schedule.schedule_id));
        schedule.worker.args = default_args;
        if (schedule_in.worker.args.has_value())
        {
            schedule.worker.args.insert(schedule.worker.args.end(),
                                        schedule_in.worker.args->begin(),
                                        schedule_in.worker.args->end());
        }
        const std::string working_directory =
            schedule_in.working_directory.value_or(
                schedule_in.worker.working_directory.value_or(
                    std::to_string(schedule.schedule_id)));
        schedule.worker.working_directory = ResolveWorkingDirectory(
            config_dir, parsed.worker_defaults.working_directory_root,
            working_directory);
        schedule.worker.transport =
            !manager_transport.empty() ? manager_transport : "tcp";

        if (schedule_in.worker.executable_path.has_value())
        {
            schedule.worker.executable_path = ResolveMaybeRelativePath(
                config_dir, *schedule_in.worker.executable_path);
        }
        else if (default_executable.has_value())
        {
            schedule.worker.executable_path =
                ResolveMaybeRelativePath(config_dir, *default_executable);
        }
        else
        {
            schedule.worker.executable_path =
                ResolveDefaultSpongeExecutable(manager_exe);
        }

        out.manager.schedules.push_back(std::move(schedule));
    }
    ValidateScheduleIds(out.manager.schedules);
    ValidateExchangeInputs(out.remd_mode, out.manager.schedules);
    ValidateOutputPathConflicts(out.manager.schedules,
                                out.manager.exchange_log_path);

    return out;
}

}  // namespace sponge::manager
