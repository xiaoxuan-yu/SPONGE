#include "config.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
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
    std::optional<std::vector<std::string>> args;
    std::optional<std::string> working_directory_root;
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
    std::vector<ScheduleTomlSection> schedules;
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
    if (remd_mode != "tremd" && remd_mode != "hremd" && remd_mode != "htremd")
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
             remd_mode == "htremd") &&
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
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::WorkerDefaultsTomlSection, args),
    SPONGE_TOML_DECODE_MEMBER(sponge::manager::WorkerDefaultsTomlSection,
                              working_directory_root))

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
    const fs::path default_sponge_worker = manager_exe.parent_path() / "SPONGE";

    const auto parsed =
        sponge::toml_decode::parse_file<ManagerTomlRoot>(path.string());

    ManagerExecutionConfig out;
    out.manager.block_steps = parsed.manager.block_steps.value_or(1000);
    out.epochs = parsed.manager.epochs.value_or(1);
    out.emit_output = parsed.manager.emit_output.value_or(false);
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

    if (parsed.schedules.empty())
    {
        throw std::runtime_error(
            "manager config must define at least one [[schedules]] entry");
    }

    out.manager.schedules.reserve(parsed.schedules.size());
    const auto default_args =
        parsed.worker_defaults.args.value_or(std::vector<std::string>{});
    const auto default_executable =
        parsed.worker_defaults.executable.has_value()
            ? parsed.worker_defaults.executable
            : parsed.worker_defaults.executable_path;
    for (std::size_t i = 0; i < parsed.schedules.size(); i++)
    {
        const auto& schedule_in = parsed.schedules[i];
        ScheduleConfig schedule;
        schedule.schedule_id =
            schedule_in.schedule_id.value_or(static_cast<int>(i));
        schedule.name = schedule_in.label.value_or(schedule_in.name.value_or(
            "schedule_" + std::to_string(schedule.schedule_id)));
        if (schedule_in.inputs.has_value())
        {
            schedule.inputs = DecodeScheduleInputs(*schedule_in.inputs);
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
                schedule_in.worker.working_directory.value_or("."));
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
                default_sponge_worker.lexically_normal().string();
        }

        out.manager.schedules.push_back(std::move(schedule));
    }
    ValidateExchangeInputs(out.remd_mode, out.manager.schedules);

    return out;
}

}  // namespace sponge::manager
