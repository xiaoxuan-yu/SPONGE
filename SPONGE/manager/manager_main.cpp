#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "algorithms/remd/hamiltonian_remd.h"
#include "algorithms/remd/temperature_hamiltonian_remd.h"
#include "algorithms/remd/temperature_remd.h"
#include "config.h"
#include "core/manager.h"

namespace fs = std::filesystem;

namespace
{

std::vector<std::string> SplitCommaSeparated(const std::string& value)
{
    std::vector<std::string> tokens;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        if (!token.empty())
        {
            tokens.push_back(token);
        }
    }
    return tokens;
}

std::vector<int> ParseIntList(const std::string& value)
{
    std::vector<int> result;
    for (const auto& token : SplitCommaSeparated(value))
    {
        result.push_back(std::stoi(token));
    }
    return result;
}

std::vector<double> ParseDoubleList(const std::string& value)
{
    std::vector<double> result;
    for (const auto& token : SplitCommaSeparated(value))
    {
        result.push_back(std::stod(token));
    }
    return result;
}

std::vector<int> DiscoverNumericStateDirectories(const fs::path& root)
{
    std::vector<int> state_ids;
    for (const auto& entry : fs::directory_iterator(root))
    {
        if (!entry.is_directory())
        {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (!name.empty() &&
            std::all_of(name.begin(), name.end(),
                        [](unsigned char c) { return std::isdigit(c) != 0; }))
        {
            state_ids.push_back(std::stoi(name));
        }
    }
    std::sort(state_ids.begin(), state_ids.end());
    return state_ids;
}

std::vector<double> BuildDefaultLambdaLadder(std::size_t count)
{
    std::vector<double> lambdas(count, 0.0);
    if (count == 0)
    {
        return lambdas;
    }
    if (count == 1)
    {
        lambdas[0] = 0.0;
        return lambdas;
    }
    for (std::size_t i = 0; i < count; i++)
    {
        lambdas[i] = static_cast<double>(i) / static_cast<double>(count - 1);
    }
    return lambdas;
}

void PrintUsage()
{
    std::cout << "Usage:\n"
              << "  SPONGE_MANAGER\n"
              << "  SPONGE_MANAGER --config <manager.toml>\n"
              << "  SPONGE_MANAGER --fep-root <path> [--state-ids 0,1,2,3]\n"
              << "                [--lambda-lj-list 0.0,0.33,0.66,1.0]\n"
              << "                [--thermo-temperatures 300,320,340,360]\n"
              << "                [--block-steps N] [--epochs N]\n"
              << "                [--emit-output 0|1]\n"
              << "                [--remd-mode tremd|hremd|htremd|rest2]\n"
              << "                [--exchange-round N]\n";
}

int RunManagerExecution(
    const sponge::manager::ManagerExecutionConfig& execution)
{
    sponge::manager::Manager::ManagerConfig manager_config = execution.manager;
    manager_config.managed_step_limit =
        execution.manager.block_steps * execution.epochs;
    sponge::manager::Manager manager(std::move(manager_config));
    std::cout << manager.DescribePlan();
    if (execution.remd_mode.empty())
    {
        const auto results =
            manager.ExecuteAllSchedulesOnce(execution.emit_output);
        std::cout << "Executed " << results.size() << " schedule block(s)\n";
        for (const auto& result : results)
        {
            std::cout << "schedule[" << result.schedule_id
                      << "] step=" << result.observable.step
                      << " time_ps=" << result.observable.time_ps
                      << " potential=" << result.observable.potential_energy
                      << " temperature=" << result.observable.temperature
                      << " snapshot_time_ps=" << result.snapshot.current_time_ps
                      << " snapshot_potential="
                      << result.snapshot.total_potential
                      << " snapshot_temperature=" << result.snapshot.temperature
                      << " runtime_state=" << result.runtime_state.location
                      << '\n';
        }
        return 0;
    }
    sponge::manager::remd::TemperatureReplicaExchangePolicy tremd_policy;
    sponge::manager::remd::HamiltonianReplicaExchangePolicy hremd_policy;
    sponge::manager::remd::HamiltonianReplicaExchangePolicy::Config
        rest2_policy_config;
    rest2_policy_config.random_seed = 20260515u;
    rest2_policy_config.hamiltonian_key = "REST2_lambda_m";
    rest2_policy_config.hamiltonian_key_is_integer = false;
    sponge::manager::remd::HamiltonianReplicaExchangePolicy rest2_policy(
        rest2_policy_config);
    sponge::manager::remd::TemperatureHamiltonianReplicaExchangePolicy
        htremd_policy;
    for (int epoch = 0; epoch < execution.epochs; epoch++)
    {
        std::vector<sponge::manager::BlockExecutionResult> block_results;
        std::vector<sponge::manager::ExchangeAttempt> exchange_attempts;
        if (execution.remd_mode == "tremd")
        {
            const auto epoch_result = tremd_policy.ExecuteEpoch(
                &manager, execution.exchange_round + epoch,
                execution.emit_output);
            block_results = epoch_result.block_results;
            exchange_attempts = epoch_result.exchange_attempts;
            std::cout << "T-REMD epoch " << epoch << " round "
                      << epoch_result.exchange_round
                      << " blocks=" << epoch_result.block_results.size()
                      << " attempts=" << epoch_result.exchange_attempts.size()
                      << '\n';
        }
        else if (execution.remd_mode == "hremd")
        {
            const auto epoch_result = hremd_policy.ExecuteEpoch(
                &manager, execution.exchange_round + epoch,
                execution.emit_output);
            block_results = epoch_result.block_results;
            exchange_attempts = epoch_result.exchange_attempts;
            std::cout << "H-REMD epoch " << epoch << " round "
                      << epoch_result.exchange_round
                      << " blocks=" << epoch_result.block_results.size()
                      << " attempts=" << epoch_result.exchange_attempts.size()
                      << '\n';
        }
        else if (execution.remd_mode == "htremd")
        {
            const auto epoch_result = htremd_policy.ExecuteEpoch(
                &manager, execution.exchange_round + epoch,
                execution.emit_output);
            block_results = epoch_result.block_results;
            exchange_attempts = epoch_result.exchange_attempts;
            std::cout << "HT-REMD epoch " << epoch << " round "
                      << epoch_result.exchange_round
                      << " blocks=" << epoch_result.block_results.size()
                      << " attempts=" << epoch_result.exchange_attempts.size()
                      << '\n';
        }
        else if (execution.remd_mode == "rest2")
        {
            const auto epoch_result = rest2_policy.ExecuteEpoch(
                &manager, execution.exchange_round + epoch,
                execution.emit_output);
            block_results = epoch_result.block_results;
            exchange_attempts = epoch_result.exchange_attempts;
            std::cout << "REST2-REMD epoch " << epoch << " round "
                      << epoch_result.exchange_round
                      << " blocks=" << epoch_result.block_results.size()
                      << " attempts=" << epoch_result.exchange_attempts.size()
                      << '\n';
        }
        else
        {
            throw std::runtime_error("unsupported remd mode: " +
                                     execution.remd_mode);
        }
        for (const auto& result : block_results)
        {
            std::cout << "schedule[" << result.schedule_id
                      << "] step=" << result.observable.step
                      << " time_ps=" << result.observable.time_ps
                      << " potential=" << result.observable.potential_energy
                      << " temperature=" << result.observable.temperature
                      << " runtime_state=" << result.runtime_state.location
                      << " walker=" << result.runtime_state.walker_id << '\n';
        }
        for (const auto& attempt : exchange_attempts)
        {
            std::cout << "pair[" << attempt.pair_index << "] "
                      << attempt.pair.left_schedule_id << "<->"
                      << attempt.pair.right_schedule_id
                      << " log_acc=" << attempt.log_acceptance
                      << " p=" << attempt.acceptance_probability
                      << " r=" << attempt.random_value
                      << " accepted=" << (attempt.accepted ? 1 : 0) << '\n';
        }
        manager.AppendExchangeAttempts(execution.remd_mode, epoch,
                                       exchange_attempts);
        manager.AppendScheduleStates(execution.remd_mode, epoch);
        std::cout << manager.DescribePlan();
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    std::string config_path;
    std::string fep_root;
    std::vector<int> state_ids;
    std::vector<double> lambda_ladder;
    std::vector<double> thermo_temperatures;
    int block_steps = 1000;
    bool emit_output = false;
    std::string remd_mode;
    int epochs = 1;
    int exchange_round = 0;

    for (int i = 1; i < argc; i++)
    {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(std::string("missing value for ") +
                                         name);
            }
            i += 1;
            return argv[i];
        };

        if (arg == "--help" || arg == "-h")
        {
            PrintUsage();
            return 0;
        }
        if (arg == "--fep-root")
        {
            fep_root = require_value("--fep-root");
        }
        else if (arg == "--config")
        {
            config_path = require_value("--config");
        }
        else if (arg == "--state-ids")
        {
            state_ids = ParseIntList(require_value("--state-ids"));
        }
        else if (arg == "--lambda-lj-list")
        {
            lambda_ladder = ParseDoubleList(require_value("--lambda-lj-list"));
        }
        else if (arg == "--thermo-temperatures")
        {
            thermo_temperatures =
                ParseDoubleList(require_value("--thermo-temperatures"));
        }
        else if (arg == "--block-steps")
        {
            block_steps = std::stoi(require_value("--block-steps"));
        }
        else if (arg == "--epochs")
        {
            epochs = std::stoi(require_value("--epochs"));
        }
        else if (arg == "--emit-output")
        {
            emit_output = std::stoi(require_value("--emit-output")) != 0;
        }
        else if (arg == "--remd-mode")
        {
            remd_mode = require_value("--remd-mode");
        }
        else if (arg == "--exchange-round")
        {
            exchange_round = std::stoi(require_value("--exchange-round"));
        }
        else
        {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (!config_path.empty() && !fep_root.empty())
    {
        throw std::runtime_error(
            "--config and --fep-root are mutually exclusive");
    }
    if (!config_path.empty())
    {
        return RunManagerExecution(
            sponge::manager::LoadManagerExecutionConfigFromToml(config_path,
                                                                argv[0]));
    }

    if (fep_root.empty())
    {
        sponge::manager::Manager::ManagerConfig config;
        config.block_steps = 1000;
        config.exchange_log_path = "exchange.log";

        sponge::manager::ScheduleConfig schedule0;
        schedule0.schedule_id = 0;
        schedule0.name = "replica_0";
        schedule0.inputs.values["target_temperature"] = 300.0;
        schedule0.inputs.values["hamiltonian_id"] =
            static_cast<std::int64_t>(0);
        schedule0.worker.name = "worker_0";

        sponge::manager::ScheduleConfig schedule1;
        schedule1.schedule_id = 1;
        schedule1.name = "replica_1";
        schedule1.inputs.values["target_temperature"] = 320.0;
        schedule1.inputs.values["hamiltonian_id"] =
            static_cast<std::int64_t>(1);
        schedule1.worker.name = "worker_1";

        config.schedules.push_back(schedule0);
        config.schedules.push_back(schedule1);

        sponge::manager::Manager manager(config);
        std::cout << manager.DescribePlan();
        return 0;
    }

    fs::path root = fs::absolute(fep_root).lexically_normal();
    if (state_ids.empty())
    {
        state_ids = DiscoverNumericStateDirectories(root);
    }
    if (state_ids.empty())
    {
        throw std::runtime_error("no numeric state directories found under " +
                                 root.string());
    }
    if (epochs <= 0)
    {
        throw std::runtime_error("epochs must be > 0");
    }
    if (lambda_ladder.empty())
    {
        lambda_ladder = BuildDefaultLambdaLadder(state_ids.size());
    }
    if (lambda_ladder.size() != state_ids.size())
    {
        throw std::runtime_error(
            "lambda ladder size must match state directory count");
    }
    if (thermo_temperatures.empty())
    {
        thermo_temperatures.assign(state_ids.size(), 300.0);
    }
    if (thermo_temperatures.size() != state_ids.size())
    {
        throw std::runtime_error(
            "thermo temperature list size must match state directory count");
    }
    sponge::manager::ManagerExecutionConfig execution;
    execution.manager.block_steps = block_steps;
    execution.manager.exchange_log_path =
        (root / "manager_exchange.log").string();
    execution.epochs = epochs;
    execution.emit_output = emit_output;
    execution.remd_mode = remd_mode;
    execution.exchange_round = exchange_round;
    const fs::path manager_exe = fs::absolute(argv[0]).lexically_normal();
    const fs::path sponge_worker_exe = manager_exe.parent_path() / "SPONGE";

    const fs::path mdin_path = root / "step2_mdin.txt";
    for (std::size_t i = 0; i < state_ids.size(); i++)
    {
        const int state_id = state_ids[i];
        const double lambda = lambda_ladder[i];
        const fs::path state_dir = root / std::to_string(state_id);
        if (!fs::exists(state_dir))
        {
            throw std::runtime_error("state directory does not exist: " +
                                     state_dir.string());
        }

        sponge::manager::ScheduleConfig schedule;
        schedule.schedule_id = state_id;
        schedule.name = "fep_state_" + std::to_string(state_id);
        schedule.inputs.values["hamiltonian_id"] =
            static_cast<std::int64_t>(state_id);
        schedule.inputs.values["lambda_lj"] = lambda;
        schedule.inputs.values["target_temperature"] = thermo_temperatures[i];
        schedule.inputs.values["default_out_file_prefix"] =
            std::string("manager_smoke");
        schedule.worker.name = "worker_" + std::to_string(state_id);
        schedule.worker.working_directory = state_dir.string();
        schedule.worker.executable_path = sponge_worker_exe.string();
        schedule.worker.transport = "tcp";
        schedule.worker.args = {
            "-mdin",
            mdin_path.string(),
            "-workspace",
            ".",
            "-default_in_file_prefix",
            "TMP",
            "-write_information_interval",
            "1",
            "-dont_check_input",
            "1",
        };
        execution.manager.schedules.push_back(schedule);
    }
    return RunManagerExecution(execution);
}
