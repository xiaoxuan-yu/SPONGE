#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "algorithms/remd/hamiltonian_remd.h"
#include "algorithms/remd/temperature_hamiltonian_remd.h"
#include "algorithms/remd/temperature_remd.h"
#include "config.h"
#include "core/manager.h"

namespace
{

void PrintUsage()
{
    std::cout << "Usage:\n"
              << "  SPONGE_MANAGER --config <manager.toml>\n"
              << "  SPONGE_MANAGER --help\n";
}

int RunManagerExecution(
    const sponge::manager::ManagerExecutionConfig& execution)
{
    constexpr bool kEmitOutput = true;
    sponge::manager::Manager::ManagerConfig manager_config = execution.manager;
    manager_config.managed_step_limit =
        execution.manager.block_steps * execution.epochs;
    sponge::manager::Manager manager(std::move(manager_config));
    std::cout << manager.DescribePlan();
    if (execution.remd_mode.empty())
    {
        const auto results = manager.ExecuteAllSchedulesOnce(kEmitOutput);
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
                &manager, execution.exchange_round + epoch, kEmitOutput);
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
                &manager, execution.exchange_round + epoch, kEmitOutput);
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
                &manager, execution.exchange_round + epoch, kEmitOutput);
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
                &manager, execution.exchange_round + epoch, kEmitOutput);
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
    try
    {
        std::string config_path;

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
            if (arg == "--config")
            {
                if (!config_path.empty())
                {
                    throw std::runtime_error(
                        "--config can only be specified once");
                }
                config_path = require_value("--config");
            }
            else
            {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }

        if (config_path.empty())
        {
            PrintUsage();
            throw std::runtime_error(
                "missing required --config <manager.toml>");
        }
        return RunManagerExecution(
            sponge::manager::LoadManagerExecutionConfigFromToml(config_path,
                                                                argv[0]));
    }
    catch (const std::exception& error)
    {
        std::cerr << "SPONGE_MANAGER error: " << error.what() << '\n';
        return 1;
    }
}
