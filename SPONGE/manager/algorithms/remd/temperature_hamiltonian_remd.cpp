#include "temperature_hamiltonian_remd.h"

#include <cmath>
#include <stdexcept>

#include "../../../common.h"

namespace sponge::manager::remd
{

namespace
{

double ComputeBeta(double temperature_kelvin)
{
    if (temperature_kelvin <= 0.0)
    {
        throw std::runtime_error(
            "TemperatureHamiltonianReplicaExchangePolicy requires "
            "temperature > 0");
    }
    return 1.0 / (static_cast<double>(CONSTANT_kB) * temperature_kelvin);
}

double RequireInputDouble(const ScheduleRecord& schedule, const char* key)
{
    const auto value = schedule.config.inputs.FindDouble(key);
    if (!value.has_value())
    {
        throw std::runtime_error(
            "TemperatureHamiltonianReplicaExchangePolicy requires "
            "schedules.inputs." +
            std::string(key));
    }
    return *value;
}

void RequireHamiltonianId(const ScheduleRecord& schedule)
{
    if (!schedule.config.inputs.FindInt("hamiltonian_id").has_value())
    {
        throw std::runtime_error(
            "TemperatureHamiltonianReplicaExchangePolicy requires "
            "schedules.inputs.hamiltonian_id");
    }
}

}  // namespace

TemperatureHamiltonianReplicaExchangePolicy::
    TemperatureHamiltonianReplicaExchangePolicy()
    : TemperatureHamiltonianReplicaExchangePolicy(Config{})
{
}

TemperatureHamiltonianReplicaExchangePolicy::
    TemperatureHamiltonianReplicaExchangePolicy(Config config)
    : rng_(config.random_seed)
{
}

std::vector<ExchangePair>
TemperatureHamiltonianReplicaExchangePolicy::BuildOddEvenPairs(
    const std::vector<ScheduleRecord>& schedules, int exchange_round) const
{
    std::vector<ExchangePair> pairs;
    const int parity = exchange_round % 2;
    for (int i = parity; i + 1 < static_cast<int>(schedules.size()); i += 2)
    {
        ExchangePair pair;
        pair.left_schedule_id = schedules[i].config.schedule_id;
        pair.right_schedule_id = schedules[i + 1].config.schedule_id;
        pairs.push_back(pair);
    }
    return pairs;
}

ExchangeAttempt TemperatureHamiltonianReplicaExchangePolicy::EvaluatePair(
    Manager* manager, const ScheduleRecord& left, const ScheduleRecord& right,
    int exchange_round, int pair_index)
{
    if (manager == nullptr)
    {
        throw std::runtime_error(
            "TemperatureHamiltonianReplicaExchangePolicy requires a non-null "
            "manager");
    }
    if (left.last_observable.step < 0 || right.last_observable.step < 0)
    {
        throw std::runtime_error(
            "TemperatureHamiltonianReplicaExchangePolicy requires valid "
            "observables");
    }

    const auto left_under_right = manager->ProbeObservable(
        left.config.schedule_id, right.config.schedule_id);
    const auto right_under_left = manager->ProbeObservable(
        right.config.schedule_id, left.config.schedule_id);

    RequireHamiltonianId(left);
    RequireHamiltonianId(right);
    const double temperature_left =
        RequireInputDouble(left, "target_temperature");
    const double temperature_right =
        RequireInputDouble(right, "target_temperature");
    const double beta_left = ComputeBeta(temperature_left);
    const double beta_right = ComputeBeta(temperature_right);

    ExchangeAttempt attempt;
    attempt.exchange_round = exchange_round;
    attempt.pair_index = pair_index;
    attempt.pair.left_schedule_id = left.config.schedule_id;
    attempt.pair.right_schedule_id = right.config.schedule_id;
    attempt.log_acceptance =
        -(beta_left * right_under_left.potential_energy +
          beta_right * left_under_right.potential_energy -
          beta_left * left.last_observable.potential_energy -
          beta_right * right.last_observable.potential_energy);
    attempt.acceptance_probability =
        attempt.log_acceptance >= 0.0
            ? 1.0
            : std::exp(std::max(-700.0, attempt.log_acceptance));

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    attempt.random_value = uniform(rng_);
    attempt.accepted = attempt.log_acceptance >= 0.0 ||
                       attempt.random_value < attempt.acceptance_probability;
    return attempt;
}

std::vector<ExchangeAttempt>
TemperatureHamiltonianReplicaExchangePolicy::AttemptOddEvenRound(
    Manager* manager, int exchange_round)
{
    if (manager == nullptr)
    {
        throw std::runtime_error(
            "TemperatureHamiltonianReplicaExchangePolicy requires a non-null "
            "manager");
    }

    std::vector<ExchangeAttempt> attempts;
    const auto pairs = BuildOddEvenPairs(manager->schedules(), exchange_round);
    attempts.reserve(pairs.size());
    for (int pair_index = 0; pair_index < static_cast<int>(pairs.size());
         pair_index++)
    {
        const auto& pair = pairs[pair_index];
        const auto& left = manager->GetSchedule(pair.left_schedule_id);
        const auto& right = manager->GetSchedule(pair.right_schedule_id);
        ExchangeAttempt attempt =
            EvaluatePair(manager, left, right, exchange_round, pair_index);
        if (attempt.accepted)
        {
            RuntimeStateSwapOperation swap;
            swap.schedule_id_a = pair.left_schedule_id;
            swap.schedule_id_b = pair.right_schedule_id;
            const double temperature_left =
                RequireInputDouble(left, "target_temperature");
            const double temperature_right =
                RequireInputDouble(right, "target_temperature");
            swap.scale_a_to_b = std::sqrt(temperature_right / temperature_left);
            swap.scale_b_to_a = std::sqrt(temperature_left / temperature_right);
            manager->ApplyRuntimeStateSwap(swap);
        }
        attempts.push_back(attempt);
    }
    return attempts;
}

TemperatureHamiltonianReplicaExchangePolicy::EpochResult
TemperatureHamiltonianReplicaExchangePolicy::ExecuteEpoch(Manager* manager,
                                                          int exchange_round,
                                                          bool emit_output)
{
    if (manager == nullptr)
    {
        throw std::runtime_error(
            "TemperatureHamiltonianReplicaExchangePolicy requires a non-null "
            "manager");
    }
    EpochResult result;
    result.exchange_round = exchange_round;
    result.block_results = manager->ExecuteAllSchedulesOnce(emit_output);
    result.exchange_attempts = AttemptOddEvenRound(manager, exchange_round);
    return result;
}

}  // namespace sponge::manager::remd
