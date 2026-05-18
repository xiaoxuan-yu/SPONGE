#pragma once

#include <random>
#include <vector>

#include "../../core/manager.h"

namespace sponge::manager::remd
{

class TemperatureReplicaExchangePolicy
{
   public:
    struct EpochResult
    {
        int exchange_round = -1;
        std::vector<BlockExecutionResult> block_results;
        std::vector<ExchangeAttempt> exchange_attempts;
    };

    struct Config
    {
        unsigned int random_seed = 20260515u;
    };

    TemperatureReplicaExchangePolicy();
    explicit TemperatureReplicaExchangePolicy(Config config);

    std::vector<ExchangePair> BuildOddEvenPairs(
        const std::vector<ScheduleRecord>& schedules, int exchange_round) const;
    ExchangeAttempt EvaluatePair(const ScheduleRecord& left,
                                 const ScheduleRecord& right,
                                 int exchange_round, int pair_index);
    std::vector<ExchangeAttempt> AttemptOddEvenRound(Manager* manager,
                                                     int exchange_round);
    EpochResult ExecuteEpoch(Manager* manager, int exchange_round,
                             bool emit_output = false);

   private:
    std::mt19937 rng_;
};

}  // namespace sponge::manager::remd
