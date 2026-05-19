#pragma once

#include <random>
#include <string>
#include <vector>

#include "../../core/manager.h"

namespace sponge::manager::remd
{

class HamiltonianReplicaExchangePolicy
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
        std::string hamiltonian_key = "hamiltonian_id";
        bool hamiltonian_key_is_integer = true;
    };

    HamiltonianReplicaExchangePolicy();
    explicit HamiltonianReplicaExchangePolicy(Config config);

    std::vector<ExchangePair> BuildOddEvenPairs(
        const std::vector<ScheduleRecord>& schedules, int exchange_round) const;
    ExchangeAttempt EvaluatePair(Manager* manager, const ScheduleRecord& left,
                                 const ScheduleRecord& right,
                                 int exchange_round, int pair_index);
    std::vector<ExchangeAttempt> AttemptOddEvenRound(Manager* manager,
                                                     int exchange_round);
    EpochResult ExecuteEpoch(Manager* manager, int exchange_round,
                             bool emit_output = false);

   private:
    Config config_;
    std::mt19937 rng_;
};

}  // namespace sponge::manager::remd
