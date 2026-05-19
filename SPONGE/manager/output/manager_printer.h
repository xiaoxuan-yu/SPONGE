#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include "../config.h"
#include "../core/manager.h"

namespace sponge::manager
{

class ManagerPrinter
{
   public:
    explicit ManagerPrinter(std::ostream& out);

    void PrintStartupSummary(const ManagerExecutionConfig& execution,
                             const Manager& manager) const;
    void PrintSingleRunSummary(
        const std::vector<BlockExecutionResult>& results) const;
    void PrintEpochReport(const std::string& mode, int epoch,
                          int total_epochs, int exchange_round,
                          int block_steps,
                          const std::vector<BlockExecutionResult>& results,
                          const std::vector<ExchangeAttempt>& attempts,
                          int cumulative_accepted,
                          int cumulative_attempts) const;

   private:
    std::ostream& out_;
};

}  // namespace sponge::manager
