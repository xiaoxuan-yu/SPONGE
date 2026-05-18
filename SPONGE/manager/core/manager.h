#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../../scheduler/scheduler.h"
#include "types.h"

namespace sponge::worker_protocol
{
class WorkerSession;
}

namespace sponge::manager
{

class Manager
{
   public:
    struct ManagerConfig
    {
        int block_steps = 0;
        int managed_step_limit = 0;
        std::string exchange_log_path;
        std::vector<ScheduleConfig> schedules;
    };

    explicit Manager(ManagerConfig config);
    ~Manager();

    const ManagerConfig& config() const;
    const std::vector<ScheduleRecord>& schedules() const;
    const std::vector<WorkerHandle>& workers() const;
    const ScheduleRecord& GetSchedule(int schedule_id) const;
    std::string DescribePlan() const;
    BlockExecutionResult ExecuteScheduleBlock(const BlockExecutionPlan& plan);
    std::vector<BlockExecutionResult> ExecuteAllSchedulesOnce(
        bool emit_output = false);
    void ApplyRuntimeStateSwap(const RuntimeStateSwapOperation& operation);
    ExchangeObservable ProbeObservable(int source_schedule_id,
                                       int target_schedule_id);
    void AppendExchangeAttempts(const std::string& mode, int epoch,
                                const std::vector<ExchangeAttempt>& attempts);
    void AppendScheduleStates(const std::string& mode, int epoch);

   private:
    void BuildScheduleRecords();
    int FindScheduleIndex(int schedule_id) const;
    sponge::worker_protocol::WorkerSession& GetWorkerSession(
        int schedule_index);

    ManagerConfig config_;
    std::vector<ScheduleRecord> schedules_;
    std::vector<WorkerHandle> workers_;
    std::vector<sponge::RuntimeState> runtime_state_buffers_;
    std::vector<std::unique_ptr<sponge::worker_protocol::WorkerSession>>
        worker_sessions_;
};

}  // namespace sponge::manager
