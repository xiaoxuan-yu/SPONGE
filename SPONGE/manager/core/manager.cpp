#include "manager.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <initializer_list>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "../../worker_protocol/worker_protocol.h"

namespace sponge::manager
{

namespace fs = std::filesystem;

namespace
{

const char* RuntimeStateBackendName(RuntimeStateBackend backend)
{
    switch (backend)
    {
        case RuntimeStateBackend::kUnknown:
            return "unknown";
        case RuntimeStateBackend::kMemoryImage:
            return "memory";
        case RuntimeStateBackend::kRestartFile:
            return "restart-file";
    }
    return "unknown";
}

std::string InputValueToString(const ScheduleInputValue& value)
{
    if (const auto* integer = std::get_if<std::int64_t>(&value))
    {
        return std::to_string(*integer);
    }
    if (const auto* floating = std::get_if<double>(&value))
    {
        std::ostringstream oss;
        oss << std::setprecision(12) << *floating;
        return oss.str();
    }
    if (const auto* boolean = std::get_if<bool>(&value))
    {
        return *boolean ? "1" : "0";
    }
    return std::get<std::string>(value);
}

bool IsManagerOnlyInput(const std::string& key)
{
    return key == "hamiltonian_id";
}

std::string SchedulePrefix(int schedule_id, const ScheduleInputs& inputs)
{
    const std::string base_prefix =
        inputs.FindString("default_out_file_prefix").value_or("mdout");
    return base_prefix + "_" + std::to_string(schedule_id);
}

void AppendWorkerInputArgs(ScheduleConfig* schedule)
{
    if (schedule == nullptr)
    {
        return;
    }
    schedule->inputs.values["default_out_file_prefix"] =
        SchedulePrefix(schedule->schedule_id, schedule->inputs);

    std::vector<std::string> keys;
    keys.reserve(schedule->inputs.values.size());
    for (const auto& item : schedule->inputs.values)
    {
        keys.push_back(item.first);
    }
    std::sort(keys.begin(), keys.end());
    for (const auto& key : keys)
    {
        if (IsManagerOnlyInput(key))
        {
            continue;
        }
        schedule->worker.args.push_back("-" + key);
        schedule->worker.args.push_back(
            InputValueToString(schedule->inputs.values.at(key)));
    }
}

void RemoveWorkerArg(std::vector<std::string>* args, const std::string& key)
{
    if (args == nullptr)
    {
        return;
    }
    std::vector<std::string> filtered;
    filtered.reserve(args->size());
    for (std::size_t i = 0; i < args->size(); i++)
    {
        if ((*args)[i] == key)
        {
            if (i + 1 < args->size())
            {
                i += 1;
            }
            continue;
        }
        filtered.push_back((*args)[i]);
    }
    *args = std::move(filtered);
}

void ApplyManagerControlledArgs(ScheduleConfig* schedule,
                                int managed_step_limit)
{
    if (schedule == nullptr || managed_step_limit <= 0)
    {
        return;
    }
    RemoveWorkerArg(&schedule->worker.args, "-step_limit");
    schedule->worker.managed_step_limit = managed_step_limit;
}

void EnsureWorkerWorkingDirectory(const WorkerConfig& worker)
{
    if (worker.working_directory.empty())
    {
        return;
    }
    const fs::path directory(worker.working_directory);
    std::error_code error;
    fs::create_directories(directory, error);
    if (error)
    {
        throw std::runtime_error(
            "failed to create worker directory: " + directory.string() + " (" +
            error.message() + ")");
    }
    if (!fs::is_directory(directory))
    {
        throw std::runtime_error("worker path is not a directory: " +
                                 directory.string());
    }
}

ExchangeObservable MakeExchangeObservable(
    int schedule_id, const sponge::WorkerExchangeObservable& observable)
{
    ExchangeObservable result;
    result.schedule_id = schedule_id;
    result.step = observable.step;
    result.time_ps = observable.time_ps;
    result.potential_energy = observable.total_potential;
    result.effective_potential = observable.effective_potential;
    result.volume = observable.volume;
    result.temperature = observable.temperature;
    return result;
}

void ValidateInputMatchesObservable(const ScheduleConfig& schedule,
                                    const std::string& key, double actual)
{
    const auto expected = schedule.inputs.FindDouble(key);
    if (!expected.has_value())
    {
        return;
    }
    const double tolerance = std::max(1.0e-3, std::abs(*expected) * 1.0e-5);
    if (std::abs(*expected - actual) > tolerance)
    {
        std::ostringstream oss;
        oss << "worker-reported " << key << " for schedule "
            << schedule.schedule_id << " is " << actual
            << ", but schedules.inputs." << key << " is " << *expected;
        throw std::runtime_error(oss.str());
    }
}

void ValidateWorkerObservableInputs(
    const ScheduleConfig& schedule,
    const sponge::WorkerExchangeObservable& observable)
{
    ValidateInputMatchesObservable(schedule, "target_temperature",
                                   observable.target_temperature);
    ValidateInputMatchesObservable(schedule, "target_pressure",
                                   observable.target_pressure);
}

RuntimeStateRef MakeRuntimeStateRef(int schedule_id,
                                    const sponge::RuntimeState& state,
                                    int walker_id)
{
    RuntimeStateRef reference;
    reference.walker_id = walker_id;
    reference.backend = RuntimeStateBackend::kMemoryImage;
    reference.location =
        "manager-memory:schedule_" + std::to_string(schedule_id);
    reference.step = state.step;
    reference.time_ps = state.current_time_ps;
    reference.valid = state.valid;
    return reference;
}

void ScaleRuntimeStateVelocities(sponge::RuntimeState* state, double factor)
{
    if (state == nullptr || !state->valid || factor == 1.0)
    {
        return;
    }
    for (auto& velocity : state->velocities)
    {
        velocity.x *= factor;
        velocity.y *= factor;
        velocity.z *= factor;
    }
}

void Write_Exchange_Log_Header(std::ofstream* out)
{
    if (out == nullptr) return;
    (*out) << "record_type,mode,epoch,exchange_round,pair_index,schedule_id,"
              "walker_id,left_schedule,right_schedule,step,time_ps,"
              "potential_energy,effective_potential,temperature,volume,"
              "log_acceptance,acceptance_probability,random_value,accepted\n";
}

std::string DoubleToLogString(double value)
{
    std::ostringstream oss;
    oss << std::setprecision(12) << value;
    return oss.str();
}

std::string IntToLogString(int value) { return std::to_string(value); }

void WriteCsvRow(std::ofstream* out, std::initializer_list<std::string> fields)
{
    if (out == nullptr) return;
    bool first = true;
    for (const auto& field : fields)
    {
        if (!first)
        {
            (*out) << ',';
        }
        (*out) << field;
        first = false;
    }
    (*out) << '\n';
}

bool CanRunBlockInParallel(const std::vector<WorkerHandle>& workers)
{
    return !workers.empty();
}

}  // namespace

Manager::Manager(ManagerConfig config) : config_(std::move(config))
{
    BuildScheduleRecords();
}

Manager::~Manager() = default;

const Manager::ManagerConfig& Manager::config() const { return config_; }

const std::vector<ScheduleRecord>& Manager::schedules() const
{
    return schedules_;
}

const std::vector<WorkerHandle>& Manager::workers() const { return workers_; }

const ScheduleRecord& Manager::GetSchedule(int schedule_id) const
{
    return schedules_.at(FindScheduleIndex(schedule_id));
}

void Manager::BuildScheduleRecords()
{
    schedules_.clear();
    workers_.clear();
    runtime_state_buffers_.clear();
    worker_sessions_.clear();
    schedules_.reserve(config_.schedules.size());
    workers_.reserve(config_.schedules.size());
    runtime_state_buffers_.resize(config_.schedules.size());
    worker_sessions_.resize(config_.schedules.size());
    for (const auto& schedule : config_.schedules)
    {
        ScheduleRecord record;
        record.config = schedule;
        AppendWorkerInputArgs(&record.config);
        ApplyManagerControlledArgs(&record.config, config_.managed_step_limit);
        EnsureWorkerWorkingDirectory(record.config.worker);
        schedules_.push_back(record);

        WorkerHandle worker;
        worker.config = record.config.worker;
        workers_.push_back(worker);
    }
}

int Manager::FindScheduleIndex(int schedule_id) const
{
    for (int i = 0; i < static_cast<int>(schedules_.size()); i++)
    {
        if (schedules_[i].config.schedule_id == schedule_id)
        {
            return i;
        }
    }
    throw std::runtime_error("Manager::FindScheduleIndex failed for schedule " +
                             std::to_string(schedule_id));
}

sponge::worker_protocol::WorkerSession& Manager::GetWorkerSession(
    int schedule_index)
{
    auto& session = worker_sessions_.at(schedule_index);
    if (session == nullptr)
    {
        session = sponge::worker_protocol::CreateWorkerSession(
            workers_.at(schedule_index).config);
    }
    return *session;
}

BlockExecutionResult Manager::ExecuteScheduleBlock(
    const BlockExecutionPlan& plan)
{
    const int schedule_index = FindScheduleIndex(plan.schedule_id);
    auto& schedule = schedules_[schedule_index];
    auto& worker = workers_[schedule_index];
    auto& runtime_state = runtime_state_buffers_[schedule_index];

    if (plan.steps <= 0)
    {
        throw std::runtime_error(
            "Manager::ExecuteScheduleBlock requires steps > 0");
    }
    if (worker.config.args.empty())
    {
        throw std::runtime_error(
            "Manager::ExecuteScheduleBlock requires non-empty worker args");
    }

    worker.initialized = true;
    const sponge::RuntimeState* imported_state =
        (worker.has_runtime_state && runtime_state.valid) ? &runtime_state
                                                          : nullptr;
    const auto response =
        GetWorkerSession(schedule_index)
            .RunBlock(plan.steps, plan.emit_output, imported_state);
    ValidateWorkerObservableInputs(schedule.config, response.observable);
    runtime_state = response.runtime_state;
    worker.has_runtime_state = runtime_state.valid;
    const int walker_id = schedule.runtime_state.valid
                              ? schedule.runtime_state.walker_id
                              : schedule.config.schedule_id;
    schedule.runtime_state = MakeRuntimeStateRef(schedule.config.schedule_id,
                                                 runtime_state, walker_id);
    schedule.last_observable = MakeExchangeObservable(
        schedule.config.schedule_id, response.observable);

    BlockExecutionResult result;
    result.schedule_id = schedule.config.schedule_id;
    result.requested_steps = plan.steps;
    result.emit_output = plan.emit_output;
    result.executed = true;
    result.finished = response.finished;
    result.snapshot = response.snapshot;
    result.runtime_state = schedule.runtime_state;
    result.observable = schedule.last_observable;
    return result;
}

std::vector<BlockExecutionResult> Manager::ExecuteAllSchedulesOnce(
    bool emit_output)
{
    std::vector<BlockExecutionResult> results;
    results.reserve(schedules_.size());
    if (CanRunBlockInParallel(workers_))
    {
        struct PendingBlock
        {
            int schedule_index = -1;
            BlockExecutionPlan plan;
            std::future<sponge::worker_protocol::WorkerExecutionResponse>
                response;
        };

        std::vector<PendingBlock> pending_blocks;
        pending_blocks.reserve(schedules_.size());
        for (int i = 0; i < static_cast<int>(schedules_.size()); i++)
        {
            auto& worker = workers_[i];
            const auto& runtime_state = runtime_state_buffers_[i];
            if (config_.block_steps <= 0)
            {
                throw std::runtime_error(
                    "Manager::ExecuteAllSchedulesOnce requires block_steps > "
                    "0");
            }
            if (worker.config.args.empty())
            {
                throw std::runtime_error(
                    "Manager::ExecuteAllSchedulesOnce requires non-empty "
                    "worker args");
            }

            BlockExecutionPlan plan;
            plan.schedule_id = schedules_[i].config.schedule_id;
            plan.steps = config_.block_steps;
            plan.emit_output = emit_output;
            const bool has_imported_state =
                worker.has_runtime_state && runtime_state.valid;
            const sponge::RuntimeState imported_state = runtime_state;
            auto* worker_session = &GetWorkerSession(i);
            worker.initialized = true;

            PendingBlock pending;
            pending.schedule_index = i;
            pending.plan = plan;
            pending.response = std::async(
                std::launch::async,
                [worker_session, plan, has_imported_state, imported_state]()
                {
                    const sponge::RuntimeState* imported_state_ptr =
                        has_imported_state ? &imported_state : nullptr;
                    return worker_session->RunBlock(
                        plan.steps, plan.emit_output, imported_state_ptr);
                });
            pending_blocks.push_back(std::move(pending));
        }

        for (auto& pending : pending_blocks)
        {
            const auto response = pending.response.get();
            const int schedule_index = pending.schedule_index;
            auto& schedule = schedules_[schedule_index];
            auto& worker = workers_[schedule_index];
            auto& runtime_state = runtime_state_buffers_[schedule_index];
            ValidateWorkerObservableInputs(schedule.config,
                                           response.observable);
            runtime_state = response.runtime_state;
            worker.has_runtime_state = runtime_state.valid;
            const int walker_id = schedule.runtime_state.valid
                                      ? schedule.runtime_state.walker_id
                                      : schedule.config.schedule_id;
            schedule.runtime_state = MakeRuntimeStateRef(
                schedule.config.schedule_id, runtime_state, walker_id);
            schedule.last_observable = MakeExchangeObservable(
                schedule.config.schedule_id, response.observable);

            BlockExecutionResult result;
            result.schedule_id = schedule.config.schedule_id;
            result.requested_steps = pending.plan.steps;
            result.emit_output = pending.plan.emit_output;
            result.executed = true;
            result.finished = response.finished;
            result.snapshot = response.snapshot;
            result.runtime_state = schedule.runtime_state;
            result.observable = schedule.last_observable;
            results.push_back(result);
        }
        return results;
    }

    for (const auto& schedule : schedules_)
    {
        BlockExecutionPlan plan;
        plan.schedule_id = schedule.config.schedule_id;
        plan.steps = config_.block_steps;
        plan.emit_output = emit_output;
        results.push_back(ExecuteScheduleBlock(plan));
    }
    return results;
}

void Manager::ApplyRuntimeStateSwap(const RuntimeStateSwapOperation& operation)
{
    const int index_a = FindScheduleIndex(operation.schedule_id_a);
    const int index_b = FindScheduleIndex(operation.schedule_id_b);
    auto& state_a = runtime_state_buffers_.at(index_a);
    auto& state_b = runtime_state_buffers_.at(index_b);
    auto& schedule_a = schedules_.at(index_a);
    auto& schedule_b = schedules_.at(index_b);
    auto& worker_a = workers_.at(index_a);
    auto& worker_b = workers_.at(index_b);

    if (!state_a.valid || !state_b.valid)
    {
        throw std::runtime_error(
            "Manager::ApplyRuntimeStateSwap requires valid runtime states");
    }

    sponge::RuntimeState outgoing_a = state_a;
    sponge::RuntimeState outgoing_b = state_b;
    ScaleRuntimeStateVelocities(&outgoing_a, operation.scale_a_to_b);
    ScaleRuntimeStateVelocities(&outgoing_b, operation.scale_b_to_a);

    state_a = std::move(outgoing_b);
    state_b = std::move(outgoing_a);

    const int walker_id_a = schedule_a.runtime_state.valid
                                ? schedule_a.runtime_state.walker_id
                                : schedule_a.config.schedule_id;
    const int walker_id_b = schedule_b.runtime_state.valid
                                ? schedule_b.runtime_state.walker_id
                                : schedule_b.config.schedule_id;

    schedule_a.runtime_state = MakeRuntimeStateRef(
        schedule_a.config.schedule_id, state_a, walker_id_b);
    schedule_b.runtime_state = MakeRuntimeStateRef(
        schedule_b.config.schedule_id, state_b, walker_id_a);
    worker_a.has_runtime_state = state_a.valid;
    worker_b.has_runtime_state = state_b.valid;
}

ExchangeObservable Manager::ProbeObservable(int source_schedule_id,
                                            int target_schedule_id)
{
    const int source_index = FindScheduleIndex(source_schedule_id);
    const int target_index = FindScheduleIndex(target_schedule_id);
    const auto& source_state = runtime_state_buffers_.at(source_index);
    const auto& target_schedule = schedules_.at(target_index);
    const auto& target_worker = workers_.at(target_index);

    if (!source_state.valid)
    {
        throw std::runtime_error(
            "Manager::ProbeObservable requires a valid source runtime state");
    }
    if (target_worker.config.args.empty())
    {
        throw std::runtime_error(
            "Manager::ProbeObservable requires non-empty target worker args");
    }

    const auto observable =
        GetWorkerSession(target_index).ProbeObservable(source_state);
    return MakeExchangeObservable(source_schedule_id, observable);
}

std::string Manager::DescribePlan() const
{
    std::ostringstream oss;
    oss << "SPONGE manager core skeleton\n";
    oss << "block_steps=" << config_.block_steps << '\n';
    oss << "schedules=" << schedules_.size() << '\n';
    if (!config_.exchange_log_path.empty())
    {
        oss << "exchange_log=" << config_.exchange_log_path << '\n';
    }
    for (const auto& schedule : schedules_)
    {
        oss << "schedule[" << schedule.config.schedule_id << "] ";
        if (!schedule.config.name.empty())
        {
            oss << schedule.config.name << ' ';
        }
        std::vector<std::string> input_keys;
        input_keys.reserve(schedule.config.inputs.values.size());
        for (const auto& input : schedule.config.inputs.values)
        {
            input_keys.push_back(input.first);
        }
        std::sort(input_keys.begin(), input_keys.end());
        for (const auto& key : input_keys)
        {
            oss << key << '='
                << InputValueToString(schedule.config.inputs.values.at(key))
                << ' ';
        }
        if (!schedule.config.worker.name.empty())
        {
            oss << " worker=" << schedule.config.worker.name;
        }
        oss << " transport=" << schedule.config.worker.transport;
        if (!schedule.runtime_state.location.empty())
        {
            oss << " runtime_state="
                << RuntimeStateBackendName(schedule.runtime_state.backend)
                << ':' << schedule.runtime_state.location;
            if (schedule.runtime_state.walker_id >= 0)
            {
                oss << " walker=" << schedule.runtime_state.walker_id;
            }
        }
        if (schedule.last_observable.step > 0)
        {
            oss << " last_step=" << schedule.last_observable.step;
        }
        oss << '\n';
    }
    return oss.str();
}

void Manager::AppendExchangeAttempts(
    const std::string& mode, int epoch,
    const std::vector<ExchangeAttempt>& attempts)
{
    if (config_.exchange_log_path.empty())
    {
        return;
    }
    const fs::path path(config_.exchange_log_path);
    const bool write_header = !fs::exists(path) || fs::file_size(path) == 0;
    std::ofstream out(path, std::ios::app);
    if (!out)
    {
        throw std::runtime_error("failed to open exchange log: " +
                                 path.string());
    }
    if (write_header)
    {
        Write_Exchange_Log_Header(&out);
    }
    for (const auto& attempt : attempts)
    {
        WriteCsvRow(&out,
                    {"exchange_attempt", mode, IntToLogString(epoch),
                     IntToLogString(attempt.exchange_round),
                     IntToLogString(attempt.pair_index), "", "",
                     IntToLogString(attempt.pair.left_schedule_id),
                     IntToLogString(attempt.pair.right_schedule_id), "", "", "",
                     "", "", "", DoubleToLogString(attempt.log_acceptance),
                     DoubleToLogString(attempt.acceptance_probability),
                     DoubleToLogString(attempt.random_value),
                     attempt.accepted ? "1" : "0"});
    }
}

void Manager::AppendScheduleStates(const std::string& mode, int epoch)
{
    if (config_.exchange_log_path.empty())
    {
        return;
    }
    const fs::path path(config_.exchange_log_path);
    const bool write_header = !fs::exists(path) || fs::file_size(path) == 0;
    std::ofstream out(path, std::ios::app);
    if (!out)
    {
        throw std::runtime_error("failed to open exchange log: " +
                                 path.string());
    }
    if (write_header)
    {
        Write_Exchange_Log_Header(&out);
    }
    for (const auto& schedule : schedules_)
    {
        WriteCsvRow(
            &out,
            {"schedule_state", mode, IntToLogString(epoch), "", "",
             IntToLogString(schedule.config.schedule_id),
             IntToLogString(schedule.runtime_state.walker_id), "", "",
             IntToLogString(schedule.last_observable.step),
             DoubleToLogString(schedule.last_observable.time_ps),
             DoubleToLogString(schedule.last_observable.potential_energy),
             DoubleToLogString(schedule.last_observable.effective_potential),
             DoubleToLogString(schedule.last_observable.temperature),
             DoubleToLogString(schedule.last_observable.volume), "", "", "",
             ""});
    }
}

}  // namespace sponge::manager
