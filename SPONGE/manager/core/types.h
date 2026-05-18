#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "../../scheduler/scheduler.h"

namespace sponge::manager
{

using ScheduleInputValue =
    std::variant<std::int64_t, double, bool, std::string>;

struct ScheduleInputs
{
    std::unordered_map<std::string, ScheduleInputValue> values;

    bool Has(const std::string& key) const
    {
        return values.find(key) != values.end();
    }

    std::optional<double> FindDouble(const std::string& key) const
    {
        const auto it = values.find(key);
        if (it == values.end())
        {
            return std::nullopt;
        }
        if (const auto* value = std::get_if<double>(&it->second))
        {
            return *value;
        }
        if (const auto* value = std::get_if<std::int64_t>(&it->second))
        {
            return static_cast<double>(*value);
        }
        return std::nullopt;
    }

    std::optional<int> FindInt(const std::string& key) const
    {
        const auto it = values.find(key);
        if (it == values.end())
        {
            return std::nullopt;
        }
        if (const auto* value = std::get_if<std::int64_t>(&it->second))
        {
            return static_cast<int>(*value);
        }
        return std::nullopt;
    }

    std::optional<std::string> FindString(const std::string& key) const
    {
        const auto it = values.find(key);
        if (it == values.end())
        {
            return std::nullopt;
        }
        if (const auto* value = std::get_if<std::string>(&it->second))
        {
            return *value;
        }
        return std::nullopt;
    }
};

enum class RuntimeStateBackend
{
    kUnknown,
    kMemoryImage,
    kRestartFile,
};

struct RuntimeStateRef
{
    int walker_id = -1;
    RuntimeStateBackend backend = RuntimeStateBackend::kUnknown;
    std::string location;
    int step = -1;
    double time_ps = 0.0;
    bool valid = false;
};

struct ExchangeObservable
{
    int schedule_id = -1;
    int step = -1;
    double time_ps = 0.0;
    double potential_energy = 0.0;
    double effective_potential = 0.0;
    double volume = 0.0;
    double temperature = 0.0;
};

struct WorkerConfig
{
    std::string name;
    std::vector<std::string> args;
    std::string working_directory;
    std::string executable_path;
    std::string transport = "file";
    bool child_process = false;
    bool persistent = false;
};

struct ScheduleConfig
{
    int schedule_id = -1;
    std::string name;
    ScheduleInputs inputs;
    WorkerConfig worker;
};

struct ScheduleRecord
{
    ScheduleConfig config;
    RuntimeStateRef runtime_state;
    ExchangeObservable last_observable;
    int accepted_exchange_count = 0;
};

struct WorkerHandle
{
    WorkerConfig config;
    bool initialized = false;
    bool has_runtime_state = false;
};

struct BlockExecutionPlan
{
    int schedule_id = -1;
    int steps = 0;
    bool emit_output = false;
};

struct BlockExecutionResult
{
    int schedule_id = -1;
    int requested_steps = 0;
    bool emit_output = false;
    bool executed = false;
    bool finished = false;
    sponge::SchedulerSnapshot snapshot;
    RuntimeStateRef runtime_state;
    ExchangeObservable observable;
};

struct ExchangePair
{
    int left_schedule_id = -1;
    int right_schedule_id = -1;
};

struct ExchangeAttempt
{
    int exchange_round = -1;
    int pair_index = -1;
    ExchangePair pair;
    double log_acceptance = 0.0;
    double acceptance_probability = 0.0;
    double random_value = 0.0;
    bool accepted = false;
};

struct RuntimeStateSwapOperation
{
    int schedule_id_a = -1;
    int schedule_id_b = -1;
    double scale_a_to_b = 1.0;
    double scale_b_to_a = 1.0;
};

}  // namespace sponge::manager
