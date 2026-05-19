#include "manager_printer.h"

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <sstream>

namespace sponge::manager
{

namespace
{

constexpr int kLineWidth = 80;

std::string Rule(char value = '=')
{
    return std::string(kLineWidth, value);
}

std::string ModeLabel(const std::string& mode)
{
    if (mode == "tremd")
    {
        return "T-REMD";
    }
    if (mode == "hremd")
    {
        return "H-REMD";
    }
    if (mode == "htremd")
    {
        return "HT-REMD";
    }
    if (mode == "rest2")
    {
        return "REST2-REMD";
    }
    if (mode.empty())
    {
        return "MANAGER";
    }
    return mode;
}

std::string FormatInputValue(const ScheduleInputValue& value)
{
    if (const auto* integer = std::get_if<std::int64_t>(&value))
    {
        return std::to_string(*integer);
    }
    if (const auto* floating = std::get_if<double>(&value))
    {
        std::ostringstream oss;
        oss << std::setprecision(6) << *floating;
        return oss.str();
    }
    if (const auto* boolean = std::get_if<bool>(&value))
    {
        return *boolean ? "on" : "off";
    }
    return std::get<std::string>(value);
}

void AppendInputIfPresent(std::vector<std::string>* parts,
                          const ScheduleInputs& inputs,
                          const std::string& key)
{
    if (parts == nullptr)
    {
        return;
    }
    const auto it = inputs.values.find(key);
    if (it == inputs.values.end())
    {
        return;
    }
    parts->push_back(key + "=" + FormatInputValue(it->second));
}

std::string KeyParameterSummary(const ScheduleInputs& inputs,
                                const std::string& mode)
{
    std::vector<std::string> parts;
    if (mode == "rest2")
    {
        AppendInputIfPresent(&parts, inputs, "REST2_lambda_m");
        AppendInputIfPresent(&parts, inputs, "lambda_lj");
    }
    else if (mode == "tremd")
    {
        AppendInputIfPresent(&parts, inputs, "target_temperature");
    }
    else if (mode == "hremd")
    {
        AppendInputIfPresent(&parts, inputs, "hamiltonian_id");
        AppendInputIfPresent(&parts, inputs, "lambda_lj");
    }
    else if (mode == "htremd")
    {
        AppendInputIfPresent(&parts, inputs, "target_temperature");
        AppendInputIfPresent(&parts, inputs, "hamiltonian_id");
        AppendInputIfPresent(&parts, inputs, "lambda_lj");
    }
    AppendInputIfPresent(&parts, inputs, "target_pressure");
    if (parts.empty())
    {
        return "-";
    }

    std::ostringstream oss;
    for (std::size_t i = 0; i < parts.size(); i++)
    {
        if (i > 0)
        {
            oss << ", ";
        }
        oss << parts[i];
    }
    return oss.str();
}

int MaxObservedStep(const std::vector<BlockExecutionResult>& results)
{
    int step = 0;
    for (const auto& result : results)
    {
        step = std::max(step, result.observable.step);
    }
    return step;
}

int CountAccepted(const std::vector<ExchangeAttempt>& attempts)
{
    return static_cast<int>(std::count_if(
        attempts.begin(), attempts.end(),
        [](const ExchangeAttempt& attempt) { return attempt.accepted; }));
}

void PrintWorkerTable(std::ostream& out,
                      const std::vector<BlockExecutionResult>& results)
{
    out << "Schedule  Walker  Step      Time(ps)      Temp(K)      Potential"
           "      Eff.Potential\n";
    for (const auto& result : results)
    {
        out << std::left << std::setw(10) << result.schedule_id << std::setw(8)
            << result.runtime_state.walker_id << std::right << std::setw(6)
            << result.observable.step << std::setw(14) << std::fixed
            << std::setprecision(3) << result.observable.time_ps
            << std::setw(13) << std::setprecision(2)
            << result.observable.temperature << std::setw(17)
            << std::setprecision(2) << result.observable.potential_energy
            << std::setw(19) << std::setprecision(2)
            << result.observable.effective_potential << '\n';
    }
    out.unsetf(std::ios::floatfield);
}

void PrintExchangeTable(std::ostream& out,
                        const std::vector<ExchangeAttempt>& attempts)
{
    out << "\nExchange attempts\n";
    if (attempts.empty())
    {
        out << "  none\n";
        return;
    }
    out << "Pair       log(Pacc)      Pacc    Random    Result\n";
    for (const auto& attempt : attempts)
    {
        std::ostringstream pair;
        pair << attempt.pair.left_schedule_id << " <-> "
             << attempt.pair.right_schedule_id;
        out << std::left << std::setw(10) << pair.str() << std::right
            << std::setw(10) << std::fixed << std::setprecision(4)
            << attempt.log_acceptance << std::setw(10) << std::setprecision(4)
            << attempt.acceptance_probability << std::setw(10)
            << std::setprecision(4) << attempt.random_value << "    "
            << (attempt.accepted ? "ACCEPT" : "REJECT") << '\n';
    }
    out.unsetf(std::ios::floatfield);
}

std::string FirstTransport(const Manager& manager)
{
    const auto& workers = manager.workers();
    if (workers.empty())
    {
        return "-";
    }
    return workers.front().config.transport;
}

}  // namespace

ManagerPrinter::ManagerPrinter(std::ostream& out) : out_(out) {}

void ManagerPrinter::PrintStartupSummary(
    const ManagerExecutionConfig& execution, const Manager& manager) const
{
    const auto& config = manager.config();
    out_ << Rule() << '\n';
    out_ << "SPONGE_MANAGER\n";
    out_ << Rule() << '\n';
    out_ << "Mode              : " << ModeLabel(execution.remd_mode) << '\n';
    out_ << "Schedules         : " << manager.schedules().size() << '\n';
    out_ << "Block steps       : " << config.block_steps << '\n';
    out_ << "Epochs            : " << execution.epochs << '\n';
    out_ << "Transport         : " << FirstTransport(manager) << '\n';
    out_ << "Exchange log      : "
         << (config.exchange_log_path.empty() ? "-" : config.exchange_log_path)
         << '\n';
    out_ << Rule('-') << '\n';
    out_ << "Schedule  Worker      Walker  Key parameters\n";
    for (const auto& schedule : manager.schedules())
    {
        const int walker_id = schedule.runtime_state.valid
                                  ? schedule.runtime_state.walker_id
                                  : schedule.config.schedule_id;
        out_ << std::left << std::setw(10) << schedule.config.schedule_id
             << std::setw(12) << schedule.config.worker.name << std::setw(8)
             << walker_id << KeyParameterSummary(schedule.config.inputs,
                                                 execution.remd_mode)
             << '\n';
    }
    out_ << Rule() << '\n';
}

void ManagerPrinter::PrintSingleRunSummary(
    const std::vector<BlockExecutionResult>& results) const
{
    out_ << Rule('-') << '\n';
    out_ << "Single manager block finished | schedules=" << results.size()
         << " | current step=" << MaxObservedStep(results) << '\n';
    out_ << Rule('-') << '\n';
    PrintWorkerTable(out_, results);
    out_ << Rule('-') << '\n';
}

void ManagerPrinter::PrintEpochReport(
    const std::string& mode, int epoch, int total_epochs, int exchange_round,
    int block_steps, const std::vector<BlockExecutionResult>& results,
    const std::vector<ExchangeAttempt>& attempts, int cumulative_accepted,
    int cumulative_attempts) const
{
    const int accepted = CountAccepted(attempts);
    out_ << Rule('-') << '\n';
    out_ << ModeLabel(mode) << " Epoch " << (epoch + 1) << " / "
         << total_epochs << " | Round " << exchange_round
         << " | Block steps " << block_steps << " | Current step "
         << MaxObservedStep(results) << '\n';
    out_ << Rule('-') << '\n';
    PrintWorkerTable(out_, results);
    PrintExchangeTable(out_, attempts);
    out_ << Rule('-') << '\n';
    out_ << "Epoch summary: accepted " << accepted << " / "
         << attempts.size() << ", cumulative accepted " << cumulative_accepted
         << " / " << cumulative_attempts << '\n';
}

}  // namespace sponge::manager
