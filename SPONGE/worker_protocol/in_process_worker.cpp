#include <chrono>
#include <climits>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "file_protocol.h"
#include "tcp_protocol.h"
#include "worker_protocol.h"

namespace sponge::worker_protocol
{

namespace fs = std::filesystem;

namespace
{

struct ScopedCurrentPath
{
    explicit ScopedCurrentPath(const std::string& path)
        : original(fs::current_path()), changed(false)
    {
        if (!path.empty())
        {
            fs::current_path(path);
            changed = true;
        }
    }

    ~ScopedCurrentPath()
    {
        if (changed)
        {
            fs::current_path(original);
        }
    }

    fs::path original;
    bool changed;
};

std::string ShellQuote(const std::string& value)
{
    std::string quoted = "'";
    for (const char c : value)
    {
        if (c == '\'')
        {
            quoted += "'\\''";
        }
        else
        {
            quoted.push_back(c);
        }
    }
    quoted += "'";
    return quoted;
}

fs::path MakeProtocolTempPath(const std::string& prefix)
{
    const auto stamp = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    return fs::temp_directory_path() /
           (prefix + "_" + std::to_string(stamp) + ".bin");
}

void SetOrAppendArg(std::vector<std::string>* args, const std::string& key,
                    const std::string& value)
{
    if (args == nullptr)
    {
        return;
    }
    for (std::size_t i = 0; i + 1 < args->size(); i++)
    {
        if ((*args)[i] == key)
        {
            (*args)[i + 1] = value;
            return;
        }
    }
    args->push_back(key);
    args->push_back(value);
}

WorkerExecutionResponse RunChildProcessWorker(
    const sponge::manager::WorkerConfig& worker_config,
    const WorkerFileRequest& request)
{
    if (worker_config.executable_path.empty())
    {
        throw std::runtime_error(
            "ChildProcessWorkerProtocol requires executable_path");
    }
    if (worker_config.args.empty())
    {
        throw std::runtime_error(
            "ChildProcessWorkerProtocol requires non-empty args");
    }

    const fs::path request_path = MakeProtocolTempPath("sponge_worker_request");
    const fs::path response_path =
        MakeProtocolTempPath("sponge_worker_response");
    WriteWorkerFileRequest(request_path.string(), request);

    std::vector<std::string> child_args = worker_config.args;
    if (request.probe_only)
    {
        const std::string temp_prefix =
            "manager_probe_" + request_path.stem().string();
        SetOrAppendArg(&child_args, "-default_out_file_prefix", temp_prefix);
        SetOrAppendArg(&child_args, "-write_trajectory_interval", "0");
        SetOrAppendArg(&child_args, "-write_restart_file_interval", "0");
        SetOrAppendArg(&child_args, "-write_mdout_interval",
                       std::to_string(INT_MAX));
    }

    std::ostringstream command;
    if (!worker_config.working_directory.empty())
    {
        command << "cd " << ShellQuote(worker_config.working_directory)
                << " && ";
    }
    command << ShellQuote(worker_config.executable_path);
    for (const auto& arg : child_args)
    {
        command << ' ' << ShellQuote(arg);
    }
    command << " --worker-request " << ShellQuote(request_path.string())
            << " --worker-response " << ShellQuote(response_path.string());

    const int exit_code = std::system(command.str().c_str());
    if (exit_code != 0)
    {
        fs::remove(request_path);
        fs::remove(response_path);
        throw std::runtime_error("child worker command failed with exit code " +
                                 std::to_string(exit_code));
    }

    const auto response = ReadWorkerFileResponse(response_path.string());
    fs::remove(request_path);
    fs::remove(response_path);
    return response.execution;
}

WorkerExecutionResponse RunTcpChildProcessWorker(
    const sponge::manager::WorkerConfig& worker_config,
    const WorkerFileRequest& request)
{
    if (worker_config.executable_path.empty())
    {
        throw std::runtime_error(
            "ChildProcessWorkerProtocol requires executable_path");
    }
    if (worker_config.args.empty())
    {
        throw std::runtime_error(
            "ChildProcessWorkerProtocol requires non-empty args");
    }

    auto listener = TcpSocket::ListenLoopback(0);
    const int port = listener.LocalPort();

    std::vector<std::string> child_args = worker_config.args;
    if (request.probe_only)
    {
        const std::string temp_prefix =
            "manager_probe_tcp_" + std::to_string(port);
        SetOrAppendArg(&child_args, "-default_out_file_prefix", temp_prefix);
        SetOrAppendArg(&child_args, "-write_trajectory_interval", "0");
        SetOrAppendArg(&child_args, "-write_restart_file_interval", "0");
        SetOrAppendArg(&child_args, "-write_mdout_interval",
                       std::to_string(INT_MAX));
    }

    std::ostringstream command;
    if (!worker_config.working_directory.empty())
    {
        command << "cd " << ShellQuote(worker_config.working_directory)
                << " && ";
    }
    command << ShellQuote(worker_config.executable_path);
    for (const auto& arg : child_args)
    {
        command << ' ' << ShellQuote(arg);
    }
    command << " --worker-tcp "
            << ShellQuote("127.0.0.1:" + std::to_string(port));

    auto child_exit =
        std::async(std::launch::async, [command_text = command.str()]()
                   { return std::system(command_text.c_str()); });

    auto socket = listener.Accept();
    const auto hello = ReadWorkerTcpMessage(socket);
    if (hello.header.message_type != WORKER_MESSAGE_TYPE::HELLO)
    {
        throw std::runtime_error("worker TCP session did not start with HELLO");
    }

    const auto request_type = request.probe_only
                                  ? WORKER_MESSAGE_TYPE::PROBE_OBSERVABLE
                                  : WORKER_MESSAGE_TYPE::RUN_BLOCK;
    WriteWorkerTcpMessage(socket, request_type, 1,
                          SerializeWorkerRequest(request));
    const auto response_message = ReadWorkerTcpMessage(socket);
    const auto expected_response_type = request.probe_only
                                            ? WORKER_MESSAGE_TYPE::PROBE_RESULT
                                            : WORKER_MESSAGE_TYPE::RUN_RESULT;
    if (response_message.header.message_type != expected_response_type)
    {
        throw std::runtime_error(
            "worker TCP session returned unexpected "
            "message type");
    }

    WriteWorkerTcpMessage(socket, WORKER_MESSAGE_TYPE::SHUTDOWN,
                          response_message.header.request_id, "");
    const int exit_code = child_exit.get();
    if (exit_code != 0)
    {
        throw std::runtime_error("child worker command failed with exit code " +
                                 std::to_string(exit_code));
    }

    return DeserializeWorkerResponse(response_message.payload).execution;
}

}  // namespace

TcpChildProcessWorkerSession::TcpChildProcessWorkerSession(
    sponge::manager::WorkerConfig worker_config)
    : worker_config_(std::move(worker_config))
{
}

TcpChildProcessWorkerSession::~TcpChildProcessWorkerSession()
{
    try
    {
        Shutdown();
    }
    catch (...)
    {
    }
}

void TcpChildProcessWorkerSession::Start()
{
    if (started_)
    {
        return;
    }
    if (worker_config_.executable_path.empty())
    {
        throw std::runtime_error(
            "TcpChildProcessWorkerSession requires executable_path");
    }
    if (worker_config_.args.empty())
    {
        throw std::runtime_error(
            "TcpChildProcessWorkerSession requires non-empty args");
    }

    auto listener = TcpSocket::ListenLoopback(0);
    const int port = listener.LocalPort();

    std::ostringstream command;
    if (!worker_config_.working_directory.empty())
    {
        command << "cd " << ShellQuote(worker_config_.working_directory)
                << " && ";
    }
    command << ShellQuote(worker_config_.executable_path);
    for (const auto& arg : worker_config_.args)
    {
        command << ' ' << ShellQuote(arg);
    }
    command << " --worker-tcp "
            << ShellQuote("127.0.0.1:" + std::to_string(port));

    child_exit_ =
        std::async(std::launch::async, [command_text = command.str()]()
                   { return std::system(command_text.c_str()); });

    auto socket = listener.Accept();
    const auto hello = ReadWorkerTcpMessage(socket);
    if (hello.header.message_type != WORKER_MESSAGE_TYPE::HELLO)
    {
        throw std::runtime_error("worker TCP session did not start with HELLO");
    }

    socket_ = std::make_unique<TcpSocket>(std::move(socket));
    started_ = true;
}

WorkerExecutionResponse TcpChildProcessWorkerSession::SendRequest(
    const WorkerFileRequest& request)
{
    Start();
    if (socket_ == nullptr || !socket_->Valid())
    {
        throw std::runtime_error("worker TCP session is not connected");
    }

    const auto request_id = next_request_id_++;
    const auto request_type = request.probe_only
                                  ? WORKER_MESSAGE_TYPE::PROBE_OBSERVABLE
                                  : WORKER_MESSAGE_TYPE::RUN_BLOCK;
    WriteWorkerTcpMessage(*socket_, request_type, request_id,
                          SerializeWorkerRequest(request));
    const auto response_message = ReadWorkerTcpMessage(*socket_);
    const auto expected_response_type = request.probe_only
                                            ? WORKER_MESSAGE_TYPE::PROBE_RESULT
                                            : WORKER_MESSAGE_TYPE::RUN_RESULT;
    if (response_message.header.message_type != expected_response_type)
    {
        throw std::runtime_error(
            "worker TCP session returned unexpected message type");
    }
    if (response_message.header.request_id != request_id)
    {
        throw std::runtime_error(
            "worker TCP session returned unexpected request id");
    }
    return DeserializeWorkerResponse(response_message.payload).execution;
}

WorkerExecutionResponse TcpChildProcessWorkerSession::ExecuteBlock(
    int steps, bool emit_output, const sponge::RuntimeState* imported_state)
{
    WorkerFileRequest request;
    request.steps = steps;
    request.emit_output = emit_output;
    request.probe_only = false;
    request.has_runtime_state =
        imported_state != nullptr && imported_state->valid;
    if (request.has_runtime_state)
    {
        request.runtime_state = *imported_state;
    }
    return SendRequest(request);
}

void TcpChildProcessWorkerSession::Shutdown()
{
    if (shutdown_)
    {
        return;
    }
    shutdown_ = true;
    if (started_ && socket_ != nullptr && socket_->Valid())
    {
        WriteWorkerTcpMessage(*socket_, WORKER_MESSAGE_TYPE::SHUTDOWN,
                              next_request_id_++, "");
        socket_.reset();
    }
    if (child_exit_.valid())
    {
        const int exit_code = child_exit_.get();
        if (exit_code != 0)
        {
            throw std::runtime_error(
                "child worker command failed with exit code " +
                std::to_string(exit_code));
        }
    }
}

WorkerExecutionResponse InProcessWorkerProtocol::ExecuteBlock(
    const sponge::manager::WorkerConfig& worker_config, int steps,
    bool emit_output, const sponge::RuntimeState* imported_state)
{
    if (steps <= 0)
    {
        throw std::runtime_error(
            "InProcessWorkerProtocol::ExecuteBlock requires steps > 0");
    }
    if (worker_config.args.empty())
    {
        throw std::runtime_error(
            "InProcessWorkerProtocol::ExecuteBlock requires non-empty args");
    }

    sponge::SpongeScheduler scheduler;
    ScopedCurrentPath scoped_path(worker_config.working_directory);
    scheduler.InitializeFromArgs(worker_config.args);
    if (imported_state != nullptr && imported_state->valid)
    {
        scheduler.ImportRuntimeState(*imported_state);
    }
    scheduler.RunSteps(steps, emit_output);

    WorkerExecutionResponse response;
    response.runtime_state = scheduler.ExportRuntimeState();
    response.snapshot = scheduler.Snapshot();
    response.observable = scheduler.CollectExchangeObservables();
    response.finished = response.snapshot.finished;
    scheduler.Finalize();
    return response;
}

sponge::worker_protocol::WorkerExecutionResponse
ChildProcessWorkerProtocol::ExecuteBlock(
    const sponge::manager::WorkerConfig& worker_config, int steps,
    bool emit_output, const sponge::RuntimeState* imported_state)
{
    WorkerFileRequest request;
    request.steps = steps;
    request.emit_output = emit_output;
    request.probe_only = false;
    request.has_runtime_state =
        imported_state != nullptr && imported_state->valid;
    if (request.has_runtime_state)
    {
        request.runtime_state = *imported_state;
    }
    if (worker_config.persistent)
    {
        return RunTcpChildProcessWorker(worker_config, request);
    }
    return RunChildProcessWorker(worker_config, request);
}

sponge::WorkerExchangeObservable InProcessWorkerProtocol::ProbeObservable(
    const sponge::manager::WorkerConfig& worker_config,
    const sponge::RuntimeState& imported_state)
{
    if (!imported_state.valid)
    {
        throw std::runtime_error(
            "InProcessWorkerProtocol::ProbeObservable requires a valid "
            "runtime state");
    }
    if (worker_config.args.empty())
    {
        throw std::runtime_error(
            "InProcessWorkerProtocol::ProbeObservable requires non-empty args");
    }

    sponge::SpongeScheduler scheduler;
    ScopedCurrentPath scoped_path(worker_config.working_directory);
    scheduler.InitializeFromArgs(worker_config.args);
    scheduler.EnsureForeignStateProbeSafe();
    scheduler.ImportRuntimeState(imported_state);
    const auto observable = scheduler.CollectExchangeObservables();
    scheduler.Finalize();
    return observable;
}

sponge::WorkerExchangeObservable ChildProcessWorkerProtocol::ProbeObservable(
    const sponge::manager::WorkerConfig& worker_config,
    const sponge::RuntimeState& imported_state)
{
    WorkerFileRequest request;
    request.steps = 0;
    request.emit_output = false;
    request.probe_only = true;
    request.has_runtime_state = true;
    request.runtime_state = imported_state;
    if (worker_config.persistent)
    {
        return RunTcpChildProcessWorker(worker_config, request).observable;
    }
    return RunChildProcessWorker(worker_config, request).observable;
}

}  // namespace sponge::worker_protocol
