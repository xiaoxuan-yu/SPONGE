#include <chrono>
#include <climits>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

#include "file_protocol.h"
#include "shm_transport.h"
#include "tcp_protocol.h"
#include "worker_protocol.h"

namespace sponge::worker_protocol
{

namespace fs = std::filesystem;

namespace
{

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

fs::path MakeProtocolTempDirectory(const std::string& prefix)
{
    const auto stamp = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    fs::path directory =
        fs::temp_directory_path() / (prefix + "_" + std::to_string(stamp));
    fs::create_directories(directory);
    return directory;
}

void TouchFile(const fs::path& path)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        throw std::runtime_error("failed to create file: " + path.string());
    }
}

void WaitForFile(const fs::path& path, const std::future<int>* child_exit,
                 const std::string& context)
{
    while (!fs::exists(path))
    {
        if (child_exit != nullptr && child_exit->valid() &&
            child_exit->wait_for(std::chrono::milliseconds(0)) ==
                std::future_status::ready)
        {
            throw std::runtime_error("child worker exited while waiting for " +
                                     context);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

fs::path ManagerWorkerLogPath(
    const sponge::manager::WorkerConfig& worker_config)
{
    const fs::path directory = worker_config.working_directory.empty()
                                   ? fs::current_path()
                                   : fs::path(worker_config.working_directory);
    const std::string worker_name =
        worker_config.name.empty() ? "worker" : worker_config.name;
    return directory / (worker_name + "_manager.log");
}

void AppendManagerOutputRedirection(
    std::ostringstream* command,
    const sponge::manager::WorkerConfig& worker_config)
{
    if (command == nullptr)
    {
        return;
    }
    (*command) << " >> "
               << ShellQuote(ManagerWorkerLogPath(worker_config).string())
               << " 2>&1";
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
    AppendManagerOutputRedirection(&command, worker_config);

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
    const bool use_shared_memory_payloads = worker_config.transport == "shm";
    if (use_shared_memory_payloads)
    {
        command << " --worker-transport " << ShellQuote("shm");
    }
    command << " --worker-tcp "
            << ShellQuote("127.0.0.1:" + std::to_string(port));
    AppendManagerOutputRedirection(&command, worker_config);

    auto child_exit =
        std::async(std::launch::async, [command_text = command.str()]()
                   { return std::system(command_text.c_str()); });

    auto transport = CreateTcpControlTransport(listener.Accept(),
                                               use_shared_memory_payloads);
    const auto hello = transport->Receive();
    if (hello.type != WORKER_MESSAGE_TYPE::HELLO)
    {
        throw std::runtime_error("worker TCP session did not start with HELLO");
    }

    const auto request_type = request.probe_only
                                  ? WORKER_MESSAGE_TYPE::PROBE_OBSERVABLE
                                  : WORKER_MESSAGE_TYPE::RUN_BLOCK;
    WorkerMessage request_message;
    request_message.type = request_type;
    request_message.request_id = 1;
    request_message.inline_payload = SerializeWorkerRequest(request);
    transport->Send(request_message);
    const auto response_message = transport->Receive();
    const auto expected_response_type = request.probe_only
                                            ? WORKER_MESSAGE_TYPE::PROBE_RESULT
                                            : WORKER_MESSAGE_TYPE::RUN_RESULT;
    if (response_message.type != expected_response_type)
    {
        throw std::runtime_error(
            "worker TCP session returned unexpected "
            "message type");
    }

    WorkerMessage shutdown_message;
    shutdown_message.type = WORKER_MESSAGE_TYPE::SHUTDOWN;
    shutdown_message.request_id = response_message.request_id;
    transport->Send(shutdown_message);
    const int exit_code = child_exit.get();
    if (exit_code != 0)
    {
        throw std::runtime_error("child worker command failed with exit code " +
                                 std::to_string(exit_code));
    }

    return DeserializeWorkerResponse(response_message.inline_payload).execution;
}

}  // namespace

TcpChildProcessWorkerSession::TcpChildProcessWorkerSession(
    sponge::manager::WorkerConfig worker_config,
    bool use_shared_memory_payloads)
    : worker_config_(std::move(worker_config)),
      use_shared_memory_payloads_(use_shared_memory_payloads)
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
    if (use_shared_memory_payloads_)
    {
        command << " --worker-transport " << ShellQuote("shm");
    }
    command << " --worker-tcp "
            << ShellQuote("127.0.0.1:" + std::to_string(port));
    AppendManagerOutputRedirection(&command, worker_config_);

    child_exit_ =
        std::async(std::launch::async, [command_text = command.str()]()
                   { return std::system(command_text.c_str()); });

    transport_ = CreateTcpControlTransport(listener.Accept(),
                                           use_shared_memory_payloads_);
    const auto hello = transport_->Receive();
    if (hello.type != WORKER_MESSAGE_TYPE::HELLO)
    {
        throw std::runtime_error("worker TCP session did not start with HELLO");
    }
    started_ = true;
}

WorkerExecutionResponse TcpChildProcessWorkerSession::SendRequest(
    const WorkerFileRequest& request)
{
    Start();
    if (transport_ == nullptr)
    {
        throw std::runtime_error("worker TCP session is not connected");
    }

    const auto request_id = next_request_id_++;
    const auto request_type = request.probe_only
                                  ? WORKER_MESSAGE_TYPE::PROBE_OBSERVABLE
                                  : WORKER_MESSAGE_TYPE::RUN_BLOCK;
    WorkerMessage request_message;
    request_message.type = request_type;
    request_message.request_id = request_id;
    request_message.inline_payload = SerializeWorkerRequest(request);
    transport_->Send(request_message);
    const auto response_message = transport_->Receive();
    const auto expected_response_type = request.probe_only
                                            ? WORKER_MESSAGE_TYPE::PROBE_RESULT
                                            : WORKER_MESSAGE_TYPE::RUN_RESULT;
    if (response_message.type != expected_response_type)
    {
        throw std::runtime_error(
            "worker TCP session returned unexpected message type");
    }
    if (response_message.request_id != request_id)
    {
        throw std::runtime_error(
            "worker TCP session returned unexpected request id");
    }
    return DeserializeWorkerResponse(response_message.inline_payload).execution;
}

WorkerExecutionResponse TcpChildProcessWorkerSession::ExecuteBlock(
    int steps, bool emit_output, const sponge::RuntimeState* imported_state)
{
    WorkerFileRequest request;
    request.steps = steps;
    request.managed_step_limit = worker_config_.managed_step_limit;
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

sponge::WorkerExchangeObservable TcpChildProcessWorkerSession::ProbeObservable(
    const sponge::RuntimeState& imported_state)
{
    WorkerFileRequest request;
    request.steps = 0;
    request.managed_step_limit = worker_config_.managed_step_limit;
    request.emit_output = false;
    request.probe_only = true;
    request.has_runtime_state = true;
    request.runtime_state = imported_state;
    return SendRequest(request).observable;
}

void TcpChildProcessWorkerSession::Shutdown()
{
    if (shutdown_)
    {
        return;
    }
    shutdown_ = true;
    if (started_ && transport_ != nullptr)
    {
        WorkerMessage shutdown_message;
        shutdown_message.type = WORKER_MESSAGE_TYPE::SHUTDOWN;
        shutdown_message.request_id = next_request_id_++;
        transport_->Send(shutdown_message);
        transport_.reset();
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

FileChildProcessWorkerSession::FileChildProcessWorkerSession(
    sponge::manager::WorkerConfig worker_config)
    : worker_config_(std::move(worker_config))
{
}

FileChildProcessWorkerSession::~FileChildProcessWorkerSession()
{
    try
    {
        Shutdown();
    }
    catch (...)
    {
    }
}

void FileChildProcessWorkerSession::Start()
{
    if (started_)
    {
        return;
    }
    if (worker_config_.executable_path.empty())
    {
        throw std::runtime_error(
            "FileChildProcessWorkerSession requires executable_path");
    }
    if (worker_config_.args.empty())
    {
        throw std::runtime_error(
            "FileChildProcessWorkerSession requires non-empty args");
    }

    const fs::path session_directory =
        MakeProtocolTempDirectory("sponge_worker_file_session");
    session_directory_ = session_directory.string();

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
    command << " --worker-file-session "
            << ShellQuote(session_directory.string());
    AppendManagerOutputRedirection(&command, worker_config_);

    child_exit_ =
        std::async(std::launch::async, [command_text = command.str()]()
                   { return std::system(command_text.c_str()); });
    WaitForFile(session_directory / "ready", &child_exit_,
                "file worker ready marker");
    started_ = true;
}

WorkerExecutionResponse FileChildProcessWorkerSession::SendRequest(
    const WorkerFileRequest& request)
{
    Start();
    const auto request_id = next_request_id_++;
    const fs::path session_directory(session_directory_);
    const std::string request_name =
        "request_" + std::to_string(request_id) + ".bin";
    const std::string response_name =
        "response_" + std::to_string(request_id) + ".bin";
    const fs::path request_path = session_directory / request_name;
    const fs::path response_path = session_directory / response_name;
    const fs::path request_tmp_path =
        session_directory / (request_name + ".tmp");

    WriteWorkerFileRequest(request_tmp_path.string(), request);
    fs::rename(request_tmp_path, request_path);
    WaitForFile(response_path, &child_exit_, "file worker response");

    const auto response = ReadWorkerFileResponse(response_path.string());
    fs::remove(request_path);
    fs::remove(response_path);
    return response.execution;
}

WorkerExecutionResponse FileChildProcessWorkerSession::ExecuteBlock(
    int steps, bool emit_output, const sponge::RuntimeState* imported_state)
{
    WorkerFileRequest request;
    request.steps = steps;
    request.managed_step_limit = worker_config_.managed_step_limit;
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

sponge::WorkerExchangeObservable FileChildProcessWorkerSession::ProbeObservable(
    const sponge::RuntimeState& imported_state)
{
    WorkerFileRequest request;
    request.steps = 0;
    request.managed_step_limit = worker_config_.managed_step_limit;
    request.emit_output = false;
    request.probe_only = true;
    request.has_runtime_state = true;
    request.runtime_state = imported_state;
    return SendRequest(request).observable;
}

void FileChildProcessWorkerSession::Shutdown()
{
    if (shutdown_)
    {
        return;
    }
    shutdown_ = true;
    if (!session_directory_.empty())
    {
        const fs::path session_directory(session_directory_);
        const fs::path shutdown_tmp_path = session_directory / "shutdown.tmp";
        const fs::path shutdown_path = session_directory / "shutdown";
        TouchFile(shutdown_tmp_path);
        fs::rename(shutdown_tmp_path, shutdown_path);
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
    if (!session_directory_.empty())
    {
        fs::remove_all(session_directory_);
    }
}

ChildProcessWorkerSession::ChildProcessWorkerSession(
    sponge::manager::WorkerConfig config)
    : config_(std::move(config))
{
}

ChildProcessWorkerSession::~ChildProcessWorkerSession()
{
    try
    {
        Shutdown();
    }
    catch (...)
    {
    }
}

WorkerExecutionResponse ChildProcessWorkerSession::RunBlock(
    int steps, bool emit_output, const sponge::RuntimeState* imported_state)
{
    if (config_.transport == "file")
    {
        if (file_session_ == nullptr)
        {
            file_session_ =
                std::make_unique<FileChildProcessWorkerSession>(config_);
        }
        return file_session_->ExecuteBlock(steps, emit_output, imported_state);
    }

    if (tcp_session_ == nullptr)
    {
        tcp_session_ = std::make_unique<TcpChildProcessWorkerSession>(
            config_, config_.transport == "shm");
    }
    return tcp_session_->ExecuteBlock(steps, emit_output, imported_state);
}

sponge::WorkerExchangeObservable ChildProcessWorkerSession::ProbeObservable(
    const sponge::RuntimeState& imported_state)
{
    if (config_.transport == "file")
    {
        if (file_session_ == nullptr)
        {
            file_session_ =
                std::make_unique<FileChildProcessWorkerSession>(config_);
        }
        return file_session_->ProbeObservable(imported_state);
    }

    if (tcp_session_ == nullptr)
    {
        tcp_session_ = std::make_unique<TcpChildProcessWorkerSession>(
            config_, config_.transport == "shm");
    }
    return tcp_session_->ProbeObservable(imported_state);
}

void ChildProcessWorkerSession::Shutdown()
{
    if (file_session_ != nullptr)
    {
        file_session_->Shutdown();
        file_session_.reset();
    }
    if (tcp_session_ != nullptr)
    {
        tcp_session_->Shutdown();
        tcp_session_.reset();
    }
}

std::unique_ptr<WorkerSession> CreateWorkerSession(
    sponge::manager::WorkerConfig config)
{
    return std::make_unique<ChildProcessWorkerSession>(std::move(config));
}

sponge::worker_protocol::WorkerExecutionResponse
ChildProcessWorkerProtocol::ExecuteBlock(
    const sponge::manager::WorkerConfig& worker_config, int steps,
    bool emit_output, const sponge::RuntimeState* imported_state)
{
    WorkerFileRequest request;
    request.steps = steps;
    request.managed_step_limit = worker_config.managed_step_limit;
    request.emit_output = emit_output;
    request.probe_only = false;
    request.has_runtime_state =
        imported_state != nullptr && imported_state->valid;
    if (request.has_runtime_state)
    {
        request.runtime_state = *imported_state;
    }
    if (worker_config.transport == "tcp" || worker_config.transport == "shm")
    {
        return RunTcpChildProcessWorker(worker_config, request);
    }
    return RunChildProcessWorker(worker_config, request);
}

sponge::WorkerExchangeObservable ChildProcessWorkerProtocol::ProbeObservable(
    const sponge::manager::WorkerConfig& worker_config,
    const sponge::RuntimeState& imported_state)
{
    WorkerFileRequest request;
    request.steps = 0;
    request.managed_step_limit = worker_config.managed_step_limit;
    request.emit_output = false;
    request.probe_only = true;
    request.has_runtime_state = true;
    request.runtime_state = imported_state;
    if (worker_config.transport == "tcp" || worker_config.transport == "shm")
    {
        return RunTcpChildProcessWorker(worker_config, request).observable;
    }
    return RunChildProcessWorker(worker_config, request).observable;
}

}  // namespace sponge::worker_protocol
