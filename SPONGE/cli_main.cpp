#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "scheduler/scheduler.h"
#include "worker_protocol/file_protocol.h"
#include "worker_protocol/shm_transport.h"
#include "worker_protocol/tcp_protocol.h"

namespace
{

namespace fs = std::filesystem;

void TouchFile(const fs::path& path)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        throw std::runtime_error("failed to create file: " + path.string());
    }
}

bool IsWorkerRequestFile(const fs::path& path)
{
    const std::string name = path.filename().string();
    return name.rfind("request_", 0) == 0 &&
           name.size() > std::string("request_.bin").size() &&
           name.substr(name.size() - 4) == ".bin";
}

fs::path FindNextWorkerRequest(const fs::path& session_directory)
{
    std::vector<fs::path> requests;
    for (const auto& entry : fs::directory_iterator(session_directory))
    {
        if (entry.is_regular_file() && IsWorkerRequestFile(entry.path()))
        {
            requests.push_back(entry.path());
        }
    }
    std::sort(requests.begin(), requests.end());
    return requests.empty() ? fs::path{} : requests.front();
}

fs::path ResponsePathForRequest(const fs::path& request_path)
{
    const std::string request_name = request_path.filename().string();
    const std::string request_id = request_name.substr(
        std::string("request_").size(),
        request_name.size() - std::string("request_").size() - 4);
    return request_path.parent_path() / ("response_" + request_id + ".bin");
}

sponge::worker_protocol::WorkerFileResponse ExecuteWorkerRequest(
    const std::vector<std::string>& scheduler_args,
    const sponge::worker_protocol::WorkerFileRequest& request)
{
    sponge::SpongeScheduler scheduler;
    scheduler.InitializeFromArgs(scheduler_args);
    if (request.probe_only)
    {
        scheduler.EnsureForeignStateProbeSafe();
    }
    if (request.has_runtime_state && request.runtime_state.valid)
    {
        scheduler.ImportRuntimeState(request.runtime_state);
    }
    if (!request.probe_only)
    {
        scheduler.RunSteps(request.steps, request.emit_output);
    }

    sponge::worker_protocol::WorkerFileResponse response;
    response.execution.runtime_state = scheduler.ExportRuntimeState();
    response.execution.snapshot = scheduler.Snapshot();
    response.execution.observable = scheduler.CollectExchangeObservables();
    response.execution.finished = response.execution.snapshot.finished;
    scheduler.Finalize();
    return response;
}

int RunWorkerMode(const std::vector<std::string>& scheduler_args,
                  const std::string& request_path,
                  const std::string& response_path)
{
    const auto request =
        sponge::worker_protocol::ReadWorkerFileRequest(request_path);
    const auto response = ExecuteWorkerRequest(scheduler_args, request);
    sponge::worker_protocol::WriteWorkerFileResponse(response_path, response);
    return 0;
}

int RunWorkerFileSessionMode(const std::vector<std::string>& scheduler_args,
                             const std::string& session_directory_text)
{
    const fs::path session_directory(session_directory_text);
    fs::create_directories(session_directory);
    const fs::path ready_tmp_path = session_directory / "ready.tmp";
    const fs::path ready_path = session_directory / "ready";
    TouchFile(ready_tmp_path);
    fs::rename(ready_tmp_path, ready_path);

    std::unique_ptr<sponge::SpongeScheduler> scheduler;
    while (true)
    {
        if (fs::exists(session_directory / "shutdown"))
        {
            break;
        }

        const fs::path request_path = FindNextWorkerRequest(session_directory);
        if (request_path.empty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        auto request = sponge::worker_protocol::ReadWorkerFileRequest(
            request_path.string());
        sponge::worker_protocol::WorkerFileResponse response;
        if (request.probe_only)
        {
            response = ExecuteWorkerRequest(scheduler_args, request);
        }
        else
        {
            if (scheduler == nullptr)
            {
                scheduler = std::make_unique<sponge::SpongeScheduler>();
                scheduler->InitializeFromArgs(scheduler_args);
            }
            if (request.has_runtime_state && request.runtime_state.valid)
            {
                scheduler->ImportRuntimeState(request.runtime_state);
            }
            scheduler->RunSteps(request.steps, request.emit_output);
            response.execution.runtime_state = scheduler->ExportRuntimeState();
            response.execution.snapshot = scheduler->Snapshot();
            response.execution.observable =
                scheduler->CollectExchangeObservables();
            response.execution.finished = response.execution.snapshot.finished;
        }

        const fs::path response_path = ResponsePathForRequest(request_path);
        const fs::path response_tmp_path =
            response_path.parent_path() /
            (response_path.filename().string() + ".tmp");
        sponge::worker_protocol::WriteWorkerFileResponse(
            response_tmp_path.string(), response);
        fs::rename(response_tmp_path, response_path);
        fs::remove(request_path);
    }
    if (scheduler != nullptr)
    {
        scheduler->Finalize();
    }
    return 0;
}

int RunWorkerTcpMode(const std::vector<std::string>& scheduler_args,
                     const std::string& endpoint,
                     const std::string& worker_transport)
{
    const auto parsed = sponge::worker_protocol::ParseTcpEndpoint(endpoint);
    auto transport = sponge::worker_protocol::CreateTcpControlTransport(
        sponge::worker_protocol::TcpSocket::Connect(parsed.host, parsed.port),
        worker_transport == "shm");
    sponge::worker_protocol::WorkerMessage hello;
    hello.type = sponge::worker_protocol::WORKER_MESSAGE_TYPE::HELLO;
    transport->Send(hello);

    std::unique_ptr<sponge::SpongeScheduler> scheduler;
    while (true)
    {
        const auto message = transport->Receive();
        if (message.type ==
            sponge::worker_protocol::WORKER_MESSAGE_TYPE::SHUTDOWN)
        {
            break;
        }
        if (message.type !=
                sponge::worker_protocol::WORKER_MESSAGE_TYPE::RUN_BLOCK &&
            message.type !=
                sponge::worker_protocol::WORKER_MESSAGE_TYPE::PROBE_OBSERVABLE)
        {
            throw std::runtime_error("unsupported worker TCP message type");
        }

        auto request = sponge::worker_protocol::DeserializeWorkerRequest(
            message.inline_payload);
        if (message.type ==
            sponge::worker_protocol::WORKER_MESSAGE_TYPE::PROBE_OBSERVABLE)
        {
            request.probe_only = true;
        }
        sponge::worker_protocol::WorkerFileResponse response;
        if (request.probe_only)
        {
            response = ExecuteWorkerRequest(scheduler_args, request);
        }
        else
        {
            if (scheduler == nullptr)
            {
                scheduler = std::make_unique<sponge::SpongeScheduler>();
                scheduler->InitializeFromArgs(scheduler_args);
            }
            if (request.has_runtime_state && request.runtime_state.valid)
            {
                scheduler->ImportRuntimeState(request.runtime_state);
            }
            scheduler->RunSteps(request.steps, request.emit_output);
            response.execution.runtime_state = scheduler->ExportRuntimeState();
            response.execution.snapshot = scheduler->Snapshot();
            response.execution.observable =
                scheduler->CollectExchangeObservables();
            response.execution.finished = response.execution.snapshot.finished;
        }
        const auto response_type =
            request.probe_only
                ? sponge::worker_protocol::WORKER_MESSAGE_TYPE::PROBE_RESULT
                : sponge::worker_protocol::WORKER_MESSAGE_TYPE::RUN_RESULT;
        sponge::worker_protocol::WorkerMessage response_message;
        response_message.type = response_type;
        response_message.request_id = message.request_id;
        response_message.inline_payload =
            sponge::worker_protocol::SerializeWorkerResponse(response);
        transport->Send(response_message);
    }
    if (scheduler != nullptr)
    {
        scheduler->Finalize();
    }
    return 0;
}

}  // namespace

int main(int argc, char* argv[])
{
    std::vector<std::string> scheduler_args;
    scheduler_args.reserve(argc > 0 ? argc : 1);
    scheduler_args.push_back(argc > 0 ? argv[0] : "SPONGE");

    std::string worker_request_path;
    std::string worker_response_path;
    std::string worker_file_session;
    std::string worker_tcp_endpoint;
    std::string worker_transport = "tcp";

    for (int i = 1; i < argc; i++)
    {
        const std::string arg = argv[i];
        if (arg == "--worker-request" || arg == "--worker-response" ||
            arg == "--worker-file-session" || arg == "--worker-tcp" ||
            arg == "--worker-transport")
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error("missing value for " + arg);
            }
            const std::string value = argv[++i];
            if (arg == "--worker-request")
            {
                worker_request_path = value;
            }
            else if (arg == "--worker-response")
            {
                worker_response_path = value;
            }
            else if (arg == "--worker-tcp")
            {
                worker_tcp_endpoint = value;
            }
            else if (arg == "--worker-file-session")
            {
                worker_file_session = value;
            }
            else
            {
                if (value != "tcp" && value != "shm")
                {
                    throw std::runtime_error(
                        "--worker-transport must be tcp or shm");
                }
                worker_transport = value;
            }
            continue;
        }
        scheduler_args.push_back(arg);
    }

    if (!worker_tcp_endpoint.empty())
    {
        if (!worker_request_path.empty() || !worker_response_path.empty() ||
            !worker_file_session.empty())
        {
            throw std::runtime_error(
                "--worker-tcp cannot be combined with file worker mode");
        }
        return RunWorkerTcpMode(scheduler_args, worker_tcp_endpoint,
                                worker_transport);
    }

    if (!worker_file_session.empty())
    {
        if (!worker_request_path.empty() || !worker_response_path.empty())
        {
            throw std::runtime_error(
                "--worker-file-session cannot be combined with one-shot file "
                "worker mode");
        }
        return RunWorkerFileSessionMode(scheduler_args, worker_file_session);
    }

    if (!worker_request_path.empty() || !worker_response_path.empty())
    {
        if (worker_request_path.empty() || worker_response_path.empty())
        {
            throw std::runtime_error(
                "worker mode requires both --worker-request and "
                "--worker-response");
        }
        return RunWorkerMode(scheduler_args, worker_request_path,
                             worker_response_path);
    }

    sponge::SpongeScheduler scheduler;
    scheduler.InitializeFromArgs(scheduler_args);
    scheduler.RunToEnd(true);
    scheduler.Finalize();
    return 0;
}
