#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "scheduler/scheduler.h"
#include "worker_protocol/file_protocol.h"
#include "worker_protocol/shm_transport.h"
#include "worker_protocol/tcp_protocol.h"

namespace
{

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
    std::string worker_tcp_endpoint;
    std::string worker_transport = "tcp";

    for (int i = 1; i < argc; i++)
    {
        const std::string arg = argv[i];
        if (arg == "--worker-request" || arg == "--worker-response" ||
            arg == "--worker-tcp" || arg == "--worker-transport")
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
        if (!worker_request_path.empty() || !worker_response_path.empty())
        {
            throw std::runtime_error(
                "--worker-tcp cannot be combined with file worker mode");
        }
        return RunWorkerTcpMode(scheduler_args, worker_tcp_endpoint,
                                worker_transport);
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
