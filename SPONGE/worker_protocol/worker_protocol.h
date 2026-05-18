#pragma once

#include <cstdint>
#include <future>
#include <memory>

#include "../manager/core/types.h"
#include "tcp_socket.h"

namespace sponge::worker_protocol
{

struct WorkerFileRequest;

struct WorkerExecutionResponse
{
    sponge::SchedulerSnapshot snapshot;
    sponge::WorkerExchangeObservable observable;
    sponge::RuntimeState runtime_state;
    bool finished = false;
};

class WorkerProtocol
{
   public:
    virtual ~WorkerProtocol() = default;

    virtual WorkerExecutionResponse ExecuteBlock(
        const sponge::manager::WorkerConfig& worker_config, int steps,
        bool emit_output, const sponge::RuntimeState* imported_state) = 0;

    virtual sponge::WorkerExchangeObservable ProbeObservable(
        const sponge::manager::WorkerConfig& worker_config,
        const sponge::RuntimeState& imported_state) = 0;
};

class InProcessWorkerProtocol : public WorkerProtocol
{
   public:
    WorkerExecutionResponse ExecuteBlock(
        const sponge::manager::WorkerConfig& worker_config, int steps,
        bool emit_output, const sponge::RuntimeState* imported_state) override;

    sponge::WorkerExchangeObservable ProbeObservable(
        const sponge::manager::WorkerConfig& worker_config,
        const sponge::RuntimeState& imported_state) override;
};

class ChildProcessWorkerProtocol : public WorkerProtocol
{
   public:
    WorkerExecutionResponse ExecuteBlock(
        const sponge::manager::WorkerConfig& worker_config, int steps,
        bool emit_output, const sponge::RuntimeState* imported_state) override;

    sponge::WorkerExchangeObservable ProbeObservable(
        const sponge::manager::WorkerConfig& worker_config,
        const sponge::RuntimeState& imported_state) override;
};

class TcpChildProcessWorkerSession
{
   public:
    explicit TcpChildProcessWorkerSession(
        sponge::manager::WorkerConfig worker_config);
    ~TcpChildProcessWorkerSession();

    TcpChildProcessWorkerSession(const TcpChildProcessWorkerSession&) = delete;
    TcpChildProcessWorkerSession& operator=(
        const TcpChildProcessWorkerSession&) = delete;

    WorkerExecutionResponse ExecuteBlock(
        int steps, bool emit_output,
        const sponge::RuntimeState* imported_state);
    void Shutdown();

   private:
    void Start();
    WorkerExecutionResponse SendRequest(const WorkerFileRequest& request);

    sponge::manager::WorkerConfig worker_config_;
    std::unique_ptr<TcpSocket> socket_;
    std::future<int> child_exit_;
    std::uint64_t next_request_id_ = 1;
    bool started_ = false;
    bool shutdown_ = false;
};

}  // namespace sponge::worker_protocol
