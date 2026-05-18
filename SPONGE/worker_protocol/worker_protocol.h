#pragma once

#include <cstdint>
#include <future>
#include <memory>

#include "../manager/core/types.h"

namespace sponge::worker_protocol
{

struct WorkerFileRequest;
class TcpTransport;
class WorkerTransport;

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

class WorkerSession
{
   public:
    virtual ~WorkerSession() = default;

    virtual WorkerExecutionResponse RunBlock(
        int steps, bool emit_output,
        const sponge::RuntimeState* imported_state) = 0;

    virtual sponge::WorkerExchangeObservable ProbeObservable(
        const sponge::RuntimeState& imported_state) = 0;

    virtual void Shutdown() = 0;
};

class TcpChildProcessWorkerSession
{
   public:
    explicit TcpChildProcessWorkerSession(
        sponge::manager::WorkerConfig worker_config,
        bool use_shared_memory_payloads = false);
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
    std::unique_ptr<WorkerTransport> transport_;
    std::future<int> child_exit_;
    std::uint64_t next_request_id_ = 1;
    bool use_shared_memory_payloads_ = false;
    bool started_ = false;
    bool shutdown_ = false;
};

class InProcessWorkerSession : public WorkerSession
{
   public:
    explicit InProcessWorkerSession(sponge::manager::WorkerConfig config);

    WorkerExecutionResponse RunBlock(
        int steps, bool emit_output,
        const sponge::RuntimeState* imported_state) override;

    sponge::WorkerExchangeObservable ProbeObservable(
        const sponge::RuntimeState& imported_state) override;

    void Shutdown() override;

   private:
    sponge::manager::WorkerConfig config_;
};

class ChildProcessWorkerSession : public WorkerSession
{
   public:
    explicit ChildProcessWorkerSession(sponge::manager::WorkerConfig config);
    ~ChildProcessWorkerSession() override;

    WorkerExecutionResponse RunBlock(
        int steps, bool emit_output,
        const sponge::RuntimeState* imported_state) override;

    sponge::WorkerExchangeObservable ProbeObservable(
        const sponge::RuntimeState& imported_state) override;

    void Shutdown() override;

   private:
    sponge::manager::WorkerConfig config_;
    std::unique_ptr<TcpChildProcessWorkerSession> tcp_session_;
};

std::unique_ptr<WorkerSession> CreateWorkerSession(
    sponge::manager::WorkerConfig config);

}  // namespace sponge::worker_protocol
