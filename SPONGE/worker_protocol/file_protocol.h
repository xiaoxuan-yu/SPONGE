#pragma once

#include <string>

#include "worker_protocol.h"

namespace sponge::worker_protocol
{

struct WorkerFileRequest
{
    int steps = 0;
    int managed_step_limit = 0;
    bool emit_output = false;
    bool probe_only = false;
    bool has_runtime_state = false;
    sponge::RuntimeState runtime_state;
};

struct WorkerFileResponse
{
    WorkerExecutionResponse execution;
};

void WriteWorkerFileRequest(const std::string& path,
                            const WorkerFileRequest& request);
WorkerFileRequest ReadWorkerFileRequest(const std::string& path);

void WriteWorkerFileResponse(const std::string& path,
                             const WorkerFileResponse& response);
WorkerFileResponse ReadWorkerFileResponse(const std::string& path);

}  // namespace sponge::worker_protocol
