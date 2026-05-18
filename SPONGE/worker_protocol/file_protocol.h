#pragma once

#include <iosfwd>
#include <string>

#include "worker_protocol.h"

namespace sponge::worker_protocol
{

struct WorkerFileRequest
{
    int steps = 0;
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
void WriteWorkerRequest(std::ostream* out, const WorkerFileRequest& request);
WorkerFileRequest ReadWorkerRequest(std::istream* in);

void WriteWorkerFileResponse(const std::string& path,
                             const WorkerFileResponse& response);
WorkerFileResponse ReadWorkerFileResponse(const std::string& path);
void WriteWorkerResponse(std::ostream* out, const WorkerFileResponse& response);
WorkerFileResponse ReadWorkerResponse(std::istream* in);

}  // namespace sponge::worker_protocol
