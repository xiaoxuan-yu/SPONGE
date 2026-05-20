#pragma once

#include <iosfwd>
#include <string>

#include "file_protocol.h"

namespace sponge::worker_protocol
{

void WriteWorkerRequest(std::ostream* out, const WorkerFileRequest& request);
WorkerFileRequest ReadWorkerRequest(std::istream* in);

void WriteWorkerResponse(std::ostream* out, const WorkerFileResponse& response);
WorkerFileResponse ReadWorkerResponse(std::istream* in);

std::string SerializeWorkerRequest(const WorkerFileRequest& request);
WorkerFileRequest DeserializeWorkerRequest(const std::string& payload);

std::string SerializeWorkerResponse(const WorkerFileResponse& response);
WorkerFileResponse DeserializeWorkerResponse(const std::string& payload);

}  // namespace sponge::worker_protocol
