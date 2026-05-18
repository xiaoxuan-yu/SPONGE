#pragma once

#include <cstdint>
#include <string>

#include "file_protocol.h"
#include "message_protocol.h"
#include "tcp_socket.h"

namespace sponge::worker_protocol
{

struct WorkerTcpMessage
{
    WORKER_MESSAGE_HEADER header;
    std::string payload;
};

void WriteWorkerTcpMessage(const TcpSocket& socket,
                           WORKER_MESSAGE_TYPE message_type,
                           std::uint64_t request_id,
                           const std::string& payload);
WorkerTcpMessage ReadWorkerTcpMessage(const TcpSocket& socket);

std::string SerializeWorkerRequest(const WorkerFileRequest& request);
WorkerFileRequest DeserializeWorkerRequest(const std::string& payload);
std::string SerializeWorkerResponse(const WorkerFileResponse& response);
WorkerFileResponse DeserializeWorkerResponse(const std::string& payload);

}  // namespace sponge::worker_protocol
