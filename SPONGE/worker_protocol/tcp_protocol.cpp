#include "tcp_protocol.h"

#include <sstream>
#include <stdexcept>
#include <vector>

namespace sponge::worker_protocol
{

void WriteWorkerTcpMessage(const TcpSocket& socket,
                           WORKER_MESSAGE_TYPE message_type,
                           std::uint64_t request_id,
                           const std::string& payload)
{
    WORKER_MESSAGE_HEADER header;
    header.message_type = message_type;
    header.request_id = request_id;
    header.payload_size = payload.size();

    std::ostringstream header_stream;
    Write_Worker_Message_Header(&header_stream, header);
    const auto header_payload = header_stream.str();
    socket.WriteAll(header_payload.data(), header_payload.size());
    if (!payload.empty())
    {
        socket.WriteAll(payload.data(), payload.size());
    }
}

WorkerTcpMessage ReadWorkerTcpMessage(const TcpSocket& socket)
{
    std::string header_payload(WORKER_MESSAGE_HEADER_SIZE, '\0');
    socket.ReadExact(header_payload.data(), header_payload.size());

    std::istringstream header_stream(header_payload);
    WorkerTcpMessage message;
    message.header = Read_Worker_Message_Header(&header_stream);
    if (message.header.payload_size > 0)
    {
        message.payload.resize(
            static_cast<std::size_t>(message.header.payload_size));
        socket.ReadExact(message.payload.data(), message.payload.size());
    }
    return message;
}

std::string SerializeWorkerRequest(const WorkerFileRequest& request)
{
    std::ostringstream out;
    WriteWorkerRequest(&out, request);
    return out.str();
}

WorkerFileRequest DeserializeWorkerRequest(const std::string& payload)
{
    std::istringstream in(payload);
    return ReadWorkerRequest(&in);
}

std::string SerializeWorkerResponse(const WorkerFileResponse& response)
{
    std::ostringstream out;
    WriteWorkerResponse(&out, response);
    return out.str();
}

WorkerFileResponse DeserializeWorkerResponse(const std::string& payload)
{
    std::istringstream in(payload);
    return ReadWorkerResponse(&in);
}

}  // namespace sponge::worker_protocol
