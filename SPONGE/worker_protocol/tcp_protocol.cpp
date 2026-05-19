#include "tcp_protocol.h"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace sponge::worker_protocol
{

void WriteWorkerTcpMessage(const TcpSocket& socket,
                           WORKER_MESSAGE_TYPE message_type,
                           std::uint64_t request_id, const std::string& payload)
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

TcpTransport::TcpTransport(TcpSocket socket) : socket_(std::move(socket)) {}

void TcpTransport::Send(const WorkerMessage& message)
{
    if (message.payload_ref.kind != WorkerPayloadKind::kInlineBinary)
    {
        throw std::runtime_error(
            "TcpTransport currently supports only inline payloads");
    }
    WriteWorkerTcpMessage(socket_, message.type, message.request_id,
                          message.inline_payload);
}

WorkerMessage TcpTransport::Receive()
{
    const auto tcp_message = ReadWorkerTcpMessage(socket_);
    WorkerMessage message;
    message.type = tcp_message.header.message_type;
    message.request_id = tcp_message.header.request_id;
    message.inline_payload = tcp_message.payload;
    message.payload_ref.kind = WorkerPayloadKind::kInlineBinary;
    message.payload_ref.size = message.inline_payload.size();
    return message;
}

void TcpTransport::Close() { socket_.Close(); }

}  // namespace sponge::worker_protocol
