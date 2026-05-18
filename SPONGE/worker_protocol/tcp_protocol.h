#pragma once

#include <cstdint>
#include <string>

#include "message_protocol.h"
#include "runtime_state_codec.h"
#include "tcp_socket.h"
#include "transport.h"

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

class TcpTransport : public WorkerTransport
{
   public:
    explicit TcpTransport(TcpSocket socket);

    void Send(const WorkerMessage& message) override;
    WorkerMessage Receive() override;
    void Close() override;

   private:
    TcpSocket socket_;
};

}  // namespace sponge::worker_protocol
