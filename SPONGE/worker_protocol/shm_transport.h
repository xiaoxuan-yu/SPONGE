#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tcp_socket.h"
#include "transport.h"

namespace sponge::worker_protocol
{

class ShmTransport : public WorkerTransport
{
   public:
    explicit ShmTransport(std::unique_ptr<WorkerTransport> control_transport);
    ~ShmTransport() override;

    void Send(const WorkerMessage& message) override;
    WorkerMessage Receive() override;
    void Close() override;

   private:
    WorkerPayloadRef WritePayload(const std::string& payload);
    std::string ReadPayload(const WorkerPayloadRef& ref);
    void ReleaseOwnedPayloads();

    std::unique_ptr<WorkerTransport> control_transport_;
    std::vector<void*> owned_payloads_;
};

std::unique_ptr<WorkerTransport> CreateTcpControlTransport(
    TcpSocket socket, bool use_shared_memory_payloads);

}  // namespace sponge::worker_protocol
