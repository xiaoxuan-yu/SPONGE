#pragma once

#include "message_protocol.h"

namespace sponge::worker_protocol
{

class WorkerTransport
{
   public:
    virtual ~WorkerTransport() = default;

    virtual void Send(const WorkerMessage& message) = 0;
    virtual WorkerMessage Receive() = 0;
    virtual void Close() = 0;
};

}  // namespace sponge::worker_protocol
