#include "file_protocol.h"

#include <fstream>
#include <stdexcept>

#include "runtime_state_codec.h"

namespace sponge::worker_protocol
{

void WriteWorkerFileRequest(const std::string& path,
                            const WorkerFileRequest& request)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        throw std::runtime_error("failed to open worker request file: " + path);
    }
    WriteWorkerRequest(&out, request);
}

WorkerFileRequest ReadWorkerFileRequest(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("failed to open worker request file: " + path);
    }
    return ReadWorkerRequest(&in);
}

void WriteWorkerFileResponse(const std::string& path,
                             const WorkerFileResponse& response)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        throw std::runtime_error("failed to open worker response file: " +
                                 path);
    }
    WriteWorkerResponse(&out, response);
}

WorkerFileResponse ReadWorkerFileResponse(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("failed to open worker response file: " +
                                 path);
    }
    return ReadWorkerResponse(&in);
}

}  // namespace sponge::worker_protocol
