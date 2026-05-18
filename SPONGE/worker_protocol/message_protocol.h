#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>

namespace sponge::worker_protocol
{

constexpr std::uint32_t WORKER_PROTOCOL_MAGIC = 0x5350574Bu;
constexpr std::uint32_t WORKER_PROTOCOL_VERSION = 1u;
constexpr std::uint64_t WORKER_PROTOCOL_MAX_PAYLOAD_SIZE =
    16ull * 1024ull * 1024ull * 1024ull;
constexpr std::size_t WORKER_MESSAGE_HEADER_SIZE = 28;

enum class WORKER_MESSAGE_TYPE : std::uint32_t
{
    HELLO = 1,
    RUN_BLOCK = 2,
    RUN_RESULT = 3,
    IMPORT_STATE = 4,
    PROBE_OBSERVABLE = 5,
    PROBE_RESULT = 6,
    GET_STATUS = 7,
    SHUTDOWN = 8,
    ERROR = 9,
};

struct WORKER_MESSAGE_HEADER
{
    std::uint32_t magic = WORKER_PROTOCOL_MAGIC;
    std::uint32_t version = WORKER_PROTOCOL_VERSION;
    WORKER_MESSAGE_TYPE message_type = WORKER_MESSAGE_TYPE::ERROR;
    std::uint64_t request_id = 0;
    std::uint64_t payload_size = 0;
};

bool Is_Known_Worker_Message_Type(std::uint32_t message_type);
void Validate_Worker_Message_Header(const WORKER_MESSAGE_HEADER& header);
void Write_Worker_Message_Header(std::ostream* out,
                                 const WORKER_MESSAGE_HEADER& header);
WORKER_MESSAGE_HEADER Read_Worker_Message_Header(std::istream* in);

}  // namespace sponge::worker_protocol
