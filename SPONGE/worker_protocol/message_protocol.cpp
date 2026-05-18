#include "message_protocol.h"

#include <array>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace sponge::worker_protocol
{

namespace
{

void Write_U32(std::ostream* out, std::uint32_t value)
{
    const std::array<unsigned char, 4> bytes = {
        static_cast<unsigned char>(value & 0xffu),
        static_cast<unsigned char>((value >> 8u) & 0xffu),
        static_cast<unsigned char>((value >> 16u) & 0xffu),
        static_cast<unsigned char>((value >> 24u) & 0xffu),
    };
    out->write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

void Write_U64(std::ostream* out, std::uint64_t value)
{
    std::array<unsigned char, 8> bytes{};
    for (std::size_t i = 0; i < bytes.size(); i++)
    {
        bytes[i] = static_cast<unsigned char>((value >> (8u * i)) & 0xffu);
    }
    out->write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

std::uint32_t Read_U32(std::istream* in)
{
    std::array<unsigned char, 4> bytes{};
    in->read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!(*in))
    {
        throw std::runtime_error("failed to read worker message u32");
    }
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8u) |
           (static_cast<std::uint32_t>(bytes[2]) << 16u) |
           (static_cast<std::uint32_t>(bytes[3]) << 24u);
}

std::uint64_t Read_U64(std::istream* in)
{
    std::array<unsigned char, 8> bytes{};
    in->read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!(*in))
    {
        throw std::runtime_error("failed to read worker message u64");
    }
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < bytes.size(); i++)
    {
        value |= static_cast<std::uint64_t>(bytes[i]) << (8u * i);
    }
    return value;
}

}  // namespace

bool Is_Known_Worker_Message_Type(std::uint32_t message_type)
{
    switch (static_cast<WORKER_MESSAGE_TYPE>(message_type))
    {
        case WORKER_MESSAGE_TYPE::HELLO:
        case WORKER_MESSAGE_TYPE::RUN_BLOCK:
        case WORKER_MESSAGE_TYPE::RUN_RESULT:
        case WORKER_MESSAGE_TYPE::IMPORT_STATE:
        case WORKER_MESSAGE_TYPE::PROBE_OBSERVABLE:
        case WORKER_MESSAGE_TYPE::PROBE_RESULT:
        case WORKER_MESSAGE_TYPE::GET_STATUS:
        case WORKER_MESSAGE_TYPE::SHUTDOWN:
        case WORKER_MESSAGE_TYPE::ERROR:
            return true;
    }
    return false;
}

void Validate_Worker_Message_Header(const WORKER_MESSAGE_HEADER& header)
{
    if (header.magic != WORKER_PROTOCOL_MAGIC)
    {
        throw std::runtime_error("invalid worker message magic");
    }
    if (header.version != WORKER_PROTOCOL_VERSION)
    {
        throw std::runtime_error("unsupported worker protocol version");
    }
    if (!Is_Known_Worker_Message_Type(
            static_cast<std::uint32_t>(header.message_type)))
    {
        throw std::runtime_error("unknown worker message type");
    }
    if (header.payload_size > WORKER_PROTOCOL_MAX_PAYLOAD_SIZE)
    {
        throw std::runtime_error("worker message payload is too large");
    }
}

void Write_Worker_Message_Header(std::ostream* out,
                                 const WORKER_MESSAGE_HEADER& header)
{
    if (out == nullptr)
    {
        throw std::runtime_error("cannot write worker header to null stream");
    }
    Validate_Worker_Message_Header(header);
    Write_U32(out, header.magic);
    Write_U32(out, header.version);
    Write_U32(out, static_cast<std::uint32_t>(header.message_type));
    Write_U64(out, header.request_id);
    Write_U64(out, header.payload_size);
}

WORKER_MESSAGE_HEADER Read_Worker_Message_Header(std::istream* in)
{
    if (in == nullptr)
    {
        throw std::runtime_error("cannot read worker header from null stream");
    }
    WORKER_MESSAGE_HEADER header;
    header.magic = Read_U32(in);
    header.version = Read_U32(in);
    const auto message_type = Read_U32(in);
    header.message_type = static_cast<WORKER_MESSAGE_TYPE>(message_type);
    header.request_id = Read_U64(in);
    header.payload_size = Read_U64(in);
    Validate_Worker_Message_Header(header);
    return header;
}

}  // namespace sponge::worker_protocol
