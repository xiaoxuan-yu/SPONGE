#include "shm_transport.h"

#include <atomic>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <utility>

#include "tcp_protocol.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace sponge::worker_protocol
{

namespace
{

constexpr const char* kShmEnvelopeMagic = "SPONGE_SHM_V1";

std::atomic<std::uint64_t> g_shm_generation{1};

std::string MakeSharedMemoryName(std::uint64_t generation)
{
#if defined(_WIN32)
    return "Local\\sponge_worker_" + std::to_string(GetCurrentProcessId()) +
           "_" + std::to_string(generation);
#else
    return "/sponge_worker_" + std::to_string(getpid()) + "_" +
           std::to_string(generation);
#endif
}

std::string EncodePayloadRef(const WorkerPayloadRef& ref)
{
    std::ostringstream out;
    out << kShmEnvelopeMagic << '\n'
        << ref.name << '\n'
        << ref.offset << '\n'
        << ref.size << '\n'
        << ref.generation << '\n';
    return out.str();
}

bool DecodePayloadRef(const std::string& payload, WorkerPayloadRef* ref)
{
    if (ref == nullptr)
    {
        return false;
    }
    std::istringstream in(payload);
    std::string magic;
    if (!std::getline(in, magic) || magic != kShmEnvelopeMagic)
    {
        return false;
    }
    WorkerPayloadRef decoded;
    decoded.kind = WorkerPayloadKind::kSharedMemoryRef;
    if (!std::getline(in, decoded.name))
    {
        return false;
    }
    in >> decoded.offset;
    in >> decoded.size;
    in >> decoded.generation;
    if (!in)
    {
        return false;
    }
    *ref = decoded;
    return true;
}

#if defined(_WIN32)
std::string WindowsErrorMessage(const std::string& action)
{
    return action + " failed with Windows error " +
           std::to_string(GetLastError());
}
#else
std::string PosixErrorMessage(const std::string& action)
{
    return action + " failed: " + std::strerror(errno);
}
#endif

}  // namespace

ShmTransport::ShmTransport(std::unique_ptr<WorkerTransport> control_transport)
    : control_transport_(std::move(control_transport))
{
    if (control_transport_ == nullptr)
    {
        throw std::runtime_error("ShmTransport requires a control transport");
    }
}

ShmTransport::~ShmTransport()
{
    try
    {
        Close();
    }
    catch (...)
    {
    }
}

void ShmTransport::Send(const WorkerMessage& message)
{
    WorkerMessage control_message = message;
    if (!message.inline_payload.empty())
    {
        control_message.payload_ref = WritePayload(message.inline_payload);
        control_message.inline_payload =
            EncodePayloadRef(control_message.payload_ref);
        control_message.payload_ref = WorkerPayloadRef{};
        control_message.payload_ref.size = control_message.inline_payload.size();
    }
    control_transport_->Send(control_message);
}

WorkerMessage ShmTransport::Receive()
{
    WorkerMessage message = control_transport_->Receive();
    WorkerPayloadRef ref;
    if (DecodePayloadRef(message.inline_payload, &ref))
    {
        message.payload_ref = ref;
        message.inline_payload = ReadPayload(ref);
    }
    ReleaseOwnedPayloads();
    return message;
}

void ShmTransport::Close()
{
    ReleaseOwnedPayloads();
    if (control_transport_ != nullptr)
    {
        control_transport_->Close();
    }
}

WorkerPayloadRef ShmTransport::WritePayload(const std::string& payload)
{
    WorkerPayloadRef ref;
    ref.kind = WorkerPayloadKind::kSharedMemoryRef;
    ref.generation = g_shm_generation.fetch_add(1);
    ref.name = MakeSharedMemoryName(ref.generation);
    ref.offset = 0;
    ref.size = payload.size();

#if defined(_WIN32)
    if (ref.size > std::numeric_limits<DWORD>::max())
    {
        throw std::runtime_error(
            "ShmTransport Windows payload exceeds 4 GiB");
    }
    HANDLE mapping = CreateFileMappingA(
        INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
        static_cast<DWORD>(ref.size), ref.name.c_str());
    if (mapping == nullptr)
    {
        throw std::runtime_error(WindowsErrorMessage("CreateFileMapping"));
    }
    void* view =
        MapViewOfFile(mapping, FILE_MAP_WRITE, 0, 0,
                      static_cast<SIZE_T>(ref.size));
    if (view == nullptr)
    {
        CloseHandle(mapping);
        throw std::runtime_error(WindowsErrorMessage("MapViewOfFile"));
    }
    std::memcpy(view, payload.data(), payload.size());
    UnmapViewOfFile(view);
    owned_payloads_.push_back(mapping);
#else
    const int fd = shm_open(ref.name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0)
    {
        throw std::runtime_error(PosixErrorMessage("shm_open"));
    }
    if (ftruncate(fd, static_cast<off_t>(ref.size)) != 0)
    {
        close(fd);
        shm_unlink(ref.name.c_str());
        throw std::runtime_error(PosixErrorMessage("ftruncate"));
    }
    void* view =
        mmap(nullptr, ref.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (view == MAP_FAILED)
    {
        close(fd);
        shm_unlink(ref.name.c_str());
        throw std::runtime_error(PosixErrorMessage("mmap"));
    }
    std::memcpy(view, payload.data(), payload.size());
    munmap(view, ref.size);
    close(fd);
#endif
    return ref;
}

std::string ShmTransport::ReadPayload(const WorkerPayloadRef& ref)
{
    if (ref.kind != WorkerPayloadKind::kSharedMemoryRef)
    {
        throw std::runtime_error("ShmTransport expected shared-memory payload");
    }
    std::string payload(ref.size, '\0');

#if defined(_WIN32)
    HANDLE mapping = OpenFileMappingA(FILE_MAP_READ, FALSE, ref.name.c_str());
    if (mapping == nullptr)
    {
        throw std::runtime_error(WindowsErrorMessage("OpenFileMapping"));
    }
    void* view =
        MapViewOfFile(mapping, FILE_MAP_READ,
                      static_cast<DWORD>(ref.offset >> 32),
                      static_cast<DWORD>(ref.offset & 0xffffffffu),
                      static_cast<SIZE_T>(ref.size));
    if (view == nullptr)
    {
        CloseHandle(mapping);
        throw std::runtime_error(WindowsErrorMessage("MapViewOfFile"));
    }
    std::memcpy(payload.data(), view, ref.size);
    UnmapViewOfFile(view);
    CloseHandle(mapping);
#else
    const int fd = shm_open(ref.name.c_str(), O_RDONLY, 0600);
    if (fd < 0)
    {
        throw std::runtime_error(PosixErrorMessage("shm_open"));
    }
    void* view =
        mmap(nullptr, ref.size, PROT_READ, MAP_SHARED, fd, ref.offset);
    if (view == MAP_FAILED)
    {
        close(fd);
        throw std::runtime_error(PosixErrorMessage("mmap"));
    }
    std::memcpy(payload.data(), view, ref.size);
    munmap(view, ref.size);
    close(fd);
    shm_unlink(ref.name.c_str());
#endif
    return payload;
}

void ShmTransport::ReleaseOwnedPayloads()
{
#if defined(_WIN32)
    for (void* payload : owned_payloads_)
    {
        CloseHandle(static_cast<HANDLE>(payload));
    }
#endif
    owned_payloads_.clear();
}

std::unique_ptr<WorkerTransport> CreateTcpControlTransport(
    TcpSocket socket, bool use_shared_memory_payloads)
{
    auto tcp_transport =
        std::make_unique<TcpTransport>(std::move(socket));
    if (use_shared_memory_payloads)
    {
        return std::make_unique<ShmTransport>(std::move(tcp_transport));
    }
    return tcp_transport;
}

}  // namespace sponge::worker_protocol
