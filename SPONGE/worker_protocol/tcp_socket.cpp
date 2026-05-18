#include "tcp_socket.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace sponge::worker_protocol
{

namespace
{

#if defined(_WIN32)
constexpr TcpSocket::NativeHandle kInvalidSocket =
    static_cast<TcpSocket::NativeHandle>(INVALID_SOCKET);
using SockLen = int;

SOCKET ToNativeSocket(TcpSocket::NativeHandle handle)
{
    return static_cast<SOCKET>(handle);
}

void EnsureSocketRuntime()
{
    static const bool initialized = []() {
        WSADATA data;
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0)
        {
            throw std::runtime_error("failed to initialize Winsock");
        }
        return true;
    }();
    (void)initialized;
}

std::string SocketErrorMessage(const std::string& action)
{
    return action + " failed with Winsock error " +
           std::to_string(WSAGetLastError());
}

void CloseNativeSocket(TcpSocket::NativeHandle handle)
{
    if (handle != kInvalidSocket)
    {
        closesocket(static_cast<SOCKET>(handle));
    }
}
#else
constexpr TcpSocket::NativeHandle kInvalidSocket = -1;
using SockLen = socklen_t;

int ToNativeSocket(TcpSocket::NativeHandle handle)
{
    return handle;
}

void EnsureSocketRuntime() {}

std::string SocketErrorMessage(const std::string& action)
{
    return action + " failed: " + std::strerror(errno);
}

void CloseNativeSocket(TcpSocket::NativeHandle handle)
{
    if (handle != kInvalidSocket)
    {
        close(handle);
    }
}
#endif

sockaddr_in MakeLoopbackAddress(int port)
{
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    return address;
}

sockaddr_in MakeAddress(const std::string& host, int port)
{
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    if (inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1)
    {
        throw std::runtime_error("invalid IPv4 worker endpoint host: " + host);
    }
    return address;
}

}  // namespace

TcpSocket::TcpSocket() : handle_(kInvalidSocket) {}

TcpSocket::TcpSocket(NativeHandle handle) : handle_(handle) {}

TcpSocket::TcpSocket(TcpSocket&& other) noexcept : handle_(other.handle_)
{
    other.handle_ = kInvalidSocket;
}

TcpSocket& TcpSocket::operator=(TcpSocket&& other) noexcept
{
    if (this != &other)
    {
        Close();
        handle_ = other.handle_;
        other.handle_ = kInvalidSocket;
    }
    return *this;
}

TcpSocket::~TcpSocket()
{
    Close();
}

TcpSocket TcpSocket::Connect(const std::string& host, int port)
{
    EnsureSocketRuntime();
    const NativeHandle handle =
        static_cast<NativeHandle>(socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
    if (handle == kInvalidSocket)
    {
        throw std::runtime_error(SocketErrorMessage("socket"));
    }

    TcpSocket socket(handle);
    const sockaddr_in address = MakeAddress(host, port);
    if (connect(ToNativeSocket(handle),
                reinterpret_cast<const sockaddr*>(&address),
                sizeof(address)) != 0)
    {
        throw std::runtime_error(SocketErrorMessage("connect"));
    }
    return socket;
}

TcpSocket TcpSocket::ListenLoopback(int port)
{
    EnsureSocketRuntime();
    const NativeHandle handle =
        static_cast<NativeHandle>(socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
    if (handle == kInvalidSocket)
    {
        throw std::runtime_error(SocketErrorMessage("socket"));
    }

    TcpSocket socket(handle);
    int reuse = 1;
    setsockopt(ToNativeSocket(handle), SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&reuse), sizeof(reuse));

    const sockaddr_in address = MakeLoopbackAddress(port);
    if (bind(ToNativeSocket(handle),
             reinterpret_cast<const sockaddr*>(&address),
             sizeof(address)) != 0)
    {
        throw std::runtime_error(SocketErrorMessage("bind"));
    }
    if (listen(ToNativeSocket(handle), 1) != 0)
    {
        throw std::runtime_error(SocketErrorMessage("listen"));
    }
    return socket;
}

TcpSocket TcpSocket::Accept() const
{
    if (!Valid())
    {
        throw std::runtime_error("cannot accept on an invalid TCP socket");
    }
    const NativeHandle handle =
        static_cast<NativeHandle>(accept(ToNativeSocket(handle_), nullptr,
                                         nullptr));
    if (handle == kInvalidSocket)
    {
        throw std::runtime_error(SocketErrorMessage("accept"));
    }
    return TcpSocket(handle);
}

int TcpSocket::LocalPort() const
{
    if (!Valid())
    {
        throw std::runtime_error("cannot query an invalid TCP socket");
    }
    sockaddr_in address{};
    SockLen length = sizeof(address);
    if (getsockname(ToNativeSocket(handle_),
                    reinterpret_cast<sockaddr*>(&address), &length) != 0)
    {
        throw std::runtime_error(SocketErrorMessage("getsockname"));
    }
    return ntohs(address.sin_port);
}

bool TcpSocket::Valid() const
{
    return handle_ != kInvalidSocket;
}

void TcpSocket::Close()
{
    CloseNativeSocket(handle_);
    handle_ = kInvalidSocket;
}

void TcpSocket::ReadExact(char* data, std::size_t size) const
{
    std::size_t offset = 0;
    while (offset < size)
    {
        const auto chunk =
            std::min<std::size_t>(size - offset,
                                  static_cast<std::size_t>(
                                      std::numeric_limits<int>::max()));
        const int received =
            recv(ToNativeSocket(handle_), data + offset,
                 static_cast<int>(chunk), 0);
        if (received == 0)
        {
            throw std::runtime_error("worker TCP connection closed");
        }
        if (received < 0)
        {
            throw std::runtime_error(SocketErrorMessage("recv"));
        }
        offset += static_cast<std::size_t>(received);
    }
}

void TcpSocket::WriteAll(const char* data, std::size_t size) const
{
    std::size_t offset = 0;
    while (offset < size)
    {
        const auto chunk =
            std::min<std::size_t>(size - offset,
                                  static_cast<std::size_t>(
                                      std::numeric_limits<int>::max()));
        const int sent = send(ToNativeSocket(handle_), data + offset,
                              static_cast<int>(chunk), 0);
        if (sent == 0)
        {
            throw std::runtime_error("worker TCP connection closed");
        }
        if (sent < 0)
        {
            throw std::runtime_error(SocketErrorMessage("send"));
        }
        offset += static_cast<std::size_t>(sent);
    }
}

TcpEndpoint ParseTcpEndpoint(const std::string& endpoint)
{
    const auto colon = endpoint.rfind(':');
    if (colon == std::string::npos || colon == 0 ||
        colon + 1 >= endpoint.size())
    {
        throw std::runtime_error("worker TCP endpoint must be host:port");
    }
    TcpEndpoint parsed;
    parsed.host = endpoint.substr(0, colon);
    parsed.port = std::stoi(endpoint.substr(colon + 1));
    if (parsed.port <= 0 || parsed.port > 65535)
    {
        throw std::runtime_error("worker TCP endpoint port is out of range");
    }
    return parsed;
}

}  // namespace sponge::worker_protocol
