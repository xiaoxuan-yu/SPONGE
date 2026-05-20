#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace sponge::worker_protocol
{

class TcpSocket
{
   public:
#if defined(_WIN32)
    using NativeHandle = std::uintptr_t;
#else
    using NativeHandle = int;
#endif

    TcpSocket();
    explicit TcpSocket(NativeHandle handle);
    TcpSocket(const TcpSocket&) = delete;
    TcpSocket& operator=(const TcpSocket&) = delete;
    TcpSocket(TcpSocket&& other) noexcept;
    TcpSocket& operator=(TcpSocket&& other) noexcept;
    ~TcpSocket();

    static TcpSocket Connect(const std::string& host, int port);
    static TcpSocket ListenLoopback(int port);

    TcpSocket Accept() const;
    int LocalPort() const;
    bool Valid() const;
    void Close();

    void ReadExact(char* data, std::size_t size) const;
    void WriteAll(const char* data, std::size_t size) const;

   private:
    NativeHandle handle_;
};

struct TcpEndpoint
{
    std::string host;
    int port = 0;
};

TcpEndpoint ParseTcpEndpoint(const std::string& endpoint);

}  // namespace sponge::worker_protocol
