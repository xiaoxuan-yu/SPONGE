# SPONGE Manager Transport Refactor Plan

## Goal

Refactor the current manager/worker communication code into a transport-neutral
framework so `SPONGE_MANAGER` can support:

- child-process workers over TCP
- child-process workers over shared memory
- future transports such as MPI, named pipes, or CUDA IPC

The main objective is to keep the manager scheduler logic independent from
transport details. `Manager` should schedule workers and exchange runtime
states; it should not know whether a request is sent through TCP, shared memory,
files, or another backend.

## Current Problem

The current implementation already has useful pieces:

- `SPONGE_MANAGER` can launch child-process workers.
- `SPONGE --worker-tcp host:port` supports TCP loopback worker mode.
- TCP child-process workers can stay alive across `RUN_BLOCK` requests.
- the legacy file protocol still works as a fallback.

However, transport details are still too visible in the orchestration layer:

- `Manager` directly handles persistent TCP sessions.
- child-process session logic, TCP socket logic, request serialization, and
  process launching are coupled in the same implementation area.
- adding `shm` directly to this layout would spread shared-memory-specific code
  into manager scheduling code.

Before adding shared memory, the communication stack should be split into clear
layers.

## Target Layering

The target architecture should have three layers:

```text
Manager Core
  owns schedules, block dispatch, exchange, state routing
        |
Worker Session
  owns worker lifecycle and request/response semantics
        |
Transport Backend
  owns bytes, shared-memory references, sockets, files, or future IPC
```

### Manager Core

`Manager` should depend only on a `WorkerSession` interface.

It should not include or directly use:

- TCP socket classes
- shared memory classes
- request/response file paths
- child process command construction
- transport-specific message headers

Example shape:

```cpp
class WorkerSession
{
   public:
    virtual ~WorkerSession() = default;

    virtual WorkerExecutionResponse RunBlock(
        int steps, bool emit_output,
        const sponge::RuntimeState* imported_state) = 0;

    virtual sponge::WorkerExchangeObservable ProbeObservable(
        const sponge::RuntimeState& imported_state) = 0;

    virtual void Shutdown() = 0;
};
```

`Manager::ExecuteAllSchedulesOnce()` should eventually call only:

```cpp
sessions_[i]->RunBlock(...);
```

### Worker Session

Worker sessions own the lifecycle and request semantics for one worker.

Recommended implementations:

- `ChildProcessWorkerSession`

`ChildProcessWorkerSession` owns:

- child process launch
- worker command-line construction
- worker connection setup
- request id management
- shutdown behavior
- selected transport backend

The manager should not need separate code paths for TCP versus shared memory.

### Transport Backend

Transport backends should implement a small transport interface.

Example shape:

```cpp
class WorkerTransport
{
   public:
    virtual ~WorkerTransport() = default;

    virtual void Send(const WorkerMessage& message) = 0;
    virtual WorkerMessage Receive() = 0;
    virtual void Close() = 0;
};
```

Recommended initial backends:

- `TcpTransport`
- `ShmTransport`
- `FileTransport` as a migration fallback

Future backends can include:

- `MpiTransport`
- `NamedPipeTransport`
- `CudaIpcTransport`

## Protocol Versus Transport

The worker protocol should be transport-neutral.

Protocol concepts:

- `HELLO`
- `RUN_BLOCK`
- `RUN_RESULT`
- `PROBE_OBSERVABLE`
- `PROBE_RESULT`
- `IMPORT_STATE`
- `GET_STATUS`
- `SHUTDOWN`
- `ERROR`

These message types are semantic. They should not be tied to TCP.

Transport concepts:

- inline binary payload
- shared-memory payload reference
- file payload reference
- future MPI payload
- future CUDA IPC handle

Recommended common message shape:

```cpp
enum class WorkerPayloadKind
{
    kInlineBinary,
    kSharedMemoryRef,
    kFileRef,
};

struct WorkerPayloadRef
{
    WorkerPayloadKind kind = WorkerPayloadKind::kInlineBinary;
    std::string name;
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
    std::uint64_t generation = 0;
};

struct WorkerMessage
{
    WorkerMessageType type;
    std::uint64_t request_id = 0;
    std::string inline_payload;
    WorkerPayloadRef payload_ref;
};
```

For TCP-only mode, `inline_payload` can carry the serialized request/response.

For shared-memory mode, TCP can carry a small `WorkerMessage` containing a
`WorkerPayloadRef`, while the large `RuntimeState` payload lives in shared
memory.

## RuntimeState Codec

`RuntimeState` serialization should be separated from file protocol code.

Recommended module:

```text
SPONGE/worker_protocol/runtime_state_codec.h
SPONGE/worker_protocol/runtime_state_codec.cpp
```

Responsibilities:

- serialize `RuntimeState`
- deserialize `RuntimeState`
- serialize `SchedulerSnapshot`
- serialize `WorkerExchangeObservable`
- serialize `WorkerRequest`
- serialize `WorkerResponse`

This codec can then be reused by:

- TCP inline payload mode
- shared-memory payload mode
- file fallback mode
- future testing tools

## Shared Memory Design

The recommended first shared-memory implementation is hybrid:

```text
control channel: TCP loopback
bulk RuntimeState payload: shared memory
```

This avoids redesigning process control while removing most large payload copies
from TCP.

### Why Hybrid

Control messages are small and benefit from TCP's simplicity:

- `RUN_BLOCK`
- `PROBE_OBSERVABLE`
- `SHUTDOWN`
- `ERROR`
- request ids
- payload references

`RuntimeState` is the large payload:

- coordinates
- velocities
- box
- step/time
- thermostat hidden state
- barostat hidden state
- RNG state
- enhanced-sampling state as it becomes serializable

Shared memory should target this bulk payload first.

### Shm Payload Store

Recommended abstraction:

```cpp
class ShmPayloadStore
{
   public:
    virtual ~ShmPayloadStore() = default;

    virtual WorkerPayloadRef Allocate(std::uint64_t size) = 0;
    virtual void Write(const WorkerPayloadRef& ref, const char* data,
                       std::uint64_t size) = 0;
    virtual std::string Read(const WorkerPayloadRef& ref) = 0;
    virtual void Release(const WorkerPayloadRef& ref) = 0;
};
```

Platform implementations:

- Linux/macOS: POSIX shared memory or memory-mapped temporary files
- Windows: file mapping objects

The public manager configuration should not expose these OS-specific details by
default.

### Slot Model

Use fixed or growable slots per worker.

Example:

```text
worker 0:
  request slot 0
  response slot 0

worker 1:
  request slot 1
  response slot 1
```

Each slot should carry a generation counter so stale references can be detected.

## Configuration Model

User-facing config should remain simple.

Recommended default:

```toml
[manager]
transport = "tcp"
```

Shared-memory mode:

```toml
[manager]
transport = "shm"
```

Optional advanced settings:

```toml
[transport.shm]
control = "tcp"
segment_size_mb = 512
slots_per_worker = 2
```

Semantics:

- `transport = "tcp"` means control and payload both use TCP.
- `transport = "shm"` means control uses TCP and large runtime-state payloads
  use shared memory.
- `transport = "file"` keeps the legacy file request/response path as a
  fallback.

The transport setting should be interpreted by the worker/session layer, not by
the core scheduler logic.

## Proposed Source Layout

Recommended target layout:

```text
SPONGE/worker_protocol/
  worker_session.h
  worker_session.cpp

  protocol_message.h
  protocol_message.cpp

  runtime_state_codec.h
  runtime_state_codec.cpp

  transport.h

  transports/
    tcp_transport.h
    tcp_transport.cpp
    shm_transport.h
    shm_transport.cpp
    file_transport.h
    file_transport.cpp

  child_process_session.h
  child_process_session.cpp
```

Current files can be migrated as follows:

```text
message_protocol.*  -> protocol_message.*
tcp_socket.*        -> transports/tcp_socket.*
tcp_protocol.*      -> transports/tcp_transport.*
file_protocol.*     -> runtime_state_codec.* + transports/file_transport.*
child_process_worker.* -> child_process_session.*
```

## Migration Plan

### Phase 1: Introduce WorkerSession

- add `WorkerSession` interface
- add `ChildProcessWorkerSession`
- move TCP child session ownership out of `Manager`
- make `Manager` call only `sessions_[i]->RunBlock(...)`
- preserve existing TCP/file behavior

Validation:

- direct `SPONGE` run still works
- `SPONGE_MANAGER --config` still works
- T/H/HT-REMD TCP smoke tests still pass
- file fallback smoke still passes

### Phase 2: Extract RuntimeState Codec

- move binary serialization helpers out of `file_protocol.cpp`
- create reusable request/response serialization APIs
- update TCP and file backends to use the shared codec
- keep wire compatibility unless there is a deliberate version bump

Validation:

- runtime-state roundtrip smoke still passes
- old file fallback still passes
- TCP REMD smoke still passes

### Phase 3: Introduce WorkerTransport

- add `WorkerTransport` interface
- implement `TcpTransport`
- implement `FileTransport` as fallback
- make `ChildProcessWorkerSession` depend on transport interface
- keep protocol message semantics transport-neutral

Validation:

- no TCP socket details in `Manager`
- no file path request/response details in `Manager`
- no transport-specific branches in REMD algorithms

### Phase 4: Add Shared-Memory Payload Store

- add `ShmPayloadStore`
- implement platform-specific shared memory layer
- add payload references with segment name, offset, size, and generation
- keep control messages on TCP
- support large `RuntimeState` payloads through shared memory

Validation:

- `transport = "shm"` runs a 2-worker T-REMD smoke
- `transport = "shm"` runs an H-REMD smoke with foreign-state probes
- stale generation or missing segment produces a clear error
- `transport = "tcp"` remains unchanged

### Phase 5: Benchmark TCP Versus Shm

Benchmark scenarios:

- 2 workers, 1 GPU, short block
- 2 workers, 1 GPU, longer block
- N workers, multi-GPU node if available
- large system with frequent exchange
- H/HT-REMD with probe requests

Expected result:

- `shm` should help most when `RuntimeState` is large and block length is
  short.
- `shm` may not help much when MD compute dominates.
- single-GPU multi-worker runs may still be limited by GPU contention rather
  than transport cost.

## Important Non-Goals For First Shm Version

The first shared-memory implementation should not attempt:

- CUDA IPC
- direct GPU pointer exchange
- cross-node shared memory
- zero-copy GPU-to-GPU state swap
- replacing TCP control messages

Those can be future optimizations after the host shared-memory transport is
stable.

## Open Questions

- Should probe workers use shared memory payloads, or remain TCP-only because
  they are one-shot and isolated?
- Should shared-memory segments be manager-owned or worker-owned?
- Should slots be fixed-size for speed or growable for memory efficiency?
- How should crash cleanup work on Windows and POSIX platforms?
- Should `transport = "shm"` silently fall back to TCP if shared memory is not
  available, or fail fast?

The recommended default is to fail fast for `transport = "shm"` if shared
memory cannot be initialized. Silent fallback can hide performance and
correctness issues.

## Current Implementation Status

Implemented in the current migration branch:

- `Manager` depends on the transport-neutral `WorkerSession` interface for
  schedule execution and observable probes.
- `RuntimeState` and worker request/response serialization live in
  `worker_protocol/runtime_state_codec.*`, shared by TCP, file fallback, and
  shared-memory transport.
- `WorkerTransport` is the byte/message boundary used by persistent
  child-process workers.
- `TcpTransport` carries inline payloads over TCP loopback.
- `ShmTransport` keeps TCP loopback as the control channel and moves non-empty
  serialized payloads through shared memory.
- POSIX platforms use `shm_open`/`mmap`; Windows uses named file mapping
  objects.
- `manager.transport = "file"` remains as the old one-shot migration fallback.

Still intentionally out of scope for this migration:

- CUDA IPC or direct GPU pointer exchange.
- Cross-node shared memory.
- Replacing TCP as the control channel.
- A specialized lightweight H/HT-REMD probe kernel.
