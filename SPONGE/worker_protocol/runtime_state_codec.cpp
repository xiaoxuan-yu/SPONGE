#include "runtime_state_codec.h"

#include <array>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace sponge::worker_protocol
{

namespace
{

template <typename T>
void WritePod(std::ostream* out, const T& value)
{
    static_assert(std::is_trivially_copyable<T>::value,
                  "WritePod requires trivially copyable type");
    out->write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T ReadPod(std::istream* in)
{
    static_assert(std::is_trivially_copyable<T>::value,
                  "ReadPod requires trivially copyable type");
    T value{};
    in->read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!(*in))
    {
        throw std::runtime_error("failed to read POD value from worker codec");
    }
    return value;
}

void WriteString(std::ostream* out, const std::string& value)
{
    const std::uint64_t size = value.size();
    WritePod(out, size);
    if (size > 0)
    {
        out->write(value.data(), static_cast<std::streamsize>(size));
    }
}

std::string ReadString(std::istream* in)
{
    const auto size = ReadPod<std::uint64_t>(in);
    std::string value(size, '\0');
    if (size > 0)
    {
        in->read(value.data(), static_cast<std::streamsize>(size));
        if (!(*in))
        {
            throw std::runtime_error(
                "failed to read string payload from worker codec");
        }
    }
    return value;
}

template <typename T>
void WriteVector(std::ostream* out, const std::vector<T>& values)
{
    const std::uint64_t size = values.size();
    WritePod(out, size);
    for (const auto& value : values)
    {
        WritePod(out, value);
    }
}

template <typename T>
std::vector<T> ReadVector(std::istream* in)
{
    const auto size = ReadPod<std::uint64_t>(in);
    std::vector<T> values(size);
    for (std::uint64_t i = 0; i < size; i++)
    {
        values[i] = ReadPod<T>(in);
    }
    return values;
}

template <typename T, std::size_t N>
void WriteArray(std::ostream* out, const std::array<T, N>& values)
{
    for (const auto& value : values)
    {
        WritePod(out, value);
    }
}

template <typename T, std::size_t N>
std::array<T, N> ReadArray(std::istream* in)
{
    std::array<T, N> values{};
    for (std::size_t i = 0; i < N; i++)
    {
        values[i] = ReadPod<T>(in);
    }
    return values;
}

void WriteRuntimeState(std::ostream* out, const sponge::RuntimeState& state)
{
    WritePod(out, state.atom_count);
    WritePod(out, state.step);
    WritePod(out, state.step_limit);
    WritePod(out, state.start_time_ps);
    WritePod(out, state.current_time_ps);
    WriteArray(out, state.box_length);
    WriteArray(out, state.box_angle);
    WriteVector(out, state.coordinates);
    WriteVector(out, state.velocities);
    WriteVector(out, state.local_accelerations);
    WriteVector(out, state.nhc_coordinates);
    WriteVector(out, state.nhc_velocities);
    WriteVector(out, state.settle_last_pair_ab);
    WriteVector(out, state.settle_last_triangle_ba);
    WriteVector(out, state.settle_last_triangle_ca);
    WriteVector(out, state.shake_last_pair_dr);
    WriteArray(out, state.pressure_barostat_g);
    WritePod(out, state.pressure_barostat_v0);
    WriteString(out, state.pressure_barostat_rng_state);
    WriteString(out, state.pressure_barostat_distribution_state);
    WriteArray(out, state.mc_barostat_total_count);
    WriteArray(out, state.mc_barostat_accept_count);
    WriteArray(out, state.mc_barostat_accept_rate);
    WriteArray(out, state.mc_barostat_delta_box_length_max);
    WriteString(out, state.mc_barostat_rng_state);
    WriteVector(out, state.middle_langevin_rng_state);
    WriteVector(out, state.andersen_rng_state);
    WriteString(out, state.bussi_rng_state);
    WriteString(out, state.bussi_distribution_state);
    WritePod(out, state.has_local_accelerations);
    WritePod(out, state.has_nhc_state);
    WritePod(out, state.has_settle_state);
    WritePod(out, state.has_shake_state);
    WritePod(out, state.has_pressure_barostat_state);
    WritePod(out, state.has_mc_barostat_state);
    WritePod(out, state.has_middle_langevin_rng_state);
    WritePod(out, state.has_andersen_rng_state);
    WritePod(out, state.has_bussi_rng_state);
    WritePod(out, state.valid);
}

sponge::RuntimeState ReadRuntimeState(std::istream* in)
{
    sponge::RuntimeState state;
    state.atom_count = ReadPod<int>(in);
    state.step = ReadPod<int>(in);
    state.step_limit = ReadPod<int>(in);
    state.start_time_ps = ReadPod<double>(in);
    state.current_time_ps = ReadPod<double>(in);
    state.box_length = ReadArray<float, 3>(in);
    state.box_angle = ReadArray<float, 3>(in);
    state.coordinates = ReadVector<sponge::RuntimeStateAtom>(in);
    state.velocities = ReadVector<sponge::RuntimeStateAtom>(in);
    state.local_accelerations = ReadVector<sponge::RuntimeStateAtom>(in);
    state.nhc_coordinates = ReadVector<float>(in);
    state.nhc_velocities = ReadVector<float>(in);
    state.settle_last_pair_ab = ReadVector<sponge::RuntimeStateAtom>(in);
    state.settle_last_triangle_ba = ReadVector<sponge::RuntimeStateAtom>(in);
    state.settle_last_triangle_ca = ReadVector<sponge::RuntimeStateAtom>(in);
    state.shake_last_pair_dr = ReadVector<sponge::RuntimeStateAtom>(in);
    state.pressure_barostat_g = ReadArray<float, 6>(in);
    state.pressure_barostat_v0 = ReadPod<float>(in);
    state.pressure_barostat_rng_state = ReadString(in);
    state.pressure_barostat_distribution_state = ReadString(in);
    state.mc_barostat_total_count = ReadArray<int, 3>(in);
    state.mc_barostat_accept_count = ReadArray<int, 3>(in);
    state.mc_barostat_accept_rate = ReadArray<float, 3>(in);
    state.mc_barostat_delta_box_length_max = ReadArray<float, 3>(in);
    state.mc_barostat_rng_state = ReadString(in);
    state.middle_langevin_rng_state = ReadVector<std::uint8_t>(in);
    state.andersen_rng_state = ReadVector<std::uint8_t>(in);
    state.bussi_rng_state = ReadString(in);
    state.bussi_distribution_state = ReadString(in);
    state.has_local_accelerations = ReadPod<bool>(in);
    state.has_nhc_state = ReadPod<bool>(in);
    state.has_settle_state = ReadPod<bool>(in);
    state.has_shake_state = ReadPod<bool>(in);
    state.has_pressure_barostat_state = ReadPod<bool>(in);
    state.has_mc_barostat_state = ReadPod<bool>(in);
    state.has_middle_langevin_rng_state = ReadPod<bool>(in);
    state.has_andersen_rng_state = ReadPod<bool>(in);
    state.has_bussi_rng_state = ReadPod<bool>(in);
    state.valid = ReadPod<bool>(in);
    return state;
}

void WriteSchedulerSnapshot(std::ostream* out,
                            const sponge::SchedulerSnapshot& snapshot)
{
    WritePod(out, snapshot.next_step);
    WritePod(out, snapshot.last_completed_step);
    WritePod(out, snapshot.step_limit);
    WritePod(out, snapshot.current_time_ps);
    WritePod(out, snapshot.dt_ps);
    WritePod(out, snapshot.temperature);
    WritePod(out, snapshot.target_temperature);
    WritePod(out, snapshot.pressure);
    WritePod(out, snapshot.target_pressure);
    WritePod(out, snapshot.total_potential);
    WritePod(out, snapshot.effective_potential);
    WriteArray(out, snapshot.box_length);
    WritePod(out, snapshot.initialized);
    WritePod(out, snapshot.finished);
}

sponge::SchedulerSnapshot ReadSchedulerSnapshot(std::istream* in)
{
    sponge::SchedulerSnapshot snapshot;
    snapshot.next_step = ReadPod<int>(in);
    snapshot.last_completed_step = ReadPod<int>(in);
    snapshot.step_limit = ReadPod<int>(in);
    snapshot.current_time_ps = ReadPod<double>(in);
    snapshot.dt_ps = ReadPod<double>(in);
    snapshot.temperature = ReadPod<float>(in);
    snapshot.target_temperature = ReadPod<float>(in);
    snapshot.pressure = ReadPod<float>(in);
    snapshot.target_pressure = ReadPod<float>(in);
    snapshot.total_potential = ReadPod<float>(in);
    snapshot.effective_potential = ReadPod<float>(in);
    snapshot.box_length = ReadArray<float, 3>(in);
    snapshot.initialized = ReadPod<bool>(in);
    snapshot.finished = ReadPod<bool>(in);
    return snapshot;
}

void WriteWorkerExchangeObservable(std::ostream* out,
                                   const sponge::WorkerExchangeObservable& o)
{
    WritePod(out, o.step);
    WritePod(out, o.time_ps);
    WritePod(out, o.total_potential);
    WritePod(out, o.effective_potential);
    WritePod(out, o.temperature);
    WritePod(out, o.target_temperature);
    WritePod(out, o.pressure);
    WritePod(out, o.target_pressure);
    WritePod(out, o.volume);
}

sponge::WorkerExchangeObservable ReadWorkerExchangeObservable(std::istream* in)
{
    sponge::WorkerExchangeObservable observable;
    observable.step = ReadPod<int>(in);
    observable.time_ps = ReadPod<double>(in);
    observable.total_potential = ReadPod<float>(in);
    observable.effective_potential = ReadPod<float>(in);
    observable.temperature = ReadPod<float>(in);
    observable.target_temperature = ReadPod<float>(in);
    observable.pressure = ReadPod<float>(in);
    observable.target_pressure = ReadPod<float>(in);
    observable.volume = ReadPod<float>(in);
    return observable;
}

}  // namespace

void WriteWorkerRequest(std::ostream* out, const WorkerFileRequest& request)
{
    WritePod(out, request.steps);
    WritePod(out, request.managed_step_limit);
    WritePod(out, request.emit_output);
    WritePod(out, request.probe_only);
    WritePod(out, request.has_runtime_state);
    WriteRuntimeState(out, request.runtime_state);
}

WorkerFileRequest ReadWorkerRequest(std::istream* in)
{
    WorkerFileRequest request;
    request.steps = ReadPod<int>(in);
    request.managed_step_limit = ReadPod<int>(in);
    request.emit_output = ReadPod<bool>(in);
    request.probe_only = ReadPod<bool>(in);
    request.has_runtime_state = ReadPod<bool>(in);
    request.runtime_state = ReadRuntimeState(in);
    return request;
}

void WriteWorkerResponse(std::ostream* out, const WorkerFileResponse& response)
{
    WriteSchedulerSnapshot(out, response.execution.snapshot);
    WriteWorkerExchangeObservable(out, response.execution.observable);
    WriteRuntimeState(out, response.execution.runtime_state);
    WritePod(out, response.execution.finished);
}

WorkerFileResponse ReadWorkerResponse(std::istream* in)
{
    WorkerFileResponse response;
    response.execution.snapshot = ReadSchedulerSnapshot(in);
    response.execution.observable = ReadWorkerExchangeObservable(in);
    response.execution.runtime_state = ReadRuntimeState(in);
    response.execution.finished = ReadPod<bool>(in);
    return response;
}

std::string SerializeWorkerRequest(const WorkerFileRequest& request)
{
    std::ostringstream out;
    WriteWorkerRequest(&out, request);
    return out.str();
}

WorkerFileRequest DeserializeWorkerRequest(const std::string& payload)
{
    std::istringstream in(payload);
    return ReadWorkerRequest(&in);
}

std::string SerializeWorkerResponse(const WorkerFileResponse& response)
{
    std::ostringstream out;
    WriteWorkerResponse(&out, response);
    return out.str();
}

WorkerFileResponse DeserializeWorkerResponse(const std::string& payload)
{
    std::istringstream in(payload);
    return ReadWorkerResponse(&in);
}

}  // namespace sponge::worker_protocol
