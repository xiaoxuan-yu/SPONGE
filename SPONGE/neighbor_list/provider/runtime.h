#pragma once

#include <cmath>
#include "../device_buffer.h"
#include "state.h"

namespace clustered_neighbor_runtime
{

struct RecorderScope
{
    TIME_RECORDER* time_recorder;

    explicit RecorderScope(TIME_RECORDER* recorder) : time_recorder(recorder)
    {
        if (time_recorder != NULL)
        {
            time_recorder->Start();
        }
    }

    ~RecorderScope()
    {
        if (time_recorder != NULL)
        {
            time_recorder->Stop();
        }
    }
};

inline int Note_Clustered_Step_Counter(int current_step, int* counter_step,
                                       int* count_this_step,
                                       long long* count_total)
{
    if (*counter_step != current_step)
    {
        *counter_step = current_step;
        *count_this_step = 0;
    }
    *count_this_step += 1;
    *count_total += 1;
    return *count_this_step;
}

template <typename T>
inline void Reserve_Device_Buffer(int capacity, DeviceBuffer<T>* buffer,
                                  const char* tag = "reserve-device-buffer")
{
    Reserve_Device_Buffer(capacity, &buffer->data, &buffer->capacity, tag);
}

inline void Reserve_Raw_Device_Workspace(
    size_t bytes, RawDeviceWorkspace* workspace,
    const char* tag = "reserve-raw-device-workspace")
{
    if (bytes == 0 || (bytes <= workspace->bytes && workspace->data != nullptr))
    {
        return;
    }
    if (workspace->data != nullptr)
    {
        Free_Single_Device_Pointer(&workspace->data);
    }
#ifndef USE_CPU
    Clustered_Device_Malloc_Safely(&workspace->data, bytes, tag);
#else
    (void)tag;
    Device_Malloc_Safely(&workspace->data, bytes);
#endif
    workspace->bytes = bytes;
}

template <typename T>
inline void Release_Device_Buffer(DeviceBuffer<T>* buffer)
{
    Free_Single_Device_Pointer(reinterpret_cast<void**>(&buffer->data));
    buffer->capacity = 0;
}

inline void Release_Raw_Device_Workspace(RawDeviceWorkspace* workspace)
{
    Free_Single_Device_Pointer(&workspace->data);
    workspace->bytes = 0;
}

#ifndef USE_CPU
inline int Normalize_Clustered_Working_Device(int working_device)
{
    int device_count = 0;
    deviceGetDeviceCount(&device_count);
    if (device_count <= 0)
    {
        return 0;
    }
    return working_device >= 0 && working_device < device_count ? working_device
                                                                : 0;
}

inline void Bind_Clustered_Working_Device(int* working_device)
{
    const int target_device = Normalize_Clustered_Working_Device(
        working_device != NULL ? *working_device : 0);
    if (working_device != NULL)
    {
        *working_device = target_device;
    }
    setWorkingDevice(target_device);
}
#endif

inline float Clustered_Minimum_Box_Face_Height(const LTMatrix3 rcell)
{
    const float reciprocal_a = sqrtf(
        rcell.a11 * rcell.a11 + rcell.a21 * rcell.a21 + rcell.a31 * rcell.a31);
    const float reciprocal_b =
        sqrtf(rcell.a22 * rcell.a22 + rcell.a32 * rcell.a32);
    const float reciprocal_c = fabsf(rcell.a33);
    if (reciprocal_a <= 0.0f || reciprocal_b <= 0.0f || reciprocal_c <= 0.0f)
    {
        return 0.0f;
    }
    return fminf(1.0f / reciprocal_a,
                 fminf(1.0f / reciprocal_b, 1.0f / reciprocal_c));
}

}  // namespace clustered_neighbor_runtime
