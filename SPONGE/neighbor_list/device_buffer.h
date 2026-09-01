#pragma once

#include <cstdio>
#include <cstdlib>

#include "../common.h"
#include "../control.h"

namespace clustered_neighbor_runtime
{

#ifndef USE_CPU
inline void Clustered_Device_Malloc_Safely(void** pointer, size_t bytes,
                                           const char* tag)
{
    if (bytes == 0)
    {
        *pointer = NULL;
        return;
    }
    const deviceError_t error = deviceMalloc(pointer, bytes);
    if (error != DEVICE_MALLOC_SUCCESS)
    {
        int current_device = -1;
        cudaGetDevice(&current_device);
        fprintf(stderr,
                "clustered alloc failed: tag=%s bytes=%zu device=%d error=%s\n",
                tag, bytes, current_device, deviceGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#else
inline void Clustered_Device_Malloc_Safely(void** pointer, size_t bytes,
                                           const char*)
{
    Device_Malloc_Safely(pointer, bytes);
}
#endif

template <typename T>
inline void Reserve_Device_Buffer(int capacity, T** pointer,
                                  int* current_capacity,
                                  const char* tag = "reserve-device-buffer")
{
    if (capacity <= *current_capacity && *pointer != NULL)
    {
        return;
    }
    if (*pointer != NULL)
    {
        Free_Single_Device_Pointer(reinterpret_cast<void**>(pointer));
    }
#ifndef USE_CPU
    Clustered_Device_Malloc_Safely(reinterpret_cast<void**>(pointer),
                                   sizeof(T) * capacity, tag);
#else
    (void)tag;
    Device_Malloc_Safely(reinterpret_cast<void**>(pointer),
                         sizeof(T) * capacity);
#endif
    *current_capacity = capacity;
}

}  // namespace clustered_neighbor_runtime
