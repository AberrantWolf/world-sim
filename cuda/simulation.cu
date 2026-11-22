#include "simulation.h"
#include "device_buffer.hpp"

#include <memory>
#include <new>

#include "heat.cuh"

struct SimHandle {
    int32_t device_idx = -1;
    int32_t width = 0, height = 0;
    float r = 0.1f;
    DeviceBuffer d_buffers[2];
    uint8_t current_buffer = 0;

    DeviceBuffer& read() noexcept { return d_buffers[current_buffer]; }
    DeviceBuffer& write() noexcept { return d_buffers[current_buffer ^ 1]; }
    void swap() noexcept { current_buffer ^= 1; }
};

SimResult sim_init(const SimSettings* settings, SimHandle** out_handle) {
    if (settings == NULL || out_handle == NULL || settings->r <= 0 || settings->r > 0.25) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    *out_handle = NULL;

    const auto width = settings->width;
    const auto height = settings->height;
    const auto device = settings->device;
    const auto r = settings->r;
    if (width < 1 || height < 1 || device < 0) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    int device_count = 0;
    CUDA_TRY(cudaGetDeviceCount(&device_count));
    if (device >= device_count) {
        return SIM_RESULT_INVALID_DEVICE;
    }

    size_t count = (size_t)width * (size_t)height;

    std::unique_ptr<SimHandle> handle(new (std::nothrow) SimHandle());
    if (!handle) return SIM_RESULT_CREATION_ERROR;

    handle->device_idx = device;
    handle->width = width;
    handle->height = height;
    handle->r = r;
    SIM_TRY(DeviceBuffer::create(count, device, &handle->d_buffers[0]));
    SIM_TRY(DeviceBuffer::create(count, device, &handle->d_buffers[1]));

    SIM_TRY(handle->read().zero());
    SIM_TRY(handle->write().zero());

    *out_handle = handle.release();

    return SIM_RESULT_OKAY;
}

SimResult sim_fill_rect(SimHandle* handle, int32_t x, int32_t y, int32_t size_x, int32_t size_y, float value) {
    if (handle == NULL) {
        return SIM_RESULT_INVALID_HANDLE;
    }

    const auto width = handle->width;
    const auto height = handle->height;

    if (x < 0 || y < 0 || size_x < 1 || size_y < 1 || size_x > width - x || size_y > height - y) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const auto device = handle->device_idx;
    CUDA_TRY(cudaSetDevice(device));
    CUDA_TRY(launch_fill_rect(handle->read().data(), width, x, y, size_x, size_y, value));

    return SIM_RESULT_OKAY;
}

SimResult sim_step(SimHandle* handle, int32_t count) {
    if (handle == NULL) {
        return SIM_RESULT_INVALID_HANDLE;
    }

    if (count < 1) {
        return SIM_RESULT_OKAY;
    }

    const auto device = handle->device_idx;
    CUDA_TRY(cudaSetDevice(device));

    const auto width = handle->width;
    const auto height = handle->height;
    const auto r = handle->r;

    for (int32_t i = 0; i < count; ++i) {
        const auto in = handle->read().data();
        const auto out = handle->write().data();
        CUDA_TRY(launch_heat_step(in, out, width, height, r));

        // Swap the current buffer
        handle->swap();
    }

    CUDA_TRY(cudaDeviceSynchronize());

    return SIM_RESULT_OKAY;
}

void sim_destroy(SimHandle* handle) {
    delete handle;
}
