#pragma once

#include "sim_utils.h"
#include "simulation.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include <cuda_runtime.h>

class DeviceBuffer {
private:
  int32_t device_ = -1;
  size_t count_ = 0;
  float *d_buffer_ = nullptr;

  DeviceBuffer(float *ptr, size_t count, int32_t device) noexcept
      : device_(device), count_(count), d_buffer_(ptr) {
  }

public:
  [[nodiscard]] static SimResult create(const size_t count,
                                        const int32_t device_idx,
                                        DeviceBuffer *out) noexcept {
    if (out == nullptr || count < 1 || count > (SIZE_MAX / sizeof(float))) {
      return SIM_RESULT_INVALID_ARGUMENT;
    }

    float *data = nullptr;
    const size_t bytes = count * sizeof(float);

    CUDA_TRY(cudaSetDevice(device_idx));
    CUDA_TRY(cudaMalloc(&data, bytes));

    *out = DeviceBuffer{data, count, device_idx};
    return SIM_RESULT_OKAY;
  }

  DeviceBuffer() noexcept = default;

  // No copying
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer& operator=(const DeviceBuffer &) = delete;

  // Moving is okay
  DeviceBuffer(DeviceBuffer&& other) noexcept : device_(other.device_), count_(other.count_), d_buffer_(other.d_buffer_) {
      other.d_buffer_ = nullptr;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
      std::swap(device_, other.device_);
      std::swap(count_, other.count_);
      std::swap(d_buffer_, other.d_buffer_);

      return *this;
  }

  ~DeviceBuffer() noexcept {
    if (d_buffer_ != nullptr) {
      cudaSetDevice(device_);
      (void)cudaFree(d_buffer_);
    }
  }

  float* data() const noexcept {
      return d_buffer_;
  }

  size_t count() const noexcept {
      return count_;
  }

  size_t size() const noexcept {
      return count_ * sizeof(float);
  }

  SimResult zero() noexcept {
      CUDA_TRY(cudaSetDevice(device_));
      CUDA_TRY(cudaMemset(data(), 0, size()));

      return SIM_RESULT_OKAY;
  }
};
