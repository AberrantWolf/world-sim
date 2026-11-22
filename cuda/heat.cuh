#pragma once
#include <cstdint>
#include <cuda_runtime.h>

cudaError_t launch_fill_rect(float* buf, int32_t width,
                             int32_t x0, int32_t y0, int32_t rect_w, int32_t rect_h,
                             float value);

cudaError_t launch_heat_step(const float* in, float* out,
                             int32_t width, int32_t height, float r);
