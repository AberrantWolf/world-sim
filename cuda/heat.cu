#include "heat.cuh"
#include "device_utils.cuh"

namespace {

__global__ void fill_rect_kern(
    float* __restrict__ buf,
    int32_t width,
    int32_t x0, int32_t y0,
    int32_t rect_w,
    int32_t rect_h,
    float value
) {
    const int2 p = global_xy();
    if (p.x >= rect_w || p.y >= rect_h) {
        return;
    }

    const int x = x0 + p.x;
    const int y = y0 + p.y;
    buf[y * width + x] = value;
}

__global__ void heat_step_kern(
  const float* __restrict__ in,
  float* __restrict__ out,
  int32_t width,
  int32_t height,
  float r
) {
    const int2 p = global_xy();

    if (p.x >= width || p.y >= height) {
        return;
    }

    const int x_left = max(p.x - 1, 0);
    const int x_right = min(p.x + 1, width - 1);
    const int y_up = max(p.y - 1, 0);
    const int y_down = min(p.y + 1, height - 1);

    const int target = p.y * width + p.x;

    const float c = in[target];
    const float w = in[p.y * width + x_left];
    const float e = in[p.y * width + x_right];
    const float n = in[y_up * width + p.x];
    const float s = in[y_down * width + p.x];

    out[target] = c + r * (n + e + s + w - 4 * c);
}

}

cudaError_t launch_fill_rect(
    float* buf, int32_t width,
    int32_t x0, int32_t y0, int32_t rect_w, int32_t rect_h,
    float value)
{
    const dim3 block(32, 8);
    const dim3 grid((rect_w + block.x - 1) / block.x,
                    (rect_h + block.y - 1) / block.y);

    fill_rect_kern<<<grid, block>>>(buf, width, x0, y0, rect_w, rect_h, value);

    return cudaGetLastError();
}

cudaError_t launch_heat_step(
    const float* in, float* out,
    int32_t width, int32_t height, float r)
{
    const dim3 block(32, 8);
    const dim3 grid((width + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);

    heat_step_kern<<<grid, block>>>(in, out, width, height, r);
    return cudaGetLastError();
}
