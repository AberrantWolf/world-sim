#pragma once

__device__ __forceinline__ int2 global_xy() {
    return make_int2(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);
}
