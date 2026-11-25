#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t SimResult;

#define SIM_RESULT_UNKNOWN_ERR      INT32_MAX
#define SIM_RESULT_OKAY             0
#define SIM_RESULT_INVALID_HANDLE   1
#define SIM_RESULT_INVALID_ARGUMENT 2
#define SIM_RESULT_CUDA_ERROR       3
#define SIM_RESULT_CREATION_ERROR   4
#define SIM_RESULT_INVALID_DEVICE   5

typedef struct SimSettings {
  int32_t device;
  int32_t width, height;
  float r;
} SimSettings;
static_assert(sizeof(SimSettings) == 16, "update ffi.rs SimSettings");

typedef struct SimHandle SimHandle;

// ===== sim_init
//
// Initialize a new simulation. Sets *out_handle to NULL on invalid settings
// or failure.
//
// Valid settings:
// - width > 0
// - height > 0
// - 0 <= device < number of GPUs
//
SimResult sim_init(const SimSettings *settings, SimHandle **out_handle);

// ===== fill_rect
//
// Fill a rectangular region of the simulation buffer to a specified value
//
// Valid arguments:
// - handle != NULL
// - 0 <= x < width
// - 0 <= y < height
// - size_x > 0
// - size_y > 0
// - x + size_x <= width
// - y + size_y <= height
SimResult sim_fill_rect(SimHandle *handle, int32_t x, int32_t y, int32_t size_x, int32_t size_y, float value);

// ===== sim_step
//
// Step the simulation. Does nothing if 'count' < 1. Blocks until the GPU is
// finished.
//
SimResult sim_step(SimHandle *handle, int32_t count);

// ===== sim_download
//
// Download the complete state of the simulation. Data copied out is row-major.
// Blocks on any pending GPU work and until the copy is complete.
//
// Valid arguments:
// - handle != NULL
// - dst has space for at least dst_count elements
// - dst_count == (width * height) from simulation creation
//
SimResult sim_download(const SimHandle *handle, float *dst, size_t dst_count);

// ===== sim_destroy
//
// Destroy the simulation. All handles associated with this simulation become
// invalid hereafter. Passing in a NULL handle is a no-op.
//
void sim_destroy(SimHandle *handle);

#ifdef __cplusplus
}
#endif
