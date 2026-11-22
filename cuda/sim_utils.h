#pragma once

#include <stdio.h>

#include "simulation.h"
#include <cuda_runtime.h>

#define SIM_TRY(expr) \
  do { \
    SimResult sim_try_result_ = (expr); \
    if (sim_try_result_ != SIM_RESULT_OKAY) { \
      return sim_try_result_; \
    } \
  } while(0)

#define CUDA_TRY(expr) \
  do { \
    cudaError_t cuda_try_error_ = (expr); \
    if (cuda_try_error_ != cudaSuccess) { \
       fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(cuda_try_error_)); \
       return SIM_RESULT_CUDA_ERROR; \
    }\
  } while(0)
