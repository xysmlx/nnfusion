#pragma once

typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long int int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
extern "C" int kernel_entry(float* Parameter_8_0, float** Result_31_0);
extern "C" void cuda_init();
extern "C" void cuda_free();
extern "C" __global__  void QuantizeDot_float_float_float_float_float_float_float_float_float_cuda_QuantizeDot_39(float* __restrict__ input0,float* __restrict__ input1,float* __restrict__ input2,float* __restrict__ input3,float* __restrict__ input4,float* __restrict__ input5,float* __restrict__ input6,float* __restrict__ input7, float* __restrict__ output0);
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

