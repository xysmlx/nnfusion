#include <cudnn.h>
#include <cublas_v2.h>
#include <sstream>
#include <assert.h>
#include <stdexcept>
#include <fstream>
#include <cuda.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "nnfusion_rt.h"
#ifndef __HALF_COMPARE_EX__
#define __HALF_COMPARE_EX__
inline __device__ half max(half x, half y) { return x > y ? x : y; }
inline __device__ half min(half x, half y) { return x < y ? x : y; }
#endif
#define CUDA_SAFE_CALL(x)                                                                          \
    do                                                                                             \
    {                                                                                              \
        cudaError_t result = (x);                                                                  \
        if (result != cudaSuccess)                                                                 \
        {                                                                                          \
            const char* msg = cudaGetErrorString(result);                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #x " failed with error"                                    \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
#define CUDNN_SAFE_CALL(func)                                                                      \
    do                                                                                             \
    {                                                                                              \
        cudnnStatus_t e = (func);                                                                  \
        if (e != CUDNN_STATUS_SUCCESS)                                                             \
        {                                                                                          \
            const char* msg = cudnnGetErrorString(e);                                              \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #func " failed with error"                                 \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
#define CUBLAS_SAFE_CALL(func)                                                                     \
    do                                                                                             \
    {                                                                                              \
        cublasStatus_t e = (func);                                                                 \
        if (e != CUBLAS_STATUS_SUCCESS)                                                            \
        {                                                                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #func " failed with error"                                 \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << e;    \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
using namespace nvcuda;
char* group_0_CUDA_GPU0_allocator_memory_pool;
float* Reshape_10_0;
float* BitConverter_34_0;
float* Add_15_0;
float* Add_20_0;
float* Add_25_0;
float* Dot_26_0;
float* Multiply_28_0;
float* Broadcast_29_0;
float* tensor_30;
float* Result_31_0;
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}
__device__ __forceinline__ float mul(float x0, float x1)
{
    return x0 * x1;
}
__device__ __forceinline__ int division_by_invariant_multiplication(int value, int magic, int shift)
{
    int result;
    asm("{\n\t"
        ".reg .pred p;\n\t"
        ".reg .u64 res64;\n\t"
        ".reg .u32 lo32, hi32;\n\t"
        "setp.ne.s32 p, %2, 1;\n\t"
        "mul.wide.u32 res64, %1, %2;\n\t"
        "mov.b64 {lo32, hi32}, res64;\n\t"
        "selp.u32 hi32, hi32, %1, p;\n\t"
        "shr.u32 %0, hi32, %3;\n\t"
        "}" : "=r"(result) : "r"(value), "r"(magic), "r"(shift));
    return result;
}
__device__ __forceinline__ char  load(const char*  __restrict__ in, int i=0, bool b=true)
{
    char v = 0;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
} 
__device__ __forceinline__ float  load(const float*  __restrict__ in, int i=0, bool b=true)
{
    float v = 0.0f;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
__device__ __forceinline__ half  load(const half*  __restrict__ in, int i=0, bool b=true)
{
    half v = 0.0f;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
__device__ __forceinline__ int32_t  load(const int32_t*  __restrict__ in, int i=0, bool b=true)
{
    int32_t v = 0;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
__device__ __forceinline__ int64_t  load(const int64_t*  __restrict__ in, int i=0, bool b=true)
{
    int64_t v = 0;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
cublasHandle_t cublas_handle_0;
char* group_persist_CUDA_GPU0_allocator_memory_pool;
float* Constant_7_0;
float* Constant_40_0;
float* Constant_39_0;
float* Constant_38_0;
float* Constant_37_0;
float* Constant_36_0;
float* Constant_35_0;
float* Constant_33_0;
float* Constant_32_0;
float* Constant_47_0;
float* Constant_46_0;
float* Constant_45_0;
float* Constant_44_0;
float* Constant_43_0;
float* Constant_42_0;
float* Constant_54_0;
float* Constant_53_0;
float* Constant_52_0;
float* Constant_51_0;
float* Constant_50_0;
float* Constant_49_0;
float* Constant_6_0;
float* Constant_27_0;

typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long int int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;
// Node name:	Result_31
// Description:	Result
// Input:
//	- name: tensor_30	type: float	shape: Shape{1024, 10}
// Output:
//	- name: Result_31_0	type: float	shape: Shape{1024, 10}
void Result_float_float_cuda_lib_Result_31(float* input0, float** output0)
{
    *output0 = input0;
}
// Node name:	Constant_52
// Description:	Constant
// Input:
// Output:
//	- name: Constant_52_0	type: float	shape: Shape{1}
void Constant_float_cuda_Constant_52(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_52_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_52_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_45
// Description:	Constant
// Input:
// Output:
//	- name: Constant_45_0	type: float	shape: Shape{1}
void Constant_float_cuda_Constant_45(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_45_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_45_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_6
// Description:	Constant
// Input:
// Output:
//	- name: Constant_6_0	type: float	shape: Shape{10}
void Constant_float_cuda_Constant_6(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_6_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_6_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[40];
    bin_file.read(tmp_mem, 40);
    cudaMemcpyAsync(output0, tmp_mem, 40, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_42
// Description:	Constant
// Input:
// Output:
//	- name: Constant_42_0	type: float	shape: Shape{1024, 1024}
void Constant_float_cuda_Constant_42(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_42_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_42_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_53
// Description:	Constant
// Input:
// Output:
//	- name: Constant_53_0	type: float	shape: Shape{1}
void Constant_float_cuda_Constant_53(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_53_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_53_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_43
// Description:	Constant
// Input:
// Output:
//	- name: Constant_43_0	type: float	shape: Shape{1024, 1024}
void Constant_float_cuda_Constant_43(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_43_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_43_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_32
// Description:	Constant
// Input:
// Output:
//	- name: Constant_32_0	type: float	shape: Shape{1}
void Constant_float_cuda_Constant_32(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_32_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_32_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_35
// Description:	Constant
// Input:
// Output:
//	- name: Constant_35_0	type: float	shape: Shape{1024, 1024}
void Constant_float_cuda_Constant_35(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_35_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_35_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_38
// Description:	Constant
// Input:
// Output:
//	- name: Constant_38_0	type: float	shape: Shape{1}
void Constant_float_cuda_Constant_38(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_38_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_38_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_39
// Description:	Constant
// Input:
// Output:
//	- name: Constant_39_0	type: float	shape: Shape{1}
void Constant_float_cuda_Constant_39(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_39_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_39_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_46
// Description:	Constant
// Input:
// Output:
//	- name: Constant_46_0	type: float	shape: Shape{1}
void Constant_float_cuda_Constant_46(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_46_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_46_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_40
// Description:	Constant
// Input:
// Output:
//	- name: Constant_40_0	type: float	shape: Shape{1048576}
void Constant_float_cuda_Constant_40(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_40_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_40_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_36
// Description:	Constant
// Input:
// Output:
//	- name: Constant_36_0	type: float	shape: Shape{1024, 1024}
void Constant_float_cuda_Constant_36(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_36_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_36_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_44
// Description:	Constant
// Input:
// Output:
//	- name: Constant_44_0	type: float	shape: Shape{1024, 1024}
void Constant_float_cuda_Constant_44(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_44_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_44_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_27
// Description:	Constant
// Input:
// Output:
//	- name: Constant_27_0	type: float	shape: Shape{10}
void Constant_float_cuda_Constant_27(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_27_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_27_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[40];
    bin_file.read(tmp_mem, 40);
    cudaMemcpyAsync(output0, tmp_mem, 40, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_49
// Description:	Constant
// Input:
// Output:
//	- name: Constant_49_0	type: float	shape: Shape{512, 1024}
void Constant_float_cuda_Constant_49(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_49_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_49_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2097152];
    bin_file.read(tmp_mem, 2097152);
    cudaMemcpyAsync(output0, tmp_mem, 2097152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_37
// Description:	Constant
// Input:
// Output:
//	- name: Constant_37_0	type: float	shape: Shape{1024, 1024}
void Constant_float_cuda_Constant_37(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_37_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_37_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_47
// Description:	Constant
// Input:
// Output:
//	- name: Constant_47_0	type: float	shape: Shape{1048576}
void Constant_float_cuda_Constant_47(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_47_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_47_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4194304];
    bin_file.read(tmp_mem, 4194304);
    cudaMemcpyAsync(output0, tmp_mem, 4194304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_51
// Description:	Constant
// Input:
// Output:
//	- name: Constant_51_0	type: float	shape: Shape{512, 1024}
void Constant_float_cuda_Constant_51(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_51_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_51_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2097152];
    bin_file.read(tmp_mem, 2097152);
    cudaMemcpyAsync(output0, tmp_mem, 2097152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_33
// Description:	Constant
// Input:
// Output:
//	- name: Constant_33_0	type: float	shape: Shape{1}
void Constant_float_cuda_Constant_33(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_33_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_33_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_7
// Description:	Constant
// Input:
// Output:
//	- name: Constant_7_0	type: float	shape: Shape{10, 512}
void Constant_float_cuda_Constant_7(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_7_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_7_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[20480];
    bin_file.read(tmp_mem, 20480);
    cudaMemcpyAsync(output0, tmp_mem, 20480, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_50
// Description:	Constant
// Input:
// Output:
//	- name: Constant_50_0	type: float	shape: Shape{512, 1024}
void Constant_float_cuda_Constant_50(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_50_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_50_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2097152];
    bin_file.read(tmp_mem, 2097152);
    cudaMemcpyAsync(output0, tmp_mem, 2097152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_54
// Description:	Constant
// Input:
// Output:
//	- name: Constant_54_0	type: float	shape: Shape{524288}
void Constant_float_cuda_Constant_54(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_54_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_54_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2097152];
    bin_file.read(tmp_mem, 2097152);
    cudaMemcpyAsync(output0, tmp_mem, 2097152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	QuantizeDot_55
// Description:	QuantizeDot
// Input:
//	- name: Add_20_0	type: float	shape: Shape{1024, 1024}
//	- name: Constant_49_0	type: float	shape: Shape{512, 1024}
//	- name: Constant_50_0	type: float	shape: Shape{512, 1024}
//	- name: Constant_51_0	type: float	shape: Shape{512, 1024}
//	- name: Constant_52_0	type: float	shape: Shape{1}
//	- name: Constant_53_0	type: float	shape: Shape{1}
//	- name: Constant_54_0	type: float	shape: Shape{524288}
// Output:
//	- name: Add_25_0	type: float	shape: Shape{1024, 512}
extern "C" __global__  void QuantizeDot_float_float_float_float_float_float_float_float_cuda_QuantizeDot_55(float* __restrict__ input0,float* __restrict__ input1,float* __restrict__ input2,float* __restrict__ input3,float* __restrict__ input4,float* __restrict__ input5,float* __restrict__ input6, float* __restrict__ output0)
{
    {

        const unsigned int M_GLOBAL=1024;
        const unsigned int N_GLOBAL=512;
        const unsigned int K_GLOBAL=1024;
        // const parameters
        const unsigned int  WARP_SIZE=32;
        const unsigned int  M=16;
        const unsigned int  N=16;
        const unsigned int  K=16;
        const unsigned int  WMMA_M=16;
        const unsigned int  WMMA_N=16;
        const unsigned int  WMMA_K=16;

        const unsigned int  M_TILES=64;
        const unsigned int  N_TILES=32;
        const unsigned int  K_TILES=64;


        // typedef C_LAYOUT wmma::mem_row_major;

        const unsigned int WARPS_PER_BLOCK=8;
        const unsigned int THREADS_PER_BLOCK= (WARP_SIZE * WARPS_PER_BLOCK);
        const unsigned int CHUNK_K=8;
        const unsigned int CHUNK_LINE_BYTES=(CHUNK_K * K * sizeof(uint8_t));
        const unsigned int WARP_COPY_BYTES=(WARP_SIZE * sizeof(int4));
        const unsigned int CHUNK_COPY_LINES_PER_WARP=(WARP_COPY_BYTES / CHUNK_LINE_BYTES);
        const unsigned int CHUNK_COPY_LINE_LANES=(WARP_SIZE / CHUNK_COPY_LINES_PER_WARP);
        
        const unsigned int BLOCK_ROW_WARPS=2;
        const unsigned int BLOCK_COL_WARPS=4;
        
        const unsigned int WARP_ROW_TILES =4;
        const unsigned int WARP_COL_TILES =2;
        
        const unsigned int BLOCK_ROW_TILES =(WARP_ROW_TILES * BLOCK_ROW_WARPS);
        const unsigned int BLOCK_COL_TILES =(WARP_COL_TILES * BLOCK_COL_WARPS);
        
        const unsigned int GLOBAL_MEM_STRIDE =N_GLOBAL;
        
        const unsigned int SHMEM_STRIDE=(N * BLOCK_ROW_TILES);
        const unsigned int SHMEM_OFFSET=(N * WARP_ROW_TILES);
        
        const unsigned int SKEW_UINT8=32;
        

        // Convert the input pointers
        const uint8_t * A = reinterpret_cast<uint8_t*>(input0); // activation
        const uint8_t * B_val =  reinterpret_cast<uint8_t*>(input1); // weight
        const int * B_row = reinterpret_cast< int *>(input2);
        const int * B_col = reinterpret_cast< int *>(input3);
        const float scale_s1s2 = (float)(*input4);
        const int * C = reinterpret_cast< int *>(input6);
        float * D = reinterpret_cast<float*>(output0);


        extern __shared__ uint8_t shmem[][CHUNK_K * K + SKEW_UINT8];

        // Warp and lane identification.
        const unsigned int warpId = threadIdx.x / WARP_SIZE;
        const unsigned int laneId = threadIdx.x % WARP_SIZE;
      
        // Offset in shared memory from which the B matrix is stored.
        const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;       // BLOCK_COL_TILES * M is shared_A row numbers in one block
      
        // This pointer is used to access the C and D matrix tiles this warp computes.
        int *shmem_warp_tile_ptr = (int *)&shmem[0][0] +
                                   (warpId / 2) * SHMEM_STRIDE * K * 2 +    // K * 2 is because one warp calculate k * 2 rows.
                                   (warpId % 2) * SHMEM_OFFSET;
      
        // This pointer is used to stream the C and D matrices block-wide tile to and
        // from shared memory.
        int *shmem_warp_stream_ptr = (int *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;   // confuse, may be used to read from global memory?
      
        // Each CTA slides along the 128 x 128 tiles from the top left corner of the
        // matrix to the right and down, and selects the next tile to compute. Once
        // there's no such tile, all warps in this CTA exit.
        for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
          const unsigned int block_tile_i =                                   // get the i (row) index of all tiles
              ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
          const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;
      
          // Stop when there are no more D matrix tiles to compute in this CTA.
          if (block_tile_i >= M_TILES) {
            break;
          }
      
          // This warp's pointer to the C matrix data to copy memory from to shared
          // memory.
          const size_t gmem_idx =
              (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
          const int *src_gmem_warp_stream_ptr = &C[gmem_idx];
      
          // Stream multiple C tiles to shared memory.
      #pragma unroll
          for (int i = 0; i < K; i++) {
            typedef int4 copy_t;
      
            *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
                *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
                  laneId);
          }
      
          __syncthreads();
      
          // These fragments will accumulate the result of A and B matrix fragment
          // multiplications along the K_GLOBAL dimension.
          wmma::fragment<wmma::accumulator, M, N, K, int> c[WARP_COL_TILES]
                                                           [WARP_ROW_TILES];
      
          // Load the C matrix tiles into fragments from shared memory.
      #pragma unroll
          for (int i = 0; i < WARP_COL_TILES; i++) {
      #pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
              const int *tile_ptr =
                  shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;
      
              wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, wmma::mem_row_major);
            }
          }
      
          __syncthreads();
      
          // Select what warp copies what matrix to shared memory.
          // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
          int start_tile = B_row[block_tile_j / 2 + (warpId % 4)];
          int end_tile = B_row[block_tile_j / 2 + (warpId % 4) + 1];
          // Go through the global K dimension by a fixed step at a time.
      #pragma unroll
          for(int tile_k_idx = start_tile; tile_k_idx < end_tile; tile_k_idx++){
            size_t shmem_idx = 
              warpId < (WARPS_PER_BLOCK / 2)
                ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
                : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);
      
            int4 *lane_ptr = NULL;
            if(warpId < 4){
              const uint8_t *warp_ptr = &A[block_tile_i * M * K_GLOBAL] +
                M * K_GLOBAL * (warpId % 4) * 2;
              lane_ptr = (int4 *)(warp_ptr + B_col[tile_k_idx] * K * CHUNK_K +
                                  (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
                        (laneId % CHUNK_COPY_LINE_LANES);
            }else{
              const uint8_t *warp_ptr = B_val + tile_k_idx * (M * WARP_COL_TILES) * (K * CHUNK_K);
              lane_ptr = (int4 *)(warp_ptr + (laneId / CHUNK_COPY_LINE_LANES) * (K * CHUNK_K)) + (laneId % CHUNK_COPY_LINE_LANES);
            }
            // Shift the second half of the warp to the next row / column in the
            // shared memory.
            shmem_idx += laneId / CHUNK_COPY_LINE_LANES;
      
            #pragma unroll
            for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
                 i++) {
              // Copy 16 bytes at once in each lane.
              *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
                  *lane_ptr;
      
              // Advance the global memory pointer and the shared memory index.
              lane_ptr = warpId < 4 ?
                          (int4 *)((uint8_t *)lane_ptr +
                                              K_GLOBAL * CHUNK_COPY_LINES_PER_WARP):
                          (int4 *)((uint8_t *)lane_ptr +
                                              K * CHUNK_K * CHUNK_COPY_LINES_PER_WARP);
              shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }
      
            __syncthreads();
      
            for (int k_step = 0; k_step < CHUNK_K; k_step++) {
              wmma::fragment<wmma::matrix_a, M, N, K, uint8_t, wmma::row_major>
                  a[WARP_COL_TILES];
              wmma::fragment<wmma::matrix_b, M, N, K, uint8_t, wmma::col_major>
                  b[WARP_ROW_TILES];
      
      #pragma unroll
              for (int i = 0; i < WARP_COL_TILES; i++) {
                size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
                const uint8_t *tile_ptr = &shmem[shmem_idx_a][k_step * K];
      
                wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_UINT8);
      
      #pragma unroll
                for (int j = 0; j < WARP_ROW_TILES; j++) {
                  if (i == 0) {
                    // Load the B matrix fragment once, because it is going to be
                    // reused against the other A matrix fragments.
                    size_t shmem_idx_b = shmem_idx_b_off +
                                         (WARP_ROW_TILES * N) * (warpId % 2) +
                                         (j * N);
                    const uint8_t *tile_ptr = &shmem[shmem_idx_b][k_step * K];
      
                    wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_UINT8);
                  }
      
                  wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                }
              }
            }
      
            __syncthreads();
          }
      
            // Store the D fragments to shared memory.
      #pragma unroll
          for (int i = 0; i < WARP_COL_TILES; i++) {
      #pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
              // Uniform, point-wise transformations of ALL fragment elements by ALL
              // threads in the warp are well-defined even though element indices
              // within fragment storage are not defined.
      
              int *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;
      
              wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, wmma::mem_row_major);
            }
          }
      
          __syncthreads();
      
          // Now that shared memory contains all the D tiles, stream them to global
          // memory.
          float *dst_gmem_warp_stream_ptr = &D[gmem_idx];
      
      #pragma unroll
          for (int i = 0; i < K; i++) {
            for(int k = 0; k < 4; k++){
              *(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i + laneId * 4 + k) =
                scale_s1s2*(float)*((int *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i + laneId * 4 + k));
            }
          }
          __syncthreads();
        }
      }
}
extern void QuantizeDot_float_float_float_float_float_float_float_float_cuda_QuantizeDot_55_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* __restrict__ input0,float* __restrict__ input1,float* __restrict__ input2,float* __restrict__ input3,float* __restrict__ input4,float* __restrict__ input5,float* __restrict__ input6, float* __restrict__ output0) {
    QuantizeDot_float_float_float_float_float_float_float_float_cuda_QuantizeDot_55<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, output0);
}
// Node name:	Add_30
// Description:	Add
// Input:
//	- name: Dot_26_0	type: float	shape: Shape{1024, 10}
//	- name: Broadcast_29_0	type: float	shape: Shape{1024, 10}
// Output:
//	- name: tensor_30	type: float	shape: Shape{1024, 10}
extern "C" __launch_bounds__(512) __global__ void Add_float_float_float_cuda_Add_30(float* input0, float* input1, float* output0)
{
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
extern void Add_float_float_float_cuda_Add_30_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    Add_float_float_float_cuda_Add_30<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Multiply_28
// Description:	Multiply
// Input:
//	- name: Constant_6_0	type: float	shape: Shape{10}
//	- name: Constant_27_0	type: float	shape: Shape{10}
// Output:
//	- name: Multiply_28_0	type: float	shape: Shape{10}
extern "C" __launch_bounds__(10) __global__ void Multiply_float_float_float_cuda_Multiply_28(float* input0, float* input1, float* output0)
{
    output0[threadIdx.x] = mul(input0[threadIdx.x], input1[threadIdx.x]);

}
extern void Multiply_float_float_float_cuda_Multiply_28_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    Multiply_float_float_float_cuda_Multiply_28<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}

extern "C" void cuda_init()
{
CUDA_SAFE_CALL(cudaDeviceReset());
// total memory:50352768
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaMalloc((void**)&group_0_CUDA_GPU0_allocator_memory_pool,8388608));
CUDA_SAFE_CALL(cudaMemset((void*)group_0_CUDA_GPU0_allocator_memory_pool, 0, 8388608));
Reshape_10_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
BitConverter_34_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4194304);
Add_15_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Add_20_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+4194304);
Add_25_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Dot_26_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2097152);
Multiply_28_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Broadcast_29_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+64);
tensor_30 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2097152);
Result_31_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2097152);
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaMalloc((void**)&group_persist_CUDA_GPU0_allocator_memory_pool,41964160));
CUDA_SAFE_CALL(cudaMemset((void*)group_persist_CUDA_GPU0_allocator_memory_pool, 0, 41964160));
Constant_7_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+0);
Constant_40_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+20480);
Constant_39_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4214784);
Constant_38_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4214848);
Constant_37_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+4214912);
Constant_36_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8409216);
Constant_35_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+12603520);
Constant_33_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+16797824);
Constant_32_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+16797888);
Constant_47_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+16797952);
Constant_46_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+20992256);
Constant_45_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+20992320);
Constant_44_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+20992384);
Constant_43_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+25186688);
Constant_42_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+29380992);
Constant_54_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+33575296);
Constant_53_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+35672448);
Constant_52_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+35672512);
Constant_51_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+35672576);
Constant_50_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+37769728);
Constant_49_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+39866880);
Constant_6_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+41964032);
Constant_27_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+41964096);
// create streams/handles
CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle_0));
 // name=fc4.weight
Constant_float_cuda_Constant_7(0, Constant_7_0);
 // name=Constant_40
Constant_float_cuda_Constant_40(0, Constant_40_0);
 // name=Constant_39
Constant_float_cuda_Constant_39(0, Constant_39_0);
 // name=Constant_38
Constant_float_cuda_Constant_38(0, Constant_38_0);
 // name=Constant_37
Constant_float_cuda_Constant_37(0, Constant_37_0);
 // name=Constant_36
Constant_float_cuda_Constant_36(0, Constant_36_0);
 // name=Constant_35
Constant_float_cuda_Constant_35(0, Constant_35_0);
 // name=Constant_33
Constant_float_cuda_Constant_33(0, Constant_33_0);
 // name=Constant_32
Constant_float_cuda_Constant_32(0, Constant_32_0);
 // name=Constant_47
Constant_float_cuda_Constant_47(0, Constant_47_0);
 // name=Constant_46
Constant_float_cuda_Constant_46(0, Constant_46_0);
 // name=Constant_45
Constant_float_cuda_Constant_45(0, Constant_45_0);
 // name=Constant_44
Constant_float_cuda_Constant_44(0, Constant_44_0);
 // name=Constant_43
Constant_float_cuda_Constant_43(0, Constant_43_0);
 // name=Constant_42
Constant_float_cuda_Constant_42(0, Constant_42_0);
 // name=Constant_54
Constant_float_cuda_Constant_54(0, Constant_54_0);
 // name=Constant_53
Constant_float_cuda_Constant_53(0, Constant_53_0);
 // name=Constant_52
Constant_float_cuda_Constant_52(0, Constant_52_0);
 // name=Constant_51
Constant_float_cuda_Constant_51(0, Constant_51_0);
 // name=Constant_50
Constant_float_cuda_Constant_50(0, Constant_50_0);
 // name=Constant_49
Constant_float_cuda_Constant_49(0, Constant_49_0);
 // name=fc4.bias
Constant_float_cuda_Constant_6(0, Constant_6_0);
 // name=Constant_27
Constant_float_cuda_Constant_27(0, Constant_27_0);
}

// Node name:	BitConverter_34
// Description:	BitConverter
// Input:
//	- name: Reshape_10_0	type: float	shape: Shape{1024, 1024}
//	- name: Constant_32_0	type: float	shape: Shape{1}
//	- name: Constant_33_0	type: float	shape: Shape{1}
// Output:
//	- name: BitConverter_34_0	type: float	shape: Shape{1024, 1024}
extern "C" __global__  void BitConverter_float_float_float_float_cuda_BitConverter_34(float* __restrict__ input0,float* __restrict__ input1,float* __restrict__ input2, float* __restrict__ output0)
{
    {

        const int BLOCK_SIZE = 32;
        const int N=1024;

        const float * A = input0;
        uint8_t * B = reinterpret_cast<uint8_t*>(output0);
        float scale = *input1;
        int zp=(int)(*input2);

        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
     
        int startrow = by * BLOCK_SIZE + ty;
        int startcol = bx * BLOCK_SIZE + tx;
        int idx = startrow * N + startcol;
     
        int cast_val = (int(A[idx] * scale) >> zp)+zp;
     
        B[idx] = (uint8_t)cast_val;
    }
}
extern void BitConverter_float_float_float_float_cuda_BitConverter_34_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* __restrict__ input0,float* __restrict__ input1,float* __restrict__ input2, float* __restrict__ output0) {
    BitConverter_float_float_float_float_cuda_BitConverter_34<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}
// Node name:	Dot_26
// Description:	Dot
// Input:
//	- name: Add_25_0	type: float	shape: Shape{1024, 512}
//	- name: Constant_7_0	type: float	shape: Shape{10, 512}
// Output:
//	- name: Dot_26_0	type: float	shape: Shape{1024, 10}
void Dot_float_float_float_cuda_lib_Dot_26(cublasHandle_t cublas_handle, float* input0, float* input1, float* output0)
{
    const float alpha = 1.0;
    const float beta = 0;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 10, 1024, 512, &alpha, static_cast<const float*>(input1), 512, static_cast<const float*>(input0), 512, &beta, static_cast<float*>(output0), 10));

}
// Node name:	Broadcast_29
// Description:	Broadcast
// Input:
//	- name: Multiply_28_0	type: float	shape: Shape{10}
// Output:
//	- name: Broadcast_29_0	type: float	shape: Shape{1024, 10}
extern "C" __launch_bounds__(64) __global__ void Broadcast_float_float_cuda_Broadcast_29(float* input0, float* output0)
{
    size_t nthreads = 10240;
    uint32_t strides0 = 10;
    uint32_t strides1 = 1;
    int stride_magic0 = 1717986919;
    int stride_magic1 = 1;
    int stride_shift0 = 2;
    int stride_shift1 = 0;
    uint32_t reduced_strides0 = 0;
    uint32_t reduced_strides1 = 1;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nthreads)
    {
        int coordinate_product = tid;
        int coordinate0 = division_by_invariant_multiplication(coordinate_product, stride_magic0, stride_shift0);
        coordinate_product -= (coordinate0 * strides0);
        int coordinate1 = division_by_invariant_multiplication(coordinate_product, stride_magic1, stride_shift1);
        coordinate_product -= (coordinate1 * strides1);
        uint32_t reduced_idx = 0;
        reduced_idx += coordinate0 * reduced_strides0;
        reduced_idx += coordinate1 * reduced_strides1;
        output0[tid] = load(input0, reduced_idx);
    }

}
extern void Broadcast_float_float_cuda_Broadcast_29_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Broadcast_float_float_cuda_Broadcast_29<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_10
// Description:	Reshape
// Input:
//	- name: Parameter_8_0	type: float	shape: Shape{1024, 1, 32, 32}
// Output:
//	- name: Reshape_10_0	type: float	shape: Shape{1024, 1024}
void Reshape_float_float_cuda_lib_Reshape_10(cudaStream_t stream, float* input0, float* output0)
{
    if (input0 != output0) {
       cudaMemcpyAsync(output0, input0, 1048576 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

}
// Node name:	QuantizeDot_41
// Description:	QuantizeDot
// Input:
//	- name: BitConverter_34_0	type: float	shape: Shape{1024, 1024}
//	- name: Constant_35_0	type: float	shape: Shape{1024, 1024}
//	- name: Constant_36_0	type: float	shape: Shape{1024, 1024}
//	- name: Constant_37_0	type: float	shape: Shape{1024, 1024}
//	- name: Constant_38_0	type: float	shape: Shape{1}
//	- name: Constant_39_0	type: float	shape: Shape{1}
//	- name: Constant_40_0	type: float	shape: Shape{1048576}
// Output:
//	- name: Add_15_0	type: float	shape: Shape{1024, 1024}
extern "C" __global__  void QuantizeDot_float_float_float_float_float_float_float_float_cuda_QuantizeDot_41(float* __restrict__ input0,float* __restrict__ input1,float* __restrict__ input2,float* __restrict__ input3,float* __restrict__ input4,float* __restrict__ input5,float* __restrict__ input6, float* __restrict__ output0)
{
    {

        const unsigned int M_GLOBAL=1024;
        const unsigned int N_GLOBAL=1024;
        const unsigned int K_GLOBAL=1024;
        // const parameters
        const unsigned int  WARP_SIZE=32;
        const unsigned int  M=16;
        const unsigned int  N=16;
        const unsigned int  K=16;
        const unsigned int  WMMA_M=16;
        const unsigned int  WMMA_N=16;
        const unsigned int  WMMA_K=16;

        const unsigned int  M_TILES=64;
        const unsigned int  N_TILES=64;
        const unsigned int  K_TILES=64;


        // typedef C_LAYOUT wmma::mem_row_major;

        const unsigned int WARPS_PER_BLOCK=8;
        const unsigned int THREADS_PER_BLOCK= (WARP_SIZE * WARPS_PER_BLOCK);
        const unsigned int CHUNK_K=8;
        const unsigned int CHUNK_LINE_BYTES=(CHUNK_K * K * sizeof(uint8_t));
        const unsigned int WARP_COPY_BYTES=(WARP_SIZE * sizeof(int4));
        const unsigned int CHUNK_COPY_LINES_PER_WARP=(WARP_COPY_BYTES / CHUNK_LINE_BYTES);
        const unsigned int CHUNK_COPY_LINE_LANES=(WARP_SIZE / CHUNK_COPY_LINES_PER_WARP);
        
        const unsigned int BLOCK_ROW_WARPS=2;
        const unsigned int BLOCK_COL_WARPS=4;
        
        const unsigned int WARP_ROW_TILES =4;
        const unsigned int WARP_COL_TILES =2;
        
        const unsigned int BLOCK_ROW_TILES =(WARP_ROW_TILES * BLOCK_ROW_WARPS);
        const unsigned int BLOCK_COL_TILES =(WARP_COL_TILES * BLOCK_COL_WARPS);
        
        const unsigned int GLOBAL_MEM_STRIDE =N_GLOBAL;
        
        const unsigned int SHMEM_STRIDE=(N * BLOCK_ROW_TILES);
        const unsigned int SHMEM_OFFSET=(N * WARP_ROW_TILES);
        
        const unsigned int SKEW_UINT8=32;
        

        // Convert the input pointers
        const uint8_t * A = reinterpret_cast<uint8_t*>(input0); // activation
        const uint8_t * B_val =  reinterpret_cast<uint8_t*>(input1); // weight
        const int * B_row = reinterpret_cast< int *>(input2);
        const int * B_col = reinterpret_cast< int *>(input3);
        const int alpha = (int)(*input4);
        const int integer = (int)(*input5);
        const int * C = reinterpret_cast< int *>(input6);
        uint8_t * D = reinterpret_cast<uint8_t*>(output0);


        extern __shared__ uint8_t shmem[][CHUNK_K * K + SKEW_UINT8];

        // Warp and lane identification.
        const unsigned int warpId = threadIdx.x / WARP_SIZE;
        const unsigned int laneId = threadIdx.x % WARP_SIZE;
      
        // Offset in shared memory from which the B matrix is stored.
        const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;       // BLOCK_COL_TILES * M is shared_A row numbers in one block
      
        // This pointer is used to access the C and D matrix tiles this warp computes.
        int *shmem_warp_tile_ptr = (int *)&shmem[0][0] +
                                   (warpId / 2) * SHMEM_STRIDE * K * 2 +    // K * 2 is because one warp calculate k * 2 rows.
                                   (warpId % 2) * SHMEM_OFFSET;
      
        // This pointer is used to stream the C and D matrices block-wide tile to and
        // from shared memory.
        int *shmem_warp_stream_ptr = (int *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;   // confuse, may be used to read from global memory?
      
        // Each CTA slides along the 128 x 128 tiles from the top left corner of the
        // matrix to the right and down, and selects the next tile to compute. Once
        // there's no such tile, all warps in this CTA exit.
        for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
          const unsigned int block_tile_i =                                   // get the i (row) index of all tiles
              ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
          const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;
      
          // Stop when there are no more D matrix tiles to compute in this CTA.
          if (block_tile_i >= M_TILES) {
            break;
          }
      
          // This warp's pointer to the C matrix data to copy memory from to shared
          // memory.
          const size_t gmem_idx =
              (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
          const int *src_gmem_warp_stream_ptr = &C[gmem_idx];
      
          // Stream multiple C tiles to shared memory.
      #pragma unroll
          for (int i = 0; i < K; i++) {
            typedef int4 copy_t;
      
            *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
                *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
                  laneId);
          }
      
          __syncthreads();
      
          // These fragments will accumulate the result of A and B matrix fragment
          // multiplications along the K_GLOBAL dimension.
          wmma::fragment<wmma::accumulator, M, N, K, int> c[WARP_COL_TILES]
                                                           [WARP_ROW_TILES];
      
          // Load the C matrix tiles into fragments from shared memory.
      #pragma unroll
          for (int i = 0; i < WARP_COL_TILES; i++) {
      #pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
              const int *tile_ptr =
                  shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;
      
              wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, wmma::mem_row_major);
            }
          }
      
          __syncthreads();
      
          // Select what warp copies what matrix to shared memory.
          // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
          int start_tile = B_row[block_tile_j / 2 + (warpId % 4)];
          int end_tile = B_row[block_tile_j / 2 + (warpId % 4) + 1];
          // Go through the global K dimension by a fixed step at a time.
      #pragma unroll
          for(int tile_k_idx = start_tile; tile_k_idx < end_tile; tile_k_idx++){
            size_t shmem_idx = 
              warpId < (WARPS_PER_BLOCK / 2)
                ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
                : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);
      
            int4 *lane_ptr = NULL;
            if(warpId < 4){
              const uint8_t *warp_ptr = &A[block_tile_i * M * K_GLOBAL] +
                M * K_GLOBAL * (warpId % 4) * 2;
              lane_ptr = (int4 *)(warp_ptr + B_col[tile_k_idx] * K * CHUNK_K +
                                  (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
                        (laneId % CHUNK_COPY_LINE_LANES);
            }else{
              const uint8_t *warp_ptr = B_val + tile_k_idx * (M * WARP_COL_TILES) * (K * CHUNK_K);
              lane_ptr = (int4 *)(warp_ptr + (laneId / CHUNK_COPY_LINE_LANES) * (K * CHUNK_K)) + (laneId % CHUNK_COPY_LINE_LANES);
            }
            // Shift the second half of the warp to the next row / column in the
            // shared memory.
            shmem_idx += laneId / CHUNK_COPY_LINE_LANES;
      
            #pragma unroll
            for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
                 i++) {
              // Copy 16 bytes at once in each lane.
              *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
                  *lane_ptr;
      
              // Advance the global memory pointer and the shared memory index.
              lane_ptr = warpId < 4 ?
                          (int4 *)((uint8_t *)lane_ptr +
                                              K_GLOBAL * CHUNK_COPY_LINES_PER_WARP):
                          (int4 *)((uint8_t *)lane_ptr +
                                              K * CHUNK_K * CHUNK_COPY_LINES_PER_WARP);
              shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }
      
            __syncthreads();
      
            for (int k_step = 0; k_step < CHUNK_K; k_step++) {
              wmma::fragment<wmma::matrix_a, M, N, K, uint8_t, wmma::row_major>
                  a[WARP_COL_TILES];
              wmma::fragment<wmma::matrix_b, M, N, K, uint8_t, wmma::col_major>
                  b[WARP_ROW_TILES];
      
      #pragma unroll
              for (int i = 0; i < WARP_COL_TILES; i++) {
                size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
                const uint8_t *tile_ptr = &shmem[shmem_idx_a][k_step * K];
      
                wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_UINT8);
      
      #pragma unroll
                for (int j = 0; j < WARP_ROW_TILES; j++) {
                  if (i == 0) {
                    // Load the B matrix fragment once, because it is going to be
                    // reused against the other A matrix fragments.
                    size_t shmem_idx_b = shmem_idx_b_off +
                                         (WARP_ROW_TILES * N) * (warpId % 2) +
                                         (j * N);
                    const uint8_t *tile_ptr = &shmem[shmem_idx_b][k_step * K];
      
                    wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_UINT8);
                  }
      
                  wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                }
              }
            }
      
            __syncthreads();
          }
      
            // Store the D fragments to shared memory.
      #pragma unroll
          for (int i = 0; i < WARP_COL_TILES; i++) {
      #pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
      #pragma unroll
              // Uniform, point-wise transformations of ALL fragment elements by ALL
              // threads in the warp are well-defined even though element indices
              // within fragment storage are not defined.
              for (int t = 0; t < c[i][j].num_elements; t++) {
                c[i][j].x[t] = ((c[i][j].x[t] * alpha) >> integer);
              }
      
              int *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;
      
              wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, wmma::mem_row_major);
            }
          }
      
          __syncthreads();
      
          // Now that shared memory contains all the D tiles, stream them to global
          // memory.
          uint8_t *dst_gmem_warp_stream_ptr = &D[gmem_idx];
      
      #pragma unroll
          for (int i = 0; i < K; i++) {
            for(int k = 0; k < 4; k++){
              *(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i + laneId * 4 + k) =
                (uint8_t)*((int *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i + laneId * 4 + k));
            }
          }
          __syncthreads();
        }



    }
}
extern void QuantizeDot_float_float_float_float_float_float_float_float_cuda_QuantizeDot_41_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* __restrict__ input0,float* __restrict__ input1,float* __restrict__ input2,float* __restrict__ input3,float* __restrict__ input4,float* __restrict__ input5,float* __restrict__ input6, float* __restrict__ output0) {
    QuantizeDot_float_float_float_float_float_float_float_float_cuda_QuantizeDot_41<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, output0);
}

#ifndef __NNFUSION_GRAPH_CONFIG__
#define __NNFUSION_GRAPH_CONFIG__
#define NNFUSION_GRAPH_INPUT_NUM 1
#define NNFUSION_GRAPH_OUTPUT_NUM 1
#define NNFUSION_GRAPH_INPUT_DTYPE_0 float
#define NNFUSION_GRAPH_INPUT_SHAPE_0 {1024, 1, 32, 32}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_0 float
#define NNFUSION_GRAPH_OUTPUT_SHAPE_0 {1024, 10}
#endif


extern "C" int kernel_entry(float* Parameter_8_0, float** Result_31_0)
{
// kernel_entry_init
 // name=10
cudaFuncSetAttribute(
   QuantizeDot_float_float_float_float_float_float_float_float_cuda_QuantizeDot_41 , cudaFuncAttributeMaxDynamicSharedMemorySize,65536);
cudaFuncSetAttribute(
   QuantizeDot_float_float_float_float_float_float_float_float_cuda_QuantizeDot_55, cudaFuncAttributeMaxDynamicSharedMemorySize,65536);
cudaFuncSetAttribute(
   BitConverter_float_float_float_float_cuda_BitConverter_34 , cudaFuncAttributeMaxDynamicSharedMemorySize,65536);

Reshape_float_float_cuda_lib_Reshape_10(0, Parameter_8_0, Reshape_10_0);
 // name=BitConverter_34
BitConverter_float_float_float_float_cuda_BitConverter_34_Call(dim3(32, 32, 1), dim3(32, 32, 1), 65536, 0, Reshape_10_0, Constant_32_0, Constant_33_0, BitConverter_34_0);
 // name=QuantizeDot_41
QuantizeDot_float_float_float_float_float_float_float_float_cuda_QuantizeDot_41_Call(dim3(68, 1, 1), dim3(256, 1, 1), 65536, 0, BitConverter_34_0, Constant_35_0, Constant_36_0, Constant_37_0, Constant_38_0, Constant_39_0, Constant_40_0, Add_15_0);
 // name=QuantizeDot_48
QuantizeDot_float_float_float_float_float_float_float_float_cuda_QuantizeDot_41_Call(dim3(68, 1, 1), dim3(256, 1, 1), 65536, 0, Add_15_0, Constant_42_0, Constant_43_0, Constant_44_0, Constant_45_0, Constant_46_0, Constant_47_0, Add_20_0);
 // name=QuantizeDot_55
QuantizeDot_float_float_float_float_float_float_float_float_cuda_QuantizeDot_55_Call(dim3(68, 1, 1), dim3(256, 1, 1), 65536, 0, Add_20_0, Constant_49_0, Constant_50_0, Constant_51_0, Constant_52_0, Constant_53_0, Constant_54_0, Add_25_0);
 // name=Dot_26
Dot_float_float_float_cuda_lib_Dot_26(cublas_handle_0, Add_25_0, Constant_7_0, Dot_26_0);
 // name=Multiply_28
Multiply_float_float_float_cuda_Multiply_28_Call(dim3(1, 1, 1), dim3(10, 1, 1), 0, 0, Constant_6_0, Constant_27_0, Multiply_28_0);
 // name=Broadcast_29
Broadcast_float_float_cuda_Broadcast_29_Call(dim3(160, 1, 1), dim3(64, 1, 1), 0, 0, Multiply_28_0, Broadcast_29_0);
 // name=Add_30
Add_float_float_float_cuda_Add_30_Call(dim3(20, 1, 1), dim3(512, 1, 1), 0, 0, Dot_26_0, Broadcast_29_0, tensor_30);
 // name=Result_31
Result_float_float_cuda_lib_Result_31(tensor_30, Result_31_0);
return 0;
}


extern "C" void cuda_free()
{
CUBLAS_SAFE_CALL(cublasDestroy(cublas_handle_0));
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaFree(group_0_CUDA_GPU0_allocator_memory_pool));
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaFree(group_persist_CUDA_GPU0_allocator_memory_pool));
}

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

