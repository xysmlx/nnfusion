extern "C" __global__ void FP32toInt8Shift(float* input0, float* input1, float* input2, float* output0){

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