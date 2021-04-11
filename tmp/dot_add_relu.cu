// This kernel's input datatype and output datatype are all 8bit, using cuda core.
extern "C" __global__ void MatrixMulCUDA_8bit_biasRelu(float *input0, float *input1, float *input2, float *input3, float *input4, float *input5, float * input6,float  *input7, float *input8, float *output0){
    /*Prepare Stage*/
    // set the block size and matrix size constants here
    const int BLOCK_SIZE=32;
    const int k=256, n=256;
    // convert the input pointer to the correct type
    uint8_t * A =  reinterpret_cast<uint8_t*>(input0);
    uint8_t * W =  reinterpret_cast<uint8_t*>(input1);
    uint8_t * C =  reinterpret_cast<uint8_t*>(output0);
    unsigned int * Multi_w_zp =  reinterpret_cast<unsigned int *>(input2);
    uint8_t * W_zp =  reinterpret_cast<uint8_t *>(input3);
    unsigned int * ZP_accu =  reinterpret_cast<unsigned int *>(input4);
    const int integer = (int)(*input5);
    const int shift_val = (int)(*input6);
    unsigned int * bias = reinterpret_cast<unsigned int *>(input7);
    uint8_t output_zp = (uint8_t)(*input8);

    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
 
    int aBegin = BLOCK_SIZE * by * k;
    int aEnd = aBegin + k - 1;
 
    int bBegin = BLOCK_SIZE * bx * k;
 
    __shared__ uint8_t As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ uint8_t Bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ uint8_t Wzp[BLOCK_SIZE][BLOCK_SIZE];
 
    unsigned int cSub = 0;
    unsigned int azpSub = 0;
    for(int i = aBegin, j = bBegin; i <= aEnd; i += BLOCK_SIZE, j += BLOCK_SIZE){
        As[ty][tx] = A[i + ty * k + tx];
        Bs[ty][tx] = W[j + ty * k + tx];
        Wzp[ty][tx] = W_zp[j + ty * k + tx];
 
        __syncthreads();
 
        for(int k = 0; k < 32; k += 4){
            unsigned int pack_val1 = FETCH_UINT32(As[ty][k]);
            unsigned int pack_val2 = FETCH_UINT32(Bs[tx][k]);
            unsigned int pack_val3 = FETCH_UINT32(Wzp[tx][k]);
            cSub = __dp4a(pack_val1, pack_val2, cSub);
            azpSub = __dp4a(pack_val1, pack_val3, azpSub);
        }
        __syncthreads();
    }
    
    int cx = by * BLOCK_SIZE + ty;
    int cy = bx * BLOCK_SIZE + tx;
 
    cSub = cSub - Multi_w_zp[cx*n+cy] - azpSub + ZP_accu[cx*n+cy] + bias[cy];
    cSub = (cSub * integer) >> shift_val;
 
    C[cx*n+cy] = (uint8_t)cSub;
    if(C[cx*n+cy] < output_zp) C[cx*n+cy] = 0;
}