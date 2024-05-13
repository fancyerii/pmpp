#include <cuda.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cstdlib>

#define N 2048
#define N2 204800
#define BLOCK_DIM 1024
#define COARSE_FACTOR 4

using namespace std;


void generate_rand_int(float * array, int len, int seed){
    srand(seed);
    for(int i = 0; i < len; ++i){
        array[i] = rand() % 100;
    }
}

void cpu_sum_reduction(float * input, int len, float * output){
    float sum = 0;
    for(int i = 0; i < len; ++i){
        sum += input[i];
    }
    *output = sum;
}

__global__
void SimpleSumReductionKernel(float* input, float* output){
    unsigned int i = 2 * threadIdx.x;
    for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2){
        if(threadIdx.x % stride == 0){
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        *output = input[0];
    }
}

__global__
void ConvergentSumReductionKernel(float* input, float* output){
    unsigned int i = threadIdx.x;
    for(unsigned int stride = blockDim.x; stride >= 1; stride /= 2){
        if(threadIdx.x < stride){
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        *output = input[0];
    }
}

__global__
void ShareMemorySumReductionKernel(float* input, float* output){
    __shared__ float input_s[BLOCK_DIM];
    unsigned int i = threadIdx.x;
    input_s[i] = input[i] + input[i + BLOCK_DIM];
    for(unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2){
        __syncthreads();
        if(threadIdx.x < stride){
            input_s[i] += input_s[i + stride];
        }
        
    }
    if(threadIdx.x == 0){
        *output = input_s[0];
    }
}

__global__
void SegmentedSumReductionKernel(float* input, float* output){
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    input_s[t] = input[i] + input[i + BLOCK_DIM];
    for(unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2){
        __syncthreads();
        if(t < stride){
            input_s[t] += input_s[t + stride];
        }
        
    }
    if(t == 0){
        atomicAdd(output, input_s[0]);
    }
}

__global__
void CoarsenedSumReductionKernel(float* input, float* output){
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = 2 * COARSE_FACTOR * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    float sum = input[i];
    for(unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile){
        sum += input[i + tile * BLOCK_DIM];
    }
    input_s[t] = sum;
    for(unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2){
        __syncthreads();
        if(t < stride){
            input_s[t] += input_s[t + stride];
        }
        
    }
    if(t == 0){
        atomicAdd(output, input_s[0]);
    }
}

void check(float * array, int len, float result, string kernelName,
        dim3 blockDim, dim3 gridDim,
        void kernel(float* input, float* output)){
        float *deviceInput;
        float *deviceResult; 

        cudaMalloc((void **) &deviceInput, len * sizeof(float));
        cudaMalloc((void **) &deviceResult,  sizeof(float));

        cudaMemcpy(deviceInput, array, len * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(deviceResult, 0, sizeof(float));   

        kernel<<<gridDim, blockDim>>>(deviceInput, deviceResult);
        float hostResult;
        cudaMemcpy(&hostResult, deviceResult, sizeof(float), cudaMemcpyDeviceToHost);

        cout << "check " << kernelName << ": " << endl;
        if(result != hostResult){
            cout << "\tbad! result=" << int(result) << ", hostResult=" << int(hostResult) << endl;
        }else{
            cout << "\tok!" << endl;
        }

        cudaFree(deviceInput);
        cudaFree(deviceResult);  
}

int main(int argc, char* argv[]){
    float array[N];
    generate_rand_int(array, N, 1234);
    float result = 0;
    cpu_sum_reduction(array, N, &result);
    cout << "cpu result: " << result << endl;

    dim3 blockDim(1024);
    dim3 gridDim(1);
    check(array, N, result, "SimpleSumReductionKernel", blockDim, gridDim, SimpleSumReductionKernel);
    check(array, N, result, "ConvergentSumReductionKernel", blockDim, gridDim, ConvergentSumReductionKernel);
    check(array, N, result, "ShareMemorySumReductionKernel", blockDim, gridDim, ShareMemorySumReductionKernel);
    check(array, N, result, "SegmentedSumReductionKernel", blockDim, gridDim, SegmentedSumReductionKernel);
    
    float array2[N2];
    generate_rand_int(array2, N2, 1234);
    float result2 = 0;
    cpu_sum_reduction(array2, N2, &result2);
    cout << "cpu result2: " << int(result2) << endl;

    dim3 blockDim2(1024);
    dim3 gridDim2(ceil((float)N2 / (blockDim2.x * 2)));
    check(array2, N2, result2, "SegmentedSumReductionKernel N2", blockDim2, gridDim2, SegmentedSumReductionKernel);

    dim3 blockDim3(1024);
    dim3 gridDim3(ceil((float)N2 / (blockDim3.x * 2 * COARSE_FACTOR)));
    check(array2, N2, result2, "CoarsenedSumReductionKernel N2", blockDim3, gridDim3, CoarsenedSumReductionKernel);
    
    return 0;

}