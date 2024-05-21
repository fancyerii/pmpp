#include <cuda.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iomanip>

#define LEN 1023
#define SECTION_SIZE 1024
#define COARSE_FACTOR 4

#define LEN2 10241
#define SECTION_SIZE2 1024
#define BLOCK_SIZE2 ((LEN2 + SECTION_SIZE2 - 1)/(SECTION_SIZE2))

using namespace std;


void generate_rand_int(float * array, int len, int seed){
    srand(seed);
    for(int i = 0; i < len; ++i){
        array[i] = rand() % 10;
    }
}

void sequential_scan(float* x, float* y, unsigned int N){
    y[0] = x[0];
    for(unsigned int i = 1; i < N; ++i){
        y[i] = y[i - 1] + x[i];
    }

}

__global__
void Kogge_Stone_scan(float* x, float* y, unsigned int N){
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){
        XY[threadIdx.x] = x[i];
    }else{
        XY[threadIdx.x] = 0;
    }

    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads();
        float temp;
        if(threadIdx.x >= stride){
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        }
        __syncthreads();
        if(threadIdx.x >= stride){
            XY[threadIdx.x] = temp;
        }
    }

    if(i < N){
        y[i] = XY[threadIdx.x];
    }
}

__global__
void Kogge_Stone_scan_double_buffers(float* x, float* y, unsigned int N){
    __shared__ float XY1[SECTION_SIZE];
    __shared__ float XY2[SECTION_SIZE];
    
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){
        XY1[threadIdx.x] = x[i];
    }else{
        XY1[threadIdx.x] = 0;
    }

    int in_idx = 0;
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads();
        if(threadIdx.x >= stride){
            if(in_idx == 0){
                XY2[threadIdx.x] = XY1[threadIdx.x] +
                                   XY1[threadIdx.x - stride];
                in_idx = 1;
            }else{
                XY1[threadIdx.x] = XY2[threadIdx.x] +
                                   XY2[threadIdx.x - stride];                
                in_idx = 0;
            }
        }else if(threadIdx.x >= stride / 2){
            if(in_idx == 0){
                XY2[threadIdx.x] = XY1[threadIdx.x];
                in_idx = 1;
            }else{
                XY1[threadIdx.x] = XY2[threadIdx.x];
                in_idx = 0;
            }
        }

    }

    if(i < N){
        if(in_idx == 0){
            y[i] = XY1[threadIdx.x];
        }else{
            y[i] = XY2[threadIdx.x];
        }
    }
}

__global__
void Kogge_Stone_scan_double_buffers_v2(float* x, float* y, unsigned int N){
    __shared__ float XY1[SECTION_SIZE];
    __shared__ float XY2[SECTION_SIZE];
    
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){
        XY1[threadIdx.x] = x[i];
    }else{
        XY1[threadIdx.x] = 0;
    }

    float * in_buffer = XY1;
    float * out_buffer = XY2;
    float * temp;
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads();
        if(threadIdx.x >= stride){
            out_buffer[threadIdx.x] = in_buffer[threadIdx.x] +
                                      in_buffer[threadIdx.x - stride];
            temp = in_buffer;
            in_buffer = out_buffer;
            out_buffer = temp;
        }else if(threadIdx.x >= stride / 2){
            out_buffer[threadIdx.x] = in_buffer[threadIdx.x];
        }

    }

    if(i < N){
        y[i] = in_buffer[threadIdx.x];
    }
}


__global__
void Brent_Kung_scan(float* x, float* y, unsigned int N){
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        XY[threadIdx.x] = x[i];
    }
    if(i + blockDim.x < N){
        XY[threadIdx.x + blockDim.x] = x[i + blockDim.x];
    }

    for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2){
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if(index < SECTION_SIZE){
            XY[index] += XY[index - stride];
        }
    }

    for(unsigned int stride = SECTION_SIZE / 4; stride > 0; stride /= 2){
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if(index + stride < SECTION_SIZE){
            XY[index + stride] += XY[index];
        }
    }
    __syncthreads();

    if(i < N){
        y[i] = XY[threadIdx.x];
    }
    if(i + blockDim.x < N){
        y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
    }
}

__global__
void coarsen_Kogge_Stone_scan(float* x, float* y, unsigned int N){
    __shared__ float XY[SECTION_SIZE];
    
    // phase 1
    for(unsigned int start_idx = 0; start_idx < SECTION_SIZE; start_idx+=(SECTION_SIZE/COARSE_FACTOR)){
        if(threadIdx.x + start_idx < N){
            XY[threadIdx.x + start_idx] = x[threadIdx.x + start_idx];
        }else{
            XY[threadIdx.x + start_idx] = 0;
        }
    }
    __syncthreads();
    int v = 0;

    unsigned int i = threadIdx.x * COARSE_FACTOR;
    float prev = XY[i];
    for(unsigned int index = 1 + i; index < COARSE_FACTOR + i; ++index){
        XY[index] += prev;
        prev = XY[index];
    }

    // phase 2
    // scan XY[COARSE_FACTOR-1], XY[2*COARSE_FACTOR-1], ...
    unsigned int idx = (threadIdx.x + 1) * COARSE_FACTOR - 1;
    for(unsigned stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads();
        float temp;
        if(threadIdx.x >= stride){          
            temp = XY[idx] + XY[idx - stride * COARSE_FACTOR];
        }
        __syncthreads();
        if(threadIdx.x >= stride){
            XY[idx] = temp;
        }
    }

    // phase 3
    if(threadIdx.x < blockDim.x - 1){
        for(unsigned int i = idx + 1; i < idx + COARSE_FACTOR; ++i){
            XY[i] += XY[idx];
        }
    }

    __syncthreads();
    //write back
    for(unsigned int start_idx = 0; start_idx < SECTION_SIZE; start_idx+=(SECTION_SIZE/COARSE_FACTOR)){
        if(threadIdx.x + start_idx < N){
            y[threadIdx.x + start_idx] = XY[threadIdx.x + start_idx] + v;
        }
    }
}

__global__
void hierarchical_Kogge_Stone_scan3(float* y, unsigned int N, float * S){
    //TODO: can be optimized to use one less block and avoid check blockIdx.x > 0
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(blockIdx.x > 0 && i < N){
        y[i] += S[blockIdx.x - 1];
    }
}

__global__
void hierarchical_Kogge_Stone_scan1(float* x, float* y, unsigned int N, float * S){
    __shared__ float XY[SECTION_SIZE2];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){
        XY[threadIdx.x] = x[i];
    }else{
        XY[threadIdx.x] = 0;
    }

    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads();
        float temp;
        if(threadIdx.x >= stride){
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        }
        __syncthreads();
        if(threadIdx.x >= stride){
            XY[threadIdx.x] = temp;
        }
    }

    if(i < N){
        y[i] = XY[threadIdx.x];
    }

    if(threadIdx.x == blockDim.x - 1){
        S[blockIdx.x] = XY[SECTION_SIZE2 - 1];
    }    
}
 

void check(float * input, int len, float* result, string kernelName,
        dim3 blockDim, dim3 gridDim,
        void kernel(float* x, float* y, unsigned int N)){
        float *deviceInput;
        float *deviceResult; 

        float *hostResult = (float *)malloc(len * sizeof(float));

        cudaMalloc((void **) &deviceInput, len * sizeof(float));
        cudaMalloc((void **) &deviceResult,  len * sizeof(float));

        cudaMemcpy(deviceInput, input, len * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(deviceResult, 0, len * sizeof(float));   

        kernel<<<gridDim, blockDim>>>(deviceInput, deviceResult, len);
        
        cudaMemcpy(hostResult, deviceResult, len * sizeof(float), cudaMemcpyDeviceToHost);

        cout << "check " << kernelName << ": " << endl;
        for(unsigned int i = 0; i < len; ++i){
            if(result[i] != hostResult[i]){
                cout << "\tbad! result[" << i << "]=" << int(result[i]) << ", hostResult=" << int(hostResult[i]) << endl;
                goto bad;
            }
        }
        cout << "\tok!" << endl;
        bad:

        free(hostResult);
        cudaFree(deviceInput);
        cudaFree(deviceResult);  
}

__global__
void single_pass_Kogge_Stone_scan(float* x, float* y, unsigned int N, 
                int *blockCounter, int *flag, float *scan_value){
    // get dynamic block index
    __shared__ unsigned int bid_s;

    if(threadIdx.x == 0){
        bid_s = atomicAdd(blockCounter, 1);
    }
    __syncthreads();
    unsigned int bid = bid_s;

    // step1 scan within each block

    __shared__ float XY[SECTION_SIZE2];
    unsigned int i = bid * blockDim.x + threadIdx.x;

    if(i < N){
        XY[threadIdx.x] = x[i];
    }else{
        XY[threadIdx.x] = 0;
    }

    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads();
        float temp;
        if(threadIdx.x >= stride){
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        }
        __syncthreads();
        if(threadIdx.x >= stride){
            XY[threadIdx.x] = temp;
        }
    }

    if(i < N){
        y[i] = XY[threadIdx.x];
    }

    // step2 get previous result
    __shared__ float previous_sum;
    if(threadIdx.x == 0){
        while(atomicAdd(&flag[bid], 0) == 0){}
        previous_sum = scan_value[bid];
        scan_value[bid + 1] = previous_sum + XY[SECTION_SIZE2 - 1];
        __threadfence();
        atomicAdd(&flag[bid + 1], 1);
    }
    __syncthreads();

    // step3 add previous result 
    if(i < N){
        y[i] += previous_sum;
    }

}

void check_single_pass_Kogge_Stone_scan(float * input, int len, float* result,
                                dim3 blockDim, dim3 gridDim){
        float *deviceInput;
        float *deviceResult; 

        float *hostResult = (float *)malloc(len * sizeof(float));

        cudaMalloc((void **) &deviceInput, len * sizeof(float));
        cudaMalloc((void **) &deviceResult,  len * sizeof(float));
        

        cudaMemcpy(deviceInput, input, len * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(deviceResult, 0, len * sizeof(float));
           
        int *blockCounterDevice;
        int *flagDevice;
        float *scan_valueDevice;

        int size = gridDim.x + 1;

        cudaMalloc((void **) &blockCounterDevice, sizeof(int));
        cudaMalloc((void **) &flagDevice, size * sizeof(int));
        cudaMalloc((void **) &scan_valueDevice, size * sizeof(float));

        // initialize counter to 0
        cudaMemset(blockCounterDevice, 0, sizeof(int));

        // initialize flag[0] = 1
        vector<int> flag(size, 0);
        flag[0] = 1;
        cudaMemcpy(flagDevice, &flag[0], size * sizeof(int), cudaMemcpyHostToDevice);
        
        // initialize scan_valueDevice to 0
        cudaMemset(scan_valueDevice, 0, sizeof(float));

        single_pass_Kogge_Stone_scan<<<gridDim, blockDim>>>(deviceInput, deviceResult, len,
                                                            blockCounterDevice, flagDevice, 
                                                            scan_valueDevice);

        cudaMemcpy(hostResult, deviceResult, len * sizeof(float), cudaMemcpyDeviceToHost);

        cout << "check check_single_pass_Kogge_Stone_scan:" << endl;
        for(int i = 0; i < len; i++){
            if(hostResult[i] != result[i]){
                cout << "\tbad hostResult" << "[" << i << "]=" << hostResult[i]
                     << " result=" << result[i] << endl;
                goto bad;
            }
        }
        cout << "\tok!" << endl;
        bad:
        free(hostResult);
        cudaFree(deviceInput);
        cudaFree(deviceResult); 
        cudaFree(blockCounterDevice);
        cudaFree(flagDevice);
        cudaFree(scan_valueDevice); 
}

void check_hierarchical_Kogge_Stone_scan(float * input, int len, float* result,
        dim3 blockDim, dim3 gridDim,
        dim3 blockDim2, dim3 gridDim2){
        float *deviceInput;
        float *deviceResult; 
        float *deviceS;

        float *hostResult = (float *)malloc(len * sizeof(float));
        float *hostS = (float *)malloc(BLOCK_SIZE2 * sizeof(float));

        cudaMalloc((void **) &deviceInput, len * sizeof(float));
        cudaMalloc((void **) &deviceResult,  len * sizeof(float));
        cudaMalloc((void **) &deviceS,  BLOCK_SIZE2 * sizeof(float));
        

        cudaMemcpy(deviceInput, input, len * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(deviceResult, 0, len * sizeof(float));
        cudaMemset(deviceS, 0, BLOCK_SIZE2 * sizeof(float));
           

        hierarchical_Kogge_Stone_scan1<<<gridDim, blockDim>>>(deviceInput, deviceResult, len, deviceS);
        

        Kogge_Stone_scan<<<gridDim2, blockDim2>>>(deviceS, deviceS, BLOCK_SIZE2);


        hierarchical_Kogge_Stone_scan3<<<gridDim, blockDim>>>(deviceResult, len, deviceS);

        cudaMemcpy(hostResult, deviceResult, len * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostS, deviceS, BLOCK_SIZE2 * sizeof(float), cudaMemcpyDeviceToHost);

        cout << "check hierarchical_Kogge_Stone_scan:" << endl;
        for(int i = 0; i < len; i++){
            if(hostResult[i] != result[i]){
                cout << "\tbad " << hostResult << "[" << i << "]=" << hostResult[i]
                     << " result=" << result[i] << endl;
                goto bad;
            }
        }
        cout << "\tok!" << endl;
        bad:
        free(hostResult);
        free(hostS);
        cudaFree(deviceInput);
        cudaFree(deviceResult);  
        cudaFree(deviceS);
}


int main(int argc, char* argv[]){
    float x[LEN];
    generate_rand_int(x, LEN, 1234);
    float y[LEN] = {};
    sequential_scan(x, y, LEN);
    for(int i = 0; i < 16; i++){
        cout << int(y[i]) << " ";
    }
    cout << endl;



    dim3 blockDim(SECTION_SIZE);
    dim3 gridDim(1);
    check(x, LEN, y, "Kogge_Stone_scan", blockDim, gridDim, Kogge_Stone_scan);
    check(x, LEN, y, "Kogge_Stone_scan_double_buffers", blockDim, gridDim, Kogge_Stone_scan_double_buffers);
    check(x, LEN, y, "Kogge_Stone_scan_double_buffers_v2", blockDim, gridDim, Kogge_Stone_scan_double_buffers_v2);
    
    dim3 blockDim2(SECTION_SIZE/2);
    dim3 gridDim2(1);
    check(x, LEN, y, "Brent_Kung_scan", blockDim2, gridDim2, Brent_Kung_scan);
    
    dim3 blockDim3(SECTION_SIZE/COARSE_FACTOR);
    dim3 gridDim3(1);   
    check(x, LEN, y, "coarsen_Kogge_Stone_scan", blockDim3, gridDim3, coarsen_Kogge_Stone_scan);

    
    float x2[LEN2];
    generate_rand_int(x2, LEN2, 1234);
    float y2[LEN2] = {};
    sequential_scan(x2, y2, LEN2); 

    dim3 blockDim4(SECTION_SIZE2);
    dim3 gridDim4(BLOCK_SIZE2);
    dim3 blockDim5(BLOCK_SIZE2);
    dim3 gridDim5(1);
    check_hierarchical_Kogge_Stone_scan(x2, LEN2, y2, blockDim4, gridDim4, blockDim5, gridDim5);
    check_single_pass_Kogge_Stone_scan(x2, LEN2, y2, blockDim4, gridDim4);

    return 0;

}