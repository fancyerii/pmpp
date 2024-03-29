#include <cuda.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>
#define FILTER_RADIUS 2
#define FILTER_SIZE (2*(FILTER_RADIUS)+1)
#define IN_TILE_DIM ((OUT_TILE_DIM) + 2*(FILTER_RADIUS))
#define OUT_TILE_DIM 32
#define TILE_DIM 32


using json = nlohmann::json;
using namespace std;

__constant__ float F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

__global__ 
void conv_cached_tiled_2d_constant_mem_kernel(float *N, float *P,
                    int width, int height){
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x; 
    
    __shared__ float ds_N[TILE_DIM][TILE_DIM];
    if(row < height && col < width){
        ds_N[threadIdx.y][threadIdx.x] = N[row * width + col];
    }else{
        ds_N[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    if(row < height && col < width){
        float Pvalue = 0.0f;
        for(int fRow = 0; fRow < FILTER_SIZE; fRow++){
            int ds_y = threadIdx.y - FILTER_RADIUS + fRow;
            int y = row - FILTER_RADIUS + fRow;
            for(int fCol = 0; fCol < FILTER_SIZE; fCol++){
                int ds_x = threadIdx.x - FILTER_RADIUS + fCol;
                if(ds_x >= 0 && ds_x < TILE_DIM && ds_y >= 0 && ds_y < TILE_DIM){
                    Pvalue += F[fRow][fCol] * ds_N[ds_y][ds_x];
                }else{
                    int x = col - FILTER_RADIUS + fCol;
                    if(y >= 0 && y < height && x >= 0 && x < width){
                        Pvalue += F[fRow][fCol] * N[y * width + x];
                    }
                }
            }
        }
        P[row * width + col] = Pvalue;

    }

}

__global__ 
void conv_tiled_2d_constant_mem_kernel(float *N, float *P,
                    int width, int height){
    int outRow = blockIdx.y * OUT_TILE_DIM + threadIdx.y;
    int outCol = blockIdx.x * OUT_TILE_DIM + threadIdx.x;   

    __shared__ float ds_N[IN_TILE_DIM][IN_TILE_DIM];
    int loop_count = ceil(IN_TILE_DIM/(float)OUT_TILE_DIM);
    for(int i = 0; i < loop_count; i++){
        int rIdx = OUT_TILE_DIM * i + threadIdx.y;
        int y = rIdx - FILTER_RADIUS + blockIdx.y * OUT_TILE_DIM;
        for(int j = 0; j < loop_count; j++){
            int cIdx = OUT_TILE_DIM * j + threadIdx.x;
            int x = cIdx - FILTER_RADIUS + blockIdx.x * OUT_TILE_DIM;
            if(rIdx < IN_TILE_DIM && cIdx < IN_TILE_DIM){
                if(y >= 0 && y < height && x >= 0 && x < width){
                    ds_N[rIdx][cIdx] = N[y * width + x]; 
                }else{
                    ds_N[rIdx][cIdx] = 0;
                }
            }
        }
    }
    __syncthreads();


    if(outRow < height && outCol < width){
        float Pvalue = 0.0f;
        for(int row = 0; row < FILTER_SIZE; row++){
            for(int col = 0; col < FILTER_SIZE; col++){
                Pvalue += F[row][col] * ds_N[row + threadIdx.y][col + threadIdx.x];
            }
        }

        P[outRow * width + outCol] = Pvalue;
    }


}

__global__ 
void conv_2d_constant_mem_kernel(float *N, float *P,
                    int r, int width, int height){
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x; 


    float Pvalue = 0.0f;
    int inRow, inCol;
    int filter_width = 2 * r + 1;
    for(int row = 0; row < filter_width; row++){
        inRow = outRow - r + row;
        for(int col = 0; col < filter_width; col++){
            inCol = outCol - r + col;
            if(inRow >= 0 && inRow < height && inCol >= 0 && inCol < width){
                Pvalue += F[row][col] * N[inRow * width + inCol];
            }
        }
    }


    P[outRow * width + outCol] = Pvalue;
}

__global__ 
void conv_2d_basic_kernel(float *N, float *F, float *P,
                    int r, int width, int height){
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x; 


    float Pvalue = 0.0f;
    int inRow, inCol;
    int filter_width = 2 * r + 1;
    for(int row = 0; row < filter_width; row++){
        inRow = outRow - r + row;
        for(int col = 0; col < filter_width; col++){
            inCol = outCol - r + col;
            if(inRow >= 0 && inRow < height && inCol >= 0 && inCol < width){
                Pvalue += F[row * filter_width + col] * N[inRow * width + inCol];
            }
        }
    }


    P[outRow * width + outCol] = Pvalue;
}
                    

int main(int argc, char* argv[])
{
    // Query GPU properties
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);
    cout << "---------------------------------------------" << endl;
    cout << "               GPU PROPERTIES                " << endl;
    cout << "---------------------------------------------" << endl;
    cout << "Device Name: " << dev_prop.name << endl;
    cout << "Memory Clock Rate: " << dev_prop.memoryClockRate/1.0e6 <<  " GHz" << endl;
    cout << "Memory Bandwidth: " << 2.0*dev_prop.memoryClockRate*(dev_prop.memoryBusWidth/8)/1.0e6 <<  " GB/s" << endl;
    cout << "Number of SM: " << dev_prop.multiProcessorCount << endl;
    cout << "Max Threads per SM: " << dev_prop.maxThreadsPerMultiProcessor << endl;
    cout << "Registers per Block: " << dev_prop.regsPerBlock << endl;
    cout << "Shared Memory per Block: " << dev_prop.sharedMemPerBlock << " B" << endl;
    cout << "Total Global Memory per Block: " << dev_prop.totalGlobalMem/1.0e9 << " GB" << endl;
    cout << endl;


    std::ifstream f("conv.json");
    json data = json::parse(f);
    auto input = data["input"].template get<std::vector<std::vector<float>>>(); 
    auto output = data["output"].template get<std::vector<std::vector<float>>>(); 
    auto filter = data["filter"].template get<std::vector<std::vector<float>>>(); 
    
    int height = input.size();
    int width = input[0].size();
    int filter_size = filter[0].size();
    int r = filter_size / 2;

    float *hostImg, *hostFilter, *hostResult;
    float *deviceImg, *deviceFilter, *deviceResult; 


    // Allocate memory on host
    hostImg = (float *) malloc(height * width * sizeof(float)); 
    hostResult = (float *) malloc(height * width * sizeof(float));
    hostFilter = (float *) malloc(filter_size * filter_size * sizeof(float));

    // copy to hostImage and hostFilter
    int idx = 0;
    for(auto row : input){
        for(auto col : row){
            hostImg[idx++] = col;
        }
    }

    idx = 0; 
    for(auto row: filter){
        for(auto col: row){
            hostFilter[idx++] = col;
        }
    }

    float filter_constant[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];
    for(int i = 0; i < 2 * FILTER_RADIUS + 1; i++){
        for(int j = 0; j < 2 * FILTER_RADIUS + 1; j++){
            filter_constant[i][j] = filter[i][j];
        }
    }
    cudaMemcpyToSymbol(F, filter_constant, (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float));


    // Allocate memory on device
    cudaMalloc((void **) &deviceImg, height * width * sizeof(float));
    cudaMalloc((void **) &deviceResult, height * width * sizeof(float));
    cudaMalloc((void **) &deviceFilter, filter_size * filter_size * sizeof(float));

    // Copy host memory to device
    cudaMemcpy(deviceImg, hostImg, height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFilter, hostFilter, filter_size * filter_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(32, 32);
    dim3 gridDim(ceil((float)width / blockDim.x), ceil((float)height / blockDim.y));

    conv_2d_basic_kernel<<<gridDim, blockDim>>>(deviceImg, deviceFilter, deviceResult, r, width, height);

    cudaMemcpy(hostResult, deviceResult, height * width * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "check conv_2d_basic_kernel" << endl;
    //check
    idx = 0;
    for(auto row : output){
        for(auto col: row){
            if(abs(col - hostResult[idx]) > 1e-6){
                cout << idx << " bad: " << col << " - " << hostResult[idx] << " > 1e-6" << endl;
            }
            idx++;
        }
    } 

    conv_2d_constant_mem_kernel<<<gridDim, blockDim>>>(deviceImg, deviceResult, r, width, height);

    cudaMemcpy(hostResult, deviceResult, height * width * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "check conv_2d_constant_mem_kernel" << endl;
    //check
    idx = 0;
    for(auto row : output){
        for(auto col: row){
            if(abs(col - hostResult[idx]) > 1e-6){
                cout << idx << " bad: " << col << " - " << hostResult[idx] << " > 1e-6" << endl;
            }
            idx++;
        }
    } 

    dim3 blockDim2(OUT_TILE_DIM, OUT_TILE_DIM);
    dim3 gridDim2(ceil((float)width / blockDim.x), ceil((float)height / blockDim.y));

    conv_tiled_2d_constant_mem_kernel<<<gridDim2, blockDim2>>>(deviceImg, deviceResult, width, height);

    cudaMemcpy(hostResult, deviceResult, height * width * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "check conv_tiled_2d_constant_mem_kernel" << endl;
    //check
    idx = 0;
    for(auto row : output){
        for(auto col: row){
            if(abs(col - hostResult[idx]) > 1e-6){
                cout << idx << " bad: " << col << " - " << hostResult[idx] << " > 1e-6" << endl;

            }
            idx++;
        }
    } 

    dim3 blockDim3(TILE_DIM, TILE_DIM);
    dim3 gridDim3(ceil((float)width / blockDim.x), ceil((float)height / blockDim.y));

    conv_cached_tiled_2d_constant_mem_kernel<<<gridDim3, blockDim3>>>(deviceImg, deviceResult, width, height);

    cudaMemcpy(hostResult, deviceResult, height * width * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "check conv_cached_tiled_2d_constant_mem_kernel" << endl;
    //check
    idx = 0;
    for(auto row : output){
        for(auto col: row){
            if(abs(col - hostResult[idx]) > 1e-6){
                cout << idx << " bad: " << col << " - " << hostResult[idx] << " > 1e-6" << endl;

            }
            idx++;
        }
    } 

    free(hostImg);
    free(hostFilter);
    free(hostResult); 

    cudaFree(deviceImg);
    cudaFree(deviceFilter);
    cudaFree(deviceResult);
    
    return 0;

}