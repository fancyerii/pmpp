#include <cuda.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>
#define OUT_TILE_DIM 8
#define RADIUS 1
#define IN_TILE_DIM ((OUT_TILE_DIM) + 2*(RADIUS))


using json = nlohmann::json;
using namespace std;


__global__ 
void stencil_3d_basic_kernel(float *in, float *out, float *w,
                    int Z, int Y, int X){
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(i > 0 && i < Z - 1 && j > 0 && j < Y - 1 && k > 0 && k < X - 1){
        out[i * Y * X + j * X + k] = w[0] * in[i * Y * X + j * X + k] 
                                   + w[1] * in[i * Y * X + j * X + (k - 1)]
                                   + w[2] * in[i * Y * X + j * X + (k + 1)]
                                   + w[3] * in[i * Y * X + (j - 1) * X + k]
                                   + w[4] * in[i * Y * X + (j + 1) * X + k]
                                   + w[5] * in[(i - 1) * Y * X + j * X + k] 
                                   + w[6] * in[(i + 1) * Y * X + j * X + k]; 
    }
}

__global__
void stencil_3d_tile_in_coarsen_kernel(float *in, float *out, float *w,
                    int Z, int Y, int X){
    
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];
    
    if(iStart - 1 >= 0 && iStart - 1 < Z && j >= 0 && j < Y && k >= 0 && k < X){
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart - 1) * Y * X + j * X + k];
    }

    if(iStart >= 0 && iStart < Z && j >= 0 && j < Y && k >= 0 && k < X){
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart * Y * X + j * X + k];
    }

    for(int i = iStart; i < iStart + OUT_TILE_DIM; i++){
        if(i + 1 >= 0 && i + 1 < Z && j >= 0 && j < Y && k >= 0 && k < X){
            inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1) * Y * X + j * X + k];
        }
        __syncthreads();

        if(i >= 1 && i < Z - 1 && j >= 1 && j < Y - 1 && k >= 1 && k < X - 1){       
            if(threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
                threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1){
                out[i * Y * X + j * X + k] = 
                          w[0] * inCurr_s[threadIdx.y][threadIdx.x]
                        + w[1] * inCurr_s[threadIdx.y][threadIdx.x - 1]
                        + w[2] * inCurr_s[threadIdx.y][threadIdx.x + 1]
                        + w[3] * inCurr_s[threadIdx.y - 1][threadIdx.x]
                        + w[4] * inCurr_s[threadIdx.y + 1][threadIdx.x]
                        + w[5] * inPrev_s[threadIdx.y][threadIdx.x]
                        + w[6] * inNext_s[threadIdx.y][threadIdx.x];
            }
        } 
        __syncthreads();   
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];    
    }

}

__global__
void stencil_3d_tile_in_coarsen_register_kernel(float *in, float *out, float *w,
                    int Z, int Y, int X){
    
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    float inPrev;
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    float inCurr;
    float inNext;

    if(iStart - 1 >= 0 && iStart - 1 < Z && j >= 0 && j < Y && k >= 0 && k < X){
        inPrev = in[(iStart - 1) * Y * X + j * X + k];
    }

    if(iStart >= 0 && iStart < Z && j >= 0 && j < Y && k >= 0 && k < X){
        inCurr = in[iStart * Y * X + j * X + k];
        inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
    }

    for(int i = iStart; i < iStart + OUT_TILE_DIM; i++){
        if(i + 1 >= 0 && i + 1 < Z && j >= 0 && j < Y && k >= 0 && k < X){
            inNext = in[(i + 1) * Y * X + j * X + k];
        }
        __syncthreads();

        if(i >= 1 && i < Z - 1 && j >= 1 && j < Y - 1 && k >= 1 && k < X - 1){       
            if(threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
                threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1){
                out[i * Y * X + j * X + k] = 
                          w[0] * inCurr
                        + w[1] * inCurr_s[threadIdx.y][threadIdx.x - 1]
                        + w[2] * inCurr_s[threadIdx.y][threadIdx.x + 1]
                        + w[3] * inCurr_s[threadIdx.y - 1][threadIdx.x]
                        + w[4] * inCurr_s[threadIdx.y + 1][threadIdx.x]
                        + w[5] * inPrev
                        + w[6] * inNext;
            }
        } 
        __syncthreads();
        inPrev = inCurr;
        inCurr = inNext;   
        inCurr_s[threadIdx.y][threadIdx.x] = inNext;    
    }

}

__global__ 
void stencil_3d_tile_in_kernel(float *in, float *out, float *w,
                    int Z, int Y, int X){
    
    int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    if(i >= 0 && i < Z && j >= 0 && j < Y && k >= 0 && k < X){
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * Y * X + j * X + k];
    }
    __syncthreads();

    if(i >= 1 && i < Z - 1 && j >= 1 && j < Y - 1 && k >= 1 && k < X - 1){
        if(threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1 &&
           threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
           threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1){

            out[i * Y * X + j * X + k] = 
                        w[0] * in_s[threadIdx.z][threadIdx.y][threadIdx.x]
                      + w[1] * in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1]
                      + w[2] * in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1]
                      + w[3] * in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x]
                      + w[4] * in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x]
                      + w[5] * in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x]
                      + w[6] * in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];

        }
    }
}

__global__ 
void stencil_3d_tile_kernel(float *in, float *out, float *w,
                    int Z, int Y, int X){
    unsigned int out_z = blockIdx.z * OUT_TILE_DIM + threadIdx.z;
    unsigned int out_y = blockIdx.y * OUT_TILE_DIM + threadIdx.y;
    unsigned int out_x = blockIdx.x * OUT_TILE_DIM + threadIdx.x;  

    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    int loop_count = ceil(IN_TILE_DIM/(float)OUT_TILE_DIM);
    for(int i = 0; i < loop_count; i++){
        int zIdx = OUT_TILE_DIM * i + threadIdx.z;
        int z = zIdx - RADIUS + blockIdx.z * OUT_TILE_DIM;
        for(int j = 0; j < loop_count; j++){
            int yIdx = OUT_TILE_DIM * j + threadIdx.y;
            int y = yIdx - RADIUS + blockIdx.y * OUT_TILE_DIM;
            for(int k = 0; k < loop_count; k++){
                int xIdx = OUT_TILE_DIM * k + threadIdx.x;
                int x = xIdx - RADIUS + blockIdx.x * OUT_TILE_DIM;
                if(zIdx < IN_TILE_DIM && yIdx < IN_TILE_DIM && xIdx < IN_TILE_DIM){
                    if(z >= 0 && z < Z && y >= 0 && y < Y && x >= 0 && x < X){
                        in_s[zIdx][yIdx][xIdx] = in[z * Y * X + y * X + x]; 
                    }else{
                        in_s[zIdx][yIdx][xIdx] = 0;
                    }
                }
            }
        }
    }
    __syncthreads();

    if(out_z > 0 && out_z < Z - 1 && out_y > 0 && out_y < Y - 1 && out_x > 0 && out_x < X - 1){
        int zs = threadIdx.z + RADIUS;
        int ys = threadIdx.y + RADIUS;
        int xs = threadIdx.x + RADIUS;
        float Pvalue = w[0] * in_s[zs][ys][xs];
        Pvalue += w[1] * in_s[zs][ys][xs - 1];
        Pvalue += w[2] * in_s[zs][ys][xs + 1];
        Pvalue += w[3] * in_s[zs][ys - 1][xs];
        Pvalue += w[4] * in_s[zs][ys + 1][xs];
        Pvalue += w[5] * in_s[zs - 1][ys][xs];
        Pvalue += w[6] * in_s[zs + 1][ys][xs];

        out[out_z * Y * X + out_y * X + out_x] = Pvalue;
    }
}

void check_result(const std::vector<std::vector<std::vector<float>>> & input,
                  const std::vector<float> & w,
                  const std::vector<std::vector<std::vector<float>>> & output,
                  int X, int Y, int Z, int W,
                  dim3 blockDim,
                  dim3 gridDim,
                  std::string kernel_name,
                  void kernel(float *in, float *out, float *w,
                    int Z, int Y, int X)

){

    float *hostInput, *hostW, *hostResult;
    float *deviceInput, *deviceW, *deviceResult; 


    // Allocate memory on host
    hostInput = (float *) malloc(Z * Y * X * sizeof(float)); 
    hostResult = (float *) malloc(Z * Y * X * sizeof(float));
    hostW = (float *) malloc(W * sizeof(float));

    int idx = 0;
    for(int i = 0; i < Z; i++){
        auto zz = input[i];
        for(int j = 0; j < Y; j++){
            auto yy = zz[j];
            for(int k = 0; k < X; k++){
                auto xx = yy[k];
                hostInput[idx++] = xx;
                if(i == 0 || i == Z - 1 ||
                   j == 0 || j == Y - 1 ||
                   k == 0 || k == X - 1){
                    hostResult[idx] = xx;
                }
            }
        }
    }

    idx = 0;
    for(auto i : w){
        hostW[idx++] = i;
    }
    

    // Allocate memory on device
    cudaMalloc((void **) &deviceInput, Z * Y * X * sizeof(float));
    cudaMalloc((void **) &deviceResult, Z * Y * X * sizeof(float));
    cudaMalloc((void **) &deviceW, W * sizeof(float));

    // Copy host memory to device
    cudaMemcpy(deviceInput, hostInput, Z * Y * X  * sizeof(float), cudaMemcpyHostToDevice);
    // border value of result is equal to input
    cudaMemcpy(deviceResult, hostInput, Z * Y * X  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceW, hostW, W * sizeof(float), cudaMemcpyHostToDevice);


    kernel<<<gridDim, blockDim>>>(deviceInput, deviceResult, deviceW, Z, Y, X);

    cudaMemcpy(hostResult, deviceResult, Z * Y * X * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "check " << kernel_name << endl;

    idx = 0;
    for(auto zz : output){
        for(auto yy: zz){
            for(auto xx: yy){
                if(abs(xx - hostResult[idx]) > 1e-6){
                    cout << idx << " bad: output=" << xx << " - result=" << hostResult[idx] << " > 1e-6" << endl;
                    goto fail;
                }
                idx++;
            }
            
            
        }
    }
    fail:

    free(hostInput);
    free(hostW);
    free(hostResult); 

    cudaFree(deviceInput);
    cudaFree(deviceW);
    cudaFree(deviceResult);
}

int main(int argc, char* argv[]){
    std::ifstream f("stencil.json");
    json data = json::parse(f);
    auto input = data["input"].template get<std::vector<std::vector<std::vector<float>>>>(); 
    auto w = data["w"].template get<std::vector<float>>(); 
    auto output = data["output"].template get<std::vector<std::vector<std::vector<float>>>>(); 
    

    int Z = input.size();
    int Y = input[0].size();
    int X = input[0][0].size();

    int W = w.size();
    
    {
        dim3 blockDim(8, 8, 8);
        dim3 gridDim(ceil((float)X / blockDim.x), ceil((float)Y / blockDim.y), ceil((float)Z / blockDim.z));
        check_result(input, w, output, X, Y, Z, W, blockDim, gridDim, "stencil_3d_basic_kernel", stencil_3d_basic_kernel);
    }

    {
        dim3 blockDim(OUT_TILE_DIM, OUT_TILE_DIM, OUT_TILE_DIM);
        dim3 gridDim(ceil((float)X / blockDim.x), ceil((float)Y / blockDim.y), ceil((float)Z / blockDim.z));
        check_result(input, w, output, X, Y, Z, W, blockDim, gridDim, "stencil_3d_tile_kernel", stencil_3d_tile_kernel);
    }

    {
        dim3 blockDim(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
        dim3 gridDim(ceil((float)X / OUT_TILE_DIM), ceil((float)Y / OUT_TILE_DIM), ceil((float)Z / OUT_TILE_DIM));
        check_result(input, w, output, X, Y, Z, W, blockDim, gridDim, "stencil_3d_tile_in_kernel", stencil_3d_tile_in_kernel);
    } 

    {
        dim3 blockDim(IN_TILE_DIM, IN_TILE_DIM, 1);
        dim3 gridDim(ceil((float)X / OUT_TILE_DIM), ceil((float)Y / OUT_TILE_DIM), ceil((float)Z / OUT_TILE_DIM));
        check_result(input, w, output, X, Y, Z, W, blockDim, gridDim, "stencil_3d_tile_in_coarsen_kernel", stencil_3d_tile_in_coarsen_kernel);
    } 

    {
        dim3 blockDim(IN_TILE_DIM, IN_TILE_DIM, 1);
        dim3 gridDim(ceil((float)X / OUT_TILE_DIM), ceil((float)Y / OUT_TILE_DIM), ceil((float)Z / OUT_TILE_DIM));
        check_result(input, w, output, X, Y, Z, W, blockDim, gridDim, "stencil_3d_tile_in_coarsen_register_kernel", stencil_3d_tile_in_coarsen_register_kernel);
    } 
    return 0;

}