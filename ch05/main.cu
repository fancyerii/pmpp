#include <cuda.h>
#include <iostream>
#include <vector>

// Matrix Multiply: A x B  on the GPU. Results stored into C
__global__
void vanillaMatrixMulKernel(float *A, float *B, float *C, 
                            int numARows, int numAColumns, 
                            int numBRows, int numBColumns){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;                            
    float sum = 0.0f;

    if (row >= numARows || col >= numBColumns) return;

    for(int k = 0; k < numAColumns; k++){
        sum += A[row * numAColumns + k] * B[k * numBColumns + col];
    }

    C[row * numBColumns + col] = sum;
}

#define TILE_WIDTH 32
__global__
void tiledMatrixMulKernel(float *A, float *B, float *C, 
                            int numARows, int numAColumns, 
                            int numBRows, int numBColumns){
        
    // Initialize shared memory
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;
    // Loop over the M and N tiles required to compute the P element
    for (int ph = 0; ph < ceil(numAColumns/(float)TILE_WIDTH); ph++) {

        // Collaborative loading of M and N tiles into shared memory
        if (row < numARows && ph * TILE_WIDTH + tx < numAColumns)
            ds_M[ty][tx] = A[row * numAColumns + ph * TILE_WIDTH + tx];
        else
            ds_M[ty][tx] = 0;

        if (col < numBColumns && ph * TILE_WIDTH + ty < numBRows)
            ds_N[ty][tx] = B[(ph * TILE_WIDTH + ty) * numBColumns + col];
        else
            ds_N[ty][tx] = 0;

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Compute the P element
        for (int k = 0; k < TILE_WIDTH; k++)
            Pvalue += ds_M[ty][k] * ds_N[k][tx];
        
        // Synchronize to make sure the P elembasic_matrix_mulent is computed
        // before other threads load new tiles
        __syncthreads();
    }

    // Store the P element in C
    if (row < numARows && col < numBColumns)
        C[row * numBColumns + col] = Pvalue;
}


// C = A + B on a GPU, where A is a vector of 1.0f and B a vector of 2.0f
// The main function takes one argument, the size of the vectors
int main(int argc, char* argv[])
{

    float *hostA, *hostB, *hostC1, *hostC2;
    float *deviceA1, *deviceB1, *deviceC1;
    float *deviceA2, *deviceB2, *deviceC2;
    int numARows, numAColumns;
    int numBRows, numBColumns;
    int numCRows, numCColumns;

    if (argc != 5){
        printf("Usage: ./a.out <num_rows_A> <num_columns_A> <num_rows_B> <num_columns_B>\n");
        return 1;
    }

    numARows = atoi(argv[1]);
    numAColumns = atoi(argv[2]);
    numBRows = atoi(argv[3]);
    numBColumns = atoi(argv[4]);

    numCRows = numARows;
    numCColumns = numBColumns;
    
    if(numAColumns != numBRows) {
        printf("Number of columns in A must be the same as the number of rows in B\n");
        return 1;
    }

    // Allocate memory on host
    hostA = (float *) malloc(numARows * numAColumns * sizeof(float));
    hostB = (float *) malloc(numBRows * numBColumns * sizeof(float));
    hostC1 = (float *) malloc(numCRows * numCColumns * sizeof(float));
    hostC2 = (float *) malloc(numCRows * numCColumns * sizeof(float));

    // Allocate memory on device
    cudaMalloc((void **) &deviceA1, numARows * numAColumns * sizeof(float));
    cudaMalloc((void **) &deviceB1, numBRows * numBColumns * sizeof(float));
    cudaMalloc((void **) &deviceC1, numCRows * numCColumns * sizeof(float));

    cudaMalloc((void **) &deviceA2, numARows * numAColumns * sizeof(float));
    cudaMalloc((void **) &deviceB2, numBRows * numBColumns * sizeof(float));
    cudaMalloc((void **) &deviceC2, numCRows * numCColumns * sizeof(float));

    // Initialize host memory
    srand(time(NULL));
    for (int i = 0; i < numARows; i++)
        for (int j = 0; j < numAColumns; j++)
            hostA[i * numAColumns + j] = rand() / (float) RAND_MAX;

    for (int i = 0; i < numBRows; i++)
        for (int j = 0; j < numBColumns; j++)
            hostB[i * numBColumns + j] = rand() / (float) RAND_MAX;


    // Copy host memory to device
    cudaMemcpy(deviceA1, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB1, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(deviceA2, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB2, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);


    // Launch kernel
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(ceil((float)numCColumns / blockDim.x), ceil((float)numCRows / blockDim.y));
    
    vanillaMatrixMulKernel<<<gridDim, blockDim>>>(deviceA1, deviceB1, deviceC1,
                            numARows, numAColumns,
                            numBRows, numBColumns);
    tiledMatrixMulKernel<<<gridDim, blockDim>>>(deviceA2, deviceB2, deviceC2,
                            numARows, numAColumns,
                            numBRows, numBColumns);


    // Copy device memory to host
    cudaMemcpy(hostC1, deviceC1, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostC2, deviceC2, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef CHECK_EQUAL
    std::cout << "check two algorithms equal" << std::endl;
    // check hostC1 == hostC2
    for (int i = 0; i < numCRows; i++) {
        for (int j = 0; j < numCColumns; j++){
            float c1 = hostC1[i * numCColumns + j];
            float c2 = hostC2[i * numCColumns + j];
            if (abs(c1-c2) > 1e-7){
                std::cout << "i=" << i << ", j=" << j
                     << "c1=" << c1 << ", c2=" << c2
                     << std::endl;
            }
        }
 
    }
#endif

    // Free memory
    free(hostA);
    free(hostB);
    free(hostC1);
    free(hostC2);
    cudaFree(deviceA1);
    cudaFree(deviceB1);
    cudaFree(deviceC1);
    cudaFree(deviceA2);
    cudaFree(deviceB2);
    cudaFree(deviceC2);
    return 0;

}