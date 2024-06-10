#include <cuda.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <ctime>

#define MLEN 80000
#define NLEN 70000 
#define MAX 5000
#define BLOCK_SIZE 16
#define GRID_SIZE 32
#define TILE_SIZE 1024

// if you want to reproduce the bug, try using the following:
// #define MLEN 8
// #define NLEN 7 
// #define MAX 10
// #define BLOCK_SIZE 2
// #define GRID_SIZE 2
// #define TILE_SIZE 4

using namespace std;


void generate_rand_int(int * array, int len, int seed, int max){
    srand(seed);
    for(int i = 0; i < len; ++i){
        array[i] = rand() % max;
    }
}

int co_rank_host(int k, int *A, int m, int *B, int n){
    int i = k < m ? k : m;
    int j = k - i;
    int i_low = 0 > (k - n) ? 0 : (k - n);
    int j_low = 0 > (k - m) ? 0 : (k - m);
    int delta;
    bool active = true;

    while(active){
        if(i > 0 && j < n && A[i - 1] > B[j]){
            delta = (i - i_low + 1) >> 1;
            j_low = j;
            i -= delta;
            j += delta;
        }else if(j > 0 && i < m && B[j - 1] >= A[i]){
            delta = (j - j_low + 1) >> 1;
            i_low = i;
            j -= delta;
            i += delta;
        }else{
            active = false;
        }
    }

    return i;
}

__device__
int co_rank(int k, int *A, int m, int *B, int n){
    int i = k < m ? k : m;
    int j = k - i;
    int i_low = 0 > (k - n) ? 0 : (k - n);
    int j_low = 0 > (k - m) ? 0 : (k - m);
    int delta;
    bool active = true;

    while(active){
        if(i > 0 && j < n && A[i - 1] > B[j]){
            delta = (i - i_low + 1) >> 1;
            j_low = j;
            i -= delta;
            j += delta;
        }else if(j > 0 && i < m && B[j - 1] >= A[i]){
            delta = (j - j_low + 1) >> 1;
            i_low = i;
            j -= delta;
            i += delta;
        }else{
            active = false;
        }
    }

    return i;
}

__device__
int co_rank_circular(int k, int *A, int m, int *B, int n,
                     int A_S_start, int B_S_start, int tile_size){
    int i = k < m ? k : m;
    int j = k - i;
    int i_low = 0 > (k - n) ? 0 : (k - n);
    int j_low = 0 > (k - m) ? 0 : (k - m);
    int delta;
    bool active = true;

    while(active){
        int i_cir = (i + A_S_start) % tile_size;
        int i_m_1_cir = (i + A_S_start - 1) % tile_size;
        int j_cir = (j + B_S_start) % tile_size;
        int j_m_1_cir = (j + B_S_start - 1) % tile_size;

        if(i > 0 && j < n && A[i_m_1_cir] > B[j_cir]){
            delta = (i - i_low + 1) >> 1;
            j_low = j;
            i -= delta;
            j += delta;
        }else if(j > 0 && i < m && B[j_m_1_cir] >= A[i_cir]){
            delta = (j - j_low + 1) >> 1;
            i_low = i;
            j -= delta;
            i += delta;
        }else{
            active = false;
        }
    }

    return i;
}

__device__ 
void merge_sequential(int *A, int m, int *B, int n, int *C) {
    int i = 0;
    int j = 0;
    int k = 0;

    while((i < m) && (j < n)){
        if(A[i] <= B[j]){
            C[k++] = A[i++];
        }else{
            C[k++] = B[j++];
        }
    }

    if(i == m){
        while(j < n){
            C[k++] = B[j++];
        }
    }else{
        while(i < m){
            C[k++] = A[i++];
        }
    }
}

__device__ 
void merge_sequential_circular(int *A, int m, int *B, int n, int *C,
                               int A_S_start, int B_S_start, int tile_size) {
    int i = 0;
    int j = 0;
    int k = 0;

    while((i < m) && (j < n)){
        int i_cir = (i + A_S_start) % tile_size;
        int j_cir = (j + B_S_start) % tile_size;
        if(A[i_cir] <= B[j_cir]){
            C[k++] = A[i_cir];
            i++;
        }else{
            C[k++] = B[j_cir];
            j++;
        }
    }

    if(i == m){
        while(j < n){
            int j_cir = (j + B_S_start) % tile_size;
            C[k++] = B[j_cir];
            j++;
        }
    }else{
        while(i < m){
            int i_cir = (i + A_S_start) % tile_size;
            C[k++] = A[i_cir];
            i++;
        }
    }
}

void merge_sequential_host(int *A, int m, int *B, int n, int *C) {
    int i = 0;
    int j = 0;
    int k = 0;

    while((i < m) && (j < n)){
        if(A[i] <= B[j]){
            C[k++] = A[i++];
        }else{
            C[k++] = B[j++];
        }
    }

    if(i == m){
        while(j < n){
            C[k++] = B[j++];
        }
    }else{
        while(i < m){
            C[k++] = A[i++];
        }
    }
}

__global__
void merge_basic_kernel(int *A, int m, int *B, int n, int *C){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int elementsPerThread = ceil((m + n) * 1.0 / (blockDim.x * gridDim.x));

    int k_curr = tid * elementsPerThread;
    int k_next = min((tid + 1) * elementsPerThread, (m + n));

    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);

    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;

    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
}

__global__
void merge_tiled_kernel_bug(int *A, int m, int *B, int n, int *C, int tile_size){
    // for dynamic shared memory allocation, see 
    // https://stackoverflow.com/questions/24942073/dynamic-shared-memory-in-cuda
    extern __shared__ int shareAB[];

    int * A_S = &shareAB[0];
    int * B_S = &shareAB[tile_size];



    // the original codes in the book is not correct because n/d will descard
    // decimal part, see https://stackoverflow.com/questions/26105925/use-of-ceil-and-integers 
    // we can also use ceil(n/d) = (n+d-1)/d but it seems hard to understand.
    // when calculate C_curr, float is converted to int implicitly by compiler
    // but min function don't know call which overloaded function if we pass
    // min(2.0, 3). So we convert 2.0 to 2.
    int C_curr = blockIdx.x * ceil(1.0*(m+n)/gridDim.x);
    int C_next = min((blockIdx.x + 1) * int(ceil(1.0*(m+n)/gridDim.x)), m + n);

    if(threadIdx.x == 0){
        A_S[0] = co_rank(C_curr, A, m, B, n);
        B_S[0] = co_rank(C_next, A, m, B, n);
    }

    __syncthreads();
    int A_curr = A_S[0];
    int A_next = B_S[0];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads(); // we need this because A_S[0] and B_S[0] will be modified.


    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = ceil(1.0 * C_length / tile_size);
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    while(counter < total_iteration){
        for(int i = 0; i < tile_size; i += blockDim.x){
            if(i + threadIdx.x < A_length - A_consumed){
                A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
            }
        }
        for(int i = 0; i < tile_size; i += blockDim.x){
            if(i + threadIdx.x < B_length - B_consumed){
                B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();

        int c_curr = threadIdx.x * (tile_size / blockDim.x);
        int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);

        c_curr = (c_curr < (C_length - C_completed)) ? c_curr : (C_length - C_completed);
        c_next = (c_next < (C_length - C_completed)) ? c_next : (C_length - C_completed);
        
        int a_curr = co_rank(c_curr, A_S, min(tile_size, A_length - A_consumed),
                                     B_S, min(tile_size, B_length - B_consumed));
        int b_curr = c_curr - a_curr;

        int a_next = co_rank(c_next, A_S, min(tile_size, A_length - A_consumed),
                                     B_S, min(tile_size, B_length - B_consumed));
        int b_next = c_next - a_next;

        merge_sequential(A_S + a_curr, a_next - a_curr, B_S + b_curr, b_next - b_curr,
                         C + C_curr + c_curr + C_completed);

        counter++;
        C_completed += tile_size;

        A_consumed += co_rank(tile_size, A_S, tile_size, B_S, tile_size);
        B_consumed = C_completed - A_consumed;   
        __syncthreads();
    }
}

__global__
void merge_tiled_kernel(int *A, int m, int *B, int n, int *C, int tile_size){
    // for dynamic shared memory allocation, see 
    // https://stackoverflow.com/questions/24942073/dynamic-shared-memory-in-cuda
    extern __shared__ int shareAB[];

    int * A_S = &shareAB[0];
    int * B_S = &shareAB[tile_size];



    // the original codes in the book is not correct because n/d will descard
    // decimal part, see https://stackoverflow.com/questions/26105925/use-of-ceil-and-integers 
    // we can also use ceil(n/d) = (n+d-1)/d but it seems hard to understand.
    // when calculate C_curr, float is converted to int implicitly by compiler
    // but min function don't know call which overloaded function if we pass
    // min(2.0, 3). So we convert 2.0 to 2.
    int C_curr = blockIdx.x * ceil(1.0*(m+n)/gridDim.x);
    int C_next = min((blockIdx.x + 1) * int(ceil(1.0*(m+n)/gridDim.x)), m + n);

    if(threadIdx.x == 0){
        A_S[0] = co_rank(C_curr, A, m, B, n);
        B_S[0] = co_rank(C_next, A, m, B, n);
    }

    __syncthreads();
    int A_curr = A_S[0];
    int A_next = B_S[0];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads(); // we need this because A_S[0] and B_S[0] will be modified.

    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = ceil(1.0 * C_length / tile_size);
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    while(counter < total_iteration){
        for(int i = 0; i < tile_size; i += blockDim.x){
            if(i + threadIdx.x < A_length - A_consumed){
                A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
            }
        }
        for(int i = 0; i < tile_size; i += blockDim.x){
            if(i + threadIdx.x < B_length - B_consumed){
                B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();

        int c_curr = threadIdx.x * (tile_size / blockDim.x);
        int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);

        c_curr = (c_curr < (C_length - C_completed)) ? c_curr : (C_length - C_completed);
        c_next = (c_next < (C_length - C_completed)) ? c_next : (C_length - C_completed);
        
        int a_curr = co_rank(c_curr, A_S, min(tile_size, A_length - A_consumed),
                                     B_S, min(tile_size, B_length - B_consumed));
        int b_curr = c_curr - a_curr;

        int a_next = co_rank(c_next, A_S, min(tile_size, A_length - A_consumed),
                                     B_S, min(tile_size, B_length - B_consumed));
        int b_next = c_next - a_next;

        merge_sequential(A_S + a_curr, a_next - a_curr, B_S + b_curr, b_next - b_curr,
                         C + C_curr + c_curr + C_completed);

        counter++;
        C_completed += tile_size;
        // bug !!!
        // A_consumed += co_rank(tile_size, A_S, tile_size, B_S, tile_size);
        A_consumed += co_rank(tile_size, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed));
        B_consumed = C_completed - A_consumed;   
        __syncthreads();
    }
}

__global__
void merge_tiled_kernel_v2(int *A, int m, int *B, int n, int *C, int tile_size){
    // for dynamic shared memory allocation, see 
    // https://stackoverflow.com/questions/24942073/dynamic-shared-memory-in-cuda
    extern __shared__ int shareAB[];
    __shared__ int A_consumed;
    int * A_S = &shareAB[0];
    int * B_S = &shareAB[tile_size];



    // the original codes in the book is not correct because n/d will descard
    // decimal part, see https://stackoverflow.com/questions/26105925/use-of-ceil-and-integers 
    // we can also use ceil(n/d) = (n+d-1)/d but it seems hard to understand.
    // when calculate C_curr, float is converted to int implicitly by compiler
    // but min function don't know call which overloaded function if we pass
    // min(2.0, 3). So we convert 2.0 to 2.
    int C_curr = blockIdx.x * ceil(1.0*(m+n)/gridDim.x);
    int C_next = min((blockIdx.x + 1) * int(ceil(1.0*(m+n)/gridDim.x)), m + n);

    if(threadIdx.x == 0){
        A_S[0] = co_rank(C_curr, A, m, B, n);
        B_S[0] = co_rank(C_next, A, m, B, n);
        A_consumed = 0;
    }

    __syncthreads();
    int A_curr = A_S[0];
    int A_next = B_S[0];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads(); // we need this because A_S[0] and B_S[0] will be modified.

    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = ceil(1.0 * C_length / tile_size);
    int C_completed = 0;

    int B_consumed = 0;

    while(counter < total_iteration){
        for(int i = 0; i < tile_size; i += blockDim.x){
            if(i + threadIdx.x < A_length - A_consumed){
                A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
            }
        }
        for(int i = 0; i < tile_size; i += blockDim.x){
            if(i + threadIdx.x < B_length - B_consumed){
                B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();

        int c_curr = threadIdx.x * (tile_size / blockDim.x);
        int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);

        c_curr = (c_curr < (C_length - C_completed)) ? c_curr : (C_length - C_completed);
        c_next = (c_next < (C_length - C_completed)) ? c_next : (C_length - C_completed);
        
        int a_curr = co_rank(c_curr, A_S, min(tile_size, A_length - A_consumed),
                                     B_S, min(tile_size, B_length - B_consumed));
        int b_curr = c_curr - a_curr;

        int a_next = co_rank(c_next, A_S, min(tile_size, A_length - A_consumed),
                                     B_S, min(tile_size, B_length - B_consumed));
        int b_next = c_next - a_next;

        merge_sequential(A_S + a_curr, a_next - a_curr, B_S + b_curr, b_next - b_curr,
                         C + C_curr + c_curr + C_completed);

        counter++;
        C_completed += tile_size;
        if(threadIdx.x == 0){
            A_consumed += co_rank(tile_size, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed));
        }
        __syncthreads();
        B_consumed = C_completed - A_consumed;     
    }
}

__global__
void merge_tiled_kernel_v3(int *A, int m, int *B, int n, int *C, int tile_size){
    // for dynamic shared memory allocation, see 
    // https://stackoverflow.com/questions/24942073/dynamic-shared-memory-in-cuda
    extern __shared__ int shareAB[];
    __shared__ int A_consumed;
    int * A_S = &shareAB[0];
    int * B_S = &shareAB[tile_size];



    // the original codes in the book is not correct because n/d will descard
    // decimal part, see https://stackoverflow.com/questions/26105925/use-of-ceil-and-integers 
    // we can also use ceil(n/d) = (n+d-1)/d but it seems hard to understand.
    // when calculate C_curr, float is converted to int implicitly by compiler
    // but min function don't know call which overloaded function if we pass
    // min(2.0, 3). So we convert 2.0 to 2.
    int C_curr = blockIdx.x * ceil(1.0*(m+n)/gridDim.x);
    int C_next = min((blockIdx.x + 1) * int(ceil(1.0*(m+n)/gridDim.x)), m + n);

    if(threadIdx.x == 0){
        A_S[0] = co_rank(C_curr, A, m, B, n);
        B_S[0] = co_rank(C_next, A, m, B, n);
        A_consumed = 0;
    }

    __syncthreads();
    int A_curr = A_S[0];
    int A_next = B_S[0];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads(); // we need this because A_S[0] and B_S[0] will be modified.

    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = ceil(1.0 * C_length / tile_size);
    int C_completed = 0;

    int B_consumed = 0;

    while(counter < total_iteration){
        for(int i = 0; i < tile_size; i += blockDim.x){
            if(i + threadIdx.x < A_length - A_consumed){
                A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
            }
        }
        for(int i = 0; i < tile_size; i += blockDim.x){
            if(i + threadIdx.x < B_length - B_consumed){
                B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();

        int c_curr = threadIdx.x * (tile_size / blockDim.x);
        int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);

        c_curr = (c_curr < (C_length - C_completed)) ? c_curr : (C_length - C_completed);
        c_next = (c_next < (C_length - C_completed)) ? c_next : (C_length - C_completed);
        
        int a_curr = co_rank(c_curr, A_S, min(tile_size, A_length - A_consumed),
                                     B_S, min(tile_size, B_length - B_consumed));
        int b_curr = c_curr - a_curr;

        int a_next = co_rank(c_next, A_S, min(tile_size, A_length - A_consumed),
                                     B_S, min(tile_size, B_length - B_consumed));
        int b_next = c_next - a_next;

        merge_sequential(A_S + a_curr, a_next - a_curr, B_S + b_curr, b_next - b_curr,
                         C + C_curr + c_curr + C_completed);

        counter++;
        C_completed += tile_size;

        atomicAdd(&A_consumed, a_next - a_curr);
        __syncthreads();
        B_consumed = C_completed - A_consumed;     
        
    }
}


__global__
void merge_circular_buffer_kernel(int *A, int m, int *B, int n, int *C, int tile_size){
    // for dynamic shared memory allocation, see 
    // https://stackoverflow.com/questions/24942073/dynamic-shared-memory-in-cuda
    extern __shared__ int shareAB[];

    int * A_S = &shareAB[0];
    int * B_S = &shareAB[tile_size];

    // the original codes in the book is not correct because n/d will descard
    // decimal part, see https://stackoverflow.com/questions/26105925/use-of-ceil-and-integers 
    // we can also use ceil(n/d) = (n+d-1)/d but it seems hard to understand.
    // when calculate C_curr, float is converted to int implicitly by compiler
    // but min function don't know call which overloaded function if we pass
    // min(2.0, 3). So we convert 2.0 to 2.
    int C_curr = blockIdx.x * ceil(1.0*(m+n)/gridDim.x);
    int C_next = min((blockIdx.x + 1) * int(ceil(1.0*(m+n)/gridDim.x)), m + n);

    if(threadIdx.x == 0){
        A_S[0] = co_rank(C_curr, A, m, B, n);
        B_S[0] = co_rank(C_next, A, m, B, n);
    }

    __syncthreads();
    int A_curr = A_S[0];
    int A_next = B_S[0];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads(); // we need this because A_S[0] and B_S[0] will be modified.

    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = ceil(1.0 * C_length / tile_size);
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    int A_S_start = 0;
    int B_S_start = 0;
    int A_S_consumed = tile_size;
    int B_S_consumed = tile_size;
    
    int A_loaded = 0;
    int B_loaded = 0;

    while(counter < total_iteration){
        for(int i = 0; i < A_S_consumed; i += blockDim.x){
            if(i + threadIdx.x < A_length - A_loaded && i + threadIdx.x < A_S_consumed){
                A_S[(A_S_start + (tile_size - A_S_consumed) + i + threadIdx.x) % tile_size] = A[A_curr + A_loaded + i + threadIdx.x];
            }
        }
        A_loaded += min(A_S_consumed, A_length - A_loaded);

        for(int i = 0; i < B_S_consumed; i += blockDim.x){
            if(i + threadIdx.x < B_length - B_loaded && i + threadIdx.x < B_S_consumed){
                B_S[(B_S_start + (tile_size - B_S_consumed) + i + threadIdx.x) % tile_size] = B[B_curr + B_loaded + i + threadIdx.x];
            }
        }
        B_loaded += min(B_S_consumed, A_length - B_loaded);

        __syncthreads();

        int c_curr = threadIdx.x * (tile_size / blockDim.x);
        int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);

        c_curr = (c_curr < (C_length - C_completed)) ? c_curr : (C_length - C_completed);
        c_next = (c_next < (C_length - C_completed)) ? c_next : (C_length - C_completed);
        
        int a_curr = co_rank_circular(c_curr, A_S, min(tile_size, A_length - A_consumed),
                                     B_S, min(tile_size, B_length - B_consumed),
                                     A_S_start, B_S_start, tile_size);
        int b_curr = c_curr - a_curr;

        int a_next = co_rank_circular(c_next, A_S, min(tile_size, A_length - A_consumed),
                                     B_S, min(tile_size, B_length - B_consumed),
                                     A_S_start, B_S_start, tile_size);
        int b_next = c_next - a_next;

        merge_sequential_circular(A_S, a_next - a_curr, B_S, b_next - b_curr,
                                  C + C_curr + c_curr + C_completed,
                                  A_S_start + a_curr, B_S_start + b_curr, tile_size);

        counter++;

        A_S_consumed = co_rank_circular(min(tile_size, C_length - C_completed),
                                        A_S, min(tile_size, A_length - A_consumed),
                                        B_S, min(tile_size, B_length - B_consumed),
                                        A_S_start, B_S_start, tile_size);

        B_S_consumed = min(tile_size, C_length - C_completed) - A_S_consumed;

        A_consumed += A_S_consumed;
        C_completed += min(tile_size, C_length - C_completed);
        B_consumed = C_completed - A_consumed;

        A_S_start = (A_S_start + A_S_consumed) % tile_size;
        B_S_start = (B_S_start + B_S_consumed) % tile_size;

        __syncthreads();
    }
}

__global__
void merge_circular_buffer_kernel_bug(int *A, int m, int *B, int n, int *C, int tile_size){
    // for dynamic shared memory allocation, see 
    // https://stackoverflow.com/questions/24942073/dynamic-shared-memory-in-cuda
    extern __shared__ int shareAB[];

    int * A_S = &shareAB[0];
    int * B_S = &shareAB[tile_size];

    // the original codes in the book is not correct because n/d will descard
    // decimal part, see https://stackoverflow.com/questions/26105925/use-of-ceil-and-integers 
    // we can also use ceil(n/d) = (n+d-1)/d but it seems hard to understand.
    // when calculate C_curr, float is converted to int implicitly by compiler
    // but min function don't know call which overloaded function if we pass
    // min(2.0, 3). So we convert 2.0 to 2.
    int C_curr = blockIdx.x * ceil(1.0*(m+n)/gridDim.x);
    int C_next = min((blockIdx.x + 1) * int(ceil(1.0*(m+n)/gridDim.x)), m + n);

    if(threadIdx.x == 0){
        A_S[0] = co_rank(C_curr, A, m, B, n);
        B_S[0] = co_rank(C_next, A, m, B, n);
    }

    __syncthreads();
    int A_curr = A_S[0];
    int A_next = B_S[0];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads(); // we need this because A_S[0] and B_S[0] will be modified.

    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = ceil(1.0 * C_length / tile_size);
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    int A_S_start = 0;
    int B_S_start = 0;
    int A_S_consumed = tile_size;
    int B_S_consumed = tile_size;

    while(counter < total_iteration){
        for(int i = 0; i < A_S_consumed; i += blockDim.x){
            if(i + threadIdx.x < A_length - A_consumed && i + threadIdx.x < A_S_consumed){
                A_S[(A_S_start + (tile_size - A_S_consumed) + i + threadIdx.x) % tile_size] = A[A_curr + A_consumed + i + threadIdx.x];
            }
        }

        for(int i = 0; i < B_S_consumed; i += blockDim.x){
            if(i + threadIdx.x < B_length - B_consumed && i + threadIdx.x < B_S_consumed){
                B_S[(B_S_start + (tile_size - B_S_consumed) + i + threadIdx.x) % tile_size] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }

        __syncthreads();

        int c_curr = threadIdx.x * (tile_size / blockDim.x);
        int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);

        c_curr = (c_curr < (C_length - C_completed)) ? c_curr : (C_length - C_completed);
        c_next = (c_next < (C_length - C_completed)) ? c_next : (C_length - C_completed);
        
        int a_curr = co_rank_circular(c_curr, A_S, min(tile_size, A_length - A_consumed),
                                     B_S, min(tile_size, B_length - B_consumed),
                                     A_S_start, B_S_start, tile_size);
        int b_curr = c_curr - a_curr;

        int a_next = co_rank_circular(c_next, A_S, min(tile_size, A_length - A_consumed),
                                     B_S, min(tile_size, B_length - B_consumed),
                                     A_S_start, B_S_start, tile_size);
        int b_next = c_next - a_next;

        merge_sequential_circular(A_S, a_next - a_curr, B_S, b_next - b_curr,
                                  C + C_curr + c_curr + C_completed,
                                  A_S_start + a_curr, B_S_start + b_curr, tile_size);

        counter++;

        A_S_consumed = co_rank_circular(min(tile_size, C_length - C_completed),
                                        A_S, min(tile_size, A_length - A_consumed),
                                        B_S, min(tile_size, B_length - B_consumed),
                                        A_S_start, B_S_start, tile_size);

        B_S_consumed = min(tile_size, C_length - C_completed) - A_S_consumed;

        A_consumed += A_S_consumed;
        C_completed += min(tile_size, C_length - C_completed);
        B_consumed = C_completed - A_consumed;

        A_S_start = (A_S_start + A_S_consumed) % tile_size;
        B_S_start = (B_S_start + B_S_consumed) % tile_size;

        __syncthreads();
    }
}


void show_bug(){
        int hostA[8];
        int hostB[7];
        int hostC[15];
        int M = sizeof(hostA)/sizeof(hostA[0]);
        int N = sizeof(hostB)/sizeof(hostB[0]);

        generate_rand_int(hostA, M, 1234, 10);
        generate_rand_int(hostB, N, 5678, 10);
        std::sort(std::begin(hostA), std::end(hostA));
        std::sort(std::begin(hostB), std::end(hostB));

        int *deviceA;
        int *deviceB;
        int *deviceC;


        cudaMalloc((void **) &deviceA, M * sizeof(int));
        cudaMalloc((void **) &deviceB, N * sizeof(int));
        cudaMalloc((void **) &deviceC, (M + N) * sizeof(int));

        cudaMemcpy(deviceA, hostA, M * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceB, hostB, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(deviceC, 0, (M + N) * sizeof(int));

        dim3 blockDim(2);
        dim3 gridDim(2);
        int tile_size = 4;
        unsigned int shared_mem_size = 2 * tile_size * sizeof(int);
        merge_tiled_kernel_bug<<<gridDim, blockDim, shared_mem_size>>>(deviceA, M, deviceB, N, deviceC, tile_size);


        cudaMemcpy(hostC, deviceC, (M + N) * sizeof(int), cudaMemcpyDeviceToHost);
        
        cout << "bug" << endl;
        cout << "A: " << endl;
        for(int i = 0; i < M; i++){
            cout << "\t" << hostA[i];
        }
        cout << endl;
        cout << "B: " << endl;
        for(int i = 0; i < N; i++){
            cout << "\t" << hostB[i];
        }
        cout << endl;

        cout << "C: " << endl;
        for(int i = 0; i < M + N; i++){
            cout << "\t" << hostC[i];
        }
        cout << endl;                

        for(int i = 1; i < (M + N); i++){
            if(hostC[i] < hostC[i-1]){
                cout << "merge_tiled_kernel_bug bad: C[" << i << "]=" << hostC[i]
                    << ", C[" << (i-1) << "]=" << hostC[i-1] << endl; 

                break;
            }
        }

        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);    
}

void test(int kernel_id){
        int hostA[MLEN];
        int hostB[NLEN];
        int hostC[MLEN + NLEN];
        generate_rand_int(hostA, MLEN, time(0), MAX);
        generate_rand_int(hostB, NLEN, time(0), MAX);
        std::sort(std::begin(hostA), std::end(hostA));
        std::sort(std::begin(hostB), std::end(hostB));

        int *deviceA;
        int *deviceB;
        int *deviceC;


        cudaMalloc((void **) &deviceA, MLEN * sizeof(int));
        cudaMalloc((void **) &deviceB, NLEN * sizeof(int));
        cudaMalloc((void **) &deviceC, (MLEN + NLEN) * sizeof(int));

        cudaMemcpy(deviceA, hostA, MLEN * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceB, hostB, NLEN * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(deviceC, 0, (MLEN + NLEN) * sizeof(int));

        if(kernel_id == 0){
            dim3 blockDim(BLOCK_SIZE);
            dim3 gridDim(GRID_SIZE);

            merge_basic_kernel<<<gridDim, blockDim>>>(deviceA, MLEN, deviceB, NLEN, deviceC);
        }else if(kernel_id == 1){
            dim3 blockDim(BLOCK_SIZE);
            dim3 gridDim(GRID_SIZE);
            unsigned int shared_mem_size = 2 * TILE_SIZE * sizeof(int);
            merge_tiled_kernel<<<gridDim, blockDim, shared_mem_size>>>(deviceA, MLEN, deviceB, NLEN, deviceC, TILE_SIZE);
        }else if(kernel_id == 2){
            dim3 blockDim(BLOCK_SIZE);
            dim3 gridDim(GRID_SIZE);
            unsigned int shared_mem_size = 2 * TILE_SIZE * sizeof(int);
            merge_tiled_kernel_v2<<<gridDim, blockDim, shared_mem_size>>>(deviceA, MLEN, deviceB, NLEN, deviceC, TILE_SIZE);            
        }else if(kernel_id == 3){
            dim3 blockDim(BLOCK_SIZE);
            dim3 gridDim(GRID_SIZE);
            unsigned int shared_mem_size = 2 * TILE_SIZE * sizeof(int);
            merge_tiled_kernel_v3<<<gridDim, blockDim, shared_mem_size>>>(deviceA, MLEN, deviceB, NLEN, deviceC, TILE_SIZE);            
        }else if(kernel_id == 4){
            dim3 blockDim(BLOCK_SIZE);
            dim3 gridDim(GRID_SIZE);
            unsigned int shared_mem_size = 2 * TILE_SIZE * sizeof(int);
            merge_circular_buffer_kernel<<<gridDim, blockDim, shared_mem_size>>>(deviceA, MLEN, deviceB, NLEN, deviceC, TILE_SIZE);            
        }


        cudaMemcpy(hostC, deviceC, (MLEN + NLEN) * sizeof(int), cudaMemcpyDeviceToHost);
        bool pass = true;
        for(int i = 1; i < (MLEN + NLEN); i++){
            if(hostC[i] < hostC[i-1]){
                cout << "kernel" << kernel_id <<" bad: C[" << i << "]=" << hostC[i]
                    << ", C[" << (i-1) << "]=" << hostC[i-1] << endl; 
                pass = false;
                break;
            }
        }
        if(pass){
            cout << "kernel " << kernel_id << " passed the test." << endl;
        }else{
            cout << "kernel " << kernel_id << " failed the test." << endl;
        }

        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
}


void show_bug2(){
        int hostA[8];
        int hostB[7];
        int hostC[15];
        int M = sizeof(hostA)/sizeof(hostA[0]);
        int N = sizeof(hostB)/sizeof(hostB[0]);

        generate_rand_int(hostA, M, 1234, 10);
        generate_rand_int(hostB, N, 5678, 10);
        std::sort(std::begin(hostA), std::end(hostA));
        std::sort(std::begin(hostB), std::end(hostB));

        int *deviceA;
        int *deviceB;
        int *deviceC;

        
        cout << "bug" << endl;
        cout << "A: " << endl;
        for(int i = 0; i < M; i++){
            cout << "\t" << hostA[i];
        }
        cout << endl;
        cout << "B: " << endl;
        for(int i = 0; i < N; i++){
            cout << "\t" << hostB[i];
        }
        cout << endl;

        cudaMalloc((void **) &deviceA, M * sizeof(int));
        cudaMalloc((void **) &deviceB, N * sizeof(int));
        cudaMalloc((void **) &deviceC, (M + N) * sizeof(int));

        cudaMemcpy(deviceA, hostA, M * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceB, hostB, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(deviceC, 0, (M + N) * sizeof(int));

        dim3 blockDim(2);
        dim3 gridDim(2);
        int tile_size = 4;
        unsigned int shared_mem_size = 2 * tile_size * sizeof(int);
        merge_circular_buffer_kernel_bug<<<gridDim, blockDim, shared_mem_size>>>(deviceA, M, deviceB, N, deviceC, tile_size);


        cudaMemcpy(hostC, deviceC, (M + N) * sizeof(int), cudaMemcpyDeviceToHost);


        cout << "C: " << endl;
        for(int i = 0; i < M + N; i++){
            cout << "\t" << hostC[i];
        }
        cout << endl;                

        for(int i = 1; i < (M + N); i++){
            if(hostC[i] < hostC[i-1]){
                cout << "merge_circular_buffer_kernel bad: C[" << i << "]=" << hostC[i]
                    << ", C[" << (i-1) << "]=" << hostC[i-1] << endl; 

                break;
            }
        }

        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);    
}

int main(int argc, char* argv[]){
    int A[MLEN];
    int B[NLEN];
    int C[MLEN + NLEN];
    generate_rand_int(A, MLEN, time(0), MAX);
    generate_rand_int(B, NLEN, time(0), MAX);
    std::sort(std::begin(A), std::end(A));
    std::sort(std::begin(B), std::end(B));

    merge_sequential_host(A, MLEN, B, NLEN, C);

    for(int i = 1; i < (MLEN + NLEN); i++){
        if(C[i] < C[i-1]){
            cout << "bad: C[" << i << "]=" << C[i]
                 << ", C[" << (i-1) << "]=" << C[i-1] << endl; 
        }
    }

    // illustrate bad co_rank call of bug
    int AA[] = {7, 8, 9, 0};
    int BB[] = {6, 6, 7, 9};
    int r = co_rank_host(4, AA, 4, BB, 4);
    cout << "bad call r[4]=" << r << endl;
    r = co_rank_host(4, AA, 3, BB, 4);
    cout << "good call r[4]=" << r << endl;
    show_bug();
    show_bug2();

    

    for(int j = 0; j < 10; j++){
        test(0);
        test(1);
        test(2);
        test(3);
        test(4);
    }

    

    return 0;

}