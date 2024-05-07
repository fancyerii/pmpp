#include <cuda.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

#define NUM_BINS 7
#define TFACTOR 32

using namespace std;
 
void histogram_sequential(char * data, unsigned int length, unsigned int * histo){
    for(unsigned int i = 0; i < length; ++i){
        int alphabet_position = data[i] - 'a';
        if(alphabet_position >= 0 && alphabet_position < 26){
            histo[alphabet_position/4]++;
        }
    }

} 

__global__
void histo_kernel(char * data, unsigned int length, unsigned int * histo){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < length){
        int alphabet_position = data[i] - 'a';
        if(alphabet_position >= 0 && alphabet_position < 26){
            atomicAdd(&(histo[alphabet_position/4]), 1);
        }        
    }
}

__global__
void histo_private_kernel(char * data, unsigned int length, unsigned int * histo){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < length){
        int alphabet_position = data[i] - 'a';
        if(alphabet_position >= 0 && alphabet_position < 26){
            atomicAdd(&(histo[blockIdx.x * NUM_BINS + alphabet_position/4]), 1);
        }        
    }

    if(blockIdx.x > 0){
        __syncthreads();
        for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
            unsigned int binValue = histo[blockIdx.x * NUM_BINS + bin];
            if(binValue > 0){
                atomicAdd(&histo[bin], binValue);
            }
        }
    }
}

__global__
void histo_share_kernel(char * data, unsigned int length, unsigned int * histo){
    __shared__ unsigned int histo_s[NUM_BINS];
    for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
        histo_s[bin] = 0u;
    }
    __syncthreads();
    
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < length){
        int alphabet_position = data[i] - 'a';
        if(alphabet_position >= 0 && alphabet_position < 26){
            atomicAdd(&(histo_s[alphabet_position/4]), 1);
        }        
    }
    __syncthreads();

    for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
        unsigned int binValue = histo_s[bin];
        if(binValue > 0){
            atomicAdd(&histo[bin], binValue);
        }        
    }
}

__global__
void histo_share_contiguous_kernel(char * data, unsigned int length, unsigned int * histo){
    __shared__ unsigned int histo_s[NUM_BINS];
    for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
        histo_s[bin] = 0u;
    }
    __syncthreads();
    
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(unsigned i = tid * TFACTOR; i < min(length, (tid + 1)*TFACTOR); i++){
        int alphabet_position = data[i] - 'a';
        if(alphabet_position >= 0 && alphabet_position < 26){
            atomicAdd(&(histo_s[alphabet_position/4]), 1);
        }          
    }
    __syncthreads();

    for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
        unsigned int binValue = histo_s[bin];
        if(binValue > 0){
            atomicAdd(&histo[bin], binValue);
        }        
    }    
}

__global__
void histo_share_interleaved_kernel(char * data, unsigned int length, unsigned int * histo){
    __shared__ unsigned int histo_s[NUM_BINS];
    for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
        histo_s[bin] = 0u;
    }
    __syncthreads();
    
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(unsigned i = tid; i < length; i += blockDim.x * gridDim.x){
        int alphabet_position = data[i] - 'a';
        if(alphabet_position >= 0 && alphabet_position < 26){
            atomicAdd(&(histo_s[alphabet_position/4]), 1);
        }          
    }
    __syncthreads();

    for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
        unsigned int binValue = histo_s[bin];
        if(binValue > 0){
            atomicAdd(&histo[bin], binValue);
        }        
    }    
}

__global__
void histo_share_interleaved_agg_kernel(char * data, unsigned int length, unsigned int * histo){
    __shared__ unsigned int histo_s[NUM_BINS];
    for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
        histo_s[bin] = 0u;
    }
    __syncthreads();
    
    unsigned int accumulator = 0;
    int prevBinIdx = -1;

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(unsigned i = tid; i < length; i += blockDim.x * gridDim.x){
        int alphabet_position = data[i] - 'a';
        if(alphabet_position >= 0 && alphabet_position < 26){
            int bin = alphabet_position/4;
            if(bin == prevBinIdx){
                accumulator++;
            }else{
                if(accumulator > 0){
                    atomicAdd(&histo_s[prevBinIdx], accumulator);
                }
                accumulator = 1;
                prevBinIdx = bin;
            }
        }          
    }
    if(accumulator > 0){
        atomicAdd(&histo_s[prevBinIdx], accumulator);
    }
    __syncthreads();

    for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
        unsigned int binValue = histo_s[bin];
        if(binValue > 0){
            atomicAdd(&histo[bin], binValue);
        }        
    }    
}

void check(const string& s, unsigned int* histo,
        dim3 blockDim, dim3 gridDim,
        string kernel_name, int private_copy,
        void kernel(char * data, unsigned int length, unsigned int * histo)
        ){
    char *deviceInput;
    unsigned int *deviceResult; 

    cudaMalloc((void **) &deviceInput, s.length() * sizeof(char));
    cudaMalloc((void **) &deviceResult, private_copy * NUM_BINS * sizeof(unsigned int));

    cudaMemcpy(deviceInput, &s[0], s.length() * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(deviceResult, 0, private_copy * NUM_BINS * sizeof(unsigned int));

    kernel<<<gridDim, blockDim>>>(deviceInput, s.length(), deviceResult);

    unsigned int hostResult[NUM_BINS] = {0};
    cudaMemcpy(hostResult, deviceResult, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cout << "check " << kernel_name << endl;
    for(int i = 0; i < NUM_BINS; ++i){
        if(histo[i] != hostResult[i]){
            cout << "bad bin " << i << " correct: " << histo[i]
                    << " bad: " << hostResult[i] << endl;
        }
    }
    
    cudaFree(deviceInput);
    cudaFree(deviceResult);
}

int main(int argc, char* argv[]){
    ifstream f("t8.shakespeare.txt");
    stringstream buffer;
    buffer << f.rdbuf();

    string s = buffer.str();
    transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c){ return std::tolower(c); });

    cout << "t8.shakespeare.txt length: " << s.length() << endl;

    unsigned int histo[NUM_BINS] = {0};
    histogram_sequential(&s[0], s.length(), histo);
    for(int i = 0; i < NUM_BINS; ++i){
        cout << "bin " << i << ": " << histo[i] << endl;
    }

    {
        dim3 blockDim(1024);
        dim3 gridDim(ceil((float)s.length() / 1024));
        check(s, histo, blockDim, gridDim, "histo_kernel", 1, histo_kernel);
    }
    
    {
        dim3 blockDim(1024);
        dim3 gridDim(ceil((float)s.length() / 1024));
        check(s, histo, blockDim, gridDim, "histo_private_kernel", gridDim.x, histo_private_kernel);
    }

    {
        dim3 blockDim(1024);
        dim3 gridDim(ceil((float)s.length() / 1024));
        check(s, histo, blockDim, gridDim, "histo_share_kernel", 1, histo_share_kernel);
    }

    {
        dim3 blockDim(1024);
        dim3 gridDim(ceil((float)s.length() / (1024 * TFACTOR)));
        check(s, histo, blockDim, gridDim, "histo_share_contiguous_kernel", 1, histo_share_contiguous_kernel);
    }

    {
        dim3 blockDim(1024);
        dim3 gridDim(ceil((float)s.length() / (1024 * TFACTOR)));
        check(s, histo, blockDim, gridDim, "histo_share_interleaved_kernel", 1, histo_share_interleaved_kernel);
    }

    {
        dim3 blockDim(1024);
        dim3 gridDim(ceil((float)s.length() / (1024 * TFACTOR)));
        check(s, histo, blockDim, gridDim, "histo_share_interleaved_kernel", 1, histo_share_interleaved_agg_kernel);
    }    
    return 0;

}