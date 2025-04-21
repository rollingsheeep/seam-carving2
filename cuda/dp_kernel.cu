#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <stdio.h>
#include <cfloat>

// Custom atomic minimum function for floats (CUDA doesn't have one built-in)
__device__ void atomicMinFloat(float* addr, float val, int* idx_addr, int idx) {
    float old = *addr;
    float assumed;
    
    do {
        assumed = old;
        old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), 
                         __float_as_int(fminf(val, assumed)));
    } while (old != assumed);
    
    // If we updated the minimum, also update the index
    if (old == assumed && val < assumed) {
        atomicExch(idx_addr, idx);
    }
}

// CUDA kernel for initializing the first row of DP matrix
__global__ void initDPMatrixKernel(const float* energy, float* dp, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (x < width) {
        dp[x] = energy[x];
    }
}

// CUDA kernel for computing one row of the DP matrix
__global__ void computeDPRowKernel(const float* energy, float* dp, int width, int y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (x < width) {
        int curr_idx = y * width + x;
        int prev_idx = (y - 1) * width + x;
        
        float min_prev = dp[prev_idx];  // Up
        
        if (x > 0) {  // Up-left
            min_prev = fminf(min_prev, dp[(y - 1) * width + (x - 1)]);
        }
        
        if (x < width - 1) {  // Up-right
            min_prev = fminf(min_prev, dp[(y - 1) * width + (x + 1)]);
        }
        
        dp[curr_idx] = energy[curr_idx] + min_prev;
    }
}

// Host function to compute the full DP matrix using CUDA
extern "C" void computeDynamicProgrammingCUDA(const float* energy_data, float* dp_data, int width, int height) {
    // Copy first row of energy to dp (initialize)
    dim3 block_init(256);
    dim3 grid_init((width + block_init.x - 1) / block_init.x);
    
    initDPMatrixKernel<<<grid_init, block_init>>>(energy_data, dp_data, width);
    cudaDeviceSynchronize();
    
    // Compute DP matrix row by row
    for (int y = 1; y < height; ++y) {
        computeDPRowKernel<<<grid_init, block_init>>>(energy_data, dp_data, width, y);
        cudaDeviceSynchronize();
    }
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in DP kernel: %s\n", cudaGetErrorString(err));
    }
}

// Finding the minimum seam needs to be done on CPU because of dependencies
// But we can still use CUDA to find the minimum value in the last row
__global__ void findMinLastRowKernel(const float* dp, int width, int height, int* min_idx, float* min_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread handles one element in the last row
    if (idx < width) {
        int last_row_idx = (height - 1) * width + idx;
        float val = dp[last_row_idx];
        
        // Atomic operation to update the minimum value and its index
        atomicMinFloat(min_val, val, min_idx, idx);
    }
}

// Host function to find the minimum index in the last row
extern "C" int findMinIndexLastRowCUDA(const float* dp_data, int width, int height) {
    int* d_min_idx;
    float* d_min_val;
    int h_min_idx;
    float h_min_val = FLT_MAX;
    
    // Allocate device memory
    cudaMalloc(&d_min_idx, sizeof(int));
    cudaMalloc(&d_min_val, sizeof(float));
    
    // Initialize device memory
    cudaMemcpy(d_min_val, &h_min_val, sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 block(256);
    dim3 grid((width + block.x - 1) / block.x);
    findMinLastRowKernel<<<grid, block>>>(dp_data, width, height, d_min_idx, d_min_val);
    
    // Copy result back to host
    cudaMemcpy(&h_min_idx, d_min_idx, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_min_idx);
    cudaFree(d_min_val);
    
    return h_min_idx;
} 