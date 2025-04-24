#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <stdio.h>
#include <cfloat>
#include "seam_carving_cuda.cuh"


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

// CUDA kernel for computing one row of the DP matrix at a time
__global__ void computeDPRowKernel(const float* energy, float* dp, int width, int height, int y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (x < width && y > 0) {
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
    else if (x < width && y == 0) {
        // Initialize first row
        dp[x] = energy[x];
    }
}

namespace seam_carving_cuda_kernels {

// Host function to compute the full DP matrix using CUDA - safer row-by-row approach
void computeDynamicProgrammingCUDA(const float* energy_data, float* dp_data, int width, int height) {
    // Define block and grid dimensions
    dim3 block(256);
    dim3 grid((width + block.x - 1) / block.x);
    
    // Process each row sequentially
    for (int y = 0; y < height; ++y) {
        computeDPRowKernel<<<grid, block>>>(energy_data, dp_data, width, height, y);
        
        // Synchronize after each row to ensure correctness
        cudaDeviceSynchronize();
        
        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in DP row kernel at row %d: %s\n", y, cudaGetErrorString(err));
            break;
        }
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
int findMinIndexLastRowCUDA(const float* dp_data, int width, int height) {
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

// Improved CUDA kernel for parallel seam backtracking
__global__ void backtrackSeamKernel(const float* dp, int* seam, int width, int height, int min_idx_last_row) {
    // Define shared memory with explicit variables
    extern __shared__ float s_data[];
    
    // Shared memory layout:
    // [seam indices (height elements), dp values (3 elements)]
    int* s_seam = (int*)s_data;
    float* s_dp = (float*)&s_seam[height];
    
    // Initialize the last element of the seam
    if (threadIdx.x == 0) {
        s_seam[height - 1] = min_idx_last_row;
    }
    __syncthreads();
    
    // Process rows in reverse, starting from the second-to-last row
    for (int y = height - 2; y >= 0; --y) {
        // Get the position in the previous row
        int x_prev = s_seam[y + 1];
        
        // Compute the three possible DP values (up-left, up, up-right)
        if (threadIdx.x < 3) {
            int x = x_prev - 1 + threadIdx.x; // -1, 0, +1
            
            // Handle boundary conditions
            if (x >= 0 && x < width) {
                s_dp[threadIdx.x] = dp[y * width + x];
            } else {
                s_dp[threadIdx.x] = FLT_MAX; // Mark out-of-bounds as invalid
            }
        }
        __syncthreads();
        
        // Thread 0 makes the decision
        if (threadIdx.x == 0) {
            // Default: go straight up
            float min_val = s_dp[1]; 
            int best_idx = x_prev;
            
            // Check up-left
            if (x_prev > 0 && s_dp[0] < min_val) {
                min_val = s_dp[0];
                best_idx = x_prev - 1;
            }
            
            // Check up-right
            if (x_prev < width - 1 && s_dp[2] < min_val) {
                min_val = s_dp[2];
                best_idx = x_prev + 1;
            }
            
            s_seam[y] = best_idx;
        }
        __syncthreads();
    }
    
    // Copy shared memory results to global memory in a coalesced manner
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    for (int i = tid; i < height; i += stride) {
        // Perform a simple bounds check
        if (i < height) {
            int path_x = s_seam[i];
            
            // Additional validation - clamp to valid range
            if (path_x < 0) path_x = 0;
            if (path_x >= width) path_x = width - 1;
            
            seam[i] = path_x;
        }
    }
}

// Host function to perform parallel seam backtracking
void backtrackSeamCUDA(const float* dp_data, int* seam_data, int width, int height, int min_idx) {
    // Define block dimensions - we only need one block
    dim3 block(256);
    dim3 grid(1);
    
    // Calculate shared memory size
    // We need space for:
    // 1. seam indices (height integers)
    // 2. temporary dp values (3 floats for comparing up-left, up, up-right)
    int shared_mem_size = height * sizeof(int) + 3 * sizeof(float);
    
    // Verify that min_idx is valid (defensive programming)
    if (min_idx < 0 || min_idx >= width) {
        printf("Warning: Invalid min_idx (%d) in backtrackSeamCUDA, clamping to valid range\n", min_idx);
        min_idx = min(max(0, min_idx), width - 1);
    }
    
    // Launch kernel
    backtrackSeamKernel<<<grid, block, shared_mem_size>>>(dp_data, seam_data, width, height, min_idx);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in backtrack seam kernel: %s\n", cudaGetErrorString(err));
    }
    
    // Synchronize
    cudaDeviceSynchronize();
}

} // namespace seam_carving_cuda_kernels