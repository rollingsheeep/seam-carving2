#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cfloat>  // For FLT_MAX

// Helper device function for atomic min operation on floats (not available natively in CUDA)
__device__ void atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int expected;
    do {
        expected = old;
        int new_val = __float_as_int(min(__int_as_float(expected), val));
        old = atomicCAS(address_as_int, expected, new_val);
    } while (expected != old);
}

// Helper device function for atomic max operation on floats (not available natively in CUDA)
__device__ void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int expected;
    do {
        expected = old;
        int new_val = __float_as_int(max(__int_as_float(expected), val));
        old = atomicCAS(address_as_int, expected, new_val);
    } while (expected != old);
}

// CUDA kernel to calculate statistics for hybrid energy
__global__ void energyStatsKernel(const float* backward_energy, const float* forward_energy,
                                 float* min_backward, float* max_backward, float* avg_backward,
                                 float* min_forward, float* max_forward, float* avg_forward,
                                 int width, int height) {
    extern __shared__ float shared_data[];
    
    // Shared memory arrangement:
    // [min_backward, max_backward, sum_backward, min_forward, max_forward, sum_forward]
    float* s_min_backward = &shared_data[0];
    float* s_max_backward = &shared_data[1];
    float* s_sum_backward = &shared_data[2];
    float* s_min_forward = &shared_data[3];
    float* s_max_forward = &shared_data[4];
    float* s_sum_forward = &shared_data[5];
    
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int block_size = blockDim.x * blockDim.y;
    
    // Initialize shared memory
    if (tid == 0) {
        s_min_backward[0] = FLT_MAX;
        s_max_backward[0] = -FLT_MAX;
        s_sum_backward[0] = 0.0f;
        s_min_forward[0] = FLT_MAX;
        s_max_forward[0] = -FLT_MAX;
        s_sum_forward[0] = 0.0f;
    }
    __syncthreads();
    
    // Each thread processes its pixels
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        float b_energy = backward_energy[idx];
        float f_energy = forward_energy[idx];
        
        // Compute local statistics
        atomicMinFloat(&s_min_backward[0], b_energy);
        atomicMaxFloat(&s_max_backward[0], b_energy);
        atomicAdd(&s_sum_backward[0], b_energy);
        
        atomicMinFloat(&s_min_forward[0], f_energy);
        atomicMaxFloat(&s_max_forward[0], f_energy);
        atomicAdd(&s_sum_forward[0], f_energy);
    }
    __syncthreads();
    
    // Only the first thread in the block updates global memory
    if (tid == 0) {
        atomicMinFloat(min_backward, s_min_backward[0]);
        atomicMaxFloat(max_backward, s_max_backward[0]);
        atomicAdd(avg_backward, s_sum_backward[0]);
        
        atomicMinFloat(min_forward, s_min_forward[0]);
        atomicMaxFloat(max_forward, s_max_forward[0]);
        atomicAdd(avg_forward, s_sum_forward[0]);
    }
}

// CUDA kernel to normalize energy values
__global__ void normalizeEnergiesKernel(float* forward_energy, float* backward_energy,
                                       float min_forward, float max_forward,
                                       float min_backward, float max_backward,
                                       int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // Normalize energies to [0, 1] range
        float forward_range = max_forward - min_forward;
        float backward_range = max_backward - min_backward;
        
        // Avoid division by zero
        if (forward_range > 0.0001f) {
            forward_energy[idx] = (forward_energy[idx] - min_forward) / forward_range;
        }
        
        if (backward_range > 0.0001f) {
            backward_energy[idx] = (backward_energy[idx] - min_backward) / backward_range;
        }
    }
}

// CUDA kernel to compute hybrid energy with normalized values
__global__ void hybridEnergyKernel(const float* luminance, float* hybrid_energy, 
                                  const float* normalized_forward_energy, 
                                  const float* normalized_backward_energy,
                                  int width, int height, 
                                  float* backward_weight, float* forward_weight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // Get the normalized energy values
        float backward_val = normalized_backward_energy[idx];
        float forward_val = normalized_forward_energy[idx];
        
        // Calculate adaptive mix factor based on normalized gradient values
        // Use a threshold to balance between forward and backward energy
        // High gradient (texture/detail) = more forward energy influence
        // Low gradient (smooth areas) = more backward energy influence
        float gradient_threshold = 0.3f;  // Adjustable parameter
        float mixFactor = min(1.0f, backward_val / gradient_threshold);
        
        // Calculate weights
        float backwardWeight = 1.0f - mixFactor;
        float forwardWeight = mixFactor;
        
        // Blend the two energy types using the normalized values
        hybrid_energy[idx] = backwardWeight * backward_val + forwardWeight * forward_val;
        
        // Accumulate weights atomically
        atomicAdd(backward_weight, backwardWeight);
        atomicAdd(forward_weight, forwardWeight);
    }
}

extern "C" {
    // Host function to compute hybrid energy directly on GPU
    void computeHybridEnergyCUDA(const float* d_luminance, float* d_energy, 
                               const float* d_forward_energy, const float* d_backward_energy,
                               int width, int height, float* h_backward_weight, float* h_forward_weight) {
        // Allocate device memory for weights and statistics
        float* d_backward_weight;
        float* d_forward_weight;
        cudaMalloc(&d_backward_weight, sizeof(float));
        cudaMalloc(&d_forward_weight, sizeof(float));
        
        // Initialize weights to zero
        cudaMemset(d_backward_weight, 0, sizeof(float));
        cudaMemset(d_forward_weight, 0, sizeof(float));
        
        // Allocate memory for energy statistics
        float* d_min_backward;
        float* d_max_backward;
        float* d_avg_backward;
        float* d_min_forward;
        float* d_max_forward;
        float* d_avg_forward;
        
        cudaMalloc(&d_min_backward, sizeof(float));
        cudaMalloc(&d_max_backward, sizeof(float));
        cudaMalloc(&d_avg_backward, sizeof(float));
        cudaMalloc(&d_min_forward, sizeof(float));
        cudaMalloc(&d_max_forward, sizeof(float));
        cudaMalloc(&d_avg_forward, sizeof(float));
        
        // Initialize statistics
        float init_min = FLT_MAX;
        float init_max = -FLT_MAX;
        float init_avg = 0.0f;
        
        cudaMemcpy(d_min_backward, &init_min, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_max_backward, &init_max, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_avg_backward, &init_avg, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_min_forward, &init_min, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_max_forward, &init_max, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_avg_forward, &init_avg, sizeof(float), cudaMemcpyHostToDevice);
        
        // Define block and grid dimensions
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        
        // Calculate shared memory size for stats kernel
        int shared_mem_size = 6 * sizeof(float); // 6 values: min, max, sum for both energy types
        
        // Step 1: Calculate statistics for both energy types
        energyStatsKernel<<<grid, block, shared_mem_size>>>(
            d_backward_energy, d_forward_energy,
            d_min_backward, d_max_backward, d_avg_backward,
            d_min_forward, d_max_forward, d_avg_forward,
            width, height
        );
        
        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in energy stats kernel: %s\n", cudaGetErrorString(err));
        }
        
        // Synchronize to ensure statistics computation is complete
        cudaDeviceSynchronize();
        
        // Copy statistics from device to host
        float h_min_backward, h_max_backward;
        float h_min_forward, h_max_forward;
        
        cudaMemcpy(&h_min_backward, d_min_backward, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_max_backward, d_max_backward, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_min_forward, d_min_forward, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_max_forward, d_max_forward, sizeof(float), cudaMemcpyDeviceToHost);
        
        // Allocate memory for normalized energies
        float* d_norm_forward_energy;
        float* d_norm_backward_energy;
        cudaMalloc(&d_norm_forward_energy, width * height * sizeof(float));
        cudaMalloc(&d_norm_backward_energy, width * height * sizeof(float));
        
        // Copy the original energies to the normalized buffers
        cudaMemcpy(d_norm_forward_energy, d_forward_energy, width * height * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_norm_backward_energy, d_backward_energy, width * height * sizeof(float), cudaMemcpyDeviceToDevice);
        
        // Step 2: Normalize both energy types
        normalizeEnergiesKernel<<<grid, block>>>(
            d_norm_forward_energy, d_norm_backward_energy,
            h_min_forward, h_max_forward,
            h_min_backward, h_max_backward,
            width, height
        );
        
        // Check for errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in normalize energies kernel: %s\n", cudaGetErrorString(err));
        }
        
        // Synchronize to ensure normalization is complete
        cudaDeviceSynchronize();
        
        // Step 3: Compute hybrid energy using normalized values
        hybridEnergyKernel<<<grid, block>>>(
            d_luminance, d_energy, 
            d_norm_forward_energy, d_norm_backward_energy,
            width, height, 
            d_backward_weight, d_forward_weight
        );
        
        // Check for errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in hybrid energy kernel: %s\n", cudaGetErrorString(err));
        }
        
        // Copy weights back to host
        cudaMemcpy(h_backward_weight, d_backward_weight, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_forward_weight, d_forward_weight, sizeof(float), cudaMemcpyDeviceToHost);
        
        // Free device memory
        cudaFree(d_backward_weight);
        cudaFree(d_forward_weight);
        cudaFree(d_min_backward);
        cudaFree(d_max_backward);
        cudaFree(d_avg_backward);
        cudaFree(d_min_forward);
        cudaFree(d_max_forward);
        cudaFree(d_avg_forward);
        cudaFree(d_norm_forward_energy);
        cudaFree(d_norm_backward_energy);
        
        // Synchronize
        cudaDeviceSynchronize();
    }
    
    // Host function to compute energy statistics on GPU
    void computeEnergyStatsCUDA(const float* d_backward_energy, const float* d_forward_energy,
                              float* h_min_backward, float* h_max_backward, float* h_avg_backward,
                              float* h_min_forward, float* h_max_forward, float* h_avg_forward,
                              int width, int height) {
        // Allocate device memory for statistics
        float* d_min_backward;
        float* d_max_backward;
        float* d_avg_backward;
        float* d_min_forward;
        float* d_max_forward;
        float* d_avg_forward;
        
        cudaMalloc(&d_min_backward, sizeof(float));
        cudaMalloc(&d_max_backward, sizeof(float));
        cudaMalloc(&d_avg_backward, sizeof(float));
        cudaMalloc(&d_min_forward, sizeof(float));
        cudaMalloc(&d_max_forward, sizeof(float));
        cudaMalloc(&d_avg_forward, sizeof(float));
        
        // Initialize device memory
        float init_min = FLT_MAX;
        float init_max = -FLT_MAX;
        float init_avg = 0.0f;
        
        cudaMemcpy(d_min_backward, &init_min, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_max_backward, &init_max, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_avg_backward, &init_avg, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_min_forward, &init_min, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_max_forward, &init_max, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_avg_forward, &init_avg, sizeof(float), cudaMemcpyHostToDevice);
        
        // Define block and grid dimensions
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        
        // Calculate shared memory size
        int shared_mem_size = 6 * sizeof(float); // 6 values: min, max, sum for both energy types
        
        // Launch CUDA kernel
        energyStatsKernel<<<grid, block, shared_mem_size>>>(d_backward_energy, d_forward_energy,
                                                         d_min_backward, d_max_backward, d_avg_backward,
                                                         d_min_forward, d_max_forward, d_avg_forward,
                                                         width, height);
        
        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in energy stats kernel: %s\n", cudaGetErrorString(err));
        }
        
        // Copy results back to host
        cudaMemcpy(h_min_backward, d_min_backward, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_max_backward, d_max_backward, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_avg_backward, d_avg_backward, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_min_forward, d_min_forward, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_max_forward, d_max_forward, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_avg_forward, d_avg_forward, sizeof(float), cudaMemcpyDeviceToHost);
        
        // Normalize averages
        *h_avg_backward /= (width * height);
        *h_avg_forward /= (width * height);
        
        // Free device memory
        cudaFree(d_min_backward);
        cudaFree(d_max_backward);
        cudaFree(d_avg_backward);
        cudaFree(d_min_forward);
        cudaFree(d_max_forward);
        cudaFree(d_avg_forward);
        
        // Synchronize
        cudaDeviceSynchronize();
    }
} 