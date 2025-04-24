#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdio.h>
#include "seam_carving_cuda.cuh"

// CUDA kernel for computing Sobel filter (backward energy)
__global__ void sobelFilterKernel(const float* luminance, float* energy, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // Sobel filter coefficients
        // Gx
        const float gx[3][3] = {
            {1.0f, 0.0f, -1.0f},
            {2.0f, 0.0f, -2.0f},
            {1.0f, 0.0f, -1.0f}
        };
        
        // Gy
        const float gy[3][3] = {
            {1.0f, 2.0f, 1.0f},
            {0.0f, 0.0f, 0.0f},
            {-1.0f, -2.0f, -1.0f}
        };
        
        float sx = 0.0f;
        float sy = 0.0f;
        
        // Apply Sobel operator
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                
                // Check boundaries
                float pixel = 0.0f;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    pixel = luminance[ny * width + nx];
                }
                
                sx += pixel * gx[dy+1][dx+1];
                sy += pixel * gy[dy+1][dx+1];
            }
        }
        
        // Magnitude of gradient
        energy[idx] = sqrtf(sx*sx + sy*sy);
    }
}

// CUDA kernel for computing forward energy
__global__ void forwardEnergyKernel(const float* luminance, float* energy, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // For the first row, we set energy to 0
        if (y == 0) {
            energy[idx] = 0.0f;
            return;
        }
        
        // For other rows, we calculate forward energy
        // Get neighboring pixel values safely
        float left = (x > 0) ? luminance[y * width + (x - 1)] : luminance[y * width + x];
        float right = (x < width - 1) ? luminance[y * width + (x + 1)] : luminance[y * width + x];
        float up = luminance[(y - 1) * width + x];
        
        // Calculate costs
        float cU = fabsf(right - left);
        float cL = cU + fabsf(up - left);
        float cR = cU + fabsf(up - right);
        
        // Get minimum previous energy (we need to do this in a separate kernel for DP)
        // We're just calculating the local costs here
        energy[idx] = cU;  // This is just the local cost, not the full DP value
    }
}

namespace seam_carving_cuda_kernels {

// Host function to compute Sobel filter (backward energy) using CUDA
void computeSobelCUDA(const float* lum_data, float* energy_data, int width, int height) {
    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // Launch CUDA kernel
    sobelFilterKernel<<<grid, block>>>(lum_data, energy_data, width, height);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in Sobel kernel: %s\n", cudaGetErrorString(err));
    }
    
    // Synchronize
    cudaDeviceSynchronize();
}

// Host function to compute forward energy using CUDA
void computeForwardEnergyCUDA(const float* lum_data, float* energy_data, int width, int height) {
    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // Launch CUDA kernel
    forwardEnergyKernel<<<grid, block>>>(lum_data, energy_data, width, height);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in forward energy kernel: %s\n", cudaGetErrorString(err));
    }
    
    // Synchronize
    cudaDeviceSynchronize();
}

} // namespace seam_carving_cuda_kernels