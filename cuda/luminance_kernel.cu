#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// CUDA kernel to convert RGB to luminance
__global__ void rgbToLuminanceKernel(const uint32_t* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        uint32_t rgb = input[idx];
        
        // Extract RGB components
        float r = ((rgb >> (8*0)) & 0xFF) / 255.0f;
        float g = ((rgb >> (8*1)) & 0xFF) / 255.0f;
        float b = ((rgb >> (8*2)) & 0xFF) / 255.0f;
        
        // Convert to luminance using standard coefficients
        output[idx] = 0.2126f*r + 0.7152f*g + 0.0722f*b;
    }
}

extern "C" {
    // Host function to compute luminance using CUDA
    void computeLuminanceCUDA(const uint32_t* image_data, float* lum_data, int width, int height) {
        // Define block and grid dimensions
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        
        // Launch CUDA kernel
        rgbToLuminanceKernel<<<grid, block>>>(image_data, lum_data, width, height);
        
        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in luminance kernel: %s\n", cudaGetErrorString(err));
        }
        
        // Synchronize
        cudaDeviceSynchronize();
    }
} 