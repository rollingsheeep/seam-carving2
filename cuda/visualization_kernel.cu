#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// CUDA kernel to copy image and highlight seam in red
__global__ void visualizeSeamKernel(const uint32_t* input, uint32_t* output, const int* seam, 
                                   int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // First copy the original image
        output[idx] = input[idx];
        
        // If this pixel is on the seam, make it red
        if (x == seam[y]) {
            output[idx] = 0xFFFF0000; // Bright red (RGBA format)
        }
    }
}

extern "C" {
    // Host function to visualize seam using CUDA
    void visualizeSeamCUDA(const uint32_t* image_data, uint32_t* output_data, 
                          const int* seam_data, int width, int height) {
        // Define block and grid dimensions
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        
        // Launch CUDA kernel
        visualizeSeamKernel<<<grid, block>>>(image_data, output_data, seam_data, width, height);
        
        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in visualization kernel: %s\n", cudaGetErrorString(err));
        }
        
        // Synchronize
        cudaDeviceSynchronize();
    }
} 