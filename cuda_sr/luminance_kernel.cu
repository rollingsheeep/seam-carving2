#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "seam_carving_cuda.cuh"

// GPT solution
// use a __device__ function to encapsulate the logic for converting RGB to luminance
// __device__ float rgb_to_luminance(uint32_t pixel) {
//     float r = (pixel >> 0) & 0xFF;
//     float g = (pixel >> 8) & 0xFF;
//     float b = (pixel >> 16) & 0xFF;
//     return 0.2126f * r + 0.7152f * g + 0.0722f * b;
// }

// __global__ void kernel_compute_luminance(const uint32_t* input_pixels, float* output_luminance, int width, int height) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (x < width && y < height) {
//         int idx = y * width + x;
//         output_luminance[idx] = rgb_to_luminance(input_pixels[idx]) / 255.0f;  // normalize to [0, 1]
//     }
// }

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

namespace seam_carving_cuda {

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

} // namespace seam_carving_cuda 