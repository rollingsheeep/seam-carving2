#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "seam_carving_cuda.cuh"

// Robust CUDA kernel to remove seam from image
__global__ void removeSeamKernel(const uint32_t* input_image, uint32_t* output_image,
                               const int* seam, int width, int height, int new_width) {
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread is within image bounds
    if (y >= 0 && y < height && x >= 0 && x < new_width) {
        // Get the seam position for this row with bounds checking
        int seam_x = seam[y];
        
        // Validate seam position (defensive programming)
        if (seam_x < 0) seam_x = 0;
        if (seam_x >= width) seam_x = width - 1;
        
        // Determine source pixel position
        int src_x = x;
        if (x >= seam_x) {
            src_x = x + 1;  // Skip the seam pixel
        }
        
        // Ensure source position is valid
        if (src_x < width) {
            // Copy pixel from input to output, skipping the seam
            output_image[y * new_width + x] = input_image[y * width + src_x];
        }
    }
}

// Robust CUDA kernel to remove seam from matrix
__global__ void removeSeamFromMatrixKernel(const float* input_matrix, float* output_matrix,
                                         const int* seam, int width, int height, int new_width) {
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread is within matrix bounds
    if (y >= 0 && y < height && x >= 0 && x < new_width) {
        // Get the seam position for this row with bounds checking
        int seam_x = seam[y];
        
        // Validate seam position (defensive programming)
        if (seam_x < 0) seam_x = 0;
        if (seam_x >= width) seam_x = width - 1;
        
        // Determine source position
        int src_x = x;
        if (x >= seam_x) {
            src_x = x + 1;  // Skip the seam pixel
        }
        
        // Ensure source position is valid
        if (src_x < width) {
            // Copy value from input to output, skipping the seam
            output_matrix[y * new_width + x] = input_matrix[y * width + src_x];
        }
    }
}

// Improved CUDA kernel to update gradient after seam removal with safer shared memory
__global__ void updateGradientKernel(float* gradient, const float* luminance,
                                   const int* seam, int width, int height) {
    // Get row index from block index
    const int y = blockIdx.y;
    const int tid = threadIdx.x;
    
    // Exit if out of bounds
    if (y >= height) return;
    
    // Get seam position for this row
    const int seam_x = seam[y];
    
    // Define the area around the seam to update (3 pixels wide)
    const int min_update_x = max(0, seam_x - 1);
    const int max_update_x = min(width - 1, seam_x + 1);
    
    // Define shared memory for 3x3 neighborhood around the update area
    extern __shared__ float s_lum[];
    
    // We need to load a patch of luminance values that includes:
    // - The update area (min_update_x to max_update_x)
    // - Plus a 1-pixel border on all sides for Sobel filter
    const int load_min_x = max(0, min_update_x - 1);
    const int load_max_x = min(width - 1, max_update_x + 1);
    const int load_width = load_max_x - load_min_x + 1;
    
    // Define regions in shared memory for y-1, y, and y+1 rows
    float* s_lum_prev = &s_lum[0];                 // y-1 row
    float* s_lum_curr = &s_lum[load_width];        // y row
    float* s_lum_next = &s_lum[2 * load_width];    // y+1 row
    
    // Load data into shared memory
    // Each thread loads up to 3 pixels (one from each row)
    for (int x_offset = tid; x_offset < load_width; x_offset += blockDim.x) {
        int x = load_min_x + x_offset;
        
        // Load from y-1 row (with boundary check)
        if (y > 0) {
            s_lum_prev[x_offset] = luminance[(y-1) * width + x];
        } else {
            s_lum_prev[x_offset] = 0.0f;
        }
        
        // Load from y row (current row)
        s_lum_curr[x_offset] = luminance[y * width + x];
        
        // Load from y+1 row (with boundary check)
        if (y < height - 1) {
            s_lum_next[x_offset] = luminance[(y+1) * width + x];
        } else {
            s_lum_next[x_offset] = 0.0f;
        }
    }
    __syncthreads();
    
    // Each thread computes gradient for one pixel in the update area
    for (int x_offset = tid; x_offset <= (max_update_x - min_update_x); x_offset += blockDim.x) {
        int x = min_update_x + x_offset;
        
        // Sobel filter coefficients
        const float gx[3][3] = {
            {1.0f, 0.0f, -1.0f},
            {2.0f, 0.0f, -2.0f},
            {1.0f, 0.0f, -1.0f}
        };
        
        const float gy[3][3] = {
            {1.0f, 2.0f, 1.0f},
            {0.0f, 0.0f, 0.0f},
            {-1.0f, -2.0f, -1.0f}
        };
        
        float sx = 0.0f;
        float sy = 0.0f;
        
        // Apply Sobel operator using shared memory
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx;
                
                if (nx >= 0 && nx < width) {
                    // Calculate position in shared memory
                    int s_x_offset = (nx - load_min_x);
                    
                    // Get value from the appropriate row
                    float pixel;
                    if (dy == -1) {
                        pixel = s_lum_prev[s_x_offset];
                    } else if (dy == 0) {
                        pixel = s_lum_curr[s_x_offset];
                    } else { // dy == 1
                        pixel = s_lum_next[s_x_offset];
                    }
                    
                    sx += pixel * gx[dy+1][dx+1];
                    sy += pixel * gy[dy+1][dx+1];
                }
            }
        }
        
        // Update gradient
        gradient[y * width + x] = sqrtf(sx*sx + sy*sy);
    }
}

namespace seam_carving_cuda {

// Host function to remove seam from image using CUDA
void cuda_removeSeamKernel(const uint32_t* d_input_image, uint32_t* d_output_image,
                  const int* d_seam, int width, int height) {
    int new_width = width - 1;
    
    // Validate input parameters
    if (width <= 1) {
        printf("Error: Invalid width (%d) for seam removal\n", width);
        return;
    }
    
    // Define block and grid dimensions for 2D parallelism
    dim3 block(16, 16);
    dim3 grid((new_width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // Launch CUDA kernel
    removeSeamKernel<<<grid, block>>>(d_input_image, d_output_image, d_seam, width, height, new_width);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in removeSeam kernel: %s\n", cudaGetErrorString(err));
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
}

// Host function to remove seam from matrix using CUDA
void removeSeamFromMatrixCUDA(const float* d_input_matrix, float* d_output_matrix,
                           const int* d_seam, int width, int height) {
    int new_width = width - 1;
    
    // Validate input parameters
    if (width <= 1) {
        printf("Error: Invalid width (%d) for seam removal\n", width);
        return;
    }
    
    // Define block and grid dimensions for 2D parallelism
    dim3 block(16, 16);
    dim3 grid((new_width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // Launch CUDA kernel
    removeSeamFromMatrixKernel<<<grid, block>>>(d_input_matrix, d_output_matrix, d_seam, width, height, new_width);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in removeSeamFromMatrix kernel: %s\n", cudaGetErrorString(err));
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
}

// Host function to update gradient after seam removal using CUDA
void updateGradientCUDA(float* d_gradient, const float* d_luminance,
                      const int* d_seam, int width, int height) {
    // For efficiency, we launch one thread block per row
    dim3 block(32);  // 32 threads per block
    dim3 grid(1, height);  // One block per row
    
    // Calculate required shared memory size (3 rows of luminance data)
    // We need at most (width+2) elements per row to account for boundary conditions
    int shared_mem_size = 3 * (width + 2) * sizeof(float);
    
    // Check for maximum shared memory limit
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (shared_mem_size > deviceProp.sharedMemPerBlock) {
        printf("Warning: Required shared memory (%d bytes) exceeds device limit (%d bytes)\n",
               shared_mem_size, deviceProp.sharedMemPerBlock);
        
        // Fall back to smaller shared memory size for very large images
        shared_mem_size = deviceProp.sharedMemPerBlock;
    }
    
    // Launch CUDA kernel
    updateGradientKernel<<<grid, block, shared_mem_size>>>(d_gradient, d_luminance, d_seam, width, height);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in updateGradient kernel: %s\n", cudaGetErrorString(err));
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
}

} // namespace seam_carving_cuda
