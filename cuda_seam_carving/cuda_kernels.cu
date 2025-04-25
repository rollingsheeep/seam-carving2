#include "cuda_kernels.cuh"
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

// Block size for CUDA kernels
#define BLOCK_SIZE 16

// Convert RGB to luminance using the standard coefficients
__device__ float rgb_to_lum(uint32_t rgb) {
    float r = ((rgb >> (8*0)) & 0xFF) / 255.0f;
    float g = ((rgb >> (8*1)) & 0xFF) / 255.0f;
    float b = ((rgb >> (8*2)) & 0xFF) / 255.0f;
    return 0.2126f*r + 0.7152f*g + 0.0722f*b;
}

// Kernel to compute luminance from RGB pixels
__global__ void computeLuminanceKernel(uint32_t* pixels, float* lum, int width, int height, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * stride + x;
        lum[idx] = rgb_to_lum(pixels[idx]);
    }
}

// Helper function to get pixel value safely with bounds checking
__device__ float getLuminanceAt(float* lum, int x, int y, int width, int height, int stride) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        return lum[y * stride + x];
    }
    return 0.0f;
}

// Applies Sobel edge detection filter at a specific pixel
__device__ float sobelFilterAt(float* lum, int cx, int cy, int width, int height, int stride) {
    static const float gx[3][3] = {
        {1.0f, 0.0f, -1.0f},
        {2.0f, 0.0f, -2.0f},
        {1.0f, 0.0f, -1.0f},
    };

    static const float gy[3][3] = {
        {1.0f, 2.0f, 1.0f},
        {0.0f, 0.0f, 0.0f},
        {-1.0f, -2.0f, -1.0f},
    };

    float sx = 0.0f;
    float sy = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int x = cx + dx;
            int y = cy + dy;
            float c = getLuminanceAt(lum, x, y, width, height, stride);
            sx += c * gx[dy + 1][dx + 1];
            sy += c * gy[dy + 1][dx + 1];
        }
    }
    return sqrtf(sx*sx + sy*sy);
}

// Kernel to compute Sobel filter for energy gradient
__global__ void computeSobelFilterKernel(float* lum, float* grad, int width, int height, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * stride + x;
        grad[idx] = sobelFilterAt(lum, x, y, width, height, stride);
    }
}

// Kernel for forward energy calculation (first row initialization)
__global__ void initForwardEnergyKernel(float* energy, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (x < width) {
        energy[x] = 0.0f;
    }
}

// Kernel for forward energy calculation (per row)
__global__ void computeForwardEnergyKernel(float* lum, float* energy, int width, int height, int stride, int row) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (x < width && row < height) {
        float cU = 0.0f, cL = 0.0f, cR = 0.0f;

        // Compute neighbor costs safely with bounds
        float left   = (x > 0)     ? lum[row * stride + (x - 1)] : lum[row * stride + x];
        float right  = (x < width - 1) ? lum[row * stride + (x + 1)] : lum[row * stride + x];
        float up     = lum[(row - 1) * stride + x];
        
        // Cost for going straight up
        cU = fabsf(right - left);

        // Cost for going up-left
        cL = cU + fabsf(up - left);

        // Cost for going up-right
        cR = cU + fabsf(up - right);

        // Get minimum previous path cost
        float min_energy = energy[(row - 1) * stride + x] + cU;
        if (x > 0)     min_energy = fminf(min_energy, energy[(row - 1) * stride + (x - 1)] + cL);
        if (x < width - 1) min_energy = fminf(min_energy, energy[(row - 1) * stride + (x + 1)] + cR);

        energy[row * stride + x] = min_energy;
    }
}

// Kernel for dynamic programming initialization (first row)
__global__ void initDynamicProgrammingKernel(float* grad, float* dp, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (x < width) {
        dp[x] = grad[x];
    }
}

// Kernel for dynamic programming (per row)
__global__ void computeDynamicProgrammingKernel(float* grad, float* dp, int width, int height, int row) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (x < width && row < height) {
        float min_prev = dp[(row - 1) * width + x];
        if (x > 0) {
            min_prev = fminf(min_prev, dp[(row - 1) * width + (x - 1)]);
        }
        if (x < width - 1) {
            min_prev = fminf(min_prev, dp[(row - 1) * width + (x + 1)]);
        }
        dp[row * width + x] = grad[row * width + x] + min_prev;
    }
}

// Kernel to find the starting point of the minimum energy seam
__global__ void findMinSeamStartKernel(float* dp, int* seam_start, float* min_energy, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float local_min[BLOCK_SIZE];
    __shared__ int local_idx[BLOCK_SIZE];
    
    local_min[threadIdx.x] = FLT_MAX;
    local_idx[threadIdx.x] = -1;
    
    if (x < width) {
        int y = height - 1;
        float energy = dp[y * width + x];
        local_min[threadIdx.x] = energy;
        local_idx[threadIdx.x] = x;
    }
    
    __syncthreads();
    
    // Reduction to find minimum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (local_min[threadIdx.x] > local_min[threadIdx.x + s]) {
                local_min[threadIdx.x] = local_min[threadIdx.x + s];
                local_idx[threadIdx.x] = local_idx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (threadIdx.x == 0) {
        atomicMinFloat(min_energy, local_min[0]);
        if (local_min[0] == *min_energy) {
            *seam_start = local_idx[0];
        }
    }
}

// Custom implementation of atomicMin for float
__device__ void atomicMinFloat(float* addr, float val) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        int new_val = __float_as_int(fminf(__int_as_float(expected), val));
        old = atomicCAS(addr_as_int, expected, new_val);
    } while (expected != old);
}

// Kernel to backtrack and find the seam
__global__ void backtrackSeamKernel(float* dp, int* seam, int width, int height, int row) {
    // Only one thread does the backtracking for each row
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int y = row;
        int x = seam[y + 1]; // Current position based on the next row

        float min_energy = dp[y * width + x];
        int min_idx = x;
        
        if (x > 0 && dp[y * width + (x - 1)] < min_energy) {
            min_energy = dp[y * width + (x - 1)];
            min_idx = x - 1;
        }
        if (x < width - 1 && dp[y * width + (x + 1)] < min_energy) {
            min_energy = dp[y * width + (x + 1)];
            min_idx = x + 1;
        }
        
        seam[y] = min_idx;
    }
}

// Kernel to remove a seam from the image (per row)
__global__ void removeSeamKernel(uint32_t* pixels_in, uint32_t* pixels_out, 
                               float* lum_in, float* lum_out,
                               float* grad_in, float* grad_out,
                               int* seam, int width, int height, int row) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < height && x < width - 1) {
        int seam_x = seam[row];
        
        if (x >= seam_x) {
            // Skip the seam pixel
            pixels_out[row * (width - 1) + x] = pixels_in[row * width + (x + 1)];
            lum_out[row * (width - 1) + x] = lum_in[row * width + (x + 1)];
            grad_out[row * (width - 1) + x] = grad_in[row * width + (x + 1)];
        } else {
            // Copy pixels before the seam
            pixels_out[row * (width - 1) + x] = pixels_in[row * width + x];
            lum_out[row * (width - 1) + x] = lum_in[row * width + x];
            grad_out[row * (width - 1) + x] = grad_in[row * width + x];
        }
    }
}

// Kernel to update gradient values for pixels adjacent to the removed seam
__global__ void updateGradientKernel(float* lum, float* grad, int* seam, int width, int height) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int dx = blockIdx.x * blockDim.x + threadIdx.x - 1; // -1, 0, 1
    
    if (y < height && dx >= -1 && dx <= 1) {
        int x = seam[y] + dx;
        
        if (x >= 0 && x < width) {
            grad[y * width + x] = sobelFilterAt(lum, x, y, width, height, width);
        }
    }
}

// Function to allocate device memory
void allocateDeviceMemory(DeviceImage* d_img, DeviceMatrix* d_lum, DeviceMatrix* d_grad, DeviceMatrix* d_dp, int** d_seam, int width, int height) {
    // Allocate device memory for image pixels
    CUDA_CHECK(cudaMalloc(&d_img->pixels, width * height * sizeof(uint32_t)));
    d_img->width = width;
    d_img->height = height;
    d_img->stride = width;
    
    // Allocate device memory for luminance matrix
    CUDA_CHECK(cudaMalloc(&d_lum->items, width * height * sizeof(float)));
    d_lum->width = width;
    d_lum->height = height;
    d_lum->stride = width;
    
    // Allocate device memory for gradient matrix
    CUDA_CHECK(cudaMalloc(&d_grad->items, width * height * sizeof(float)));
    d_grad->width = width;
    d_grad->height = height;
    d_grad->stride = width;
    
    // Allocate device memory for dynamic programming matrix
    CUDA_CHECK(cudaMalloc(&d_dp->items, width * height * sizeof(float)));
    d_dp->width = width;
    d_dp->height = height;
    d_dp->stride = width;
    
    // Allocate device memory for seam
    CUDA_CHECK(cudaMalloc(d_seam, height * sizeof(int)));
}

// Function to free device memory
void freeDeviceMemory(DeviceImage d_img, DeviceMatrix d_lum, DeviceMatrix d_grad, DeviceMatrix d_dp, int* d_seam) {
    CUDA_CHECK(cudaFree(d_img.pixels));
    CUDA_CHECK(cudaFree(d_lum.items));
    CUDA_CHECK(cudaFree(d_grad.items));
    CUDA_CHECK(cudaFree(d_dp.items));
    CUDA_CHECK(cudaFree(d_seam));
}

// Function to copy image data to device
void copyImageToDevice(uint32_t* h_pixels, DeviceImage d_img, int width, int height) {
    CUDA_CHECK(cudaMemcpy(d_img.pixels, h_pixels, width * height * sizeof(uint32_t), cudaMemcpyHostToDevice));
}

// Function to copy image data from device
void copyImageFromDevice(uint32_t* h_pixels, DeviceImage d_img, int width, int height) {
    CUDA_CHECK(cudaMemcpy(h_pixels, d_img.pixels, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

// Higher-level function to compute luminance
void computeLuminanceCuda(DeviceImage d_img, DeviceMatrix d_lum) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((d_img.width + blockSize.x - 1) / blockSize.x, (d_img.height + blockSize.y - 1) / blockSize.y);
    
    computeLuminanceKernel<<<gridSize, blockSize>>>(d_img.pixels, d_lum.items, d_lum.width, d_lum.height, d_lum.stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Higher-level function to compute Sobel filter
void computeSobelFilterCuda(DeviceMatrix d_lum, DeviceMatrix d_grad) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((d_lum.width + blockSize.x - 1) / blockSize.x, (d_lum.height + blockSize.y - 1) / blockSize.y);
    
    computeSobelFilterKernel<<<gridSize, blockSize>>>(d_lum.items, d_grad.items, d_grad.width, d_grad.height, d_grad.stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Higher-level function to compute forward energy
void computeForwardEnergyCuda(DeviceMatrix d_lum, DeviceMatrix d_grad) {
    dim3 blockSize(BLOCK_SIZE, 1);
    dim3 gridSize((d_lum.width + blockSize.x - 1) / blockSize.x, 1);
    
    // Initialize first row
    initForwardEnergyKernel<<<gridSize, blockSize>>>(d_grad.items, d_grad.width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Process each row
    for (int row = 1; row < d_lum.height; ++row) {
        computeForwardEnergyKernel<<<gridSize, blockSize>>>(d_lum.items, d_grad.items, d_grad.width, d_grad.height, d_grad.stride, row);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Higher-level function to compute hybrid energy (switching between forward and backward)
void computeHybridEnergyCuda(DeviceMatrix d_lum, DeviceMatrix d_grad, int* h_hybrid_forward_count, int* h_hybrid_backward_count) {
    // Implement a simple strategy: use backward energy (Sobel) for now
    // This could be enhanced later with the full hybrid approach
    computeSobelFilterCuda(d_lum, d_grad);
    (*h_hybrid_backward_count)++;
    
    // Note: A more sophisticated implementation would analyze the image
    // characteristics on CPU or GPU to decide which method to use
}

// Higher-level function to compute dynamic programming
void computeDynamicProgrammingCuda(DeviceMatrix d_grad, DeviceMatrix d_dp) {
    dim3 blockSize(BLOCK_SIZE, 1);
    dim3 gridSize((d_grad.width + blockSize.x - 1) / blockSize.x, 1);
    
    // Initialize first row
    initDynamicProgrammingKernel<<<gridSize, blockSize>>>(d_grad.items, d_dp.items, d_dp.width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Process each row
    for (int row = 1; row < d_grad.height; ++row) {
        computeDynamicProgrammingKernel<<<gridSize, blockSize>>>(d_grad.items, d_dp.items, d_dp.width, d_dp.height, row);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Higher-level function to compute seam
void computeSeamCuda(DeviceMatrix d_dp, int* d_seam, int* h_seam) {
    int width = d_dp.width;
    int height = d_dp.height;
    
    // Allocate device memory for temporary variables
    int* d_seam_start;
    float* d_min_energy;
    CUDA_CHECK(cudaMalloc(&d_seam_start, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_min_energy, sizeof(float)));
    
    // Initialize min_energy to a large value
    float init_min = FLT_MAX;
    CUDA_CHECK(cudaMemcpy(d_min_energy, &init_min, sizeof(float), cudaMemcpyHostToDevice));
    
    // Find the starting point of the minimum energy seam
    dim3 blockSize(BLOCK_SIZE, 1);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 1);
    findMinSeamStartKernel<<<gridSize, blockSize>>>(d_dp.items, d_seam_start, d_min_energy, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy the seam start to host
    int seam_start;
    CUDA_CHECK(cudaMemcpy(&seam_start, d_seam_start, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Copy seam start to the last row in the seam array
    CUDA_CHECK(cudaMemcpy(d_seam + (height - 1), &seam_start, sizeof(int), cudaMemcpyHostToDevice));
    
    // Backtrack to find the seam
    for (int row = height - 2; row >= 0; --row) {
        backtrackSeamKernel<<<1, 1>>>(d_dp.items, d_seam, width, height, row);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Copy the seam back to host
    CUDA_CHECK(cudaMemcpy(h_seam, d_seam, height * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Clean up
    CUDA_CHECK(cudaFree(d_seam_start));
    CUDA_CHECK(cudaFree(d_min_energy));
}

// Higher-level function to remove seam
void removeSeamCuda(DeviceImage* d_img, DeviceMatrix* d_lum, DeviceMatrix* d_grad, int* d_seam) {
    int width = d_img->width;
    int height = d_img->height;
    
    // Allocate temporary device memory for the reduced image and matrices
    uint32_t* d_pixels_out;
    float* d_lum_out;
    float* d_grad_out;
    CUDA_CHECK(cudaMalloc(&d_pixels_out, (width - 1) * height * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_lum_out, (width - 1) * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_out, (width - 1) * height * sizeof(float)));
    
    // Launch a kernel for each row to remove the seam
    dim3 blockSize(BLOCK_SIZE, 1);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 1);
    
    for (int row = 0; row < height; ++row) {
        removeSeamKernel<<<gridSize, blockSize>>>(
            d_img->pixels, d_pixels_out,
            d_lum->items, d_lum_out,
            d_grad->items, d_grad_out,
            d_seam, width, height, row
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Free old memory and update pointers
    CUDA_CHECK(cudaFree(d_img->pixels));
    CUDA_CHECK(cudaFree(d_lum->items));
    CUDA_CHECK(cudaFree(d_grad->items));
    
    d_img->pixels = d_pixels_out;
    d_lum->items = d_lum_out;
    d_grad->items = d_grad_out;
    
    // Update dimensions
    d_img->width = width - 1;
    d_lum->width = width - 1;
    d_grad->width = width - 1;
    d_img->stride = width - 1;
    d_lum->stride = width - 1;
    d_grad->stride = width - 1;
}

// Higher-level function to update gradient after seam removal
void updateGradientCuda(DeviceMatrix d_lum, DeviceMatrix d_grad, int* d_seam) {
    dim3 blockSize(3, BLOCK_SIZE); // 3 columns (x-1, x, x+1)
    dim3 gridSize(1, (d_lum.height + blockSize.y - 1) / blockSize.y);
    
    updateGradientKernel<<<gridSize, blockSize>>>(d_lum.items, d_grad.items, d_seam, d_grad.width, d_grad.height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
} 