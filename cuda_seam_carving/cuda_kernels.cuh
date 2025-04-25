#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdint.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Device and host data structures
struct DeviceImage {
    uint32_t* pixels;
    int width;
    int height;
    int stride;
};

struct DeviceMatrix {
    float* items;
    int width;
    int height;
    int stride;
};

// Device function for atomic min on float
__device__ void atomicMinFloat(float* addr, float val);

// Kernels for luminance computation
__global__ void computeLuminanceKernel(uint32_t* pixels, float* lum, int width, int height, int stride);

// Kernels for Sobel energy computation
__global__ void computeSobelFilterKernel(float* lum, float* grad, int width, int height, int stride);

// Kernels for forward energy computation
__global__ void computeForwardEnergyKernel(float* lum, float* energy, int width, int height, int stride);

// Kernels for dynamic programming
__global__ void initDynamicProgrammingKernel(float* grad, float* dp, int width);
__global__ void computeDynamicProgrammingKernel(float* grad, float* dp, int width, int height, int row);

// Kernels for seam finding
__global__ void findMinSeamStartKernel(float* dp, int* seam_start, float* min_energy, int width, int height);
__global__ void backtrackSeamKernel(float* dp, int* seam, int width, int height, int row);

// Kernels for seam removal
__global__ void removeSeamKernel(uint32_t* pixels_in, uint32_t* pixels_out, 
                              float* lum_in, float* lum_out,
                              float* grad_in, float* grad_out,
                              int* seam, int width, int height, int row);

// Kernels for updating gradient after seam removal
__global__ void updateGradientKernel(float* lum, float* grad, int* seam, int width, int height);

// Functions to allocate/free device memory and launch kernels
void allocateDeviceMemory(DeviceImage* d_img, DeviceMatrix* d_lum, DeviceMatrix* d_grad, DeviceMatrix* d_dp, int** d_seam, int width, int height);
void freeDeviceMemory(DeviceImage d_img, DeviceMatrix d_lum, DeviceMatrix d_grad, DeviceMatrix d_dp, int* d_seam);
void copyImageToDevice(uint32_t* h_pixels, DeviceImage d_img, int width, int height);
void copyImageFromDevice(uint32_t* h_pixels, DeviceImage d_img, int width, int height);

// Higher-level functions that launch appropriate kernels
void computeLuminanceCuda(DeviceImage d_img, DeviceMatrix d_lum);
void computeSobelFilterCuda(DeviceMatrix d_lum, DeviceMatrix d_grad);
void computeForwardEnergyCuda(DeviceMatrix d_lum, DeviceMatrix d_grad);
void computeHybridEnergyCuda(DeviceMatrix d_lum, DeviceMatrix d_grad, int* h_hybrid_forward_count, int* h_hybrid_backward_count);
void computeDynamicProgrammingCuda(DeviceMatrix d_grad, DeviceMatrix d_dp);
void computeSeamCuda(DeviceMatrix d_dp, int* d_seam, int* h_seam);
void removeSeamCuda(DeviceImage* d_img, DeviceMatrix* d_lum, DeviceMatrix* d_grad, int* d_seam);
void updateGradientCuda(DeviceMatrix d_lum, DeviceMatrix d_grad, int* d_seam); 