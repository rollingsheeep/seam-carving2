#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <string>
#include "data_structures.h"

namespace seam_carving_cuda {

// Functions from dp_kernel.cu
void computeDynamicProgrammingCUDA(const float* energy_data, float* dp_data, int width, int height);
int findMinIndexLastRowCUDA(const float* dp_data, int width, int height);
void backtrackSeamCUDA(const float* dp_data, int* seam_data, int width, int height, int min_idx);

// Functions from energy_kernels.cu
void computeSobelCUDA(const float* lum_data, float* energy_data, int width, int height);
void computeForwardEnergyCUDA(const float* lum_data, float* energy_data, int width, int height);

// Functions from hybrid_energy_kernel.cu
void computeHybridEnergyCUDA(const float* d_luminance, float* d_energy, 
                           const float* d_forward_energy, const float* d_backward_energy,
                           int width, int height, float* h_backward_weight, float* h_forward_weight);
void computeEnergyStatsCUDA(const float* d_backward_energy, const float* d_forward_energy,
                          float* h_min_backward, float* h_max_backward, float* h_avg_backward,
                          float* h_min_forward, float* h_max_forward, float* h_avg_forward,
                          int width, int height);

// Functions from luminance_kernel.cu
void computeLuminanceCUDA(const uint32_t* image_data, float* lum_data, int width, int height);

// Functions from seam_kernel.cu
void cuda_removeSeamKernel(const uint32_t* d_input_image, uint32_t* d_output_image,
                  const int* d_seam, int width, int height);
void removeSeamFromMatrixCUDA(const float* d_input_matrix, float* d_output_matrix,
                           const int* d_seam, int width, int height);
void updateGradientCUDA(float* d_gradient, const float* d_luminance,
                      const int* d_seam, int width, int height);

} // namespace seam_carving_cuda

// Functions from seam_carving_cuda.cpp
// Initialize CUDA and print device information
void initCUDA();
// Cleanup CUDA resources
void cleanupCUDA();
// Returns true if CUDA is available and initialized
bool isCUDAAvailable();
// Copy data between host and device
void copyImageToDevice(const Image& img);
void copyMatrixToDevice(const Matrix& mat, const char* name);
void copyMatrixFromDevice(Matrix& mat, const char* name);