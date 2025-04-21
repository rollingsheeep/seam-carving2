#pragma once

#include "data_structures.h"
#include <vector>
#include <cstdint>

namespace seam_carving_cuda {

// Initialize CUDA and print device information
void initCUDA();

// Copy data between host and device
void copyImageToDevice(const Image& img);
void copyMatrixToDevice(const Matrix& mat, const char* name);
void copyMatrixFromDevice(Matrix& mat, const char* name);

// CUDA implementations of seam carving operations
void computeLuminanceCUDA(Matrix& lum, const Image& img);
void computeSobelFilterCUDA(Matrix& energy, const Matrix& lum);
void computeForwardEnergyCUDA(Matrix& energy, const Matrix& lum);
void computeHybridEnergyCUDA(Matrix& energy, const Matrix& lum);
void computeDynamicProgrammingCUDA(Matrix& dp, const Matrix& energy);
void computeSeamCUDA(std::vector<int>& seam, const Matrix& dp);

// Cleanup CUDA resources
void cleanupCUDA();

// Returns true if CUDA is available and initialized
bool isCUDAAvailable();

// Visualization helpers
void saveVisualizationsCUDA(const Image& img, const Matrix& lum, const Matrix& energy, const Matrix& dp, const std::vector<int>& seam, int stage);

} // namespace seam_carving_cuda 