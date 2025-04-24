// seam_carving_cuda.cuh
#pragma once

#include "data_structures.h"
#include <vector>
#include <cstdint>

// Host-callable CUDA wrapper functions (externally linked only from C++)
void computeLuminanceCUDA(Matrix& lum, const Image& img);
void computeSobelFilterCUDA(Matrix& energy, const Matrix& lum);
void computeForwardEnergyCUDA(Matrix& energy, const Matrix& lum);
void computeHybridEnergyCUDA(Matrix& energy, const Matrix& lum);
void computeDynamicProgrammingCUDA(Matrix& dp, const Matrix& energy);
void computeSeamCUDA(std::vector<int>& seam, const Matrix& dp);
void removeSeamCUDA(Image& img, Matrix& lum, Matrix& grad, const std::vector<int>& seam);
void updateGradientCUDA(Matrix& grad, const Matrix& lum, const std::vector<int>& seam);
void initCUDA();
bool isCUDAAvailable();
void cleanupCUDA();
void copyImageToDevice(const Image& img);
void copyMatrixToDevice(const Matrix& mat, const char* name);
void copyMatrixFromDevice(Matrix& mat, const char* name);
