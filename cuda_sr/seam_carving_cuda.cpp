#include "seam_carving_cuda.h"
#include "seam_carving_cuda.cuh"
#include "data_structures.h"  // Use our new header instead of main.cpp
#include "cuda_utils.h"
#include <iostream>
#include <cassert>
#include <iomanip>
#include <cstring>
#include <cfloat>

namespace seam_carving_cuda {

// CUDA memory objects
cuda_utils::CudaMemory<uint32_t> d_image;
cuda_utils::CudaMemory<float> d_luminance;
cuda_utils::CudaMemory<float> d_energy;
cuda_utils::CudaMemory<float> d_dp;
cuda_utils::CudaMemory<uint32_t> d_output_image;
cuda_utils::CudaMemory<int> d_seam;
cuda_utils::CudaMemory<float> d_forward_energy;
cuda_utils::CudaMemory<float> d_backward_energy;
cuda_utils::CudaMemory<float> d_output_lum;
cuda_utils::CudaMemory<float> d_output_energy;

bool cuda_initialized = false;

// Add at the global scope within the seam_carving_cuda namespace
float total_backward_weight = 0.0f;
float total_forward_weight = 0.0f;

void initCUDA() {
    try {
        cuda_utils::printDeviceInfo();
        cuda_initialized = true;
    } catch (const std::exception& e) {
        std::cerr << "CUDA initialization failed: " << e.what() << std::endl;
        cuda_initialized = false;
    }
}

bool isCUDAAvailable() {
    return cuda_initialized;
}

void copyImageToDevice(const Image& img) {
    if (!cuda_initialized) return;
    
    try {
        // Copy image data to device
        d_image.copyToDevice(img.pixels.data(), img.width * img.height);
    } catch (const std::exception& e) {
        std::cerr << "Error copying image to device: " << e.what() << std::endl;
    }
}

void copyMatrixToDevice(const Matrix& mat, const char* name) {
    if (!cuda_initialized) return;
    
    try {
        if (strcmp(name, "lum") == 0) {
            d_luminance.copyToDevice(mat.items.data(), mat.width * mat.height);
        } else if (strcmp(name, "energy") == 0) {
            d_energy.copyToDevice(mat.items.data(), mat.width * mat.height);
        } else if (strcmp(name, "dp") == 0) {
            d_dp.copyToDevice(mat.items.data(), mat.width * mat.height);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error copying matrix " << name << " to device: " << e.what() << std::endl;
    }
}

void copyMatrixFromDevice(Matrix& mat, const char* name) {
    if (!cuda_initialized) return;
    
    try {
        if (strcmp(name, "lum") == 0) {
            d_luminance.copyToHost(mat.items.data(), mat.width * mat.height);
        } else if (strcmp(name, "energy") == 0) {
            d_energy.copyToHost(mat.items.data(), mat.width * mat.height);
        } else if (strcmp(name, "dp") == 0) {
            d_dp.copyToHost(mat.items.data(), mat.width * mat.height);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error copying matrix " << name << " from device: " << e.what() << std::endl;
    }
}

void computeLuminanceCUDA(Matrix& lum, const Image& img) {
    if (!cuda_initialized) return;
    
    try {
        int width = img.width;
        int height = img.height;
        size_t image_size = width * height * sizeof(uint32_t);
        size_t lum_size = width * height * sizeof(float);
        
        // Allocate device memory directly
        uint32_t* d_img = nullptr;
        float* d_lum = nullptr;
        CUDA_CHECK(cudaMalloc(&d_img, image_size));
        CUDA_CHECK(cudaMalloc(&d_lum, lum_size));
        
        // Copy image to device using cudaMemcpy directly
        CUDA_CHECK(cudaMemcpy(d_img, img.pixels.data(), image_size, cudaMemcpyHostToDevice));
        
        // Compute luminance
        seam_carving_cuda_kernels::computeLuminanceCUDA(d_img, d_lum, width, height);
        
        // Copy result back to host using cudaMemcpy directly
        CUDA_CHECK(cudaMemcpy(lum.items.data(), d_lum, lum_size, cudaMemcpyDeviceToHost));
        
        // Free device memory
        CUDA_CHECK(cudaFree(d_img));
        CUDA_CHECK(cudaFree(d_lum));
    } catch (const std::exception& e) {
        std::cerr << "Error in computeLuminanceCUDA: " << e.what() << std::endl;
    }
}

void computeSobelFilterCUDA(Matrix& energy, const Matrix& lum) {
    if (!cuda_initialized) return;
    
    try {
        int width = lum.width;
        int height = lum.height;
        size_t matrix_size = width * height * sizeof(float);
        
        // Allocate device memory directly
        float* d_lum = nullptr;
        float* d_energy = nullptr;
        CUDA_CHECK(cudaMalloc(&d_lum, matrix_size));
        CUDA_CHECK(cudaMalloc(&d_energy, matrix_size));
        
        // Copy luminance to device using cudaMemcpy directly
        CUDA_CHECK(cudaMemcpy(d_lum, lum.items.data(), matrix_size, cudaMemcpyHostToDevice));
        
        // Compute Sobel filter
        seam_carving_cuda_kernels::computeSobelCUDA(d_lum, d_energy, width, height);
        
        // Copy result back to host using cudaMemcpy directly
        CUDA_CHECK(cudaMemcpy(energy.items.data(), d_energy, matrix_size, cudaMemcpyDeviceToHost));
        
        // Free device memory
        CUDA_CHECK(cudaFree(d_lum));
        CUDA_CHECK(cudaFree(d_energy));
    } catch (const std::exception& e) {
        std::cerr << "Error in computeSobelFilterCUDA: " << e.what() << std::endl;
    }
}

void computeForwardEnergyCUDA(Matrix& energy, const Matrix& lum) {
    if (!cuda_initialized) return;
    
    try {
        int width = lum.width;
        int height = lum.height;
        size_t matrix_size = width * height * sizeof(float);
        
        // Allocate device memory directly
        float* d_lum = nullptr;
        float* d_energy = nullptr;
        CUDA_CHECK(cudaMalloc(&d_lum, matrix_size));
        CUDA_CHECK(cudaMalloc(&d_energy, matrix_size));
        
        // Copy luminance to device using cudaMemcpy directly
        CUDA_CHECK(cudaMemcpy(d_lum, lum.items.data(), matrix_size, cudaMemcpyHostToDevice));
        
        // Compute Forward Energy
        seam_carving_cuda_kernels::computeForwardEnergyCUDA(d_lum, d_energy, width, height);
        
        // Copy result back to host using cudaMemcpy directly
        CUDA_CHECK(cudaMemcpy(energy.items.data(), d_energy, matrix_size, cudaMemcpyDeviceToHost));
        
        // Free device memory
        CUDA_CHECK(cudaFree(d_lum));
        CUDA_CHECK(cudaFree(d_energy));
    } catch (const std::exception& e) {
        std::cerr << "Error in computeForwardEnergyCUDA: " << e.what() << std::endl;
    }
}

void computeHybridEnergyCUDA(Matrix& energy, const Matrix& lum) {
    if (!cuda_initialized) return;
    
    try {
        int width = lum.width;
        int height = lum.height;
        
        // Allocate temporary matrices for forward and backward energy
        Matrix forwardEnergy(width, height);
        Matrix backwardEnergy(width, height);
        
        // Compute both energy types
        computeSobelFilterCUDA(backwardEnergy, lum);
        computeForwardEnergyCUDA(forwardEnergy, lum);
        
        // Allocate device memory for both energy types
        d_forward_energy.allocate(width * height);
        d_backward_energy.allocate(width * height);
        d_energy.allocate(width * height);
        
        // Copy both energy matrices to device
        d_forward_energy.copyToDevice(forwardEnergy.items.data(), width * height);
        d_backward_energy.copyToDevice(backwardEnergy.items.data(), width * height);
        
        // Variables to track energy weights
        float backward_weight = 0.0f;
        float forward_weight = 0.0f;
        
        // Compute hybrid energy directly on GPU
        seam_carving_cuda_kernels::computeHybridEnergyCUDA(d_luminance.get(), d_energy.get(), 
                                                           d_forward_energy.get(), d_backward_energy.get(),
                                                           width, height, &backward_weight, &forward_weight);
        
        // Copy result back to host
        d_energy.copyToHost(energy.items.data(), width * height);
        
        // Update global counters
        total_backward_weight += backward_weight;
        total_forward_weight += forward_weight;
        
        // Calculate the ratio for this frame for debugging
        float total_pixels = width * height;
        float frame_backward_ratio = backward_weight / total_pixels;
        float frame_forward_ratio = forward_weight / total_pixels;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in computeHybridEnergyCUDA: " << e.what() << std::endl;
        
        // Fall back to CPU implementation using the new normalized hybrid energy
        float backward_weight = 0.0f;
        float forward_weight = 0.0f;
        compute_hybrid_energy(lum, energy, &backward_weight, &forward_weight);
        
        // Update global counters
        total_backward_weight += backward_weight;
        total_forward_weight += forward_weight;
    }
}

void computeDynamicProgrammingCUDA(Matrix& dp, const Matrix& energy) {
    if (!cuda_initialized) return;
    
    try {
        // Use the CPU implementation instead of CUDA
        compute_dynamic_programming(energy, dp);
    } catch (const std::exception& e) {
        std::cerr << "Error in computeDynamicProgrammingCUDA: " << e.what() << std::endl;
    }
}

void computeSeamCUDA(std::vector<int>& seam, const Matrix& dp) {
    if (!cuda_initialized) return;
    
    try {
        compute_seam(dp, seam);
    } catch (const std::exception& e) {
        std::cerr << "Error in computeSeamCUDA: " << e.what() << std::endl;
    }
}

// Optimized CUDA implementation of seam removal
void removeSeamCUDA(Image& img, Matrix& lum, Matrix& grad, const std::vector<int>& seam) {
    if (!cuda_initialized) return;
    
    try {
        int width = img.width;
        int height = img.height;
        int new_width = width - 1;
        
        size_t image_size = width * height * sizeof(uint32_t);
        size_t new_image_size = new_width * height * sizeof(uint32_t);
        size_t matrix_size = width * height * sizeof(float);
        size_t new_matrix_size = new_width * height * sizeof(float);
        size_t seam_size = height * sizeof(int);
        
        // Allocate device memory directly
        uint32_t* d_input_image = nullptr;
        uint32_t* d_output_image = nullptr;
        float* d_input_lum = nullptr;
        float* d_output_lum = nullptr;
        float* d_input_grad = nullptr;
        float* d_output_grad = nullptr;
        int* d_seam = nullptr;
        
        CUDA_CHECK(cudaMalloc(&d_input_image, image_size));
        CUDA_CHECK(cudaMalloc(&d_output_image, new_image_size));
        CUDA_CHECK(cudaMalloc(&d_input_lum, matrix_size));
        CUDA_CHECK(cudaMalloc(&d_output_lum, new_matrix_size));
        CUDA_CHECK(cudaMalloc(&d_input_grad, matrix_size));
        CUDA_CHECK(cudaMalloc(&d_output_grad, new_matrix_size));
        CUDA_CHECK(cudaMalloc(&d_seam, seam_size));
        
        // Copy data to device using cudaMemcpy directly
        CUDA_CHECK(cudaMemcpy(d_input_image, img.pixels.data(), image_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_input_lum, lum.items.data(), matrix_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_input_grad, grad.items.data(), matrix_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_seam, seam.data(), seam_size, cudaMemcpyHostToDevice));
        
        // Remove seam from image
        seam_carving_cuda_kernels::cuda_removeSeamKernel(d_input_image, d_output_image, d_seam, width, height);
        
        // Remove seam from luminance matrix
        seam_carving_cuda_kernels::removeSeamFromMatrixCUDA(d_input_lum, d_output_lum, d_seam, width, height);
        
        // Remove seam from gradient matrix
        seam_carving_cuda_kernels::removeSeamFromMatrixCUDA(d_input_grad, d_output_grad, d_seam, width, height);
        
        // Copy results back to host using cudaMemcpy directly
        CUDA_CHECK(cudaMemcpy(img.pixels.data(), d_output_image, new_image_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lum.items.data(), d_output_lum, new_matrix_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(grad.items.data(), d_output_grad, new_matrix_size, cudaMemcpyDeviceToHost));
        
        // Update dimensions
        img.width = new_width;
        lum.width = new_width;
        grad.width = new_width;
        
        // Free device memory
        CUDA_CHECK(cudaFree(d_input_image));
        CUDA_CHECK(cudaFree(d_output_image));
        CUDA_CHECK(cudaFree(d_input_lum));
        CUDA_CHECK(cudaFree(d_output_lum));
        CUDA_CHECK(cudaFree(d_input_grad));
        CUDA_CHECK(cudaFree(d_output_grad));
        CUDA_CHECK(cudaFree(d_seam));
    } catch (const std::exception& e) {
        std::cerr << "Error in removeSeamCUDA: " << e.what() << std::endl;
    }
}

// Optimized CUDA implementation of gradient update after seam removal
void updateGradientCUDA(Matrix& grad, const Matrix& lum, const std::vector<int>& seam) {
    if (!cuda_initialized) return;
    
    try {
        int width = grad.width;
        int height = grad.height;
        size_t matrix_size = width * height * sizeof(float);
        size_t seam_size = height * sizeof(int);
        
        // Allocate device memory directly
        float* d_grad = nullptr;
        float* d_lum = nullptr;
        int* d_seam = nullptr;
        
        CUDA_CHECK(cudaMalloc(&d_grad, matrix_size));
        CUDA_CHECK(cudaMalloc(&d_lum, matrix_size));
        CUDA_CHECK(cudaMalloc(&d_seam, seam_size));
        
        // Copy data to device using cudaMemcpy directly
        CUDA_CHECK(cudaMemcpy(d_grad, grad.items.data(), matrix_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lum, lum.items.data(), matrix_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_seam, seam.data(), seam_size, cudaMemcpyHostToDevice));
        
        // Update gradient
        seam_carving_cuda_kernels::updateGradientCUDA(d_grad, d_lum, d_seam, width, height);
        
        // Copy result back to host using cudaMemcpy directly
        CUDA_CHECK(cudaMemcpy(grad.items.data(), d_grad, matrix_size, cudaMemcpyDeviceToHost));
        
        // Free device memory
        CUDA_CHECK(cudaFree(d_grad));
        CUDA_CHECK(cudaFree(d_lum));
        CUDA_CHECK(cudaFree(d_seam));
    } catch (const std::exception& e) {
        std::cerr << "Error in updateGradientCUDA: " << e.what() << std::endl;
    }
}

void cleanupCUDA() {
    if (!cuda_initialized) return;
    
    try {
        // Report final energy statistics if hybrid energy was used
        if (total_backward_weight > 0.0f || total_forward_weight > 0.0f) {
            float total_weight = total_backward_weight + total_forward_weight;
            float backward_ratio = total_backward_weight / total_weight;
            float forward_ratio = total_forward_weight / total_weight;
            
            std::cout << "\nHybrid energy summary:" << std::endl;
            std::cout << "  Backward energy usage: " << std::fixed << std::setprecision(2) 
                      << backward_ratio * 100.0f << "%" << std::endl;
            std::cout << "  Forward energy usage: " << forward_ratio * 100.0f << "%" << std::endl;
            std::cout << "  Ratio (Backward:Forward): " << backward_ratio / forward_ratio << ":1" << std::endl;
        }
        
        // No need to free memory here as we're now freeing it after each operation
        cuda_initialized = false;
    } catch (const std::exception& e) {
        std::cerr << "Error cleaning up CUDA resources: " << e.what() << std::endl;
    }
}

} // namespace seam_carving_cuda 