#include "seam_carving_cuda.h"
#include "data_structures.h"  // Use our new header instead of main.cpp
#include "cuda_utils.h"
#include "visualization.h"

#include <iostream>
#include <cstring>
#include <cfloat>
#include <cassert>

// Declare external C functions from CUDA files
extern "C" {
    // From luminance_kernel.cu
    void computeLuminanceCUDA(const uint32_t* image_data, float* lum_data, int width, int height);
    
    // From energy_kernels.cu
    void computeSobelCUDA(const float* lum_data, float* energy_data, int width, int height);
    void computeForwardEnergyCUDA(const float* lum_data, float* energy_data, int width, int height);
    
    // From dp_kernel.cu
    void computeDynamicProgrammingCUDA(const float* energy_data, float* dp_data, int width, int height);
    int findMinIndexLastRowCUDA(const float* dp_data, int width, int height);
}

namespace seam_carving_cuda {

// CUDA memory objects
cuda_utils::CudaMemory<uint32_t> d_image;
cuda_utils::CudaMemory<float> d_luminance;
cuda_utils::CudaMemory<float> d_energy;
cuda_utils::CudaMemory<float> d_dp;

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
        // Ensure GPU memory is allocated
        d_image.allocate(img.width * img.height);
        d_luminance.allocate(lum.width * lum.height);
        
        // Copy image to device
        d_image.copyToDevice(img.pixels.data(), img.width * img.height);
        
        // Compute luminance
        ::computeLuminanceCUDA(d_image.get(), d_luminance.get(), img.width, img.height);
        
        // Copy result back to host
        d_luminance.copyToHost(lum.items.data(), lum.width * lum.height);
    } catch (const std::exception& e) {
        std::cerr << "Error in computeLuminanceCUDA: " << e.what() << std::endl;
    }
}

void computeSobelFilterCUDA(Matrix& energy, const Matrix& lum) {
    if (!cuda_initialized) return;
    
    try {
        // Ensure GPU memory is allocated
        d_luminance.allocate(lum.width * lum.height);
        d_energy.allocate(energy.width * energy.height);
        
        // Copy luminance to device
        d_luminance.copyToDevice(lum.items.data(), lum.width * lum.height);
        
        // Compute Sobel filter
        ::computeSobelCUDA(d_luminance.get(), d_energy.get(), lum.width, lum.height);
        
        // Copy result back to host
        d_energy.copyToHost(energy.items.data(), energy.width * energy.height);
    } catch (const std::exception& e) {
        std::cerr << "Error in computeSobelFilterCUDA: " << e.what() << std::endl;
    }
}

void computeForwardEnergyCUDA(Matrix& energy, const Matrix& lum) {
    if (!cuda_initialized) return;
    
    try {
        // Ensure GPU memory is allocated
        d_luminance.allocate(lum.width * lum.height);
        d_energy.allocate(energy.width * energy.height);
        
        // Copy luminance to device
        d_luminance.copyToDevice(lum.items.data(), lum.width * lum.height);
        
        // Compute Forward Energy
        ::computeForwardEnergyCUDA(d_luminance.get(), d_energy.get(), lum.width, lum.height);
        
        // Copy result back to host
        d_energy.copyToHost(energy.items.data(), energy.width * energy.height);
    } catch (const std::exception& e) {
        std::cerr << "Error in computeForwardEnergyCUDA: " << e.what() << std::endl;
    }
}

void computeHybridEnergyCUDA(Matrix& energy, const Matrix& lum) {
    if (!cuda_initialized) return;
    
    try {
        // Allocate a temporary matrix for the second energy type
        Matrix forwardEnergy(lum.width, lum.height);
        Matrix backwardEnergy(lum.width, lum.height);
        
        // Compute both energy types
        computeSobelFilterCUDA(backwardEnergy, lum);
        computeForwardEnergyCUDA(forwardEnergy, lum);
        
        // Ensure GPU memory is allocated for the final energy
        d_energy.allocate(energy.width * energy.height);
        
        // Reset energy ratio trackers for this calculation
        float frame_backward_weight = 0.0f;
        float frame_forward_weight = 0.0f;
        
        // Combine both energy values based on position in the image
        // Logic: Use more forward energy in highly textured regions (high gradient)
        // and more backward energy in smooth regions
        for (int y = 0; y < lum.height; ++y) {
            for (int x = 0; x < lum.width; ++x) {
                // Normalize the backward energy to determine the blending factor
                float gradient = backwardEnergy.at(y, x);
                
                // Calculate the mix factor - more gradient means more forward energy influence
                // Use a much lower threshold to increase forward energy influence
                float mixFactor = std::min(1.0f, gradient / 5.0f);
                
                // Track the weights used
                float backwardWeight = 1.0f - mixFactor;
                float forwardWeight = mixFactor;
                
                frame_backward_weight += backwardWeight;
                frame_forward_weight += forwardWeight;
                
                // Blend the two energy types
                energy.at(y, x) = backwardWeight * backwardEnergy.at(y, x) + 
                                  forwardWeight * forwardEnergy.at(y, x);
            }
        }
        
        // Update global counters
        total_backward_weight += frame_backward_weight;
        total_forward_weight += frame_forward_weight;
        
        // Calculate the ratio for this frame
        float total_pixels = lum.width * lum.height;
        float frame_backward_ratio = frame_backward_weight / total_pixels;
        float frame_forward_ratio = frame_forward_weight / total_pixels;
        
        // Output the energy ratio for this frame
        // std::cout << "  Hybrid energy ratio - Backward: " 
        //           << std::fixed << std::setprecision(2) << frame_backward_ratio * 100.0f << "%, Forward: " 
        //           << frame_forward_ratio * 100.0f << "%" << std::endl;
        
        // Copy the final result to GPU
        d_energy.copyToDevice(energy.items.data(), energy.width * energy.height);
    }
    catch (const std::exception& e) {
        std::cerr << "Error in computeHybridEnergyCUDA: " << e.what() << std::endl;
    }
}

void computeDynamicProgrammingCUDA(Matrix& dp, const Matrix& energy) {
    if (!cuda_initialized) return;
    
    try {
        // Ensure GPU memory is allocated
        d_energy.allocate(energy.width * energy.height);
        d_dp.allocate(dp.width * dp.height);
        
        // Copy energy to device
        d_energy.copyToDevice(energy.items.data(), energy.width * energy.height);
        
        // Compute dynamic programming
        ::computeDynamicProgrammingCUDA(d_energy.get(), d_dp.get(), energy.width, energy.height);
        
        // Copy result back to host
        d_dp.copyToHost(dp.items.data(), dp.width * dp.height);
    } catch (const std::exception& e) {
        std::cerr << "Error in computeDynamicProgrammingCUDA: " << e.what() << std::endl;
    }
}

void computeSeamCUDA(std::vector<int>& seam, const Matrix& dp) {
    if (!cuda_initialized) return;
    
    try {
        int height = dp.height;
        int width = dp.width;
        
        seam.resize(height);
        
        // Find the minimum value in the last row (can be done on GPU)
        int min_x = ::findMinIndexLastRowCUDA(d_dp.get(), width, height);
        
        // The rest of the seam computation is done on CPU
        seam[height - 1] = min_x;
        
        // Copy DP matrix to host for backtracking
        std::vector<float> host_dp(width * height);
        d_dp.copyToHost(host_dp.data(), width * height);
        
        // Backtrack to find the seam (this part is hard to parallelize efficiently)
        for (int y = height - 2; y >= 0; --y) {
            int x = seam[y + 1];
            seam[y] = x;  // Default: go straight up
            
            float up = host_dp[y * width + x];
            float up_left = x > 0 ? host_dp[y * width + (x - 1)] : FLT_MAX;
            float up_right = x < width - 1 ? host_dp[y * width + (x + 1)] : FLT_MAX;
            
            if (x > 0 && up_left < up && up_left <= up_right) {
                seam[y] = x - 1;  // Go up-left
            } else if (x < width - 1 && up_right < up && up_right <= up_left) {
                seam[y] = x + 1;  // Go up-right
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in computeSeamCUDA: " << e.what() << std::endl;
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
        
        d_image.free();
        d_luminance.free();
        d_energy.free();
        d_dp.free();
    } catch (const std::exception& e) {
        std::cerr << "Error cleaning up CUDA resources: " << e.what() << std::endl;
    }
    
    cuda_initialized = false;
}

void saveVisualizationsCUDA(const Image& img, const Matrix& lum, const Matrix& energy, const Matrix& dp, 
                           const std::vector<int>& seam, int stage) {
    visualization::saveStageVisualizations(img, lum, energy, dp, seam, stage);
}

} // namespace seam_carving_cuda 