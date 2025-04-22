#include "seam_carving_cuda.h"
#include "data_structures.h"  // Use our new header instead of main.cpp
#include "cuda_utils.h"
#include "visualization.h"

#include <iostream>
#include <cstring>
#include <cfloat>
#include <cassert>
#include <iomanip>

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
    void backtrackSeamCUDA(const float* dp_data, int* seam_data, int width, int height, int min_idx);
    
    // From visualization_kernel.cu
    void visualizeSeamCUDA(const uint32_t* image_data, uint32_t* output_data, const int* seam_data, int width, int height);
    
    // From seam_kernel.cu
    void cuda_removeSeamKernel(const uint32_t* d_input_image, uint32_t* d_output_image, 
                      const int* d_seam, int width, int height);
    void removeSeamFromMatrixCUDA(const float* d_input_matrix, float* d_output_matrix, 
                               const int* d_seam, int width, int height);
    void updateGradientCUDA(float* d_gradient, const float* d_luminance, 
                          const int* d_seam, int width, int height);
    
    // From hybrid_energy_kernel.cu
    void computeHybridEnergyCUDA(const float* d_luminance, float* d_energy, 
                               const float* d_forward_energy, const float* d_backward_energy,
                               int width, int height, float* h_backward_weight, float* h_forward_weight);
    void computeEnergyStatsCUDA(const float* d_backward_energy, const float* d_forward_energy,
                              float* h_min_backward, float* h_max_backward, float* h_avg_backward,
                              float* h_min_forward, float* h_max_forward, float* h_avg_forward,
                              int width, int height);
}

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
        ::computeHybridEnergyCUDA(d_luminance.get(), d_energy.get(), 
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
        
        // Find the minimum value in the last row (done on GPU)
        int min_x = ::findMinIndexLastRowCUDA(d_dp.get(), width, height);
        
        // Allocate device memory for seam
        d_seam.allocate(height);
        
        // Use new GPU-based backtracking
        ::backtrackSeamCUDA(d_dp.get(), d_seam.get(), width, height, min_x);
        
        // Copy seam data back to host
        d_seam.copyToHost(seam.data(), height);
    } catch (const std::exception& e) {
        std::cerr << "Error in computeSeamCUDA: " << e.what() << std::endl;
        
        // Fallback to CPU implementation if CUDA fails
        // First copy DP matrix to host
        std::vector<float> host_dp(dp.width * dp.height);
        d_dp.copyToHost(host_dp.data(), dp.width * dp.height);
        
        // Find minimum in last row
        int y = dp.height - 1;
        seam[y] = 0;
        float min_energy = host_dp[y * dp.width];
        for (int x = 1; x < dp.width; ++x) {
            if (host_dp[y * dp.width + x] < min_energy) {
                min_energy = host_dp[y * dp.width + x];
                seam[y] = x;
            }
        }
        
        // Backtrack to find the seam
        for (y = dp.height - 2; y >= 0; --y) {
            int x = seam[y + 1];
            seam[y] = x;  // Default: go straight up
            
            float up = host_dp[y * dp.width + x];
            float up_left = x > 0 ? host_dp[y * dp.width + (x - 1)] : FLT_MAX;
            float up_right = x < dp.width - 1 ? host_dp[y * dp.width + (x + 1)] : FLT_MAX;
            
            if (x > 0 && up_left < up && up_left <= up_right) {
                seam[y] = x - 1;  // Go up-left
            } else if (x < dp.width - 1 && up_right < up && up_right <= up_left) {
                seam[y] = x + 1;  // Go up-right
            }
        }
    }
}

// Optimized CUDA implementation of seam removal
void removeSeamCUDA(Image& img, Matrix& lum, Matrix& grad, const std::vector<int>& seam) {
    if (!cuda_initialized) {
        // Fall back to CPU implementation
        remove_seam(img, lum, grad, seam);
        return;
    }
    
    try {
        int width = img.width;
        int height = img.height;
        int new_width = width - 1;
        
        // Allocate device memory for output buffers
        d_output_image.allocate(new_width * height);
        d_output_lum.allocate(new_width * height);
        d_output_energy.allocate(new_width * height);
        d_seam.allocate(height);
        
        // Copy seam data to device
        d_seam.copyToDevice(seam.data(), height);
        
        // Copy input data to device if not already there
        d_image.copyToDevice(img.pixels.data(), width * height);
        d_luminance.copyToDevice(lum.items.data(), width * height);
        d_energy.copyToDevice(grad.items.data(), width * height);
        
        // Execute CUDA kernels in parallel streams to overlap computation
        cudaStream_t stream1, stream2, stream3;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        cudaStreamCreate(&stream3);
        
        // Remove seam from image
        cuda_removeSeamKernel(d_image.get(), d_output_image.get(), d_seam.get(), width, height);
        
        // Remove seam from luminance matrix
        removeSeamFromMatrixCUDA(d_luminance.get(), d_output_lum.get(), d_seam.get(), width, height);
        
        // Remove seam from energy matrix
        removeSeamFromMatrixCUDA(d_energy.get(), d_output_energy.get(), d_seam.get(), width, height);
        
        // Copy results back to host
        d_output_image.copyToHost(img.pixels.data(), new_width * height);
        d_output_lum.copyToHost(lum.items.data(), new_width * height);
        d_output_energy.copyToHost(grad.items.data(), new_width * height);
        
        // Update dimensions
        --img.width;
        --lum.width;
        --grad.width;
        img.stride = img.width;
        lum.stride = lum.width;
        grad.stride = grad.width;
        
        // Clean up streams
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream3);
    }
    catch (const std::exception& e) {
        std::cerr << "Error in removeSeamCUDA: " << e.what() << std::endl;
        // Fall back to CPU implementation
        remove_seam(img, lum, grad, seam);
    }
}

// Optimized CUDA implementation of gradient update after seam removal
void updateGradientCUDA(Matrix& grad, const Matrix& lum, const std::vector<int>& seam) {
    if (!cuda_initialized) {
        // Fall back to CPU implementation
        update_gradient(grad, lum, seam);
        return;
    }
    
    try {
        int width = grad.width;
        int height = grad.height;
        
        // Copy data to device
        d_luminance.copyToDevice(lum.items.data(), width * height);
        d_energy.copyToDevice(grad.items.data(), width * height);
        d_seam.copyToDevice(seam.data(), height);
        
        // Execute CUDA kernel
        ::updateGradientCUDA(d_energy.get(), d_luminance.get(), d_seam.get(), width, height);
        
        // Copy updated gradient back to host
        d_energy.copyToHost(grad.items.data(), width * height);
    }
    catch (const std::exception& e) {
        std::cerr << "Error in updateGradientCUDA: " << e.what() << std::endl;
        // Fall back to CPU implementation
        update_gradient(grad, lum, seam);
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
        d_output_image.free();
        d_seam.free();
        d_forward_energy.free();
        d_backward_energy.free();
        d_output_lum.free();
        d_output_energy.free();
    } catch (const std::exception& e) {
        std::cerr << "Error cleaning up CUDA resources: " << e.what() << std::endl;
    }
    
    cuda_initialized = false;
}

void saveVisualizationsCUDA(const Image& img, const Matrix& lum, const Matrix& energy, const Matrix& dp, 
                           const std::vector<int>& seam, int stage, bool detailed_viz) {
    visualization::saveStageVisualizations(img, lum, energy, dp, seam, stage, detailed_viz);
}

// GPU-accelerated visualization of the seam on the image
void visualizeSeamRemovalCUDA(const Image& img, const std::vector<int>& seam, const std::string& filename) {
    static int frame_counter = 0;
    frame_counter++;
    
    // Only update every 10th frame to reduce disk I/O and improve performance
    if (frame_counter % 10 != 0) {
        return;
    }
    
    if (!cuda_initialized) {
        // Fall back to CPU implementation if CUDA is not available
        visualization::visualizeSeamRemoval(img, seam, filename);
        return;
    }
    
    try {
        int width = img.width;
        int height = img.height;
        
        // Allocate device memory for image, seam, and output
        d_image.allocate(width * height);
        d_seam.allocate(height);
        d_output_image.allocate(width * height);
        
        // Copy data to device
        d_image.copyToDevice(img.pixels.data(), width * height);
        d_seam.copyToDevice(seam.data(), height);
        
        // Execute CUDA kernel to visualize seam
        ::visualizeSeamCUDA(d_image.get(), d_output_image.get(), d_seam.get(), width, height);
        
        // Copy result back to host
        std::vector<uint32_t> output_pixels(width * height);
        d_output_image.copyToHost(output_pixels.data(), width * height);
        
        // Save the image to disk
        if (!stbi_write_png(filename.c_str(), width, height, 4, output_pixels.data(), width * sizeof(uint32_t))) {
            std::cerr << "ERROR: could not save visualization to " << filename << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in visualizeSeamRemovalCUDA: " << e.what() << std::endl;
        // Fall back to CPU implementation
        visualization::visualizeSeamRemoval(img, seam, filename);
    }
}

} // namespace seam_carving_cuda 