#include "visualization.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// Implementation file for visualization functions

namespace visualization {

// Convert a matrix to a grayscale image for visualization
void matrixToGrayscaleImage(const Matrix& matrix, const std::string& filename) {
    int width = matrix.width;
    int height = matrix.height;
    
    std::vector<uint32_t> pixels(width * height);
    
    // Find min and max values for normalization
    float min_val = matrix.items[0];
    float max_val = matrix.items[0];
    
    for (int i = 0; i < width * height; ++i) {
        if (matrix.items[i] < min_val) min_val = matrix.items[i];
        if (matrix.items[i] > max_val) max_val = matrix.items[i];
    }
    
    // Normalize and convert to grayscale
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float normalized = (matrix.at(y, x) - min_val) / (max_val - min_val + 1e-6f);
            uint8_t gray = static_cast<uint8_t>(normalized * 255);
            
            // Create RGBA pixel (grayscale)
            uint32_t pixel = (0xFF << 24) | (gray << 16) | (gray << 8) | gray;
            pixels[y * width + x] = pixel;
        }
    }
    
    // Save image
    stbi_write_png(filename.c_str(), width, height, 4, pixels.data(), width * sizeof(uint32_t));
}

// Create a heatmap visualization of the energy map
void energyToHeatmap(const Matrix& energy, const std::string& filename) {
    int width = energy.width;
    int height = energy.height;
    
    std::vector<uint32_t> pixels(width * height);
    
    // Find min and max values for normalization
    float min_val = energy.items[0];
    float max_val = energy.items[0];
    
    for (int i = 0; i < width * height; ++i) {
        if (energy.items[i] < min_val) min_val = energy.items[i];
        if (energy.items[i] > max_val) max_val = energy.items[i];
    }
    
    // Create heatmap (blue to red)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float normalized = (energy.at(y, x) - min_val) / (max_val - min_val + 1e-6f);
            
            // Blue (low energy) to Red (high energy)
            uint8_t r = static_cast<uint8_t>(normalized * 255);
            uint8_t g = static_cast<uint8_t>((1.0f - std::abs(normalized - 0.5f) * 2.0f) * 255);
            uint8_t b = static_cast<uint8_t>((1.0f - normalized) * 255);
            
            uint32_t pixel = (0xFF << 24) | (r << 16) | (g << 8) | b;
            pixels[y * width + x] = pixel;
        }
    }
    
    // Save image
    stbi_write_png(filename.c_str(), width, height, 4, pixels.data(), width * sizeof(uint32_t));
}

// Visualize the current image with the seam highlighted
void visualizeSeam(const Image& img, const std::vector<int>& seam, const std::string& filename) {
    int width = img.width;
    int height = img.height;
    
    std::vector<uint32_t> pixels(width * height);
    
    // Copy the original image
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            pixels[y * width + x] = img.at(y, x);
        }
    }
    
    // Highlight the seam in red
    for (int y = 0; y < height; ++y) {
        int x = seam[y];
        if (x >= 0 && x < width) {
            pixels[y * width + x] = 0xFFFF0000;  // Bright red
        }
    }
    
    // Save image
    stbi_write_png(filename.c_str(), width, height, 4, pixels.data(), width * sizeof(uint32_t));
}

// Save all visualizations for a specific stage
void saveStageVisualizations(
    const Image& img,
    const Matrix& lum,
    const Matrix& energy,
    const Matrix& dp,
    const std::vector<int>& seam,
    int stage,
    bool detailed_viz
) {
    // Create output directory if it doesn't exist
    std::string dir = "output";
    
    #ifdef _WIN32
    // More robust Windows directory creation
    std::string mkdir_cmd = "if not exist " + dir + " mkdir " + dir;
    system(mkdir_cmd.c_str());
    
    // Print directory creation for debugging
    std::cout << "Creating visualization directory: " << dir << std::endl;
    #else
    system(("mkdir -p " + dir).c_str());
    #endif
    
    // Create full path string
    std::string prefix = dir + "/stage_" + std::to_string(stage);
    // Ensure proper padding with leading zeros
    if (stage < 10) prefix = dir + "/stage_00" + std::to_string(stage);
    else if (stage < 100) prefix = dir + "/stage_0" + std::to_string(stage);
    
    std::cout << "Saving stage " << stage << " visualization" << (detailed_viz ? " (detailed)" : "") << std::endl;
    
    // Always save the current image with seam
    visualizeSeam(img, seam, prefix + "_seam.png");
    
    // Only save detailed visualizations if requested
    if (detailed_viz) {
        // Save detailed visualizations in parallel if OpenMP is available
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                matrixToGrayscaleImage(lum, prefix + "_grayscale.png");
            }
            
            #pragma omp section
            {
                energyToHeatmap(energy, prefix + "_energy.png");
            }
            
            #pragma omp section
            {
                matrixToGrayscaleImage(dp, prefix + "_dp.png");
            }
            
            #pragma omp section
            {
                // Save current resized image
                stbi_write_png((prefix + "_image.png").c_str(), img.width, img.height, 
                              4, img.pixels.data(), img.stride * sizeof(uint32_t));
            }
        }
        
        // Create a log file with information
        std::ofstream log(prefix + "_info.txt");
        log << "Stage: " << stage << std::endl;
        log << "Image dimensions: " << img.width << " x " << img.height << std::endl;
        log << "Energy stats: " << std::endl;
        
        // Calculate energy stats
        float min_energy = energy.items[0];
        float max_energy = energy.items[0];
        float avg_energy = 0.0f;
        
        for (int i = 0; i < energy.width * energy.height; ++i) {
            min_energy = std::min(min_energy, energy.items[i]);
            max_energy = std::max(max_energy, energy.items[i]);
            avg_energy += energy.items[i];
        }
        
        avg_energy /= (energy.width * energy.height);
        
        log << "  Min: " << min_energy << std::endl;
        log << "  Max: " << max_energy << std::endl;
        log << "  Avg: " << avg_energy << std::endl;
        
        log.close();
    }
}

// Update a single output.png file showing the current image with the most recently removed seam
void visualizeSeamRemoval(const Image& img, const std::vector<int>& seam, const std::string& filename) {
    static int frame_counter = 0;
    frame_counter++;
    
    // Only update every 10th frame to reduce disk I/O and improve performance
    if (frame_counter % 10 != 0) {
        return;
    }
    
    int width = img.width;
    int height = img.height;
    
    std::vector<uint32_t> pixels(width * height);
    
    // Copy the original image with optimized memory access pattern
    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            pixels[y * width + x] = img.at(y, x);
        }
    }
    
    // Highlight the seam in red (bright, easy to see)
    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        int x = seam[y];
        if (x >= 0 && x < width) {
            pixels[y * width + x] = 0xFFFF0000;  // Bright red
        }
    }
    
    // Save image, overwriting the previous version
    if (!stbi_write_png(filename.c_str(), width, height, 4, pixels.data(), width * sizeof(uint32_t))) {
        std::cerr << "ERROR: could not save visualization to " << filename << std::endl;
    }
}

} // namespace visualization 