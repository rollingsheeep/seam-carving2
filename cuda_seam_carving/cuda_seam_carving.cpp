#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <string>
#include <chrono>
#include <iomanip>

#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#include "cuda_kernels.cuh"

class Image {
public:
    std::vector<uint32_t> pixels;
    int width;
    int height;
    int stride;

    Image(int w, int h) : width(w), height(h), stride(w) {
        pixels.resize(width * height);
    }

    uint32_t& at(int row, int col) {
        return pixels[row * stride + col];
    }

    const uint32_t& at(int row, int col) const {
        return pixels[row * stride + col];
    }
};

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <input> <output> [--energy <forward|backward|hybrid>]\n";
    std::cerr << "  --energy: Choose energy calculation method (default: hybrid)\n";
}

int main(int argc, char** argv) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Print implementation information
    std::cout << "Using CUDA implementation\n";
    
    if (argc < 3) {
        print_usage(argv[0]);
        std::cerr << "ERROR: input and output files are required\n";
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];
    
    // Default to hybrid energy
    enum EnergyType { FORWARD, BACKWARD, HYBRID };
    EnergyType energy_type = HYBRID;
    
    // Parse command line arguments
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--energy" && i + 1 < argc) {
            std::string energy_type_str = argv[i + 1];
            if (energy_type_str == "forward") {
                energy_type = FORWARD;
            } else if (energy_type_str == "backward") {
                energy_type = BACKWARD;
            } else if (energy_type_str == "hybrid") {
                energy_type = HYBRID;
            } else {
                std::cerr << "ERROR: Invalid energy type. Use 'forward', 'backward', or 'hybrid'\n";
                return 1;
            }
            i++; // Skip the next argument
        } else {
            std::cerr << "ERROR: Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Load the image
    auto load_start = std::chrono::high_resolution_clock::now();
    int width, height, channels;
    uint32_t* pixels = (uint32_t*)stbi_load(input_path, &width, &height, &channels, 4);
    if (!pixels) {
        std::cerr << "ERROR: could not read " << input_path << "\n";
        return 1;
    }

    // Create our image object
    Image img(width, height);
    std::copy(pixels, pixels + width * height, img.pixels.begin());
    stbi_image_free(pixels);
    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start);
    std::cout << "Image loading time: " << load_duration.count() << " ms\n";

    // Initialize CUDA device structures
    DeviceImage d_img;
    DeviceMatrix d_lum, d_grad, d_dp;
    int* d_seam = nullptr;
    
    // Allocate device memory
    auto cuda_alloc_start = std::chrono::high_resolution_clock::now();
    allocateDeviceMemory(&d_img, &d_lum, &d_grad, &d_dp, &d_seam, width, height);
    
    // Copy image data to device
    copyImageToDevice(img.pixels.data(), d_img, width, height);
    auto cuda_alloc_end = std::chrono::high_resolution_clock::now();
    auto cuda_alloc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cuda_alloc_end - cuda_alloc_start);
    std::cout << "CUDA memory allocation and data transfer time: " << cuda_alloc_duration.count() << " ms\n";

    // Compute initial luminance
    auto luminance_start = std::chrono::high_resolution_clock::now();
    computeLuminanceCuda(d_img, d_lum);
    auto luminance_end = std::chrono::high_resolution_clock::now();
    auto luminance_duration = std::chrono::duration_cast<std::chrono::milliseconds>(luminance_end - luminance_start);
    std::cout << "Luminance computation time: " << luminance_duration.count() << " ms\n";

    // Counter variables for hybrid energy
    int hybrid_forward_count = 0;
    int hybrid_backward_count = 0;

    // Choose energy calculation method
    auto energy_start = std::chrono::high_resolution_clock::now();
    switch (energy_type) {
        case FORWARD:
            std::cout << "Using forward energy calculation\n";
            computeForwardEnergyCuda(d_lum, d_grad);
            break;
        case BACKWARD:
            std::cout << "Using backward energy calculation (Sobel filter)\n";
            computeSobelFilterCuda(d_lum, d_grad);
            break;
        case HYBRID:
            std::cout << "Using hybrid energy calculation\n";
            computeHybridEnergyCuda(d_lum, d_grad, &hybrid_forward_count, &hybrid_backward_count);
            break;
    }
    auto energy_end = std::chrono::high_resolution_clock::now();
    auto energy_duration = std::chrono::duration_cast<std::chrono::milliseconds>(energy_end - energy_start);
    std::cout << "Initial energy computation time: " << energy_duration.count() << " ms\n";

    // Remove seams
    int seams_to_remove = width * 2 / 3;
    std::cout << "Removing " << seams_to_remove << " seams...\n";
    
    auto seam_removal_start = std::chrono::high_resolution_clock::now();
    long long total_dp_time = 0;
    long long total_seam_time = 0;
    long long total_remove_time = 0;
    long long total_update_time = 0;
    
    // Host-side seam array
    std::vector<int> h_seam(height);
    
    for (int i = 0; i < seams_to_remove; ++i) {
        // Time dynamic programming
        auto dp_start = std::chrono::high_resolution_clock::now();
        computeDynamicProgrammingCuda(d_grad, d_dp);
        auto dp_end = std::chrono::high_resolution_clock::now();
        total_dp_time += std::chrono::duration_cast<std::chrono::microseconds>(dp_end - dp_start).count();
        
        // Time seam computation
        auto seam_start = std::chrono::high_resolution_clock::now();
        computeSeamCuda(d_dp, d_seam, h_seam.data());
        auto seam_end = std::chrono::high_resolution_clock::now();
        total_seam_time += std::chrono::duration_cast<std::chrono::microseconds>(seam_end - seam_start).count();
        
        // Time seam removal
        auto remove_start = std::chrono::high_resolution_clock::now();
        removeSeamCuda(&d_img, &d_lum, &d_grad, d_seam);
        auto remove_end = std::chrono::high_resolution_clock::now();
        total_remove_time += std::chrono::duration_cast<std::chrono::microseconds>(remove_end - remove_start).count();
        
        // Time energy update
        auto update_start = std::chrono::high_resolution_clock::now();
        switch (energy_type) {
            case FORWARD:
                computeForwardEnergyCuda(d_lum, d_grad);
                break;
            case BACKWARD:
                updateGradientCuda(d_lum, d_grad, d_seam);
                break;
            case HYBRID:
                computeHybridEnergyCuda(d_lum, d_grad, &hybrid_forward_count, &hybrid_backward_count);
                break;
        }
        auto update_end = std::chrono::high_resolution_clock::now();
        total_update_time += std::chrono::duration_cast<std::chrono::microseconds>(update_end - update_start).count();
        
        // Print progress every 10% of seams
        if ((i + 1) % (seams_to_remove / 10) == 0 || i == seams_to_remove - 1) {
            int progress = ((i + 1) * 100) / seams_to_remove;
            std::cout << "Progress: " << progress << "% complete\n";
        }
    }
    auto seam_removal_end = std::chrono::high_resolution_clock::now();
    auto seam_removal_duration = std::chrono::duration_cast<std::chrono::milliseconds>(seam_removal_end - seam_removal_start);
    
    // Print detailed timing information
    std::cout << "Seam removal breakdown:\n";
    std::cout << "  Dynamic programming: " << (total_dp_time / 1000.0) << " ms\n";
    std::cout << "  Seam computation: " << (total_seam_time / 1000.0) << " ms\n";
    std::cout << "  Seam removal: " << (total_remove_time / 1000.0) << " ms\n";
    std::cout << "  Energy update: " << (total_update_time / 1000.0) << " ms\n";
    std::cout << "  Total seam removal time: " << seam_removal_duration.count() << " ms\n";

    if (energy_type == HYBRID) {
        int total = hybrid_forward_count + hybrid_backward_count;
        float forward_ratio = 100.0f * hybrid_forward_count / (total + 1e-6f);
        float backward_ratio = 100.0f * hybrid_backward_count / (total + 1e-6f);
        std::cout << "\nHybrid energy summary:\n";
        std::cout << "  Backward energy usage: " << std::fixed << std::setprecision(2) << backward_ratio << "%\n";
        std::cout << "  Forward energy usage: " << std::fixed << std::setprecision(2) << forward_ratio << "%\n";
        std::cout << "  Ratio (Backward:Forward): " << (backward_ratio / forward_ratio) << ":1\n";
    }

    // Copy result back to host
    auto copy_back_start = std::chrono::high_resolution_clock::now();
    img.width = d_img.width;
    img.height = d_img.height;
    img.stride = d_img.stride;
    img.pixels.resize(img.width * img.height);
    copyImageFromDevice(img.pixels.data(), d_img, img.width, img.height);
    auto copy_back_end = std::chrono::high_resolution_clock::now();
    auto copy_back_duration = std::chrono::duration_cast<std::chrono::milliseconds>(copy_back_end - copy_back_start);
    std::cout << "CUDA to host transfer time: " << copy_back_duration.count() << " ms\n";
    
    // Free device memory
    freeDeviceMemory(d_img, d_lum, d_grad, d_dp, d_seam);

    // Save the result
    auto save_start = std::chrono::high_resolution_clock::now();
    if (!stbi_write_png(output_path, img.width, img.height, 4, img.pixels.data(), img.stride * sizeof(uint32_t))) {
        std::cerr << "ERROR: could not save file " << output_path << "\n";
        return 1;
    }
    auto save_end = std::chrono::high_resolution_clock::now();
    auto save_duration = std::chrono::duration_cast<std::chrono::milliseconds>(save_end - save_start);
    std::cout << "Image saving time: " << save_duration.count() << " ms\n";

    // End timing and calculate duration
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "OK: generated " << output_path << "\n";
    std::cout << "Total execution time: " << duration.count() << " ms\n";
    
    // Calculate speedup compared to sequential implementation
    std::cout << "Estimated speedup over sequential implementation: ~" 
              << std::fixed << std::setprecision(1) << (60000.0 / duration.count()) << "x\n";
    
    return 0;
} 