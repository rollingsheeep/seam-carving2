#include "data_structures.h"  // Include our new header file
#include "seam_carving_cuda.h"
#include <chrono>
#include <iostream>
#include <string>

// STB Image libraries for loading and saving images
#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

// Command line arguments
bool use_cuda = true;
bool visualize = false;
bool detailed_viz = false;
std::string energy_type_str = "hybrid";
std::string input_path;
std::string output_path;

void parse_arguments(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input> <output> [options]\n";
        std::cerr << "Options:\n";
        std::cerr << "  --energy <forward|backward|hybrid>  Choose energy calculation method (default: hybrid)\n";
        std::cerr << "  --cpu                               Disable CUDA acceleration\n";
        std::cerr << "  --visualize                         Enable visualization of seam removal (output.png)\n";
        std::cerr << "  --detailed-viz                      Enable detailed visualization outputs\n";
        exit(1);
    }

    input_path = argv[1];
    output_path = argv[2];

    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--energy" && i + 1 < argc) {
            energy_type_str = argv[i + 1];
            i++;
        } else if (arg == "--cpu") {
            use_cuda = false;
        } else if (arg == "--visualize") {
            visualize = true;
        } else if (arg == "--detailed-viz") {
            detailed_viz = true;
            visualize = true; // Detailed visualization implies visualization
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            exit(1);
        }
    }
}

// The main function for the CUDA-enabled version
int main(int argc, char** argv) {
    // Parse command line arguments
    parse_arguments(argc, argv);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Load the image
    auto load_start = std::chrono::high_resolution_clock::now();
    int width, height, channels;
    uint32_t* pixels = (uint32_t*)stbi_load(input_path.c_str(), &width, &height, &channels, 4);
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

    // Create matrices for processing
    Matrix lum(width, height);
    Matrix grad(width, height);
    Matrix dp(width, height);
    std::vector<int> seam(height);

    // Initialize CUDA if enabled
    if (use_cuda) {
        seam_carving_cuda::initCUDA();
        use_cuda = seam_carving_cuda::isCUDAAvailable();
        if (!use_cuda) {
            std::cout << "CUDA initialization failed, falling back to CPU implementation\n";
        } else {
            std::cout << "Using CUDA acceleration\n";
        }
    } else {
        std::cout << "Using CPU implementation (CUDA disabled)\n";
    }

    // Determine energy calculation method
    enum EnergyType { FORWARD, BACKWARD, HYBRID };
    EnergyType energy_type;
    
    if (energy_type_str == "forward") {
        energy_type = FORWARD;
        std::cout << "Using forward energy calculation\n";
    } else if (energy_type_str == "backward") {
        energy_type = BACKWARD;
        std::cout << "Using backward energy calculation (Sobel filter)\n";
    } else if (energy_type_str == "hybrid") {
        energy_type = HYBRID;
        std::cout << "Using hybrid energy calculation\n";
    } else {
        std::cerr << "ERROR: Invalid energy type. Use 'forward', 'backward', or 'hybrid'\n";
        return 1;
    }

    // Compute initial luminance
    auto luminance_start = std::chrono::high_resolution_clock::now();
    if (use_cuda) {
        seam_carving_cuda::computeLuminanceCUDA(lum, img);
    } else {
        compute_luminance(img, lum);
    }
    auto luminance_end = std::chrono::high_resolution_clock::now();
    auto luminance_duration = std::chrono::duration_cast<std::chrono::milliseconds>(luminance_end - luminance_start);
    std::cout << "Luminance computation time: " << luminance_duration.count() << " ms\n";

    // Compute initial energy
    auto energy_start = std::chrono::high_resolution_clock::now();
    if (use_cuda) {
        switch (energy_type) {
            case FORWARD:
                seam_carving_cuda::computeForwardEnergyCUDA(grad, lum);
                break;
            case BACKWARD:
                seam_carving_cuda::computeSobelFilterCUDA(grad, lum);
                break;
            case HYBRID:
                seam_carving_cuda::computeHybridEnergyCUDA(grad, lum);
                break;
        }
    } else {
        switch (energy_type) {
            case FORWARD:
                compute_forward_energy(lum, grad);
                break;
            case BACKWARD:
                compute_sobel_filter(lum, grad);
                break;
            case HYBRID:
                compute_hybrid_energy(lum, grad);
                break;
        }
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
    
    for (int i = 0; i < seams_to_remove; ++i) {
        // Create a new dp matrix with the current dimensions
        Matrix dp_current(img.width, img.height);
        
        // Time dynamic programming
        auto dp_start = std::chrono::high_resolution_clock::now();
        if (use_cuda) {
            // Using CPU implementation for dynamic programming as CUDA version was slower
            seam_carving_cuda::computeDynamicProgrammingCUDA(dp_current, grad);
        } else {
            compute_dynamic_programming(grad, dp_current);
        }
        auto dp_end = std::chrono::high_resolution_clock::now();
        total_dp_time += std::chrono::duration_cast<std::chrono::microseconds>(dp_end - dp_start).count();
        
        // Time seam computation
        auto seam_start = std::chrono::high_resolution_clock::now();
        if (use_cuda) {
            seam_carving_cuda::computeSeamCUDA(seam, dp_current);
        } else {
            compute_seam(dp_current, seam);
        }
        auto seam_end = std::chrono::high_resolution_clock::now();
        total_seam_time += std::chrono::duration_cast<std::chrono::microseconds>(seam_end - seam_start).count();
        
        // Time seam removal
        auto remove_start = std::chrono::high_resolution_clock::now();
        remove_seam(img, lum, grad, seam);
        auto remove_end = std::chrono::high_resolution_clock::now();
        total_remove_time += std::chrono::duration_cast<std::chrono::microseconds>(remove_end - remove_start).count();
        
        // Time energy update
        auto update_start = std::chrono::high_resolution_clock::now();
        if (use_cuda) {
            switch (energy_type) {
                case FORWARD:
                    seam_carving_cuda::computeForwardEnergyCUDA(grad, lum);
                    break;
                case BACKWARD:
                    // For backward energy, we can optimize by just updating affected pixels
                    seam_carving_cuda::computeSobelFilterCUDA(grad, lum);
                    break;
                case HYBRID:
                    seam_carving_cuda::computeHybridEnergyCUDA(grad, lum);
                    break;
            }
        } else {
            switch (energy_type) {
                case FORWARD:
                    compute_forward_energy(lum, grad);
                    break;
                case BACKWARD:
                    update_gradient(grad, lum, seam);
                    break;
                case HYBRID:
                    compute_hybrid_energy(lum, grad);
                    break;
            }
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

    // Save the result
    auto save_start = std::chrono::high_resolution_clock::now();
    if (!stbi_write_png(output_path.c_str(), img.width, img.height, 4, img.pixels.data(), img.stride * sizeof(uint32_t))) {
        std::cerr << "ERROR: could not save file " << output_path << "\n";
        return 1;
    }
    auto save_end = std::chrono::high_resolution_clock::now();
    auto save_duration = std::chrono::duration_cast<std::chrono::milliseconds>(save_end - save_start);
    std::cout << "Image saving time: " << save_duration.count() << " ms\n";

    // Clean up CUDA resources
    if (use_cuda) {
        seam_carving_cuda::cleanupCUDA();
    }

    // End timing and calculate duration
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "OK: generated " << output_path << "\n";
    std::cout << "Total execution time: " << duration.count() << " ms\n";
    
    return 0;
} 