#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <string>
#include <chrono>
#include <iomanip>  // For std::fixed and std::setprecision

#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

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

class Matrix {
public:
    std::vector<float> items;
    int width;
    int height;
    int stride;

    Matrix(int w, int h) : width(w), height(h), stride(w) {
        items.resize(width * height);
    }

    float& at(int row, int col) {
        return items[row * stride + col];
    }

    const float& at(int row, int col) const {
        return items[row * stride + col];
    }

    bool within(int row, int col) const {
        return 0 <= col && col < width && 0 <= row && row < height;
    }
};

// Convert RGB to luminance using the standard coefficients
float rgb_to_lum(uint32_t rgb) {
    float r = ((rgb >> (8*0)) & 0xFF) / 255.0f;
    float g = ((rgb >> (8*1)) & 0xFF) / 255.0f;
    float b = ((rgb >> (8*2)) & 0xFF) / 255.0f;
    return 0.2126f*r + 0.7152f*g + 0.0722f*b;
}

// Converts each pixel in the image to its luminance value
void compute_luminance(const Image& img, Matrix& lum) {
    assert(img.width == lum.width && img.height == lum.height);
    for (int y = 0; y < lum.height; ++y) {
        for (int x = 0; x < lum.width; ++x) {
            lum.at(y, x) = rgb_to_lum(img.at(y, x));
        }
    }
}

// Applies Sobel edge detection filter at a specific pixel to detect edges
float sobel_filter_at(const Matrix& mat, int cx, int cy) {
    static const float gx[3][3] = {
        {1.0f, 0.0f, -1.0f},
        {2.0f, 0.0f, -2.0f},
        {1.0f, 0.0f, -1.0f},
    };

    static const float gy[3][3] = {
        {1.0f, 2.0f, 1.0f},
        {0.0f, 0.0f, 0.0f},
        {-1.0f, -2.0f, -1.0f},
    };

    float sx = 0.0f;
    float sy = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int x = cx + dx;
            int y = cy + dy;
            float c = mat.within(y, x) ? mat.at(y, x) : 0.0f;
            sx += c * gx[dy + 1][dx + 1];
            sy += c * gy[dy + 1][dx + 1];
        }
    }
    return std::sqrt(sx*sx + sy*sy);
}

// Applies Sobel filter to the entire image to create energy gradient map
void compute_sobel_filter(const Matrix& mat, Matrix& grad) {
    assert(mat.width == grad.width && mat.height == grad.height);
    for (int y = 0; y < mat.height; ++y) {
        for (int x = 0; x < mat.width; ++x) {
            grad.at(y, x) = sobel_filter_at(mat, x, y);
        }
    }
}

// Uses dynamic programming to compute cumulative energy matrix
void compute_dynamic_programming(const Matrix& grad, Matrix& dp) {
    assert(grad.width == dp.width && grad.height == dp.height);

    // Initialize first row
    for (int x = 0; x < grad.width; ++x) {
        dp.at(0, x) = grad.at(0, x);
    }

    // Fill the rest of the matrix
    for (int y = 1; y < grad.height; ++y) {
        for (int x = 0; x < grad.width; ++x) {
            float min_prev = dp.at(y - 1, x);
            if (x > 0) {
                min_prev = std::min(min_prev, dp.at(y - 1, x - 1));
            }
            if (x < grad.width - 1) {
                min_prev = std::min(min_prev, dp.at(y - 1, x + 1));
            }
            dp.at(y, x) = grad.at(y, x) + min_prev;
        }
    }
}

// Computes the minimum energy vertical seam path through the image
void compute_seam(const Matrix& dp, std::vector<int>& seam) {
    seam.resize(dp.height);
    
    // Find the minimum value in the last row
    int y = dp.height - 1;
    seam[y] = 0;
    float min_energy = dp.at(y, 0);
    for (int x = 1; x < dp.width; ++x) {
        if (dp.at(y, x) < min_energy) {
            min_energy = dp.at(y, x);
            seam[y] = x;
        }
    }

    // Backtrack to find the seam
    for (y = dp.height - 2; y >= 0; --y) {
        int x = seam[y + 1];
        seam[y] = x;
        float min_energy = dp.at(y, x);
        
        if (x > 0 && dp.at(y, x - 1) < min_energy) {
            min_energy = dp.at(y, x - 1);
            seam[y] = x - 1;
        }
        if (x < dp.width - 1 && dp.at(y, x + 1) < min_energy) {
            min_energy = dp.at(y, x + 1);
            seam[y] = x + 1;
        }
    }
}

// Removes a seam from the image and updates relevant matrices
void remove_seam(Image& img, Matrix& lum, Matrix& grad, const std::vector<int>& seam) {
    std::vector<uint32_t> new_pixels((img.width - 1) * img.height);
    std::vector<float> new_lum((img.width - 1) * img.height);
    std::vector<float> new_grad((img.width - 1) * img.height);
    
    for (int y = 0; y < img.height; ++y) {
        int x_src = 0;
        int x_dst = 0;
        while (x_dst < img.width - 1) {
            if (x_src == seam[y]) {
                ++x_src;
                continue;
            }
            new_pixels[y * (img.width - 1) + x_dst] = img.at(y, x_src);
            new_lum[y * (img.width - 1) + x_dst] = lum.at(y, x_src);
            new_grad[y * (img.width - 1) + x_dst] = grad.at(y, x_src);
            ++x_src;
            ++x_dst;
        }
    }
    
    --img.width;
    --lum.width;
    --grad.width;
    img.stride = img.width;
    lum.stride = lum.width;
    grad.stride = grad.width;
    
    img.pixels = std::move(new_pixels);
    lum.items = std::move(new_lum);
    grad.items = std::move(new_grad);
}

// Updates gradient values for pixels adjacent to the removed seam
void update_gradient(Matrix& grad, const Matrix& lum, const std::vector<int>& seam) {
    // Only update the gradient for pixels adjacent to the removed seam
    for (int y = 0; y < grad.height; ++y) {
        int x = seam[y];
        // Update one pixel to the left and right of the seam
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            if (nx >= 0 && nx < grad.width) {
                grad.at(y, nx) = sobel_filter_at(lum, nx, y);
            }
        }
    }
}

// Calculates forward energy as described in "Improved Seam Carving for Video Retargeting"
void compute_forward_energy(const Matrix& lum, Matrix& energy) {
    assert(lum.width == energy.width && lum.height == energy.height);
    int w = lum.width;
    int h = lum.height;

    // Initialize the first row to 0
    for (int x = 0; x < w; ++x) {
        energy.at(0, x) = 0.0f;
    }

    // DP forward energy computation
    for (int y = 1; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float cU = 0.0f, cL = 0.0f, cR = 0.0f;

            // Compute neighbor costs safely with bounds
            float left   = (x > 0)     ? lum.at(y, x - 1) : lum.at(y, x);
            float right  = (x < w - 1) ? lum.at(y, x + 1) : lum.at(y, x);
            float up     = lum.at(y - 1, x);
            float upLeft = (x > 0)     ? lum.at(y - 1, x - 1) : up;
            float upRight= (x < w - 1) ? lum.at(y - 1, x + 1) : up;

            // Cost for going straight up
            cU = std::abs(right - left);

            // Cost for going up-left
            cL = cU + std::abs(up - left);

            // Cost for going up-right
            cR = cU + std::abs(up - right);

            // Get minimum previous path cost
            float min_energy = energy.at(y - 1, x) + cU;
            if (x > 0)     min_energy = std::min(min_energy, energy.at(y - 1, x - 1) + cL);
            if (x < w - 1) min_energy = std::min(min_energy, energy.at(y - 1, x + 1) + cR);

            energy.at(y, x) = min_energy;
        }
    }
}

int hybrid_forward_count = 0;
int hybrid_backward_count = 0;

// Dynamically chooses between forward and backward energy based on image characteristics
void compute_hybrid_energy(const Matrix& lum, Matrix& energy) {
    assert(lum.width == energy.width && lum.height == energy.height);

    Matrix forward_energy(lum.width, lum.height);
    Matrix backward_energy(lum.width, lum.height);

    compute_forward_energy(lum, forward_energy);
    compute_sobel_filter(lum, backward_energy);

    // Find min and max values for normalization
    float min_forward = FLT_MAX, max_forward = -FLT_MAX;
    float min_backward = FLT_MAX, max_backward = -FLT_MAX;
    
    for (int y = 0; y < lum.height; ++y) {
        for (int x = 0; x < lum.width; ++x) {
            float f_val = forward_energy.at(y, x);
            float b_val = backward_energy.at(y, x);
            
            min_forward = std::min(min_forward, f_val);
            max_forward = std::max(max_forward, f_val);
            min_backward = std::min(min_backward, b_val);
            max_backward = std::max(max_backward, b_val);
        }
    }
    
    // Create normalized versions of both energy types
    Matrix norm_forward_energy(lum.width, lum.height);
    Matrix norm_backward_energy(lum.width, lum.height);
    
    float forward_range = max_forward - min_forward + 1e-6f;
    float backward_range = max_backward - min_backward + 1e-6f;
    
    for (int y = 0; y < lum.height; ++y) {
        for (int x = 0; x < lum.width; ++x) {
            // Normalize to [0, 1] range
            norm_forward_energy.at(y, x) = (forward_energy.at(y, x) - min_forward) / forward_range;
            norm_backward_energy.at(y, x) = (backward_energy.at(y, x) - min_backward) / backward_range;
        }
    }
    
    // Calculate statistics on normalized energies
    float avg_forward = 0.0f, avg_backward = 0.0f;
    float std_dev_forward = 0.0f, std_dev_backward = 0.0f;
    
    for (int y = 0; y < lum.height; ++y) {
        for (int x = 0; x < lum.width; ++x) {
            avg_forward += norm_forward_energy.at(y, x);
            avg_backward += norm_backward_energy.at(y, x);
        }
    }
    
    int total_pixels = lum.width * lum.height;
    avg_forward /= total_pixels;
    avg_backward /= total_pixels;
    
    for (int y = 0; y < lum.height; ++y) {
        for (int x = 0; x < lum.width; ++x) {
            float diff_forward = norm_forward_energy.at(y, x) - avg_forward;
            float diff_backward = norm_backward_energy.at(y, x) - avg_backward;
            std_dev_forward += diff_forward * diff_forward;
            std_dev_backward += diff_backward * diff_backward;
        }
    }
    
    std_dev_forward = std::sqrt(std_dev_forward / total_pixels);
    std_dev_backward = std::sqrt(std_dev_backward / total_pixels);
    
    // Count high energy pixels in normalized space
    int high_energy_forward = 0, high_energy_backward = 0;
    float forward_threshold = avg_forward + std_dev_forward;
    float backward_threshold = avg_backward + std_dev_backward;
    
    for (int y = 0; y < lum.height; ++y) {
        for (int x = 0; x < lum.width; ++x) {
            if (norm_forward_energy.at(y, x) > forward_threshold) high_energy_forward++;
            if (norm_backward_energy.at(y, x) > backward_threshold) high_energy_backward++;
        }
    }
    
    float edge_density_forward = static_cast<float>(high_energy_forward) / total_pixels;
    float edge_density_backward = static_cast<float>(high_energy_backward) / total_pixels;
    
    // Compare normalized energies to decide which to use
    bool use_backward = false;
    
    // Factor 1: Edge density comparison (which method detects more edges)
    if (edge_density_backward > edge_density_forward * 1.1f) {
        use_backward = true;
    } 
    // Factor 2: Standard deviation comparison (which method has more variation)
    else if (std_dev_backward > std_dev_forward * 1.1f) {
        use_backward = true;
    }
    // Factor 3: Add randomness to break ties and ensure some backward energy usage
    else if (edge_density_backward > edge_density_forward * 0.9f && 
             std_dev_backward > std_dev_forward * 0.9f) {
        float random_factor = static_cast<float>(rand()) / RAND_MAX;
        if (random_factor < 0.3f) use_backward = true; // 30% chance of using backward
    }
    
    // Update counters
    if (use_backward) {
        hybrid_backward_count++;
    } else {
        hybrid_forward_count++;
    }
    
    // Use the chosen energy method
    const Matrix& chosen = use_backward ? backward_energy : forward_energy;
    for (int y = 0; y < lum.height; ++y) {
        for (int x = 0; x < lum.width; ++x) {
            energy.at(y, x) = chosen.at(y, x);
        }
    }
}

// Displays usage information and command-line options
void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <input> <output> [--energy <forward|backward|hybrid>]\n";
    std::cerr << "  --energy: Choose energy calculation method (default: hybrid)\n";
}

// Main function that orchestrates the seam carving process
int main(int argc, char** argv) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Print implementation information
    std::cout << "Using Sequential implementation\n";
    
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

    // Create matrices for processing
    Matrix lum(width, height);
    Matrix grad(width, height);
    Matrix dp(width, height);
    std::vector<int> seam(height);

    // Compute initial luminance
    auto luminance_start = std::chrono::high_resolution_clock::now();
    compute_luminance(img, lum);
    auto luminance_end = std::chrono::high_resolution_clock::now();
    auto luminance_duration = std::chrono::duration_cast<std::chrono::milliseconds>(luminance_end - luminance_start);
    std::cout << "Luminance computation time: " << luminance_duration.count() << " ms\n";

    // Choose energy calculation method
    auto energy_start = std::chrono::high_resolution_clock::now();
    switch (energy_type) {
        case FORWARD:
            std::cout << "Using forward energy calculation\n";
            compute_forward_energy(lum, grad);
            break;
        case BACKWARD:
            std::cout << "Using backward energy calculation (Sobel filter)\n";
            compute_sobel_filter(lum, grad);
            break;
        case HYBRID:
            std::cout << "Using hybrid energy calculation\n";
            compute_hybrid_energy(lum, grad);
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
    
    for (int i = 0; i < seams_to_remove; ++i) {
        // Create a new dp matrix with the current dimensions
        Matrix dp_current(img.width, img.height);
        
        // Time dynamic programming
        auto dp_start = std::chrono::high_resolution_clock::now();
        compute_dynamic_programming(grad, dp_current);
        auto dp_end = std::chrono::high_resolution_clock::now();
        total_dp_time += std::chrono::duration_cast<std::chrono::microseconds>(dp_end - dp_start).count();
        
        // Time seam computation
        auto seam_start = std::chrono::high_resolution_clock::now();
        compute_seam(dp_current, seam);
        auto seam_end = std::chrono::high_resolution_clock::now();
        total_seam_time += std::chrono::duration_cast<std::chrono::microseconds>(seam_end - seam_start).count();
        
        // Time seam removal
        auto remove_start = std::chrono::high_resolution_clock::now();
        remove_seam(img, lum, grad, seam);
        auto remove_end = std::chrono::high_resolution_clock::now();
        total_remove_time += std::chrono::duration_cast<std::chrono::microseconds>(remove_end - remove_start).count();
        
        // Time energy update
        auto update_start = std::chrono::high_resolution_clock::now();
        switch (energy_type) {
            case FORWARD:
                compute_forward_energy(lum, grad);
                break;
            case BACKWARD:
                update_gradient(grad, lum, seam);
                break;
            case HYBRID:
                // For hybrid mode, we need to decide which method to use for each iteration
                // We'll use the same hybrid decision function
                compute_hybrid_energy(lum, grad);
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
    return 0;
} 