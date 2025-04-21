#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

void compute_luminance(const Image& img, Matrix& lum) {
    assert(img.width == lum.width && img.height == lum.height);
    for (int y = 0; y < lum.height; ++y) {
        for (int x = 0; x < lum.width; ++x) {
            lum.at(y, x) = rgb_to_lum(img.at(y, x));
        }
    }
}

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

void compute_sobel_filter(const Matrix& mat, Matrix& grad) {
    assert(mat.width == grad.width && mat.height == grad.height);
    for (int y = 0; y < mat.height; ++y) {
        for (int x = 0; x < mat.width; ++x) {
            grad.at(y, x) = sobel_filter_at(mat, x, y);
        }
    }
}

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

// Forward energy calculation as described in "Improved Seam Carving for Video Retargeting"
// by Rubinstein, Shamir, Avidan
void compute_forward_energy(const Matrix& lum, Matrix& energy) {
    assert(lum.width == energy.width && lum.height == energy.height);
    
    // Initialize first row
    for (int x = 0; x < lum.width; ++x) {
        energy.at(0, x) = lum.at(0, x);
    }
    
    // For each row
    for (int y = 1; y < lum.height; ++y) {
        for (int x = 0; x < lum.width; ++x) {
            // Calculate costs for each possible path
            float cU = 0.0f; // Cost of going up
            float cL = 0.0f; // Cost of going left
            float cR = 0.0f; // Cost of going right
            
            // Calculate vertical cost (up)
            if (y > 0) {
                cU = std::abs(lum.at(y, x) - lum.at(y-1, x));
            }
            
            // Calculate horizontal costs (left and right)
            if (x > 0) {
                cL = std::abs(lum.at(y, x) - lum.at(y, x-1));
            }
            if (x < lum.width - 1) {
                cR = std::abs(lum.at(y, x) - lum.at(y, x+1));
            }
            
            // Find minimum cost path
            float min_cost = FLT_MAX;
            
            // Check up path
            if (y > 0) {
                min_cost = std::min(min_cost, energy.at(y-1, x) + cU);
            }
            
            // Check up-left path
            if (y > 0 && x > 0) {
                min_cost = std::min(min_cost, energy.at(y-1, x-1) + cL + cU);
            }
            
            // Check up-right path
            if (y > 0 && x < lum.width - 1) {
                min_cost = std::min(min_cost, energy.at(y-1, x+1) + cR + cU);
            }
            
            // Set energy value
            energy.at(y, x) = min_cost;
        }
    }
}

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <input> <output> [--energy <forward|backward>]\n";
    std::cerr << "  --energy: Choose energy calculation method (default: forward)\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        std::cerr << "ERROR: input and output files are required\n";
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];
    
    // Default to forward energy
    bool use_forward_energy = true;
    
    // Parse command line arguments
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--energy" && i + 1 < argc) {
            std::string energy_type = argv[i + 1];
            if (energy_type == "forward") {
                use_forward_energy = true;
            } else if (energy_type == "backward") {
                use_forward_energy = false;
            } else {
                std::cerr << "ERROR: Invalid energy type. Use 'forward' or 'backward'\n";
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

    // Create matrices for processing
    Matrix lum(width, height);
    Matrix grad(width, height);
    Matrix dp(width, height);
    std::vector<int> seam(height);

    // Compute initial luminance
    compute_luminance(img, lum);

    // Choose energy calculation method
    if (use_forward_energy) {
        std::cout << "Using forward energy calculation\n";
        compute_forward_energy(lum, grad);
    } else {
        std::cout << "Using backward energy calculation (Sobel filter)\n";
        compute_sobel_filter(lum, grad);
    }

    // Remove seams
    int seams_to_remove = width * 2 / 3;
    for (int i = 0; i < seams_to_remove; ++i) {
        // Create a new dp matrix with the current dimensions
        Matrix dp_current(img.width, img.height);
        
        compute_dynamic_programming(grad, dp_current);
        compute_seam(dp_current, seam);
        remove_seam(img, lum, grad, seam);
        
        // Update energy after seam removal based on chosen method
        if (use_forward_energy) {
            compute_forward_energy(lum, grad);
        } else {
            update_gradient(grad, lum, seam);
        }
    }

    // Save the result
    if (!stbi_write_png(output_path, img.width, img.height, 4, img.pixels.data(), img.stride * sizeof(uint32_t))) {
        std::cerr << "ERROR: could not save file " << output_path << "\n";
        return 1;
    }

    std::cout << "OK: generated " << output_path << "\n";
    return 0;
} 