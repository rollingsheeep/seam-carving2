#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <limits>

// Image class definition
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

// Matrix class definition
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

// Functions for CPU-based seam carving (needed to fall back if CUDA fails)

// Convert RGB to luminance
inline float rgb_to_lum(uint32_t rgb) {
    float r = ((rgb >> (8*0)) & 0xFF) / 255.0f;
    float g = ((rgb >> (8*1)) & 0xFF) / 255.0f;
    float b = ((rgb >> (8*2)) & 0xFF) / 255.0f;
    return 0.2126f*r + 0.7152f*g + 0.0722f*b;
}

// Compute luminance from an image
inline void compute_luminance(const Image& img, Matrix& lum) {
    for (int y = 0; y < lum.height; ++y) {
        for (int x = 0; x < lum.width; ++x) {
            lum.at(y, x) = rgb_to_lum(img.at(y, x));
        }
    }
}

// Compute Sobel filter at a specific pixel
inline float sobel_filter_at(const Matrix& mat, int cx, int cy) {
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

// Compute Sobel filter for the entire matrix
inline void compute_sobel_filter(const Matrix& mat, Matrix& grad) {
    for (int y = 0; y < mat.height; ++y) {
        for (int x = 0; x < mat.width; ++x) {
            grad.at(y, x) = sobel_filter_at(mat, x, y);
        }
    }
}

// Compute forward energy
inline void compute_forward_energy(const Matrix& lum, Matrix& energy) {
    // Initialize first row to 0
    for (int x = 0; x < lum.width; ++x) {
        energy.at(0, x) = 0.0f;
    }

    // DP forward energy computation
    for (int y = 1; y < lum.height; ++y) {
        for (int x = 0; x < lum.width; ++x) {
            // Compute neighbor costs safely with bounds
            float left = (x > 0) ? lum.at(y, x - 1) : lum.at(y, x);
            float right = (x < lum.width - 1) ? lum.at(y, x + 1) : lum.at(y, x);
            float up = lum.at(y - 1, x);

            // Cost for going straight up
            float cU = std::abs(right - left);

            // Cost for going up-left
            float cL = cU + std::abs(up - left);

            // Cost for going up-right
            float cR = cU + std::abs(up - right);

            // Get minimum previous path cost
            float min_energy = energy.at(y - 1, x) + cU;
            if (x > 0) min_energy = std::min(min_energy, energy.at(y - 1, x - 1) + cL);
            if (x < lum.width - 1) min_energy = std::min(min_energy, energy.at(y - 1, x + 1) + cR);

            energy.at(y, x) = min_energy;
        }
    }
}

// Simple hybrid energy selection
inline void compute_hybrid_energy(const Matrix& lum, Matrix& energy, float* backward_weight = nullptr, float* forward_weight = nullptr) {
    // Compute both backward and forward energy
    Matrix forwardEnergy(lum.width, lum.height);
    Matrix backwardEnergy(lum.width, lum.height);
    
    compute_sobel_filter(lum, backwardEnergy);
    compute_forward_energy(lum, forwardEnergy);
    
    // Find statistics for normalization
    float min_backward = std::numeric_limits<float>::max();
    float max_backward = -std::numeric_limits<float>::max();
    float min_forward = std::numeric_limits<float>::max();
    float max_forward = -std::numeric_limits<float>::max();
    
    // Find min/max values for both energy types
    for (int y = 0; y < lum.height; ++y) {
        for (int x = 0; x < lum.width; ++x) {
            min_backward = std::min(min_backward, backwardEnergy.at(y, x));
            max_backward = std::max(max_backward, backwardEnergy.at(y, x));
            min_forward = std::min(min_forward, forwardEnergy.at(y, x));
            max_forward = std::max(max_forward, forwardEnergy.at(y, x));
        }
    }
    
    // Create normalized energy matrices
    Matrix norm_backward(lum.width, lum.height);
    Matrix norm_forward(lum.width, lum.height);
    
    // Normalize to [0, 1] range
    float backward_range = max_backward - min_backward;
    float forward_range = max_forward - min_forward;
    
    for (int y = 0; y < lum.height; ++y) {
        for (int x = 0; x < lum.width; ++x) {
            // Avoid division by zero
            if (backward_range > 0.0001f) {
                norm_backward.at(y, x) = (backwardEnergy.at(y, x) - min_backward) / backward_range;
            } else {
                norm_backward.at(y, x) = backwardEnergy.at(y, x);
            }
            
            if (forward_range > 0.0001f) {
                norm_forward.at(y, x) = (forwardEnergy.at(y, x) - min_forward) / forward_range;
            } else {
                norm_forward.at(y, x) = forwardEnergy.at(y, x);
            }
        }
    }
    
    // Accumulate weights for reporting
    float total_backward_weight = 0.0f;
    float total_forward_weight = 0.0f;
    
    // Blend the normalized energies using the same threshold as in CUDA
    float gradient_threshold = 0.3f;
    
    for (int y = 0; y < lum.height; ++y) {
        for (int x = 0; x < lum.width; ++x) {
            float gradient = norm_backward.at(y, x);
            float mixFactor = std::min(1.0f, gradient / gradient_threshold);
            
            float backwardWeight = 1.0f - mixFactor;
            float forwardWeight = mixFactor;
            
            total_backward_weight += backwardWeight;
            total_forward_weight += forwardWeight;
            
            energy.at(y, x) = backwardWeight * norm_backward.at(y, x) + 
                             forwardWeight * norm_forward.at(y, x);
        }
    }
    
    // If pointers are provided, write back the accumulated weights
    if (backward_weight) {
        *backward_weight = total_backward_weight;
    }
    
    if (forward_weight) {
        *forward_weight = total_forward_weight;
    }
}

// Compute dynamic programming for seam finding
inline void compute_dynamic_programming(const Matrix& grad, Matrix& dp) {
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

// Compute the optimal seam
inline void compute_seam(const Matrix& dp, std::vector<int>& seam) {
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

// Remove seam from an image and associated matrices
inline void remove_seam(Image& img, Matrix& lum, Matrix& grad, const std::vector<int>& seam) {
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

// Update gradient after seam removal
inline void update_gradient(Matrix& grad, const Matrix& lum, const std::vector<int>& seam) {
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