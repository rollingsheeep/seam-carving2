#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <string>
#include <chrono>
#include <iomanip>  // For std::fixed and std::setprecision

// MPI header
#if defined(_WIN32) || defined(WIN32)
#include <mpi.h>
#else
// For Linux/Unix platforms
#include <mpi.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

// Define image and matrix classes with the necessary functionality for MPI
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

// Function prototypes for MPI implementation
void print_usage(const char* program);
int mpi_main(int argc, char** argv);

// Convert RGB to luminance using the standard coefficients
float rgb_to_lum(uint32_t rgb) {
    float r = ((rgb >> (8*0)) & 0xFF) / 255.0f;
    float g = ((rgb >> (8*1)) & 0xFF) / 255.0f;
    float b = ((rgb >> (8*2)) & 0xFF) / 255.0f;
    return 0.2126f*r + 0.7152f*g + 0.0722f*b;
}

// Compute luminance in parallel
void compute_luminance(const Image& img, Matrix& lum, int start_row, int end_row) {
    assert(img.width == lum.width && img.height == lum.height);
    for (int y = start_row; y < end_row; ++y) {
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

// Compute sobel filter in parallel for a specific range of rows
void compute_sobel_filter(const Matrix& mat, Matrix& grad, int start_row, int end_row) {
    assert(mat.width == grad.width && mat.height == grad.height);
    for (int y = start_row; y < end_row; ++y) {
        for (int x = 0; x < mat.width; ++x) {
            grad.at(y, x) = sobel_filter_at(mat, x, y);
        }
    }
}

// Forward energy calculation in parallel for a specific range of rows
void compute_forward_energy_partial(const Matrix& lum, Matrix& energy, int start_row, int end_row) {
    assert(lum.width == energy.width && lum.height == energy.height);
    int w = lum.width;
    
    // Skip first row in worker processes - it will be handled by the root process or pre-initialized
    for (int y = std::max(1, start_row); y < end_row; ++y) {
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

// Hybrid energy calculation for a range of rows
void compute_hybrid_energy_partial(const Matrix& lum, Matrix& energy, int start_row, int end_row,
                                   float* min_forward, float* max_forward,
                                   float* min_backward, float* max_backward,
                                   Matrix& forward_energy, Matrix& backward_energy) {
    assert(lum.width == energy.width && lum.height == energy.height);

    // First compute forward and backward energy for the assigned rows
    if (start_row == 0) {
        // Initialize the first row to 0 for forward energy
        for (int x = 0; x < lum.width; ++x) {
            forward_energy.at(0, x) = 0.0f;
        }
    }
    
    // Compute forward energy for assigned rows
    compute_forward_energy_partial(lum, forward_energy, start_row, end_row);
    
    // Compute backward energy (sobel) for assigned rows
    compute_sobel_filter(lum, backward_energy, start_row, end_row);
    
    // Find local min and max values for normalization
    float local_min_forward = FLT_MAX, local_max_forward = -FLT_MAX;
    float local_min_backward = FLT_MAX, local_max_backward = -FLT_MAX;
    
    for (int y = start_row; y < end_row; ++y) {
        for (int x = 0; x < lum.width; ++x) {
            float f_val = forward_energy.at(y, x);
            float b_val = backward_energy.at(y, x);
            
            local_min_forward = std::min(local_min_forward, f_val);
            local_max_forward = std::max(local_max_forward, f_val);
            local_min_backward = std::min(local_min_backward, b_val);
            local_max_backward = std::max(local_max_backward, b_val);
        }
    }
    
    // Update the global min/max values
    *min_forward = local_min_forward;
    *max_forward = local_max_forward;
    *min_backward = local_min_backward;
    *max_backward = local_max_backward;
}

// Compute dynamic programming for a range of rows
void compute_dynamic_programming_partial(const Matrix& grad, Matrix& dp, int start_row, int end_row) {
    assert(grad.width == dp.width && grad.height == dp.height);

    // Skip first row in worker processes - it will be handled by the root process or pre-initialized
    for (int y = std::max(1, start_row); y < end_row; ++y) {
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

// Compute seam - this runs on root process only
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

// Remove seam - this runs on root process only
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

// Update gradient for a range of rows
void update_gradient_partial(Matrix& grad, const Matrix& lum, const std::vector<int>& seam, int start_row, int end_row) {
    // Update gradient for pixels adjacent to the removed seam
    for (int y = start_row; y < end_row; ++y) {
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

// Print usage information
void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <input> <output> [--energy <forward|backward|hybrid>]\n";
    std::cerr << "  --energy: Choose energy calculation method (default: hybrid)\n";
}

// Global counters for hybrid energy
int hybrid_forward_count = 0;
int hybrid_backward_count = 0;

// Function to calculate row ranges and displacements for MPI_Scatterv
void calculate_row_ranges(int height, int width, int num_procs, int rank,
                         int& start_row, int& end_row,
                         std::vector<int>& recvcounts, std::vector<int>& displs) {
    int rows_per_proc = height / num_procs;
    int remainder = height % num_procs;
    
    start_row = rank * rows_per_proc + std::min(rank, remainder);
    end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);
    
    // Ensure valid row ranges
    start_row = std::min(start_row, height);
    end_row = std::min(end_row, height);
    
    // Calculate receive counts and displacements for all processes
    for (int i = 0; i < num_procs; i++) {
        int proc_start = i * rows_per_proc + std::min(i, remainder);
        int proc_end = proc_start + rows_per_proc + (i < remainder ? 1 : 0);
        proc_start = std::min(proc_start, height);
        proc_end = std::min(proc_end, height);
        recvcounts[i] = width * (proc_end - proc_start);
        displs[i] = width * proc_start;
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // Start timing (only on root process)
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Print implementation information (only on root process)
    if (rank == 0) {
        std::cout << "Using MPI implementation with " << num_procs << " processes\n";
    }
    
    if (argc < 3) {
        if (rank == 0) {
            print_usage(argv[0]);
            std::cerr << "ERROR: input and output files are required\n";
        }
        MPI_Finalize();
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];
    
    // Default to hybrid energy
    enum EnergyType { FORWARD, BACKWARD, HYBRID };
    EnergyType energy_type = HYBRID;
    
    // Parse command line arguments (on root process only)
    if (rank == 0) {
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
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return 1;
                }
                i++; // Skip the next argument
            } else {
                std::cerr << "ERROR: Unknown argument: " << arg << "\n";
                print_usage(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
    }
    
    // Broadcast energy type to all processes
    MPI_Bcast(&energy_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Variables for image and processing
    int width = 0, height = 0, channels = 0;
    Image img(0, 0);  // Will be resized after loading
    Matrix lum(0, 0);
    Matrix grad(0, 0);
    std::vector<int> seam;
    
    // Only root process loads the image
    if (rank == 0) {
        auto load_start = std::chrono::high_resolution_clock::now();
        uint32_t* pixels = (uint32_t*)stbi_load(input_path, &width, &height, &channels, 4);
        if (!pixels) {
            std::cerr << "ERROR: could not read " << input_path << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        // Create our image object
        img = Image(width, height);
        std::copy(pixels, pixels + width * height, img.pixels.begin());
        stbi_image_free(pixels);
        auto load_end = std::chrono::high_resolution_clock::now();
        auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start);
        std::cout << "Image loading time: " << load_duration.count() << " ms\n";
    }
    
    // Broadcast image dimensions to all processes
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Initialize matrices on all processes with the correct dimensions
    if (rank != 0) {
        img = Image(width, height);
    }
    lum = Matrix(width, height);
    grad = Matrix(width, height);
    
    // Calculate row ranges and displacements
    int start_row, end_row;
    std::vector<int> recvcounts(num_procs);
    std::vector<int> displs(num_procs);
    calculate_row_ranges(height, width, num_procs, rank, start_row, end_row, recvcounts, displs);
    
    // Create local image and matrix objects with only necessary rows
    int local_height = end_row - start_row;
    Image local_img(width, local_height);
    Matrix local_lum(width, local_height);
    Matrix local_grad(width, local_height);
    
    // Scatter image data using MPI_Scatterv
    if (rank == 0) {
        // Root process sends data to all processes
        for (int i = 1; i < num_procs; i++) {
            int proc_start = displs[i] / width;
            int proc_end = proc_start + recvcounts[i] / width;
            MPI_Send(&img.pixels[displs[i]], recvcounts[i], MPI_UINT32_T, i, 0, MPI_COMM_WORLD);
        }
        // Root process copies its own portion
        std::copy(img.pixels.begin(), img.pixels.begin() + recvcounts[0], local_img.pixels.begin());
    } else {
        // Other processes receive their portion
        MPI_Recv(local_img.pixels.data(), recvcounts[rank], MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Compute initial luminance in parallel
    auto luminance_start = std::chrono::high_resolution_clock::now();
    compute_luminance(local_img, local_lum, 0, local_height);
    
    // Gather luminance results using MPI_Gatherv
    std::vector<float> recv_lum(width * height);
    MPI_Gatherv(local_lum.items.data(), width * local_height, MPI_FLOAT,
                recv_lum.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);
    
    // Copy received data back to lum matrix on root process
    if (rank == 0) {
        std::copy(recv_lum.begin(), recv_lum.end(), lum.items.begin());
    }
    
    // Scatter luminance data instead of broadcasting
    MPI_Scatterv(recv_lum.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                 lum.items.data() + width * start_row, width * local_height, MPI_FLOAT,
                 0, MPI_COMM_WORLD);
    
    auto luminance_end = std::chrono::high_resolution_clock::now();
    auto luminance_duration = std::chrono::duration_cast<std::chrono::milliseconds>(luminance_end - luminance_start);
    if (rank == 0) {
        std::cout << "Luminance computation time: " << luminance_duration.count() << " ms\n";
    }
    
    // Choose energy calculation method
    auto energy_start = std::chrono::high_resolution_clock::now();
    
    // Create additional matrices for hybrid energy method if needed
    Matrix forward_energy(width, height);
    Matrix backward_energy(width, height);
    
    switch (energy_type) {
        case FORWARD: {
            if (rank == 0) std::cout << "Using forward energy calculation\n";
            if (end_row > start_row) {
                compute_forward_energy_partial(lum, grad, start_row, end_row);
            }
            
            // Gather gradient results using MPI_Gatherv
            std::vector<float> recv_grad(width * height);
            MPI_Gatherv(grad.items.data() + width * start_row, width * local_height, MPI_FLOAT,
                       recv_grad.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                       0, MPI_COMM_WORLD);
            
            // Copy received data back to grad matrix on root process
            if (rank == 0) {
                std::copy(recv_grad.begin(), recv_grad.end(), grad.items.begin());
            }
            
            // Scatter gradient data instead of broadcasting
            MPI_Scatterv(recv_grad.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                         grad.items.data() + width * start_row, width * local_height, MPI_FLOAT,
                         0, MPI_COMM_WORLD);
            break;
        }
            
        case BACKWARD: {
            if (rank == 0) std::cout << "Using backward energy calculation (Sobel filter)\n";
            if (end_row > start_row) {
                compute_sobel_filter(lum, grad, start_row, end_row);
            }
            
            // Gather gradient results using MPI_Gatherv
            std::vector<float> recv_grad(width * height);
            MPI_Gatherv(grad.items.data() + width * start_row, width * local_height, MPI_FLOAT,
                       recv_grad.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                       0, MPI_COMM_WORLD);
            
            // Copy received data back to grad matrix on root process
            if (rank == 0) {
                std::copy(recv_grad.begin(), recv_grad.end(), grad.items.begin());
            }
            
            // Scatter gradient data instead of broadcasting
            MPI_Scatterv(recv_grad.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                         grad.items.data() + width * start_row, width * local_height, MPI_FLOAT,
                         0, MPI_COMM_WORLD);
            break;
        }
            
        case HYBRID: {
            if (rank == 0) std::cout << "Using hybrid energy calculation\n";
            
            // Each process computes local statistics first
            float local_min_forward = FLT_MAX, local_max_forward = -FLT_MAX;
            float local_min_backward = FLT_MAX, local_max_backward = -FLT_MAX;
            
            // Compute local min/max values
            for (int y = start_row; y < end_row; ++y) {
                for (int x = 0; x < width; ++x) {
                    float f_val = forward_energy.at(y, x);
                    float b_val = backward_energy.at(y, x);
                    
                    local_min_forward = std::min(local_min_forward, f_val);
                    local_max_forward = std::max(local_max_forward, f_val);
                    local_min_backward = std::min(local_min_backward, b_val);
                    local_max_backward = std::max(local_max_backward, b_val);
                }
            }
            
            // Use MPI_Allreduce only once for min/max values
            float global_min_forward, global_max_forward;
            float global_min_backward, global_max_backward;
            
            MPI_Allreduce(&local_min_forward, &global_min_forward, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&local_max_forward, &global_max_forward, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(&local_min_backward, &global_min_backward, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&local_max_backward, &global_max_backward, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
            
            // Normalize using global min/max values
            float forward_range = global_max_forward - global_min_forward + 1e-6f;
            float backward_range = global_max_backward - global_min_backward + 1e-6f;
            
            // Each process normalizes its portion of the data
            for (int y = start_row; y < end_row; ++y) {
                for (int x = 0; x < width; ++x) {
                    // Normalize to [0, 1] range
                    forward_energy.at(y, x) = (forward_energy.at(y, x) - global_min_forward) / forward_range;
                    backward_energy.at(y, x) = (backward_energy.at(y, x) - global_min_backward) / backward_range;
                }
            }
            
            // Calculate local statistics for energy selection
            float local_sum_forward = 0.0f, local_sum_backward = 0.0f;
            int local_high_energy_forward = 0, local_high_energy_backward = 0;
            
            for (int y = start_row; y < end_row; ++y) {
                for (int x = 0; x < width; ++x) {
                    local_sum_forward += forward_energy.at(y, x);
                    local_sum_backward += backward_energy.at(y, x);
                }
            }
            
            // Reduce to find global sums
            float global_sum_forward, global_sum_backward;
            MPI_Allreduce(&local_sum_forward, &global_sum_forward, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&local_sum_backward, &global_sum_backward, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            
            int total_pixels = width * height;
            float avg_forward = global_sum_forward / total_pixels;
            float avg_backward = global_sum_backward / total_pixels;
            
            // Calculate local standard deviations
            float local_sum_sqr_diff_forward = 0.0f, local_sum_sqr_diff_backward = 0.0f;
            
            for (int y = start_row; y < end_row; ++y) {
                for (int x = 0; x < width; ++x) {
                    float diff_forward = forward_energy.at(y, x) - avg_forward;
                    float diff_backward = backward_energy.at(y, x) - avg_backward;
                    local_sum_sqr_diff_forward += diff_forward * diff_forward;
                    local_sum_sqr_diff_backward += diff_backward * diff_backward;
                }
            }
            
            // Reduce to find global sum of squared differences
            float global_sum_sqr_diff_forward, global_sum_sqr_diff_backward;
            MPI_Allreduce(&local_sum_sqr_diff_forward, &global_sum_sqr_diff_forward, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&local_sum_sqr_diff_backward, &global_sum_sqr_diff_backward, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            
            float std_dev_forward = std::sqrt(global_sum_sqr_diff_forward / total_pixels);
            float std_dev_backward = std::sqrt(global_sum_sqr_diff_backward / total_pixels);
            
            // Count high energy pixels in normalized space
            float forward_threshold = avg_forward + std_dev_forward;
            float backward_threshold = avg_backward + std_dev_backward;
            
            for (int y = start_row; y < end_row; ++y) {
                for (int x = 0; x < width; ++x) {
                    if (forward_energy.at(y, x) > forward_threshold) local_high_energy_forward++;
                    if (backward_energy.at(y, x) > backward_threshold) local_high_energy_backward++;
                }
            }
            
            // Reduce to find global high energy pixel counts
            int global_high_energy_forward, global_high_energy_backward;
            MPI_Allreduce(&local_high_energy_forward, &global_high_energy_forward, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&local_high_energy_backward, &global_high_energy_backward, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            
            float edge_density_forward = static_cast<float>(global_high_energy_forward) / total_pixels;
            float edge_density_backward = static_cast<float>(global_high_energy_backward) / total_pixels;
            
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
                // Root process makes the random decision to ensure consistency
                if (rank == 0) {
                    float random_factor = static_cast<float>(rand()) / RAND_MAX;
                    if (random_factor < 0.3f) use_backward = true; // 30% chance of using backward
                }
                
                // Broadcast the decision to all processes
                MPI_Bcast(&use_backward, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
            }
            
            // Update counters on root process
            if (rank == 0) {
                if (use_backward) {
                    hybrid_backward_count++;
                } else {
                    hybrid_forward_count++;
                }
            }
            
            // Use the chosen energy method
            const Matrix& chosen = use_backward ? backward_energy : forward_energy;
            for (int y = start_row; y < end_row; ++y) {
                for (int x = 0; x < width; ++x) {
                    grad.at(y, x) = chosen.at(y, x);
                }
            }
            
            // Gather gradient results using MPI_Gatherv
            std::vector<float> recv_grad(width * height);
            MPI_Gatherv(grad.items.data() + width * start_row, width * local_height, MPI_FLOAT,
                       recv_grad.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                       0, MPI_COMM_WORLD);
            
            // Copy received data back to grad matrix on root process
            if (rank == 0) {
                std::copy(recv_grad.begin(), recv_grad.end(), grad.items.begin());
            }
            
            // Scatter gradient data instead of broadcasting
            MPI_Scatterv(recv_grad.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                         grad.items.data() + width * start_row, width * local_height, MPI_FLOAT,
                         0, MPI_COMM_WORLD);
            break;
        }
    }
    
    
    auto energy_end = std::chrono::high_resolution_clock::now();
    auto energy_duration = std::chrono::duration_cast<std::chrono::milliseconds>(energy_end - energy_start);
    if (rank == 0) {
        std::cout << "Initial energy computation time: " << energy_duration.count() << " ms\n";
    }
    
    // Remove seams
    int seams_to_remove = width * 2 / 3;
    if (rank == 0) {
        std::cout << "Removing " << seams_to_remove << " seams...\n";
    }
    
    auto seam_removal_start = std::chrono::high_resolution_clock::now();
    long long total_dp_time = 0;
    long long total_seam_time = 0;
    long long total_remove_time = 0;
    long long total_update_time = 0;
    
    for (int seam_idx = 0; seam_idx < seams_to_remove; ++seam_idx) {
        // Create a new dp matrix with the current dimensions
        Matrix dp(img.width, img.height);
        
        // Time dynamic programming
        auto dp_start = std::chrono::high_resolution_clock::now();
        
        // Initialize first row on root process then broadcast
        if (rank == 0) {
            for (int x = 0; x < img.width; ++x) {
                dp.at(0, x) = grad.at(0, x);
            }
        }
        
        // Broadcast the first row to all processes
        MPI_Bcast(dp.items.data(), img.width, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        // Calculate row ranges for each process
        std::vector<int> proc_start_rows(num_procs);
        std::vector<int> proc_end_rows(num_procs);
        
        // Calculate row ranges for all processes
        int rows_per_process = img.height / num_procs;
        int remainder_rows = img.height % num_procs;
        
        for (int p = 0; p < num_procs; p++) {
            int proc_start = p * rows_per_process + (p < remainder_rows ? p : remainder_rows);
            int proc_end = proc_start + rows_per_process + (p < remainder_rows ? 1 : 0);
            proc_start = std::min(proc_start, img.height);
            proc_end = std::min(proc_end, img.height);
            proc_start_rows[p] = proc_start;
            proc_end_rows[p] = proc_end;
        }
        
        // Each process computes its portion of the dp matrix
        for (int y = 1; y < img.height; ++y) {
            // Determine which process owns this row
            int owner_proc = -1;
            for (int p = 0; p < num_procs; p++) {
                if (y >= proc_start_rows[p] && y < proc_end_rows[p]) {
                    owner_proc = p;
                    break;
                }
            }
            
            // Only the process that owns this row computes it
            if (rank == owner_proc) {
                for (int x = 0; x < img.width; ++x) {
                    float min_prev = dp.at(y - 1, x);
                    if (x > 0) {
                        min_prev = std::min(min_prev, dp.at(y - 1, x - 1));
                    }
                    if (x < img.width - 1) {
                        min_prev = std::min(min_prev, dp.at(y - 1, x + 1));
                    }
                    dp.at(y, x) = grad.at(y, x) + min_prev;
                }
            }
            
            // Broadcast the computed row to all processes
            MPI_Bcast(&dp.items[y * dp.stride], img.width, MPI_FLOAT, owner_proc, MPI_COMM_WORLD);
        }
        
        auto dp_end = std::chrono::high_resolution_clock::now();
        if (rank == 0) {
            total_dp_time += std::chrono::duration_cast<std::chrono::microseconds>(dp_end - dp_start).count();
        }
        
        // Time seam computation (root process only)
        auto seam_start = std::chrono::high_resolution_clock::now();
        if (rank == 0) {
            compute_seam(dp, seam);
        }
        auto seam_end = std::chrono::high_resolution_clock::now();
        if (rank == 0) {
            total_seam_time += std::chrono::duration_cast<std::chrono::microseconds>(seam_end - seam_start).count();
        }
        
        // Ensure all processes have the seam vector properly sized
        seam.resize(img.height);
        
        // Broadcast seam to all processes
        MPI_Bcast(seam.data(), img.height, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Time seam removal (root process only)
        auto remove_start = std::chrono::high_resolution_clock::now();
        if (rank == 0) {
            remove_seam(img, lum, grad, seam);
        }
        auto remove_end = std::chrono::high_resolution_clock::now();
        if (rank == 0) {
            total_remove_time += std::chrono::duration_cast<std::chrono::microseconds>(remove_end - remove_start).count();
        }
        
        // First broadcast the new dimensions
        int new_width;
        if (rank == 0) {
            new_width = img.width;
        }
        MPI_Bcast(&new_width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Update width variable for all processes
        width = new_width;
        
        // All processes resize their containers
        if (rank != 0) {
            img.width = new_width;
            img.stride = new_width;
            lum.width = new_width;
            lum.stride = new_width;
            grad.width = new_width;
            grad.stride = new_width;
            
            img.pixels.resize(new_width * img.height);
            lum.items.resize(new_width * lum.height);
            grad.items.resize(new_width * grad.height);
        }
        
        // Synchronize all processes after resizing
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Now broadcast the actual data
        MPI_Bcast(img.pixels.data(), new_width * img.height, MPI_UINT32_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(lum.items.data(), new_width * lum.height, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(grad.items.data(), new_width * grad.height, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        // Recalculate row assignments as the matrix size changed
        rows_per_process = img.height / num_procs;
        remainder_rows = img.height % num_procs;
        start_row = rank * rows_per_process + (rank < remainder_rows ? rank : remainder_rows);
        end_row = start_row + rows_per_process + (rank < remainder_rows ? 1 : 0);
        
        // Ensure row bounds are valid
        start_row = std::min(start_row, img.height);
        end_row = std::min(end_row, img.height);
        
        // Update receive counts and displacements for new dimensions
        for (int proc = 0; proc < num_procs; proc++) {
            int proc_start = proc * rows_per_process + (proc < remainder_rows ? proc : remainder_rows);
            int proc_end = proc_start + rows_per_process + (proc < remainder_rows ? 1 : 0);
            proc_start = std::min(proc_start, img.height);
            proc_end = std::min(proc_end, img.height);
            recvcounts[proc] = width * (proc_end - proc_start);
            displs[proc] = width * proc_start;
        }
        
        // Time energy update
        auto update_start = std::chrono::high_resolution_clock::now();
        
        // Synchronize before energy update
        MPI_Barrier(MPI_COMM_WORLD);
        
        switch (energy_type) {
            case FORWARD: {
                compute_forward_energy_partial(lum, grad, start_row, end_row);
                // Create temporary buffer for gradient
                std::vector<float> temp_grad(width * (end_row - start_row));
                std::vector<float> recv_grad(width * height);
                
                // Copy local portion to temporary buffer
                std::copy(grad.items.data() + start_row * width,
                         grad.items.data() + end_row * width,
                         temp_grad.begin());
                
                // Gather gradient values to root process
                MPI_Gatherv(temp_grad.data(), width * (end_row - start_row), MPI_FLOAT,
                           recv_grad.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                           0, MPI_COMM_WORLD);
                
                // Copy received data back to grad matrix on root process
                if (rank == 0) {
                    std::copy(recv_grad.begin(), recv_grad.end(), grad.items.begin());
                }
                
                // Broadcast complete gradient matrix to all processes
                MPI_Bcast(grad.items.data(), width * height, MPI_FLOAT, 0, MPI_COMM_WORLD);
                break;
            }
                
            case BACKWARD: {
                update_gradient_partial(grad, lum, seam, start_row, end_row);
                // Create temporary buffer for gradient
                std::vector<float> temp_grad(width * (end_row - start_row));
                std::vector<float> recv_grad(width * height);
                
                // Copy local portion to temporary buffer
                std::copy(grad.items.data() + start_row * width,
                         grad.items.data() + end_row * width,
                         temp_grad.begin());
                
                // Gather gradient values to root process
                MPI_Gatherv(temp_grad.data(), width * (end_row - start_row), MPI_FLOAT,
                           recv_grad.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                           0, MPI_COMM_WORLD);
                
                // Copy received data back to grad matrix on root process
                if (rank == 0) {
                    std::copy(recv_grad.begin(), recv_grad.end(), grad.items.begin());
                }
                
                // Broadcast complete gradient matrix to all processes
                MPI_Bcast(grad.items.data(), width * height, MPI_FLOAT, 0, MPI_COMM_WORLD);
                break;
            }
                
            case HYBRID: {
                // For hybrid mode, we need to decide which method to use for each iteration
                // Reuse the matrices created earlier
                forward_energy = Matrix(img.width, img.height);
                backward_energy = Matrix(img.width, img.height);
                
                // Synchronize before computing energies
                MPI_Barrier(MPI_COMM_WORLD);
                
                // Each process computes local min/max for its portion
                float local_min_forward = FLT_MAX, local_max_forward = -FLT_MAX;
                float local_min_backward = FLT_MAX, local_max_backward = -FLT_MAX;
                
                compute_hybrid_energy_partial(lum, grad, start_row, end_row, 
                                            &local_min_forward, &local_max_forward,
                                            &local_min_backward, &local_max_backward,
                                            forward_energy, backward_energy);
                
                // Create temporary buffers for gathering
                std::vector<float> temp_forward(width * (end_row - start_row));
                std::vector<float> temp_backward(width * (end_row - start_row));
                std::vector<float> recv_forward(width * height);
                std::vector<float> recv_backward(width * height);
                
                // Copy local portions to temporary buffers
                std::copy(forward_energy.items.data() + start_row * width,
                         forward_energy.items.data() + end_row * width,
                         temp_forward.begin());
                std::copy(backward_energy.items.data() + start_row * width,
                         backward_energy.items.data() + end_row * width,
                         temp_backward.begin());
                
                // Gather forward and backward energies to root process
                MPI_Gatherv(temp_forward.data(), width * (end_row - start_row), MPI_FLOAT,
                           recv_forward.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                           0, MPI_COMM_WORLD);
                
                MPI_Gatherv(temp_backward.data(), width * (end_row - start_row), MPI_FLOAT,
                           recv_backward.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                           0, MPI_COMM_WORLD);
                
                // Copy received data back to matrices on root process
                if (rank == 0) {
                    std::copy(recv_forward.begin(), recv_forward.end(), forward_energy.items.begin());
                    std::copy(recv_backward.begin(), recv_backward.end(), backward_energy.items.begin());
                }
                
                // Broadcast complete matrices to all processes
                MPI_Bcast(forward_energy.items.data(), width * height, MPI_FLOAT, 0, MPI_COMM_WORLD);
                MPI_Bcast(backward_energy.items.data(), width * height, MPI_FLOAT, 0, MPI_COMM_WORLD);
                
                // Declare global min/max variables
                float global_min_forward, global_max_forward;
                float global_min_backward, global_max_backward;
                
                // Reduce to find global min/max values
                MPI_Allreduce(&local_min_forward, &global_min_forward, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
                MPI_Allreduce(&local_max_forward, &global_max_forward, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
                MPI_Allreduce(&local_min_backward, &global_min_backward, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
                MPI_Allreduce(&local_max_backward, &global_max_backward, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
                
                // Create normalized versions of both energy types
                Matrix norm_forward_energy(width, height);
                Matrix norm_backward_energy(width, height);
                
                float forward_range = global_max_forward - global_min_forward + 1e-6f;
                float backward_range = global_max_backward - global_min_backward + 1e-6f;
                
                // Each process normalizes its portion of the data
                for (int y = start_row; y < end_row; ++y) {
                    for (int x = 0; x < width; ++x) {
                        // Normalize to [0, 1] range
                        norm_forward_energy.at(y, x) = (forward_energy.at(y, x) - global_min_forward) / forward_range;
                        norm_backward_energy.at(y, x) = (backward_energy.at(y, x) - global_min_backward) / backward_range;
                    }
                }
                
                // Create temporary buffers for normalized energies
                std::vector<float> temp_norm_forward(width * (end_row - start_row));
                std::vector<float> temp_norm_backward(width * (end_row - start_row));
                std::vector<float> recv_norm_forward(width * height);
                std::vector<float> recv_norm_backward(width * height);
                
                // Copy local portions to temporary buffers
                std::copy(norm_forward_energy.items.data() + start_row * width,
                         norm_forward_energy.items.data() + end_row * width,
                         temp_norm_forward.begin());
                std::copy(norm_backward_energy.items.data() + start_row * width,
                         norm_backward_energy.items.data() + end_row * width,
                         temp_norm_backward.begin());
                
                // Gather normalized energies to root process
                MPI_Gatherv(temp_norm_forward.data(), width * (end_row - start_row), MPI_FLOAT,
                           recv_norm_forward.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                           0, MPI_COMM_WORLD);
                
                MPI_Gatherv(temp_norm_backward.data(), width * (end_row - start_row), MPI_FLOAT,
                           recv_norm_backward.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                           0, MPI_COMM_WORLD);
                
                // Copy received data back to matrices on root process
                if (rank == 0) {
                    std::copy(recv_norm_forward.begin(), recv_norm_forward.end(), norm_forward_energy.items.begin());
                    std::copy(recv_norm_backward.begin(), recv_norm_backward.end(), norm_backward_energy.items.begin());
                }
                
                // Broadcast complete normalized matrices to all processes
                MPI_Bcast(norm_forward_energy.items.data(), width * height, MPI_FLOAT, 0, MPI_COMM_WORLD);
                MPI_Bcast(norm_backward_energy.items.data(), width * height, MPI_FLOAT, 0, MPI_COMM_WORLD);
                
                // Calculate statistics on normalized energies
                float local_sum_forward = 0.0f, local_sum_backward = 0.0f;
                int local_high_energy_forward = 0, local_high_energy_backward = 0;
                
                for (int y = start_row; y < end_row; ++y) {
                    for (int x = 0; x < width; ++x) {
                        local_sum_forward += norm_forward_energy.at(y, x);
                        local_sum_backward += norm_backward_energy.at(y, x);
                    }
                }
                
                // Reduce to find global sums
                float global_sum_forward, global_sum_backward;
                MPI_Allreduce(&local_sum_forward, &global_sum_forward, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&local_sum_backward, &global_sum_backward, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                
                int total_pixels = width * height;
                float avg_forward = global_sum_forward / total_pixels;
                float avg_backward = global_sum_backward / total_pixels;
                
                // Calculate standard deviations
                float local_sum_sqr_diff_forward = 0.0f, local_sum_sqr_diff_backward = 0.0f;
                
                for (int y = start_row; y < end_row; ++y) {
                    for (int x = 0; x < width; ++x) {
                        float diff_forward = norm_forward_energy.at(y, x) - avg_forward;
                        float diff_backward = norm_backward_energy.at(y, x) - avg_backward;
                        local_sum_sqr_diff_forward += diff_forward * diff_forward;
                        local_sum_sqr_diff_backward += diff_backward * diff_backward;
                    }
                }
                
                // Reduce to find global sum of squared differences
                float global_sum_sqr_diff_forward, global_sum_sqr_diff_backward;
                MPI_Allreduce(&local_sum_sqr_diff_forward, &global_sum_sqr_diff_forward, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&local_sum_sqr_diff_backward, &global_sum_sqr_diff_backward, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                
                float std_dev_forward = std::sqrt(global_sum_sqr_diff_forward / total_pixels);
                float std_dev_backward = std::sqrt(global_sum_sqr_diff_backward / total_pixels);
                
                // Count high energy pixels in normalized space
                float forward_threshold = avg_forward + std_dev_forward;
                float backward_threshold = avg_backward + std_dev_backward;
                
                for (int y = start_row; y < end_row; ++y) {
                    for (int x = 0; x < width; ++x) {
                        if (norm_forward_energy.at(y, x) > forward_threshold) local_high_energy_forward++;
                        if (norm_backward_energy.at(y, x) > backward_threshold) local_high_energy_backward++;
                    }
                }
                
                // Reduce to find global high energy pixel counts
                int global_high_energy_forward, global_high_energy_backward;
                MPI_Allreduce(&local_high_energy_forward, &global_high_energy_forward, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&local_high_energy_backward, &global_high_energy_backward, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                
                float edge_density_forward = static_cast<float>(global_high_energy_forward) / total_pixels;
                float edge_density_backward = static_cast<float>(global_high_energy_backward) / total_pixels;
                
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
                    // Root process makes the random decision to ensure consistency
                    if (rank == 0) {
                        float random_factor = static_cast<float>(rand()) / RAND_MAX;
                        if (random_factor < 0.3f) use_backward = true; // 30% chance of using backward
                    }
                    
                    // Broadcast the decision to all processes
                    MPI_Bcast(&use_backward, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
                }
                
                // Update counters on root process
                if (rank == 0) {
                    if (use_backward) {
                        hybrid_backward_count++;
                    } else {
                        hybrid_forward_count++;
                    }
                }
                
                // Use the chosen energy method
                const Matrix& chosen = use_backward ? backward_energy : forward_energy;
                for (int y = start_row; y < end_row; ++y) {
                    for (int x = 0; x < width; ++x) {
                        grad.at(y, x) = chosen.at(y, x);
                    }
                }
                
                // Create temporary buffer for gradient
                std::vector<float> temp_grad(width * (end_row - start_row));
                std::vector<float> recv_grad(width * height);
                
                // Copy local portion to temporary buffer
                std::copy(grad.items.data() + start_row * width,
                         grad.items.data() + end_row * width,
                         temp_grad.begin());
                
                // Gather gradient values to root process
                MPI_Gatherv(temp_grad.data(), width * (end_row - start_row), MPI_FLOAT,
                           recv_grad.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                           0, MPI_COMM_WORLD);
                
                // Copy received data back to grad matrix on root process
                if (rank == 0) {
                    std::copy(recv_grad.begin(), recv_grad.end(), grad.items.begin());
                }
                
                // Broadcast complete gradient matrix to all processes
                MPI_Bcast(grad.items.data(), width * height, MPI_FLOAT, 0, MPI_COMM_WORLD);
                break;
            }
        }
        
        auto update_end = std::chrono::high_resolution_clock::now();
        if (rank == 0) {
            total_update_time += std::chrono::duration_cast<std::chrono::microseconds>(update_end - update_start).count();
        }
        
        // Print progress every 10% of seams (root process only)
        if (rank == 0 && ((seam_idx + 1) % (seams_to_remove / 10) == 0 || seam_idx == seams_to_remove - 1)) {
            int progress = ((seam_idx + 1) * 100) / seams_to_remove;
            std::cout << "Progress: " << progress << "% complete\n";
        }
    }
    
    auto seam_removal_end = std::chrono::high_resolution_clock::now();
    auto seam_removal_duration = std::chrono::duration_cast<std::chrono::milliseconds>(seam_removal_end - seam_removal_start);
    
    // Print detailed timing information (root process only)
    if (rank == 0) {
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
        
        // Sanity check before saving
        if (img.width == 0 || img.height == 0 || img.pixels.empty()) {
            std::cerr << "ERROR: Invalid image state before saving. Possibly due to earlier seam removal or loading failure.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        if (!stbi_write_png(output_path, img.width, img.height, 4, img.pixels.data(), img.stride * sizeof(uint32_t))) {
            std::cerr << "ERROR: could not save file " << output_path << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
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
    }
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
} 