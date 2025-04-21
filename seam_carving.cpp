#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>

#define CENTER_BIAS 0.05  // increase to push seam away from edges

// Constants
const double ENERGY_MASK_CONST = 100000.0;  // Large energy value for protective masking
const int MASK_THRESHOLD = 10;              // Minimum pixel intensity for binary mask
const bool USE_FORWARD_ENERGY = true;       // If true, use forward energy algorithm

// Function declarations
cv::Mat resize(const cv::Mat& image, int width);
cv::Mat rotate_image(const cv::Mat& image, bool clockwise);
cv::Mat backward_energy(const cv::Mat& image);
cv::Mat forward_energy(const cv::Mat& image);
cv::Mat add_seam(const cv::Mat& image, const std::vector<int>& seam_idx);
cv::Mat add_seam_grayscale(const cv::Mat& image, const std::vector<int>& seam_idx);
cv::Mat remove_seam(const cv::Mat& image, const cv::Mat& boolmask);
cv::Mat remove_seam_grayscale(const cv::Mat& image, const cv::Mat& boolmask);
std::pair<std::vector<int>, cv::Mat> get_minimum_seam(const cv::Mat& image, const cv::Mat& mask, const cv::Mat& remove_mask);
cv::Mat reduce_width(const cv::Mat& image, int num_seams, const cv::Mat& mask, bool visualize);
cv::Mat reduce_height(const cv::Mat& image, int num_seams, const cv::Mat& mask, bool visualize);
cv::Mat enlarge_width(const cv::Mat& image, int num_seams, const cv::Mat& mask, bool visualize);
cv::Mat enlarge_height(const cv::Mat& image, int num_seams, const cv::Mat& mask, bool visualize);
cv::Mat seam_carve(const cv::Mat& image, int dy, int dx, const cv::Mat& mask, bool visualize);
cv::Mat object_removal(const cv::Mat& image, const cv::Mat& rmask, const cv::Mat& mask, bool visualize);
int test_seam_carving(); // Declaration for test function

// Utility function: resize
cv::Mat resize(const cv::Mat& image, int width) {
    int h = image.rows;
    int w = image.cols;
    int new_h = static_cast<int>(h * width / static_cast<float>(w));
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(width, new_h));
    return resized;
}

// Utility function: rotate_image
cv::Mat rotate_image(const cv::Mat& image, bool clockwise) {
    cv::Mat rotated;
    int rotation_code = clockwise ? cv::ROTATE_90_CLOCKWISE : cv::ROTATE_90_COUNTERCLOCKWISE;
    cv::rotate(image, rotated, rotation_code);
    return rotated;
}

// Energy function: backward_energy
cv::Mat backward_energy(const cv::Mat& image) {
    // Convert to grayscale if the image is color
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Convert to float for calculations
    cv::Mat float_gray;
    gray.convertTo(float_gray, CV_64F, 1.0/255.0);
    
    // Calculate gradients using Sobel with proper scaling
    cv::Mat grad_x, grad_y;
    cv::Sobel(float_gray, grad_x, CV_64F, 1, 0, 3);
    cv::Sobel(float_gray, grad_y, CV_64F, 0, 1, 3);
    
    // Calculate gradient magnitude
    cv::Mat magnitude;
    cv::magnitude(grad_x, grad_y, magnitude);
    
    // Add edge detection using Laplacian for better feature preservation
    cv::Mat laplacian;
    cv::Laplacian(float_gray, laplacian, CV_64F);
    cv::Mat zeros = cv::Mat::zeros(laplacian.size(), CV_64F);
    cv::absdiff(laplacian, zeros, laplacian);
    
    // Combine gradient magnitude with Laplacian
    cv::Mat energy = 0.5 * magnitude + 0.5 * laplacian;
    
    // Normalize energy to [0,1] range
    cv::normalize(energy, energy, 0, 1, cv::NORM_MINMAX);
    
    // Apply non-linear scaling to emphasize strong edges
    cv::pow(energy, 2.0, energy);
    
    return energy;
}

// Energy function: forward_energy
cv::Mat forward_energy(const cv::Mat& image) {
    // Convert to grayscale if the image is color
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Convert to float for calculations
    cv::Mat float_gray;
    gray.convertTo(float_gray, CV_64F);
    
    int h = float_gray.rows;
    int w = float_gray.cols;
    
    // Create matrices for energy and cumulative energy
    cv::Mat energy = cv::Mat::zeros(h, w, CV_64F);
    cv::Mat m = cv::Mat::zeros(h, w, CV_64F);
    
    // Create shifted versions of the image for edge detection
    cv::Mat U, L, R;
    cv::Mat kernel_shift_up = (cv::Mat_<double>(1, 3) << 0, 0, 1);
    cv::Mat kernel_shift_left = (cv::Mat_<double>(1, 3) << 0, 1, 0);
    cv::Mat kernel_shift_right = (cv::Mat_<double>(1, 3) << 0, 0, 1);
    
    cv::filter2D(float_gray, U, -1, kernel_shift_up, cv::Point(-1, 0), 0, cv::BORDER_REPLICATE);
    cv::filter2D(float_gray, L, -1, kernel_shift_left, cv::Point(-1, 0), 0, cv::BORDER_REPLICATE);
    cv::filter2D(float_gray, R, -1, kernel_shift_right, cv::Point(1, 0), 0, cv::BORDER_REPLICATE);
    
    // Calculate costs
    cv::Mat cU, cL, cR;
    cv::absdiff(R, L, cU);
    
    cv::Mat diff_UL;
    cv::absdiff(U, L, diff_UL);
    cv::add(cU, diff_UL, cL);
    
    cv::Mat diff_UR;
    cv::absdiff(U, R, diff_UR);
    cv::add(cU, diff_UR, cR);
    
    // First row is just the costs
    cU.row(0).copyTo(m.row(0));
    
    // Dynamic programming to find minimum energy path
    for (int i = 1; i < h; i++) {
        for (int j = 0; j < w; j++) {
            double min_val = m.at<double>(i-1, j);
            int min_idx = j;
            
            if (j > 0 && m.at<double>(i-1, j-1) < min_val) {
                min_val = m.at<double>(i-1, j-1);
                min_idx = j-1;
            }
            
            if (j < w-1 && m.at<double>(i-1, j+1) < min_val) {
                min_val = m.at<double>(i-1, j+1);
                min_idx = j+1;
            }
            
            // Set the cumulative energy
            if (min_idx == j) {
                m.at<double>(i, j) = min_val + cU.at<double>(i, j);
                energy.at<double>(i, j) = cU.at<double>(i, j);
            } else if (min_idx == j-1) {
                m.at<double>(i, j) = min_val + cL.at<double>(i, j);
                energy.at<double>(i, j) = cL.at<double>(i, j);
            } else {
                m.at<double>(i, j) = min_val + cR.at<double>(i, j);
                energy.at<double>(i, j) = cR.at<double>(i, j);
            }
        }
    }
    
    return energy;
}

// Seam helper function: add_seam
cv::Mat add_seam(const cv::Mat& image, const std::vector<int>& seam_idx) {
    int h = image.rows;
    int w = image.cols;
    int channels = image.channels();
    
    cv::Mat output(h, w + 1, image.type());
    
    for (int row = 0; row < h; row++) {
        int col = seam_idx[row];
        
        for (int c = 0; c <= col; c++) {
            output.at<cv::Vec3b>(row, c) = image.at<cv::Vec3b>(row, c);
        }
        
        // Insert seam pixel
        cv::Vec3b left_pixel, right_pixel;

        if (col == 0) {
            left_pixel = image.at<cv::Vec3b>(row, col);
            right_pixel = image.at<cv::Vec3b>(row, col + 1);
        } else if (col == w - 1) {
            left_pixel = image.at<cv::Vec3b>(row, col - 1);
            right_pixel = image.at<cv::Vec3b>(row, col);
        } else {
            left_pixel = image.at<cv::Vec3b>(row, col - 1);
            right_pixel = image.at<cv::Vec3b>(row, col + 1);
        }

        // Weighted average based on pixel similarity to avoid visible seams
        double left_energy = cv::norm(left_pixel - image.at<cv::Vec3b>(row, col));
        double right_energy = cv::norm(right_pixel - image.at<cv::Vec3b>(row, col));
            double total_energy = left_energy + right_energy;
            
        cv::Vec3b new_pixel;
            if (total_energy > 0) {
            for (int ch = 0; ch < 3; ch++) {
                new_pixel[ch] = static_cast<uchar>(
                    (left_pixel[ch] * right_energy + right_pixel[ch] * left_energy) / total_energy
                );
            }
            } else {
            new_pixel = image.at<cv::Vec3b>(row, col);
        }
        
        // Seam inserted between col and col+1
        output.at<cv::Vec3b>(row, col + 1) = new_pixel;

        // Shift remaining pixels to the right
        for (int c = col + 1; c < w; c++) {
            output.at<cv::Vec3b>(row, c + 1) = image.at<cv::Vec3b>(row, c);
        }
    }
    
    return output;
}

// Seam helper function: add_seam_grayscale
cv::Mat add_seam_grayscale(const cv::Mat& image, const std::vector<int>& seam_idx) {
    std::cout << "add_seam_grayscale: Starting function" << std::endl;
    
    int h = image.rows;
    int w = image.cols;
    
    std::cout << "add_seam_grayscale: Image dimensions: " << h << "x" << w << std::endl;
    std::cout << "add_seam_grayscale: Seam indices size: " << seam_idx.size() << std::endl;
    
    // Validate seam indices
    if (seam_idx.size() != h) {
        std::cout << "add_seam_grayscale: Error - seam indices size (" << seam_idx.size() 
                  << ") does not match image height (" << h << ")" << std::endl;
        return image.clone();
    }
    
    // Check if any seam index is out of bounds
    for (int i = 0; i < h; i++) {
        if (seam_idx[i] < 0 || seam_idx[i] >= w) {
            std::cout << "add_seam_grayscale: Error - seam index at row " << i 
                      << " is out of bounds: " << seam_idx[i] << std::endl;
            return image.clone();
        }
    }
    
    // Create output image with one more column
    cv::Mat output(h, w + 1, image.type());
    std::cout << "add_seam_grayscale: Created output image with size: " << output.size() << std::endl;
    
    // For each row
    for (int row = 0; row < h; row++) {
        int col = seam_idx[row];
        
        // Copy pixels before the seam
        for (int c = 0; c < col; c++) {
            output.at<uchar>(row, c) = image.at<uchar>(row, c);
        }
        
        // Add the seam by averaging pixels with improved blending
        if (col == 0) {
            // If seam is at the left edge, blend with right neighbor
            uchar right = image.at<uchar>(row, col + 1);
            output.at<uchar>(row, col) = image.at<uchar>(row, col);
            output.at<uchar>(row, col + 1) = (image.at<uchar>(row, col) + right) / 2;
        } else if (col == w - 1) {
            // If seam is at the right edge, blend with left neighbor
            uchar left = image.at<uchar>(row, col - 1);
            output.at<uchar>(row, col) = (left + image.at<uchar>(row, col)) / 2;
            output.at<uchar>(row, col + 1) = image.at<uchar>(row, col);
        } else {
            // If seam is in the middle, use weighted average of neighbors
            uchar left = image.at<uchar>(row, col - 1);
            uchar center = image.at<uchar>(row, col);
            uchar right = image.at<uchar>(row, col + 1);
            
            // Calculate energy-based weights
            double left_energy = std::abs(left - center);
            double right_energy = std::abs(right - center);
            double total_energy = left_energy + right_energy;
            
            if (total_energy > 0) {
                double left_weight = right_energy / total_energy;
                double right_weight = left_energy / total_energy;
                output.at<uchar>(row, col) = static_cast<uchar>(
                    left * left_weight + right * right_weight
                );
            } else {
                output.at<uchar>(row, col) = (left + right) / 2;
            }
        }
        
        // Copy pixels after the seam
        for (int c = col + 1; c < w; c++) {
            output.at<uchar>(row, c + 1) = image.at<uchar>(row, c);
        }
    }
    
    std::cout << "add_seam_grayscale: Function completed successfully" << std::endl;
    return output;
}

// Seam helper function: remove_seam
cv::Mat remove_seam(const cv::Mat& image, const cv::Mat& boolmask) {
    int h = image.rows;
    int w = image.cols;
    int channels = image.channels();
    
    // Validate input dimensions
    if (h != boolmask.rows || w != boolmask.cols) {
        std::cout << "remove_seam: Error - mask dimensions don't match image" << std::endl;
        throw std::runtime_error("Mask dimensions don't match image");
    }
    
    cv::Mat output(h, w - 1, image.type());
    
    for (int row = 0; row < h; row++) {
        const uchar* mask_ptr = boolmask.ptr<uchar>(row);
        int seam_pos = -1;
        
        // Find seam position (marked by 0 in the mask)
        for (int col = 0; col < w; col++) {
            if (mask_ptr[col] == 0) {
                seam_pos = col;
                break;
            }
        }
        
        if (seam_pos == -1) {
            std::cout << "remove_seam: Error - no seam found in row " << row << std::endl;
            throw std::runtime_error("No seam found in row");
        }

        if (channels == 3) {
            const cv::Vec3b* in_ptr = image.ptr<cv::Vec3b>(row);
            cv::Vec3b* out_ptr = output.ptr<cv::Vec3b>(row);
            
            // Copy pixels before the seam
            std::copy(in_ptr, in_ptr + seam_pos, out_ptr);
            
            // Copy pixels after the seam
            std::copy(in_ptr + seam_pos + 1, in_ptr + w, out_ptr + seam_pos);
        } else {
            const uchar* in_ptr = image.ptr<uchar>(row);
            uchar* out_ptr = output.ptr<uchar>(row);
            
            // Copy pixels before the seam
            std::copy(in_ptr, in_ptr + seam_pos, out_ptr);
            
            // Copy pixels after the seam
            std::copy(in_ptr + seam_pos + 1, in_ptr + w, out_ptr + seam_pos);
        }
    }

    return output;
}

cv::Mat remove_seam_grayscale(const cv::Mat& image, const cv::Mat& boolmask) {
    int h = image.rows;
    int w = image.cols;
    
    cv::Mat output(h, w - 1, image.type());
    
    for (int row = 0; row < h; row++) {
        const uchar* mask_ptr = boolmask.ptr<uchar>(row);
        const uchar* in_ptr = image.ptr<uchar>(row);
        uchar* out_ptr = output.ptr<uchar>(row);
        
        int out_col = 0;
        for (int in_col = 0; in_col < w; in_col++) {
            if (mask_ptr[in_col] > 0) {
                out_ptr[out_col++] = in_ptr[in_col];
            }
        }
    }
    
    return output;
}

// Seam helper function: get_minimum_seam
std::pair<std::vector<int>, cv::Mat> get_minimum_seam(const cv::Mat& image, const cv::Mat& mask, const cv::Mat& remove_mask) {
    std::cout << "get_minimum_seam: Processing image of size " << image.size() << std::endl;
    
    // Calculate energy map
    cv::Mat energy;
    if (USE_FORWARD_ENERGY) {
        energy = forward_energy(image);
    } else {
        energy = backward_energy(image);
    }
    
    // Apply mask for object removal if provided
    if (!remove_mask.empty()) {
        cv::Mat masked_energy;
        cv::bitwise_and(energy, remove_mask, masked_energy);
        energy = masked_energy;
    }
    
    int rows = energy.rows;
    int cols = energy.cols;
    
    // Initialize backtracking matrix and cumulative energy matrix
    cv::Mat cumulative_energy = energy.clone();
    cv::Mat backtrack = cv::Mat::zeros(rows, cols, CV_32S);
    
    // Fill cumulative energy matrix using dynamic programming
    for (int i = 1; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float min_energy = std::numeric_limits<float>::max();
            int min_idx = j;
            
            // Check all possible paths (left, up, right)
            for (int k = -1; k <= 1; k++) {
                int prev_j = j + k;
                if (prev_j >= 0 && prev_j < cols) {
                    float energy = cumulative_energy.at<float>(i-1, prev_j);
                    // Add a small penalty for diagonal paths to prefer vertical seams
                    if (k != 0) energy += 0.1f;
                    if (energy < min_energy) {
                        min_energy = energy;
                        min_idx = prev_j;
                    }
                }
            }
            
            cumulative_energy.at<float>(i, j) += min_energy;
            backtrack.at<int>(i, j) = min_idx;
        }
    }
    
    // Find the minimum energy path in the last row
    float min_energy = std::numeric_limits<float>::max();
    int min_idx = 0;
    
    // Prefer seams closer to the center of the image
    int center = cols / 2;
    for (int j = 0; j < cols; j++) {
        float energy = cumulative_energy.at<float>(rows-1, j);
        // Add a small penalty based on distance from center
        float center_penalty = CENTER_BIAS * std::abs(j - center);
        energy += center_penalty;
        
        if (energy < min_energy) {
            min_energy = energy;
            min_idx = j;
        }
    }
    
    // Backtrack to find the seam
    std::vector<int> seam(rows);
    seam[rows-1] = min_idx;
    
    for (int i = rows-2; i >= 0; i--) {
        seam[i] = backtrack.at<int>(i+1, seam[i+1]);
    }
    
    // Create boolean mask for the seam
    cv::Mat boolmask = cv::Mat::zeros(rows, cols, CV_8U);
    for (int i = 0; i < rows; i++) {
        boolmask.at<uchar>(i, seam[i]) = 1;
    }
    
    std::cout << "get_minimum_seam: Seam created successfully" << std::endl;
    return {seam, boolmask};
}

// Function to reduce width of an image by removing vertical seams
cv::Mat reduce_width(const cv::Mat& image, int num_seams, const cv::Mat& mask, bool visualize) {
    std::cout << "reduce_width: Starting width reduction by " << num_seams << " seams" << std::endl;
    
    // Validate input image
    if (image.empty()) {
        std::cout << "reduce_width: Error: Input image is empty" << std::endl;
        throw std::runtime_error("Input image is empty");
    }
    
    std::cout << "reduce_width: Input image size: " << image.size() << ", channels: " << image.channels() << std::endl;
    
    // Make a copy of the input image
    cv::Mat output = image.clone();
    cv::Mat current_mask = mask.empty() ? cv::Mat() : mask.clone();
    
    // Validate input
    if (num_seams <= 0) {
        std::cout << "reduce_width: No seams to remove, returning original image" << std::endl;
        return output;
    }
    
    if (num_seams >= output.cols) {
        std::cout << "reduce_width: Error: Cannot remove more seams than image width" << std::endl;
        throw std::runtime_error("Cannot remove more seams than image width");
    }
    
    // Process seams in smaller batches to avoid distortion
    const int BATCH_SIZE = 10; // Process 10 seams at a time
    int remaining_seams = num_seams;
    
    while (remaining_seams > 0) {
        int batch_size = std::min(BATCH_SIZE, remaining_seams);
        std::cout << "reduce_width: Processing batch of " << batch_size << " seams (remaining: " << remaining_seams << ")" << std::endl;
        
        // For each seam in the batch
        for (int i = 0; i < batch_size; i++) {
            try {
                // Get minimum energy seam
                cv::Mat grad;
                if (USE_FORWARD_ENERGY) {
                    grad = forward_energy(output);
                } else {
                    grad = backward_energy(output);
                }

                // Compute seam with fresh energy
                auto [seam_idx, boolmask] = get_minimum_seam(output, current_mask, cv::Mat());
                
                // Validate seam
                if (seam_idx.empty() || boolmask.empty()) {
                    std::cout << "reduce_width: Error: Invalid seam returned from get_minimum_seam" << std::endl;
                    throw std::runtime_error("Invalid seam returned from get_minimum_seam");
                }
                
                // Remove the seam
                cv::Mat new_output = remove_seam(output, boolmask);
                if (new_output.empty()) {
                    std::cout << "reduce_width: Error: Failed to remove seam" << std::endl;
                    throw std::runtime_error("Failed to remove seam");
                }
                output = new_output;
                
                // Update mask if provided
                if (!current_mask.empty()) {
                    cv::Mat new_mask = remove_seam_grayscale(current_mask, boolmask);
                    if (new_mask.empty()) {
                        std::cout << "reduce_width: Error: Failed to update mask" << std::endl;
                        throw std::runtime_error("Failed to update mask");
                    }
                    current_mask = new_mask;
                }
                
                std::cout << "reduce_width: Seam " << (num_seams - remaining_seams + i + 1) << " of " << num_seams << " removed, new width: " << output.cols << std::endl;
            } catch (const std::exception& e) {
                std::cout << "reduce_width: Exception during seam removal: " << e.what() << std::endl;
                throw;
            }
        }
        
        remaining_seams -= batch_size;
    }
    
    std::cout << "reduce_width: Width reduction completed successfully" << std::endl;
    return output;
}

// Function to reduce height of an image by removing horizontal seams
cv::Mat reduce_height(const cv::Mat& image, int num_seams, const cv::Mat& mask, bool visualize) {
    std::cout << "reduce_height: Starting height reduction by " << num_seams << " seams" << std::endl;
    
    // Validate input image
    if (image.empty()) {
        std::cout << "reduce_height: Error: Input image is empty" << std::endl;
        throw std::runtime_error("Input image is empty");
    }
    
    std::cout << "reduce_height: Input image size: " << image.size() << ", channels: " << image.channels() << std::endl;
    
    // Validate input
    if (num_seams <= 0) {
        std::cout << "reduce_height: No seams to remove, returning original image" << std::endl;
        return image.clone();
    }
    
    if (num_seams >= image.rows) {
        std::cout << "reduce_height: Error: Cannot remove more seams than image height" << std::endl;
        throw std::runtime_error("Cannot remove more seams than image height");
    }
    
    try {
        // Rotate image clockwise
        std::cout << "reduce_height: Rotating image clockwise" << std::endl;
        cv::Mat rotated = rotate_image(image, true);
        cv::Mat rotated_mask;
        if (!mask.empty()) {
            rotated_mask = rotate_image(mask, true);
        }
        
        // Apply reduce_width to the rotated image
        std::cout << "reduce_height: Applying reduce_width to rotated image" << std::endl;
        cv::Mat reduced = reduce_width(rotated, num_seams, rotated_mask, visualize);
        
        // Rotate back counterclockwise
        std::cout << "reduce_height: Rotating image back counterclockwise" << std::endl;
        cv::Mat result = rotate_image(reduced, false);
        
        std::cout << "reduce_height: Final image size: " << result.size() << std::endl;
        std::cout << "reduce_height: Height reduction completed successfully" << std::endl;
        return result;
        
    } catch (const std::exception& e) {
        std::cout << "reduce_height: Error during height reduction: " << e.what() << std::endl;
        throw;
    }
}

// Function to enlarge width of an image by adding vertical seams
cv::Mat enlarge_width(const cv::Mat& image, int num_seams, const cv::Mat& mask, bool visualize) {
    std::cout << "enlarge_width: Starting width enlargement by " << num_seams << " seams" << std::endl;
    
    // Validate input image
    if (image.empty()) {
        std::cout << "enlarge_width: Error: Input image is empty" << std::endl;
        throw std::runtime_error("Input image is empty");
    }
    
    std::cout << "enlarge_width: Input image size: " << image.size() << ", channels: " << image.channels() << std::endl;
    
    // Validate input
    if (num_seams <= 0) {
        std::cout << "enlarge_width: No seams to add, returning original image" << std::endl;
        return image.clone();
    }
    
    try {
        // Make copies of input image and mask for finding seams
        cv::Mat temp_image = image.clone();
        cv::Mat temp_mask = mask.empty() ? cv::Mat() : mask.clone();
        
        // Store seams for later use
        std::vector<std::vector<int>> seams_record;
        seams_record.reserve(num_seams);
        
        // First phase: Find and store all seams
        std::cout << "enlarge_width: Finding optimal seams..." << std::endl;
        for (int i = 0; i < num_seams; i++) {
            // Find the optimal seam
            auto [seam_idx, boolmask] = get_minimum_seam(temp_image, temp_mask, cv::Mat());
            seams_record.push_back(seam_idx);
            
            // Remove the seam from temporary image and mask
            temp_image = remove_seam(temp_image, boolmask);
            if (!temp_mask.empty()) {
                temp_mask = remove_seam_grayscale(temp_mask, boolmask);
            }
            
            std::cout << "enlarge_width: Found seam " << (i + 1) << " of " << num_seams << std::endl;
        }
        
        // Second phase: Add seams back in reverse order with improved blending
        std::cout << "enlarge_width: Adding seams..." << std::endl;
        cv::Mat output = image.clone();
        cv::Mat current_mask = mask.empty() ? cv::Mat() : mask.clone();
        
        // Process seams in reverse order
        for (int i = num_seams - 1; i >= 0; i--) {
            std::vector<int>& seam = seams_record[i];
            
            // Update seam indices for previously added seams
            for (int j = i + 1; j < num_seams; j++) {
                for (size_t row = 0; row < seams_record[j].size(); row++) {
                    if (seams_record[j][row] >= seam[row]) {
                        seams_record[j][row] += 1;  // Shift by 2 because we're adding a new pixel
                    }
                }
            }
            
            // Add the seam with improved blending
            output = add_seam(output, seam);
            if (!current_mask.empty()) {
                current_mask = add_seam_grayscale(current_mask, seam);
            }
            
            // Apply adaptive post-processing to reduce artifacts
            cv::Mat blurred;
            // Use a smaller kernel and sigma for more subtle smoothing
            cv::GaussianBlur(output, blurred, cv::Size(3, 3), 0.3);
            
            // Only apply blur to the seam region to preserve image details
            for (int row = 0; row < output.rows; row++) {
                int col = seam[row];
                // Apply blur only to the seam and its immediate neighbors
                for (int c = std::max(0, col - 1); c <= std::min(output.cols - 1, col + 1); c++) {
                    output.at<cv::Vec3b>(row, c) = blurred.at<cv::Vec3b>(row, c);
                }
            }
            
            std::cout << "enlarge_width: Added seam " << (num_seams - i) << " of " << num_seams << std::endl;
        }
        
        std::cout << "enlarge_width: Final image size: " << output.size() << std::endl;
        std::cout << "enlarge_width: Width enlargement completed successfully" << std::endl;
        return output;
        
    } catch (const std::exception& e) {
        std::cout << "enlarge_width: Error during width enlargement: " << e.what() << std::endl;
        throw;
    }
}

// Function to enlarge height of an image by adding horizontal seams
cv::Mat enlarge_height(const cv::Mat& image, int num_seams, const cv::Mat& mask, bool visualize) {
    std::cout << "enlarge_height: Starting height enlargement by " << num_seams << " seams" << std::endl;
    
    // Validate input image
    if (image.empty()) {
        std::cout << "enlarge_height: Error: Input image is empty" << std::endl;
        throw std::runtime_error("Input image is empty");
    }
    
    std::cout << "enlarge_height: Input image size: " << image.size() << ", channels: " << image.channels() << std::endl;
    
    // Validate input
    if (num_seams <= 0) {
        std::cout << "enlarge_height: No seams to add, returning original image" << std::endl;
        return image.clone();
    }
    
    try {
        // Rotate image clockwise
        std::cout << "enlarge_height: Rotating image clockwise" << std::endl;
        cv::Mat rotated = rotate_image(image, true);
        cv::Mat rotated_mask;
        if (!mask.empty()) {
            rotated_mask = rotate_image(mask, true);
        }
        
        // Apply enlarge_width to the rotated image
        std::cout << "enlarge_height: Applying enlarge_width to rotated image" << std::endl;
        cv::Mat enlarged = enlarge_width(rotated, num_seams, rotated_mask, visualize);
        
        // Rotate back counterclockwise
        std::cout << "enlarge_height: Rotating image back counterclockwise" << std::endl;
        cv::Mat result = rotate_image(enlarged, false);
        
        // Apply final post-processing to reduce rotation artifacts
        cv::Mat final_result;
        cv::GaussianBlur(result, final_result, cv::Size(3, 3), 0.3);
        
        // Blend original and blurred result to preserve details
        cv::addWeighted(result, 0.7, final_result, 0.3, 0, result);
        
        std::cout << "enlarge_height: Final image size: " << result.size() << std::endl;
        std::cout << "enlarge_height: Height enlargement completed successfully" << std::endl;
        return result;
        
    } catch (const std::exception& e) {
        std::cout << "enlarge_height: Error during height enlargement: " << e.what() << std::endl;
        throw;
    }
}

// Function to perform seam carving with both width and height changes
cv::Mat seam_carve(const cv::Mat& image, int dy, int dx, const cv::Mat& mask, bool visualize) {
    std::cout << "seam_carve: Starting with dx=" << dx << ", dy=" << dy << std::endl;
    
    // Validate input image
    if (image.empty()) {
        std::cout << "seam_carve: Error: Input image is empty" << std::endl;
        throw std::runtime_error("Input image is empty");
    }
    
    // Make a copy of the input image
    cv::Mat result = image.clone();
    cv::Mat current_mask = mask.empty() ? cv::Mat() : mask.clone();
    
    // Handle width changes
    if (dx < 0) {
        // Reduce width in batches of 50 seams
        int remaining_seams = -dx;
        while (remaining_seams > 0) {
            int batch_size = std::min(50, remaining_seams);
            std::cout << "seam_carve: Reducing width by " << batch_size << " seams (remaining: " << remaining_seams << ")" << std::endl;
            result = reduce_width(result, batch_size, current_mask, visualize);
            remaining_seams -= batch_size;
        }
    } else if (dx > 0) {
        // Enlarge width in batches of 50 seams
        int remaining_seams = dx;
        while (remaining_seams > 0) {
            int batch_size = std::min(50, remaining_seams);
            std::cout << "seam_carve: Enlarging width by " << batch_size << " seams (remaining: " << remaining_seams << ")" << std::endl;
            result = enlarge_width(result, batch_size, current_mask, visualize);
            remaining_seams -= batch_size;
        }
    }
    
    // Handle height changes
    if (dy < 0) {
        // Reduce height in batches of 50 seams
        int remaining_seams = -dy;
        while (remaining_seams > 0) {
            int batch_size = std::min(50, remaining_seams);
            std::cout << "seam_carve: Reducing height by " << batch_size << " seams (remaining: " << remaining_seams << ")" << std::endl;
            result = reduce_height(result, batch_size, current_mask, visualize);
            remaining_seams -= batch_size;
        }
    } else if (dy > 0) {
        // Enlarge height in batches of 50 seams
        int remaining_seams = dy;
        while (remaining_seams > 0) {
            int batch_size = std::min(50, remaining_seams);
            std::cout << "seam_carve: Enlarging height by " << batch_size << " seams (remaining: " << remaining_seams << ")" << std::endl;
            result = enlarge_height(result, batch_size, current_mask, visualize);
            remaining_seams -= batch_size;
        }
    }
    
    std::cout << "seam_carve: Final image size: " << result.size() << std::endl;
    return result;
}

// Function to remove an object from an image using seam carving
cv::Mat object_removal(const cv::Mat& image, const cv::Mat& rmask, const cv::Mat& mask, bool visualize) {
    std::cout << "object_removal: Starting object removal" << std::endl;
    
    // Validate input image
    if (image.empty()) {
        std::cout << "object_removal: Error: Input image is empty" << std::endl;
        throw std::runtime_error("Input image is empty");
    }
    
    // Validate removal mask
    if (rmask.empty()) {
        std::cout << "object_removal: Error: Removal mask is empty" << std::endl;
        throw std::runtime_error("Removal mask is empty");
    }
    
    // Convert removal mask to binary if needed
    cv::Mat binary_mask;
    if (rmask.channels() > 1) {
        cv::cvtColor(rmask, binary_mask, cv::COLOR_BGR2GRAY);
    } else {
        binary_mask = rmask.clone();
    }
    
    // Threshold to create binary mask
    cv::threshold(binary_mask, binary_mask, MASK_THRESHOLD, 255, cv::THRESH_BINARY);
    
    // Count pixels to remove
    int pixels_to_remove = cv::countNonZero(binary_mask);
    std::cout << "object_removal: Pixels to remove: " << pixels_to_remove << std::endl;
    
    // Make a copy of the input image
    cv::Mat result = image.clone();
    cv::Mat current_mask = mask.empty() ? cv::Mat() : mask.clone();
    
    // Create a removal mask (inverse of binary mask)
    cv::Mat remove_mask;
    cv::bitwise_not(binary_mask, remove_mask);
    
    // Remove seams until all marked pixels are removed
    int max_iterations = pixels_to_remove * 2; // Safety limit
    int iterations = 0;
    
    while (cv::countNonZero(binary_mask) > 0 && iterations < max_iterations) {
        // Find minimum seam
        auto [seam_idx, boolmask] = get_minimum_seam(result, current_mask, remove_mask);
        
        // Remove the seam
        result = remove_seam(result, boolmask);
        
        // Update masks
        if (!current_mask.empty()) {
            current_mask = remove_seam_grayscale(current_mask, boolmask);
        }
        
        // Update binary mask
        binary_mask = remove_seam_grayscale(binary_mask, boolmask);
        
        iterations++;
        
        if (iterations % 10 == 0) {
            std::cout << "object_removal: Iteration " << iterations 
                      << ", pixels remaining: " << cv::countNonZero(binary_mask) << std::endl;
        }
    }
    
    std::cout << "object_removal: Completed after " << iterations << " iterations" << std::endl;
    std::cout << "object_removal: Final image size: " << result.size() << std::endl;
    
    return result;
}

// Function to print usage information
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -resize              Resize image (requires -dx and -dy)" << std::endl;
    std::cout << "  -remove              Remove object (requires -mask)" << std::endl;
    std::cout << "  -im <filename>       Input image file" << std::endl;
    std::cout << "  -out <filename>      Output image file" << std::endl;
    std::cout << "  -dx <value>          Width change (positive to enlarge, negative to reduce)" << std::endl;
    std::cout << "  -dy <value>          Height change (positive to enlarge, negative to reduce)" << std::endl;
    std::cout << "  -target_width <value> Target width in pixels (alternative to -dx)" << std::endl;
    std::cout << "  -target_height <value> Target height in pixels (alternative to -dy)" << std::endl;
    std::cout << "  -mask <filename>     Mask file for object removal" << std::endl;
    std::cout << "  -vis                 Visualize seams" << std::endl;
    std::cout << "  -test                Run test function" << std::endl;
    std::cout << "  -help                Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " -resize -im images/Lena_512.png -out output/resized.jpg -dx -100 -dy 0" << std::endl;
    std::cout << "  " << program_name << " -resize -im images/Lena_512.png -out output/resized.jpg -target_width 162" << std::endl;
    std::cout << "  " << program_name << " -remove -im images/Lena_512.png -mask images/mask.png -out output/removed.jpg" << std::endl;
    std::cout << "  " << program_name << " -test" << std::endl;
}

// Function to parse command line arguments
bool parse_args(int argc, char** argv, std::string& operation, std::string& input_file, 
                std::string& output_file, std::string& mask_file, int& dx, int& dy, bool& visualize) {
    operation = "";
    input_file = "";
    output_file = "";
    mask_file = "";
    dx = 0;
    dy = 0;
    visualize = false;
    int target_width = -1;
    int target_height = -1;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-resize") {
            operation = "resize";
        } else if (arg == "-remove") {
            operation = "remove";
        } else if (arg == "-im" && i + 1 < argc) {
            input_file = argv[++i];
        } else if (arg == "-out" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "-mask" && i + 1 < argc) {
            mask_file = argv[++i];
        } else if (arg == "-dx" && i + 1 < argc) {
            dx = std::stoi(argv[++i]);
        } else if (arg == "-dy" && i + 1 < argc) {
            dy = std::stoi(argv[++i]);
        } else if (arg == "-target_width" && i + 1 < argc) {
            target_width = std::stoi(argv[++i]);
        } else if (arg == "-target_height" && i + 1 < argc) {
            target_height = std::stoi(argv[++i]);
        } else if (arg == "-vis") {
            visualize = true;
        } else if (arg == "-test") {
            operation = "test";
        } else if (arg == "-help") {
            print_usage(argv[0]);
            return false;
        }
    }
    
    // Validate arguments
    if (operation.empty()) {
        std::cout << "Error: No operation specified. Use -resize, -remove, or -test" << std::endl;
        return false;
    }
    
    if (operation != "test" && input_file.empty()) {
        std::cout << "Error: Input file not specified. Use -im <filename>" << std::endl;
        return false;
    }
    
    if (operation != "test" && output_file.empty()) {
        std::cout << "Error: Output file not specified. Use -out <filename>" << std::endl;
        return false;
    }
    
    if (operation == "resize") {
        // If target dimensions are specified, calculate dx and dy
        if (target_width > 0 || target_height > 0) {
            // Load the image to get its dimensions
            cv::Mat image = cv::imread(input_file);
            if (image.empty()) {
                std::cout << "Error: Could not read image '" << input_file << "' to determine dimensions" << std::endl;
                return false;
            }
            
            int current_width = image.cols;
            int current_height = image.rows;
            
            if (target_width > 0) {
                // Calculate the number of seams to add or remove
                dx = target_width - current_width;
                std::cout << "Target width: " << target_width << ", current width: " << current_width << std::endl;
                std::cout << "Calculated dx: " << dx << " (negative means remove seams, positive means add seams)" << std::endl;
            }
            
            if (target_height > 0) {
                // Calculate the number of seams to add or remove
                dy = target_height - current_height;
                std::cout << "Target height: " << target_height << ", current height: " << current_height << std::endl;
                std::cout << "Calculated dy: " << dy << " (negative means remove seams, positive means add seams)" << std::endl;
            }
        } else if (dx == 0 && dy == 0) {
            std::cout << "Error: No size change specified. Use -dx and/or -dy or -target_width and/or -target_height" << std::endl;
            return false;
        }
    }
    
    if (operation == "remove" && mask_file.empty()) {
        std::cout << "Error: Mask file not specified for object removal. Use -mask <filename>" << std::endl;
        return false;
    }
    
    return true;
}

int main(int argc, char** argv) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Parse command line arguments
    std::string operation, input_file, output_file, mask_file;
    int dx = 0, dy = 0;
    bool visualize = false;
    
    if (!parse_args(argc, argv, operation, input_file, output_file, mask_file, dx, dy, visualize)) {
        return 1;
    }
    
    // Run test function if requested
    if (operation == "test") {
        return test_seam_carving();
    }
    
    // Create output directory if it doesn't exist
    #ifdef _WIN32
    system("if not exist output mkdir output");
    #else
    system("mkdir -p output");
    #endif
    
    // Load input image
    std::cout << "Loading image: " << input_file << std::endl;
    cv::Mat image = cv::imread(input_file);
    if (image.empty()) {
        std::cout << "Error: Could not read image '" << input_file << "'" << std::endl;
        return 1;
    }
    
    std::cout << "Original image size: " << image.size() << std::endl;
    
    // Process based on operation
    cv::Mat result;
    
    if (operation == "resize") {
        std::cout << "Resizing image by dx=" << dx << ", dy=" << dy << std::endl;
        result = seam_carve(image, dy, dx, cv::Mat(), visualize);
    } else if (operation == "remove") {
        std::cout << "Removing object using mask: " << mask_file << std::endl;
        cv::Mat mask = cv::imread(mask_file, cv::IMREAD_GRAYSCALE);
        if (mask.empty()) {
            std::cout << "Error: Could not read mask '" << mask_file << "'" << std::endl;
            return 1;
        }
        result = object_removal(image, mask, cv::Mat(), visualize);
    }
    
    // Save result
    std::cout << "Saving result to: " << output_file << std::endl;
    bool success = cv::imwrite(output_file, result);
    if (!success) {
        std::cout << "Error: Could not save image to '" << output_file << "'" << std::endl;
        return 1;
    }
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Final image size: " << result.size() << std::endl;
    std::cout << "Operation completed successfully" << std::endl;
    std::cout << "Total execution time: " << duration.count() << " milliseconds" << std::endl;
    
    return 0;
}

// Test function for seam carving
int test_seam_carving() {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "Program started..." << std::endl;
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    
    // Create output directory if it doesn't exist
    std::cout << "Checking output directory..." << std::endl;
    #ifdef _WIN32
    system("if not exist output mkdir output");
    #else
    system("mkdir -p output");
    #endif
    
    // Try to load an image
    std::cout << "Attempting to load image..." << std::endl;
    cv::Mat image = cv::imread("images/Lena_512.png");
    if (image.empty()) {
        std::cout << "Error: Could not read image 'images/Lena_512.png'" << std::endl;
        std::cout << "Current working directory contents:" << std::endl;
        #ifdef _WIN32
        system("dir");
        #else
        system("ls -la");
        #endif
        return 1;
    }
    
    std::cout << "Image loaded successfully" << std::endl;
    std::cout << "Original image size: " << image.size() << std::endl;
    
    // Create a simple seam for testing add_seam function
    std::cout << "Creating test seam..." << std::endl;
    std::vector<int> test_seam;
    for (int i = 0; i < image.rows; i++) {
        test_seam.push_back(image.cols / 2);  // Vertical seam in the middle
    }
    std::cout << "Test seam created with size: " << test_seam.size() << std::endl;
    
    // Test add_seam function
    std::cout << "Testing add_seam function..." << std::endl;
    cv::Mat image_with_seam = add_seam(image, test_seam);
    std::cout << "Image with seam size: " << image_with_seam.size() << std::endl;
    
    // Save the image with seam
    bool success1 = cv::imwrite("test/image_with_seam.jpg", image_with_seam);
    if (!success1) {
        std::cout << "Error: Could not save image with seam" << std::endl;
        return 1;
    }
    
    std::cout << "Image with seam saved successfully" << std::endl;
    
    // Convert image to grayscale for testing add_seam_grayscale
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    std::cout << "Grayscale image size: " << gray_image.size() << std::endl;
    
    // Test add_seam_grayscale function
    std::cout << "Testing add_seam_grayscale function..." << std::endl;
    cv::Mat gray_with_seam = add_seam_grayscale(gray_image, test_seam);
    std::cout << "Grayscale image with seam size: " << gray_with_seam.size() << std::endl;
    
    // Save the grayscale image with seam
    bool success2 = cv::imwrite("test/gray_with_seam.jpg", gray_with_seam);
    if (!success2) {
        std::cout << "Error: Could not save grayscale image with seam" << std::endl;
        return 1;
    }
    
    std::cout << "Grayscale image with seam saved successfully" << std::endl;
    
    // Create a boolean mask for testing remove_seam function
    std::cout << "Creating boolean mask for remove_seam..." << std::endl;
    cv::Mat boolmask = cv::Mat::ones(image.rows, image.cols, CV_8U);
    for (int row = 0; row < image.rows; row++) {
        boolmask.at<uchar>(row, image.cols / 2) = 0;  // Mark the middle column as the seam
    }
    std::cout << "Boolean mask created with size: " << boolmask.size() << std::endl;
    
    // Test remove_seam function
    std::cout << "Testing remove_seam function..." << std::endl;
    cv::Mat image_without_seam = remove_seam(image, boolmask);
    std::cout << "Image without seam size: " << image_without_seam.size() << std::endl;
    
    // Save the image without seam
    bool success3 = cv::imwrite("test/image_without_seam.jpg", image_without_seam);
    if (!success3) {
        std::cout << "Error: Could not save image without seam" << std::endl;
        return 1;
    }
    
    std::cout << "Image without seam saved successfully" << std::endl;
    
    // Test get_minimum_seam function
    std::cout << "Finding minimum seam..." << std::endl;
    auto [min_seam_idx, min_seam_mask] = get_minimum_seam(image, cv::Mat(), cv::Mat());
    std::cout << "Minimum seam found with " << min_seam_idx.size() << " points" << std::endl;
    
    // Visualize the minimum seam
    cv::Mat image_with_min_seam = image.clone();
    for (int i = 0; i < image.rows; i++) {
        image_with_min_seam.at<cv::Vec3b>(i, min_seam_idx[i]) = cv::Vec3b(0, 0, 255); // Red color
    }
    
    // Save the image with the minimum seam
    if (!cv::imwrite("test/minimum_seam.jpg", image_with_min_seam)) {
        std::cerr << "Error: Could not save image with minimum seam" << std::endl;
        return -1;
    }
    std::cout << "Image with minimum seam saved successfully" << std::endl;
    
    // Test reduce_width function
    std::cout << "\nTesting reduce_width function..." << std::endl;
    try {
        int num_seams_to_remove = 5;  // Start with just 5 seams
        std::cout << "Attempting to reduce width by " << num_seams_to_remove << " seams..." << std::endl;
        cv::Mat reduced_image = reduce_width(image, num_seams_to_remove, cv::Mat(), false);
        std::cout << "Reduced image size: " << reduced_image.size() << std::endl;
        
        // Save the reduced image
        std::cout << "Saving reduced width image..." << std::endl;
        bool success = cv::imwrite("test/reduced_width.jpg", reduced_image);
        if (!success) {
            std::cout << "Error: Could not save reduced width image" << std::endl;
            return 1;
        }
        std::cout << "Reduced width image saved successfully" << std::endl;
        
        // Test reduce_height function
        std::cout << "\nTesting reduce_height function..." << std::endl;
        int height_seams_to_remove = 5;  // Remove 5 horizontal seams
        std::cout << "Attempting to reduce height by " << height_seams_to_remove << " seams..." << std::endl;
        cv::Mat height_reduced_image = reduce_height(image, height_seams_to_remove, cv::Mat(), false);
        std::cout << "Height reduced image size: " << height_reduced_image.size() << std::endl;
        
        // Save the height-reduced image
        std::cout << "Saving height-reduced image..." << std::endl;
        bool success2 = cv::imwrite("test/reduced_height.jpg", height_reduced_image);
        if (!success2) {
            std::cout << "Error: Could not save height-reduced image" << std::endl;
            return 1;
        }
        std::cout << "Height-reduced image saved successfully" << std::endl;
        
        // Test enlarge_width function
        std::cout << "\nTesting enlarge_width function..." << std::endl;
        int width_seams_to_add = 5;  // Add 5 vertical seams
        std::cout << "Attempting to enlarge width by " << width_seams_to_add << " seams..." << std::endl;
        cv::Mat enlarged_image = enlarge_width(image, width_seams_to_add, cv::Mat(), false);
        std::cout << "Enlarged image size: " << enlarged_image.size() << std::endl;
        
        // Save the enlarged image
        std::cout << "Saving enlarged image..." << std::endl;
        bool success3 = cv::imwrite("test/enlarged_width.jpg", enlarged_image);
        if (!success3) {
            std::cout << "Error: Could not save enlarged image" << std::endl;
            return 1;
        }
        std::cout << "Enlarged image saved successfully" << std::endl;
        
        // Test enlarge_height function
        std::cout << "\nTesting enlarge_height function..." << std::endl;
        int height_seams_to_add = 5;  // Add 5 horizontal seams
        std::cout << "Attempting to enlarge height by " << height_seams_to_add << " seams..." << std::endl;
        cv::Mat height_enlarged_image = enlarge_height(image, height_seams_to_add, cv::Mat(), false);
        std::cout << "Height enlarged image size: " << height_enlarged_image.size() << std::endl;
        
        // Save the height-enlarged image
        std::cout << "Saving height-enlarged image..." << std::endl;
        bool success4 = cv::imwrite("test/enlarged_height.jpg", height_enlarged_image);
        if (!success4) {
            std::cout << "Error: Could not save height-enlarged image" << std::endl;
            return 1;
        }
        std::cout << "Height-enlarged image saved successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Exception occurred during seam carving: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "Unknown exception occurred during seam carving" << std::endl;
        return 1;
    }
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "\nTotal execution time: " << duration.count() << " milliseconds" << std::endl;
    
    return 0;
} 