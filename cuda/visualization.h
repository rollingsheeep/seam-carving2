#pragma once

#include "data_structures.h"
#include <string>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <fstream>

// Include STB image write
#include "../stb_image_write.h"

// Visualization utilities for seam carving CUDA implementation
namespace visualization {

// Convert a matrix to a grayscale image for visualization
void matrixToGrayscaleImage(const Matrix& matrix, const std::string& filename);

// Create a heatmap visualization of the energy map
void energyToHeatmap(const Matrix& energy, const std::string& filename);

// Visualize the current image with the seam highlighted
void visualizeSeam(const Image& img, const std::vector<int>& seam, const std::string& filename);

// Save all visualizations for a specific stage
void saveStageVisualizations(
    const Image& img,
    const Matrix& lum,
    const Matrix& energy,
    const Matrix& dp,
    const std::vector<int>& seam,
    int stage
);

} // namespace visualization 