# CUDA-Accelerated Seam Carving

This is a CUDA-accelerated implementation of the content-aware image resizing algorithm known as "seam carving". The implementation parallelizes the computationally intensive parts of the algorithm to run on NVIDIA GPUs.

## Features

- CUDA acceleration for the most compute-intensive parts of seam carving:
  - Luminance/grayscale conversion
  - Energy calculation (Sobel filter and forward energy)
  - Dynamic programming for seam finding
  - Seam selection

- Multiple energy calculation methods:
  - Backward energy (Sobel filter)
  - Forward energy (Rubinstein et al.)
  - Hybrid energy selection

- Visualization capabilities to see intermediate steps:
  - Grayscale/luminance representation
  - Energy maps
  - Dynamic programming matrices
  - Seam visualization

## Requirements

- CUDA Toolkit 11.0 or later
- CMake 3.18 or later
- C++14 compatible compiler
- NVIDIA GPU with compute capability 6.0 or higher

## Building

```bash
cd D:\Project\seam-carving2\cuda
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
```

1. Open build/ .sln file with VisualStudio
2. Build Solution

in x64
seam_carving_cuda.exe ..\..\..\images\surfer.jpg ..\..\..\output\surfer_cuda.jpg --energy hybrid

## Usage

```bash
# Basic usage
./seam_carving_cuda <input_image> <output_image>

# With options
./seam_carving_cuda <input_image> <output_image> --energy <forward|backward|hybrid> --visualize

# Use CPU implementation instead
./seam_carving_cuda <input_image> <output_image> --cpu
```

### Command Line Options

- `--energy <forward|backward|hybrid>`: Choose energy calculation method (default: hybrid)
- `--cpu`: Disable CUDA acceleration and use CPU implementation
- `--visualize`: Enable visualization of intermediate steps (stages 1, 25, 50, 100)

## Implementation Details

The CUDA implementation parallellizes the following parts of the seam carving algorithm:

1. **Luminance calculation**: Each pixel's RGB values are converted to luminance in parallel.
2. **Energy calculation**: 
   - **Backward energy**: Sobel filter is applied in parallel to each pixel
   - **Forward energy**: Each pixel's forward energy is calculated in parallel
3. **Dynamic programming**: Each row of the DP matrix is computed in parallel (with row synchronization)
4. **Seam finding**: Finding the minimum value in the last row is done in parallel

The seam removal operation is still performed on the CPU due to its serial nature and data dependencies.

## Performance

On compatible GPUs, the CUDA implementation provides significant speedup compared to the CPU version, particularly for large images. The exact performance improvement depends on:

- Image dimensions
- GPU model and compute capability
- Energy calculation method used

## Visualization

When run with the `--visualize` flag, the program generates visualization files in the `output` directory for the following stages: 1, 25, 50, and 100. Each stage includes:

- The current state of the image
- Grayscale/luminance representation
- Energy map visualization (as a heat map)
- Dynamic programming matrix visualization
- Image with highlighted seam to be removed

## License

This project is part of a larger seam carving implementation and follows the same licensing terms as the parent project. 