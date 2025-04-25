# Seam Carving Implementation

This is a C++ implementation of the seam carving algorithm for content-aware image resizing, based on the paper "Seam Carving for Content-Aware Image Resizing" by Shai Avidan and Ariel Shamir. The implementation includes four versions: sequential, OpenMP, CUDA, and MPI for performance comparison.

## Features

- Content-aware image resizing using seam carving
- Multiple energy calculation methods:
  - Forward energy (as described in "Improved Seam Carving for Video Retargeting")
  - Backward energy (classic Sobel filter approach)
  - Hybrid energy (dynamic selection between forward and backward)
- Four implementation approaches:
  - Sequential (single-threaded)
  - OpenMP (CPU multi-threading)
  - CUDA (GPU acceleration)
  - MPI (distributed computing)
- Performance benchmarking and comparison

## Project Structure

```
.
├── sequential/             # Sequential implementation
│   ├── sequential_impl.cpp # Sequential implementation code
│   └── CMakeLists.txt      # CMake build script for sequential version
├── openmp/                 # OpenMP implementation
│   ├── omp_impl.cpp        # OpenMP implementation code
│   └── CMakeLists.txt      # CMake build script for OpenMP version
├── cuda/                   # CUDA implementation
│   ├── seam_carving_cuda.cpp    # CUDA implementation code
│   ├── *.cu                     # CUDA kernel files
│   └── CMakeLists.txt           # CMake build script for CUDA version
├── mpi/                    # MPI implementation
│   ├── mpi_impl.cpp        # MPI implementation code
│   └── CMakeLists.txt      # CMake build script for MPI version
├── stb_image.h             # Image loading library
├── stb_image_write.h       # Image writing library
├── images/                 # Test images directory
└── output/                 # Output directory for processed images
```

## Prerequisites

Depending on which version you want to build, you'll need to install different tools:

1. **For all versions:**
   - CMake 3.10 or higher
   - Visual Studio 2022 or another C++ compiler with C++14 support

2. **For OpenMP version:**
   - A compiler with OpenMP support (most modern C++ compilers include this)

3. **For CUDA version:**
   - NVIDIA CUDA Toolkit 11.0 or higher
   - NVIDIA GPU with compute capability 5.0 or higher

4. **For MPI version:**
   - Microsoft MPI (MS-MPI) or another MPI implementation
   - MS-MPI can be downloaded from the Microsoft website

## Building

### Building with Visual Studio

For each implementation (sequential, openmp, cuda, mpi), follow these steps:

```bash
# Navigate to the specific implementation directory
cd D:\Project\seam-carving2\{sequential|openmp|cuda|mpi}

# Create a build directory
mkdir build
cd build

# Configure the project with CMake for Visual Studio
cmake -G "Visual Studio 17 2022" -A x64 ..
```

After running these commands:
1. Open the generated `.sln` file in Visual Studio
2. Select the "Release" configuration
3. Build the solution (F7 or Build → Build Solution)
4. The executable will be created in the `build\Release` folder

### Building with Other Compilers

```bash
# Navigate to the specific implementation directory
cd {sequential|openmp|cuda|mpi}

# Create a build directory
mkdir build
cd build

# Configure the project
cmake ..

# Build the project
cmake --build . --config Release
```

## Running the Programs

After building, you can run each implementation with the following commands:

### Sequential Version

```bash
cd D:\Project\seam-carving2\sequential\build\Release
seam_carving_sequential.exe <input_image> <output_image> [--energy <forward|backward|hybrid>]
```

### OpenMP Version

```bash
cd D:\Project\seam-carving2\openmp\build\Release
seam_carving_openmp.exe <input_image> <output_image> [--energy <forward|backward|hybrid>] [--threads <num_threads>]
```

### CUDA Version

```bash
cd D:\Project\seam-carving2\cuda\build\Release
seam_carving_cuda.exe <input_image> <output_image> [--energy <forward|backward|hybrid>]
```

### MPI Version

```bash
cd D:\Project\seam-carving2\mpi\build\Release
seam_carving_mpi.exe <input_image> <output_image> [--energy <forward|backward|hybrid>]
```

For MPI with multiple processes:

```bash
mpiexec -n <num_processes> seam_carving_mpi.exe <input_image> <output_image> [--energy <forward|backward|hybrid>]
```

## Examples

Here are some example commands for each implementation:

```bash
# Sequential version with hybrid energy
seam_carving_sequential.exe ..\..\..\images\surfer.jpg ..\..\..\output\surfer_sequential.jpg --energy hybrid

# OpenMP version with 4 threads
seam_carving_openmp.exe ..\..\..\images\surfer.jpg ..\..\..\output\surfer_omp.jpg --energy hybrid --threads 4

# CUDA version
seam_carving_cuda.exe ..\..\..\images\surfer.jpg ..\..\..\output\surfer_cuda_sr_new.jpg --energy hybrid

# MPI version with 4 processes
mpiexec -n 4 seam_carving_mpi.exe ..\..\..\images\surfer.jpg ..\..\..\output\surfer_mpi4_new.jpg --energy hybrid
```

## Command Line Arguments

- `<input_image>`: Path to the input image file
- `<output_image>`: Path where the output image will be saved
- `--energy <type>`: Energy calculation method
  - `forward`: Uses forward energy calculation
  - `backward`: Uses backward energy calculation (Sobel filter)
  - `hybrid`: Adaptively selects between forward and backward energy (default)
- `--threads <num>`: Number of threads to use (OpenMP version only)

## Implementation Details

The seam carving algorithm is implemented in four different ways to showcase different parallel computing paradigms:

1. **Sequential**: A baseline single-threaded implementation
2. **OpenMP**: Shared-memory parallelism using CPU threads
3. **CUDA**: Massively parallel GPU implementation
4. **MPI**: Distributed-memory parallelism for multi-node execution

Each implementation follows the same basic algorithm:
1. Convert image to luminance
2. Calculate energy (forward, backward, or hybrid)
3. Use dynamic programming to find minimum energy paths
4. Remove seams with minimum energy
5. Update energy after each seam removal

## Performance Comparison

Different implementations have different strengths:

- **Sequential**: Simple baseline, works on any system
- **OpenMP**: Good speedup on multi-core systems, relatively simple implementation
- **CUDA**: Excels at energy computation but struggles with dynamic programming steps
- **MPI**: Suitable for very large images across multiple compute nodes

For detailed performance analysis, see `performance_analysis.md`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
