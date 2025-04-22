# MPI Implementation of Seam Carving

This is an MPI-based parallel implementation of the seam carving algorithm for content-aware image resizing.

## Prerequisites

To build and run this implementation, you need:

1. A C++ compiler with C++14 support
2. CMake (version 3.10 or newer)
3. An MPI implementation (like OpenMPI or MPICH)

## Building the Project

### Using CMake

```bash
# Navigate to the mpi directory
cd mpi

# Create a build directory and enter it
mkdir build
cd build

# Generate build files
cmake ..

# Build the project
cmake --build .
```

## Running the Program

```bash
# Run with N processes (replace N with the desired number of processes)
mpirun -n N ./seam_carving_mpi <input_image> <output_image> [--energy <forward|backward|hybrid>]
```

### Example

```bash
# Run with 4 processes using the hybrid energy function
mpirun -n 4 ./seam_carving_mpi ../../images/example.jpg output.jpg --energy hybrid
```

## Command Line Arguments

- `<input_image>`: Path to the input image file.
- `<output_image>`: Path where the output image will be saved.
- `--energy <type>`: Optional. Specifies the energy function to use:
  - `forward`: Uses forward energy calculation
  - `backward`: Uses backward energy calculation (Sobel filter)
  - `hybrid`: Adaptively selects between forward and backward energy (default)

## Implementation Details

This MPI implementation divides the image processing work among multiple processes:

1. The root process (rank 0) loads the image and distributes necessary data to all processes.
2. Each process works on a subset of rows for most computations.
3. Communication is synchronized at key points in the algorithm.
4. The root process handles seam removal and image saving.

### Key Parallelization Strategies

1. **Row-based division**: The image is divided horizontally among processes.
2. **Collective communications**: Uses MPI collective operations for data sharing.
3. **Load balancing**: Work is distributed evenly with special handling for the last process.
4. **Minimal communication**: Optimized to reduce communication overhead where possible.

## Performance Considerations

- The MPI implementation is most effective on large images where computational work outweighs communication overhead.
- Scaling efficiency depends on image dimensions and the number of processes.
- For small images, using fewer processes may be more efficient due to communication overhead. 