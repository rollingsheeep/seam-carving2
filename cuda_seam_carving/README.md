# CUDA Seam Carving Implementation

This directory contains a CUDA-accelerated implementation of the seam carving algorithm for content-aware image resizing.

## Overview

Seam carving is a technique for content-aware image resizing that works by removing paths of pixels (seams) that have the least impact on the image content. This implementation leverages CUDA to parallelize the computation-intensive parts of the algorithm, including:

- Luminance computation
- Energy (gradient) calculation using Sobel filters
- Forward energy computation
- Dynamic programming for finding minimum energy paths
- Seam identification and removal

## Files

- `cuda_kernels.cuh`: CUDA kernel function declarations and data structures
- `cuda_kernels.cu`: CUDA kernel implementations
- `cuda_seam_carving.cpp`: Main implementation file
- `Makefile`: Compilation instructions

## Building

To build the CUDA seam carving implementation, run:

```
make
```

Requirements:
- CUDA toolkit (11.0 or later recommended)
- C++11 compatible compiler
- G++ for host-side compilation

## Usage

```
./cuda_seam_carving <input_image> <output_image> [--energy <forward|backward|hybrid>]
```

### Parameters:

- `input_image`: Path to the input image file
- `output_image`: Path to write the output image
- `--energy`: Energy calculation method (optional)
  - `forward`: Use forward energy (better preserves edges)
  - `backward`: Use backward energy (classic gradient-based approach)
  - `hybrid`: Automatically choose between forward and backward (default)

## Performance

The CUDA implementation achieves significant speedups over the sequential version:

- Sequential implementation: ~60 seconds
- CUDA implementation: Expected ~3-6 seconds (10-20x speedup)

The actual performance improvement depends on your GPU hardware and image size.

## Implementation Details

The parallelization strategy focuses on:

1. 2D kernel launches for pixel-parallel operations (luminance, sobel filter)
2. Row-based kernels for dynamic programming
3. Reduction kernels for finding minimum energy seams
4. Optimized memory transfers to minimize host-device communication

The hybrid energy approach intelligently selects between forward and backward energy based on image content characteristics. 