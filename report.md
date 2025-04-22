cd d:/project/seam-carving2

# ORIGINAL
seam_carving_sequential.exe ..\..\..\..\images\surfer.jpg ..\..\..\..\output\surfer_sequential.jpg --energy hybrid

Using Sequential implementation
Image loading time: 19 ms
Luminance computation time: 4 ms
Using hybrid energy calculation
Initial energy computation time: 56 ms
Removing 1280 seams...
Progress: 10% complete
Progress: 20% complete
Progress: 30% complete
Progress: 40% complete
Progress: 50% complete
Progress: 60% complete
Progress: 70% complete
Progress: 80% complete
Progress: 90% complete
Progress: 100% complete
Seam removal breakdown:
  Dynamic programming: 2636.36 ms
  Seam computation: 29.18 ms
  Seam removal: 11504 ms
  Energy update: 46244.4 ms
  Total seam removal time: 62004 ms

Hybrid energy summary:
  Backward energy usage: 0.00%
  Forward energy usage: 100.00%
  Ratio (Backward:Forward): 0.00:1
Image saving time: 154 ms
OK: generated ..\..\..\..\output\surfer_sequential.jpg
Total execution time: 62249 ms


# OMP
seam_carving_openmp.exe ..\..\..\..\images\surfer.jpg ..\..\..\..\output\surfer_omp.jpg --energy hybrid --threads 4

Using 4 OpenMP threads
Image loading time: 17 ms
Luminance computation time: 1 ms
Using hybrid energy calculation
Initial energy computation time: 37 ms
Removing 1280 seams...
Progress: 10% complete
Progress: 20% complete
Progress: 30% complete
Progress: 40% complete
Progress: 50% complete
Progress: 60% complete
Progress: 70% complete
Progress: 80% complete
Progress: 90% complete
Progress: 100% complete
Seam removal breakdown:
  Dynamic programming: 2143.51 ms
  Seam computation: 17.494 ms
  Seam removal: 6783.08 ms
  Energy update: 38778.6 ms
  Total seam removal time: 49058 ms
Hybrid energy summary:
  Backward energy usage: 0%
  Forward energy usage: 100%
  Ratio (Backward:Forward): 0:1
Image saving time: 154 ms
OK: generated ..\..\..\..\output\surfer_omp.jpg
Total execution time: 49276 ms

# CUDA
seam_carving_cuda.exe ..\..\..\images\surfer.jpg ..\..\..\output\surfer_cuda.jpg --energy hybrid

Image loading time: 17 ms
Found 1 CUDA device(s)
Device 0: NVIDIA GeForce RTX 3050 Laptop GPU
  Compute capability: 8.6
  Total global memory: 4095 MB
  Multiprocessors: 16
  Max threads per block: 1024
  Max block dimensions: 1024 x 1024 x 64
  Max grid dimensions: 2147483647 x 65535 x 65535
  Warp size: 32
  Memory clock rate: 6001 MHz
  Memory bus width: 128 bits
Using CUDA acceleration
Using hybrid energy calculation
Luminance computation time: 59 ms
Initial energy computation time: 30 ms
Removing 1280 seams...
Progress: 10% complete
Progress: 20% complete
Progress: 30% complete
Progress: 40% complete
Progress: 50% complete
Progress: 60% complete
Progress: 70% complete
Progress: 80% complete
Progress: 90% complete
Progress: 100% complete
Seam removal breakdown:
  Dynamic programming: 25599.7 ms
  Seam computation: 1568.64 ms
  Visualization: 0 ms
  Seam removal: 9163.25 ms
  Energy update: 24233.2 ms
  Total seam removal time: 61849 ms
Image saving time: 141 ms

Hybrid energy summary:
  Backward energy usage: 78.95%
  Forward energy usage: 21.05%
  Ratio (Backward:Forward): 3.75:1
OK: generated ..\..\..\output\surfer_cuda.jpg
Total execution time: 62132 ms


# MPI
mpiexec -n 4 seam_carving_mpi.exe ..\..\..\images\surfer.jpg ..\..\..\output\surfer_mpi.jpg --energy hybrid

Using MPI implementation with 1 processes
Image loading time: 19 ms
Luminance computation time: 4 ms
Using hybrid energy calculation
Initial energy computation time: 55 ms
Removing 1280 seams...
Progress: 10% complete
Progress: 20% complete
Progress: 30% complete
Progress: 40% complete
Progress: 50% complete
Progress: 60% complete
Progress: 70% complete
Progress: 80% complete
Progress: 90% complete
Progress: 100% complete
Seam removal breakdown:
  Dynamic programming: 2271.55 ms
  Seam computation: 22.136 ms
  Seam removal: 11696.6 ms
  Energy update: 33377.3 ms
  Total seam removal time: 49529 ms

Hybrid energy summary:
  Backward energy usage: 50.04%
  Forward energy usage: 49.96%
  Ratio (Backward:Forward): 1.00:1
Image saving time: 140 ms
OK: generated ..\..\..\output\surfer_mpi.jpg
Total execution time: 49760 ms

Using MPI implementation with 4 processes
Image loading time: 26 ms
Luminance computation time: 4 ms
Using hybrid energy calculation
Initial energy computation time: 48 ms
Removing 1280 seams...
Progress: 10% complete
Progress: 20% complete
Progress: 30% complete
Progress: 40% complete
Progress: 50% complete
Progress: 60% complete
Progress: 70% complete
Progress: 80% complete
Progress: 90% complete
Progress: 100% complete
Seam removal breakdown:
  Dynamic programming: 3795.19 ms
  Seam computation: 19.587 ms
  Seam removal: 11199.4 ms
  Energy update: 18608.4 ms
  Total seam removal time: 42518 ms

Hybrid energy summary:
  Backward energy usage: 50.04%
  Forward energy usage: 49.96%
  Ratio (Backward:Forward): 1.00:1
Image saving time: 133 ms
OK: generated ..\..\..\output\surfer_mpi.jpg
Total execution time: 42754 ms