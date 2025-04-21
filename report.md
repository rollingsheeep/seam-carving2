cd d:/project/seam-carving2

# ORIGINAL
g++ main.cpp -o main.exe -std=c++17
./main.exe images/surfer.jpg output/surfer_hybrid.png --energy hybrid

Progress: 100% complete
Seam removal breakdown:
  Dynamic programming: 52256.9 ms
  Seam computation: 88.683 ms
  Seam removal: 35944.9 ms
  Energy update: 294082 ms
  Total seam removal time: 385191 ms
Hybrid energy usage summary:
  Forward selected: 896 times (69.9454%)
  Backward selected: 385 times (30.0546%)
Image saving time: 292 ms
OK: generated output/surfer_hybrid.png
Total execution time: 385920 ms


# OMP
g++ -fopenmp -O3 main_omp.cpp -o seam_carving_omp
./seam_carving_omp images/surfer.jpg output/surfer_omp.png --energy hybrid --threads 4

Progress: 100% complete
Seam removal breakdown:
  Dynamic programming: 63807.6 ms
  Seam computation: 81.082 ms
  Seam removal: 6848.86 ms
  Energy update: 20835 ms
  Total seam removal time: 92902 ms
Hybrid energy usage summary:
  Forward selected: 896 times (69.9454%)
  Backward selected: 385 times (30.0546%)
Image saving time: 128 ms
OK: generated output/surfer_omp.png
Total execution time: 93072 ms

=== Parallelization Summary ===
Parallel Method: OpenMP
Number of Threads: 4
OpenMP Version: 201511
==============================

# CUDA
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
Initial energy computation time: 17 ms
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
  Dynamic programming: 27159.9 ms
  Seam computation: 3965.34 ms
  Seam removal: 10940.7 ms
  Energy update: 17230.6 ms
  Total seam removal time: 67030 ms
Image saving time: 144 ms

Hybrid energy summary:
  Backward energy usage: 94.61%
  Forward energy usage: 5.39%
  Ratio (Backward:Forward): 17.55:1
OK: generated ..\..\..\output\surfer_cuda.jpg
Total execution time: 67304 ms