cmake_minimum_required(VERSION 3.18)
project(seam_carving_cuda LANGUAGES CXX CUDA)

# Set standards
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Set CUDA architecture (adjust for your GPU if needed)
set(CMAKE_CUDA_ARCHITECTURES 75 86)

# If using MSVC + CUDA >= 12, help NVCC find host compiler
if (MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
endif()

# Source files
set(CUDA_SOURCES
    luminance_kernel.cu
    energy_kernels.cu
    dp_kernel.cu
    visualization_kernel.cu
    seam_kernel.cu
    hybrid_energy_kernel.cu
)

set(CXX_SOURCES
    seam_carving_cuda.cpp
    main_cuda.cpp
    visualization.cpp
)

# Add executable and all sources
add_executable(seam_carving_cuda ${CXX_SOURCES} ${CUDA_SOURCES})

# Optional: suppress deprecated target warning
target_compile_options(seam_carving_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--Wno-deprecated-gpu-targets>")

# Optional: enable separate compilation for CUDA (only needed if you're doing device linking)
set_target_properties(seam_carving_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# Include directories (your headers are all local)
target_include_directories(seam_carving_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
