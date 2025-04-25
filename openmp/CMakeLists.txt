cmake_minimum_required(VERSION 3.12)
project(seam_carving_openmp LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Source files
set(SOURCES
    omp_impl.cpp
)

# Define the executable
add_executable(seam_carving_openmp ${SOURCES})

# Link with OpenMP
target_link_libraries(seam_carving_openmp PRIVATE OpenMP::OpenMP_CXX)

# Include parent directory for stb_image headers
target_include_directories(seam_carving_openmp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Add optimization flags for release build
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        # Use LLVM OpenMP for better OpenMP support
        target_compile_options(seam_carving_openmp PRIVATE /O2 /openmp:llvm)
        # Add thread affinity for MSVC
        target_compile_options(seam_carving_openmp PRIVATE /Qopenmp-affinity)
        # Disable sprintf warning
        target_compile_definitions(seam_carving_openmp PRIVATE _CRT_SECURE_NO_WARNINGS)
    else()
        target_compile_options(seam_carving_openmp PRIVATE -O3)
        # Add thread affinity for GCC/Clang
        target_compile_options(seam_carving_openmp PRIVATE -fopenmp-affinity)
    endif()
endif()

# Install target
install(TARGETS seam_carving_openmp
    RUNTIME DESTINATION bin
) 