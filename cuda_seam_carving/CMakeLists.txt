cmake_minimum_required(VERSION 3.18)
project(cuda_seam_carving CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)

add_executable(cuda_seam_carving 
    cuda_seam_carving.cpp
    cuda_kernels.cu)

target_include_directories(cuda_seam_carving PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_target_properties(cuda_seam_carving PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)