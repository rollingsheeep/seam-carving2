cmake_minimum_required(VERSION 3.12)
project(seam_carving_sequential LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SOURCES
    sequential_impl.cpp
)

# Define the executable
add_executable(seam_carving_sequential ${SOURCES})

# Include parent directory for stb_image headers
target_include_directories(seam_carving_sequential PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Add optimization flags for release build
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(seam_carving_sequential PRIVATE /O2)
    else()
        target_compile_options(seam_carving_sequential PRIVATE -O3)
    endif()
endif()

# Install target
install(TARGETS seam_carving_sequential
    RUNTIME DESTINATION bin
) 