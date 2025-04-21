cmake_minimum_required(VERSION 3.10)
project(seam_carving)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add executable
add_executable(seam_carving seam_carving.cpp)

# Link OpenCV libraries
target_link_libraries(seam_carving ${OpenCV_LIBS}) 