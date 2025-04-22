cmake_minimum_required(VERSION 3.10)
project(seam_carving_mpi)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI package
find_package(MPI REQUIRED)

# Add executable
add_executable(seam_carving_mpi mpi_impl.cpp)

# Include MPI directories and link MPI libraries
target_include_directories(seam_carving_mpi PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(seam_carving_mpi ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})

# Set compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(seam_carving_mpi PRIVATE -Wall -Wextra -O3)
elseif(MSVC)
    target_compile_options(seam_carving_mpi PRIVATE /W4 /O2)
endif()

# Installation targets
install(TARGETS seam_carving_mpi DESTINATION bin) 