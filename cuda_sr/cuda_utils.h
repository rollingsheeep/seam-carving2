#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

// CUDA error checking utilities
namespace cuda_utils {

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cerr << "Code: " << error << ", Reason: " << cudaGetErrorString(error) << std::endl; \
            throw std::runtime_error("CUDA error"); \
        } \
    } while(0)

// Helper function to get and reset the last CUDA error
inline void checkLastCudaError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error");
    }
}

// Managed memory wrapper for CUDA
template<typename T>
class CudaMemory {
private:
    T* dev_ptr = nullptr;
    size_t size = 0;
    bool allocated = false;

public:
    CudaMemory() = default;
    
    // Allocate memory
    void allocate(size_t num_elements) {
        if (allocated) {
            free();
        }
        size = num_elements * sizeof(T);
        CUDA_CHECK(cudaMalloc(&dev_ptr, size));
        allocated = true;
    }
    
    // Free memory
    void free() {
        if (allocated && dev_ptr != nullptr) {
            CUDA_CHECK(cudaFree(dev_ptr));
            dev_ptr = nullptr;
            allocated = false;
            size = 0;
        }
    }
    
    // Copy from host to device
    void copyToDevice(const T* host_data, size_t num_elements) {
        if (!allocated || num_elements * sizeof(T) > size) {
            allocate(num_elements);
        }
        CUDA_CHECK(cudaMemcpy(dev_ptr, host_data, num_elements * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    // Copy from device to host
    void copyToHost(T* host_data, size_t num_elements) const {
        if (allocated) {
            CUDA_CHECK(cudaMemcpy(host_data, dev_ptr, num_elements * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }
    
    // Get device pointer
    T* get() const {
        return dev_ptr;
    }
    
    // Destructor
    ~CudaMemory() {
        free();
    }
};

// Calculate grid and block dimensions
inline dim3 calculateGrid(dim3 block, int width, int height) {
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    return grid;
}

// Synchronize CUDA device
inline void synchronizeDevice() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Get CUDA device properties
inline void printDeviceInfo() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total global memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max block dimensions: " << prop.maxThreadsDim[0] << " x " 
                  << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << std::endl;
        std::cout << "  Max grid dimensions: " << prop.maxGridSize[0] << " x " 
                  << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;
        std::cout << "  Warp size: " << prop.warpSize << std::endl;
        std::cout << "  Memory clock rate: " << (prop.memoryClockRate / 1000) << " MHz" << std::endl;
        std::cout << "  Memory bus width: " << prop.memoryBusWidth << " bits" << std::endl;
    }
}

} // namespace cuda_utils 