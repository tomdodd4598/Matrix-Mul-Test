#include "gpu_utils.h"
#include "type_utils.h"

#include <cuda_runtime.h>

#include <vector>

const std::vector<int> gpu_ids = get_gpu_ids();
const usize gpu_count = gpu_ids.size();

// Modified from QuEST
std::vector<int> get_gpu_ids() {
    int device_count;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&device_count);
    if (cudaResultCode != cudaSuccess) {
        device_count = 0;
    }

    std::vector<int> gpu_ids;
    cudaDeviceProp properties;
    for (int i = 0; i < device_count; ++i) {
        cudaGetDeviceProperties(&properties, i);
        if (properties.major != 9999) {
            gpu_ids.push_back(i);
        }
    }

    return gpu_ids;
}

int gpu_get_id() {
    int id;
    cudaGetDevice(&id);
    return id;
}

usize gpu_get_free_memory() {
    usize free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    return free_memory;
}

usize gpu_get_max_threads() {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, gpu_get_id());
    return props.multiProcessorCount * props.maxThreadsPerMultiProcessor;
}
