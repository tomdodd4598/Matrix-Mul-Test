#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include "type_utils.h"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

constexpr usize KERNEL_BLOCK_SIZE = 256;

#define RUN_KERNEL(NAME, SIZE, ...)\
kernel_##NAME<<<static_cast<usize>(std::ceil(static_cast<f64>(SIZE) / static_cast<f64>(KERNEL_BLOCK_SIZE))), KERNEL_BLOCK_SIZE>>>(__VA_ARGS__);\
{\
    cudaError_t err = cudaGetLastError();\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA: " << cudaGetErrorString(err) << std::endl;\
        exit(EXIT_FAILURE);\
    }\
}\

#define GPU_OR_DEFAULT(NAME, ...) gpu_count > 0 ? gpu_##NAME(__VA_ARGS__) : NAME(__VA_ARGS__)

extern const std::vector<int> gpu_ids;
extern const usize gpu_count;

std::vector<int> get_gpu_ids();

int gpu_get_id();

usize gpu_get_free_memory();

usize gpu_get_max_threads();

#endif
