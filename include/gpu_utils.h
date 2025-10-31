#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <cublas_v2.h>

#include <iostream>

#define CHECK_CUDA(X)\
{\
    cudaError_t err = X;\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA: " << cudaGetErrorString(err) << std::endl;\
        exit(EXIT_FAILURE);\
    }\
}\

#define CHECK_CUBLAS(X)\
{\
    cublasStatus_t status = X;\
    if (status != CUBLAS_STATUS_SUCCESS) {\
        std::cerr << "cuBLAS: " << status << std::endl;\
        exit(EXIT_FAILURE);\
    }\
}\

#endif
