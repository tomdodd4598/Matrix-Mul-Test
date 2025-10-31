#include "gpu_impl.h"
#include "gpu_utils.h"
#include "type_utils.h"

#include <cublas_v2.h>
#include <cuComplex.h>

#include <complex>
#include <vector>

void gpu_matmul(usize dim, std::vector<complex> const& mat_a, std::vector<complex> const& mat_b, std::vector<complex>& mat_c) {
    cuDoubleComplex* d_mat_a;
    cuDoubleComplex* d_mat_b;
    cuDoubleComplex* d_mat_c;

    const auto bytes = dim * dim * sizeof(complex);

    CHECK_CUDA(cudaMalloc(&d_mat_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_mat_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_mat_c, bytes));

    CHECK_CUDA(cudaMemcpy(d_mat_a, mat_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mat_b, mat_b.data(), bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const auto alpha = make_cuDoubleComplex(1.0, 0.0);
    const auto beta  = make_cuDoubleComplex(0.0, 0.0);

    CHECK_CUBLAS(cublasZgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        dim, dim, dim,
        &alpha,
        d_mat_b, dim,
        d_mat_a, dim,
        &beta,
        d_mat_c, dim
    ));

    CHECK_CUBLAS(cublasDestroy(handle));

    CHECK_CUDA(cudaMemcpy(mat_c.data(), d_mat_c, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_mat_a));
    CHECK_CUDA(cudaFree(d_mat_b));
    CHECK_CUDA(cudaFree(d_mat_c));
}
