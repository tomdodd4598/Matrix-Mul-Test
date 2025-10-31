#include "gpu_impl.h"
#include "type_utils.h"

#include <cuComplex.h>

#include <complex>
#include <vector>

constexpr usize TILE_SIZE = 32;

__global__ void kernel_matmul(usize dim, cuDoubleComplex* mat_a, cuDoubleComplex* mat_b, cuDoubleComplex* mat_c) {
    __shared__ cuDoubleComplex tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ cuDoubleComplex tile_b[TILE_SIZE][TILE_SIZE];

    auto tx = threadIdx.x;
    auto ty = threadIdx.y;
    auto col = blockIdx.x * TILE_SIZE + tx;
    auto row = blockIdx.y * TILE_SIZE + ty;

    auto result = make_cuDoubleComplex(0.0, 0.0);

    for (usize t = 0, t_end = (dim + TILE_SIZE - 1) / TILE_SIZE; t < t_end; ++t) {
        auto a_col = t * TILE_SIZE + tx;
        auto a_row = row;
        auto b_col = col;
        auto b_row = t * TILE_SIZE + ty;

        if (a_row < dim && a_col < dim) {
            tile_a[ty][tx] = mat_a[a_row * dim + a_col];
        } else {
            tile_a[ty][tx] = make_cuDoubleComplex(0.0, 0.0);
        }

        if (b_row < dim && b_col < dim) {
            tile_b[ty][tx] = mat_b[b_row * dim + b_col];
        } else {
            tile_b[ty][tx] = make_cuDoubleComplex(0.0, 0.0);
        }

        __syncthreads();

        for (usize k = 0; k < TILE_SIZE; ++k) {
            result = cuCadd(result, cuCmul(tile_a[ty][k], tile_b[k][tx]));
        }

        __syncthreads();
    }

    if (row < dim && col < dim) {
        mat_c[row * dim + col] = result;
    }
}

void gpu_matmul(usize dim, std::vector<complex> const& mat_a, std::vector<complex> const& mat_b, std::vector<complex>& mat_c) {
    cuDoubleComplex* d_mat_a;
    cuDoubleComplex* d_mat_b;
    cuDoubleComplex* d_mat_c;

    auto bytes = dim * sizeof(complex);

    cudaMalloc(&d_mat_a, bytes);
    cudaMalloc(&d_mat_b, bytes);
    cudaMalloc(&d_mat_c, bytes);

    cudaMemcpy(d_mat_a, mat_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat_b, mat_b.data(), bytes, cudaMemcpyHostToDevice);
    
    auto grid_size = (dim + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid_dim(grid_size, grid_size);
    dim3 block_dim(TILE_SIZE, TILE_SIZE);

    kernel_matmul<<<grid_dim, block_dim>>>(dim, d_mat_a, d_mat_b, d_mat_c);

    cudaDeviceSynchronize();

    cudaMemcpy(mat_c.data(), d_mat_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_mat_a);
    cudaFree(d_mat_b);
    cudaFree(d_mat_c);
}
