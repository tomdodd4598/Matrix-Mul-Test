#include "cpu_impl.h"
#include "type_utils.h"

#include <algorithm>
#include <complex>

constexpr usize TILE_SIZE = 32;

void matmul(usize dim, std::vector<complex> const& mat_a, std::vector<complex> const& mat_b, std::vector<complex>& mat_c) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (usize ii = 0; ii < dim; ii += TILE_SIZE) {
        for (usize jj = 0; jj < dim; jj += TILE_SIZE) {
            auto i_end = std::min(ii + TILE_SIZE, dim);
            auto j_end = std::min(jj + TILE_SIZE, dim);

            for (usize i = ii; i < i_end; ++i) {
                for (usize j = jj; j < j_end; ++j) {
                    mat_c[i * dim + j] = 0.0;
                }
            }

            for (usize kk = 0; kk < dim; kk += TILE_SIZE) {
                auto k_end = std::min(kk + TILE_SIZE, dim);

                for (usize i = ii; i < i_end; ++i) {
                    for (usize k = kk; k < k_end; ++k) {
                        const auto a_ik = mat_a[i * dim + k];
                        for (usize j = jj; j < j_end; ++j) {
                            mat_c[i * dim + j] += a_ik * mat_b[k * dim + j];
                        }
                    }
                }
            }
        }
    }
}
