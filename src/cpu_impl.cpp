#include "cpu_impl.h"
#include "macro_utils.h"
#include "type_utils.h"

#include <cblas.h>

#include <algorithm>
#include <complex>
#include <vector>

void matmul(usize dim, std::vector<complex> const& mat_a, std::vector<complex> const& mat_b, std::vector<complex>& mat_c) {
    const complex alpha = 1.0;
    const complex beta  = 0.0;

    cblas_zgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        dim, dim, dim,
        PTR_CAST(void const*, &alpha),
        PTR_CAST(void const*, mat_a.data()), dim,
        PTR_CAST(void const*, mat_b.data()), dim,
        PTR_CAST(void const*, &beta),
        PTR_CAST(void*, mat_c.data()), dim
    );
}
