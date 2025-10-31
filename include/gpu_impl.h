#ifndef GPU_IMPL_H
#define GPU_IMPL_H

#include "type_utils.h"

#include <complex>
#include <vector>

void gpu_matmul(usize dim, std::vector<complex> const& mat_a, std::vector<complex> const& mat_b, std::vector<complex>& mat_c);

#endif
