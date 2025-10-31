#ifndef CPU_IMPL_H
#define CPU_IMPL_H

#include "type_utils.h"

#include <complex>
#include <vector>

void matmul(usize dim, std::vector<complex> const& mat_a, std::vector<complex> const& mat_b, std::vector<complex>& mat_c);

#endif
