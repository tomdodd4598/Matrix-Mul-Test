#include "cpu_impl.h"
#include "gpu_impl.h"
#include "time_utils.h"
#include "type_utils.h"

#include <complex>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    const usize dim = 120 * 120;
    const usize size = dim * dim;

    std::vector<complex> mat_a(size);
    std::vector<complex> mat_b(size);
    std::vector<complex> mat_c(size);

    std::cout << "Starting CPU test..." << std::endl;
    auto cpu_start_time = time_now();
    matmul(dim, mat_a, mat_b, mat_c);
    auto cpu_run_time = time_since(cpu_start_time);
    std::cout << "CPU time (s): " << cpu_run_time << std::endl;

    std::cout << "Starting GPU test..." << std::endl;
    auto gpu_start_time = time_now();
    gpu_matmul(dim, mat_a, mat_b, mat_c);
    auto gpu_run_time = time_since(gpu_start_time);
    std::cout << "GPU time (s): " << gpu_run_time << std::endl;

    return 0;
}
