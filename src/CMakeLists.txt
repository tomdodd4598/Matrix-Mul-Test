add_library(MatMulLib STATIC cpu_impl.cpp gpu_impl.cu time_utils.cpp)
target_include_directories(MatMulLib PUBLIC ${CUDA_INCLUDE_DIRS} ${IncludeDir})

add_executable(MatMulTest matmultest.cpp)
target_link_libraries(MatMulTest PRIVATE MatMulLib)
