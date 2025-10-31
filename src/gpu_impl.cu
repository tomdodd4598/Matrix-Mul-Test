#include "gpu_impl.h"

// Modified from QuEST
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ < 600

#define DEFINE_ATOMIC_ADD(FLOAT, INT, FLOAT_AS_INT, INT_AS_FLOAT)\
__forceinline__ __device__ FLOAT atomicAdd(FLOAT* ptr, FLOAT value) {\
    INT* ptr_cast = PTR_CAST(INT*, ptr);\
    INT old = *ptr_cast, assumed;\
    \
    do {\
        assumed = old;\
        old = atomicCAS(ptr_cast, assumed, FLOAT_AS_INT(value + INT_AS_FLOAT(assumed)));\
    }\
    while (assumed != old);\
    \
    return INT_AS_FLOAT(old);\
}\

DEFINE_ATOMIC_ADD(f32, i32, __float_as_int, __int_as_float);
DEFINE_ATOMIC_ADD(f64, i64, __double_as_longlong, __longlong_as_double);

#endif
