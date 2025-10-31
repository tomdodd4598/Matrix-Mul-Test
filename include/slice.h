#ifndef SLICE_H
#define SLICE_H

#include "type_utils.h"

#define SLICE(NAME, MODIFIER)\
template<typename T>\
struct NAME {\
    typedef T value_type;\
    \
    T MODIFIER* const ptr;\
    const usize len;\
    \
    NAME() = delete;\
    \
    NAME(T MODIFIER* ptr, usize len) : ptr{ptr}, len{len} {}\
    \
    T* data() MODIFIER {\
        return ptr;\
    }\
    \
    usize size() const {\
        return len;\
    }\
    \
    T MODIFIER& operator[](usize index) MODIFIER {\
        return ptr[index];\
    }\
};\

SLICE(Slice, const)
SLICE(SliceMut,)

#endif
