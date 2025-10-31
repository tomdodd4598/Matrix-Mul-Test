#ifndef TYPE_UTILS_H
#define TYPE_UTILS_H

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>

using usize = std::size_t;
using isize = std::make_signed_t<usize>;

using iptrdiff = std::ptrdiff_t;
using uptrdiff = std::make_unsigned_t<iptrdiff>;

using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using f32 = float;
using f64 = double;

using real = f64;
using complex = std::complex<real>;

using Runnable = std::function<void()>;

template<typename T>
using Consumer = std::function<void(T)>;

template<typename T>
using Supplier = std::function<T()>;

template<typename T>
using UnaryOp = std::function<T(T)>;

template<typename T>
using BinaryOp = std::function<T(T, T)>;

template<typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename T>
using collection_value_t = typename remove_cvref_t<T>::value_type;

template<typename T>
constexpr bool is_class_v = std::is_class<T>::value;

#endif
