#ifndef MACRO_UTILS_H
#define MACRO_UTILS_H

#include "type_utils.h"

#include <utility>

#define STRINGIFY(X) #X

#define COMMA ,
#define L_PAR (
#define R_PAR )

#define EXPAND(...) __VA_ARGS__
#define EXPAND2(...) EXPAND L_PAR __VA_ARGS__ R_PAR
#define EXPAND3(...) EXPAND2 L_PAR __VA_ARGS__ R_PAR
#define EXPAND4(...) EXPAND3 L_PAR __VA_ARGS__ R_PAR
#define EXPAND5(...) EXPAND4 L_PAR __VA_ARGS__ R_PAR
#define EXPAND6(...) EXPAND5 L_PAR __VA_ARGS__ R_PAR
#define EXPAND7(...) EXPAND6 L_PAR __VA_ARGS__ R_PAR
#define EXPAND8(...) EXPAND7 L_PAR __VA_ARGS__ R_PAR
#define EXPAND9(...) EXPAND8 L_PAR __VA_ARGS__ R_PAR

#define PRAGMA(X) _Pragma(X)

#define FORWARD(X) std::forward<decltype(X)>(X)

#define PTR_CAST(T, X) reinterpret_cast<T>(X)

#endif
