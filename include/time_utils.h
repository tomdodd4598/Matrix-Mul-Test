#ifndef TIME_UTILS_H
#define TIME_UTILS_H

#include "type_utils.h"

#include <chrono>

using Clock = std::chrono::steady_clock;
using TimeUnit = std::chrono::nanoseconds;

/**
 * \brief Get current clock time.
 **/
Clock::time_point time_now();

/**
 * \brief Get time between start and end in seconds.
 **/
real time_interval(Clock::time_point start, Clock::time_point end);

/**
 * \brief Get time since start in seconds.
 **/
real time_since(Clock::time_point start);

template<typename FUNCTION>
real time_of(FUNCTION&& f) {
    const auto start_time = time_now();
    f();
    return time_since(start_time);
}

#endif
