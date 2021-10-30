#pragma once
#include <chrono>
#include <iostream>

#include "terminal_utils.hpp"

namespace chameleon {

template <bool TIMED, typename F, typename... Args>
inline std::result_of_t<F(Args...)> timed_function(F function, Args... a) {
    if constexpr (TIMED) {
        auto start = std::chrono::steady_clock::now();
        if constexpr (std::is_same_v<std::result_of_t<F(Args...)>, void>) {
            function(a...);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
        } else {
            auto res = function(a...);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
            return res;
        }
    } else {
        if constexpr (std::is_same_v<std::result_of_t<F(Args...)>, void>) {
            function(a...);
        } else {
            return function(a...);
        }
    }
}
static std::chrono::time_point<std::chrono::system_clock>
    __tick__time__start__;

#define TICK __tick__time__start__ = std::chrono::system_clock::now();

#define TOCK                                                                  \
    std::cout                                                                 \
        << TERMINAL_YELLOW << __FUNCTION__ <<" : " << __LINE__ << ": elapsed time: "              \
        << std::chrono::duration<double>(std::chrono::system_clock::now() - __tick__time__start__).count() \
        << "s" << TERMINAL_DEFAULT << std::endl;

}  // namespace chameleon