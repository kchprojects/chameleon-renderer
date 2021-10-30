#pragma once
namespace chameleon {
static constexpr const char* TERMINAL_RED = "\033[1;31m";
static constexpr const char* TERMINAL_GREEN = "\033[1;32m";
static constexpr const char* TERMINAL_YELLOW = "\033[1;33m";
static constexpr const char* TERMINAL_BLUE = "\033[1;34m";
static constexpr const char* TERMINAL_RESET = "\033[0m";
static constexpr const char* TERMINAL_DEFAULT = TERMINAL_RESET;
static constexpr const char* TERMINAL_BOLD = "\033[1;1m";

#ifndef PRINT_VAR
#define PRINT_VAR(var) std::cout << #var << "=" << var << std::endl;
#define PING                                                             \
    std::cout << TERMINAL_YELLOW << __FILE__ << "::" << __LINE__ << ": " \
              << __FUNCTION__ << TERMINAL_DEFAULT << std::endl;
#endif

}  // namespace chameleon
