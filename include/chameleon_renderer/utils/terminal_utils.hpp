#pragma once
#include <spdlog/spdlog.h>
namespace chameleon {

#ifndef PRINT_VAR
#define PRINT_VAR(var) spdlog::info("{} = {}", #var, var);
#define PING \
    spdlog::info("{} :: {} : {}", __FILE__, __LINE__, __PRETTY_FUNCTION__);
#endif

}  // namespace chameleon
