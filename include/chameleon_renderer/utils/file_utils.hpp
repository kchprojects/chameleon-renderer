#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <string>
#include <spdlog/spdlog.h>

namespace chameleon {

namespace fs = std::filesystem;

inline std::string readPTX(std::string const& filename) {
    std::ifstream inputPtx(filename);

    if (!inputPtx) {
        spdlog::error("ERROR: readPTX() Failed to open file {}", filename);
        return std::string();
    }

    std::stringstream ptx;

    ptx << inputPtx.rdbuf();

    if (inputPtx.fail()) {
        spdlog::error("ERROR: readPTX() Failed to read file {}", filename);
        return std::string();
    }

    return ptx.str();
}

}  // namespace chameleon
