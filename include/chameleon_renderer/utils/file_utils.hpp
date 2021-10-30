#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace chameleon {
inline std::string readPTX(std::string const& filename) {
    std::ifstream inputPtx(filename);

    if (!inputPtx) {
        std::cerr << "ERROR: readPTX() Failed to open file " << filename
                  << '\n';
        return std::string();
    }

    std::stringstream ptx;

    ptx << inputPtx.rdbuf();

    if (inputPtx.fail()) {
        std::cerr << "ERROR: readPTX() Failed to read file " << filename
                  << '\n';
        return std::string();
    }

    return ptx.str();
}

}  // namespace chameleon
