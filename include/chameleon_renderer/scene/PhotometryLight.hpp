#pragma once
#include <memory>
#include <vector>

#include "Light.hpp"

namespace chameleon {
class PhotometryLight
{
public:
    template<typename T>
    PhotometryLight(const Json& j)
    {
        for (const auto& light : j) {
            lights.push_back(std::make_unique<T>(j));
        }
    }
};
} // namespace chameleon