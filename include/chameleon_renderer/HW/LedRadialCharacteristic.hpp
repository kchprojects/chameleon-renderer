#pragma once
#include <array>
#include <chameleon_renderer/utils/stdutils.hpp>

namespace chameleon
{
    struct LedRadialCharacteristic{
        std::string label;
        std::array<float,90> ies; // measured values for angle of normal 0-90 deg

        /**
         * @brief LedRadialCharacteristic constructor
         * @param ies_json_path path to normalized json file with ies values
         * 
         */
        LedRadialCharacteristic(const fs::path& ies_json_path);
        LedRadialCharacteristic()=default;

    };
} // namespace chameleon
