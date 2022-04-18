#include <spdlog/spdlog.h>

#include <chameleon_renderer/HW/LedRadialCharacteristic.hpp>
#include <chameleon_renderer/utils/terminal_utils.hpp>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
namespace chameleon {
LedRadialCharacteristic::LedRadialCharacteristic(
    const fs::path& ies_json_path) {
    PING;
    nlohmann::json ies_j;
    {
        std::ifstream ifs(ies_json_path);
        PING;
        if (!ifs) {
            spdlog::error("cannot load file {}", ies_json_path.string());
            throw std::invalid_argument("cannot load file");
        }
        PING;
        ifs >> ies_j;
    }
    label = ies_j["led_model"];
    PING;
    auto att_j = ies_j["radial_attenuation"];
    if (att_j.size() < ies.size()) {
        spdlog::error("Wrong size of ies file");
        throw std::invalid_argument("Wrong size of ies file");
    }
    for (auto i = 0u; i < ies.size(); ++i) {
        ies[i] = att_j[i].get<float>();
    }
    for (auto i : ies) {
        std::cout << i << std::endl;
    }
    PING;
}
}  // namespace chameleon
