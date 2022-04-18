#include <chameleon_renderer/HW/LedRadialCharacteristic.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <nlohmann/json.hpp>
namespace chameleon {
LedRadialCharacteristic::LedRadialCharacteristic(
    const fs::path& ies_json_path) {
        std:: ifstream ifs(ies_json_path);
        if(!ifs){
            spdlog::error("cannot load file {}",ies_json_path.string());
            throw std::invalid_argument("cannot load file");
        }
        nlohmann::json ies_j;
        ifs >> ies_j;
        label = ies_j["led_model"];
        for (auto i = 0u; i < ies.size(); ++i){
            ies[i] = ies_j["radial_attenuation"][i].get<float>();
        }

    }
}  // namespace chameleon
