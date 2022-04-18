#include <unordered_map>
#include <chameleon_renderer/HW/HWSetup.hpp>
#include <chameleon_renderer/utils/terminal_utils.hpp>
#include <chameleon_renderer/scene/Light.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
namespace chameleon
{
    HWSetup::HWSetup(const nlohmann::json& setup_json){
        std::unordered_map<std::string,LedRadialCharacteristic> characteristics;
        PING;
        for(const auto& [model_id,model_json] : setup_json["models"].items()){ 
            std::string ies_file = model_json["characteristics_file"];
            PING;
            characteristics[model_id] = LedRadialCharacteristic(ies_file);
            PING;
        }
        PING;
        for(const nlohmann::json& light_json :setup_json["lights"]){
            eigen_utils::Vec3<float> pos;
            pos << light_json["position"]["x"],light_json["position"]["y"],light_json["position"]["z"];
            eigen_utils::Vec3<float> dir;
            pos << light_json["direction"]["x"],light_json["direction"]["y"],light_json["direction"]["z"];
            PING;
            if(light_json.count("led_model") > 0 && characteristics.count(light_json["led_model"]) > 0){
                lights[light_json["id"]] = std::make_shared<LedLight>();
                lights[light_json["id"]]->set_radial(characteristics.at(light_json["led_model"]));
            }else{
                lights[light_json["id"]] = std::make_shared<SpotLight>();
            }
            PING;
            lights[light_json["id"]]->set_direction(dir);
            lights[light_json["id"]]->set_position(pos);
        }
    }
    
} // namespace chameleon
