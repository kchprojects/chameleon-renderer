#pragma once 
#include <unordered_map>
#include <memory>
#include <chameleon_renderer/scene/Light.hpp>
#include <nlohmann/json.hpp>

namespace chameleon
{
    struct HWSetup
    {
        std::unordered_map<int,std::shared_ptr<ILight>> lights;

        HWSetup(const nlohmann::json& setup_json);

    };
    
} // namespace chameleon
