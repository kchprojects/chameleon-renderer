#pragma once
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

namespace chameleon {

template<typename T>
inline nlohmann::json to_json(const T& val){
    return val; 
}

template<>
inline nlohmann::json to_json(const glm::vec3& vec) {
    auto out = nlohmann::json::array();
    out[0] = vec.x;
    out[1] = vec.y;
    out[2] = vec.z;
    return out;
}

inline glm::vec3 vec_from_json(const nlohmann::json& in) {
    glm::vec3 vec;
    vec.x = in[0];
    vec.y = in[1];
    vec.z = in[2];
    return vec;
}

}  // namespace chameleon