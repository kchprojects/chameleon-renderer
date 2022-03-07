#pragma once

namespace chameleon {

namespace photometry_render {
struct RadiationRayDataBase {
    glm::vec3 normal;
    glm::vec3 uv;
    glm::vec3 view;
    uint8_t mask;
};

struct RadiationRayData {
    glm::vec3* normal;
    glm::vec3* uv;
    glm::vec3* view;
    uint8_t* mask;
};

}  // namespace photometry_render

}  // namespace chameleon
