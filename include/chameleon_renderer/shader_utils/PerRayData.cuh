#pragma once
#include <chameleon_renderer/materials/barytex/MeasurementHit.hpp>
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

namespace barytex_learn_render {
struct RadiationRayData {
    MeasurementHit hit;
};

}  // namespace barytex_learn_render

}  // namespace chameleon
