#pragma once

namespace chameleon {

namespace photometry_render {
struct RadiationRayDataBase {
    vec3f normal;
    vec3f uv;
    vec3f view;
    uint8_t mask;
};

struct RadiationRayData {
    vec3f* normal;
    vec3f* uv;
    vec3f* view;
    uint8_t* mask;
};

}  // namespace photometry_render

}  // namespace chameleon
