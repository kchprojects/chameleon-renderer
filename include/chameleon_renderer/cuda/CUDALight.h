#pragma once

#include <chameleon_renderer/utils/math_utils.hpp>
#include <chameleon_renderer/cuda/CUDAArray.h>
namespace chameleon {
enum LightType
{
    SPOT,
    POINT,
    DIRECT,
    LED_LIGHT
};

struct CUDALight
{
    LightType l_type;
    glm::vec3 position;
    glm::vec3 direction;
    float intensity;
    glm::vec3 color;
    CUDAArray<float> radial_attenuation;
};

struct CUDALightArray
{
    CUDALight* data;
    size_t count;
};

} // namespace chameleon
