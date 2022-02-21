#ifndef CUDALIGHT_H
#define CUDALIGHT_H

#include <chameleon_renderer/utils/math_utils.hpp>
namespace chameleon {
enum LightType
{
    SPOT,
    POINT,
    DIRECT
};

struct CudaLight
{
    LightType l_type;
    vec3f position;
    vec3f direction;
    float intensity;
};

struct CudaLightArray
{
    CudaLight* data;
    size_t count;
};

} // namespace chameleon
#endif // CUDALIGHT_H