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
    glm::vec3 position;
    glm::vec3 direction;
    float intensity;
    glm::vec3 color;

};

struct CudaLightArray
{
    CudaLight* data;
    size_t count;
};

} // namespace chameleon
#endif // CUDALIGHT_H