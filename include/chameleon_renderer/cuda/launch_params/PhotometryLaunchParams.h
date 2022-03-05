#pragma once

#include <chameleon_renderer/utils/optix7.hpp>
#include <chameleon_renderer/utils/math_utils.hpp>
#include <chameleon_renderer/cuda/CudaCamera.h>
#include <chameleon_renderer/cuda/CudaLight.h>

namespace chameleon {

namespace photometry_render {

struct LaunchParams
{
    struct Layers
    {
        glm::vec3* normal_map;
        uint8_t* mask;
        glm::vec3* uv_map;
        glm::vec3* view;
        vec2i size;
    };
    Layers layers;
    CudaCamera camera;
    OptixTraversableHandle traversable;
    CudaLightArray light_data;
};
} // namespace photometry_render
} // namespace chameleon
