#pragma once

#include <chameleon_renderer/utils/optix7.h>
#include <chameleon_renderer/utils/math_utils.hpp>
#include <chameleon_renderer/cuda/CUDACamera.h>
#include <chameleon_renderer/cuda/CUDALight.h>

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
    CUDACamera camera;
    OptixTraversableHandle traversable;
    CUDALightArray light_data;
};
} // namespace photometry_render
} // namespace chameleon
