#pragma once

#include <chameleon_renderer/utils/optix7.hpp>
#include <curand.h>

#include <chameleon_renderer/utils/math_utils.hpp>

#include <chameleon_renderer/cuda/CudaCamera.h>
#include <chameleon_renderer/cuda/CudaLight.h>

namespace chameleon {

namespace photometry_render {

struct LaunchParams
{
    struct Layers
    {
        vec3f* normal_map;
        uint8_t* mask;
        vec3f* uv_map;
        vec3f* view;
        vec2i size;
    };
    Layers layers;
    CudaCamera camera;
    OptixTraversableHandle traversable;
    CudaLightArray light_data;
};
} // namespace photometry_render
} // namespace chameleon