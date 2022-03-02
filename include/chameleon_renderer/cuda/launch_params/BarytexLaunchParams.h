#pragma once

#include <chameleon_renderer/cuda/CudaCamera.h>
#include <chameleon_renderer/cuda/CudaLight.h>
#include <chameleon_renderer/utils/optix7.hpp>
#include <curand.h>  // ? why

#include <chameleon_renderer/materials/barytex/MeasurementHit.hpp>

namespace chameleon {

namespace barycentric_learn_render {

struct LaunchParams {
    // render data
    struct RenderData {
        MeasurementHit* out_data;
        size_t out_size;

        vec3f** captured_images;
        size_t width;
        size_t height;
        size_t img_count;
    };
    RenderData render_data;

    // setup data
    CudaCamera camera;
    CudaLightArray light_data;

    // optix
    OptixTraversableHandle traversable;
};
}  // namespace barycentric_learn_render
}  // namespace chameleon
