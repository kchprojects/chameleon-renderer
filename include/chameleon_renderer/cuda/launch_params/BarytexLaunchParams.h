#pragma once
#include <chameleon_renderer/utils/math_utils.hpp>
#include <chameleon_renderer/cuda/CudaCamera.h>
#include <chameleon_renderer/cuda/CudaLight.h>
#include <chameleon_renderer/utils/optix7.h>
#include <chameleon_renderer/materials/barytex/MeasurementHit.hpp>

namespace chameleon {

namespace barycentric_learn_render {

struct LaunchParams {
    // render data
    struct RenderData {
        MeasurementHit* out_data;
        size_t out_size;

        glm::vec3** captured_images;
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
