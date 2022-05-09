#pragma once
#include <chameleon_renderer/utils/math_utils.hpp>
#include <chameleon_renderer/cuda/CUDACamera.h>
#include <chameleon_renderer/cuda/CUDAArray.h>
#include <chameleon_renderer/renderer/barytex/CUDABarytexObservation.cuh>
#include <chameleon_renderer/materials/barytex/CUDAMaterialLookupForest.hpp>
#include <chameleon_renderer/utils/optix7.h>
#include <chameleon_renderer/materials/barytex/MeasurementHit.hpp>

namespace chameleon {

namespace barytex_learn_render {

struct LaunchParams {

    CUDAArray<MeasurementHit> render_data;

    // setup data
    CUDACamera camera;
    CUDABarytexObservation observation;

    // optix
    OptixTraversableHandle traversable;

    const int sample_multiplier = 5;
};
}  // namespace barycentric_learn_render

namespace barytex_show_render {

struct LaunchParams {

    struct Layers
    {
        uint8_t* mask;
        glm::vec3* view;
        vec2i size;
    };
    Layers render_data;

    // setup data
    CUDACamera camera;
    CUDALightArray light_data;
    CUDAMaterialLookupForest material_forest; 

    // optix
    OptixTraversableHandle traversable;
};
}  // namespace barycentric_learn_render
}  // namespace chameleon
