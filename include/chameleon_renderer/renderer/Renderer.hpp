#pragma once
#include <chameleon_renderer/optix/OptixPrograms.hpp>
#include <chameleon_renderer/optix/OptixScene.hpp>
#include <chameleon_renderer/renderer/RenderLayers.hpp>
#include <chameleon_renderer/scene/PhotometryCamera.hpp>
#include <chameleon_renderer/utils/benchmarking.hpp>
#include <opencv2/opencv.hpp>

namespace chameleon {

struct Renderer
{
    virtual void setup() = 0;

    void add_camera(const std::string& cam_label, const CalibratedCamera& cam);

    CalibratedCamera& camera(const std::string& cam_label);
    
    void setup_scene(const OptixStaticScene& new_scene);

    virtual ~Renderer() = default;

protected:
    OptixStaticScene scene;
    OptixPrograms programs;
    OptixPipeline pipeline;

    OptixModule raygen_module;
    OptixModule miss_module;
    OptixModule anyhit_module;
    OptixModule closest_hit_module;

    std::map<std::string, CalibratedCamera> cameras;

    OptixModuleCompileOptions moduleCompileOptions{
        50,
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
        OPTIX_COMPILE_DEBUG_LEVEL_NONE,
        nullptr,
        0
    };

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions pipelineLinkOptions = {};
};

} // namespace chameleon
