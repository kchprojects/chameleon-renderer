#pragma once
#include <chameleon_renderer/cuda/LaunchParams.h>

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

struct PhotometryRenderer : Renderer
{
    using launch_params_t = photometry_render::LaunchParams;

    struct OutputLayers
    {
        ImageLayer<cv::Vec3f> normal_map;
        ImageLayer<cv::Vec3f> uv_map;
        ImageLayer<cv::Vec3f> view;
        ImageLayer<uchar> mask;

        void resize(size_t x, size_t y);
        void clear();

        launch_params_t::Layers get_cuda();
    };

    std::map<std::string, PhotometryCamera> photometry_cameras;
    OutputLayers out_layers;
    CUDABuffer launch_params_buff;
    launch_params_t launch_params;

    PhotometryRenderer();

    void setup() override;

    void add_camera(const std::string& cam_label, const PhotometryCamera& cam);

    PhotometryCamera& photometry_camera(const std::string& cam_label);

    CalibratedCamera& camera(const std::string& cam_label);

    const OutputLayers& render(const std::string& camera_label);

    launch_params_t::Layers get_cuda();
};

} // namespace chameleon
