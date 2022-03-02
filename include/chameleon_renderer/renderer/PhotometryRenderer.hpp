#pragma once
#include <chameleon_renderer/renderer/Renderer.hpp>
#include <chameleon_renderer/cuda/LaunchParams.h>

namespace chameleon {

struct PhotometryRenderer : Renderer {
    using launch_params_t = photometry_render::LaunchParams;

    struct OutputLayers {
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

    const OutputLayers& render(const std::string& camera_label);

    launch_params_t::Layers get_cuda();
};
}  // namespace chameleon
