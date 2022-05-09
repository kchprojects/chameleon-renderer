#pragma once

#pragma once
#include <chameleon_renderer/renderer/Renderer.hpp>
#include <chameleon_renderer/renderer/RenderLayers.hpp>
#include <chameleon_renderer/cuda/LaunchParams.h>
#include <chameleon_renderer/materials/barytex/MaterialLookupForest.hpp>
#include <chameleon_renderer/renderer/barytex/BarytexObservation.hpp>

namespace chameleon {

struct BarytexShowRenderer : Renderer {
    using launch_params_t = barytex_show_render::LaunchParams;

    struct OutputData {
        ImageLayer<cv::Vec3f> view;
        ImageLayer<uchar> mask;

        void resize(int x, int y);
        void clear();
    };
    
    std::map<std::string, PhotometryCamera> photometry_cameras;
    OutputData out_data;
    CUDABuffer launch_params_buff;
    launch_params_t launch_params;
    
    BarytexShowRenderer();

    void setup() override;

    void add_camera(const std::string& cam_label, const PhotometryCamera& cam);

    PhotometryCamera& photometry_camera(const std::string& cam_label);

    CalibratedCamera& camera(const std::string& cam_label);

    const OutputData& render(const std::string& camera_label);

    launch_params_t::Layers get_cuda_output();

    void setup_material(const MaterialLookupForest& mlf);


private:
    std::optional<CUDAMaterialLookupForest> material_forest;

};
}  // namespace chameleon
