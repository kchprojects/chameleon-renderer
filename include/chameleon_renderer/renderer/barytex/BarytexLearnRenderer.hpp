#pragma once
#include <chameleon_renderer/renderer/Renderer.hpp>
#include <chameleon_renderer/renderer/RenderLayers.hpp>
#include <chameleon_renderer/cuda/LaunchParams.h>
#include <chameleon_renderer/materials/barytex/MeasurementHit.hpp>

namespace chameleon {

struct BarytexLearnRenderer : Renderer {
    using launch_params_t = barycentric_learn_render::LaunchParams;

    struct OutputData {
        VectorLayer<MeasurementHit> measurements;
        //metadata ? 

        void resize(size_t maximum_hitcount);
        void clear();
    };

    struct InputData {
        std::vector<InputImageLayer<cv::Vec3b>> captured_images;
        int height = -1;
        int width = -1;
        CUDABuffer images_ptr_array;
        //metadata ? 
        void upload_source_images(std::vector<cv::Mat> images); 
        void clear();
    };

    
    std::map<std::string, PhotometryCamera> photometry_cameras;
    OutputData out_data;
    InputData in_data;
    CUDABuffer launch_params_buff;
    launch_params_t launch_params;

    BarytexLearnRenderer();

    void setup() override;

    void upload_source_img(cv::Mat img);

    void add_camera(const std::string& cam_label, const PhotometryCamera& cam);

    PhotometryCamera& photometry_camera(const std::string& cam_label);

    CalibratedCamera& camera(const std::string& cam_label);

    const OutputData& render(const std::string& camera_label);

    launch_params_t::RenderData get_cuda();
};
}  // namespace chameleon
