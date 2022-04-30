#pragma once

#pragma once
#include <chameleon_renderer/renderer/Renderer.hpp>
#include <chameleon_renderer/renderer/RenderLayers.hpp>
#include <chameleon_renderer/cuda/LaunchParams.h>
#include <chameleon_renderer/materials/barytex/MeasurementHit.hpp>
#include <chameleon_renderer/renderer/barytex/BarytexObservation.hpp>

namespace chameleon {

struct BarytexLearnRenderer : Renderer {
    using launch_params_t = barytex_show_render::LaunchParams;

    struct OutputData {
        VectorLayer<MeasurementHit> measurements;
        //metadata ? 

        void resize(size_t maximum_hitcount);
        void clear();
    };
    
    std::map<std::string, PhotometryCamera> photometry_cameras;
    OutputData out_data;
    CUDABuffer launch_params_buff;
    launch_params_t launch_params;

    BarytexLearnRenderer();

    void setup() override;

    void add_camera(const std::string& cam_label, const PhotometryCamera& cam);

    PhotometryCamera& photometry_camera(const std::string& cam_label);

    CalibratedCamera& camera(const std::string& cam_label);

    const OutputData& render(const BarytexObservation& observation);

    CUDAArray<MeasurementHit> get_cuda();
};
}  // namespace chameleon
