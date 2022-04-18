#pragma once
#include <opencv2/opencv.hpp>
#include <chameleon_renderer/scene/Light.hpp>
#include <chameleon_renderer/renderer/RenderLayers.hpp>
#include <chameleon_renderer/renderer/barytex/CUDABarytexObservation.cuh>

namespace chameleon
{
    struct BarytexObservation{
        std::string cam_label;
        cv::Mat_<cv::Vec3b> image;
        std::shared_ptr<ILight> light;


        CUDABarytexObservation get_cuda(const eigen_utils::Mat4<float>& obj_mat = eigen_utils::Mat4<float>::Identity()) const;
    };

} // namespace chameleon

