#pragma once
#include <chameleon_renderer/utils/eigen_utils.hpp>

#include <chameleon_renderer/scene/Camera.hpp>
#include <chameleon_renderer/cuda/CUDABuffer.hpp>
#include <chameleon_renderer/scene/Light.hpp>

namespace chameleon {
struct PhotometryCamera : public CalibratedCamera {
    using CalibratedCamera::get_cuda;

    std::vector<std::shared_ptr<ILight>> lights;
    CUDABuffer light_buff;
    bool is_actual = false;

    PhotometryCamera() = default;

    PhotometryCamera(eigen_utils::Vec2<int> resolution,
                     eigen_utils::Mat3<float> camera_matrix);

    PhotometryCamera(eigen_utils::Vec2<int> resolution,
                     eigen_utils::Mat3<float> camera_matrix,
                     eigen_utils::Mat4<float> view_matrix);

    PhotometryCamera(eigen_utils::Vec2<int> resolution, Json j);

    CUDALightArray get_cuda_light_array();

    template <typename light_t>
    void add_light(light_t l) {
        lights.push_back(std::make_shared(std::move(l)));
    }

    template <typename light_t>
    void add_light(std::unique_ptr<light_t> l) {
        lights.push_back(std::move(l));
    }

    void set_lights(std::vector<std::shared_ptr<ILight>> l) {
        lights = std::move(l);
    }
};

}  // namespace chameleon
