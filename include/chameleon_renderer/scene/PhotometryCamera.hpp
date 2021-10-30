#pragma once
#include <chameleon_renderer/utils/eigen_utils.hpp>

#include "Camera.hpp"
#include "Light.hpp"

namespace chameleon {
struct PhotometryCamera : public CalibratedCamera {
    using CalibratedCamera::get_cuda;

    PhotometryCamera() = default;

    PhotometryCamera(eigen_utils::Vec2<int> resolution,
                     eigen_utils::Mat3<float> camera_matrix)
        : CalibratedCamera(resolution, camera_matrix) {}

    PhotometryCamera(eigen_utils::Vec2<int> resolution,
                     eigen_utils::Mat3<float> camera_matrix,
                     eigen_utils::Mat4<float> view_matrix)
        : CalibratedCamera(resolution, camera_matrix, view_matrix) {}

    PhotometryCamera(eigen_utils::Vec2<int> resolution, Json j)
        : CalibratedCamera(resolution, j) {}

    std::vector<std::shared_ptr<ILight>> lights;
    CUDABuffer light_buff;
    bool is_actual = false;

    CudaLightArray get_cuda_light_array() {
        using namespace eigen_utils;
        std::vector<CudaLight> cuda_lights;

        cuda_lights.resize(lights.size());
        for (auto i = 0u; i < lights.size(); ++i) {
            auto lt = lights[i]->get_type();
            cuda_lights[i].l_type = lt;
            cuda_lights[i].intensity = lights[i]->get_intensity();
            if (lt == LightType::POINT || lt == LightType::SPOT) {
                cuda_lights[i].position = from_eigen_v3(homogenus_transform(
                    _object_matrix, lights[i]->get_position(), true));
            }
            if (lt == LightType::DIRECT || lt == LightType::SPOT) {
                cuda_lights[i].direction = from_eigen_v3(homogenus_transform(
                    _object_matrix, lights[i]->get_direction(), false));
            }
        }
        // TODO: reuse memory
        if (light_buff.d_ptr != 0) {
            light_buff.free();
        }
        light_buff.alloc_and_upload(cuda_lights);
        return {(CudaLight*)light_buff.d_ptr, cuda_lights.size()};
    }

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
