#include <chameleon_renderer/scene/PhotometryCamera.hpp>

namespace chameleon {

PhotometryCamera::PhotometryCamera(eigen_utils::Vec2<int> resolution,
                                   eigen_utils::Mat3<float> camera_matrix)
    : CalibratedCamera(resolution, camera_matrix) {}

PhotometryCamera::PhotometryCamera(eigen_utils::Vec2<int> resolution,
                                   eigen_utils::Mat3<float> camera_matrix,
                                   eigen_utils::Mat4<float> view_matrix)
    : CalibratedCamera(resolution, camera_matrix, view_matrix) {}

PhotometryCamera::PhotometryCamera(eigen_utils::Vec2<int> resolution, Json j)
    : CalibratedCamera(resolution, j) {}

CUDALightArray PhotometryCamera::get_cuda_light_array() {
    using namespace eigen_utils;
    std::vector<CUDALight> cuda_lights;

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
    return {(CUDALight*)light_buff.d_ptr, cuda_lights.size()};
}

}  // namespace chameleon
