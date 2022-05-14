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
    cuda_lights.reserve(lights.size());
    for (auto i = 0u; i < lights.size(); ++i) {
        cuda_lights.push_back(lights[i]->convert_to_cuda(_object_matrix));
    }
    if(light_buff.d_ptr != nullptr){
        light_buff.free();
        
    }
    light_buff.alloc_and_upload(cuda_lights);
    return {(CUDALight*)light_buff.d_ptr, cuda_lights.size()};
}

}  // namespace chameleon
