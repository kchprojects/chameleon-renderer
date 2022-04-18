#pragma once

#include <chameleon_renderer/cuda/CUDACamera.h>

#include <chameleon_renderer/utils/eigen_utils.hpp>
#include <chameleon_renderer/utils/json.hpp>
#include <chameleon_renderer/utils/math_utils.hpp>
#include <chameleon_renderer/utils/terminal_utils.hpp>

namespace chameleon {

class CalibratedCamera
{
protected:
    eigen_utils::Vec2<int> _resolution;

    eigen_utils::Mat3<float> _camera_matrix;
    eigen_utils::Mat3<float> _camera_matrix_inverse;

    eigen_utils::Mat4<float> _object_matrix;
    eigen_utils::Mat4<float> _object_matrix_inverse;

public:
    CalibratedCamera();

    CalibratedCamera(eigen_utils::Vec2<int> resolution,
                     eigen_utils::Mat3<float> camera_matrix);

    CalibratedCamera(eigen_utils::Vec2<int> resolution,
                     eigen_utils::Mat3<float> camera_matrix,
                     eigen_utils::Mat4<float> view_matrix);

    CalibratedCamera(eigen_utils::Vec2<int> resolution, Json j);

    void move_to(const eigen_utils::Mat4<float>& mat);
    void move_by(const eigen_utils::Mat4<float>& mat);

    void set_camera_matrix(eigen_utils::Mat3<float> m);
    const eigen_utils::Mat3<float>& camera_matrix() const;
    const Eigen::Vector2i& resolution() const;

    const eigen_utils::Mat3<float>& inv_camera_matrix() const;

    eigen_utils::Vec3<float> position() const;
    eigen_utils::Mat4<float> object_matrix() const;

    CUDACamera get_cuda() const;
};
} // namespace chameleon