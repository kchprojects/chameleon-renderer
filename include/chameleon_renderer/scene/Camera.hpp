#pragma once

#include <chameleon_renderer/cuda/CudaCamera.h>

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
    CalibratedCamera() { move_to(eigen_utils::Mat4<float>::Identity()); }

    CalibratedCamera(eigen_utils::Vec2<int> resolution,
                     eigen_utils::Mat3<float> camera_matrix)
        : _resolution(std::move(resolution))
    {
        set_camera_matrix(std::move(camera_matrix));
        move_to(eigen_utils::Mat4<float>::Identity());
    }

    CalibratedCamera(eigen_utils::Vec2<int> resolution,
                     eigen_utils::Mat3<float> camera_matrix,
                     eigen_utils::Mat4<float> view_matrix)
        : _resolution(std::move(resolution))
    {
        set_camera_matrix(std::move(camera_matrix));
        move_to(eigen_utils::Mat4<float>::Identity());
    }

    CalibratedCamera(eigen_utils::Vec2<int> resolution, Json j)
        : _resolution(std::move(resolution))
    {
        if (j.count("camera_matrix")) {
            auto& cam_mat_jsn = j.at("camera_matrix");
            _camera_matrix << cam_mat_jsn[0][0], cam_mat_jsn[0][1],
                cam_mat_jsn[0][2], cam_mat_jsn[1][0], cam_mat_jsn[1][1],
                cam_mat_jsn[1][2], cam_mat_jsn[2][0], cam_mat_jsn[2][1],
                cam_mat_jsn[2][2];
        } else if (j.count("K_undist")) {
            auto& cam_mat_jsn = j.at("K_undist").at("data");
            _camera_matrix << cam_mat_jsn[0], cam_mat_jsn[1], cam_mat_jsn[2],
                cam_mat_jsn[3], cam_mat_jsn[4], cam_mat_jsn[5], cam_mat_jsn[6],
                cam_mat_jsn[7], cam_mat_jsn[8];
        }
        if (j.count("imageSize")) {
            _resolution = { j.at("imageSize")[0], j.at("imageSize")[1] };
        }
        std::cout << "CAM < " << _camera_matrix << std::endl;
        std::cout << "RES < " << _resolution << std::endl;
        _camera_matrix_inverse = _camera_matrix.inverse();
        move_to(eigen_utils::Mat4<float>::Identity());
    }

    void move_to(const eigen_utils::Mat4<float>& mat)
    {
        _object_matrix = mat;
        _object_matrix_inverse = _object_matrix.inverse();
    }
    void move_by(const eigen_utils::Mat4<float>& mat)
    {
        _object_matrix = mat * _object_matrix;
        _object_matrix_inverse = _object_matrix.inverse();
    }

    void set_camera_matrix(eigen_utils::Mat3<float> m)
    {
        _camera_matrix = std::move(m);
        _camera_matrix_inverse = _camera_matrix.inverse();
    }
    const eigen_utils::Mat3<float>& camera_matrix() const
    {
        return _camera_matrix;
    }
    const auto& resolution() const { return _resolution; }

    const eigen_utils::Mat3<float>& inv_camera_matrix() const
    {
        return _camera_matrix_inverse;
    }

    eigen_utils::Vec3<float> position() const
    {
        return { _object_matrix(0, 3),
                 _object_matrix(1, 3),
                 _object_matrix(2, 3) };
    }

    CudaCamera get_cuda() const
    {
        CudaCamera out;
        out.camera_mat_inverse =
            eigen_utils::from_eigen_m3(_camera_matrix_inverse);
        out.camera_mat = eigen_utils::from_eigen_m3(_camera_matrix);
        out.obj_mat = eigen_utils::from_eigen_m4(_object_matrix);
        out.obj_mat_inverse =
            eigen_utils::from_eigen_m4(_object_matrix_inverse);
        out.pos = eigen_utils::from_eigen_homogenus(
            _object_matrix * eigen_utils::Vec4<float>(0, 0, 0, 1));
        std::cout << out.pos.x << "," << out.pos.y << "," << out.pos.z
                  << std::endl;
        // std::cout << out.camera_mat.r1.x << " " << out.camera_mat.r1.y << " "
        //           << out.camera_mat.r1.z << std::endl
        //           << out.camera_mat.r2.x << " " << out.camera_mat.r2.y << " "
        //           << out.camera_mat.r2.z << std::endl
        //           << out.camera_mat.r3.x << " " << out.camera_mat.r3.y << " "
        //           << out.camera_mat.r3.z << std::endl;
        out.res = { _resolution(0), _resolution(1) };
        return out;
    }
};
} // namespace chameleon