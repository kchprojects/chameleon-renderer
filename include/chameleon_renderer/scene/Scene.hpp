#pragma once
#include "Camera.hpp"
#include "Model.hpp"
#include "SceneObject.hpp"
#include <chameleon_renderer/utils/eigen_utils.hpp>
#include <chameleon_renderer/utils/math_utils.hpp>

namespace chameleon {
enum unit
{
    mm = 1,
    cm = 10,
    dm = 100,
    m = 1000,
};

class Scene
{
    Eigen::Matrix4f _transform;
    Eigen::Matrix4f _unit_scale;
    Eigen::Matrix4f _coordinate_transform;

    Model _model;
    CalibratedCamera _camera;
    ModifyGuard<std::vector<vec3f>> _light_positions;

public:
    Scene(unit unit, float rx = 0, float ry = 0, float rz = 0)
    {
        using namespace eigen_utils;
        _unit_scale = to_homogenus<float, 3>(
            scale_mat(1 / float(unit), 1 / float(unit), 1 / float(unit)));
        _coordinate_transform =
            to_homogenus<float, 3>(euler_to_mat(rx, ry, rz));
        _transform = _unit_scale;
    }

    void set_model(Model m) { _model = std::move(m); }
    void set_camera(CalibratedCamera c) { _camera = std::move(c); }

    void set_lights(std::vector<vec3f> lights)
    {
        _light_positions = std::move(lights);
    }

    auto& camera() { return _camera; }
    const auto& light_positions() const { return _light_positions; }
    auto& light_positions() { return _light_positions; }
    const auto& camera() const { return _camera; }

    const auto& model() const { return _model; }

    const auto& transform() { return _transform; }
};
} // namespace chameleon