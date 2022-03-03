#pragma once
#include <chameleon_renderer/utils/eigen_utils.hpp>
#include <chameleon_renderer/utils/math_utils.hpp>

#include "Camera.hpp"
#include "Model.hpp"
#include "SceneObject.hpp"

namespace chameleon {
enum unit {
    mm = 1,
    cm = 10,
    dm = 100,
    m = 1000,
};

class Scene {
    Eigen::Matrix4f _transform;
    Eigen::Matrix4f _unit_scale;
    Eigen::Matrix4f _coordinate_transform;

    Model _model;
    CalibratedCamera _camera;
    ModifyGuard<std::vector<vec3f>> _light_positions;

   public:
    Scene(unit unit, float rx = 0, float ry = 0, float rz = 0);

    void set_model(Model m);
    void set_camera(CalibratedCamera c);

    void set_lights(std::vector<vec3f> lights);

    CalibratedCamera& camera();
    const CalibratedCamera& camera() const;

    const chameleon::ModifyGuard<std::vector<gdt::vec3f>>& light_positions()
        const;
    chameleon::ModifyGuard<std::vector<gdt::vec3f>>& light_positions();

    const Model& model() const;

    const Eigen::Matrix4f& transform();
};
}  // namespace chameleon