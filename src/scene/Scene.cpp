#include <chameleon_renderer/scene/Scene.hpp>

namespace chameleon {
Scene::Scene(unit unit, float rx, float ry, float rz) {
    using namespace eigen_utils;
    _unit_scale = to_homogenus<float, 3>(
        scale_mat(1 / float(unit), 1 / float(unit), 1 / float(unit)));
    _coordinate_transform = to_homogenus<float, 3>(euler_to_mat(rx, ry, rz));
    _transform = _unit_scale;
}

void Scene::set_model(Model m) { _model = std::move(m); }
void Scene::set_camera(CalibratedCamera c) { _camera = std::move(c); }

void Scene::set_lights(std::vector<glm::vec3> lights) {
    _light_positions = std::move(lights);
}

CalibratedCamera& Scene::camera() { return _camera; }
const CalibratedCamera& Scene::camera() const { return _camera; }

const chameleon::ModifyGuard<std::vector<glm::vec3>>& Scene::light_positions()
    const {
    return _light_positions;
}
chameleon::ModifyGuard<std::vector<glm::vec3>>& Scene::light_positions() {
    return _light_positions;
}

const Model& Scene::model() const { return _model; }

const Eigen::Matrix4f& Scene::transform() { return _transform; }

}  // namespace chameleon