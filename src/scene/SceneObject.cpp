#include <chameleon_renderer/scene/SceneObject.hpp>

namespace chameleon {

SceneObject::SceneObject(mat_t obj_mat) : _object_matrix(std::move(obj_mat)) {}

SceneObject& SceneObject::operator=(SceneObject other) {
    _object_matrix = std::move(other._object_matrix);
    modified = true;
    return *this;
}
void SceneObject::set_object_matrix(mat_t mat) {
    _object_matrix.set(std::move(mat));
}

bool SceneObject::is_modified() const { return _object_matrix.is_modified(); }
const auto& SceneObject::object_matrix() { return _object_matrix->mat(); }
const auto& SceneObject::inv_object_matrix() { return _object_matrix->inv(); }
}  // namespace chameleon
