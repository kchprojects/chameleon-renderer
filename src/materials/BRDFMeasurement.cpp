#include <chameleon_renderer/materials/barytex/BRDFMeasurement.hpp>
#include <functional>
#include <glm/gtx/rotate_vector.hpp>
// #define DEBUG_MESSAGES
#include <chameleon_renderer/utils/debug_io.hpp>
#include <stdexcept>

namespace chameleon {

inline float sgn(float v) {
    if (v > std::numeric_limits<float>::epsilon()) {
        return 1;
    }
    if (v < -std::numeric_limits<float>::epsilon()) {
        return -1;
    }
    return 0;
}

static std::pair<float, float> vec_to_azimuth_elevation(glm::vec3 v) {
    glm::vec3 azimuth_axis;
    glm::vec3 v_z0;
    float azimuth;
    if (glm::length(v) == 0) {
        throw std::runtime_error("computing azimuth of zero vector");
    }
    v = glm::normalize(v);
    if (std::abs(std::abs(v.z) - 1) > std::numeric_limits<float>::epsilon()) {
        v_z0 = glm::normalize(glm::vec3(v.x, v.y, 0));
        azimuth_axis = glm::cross({1, 0, 0}, v_z0);
        azimuth = sgn(azimuth_axis.z) * glm::acos(v_z0.x);
    } else {
        v_z0 = {1, 0, 0};
        azimuth_axis = {0, 1, 0};
        azimuth = 0;
    }
    float elevation = M_PI / 2.f - glm::acos(v.z);
    return {azimuth, elevation};
}

IsotropicBRDFMeasurement::IsotropicBRDFMeasurement(const MeasurementHit& hit)
    : triangle_id(hit.triangle_id),world_coordinates(hit.world_coordinates) {
    glm::vec3 normal = glm::normalize(hit.mesh_normal);
    glm::vec3 axis_n;
    value = hit.value;
    if (std::abs(std::abs(normal.z) - 1) <
        10 * std::numeric_limits<float>::epsilon()) {
        axis_n = {0, 1, 0};
    } else {
        axis_n = glm::normalize(glm::cross(normal, {0, 0, 1}));
    }
    // print_vec(axis_n, "axis_n");
    float angle_n = std::acos(normal.z);
    // print_var(angle_n,"angle_n");
    glm::vec3 test_mesh_normal = glm::rotate(normal, angle_n, axis_n);
    // print_vec(test_mesh_normal, "test_mesh_normal");
    glm::vec3 eye = glm::rotate(glm::normalize(hit.eye), angle_n, axis_n);
    // print_vec(eye, "rot_eye");
    glm::vec3 light = glm::rotate(glm::normalize(hit.light), angle_n, axis_n);
    // print_vec(light, "rot_light");

    float eye_azimuth;
    std::tie(eye_azimuth,eye_elevation) = vec_to_azimuth_elevation(eye);
    // print_var(eye_azimuth, "eye_azimuth");
    // print_var(eye_elevation, "eye_elevation");
    light = glm::rotateZ(light, eye_azimuth);

    std::tie(light_azimuth, light_elevation) = vec_to_azimuth_elevation(light);
}

glm::vec3 IsotropicBRDFMeasurement::eye() const {
    return glm::rotateY(glm::vec3(1, 0, 0), -eye_elevation);
}
glm::vec3 IsotropicBRDFMeasurement::light() const {
    return glm::rotateZ(glm::rotateY(glm::vec3(1, 0, 0), -light_elevation),
                        light_azimuth);
}
}  // namespace chameleon
