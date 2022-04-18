#pragma once
#include <chameleon_renderer/utils/math_utils.hpp>
namespace chameleon {

/**
 * @brief Struct representing ray hit
 *
 */
struct MeasurementHit {
    unsigned int triangle_id;  // index of hitted triangle
    glm::vec3 coordinates;     // barycentric coordinates inside hitted tringle
    glm::vec3 world_coordinates;     // world coordinates of hit

    glm::vec3 eye;    // vector to camera sized by distance
    glm::vec3 light;  // vector to light sized by distance
    glm::vec3 mesh_normal;

    glm::vec3 value;  // value of observed pixel

    bool is_valid = false;  // hit is not corrupted by shadow
};
}  // namespace chameleon