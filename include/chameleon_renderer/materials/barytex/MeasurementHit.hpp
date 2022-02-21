#include <glm/glm.hpp>
namespace chameleon {

/**
 * @brief Struct representing ray hit
 *
 */
struct MeasurementHit {
    unsigned int triangle_id;  // index of hitted triangle
    glm::vec3 coordinates;     // barycentric coordinates inside hitted tringle

    glm::vec3 eye;    // vector to camera
    glm::vec3 light;  // vector to light

    glm::vec3 value;  // value of observed pixel

    bool is_valid = false;  // hit is not corrupted by shadow
};
}  // namespace chameleon