#pragma once
#include <chameleon_renderer/utils/math_utils.hpp>
namespace chameleon {
struct CUDACamera {
    glm::mat4 obj_mat;
    glm::mat4 obj_mat_inverse;
    glm::mat3 camera_mat;
    glm::mat3 camera_mat_inverse;
    glm::vec3 pos;
    vec2i res;
};

#ifdef __CUDA_ARCH__
inline glm::vec3 __device__ get_pos(const glm::mat4& view_matrix) {
    return {view_matrix[0][4], view_matrix[1][4], view_matrix[2][4]};
}
#endif
}  // namespace chameleon