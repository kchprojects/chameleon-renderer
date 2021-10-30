#pragma once
#include <chameleon_renderer/utils/math_utils.hpp>
namespace chameleon {
struct CudaCamera {
    mat4f obj_mat;
    mat4f obj_mat_inverse;
    mat3f camera_mat;
    mat3f camera_mat_inverse;
    vec3f pos;
    vec2i res;
};

inline vec3f __device__ get_pos(mat4f view_matrix) {
    return {view_matrix.r1.w, view_matrix.r2.w, view_matrix.r3.w};
}

}  // namespace chameleon