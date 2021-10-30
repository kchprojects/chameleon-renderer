#pragma once

#include <gdt/math/AffineSpace.h>
#include <gdt/math/vec.h>
namespace chameleon {
using gdt::box3f;
using gdt::vec2f;
using gdt::vec2i;
using gdt::vec3f;
using gdt::vec3i;
using gdt::vec3uc;
using gdt::vec4f;

struct mat3f {
    vec3f r1;
    vec3f r2;
    vec3f r3;
};

struct mat4f {
    vec4f r1;
    vec4f r2;
    vec4f r3;
    vec4f r4;
};

struct affine_mat4f {
    mat3f R;
    vec3f translation;
};

}  // namespace chameleon