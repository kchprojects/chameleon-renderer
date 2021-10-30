#pragma once

#include <chameleon_renderer/utils/optix7.h>

#include <chameleon_renderer/utils/math_utils.hpp>

namespace chameleon {
struct TriangleMeshSBTData {
    vec3f color;
    vec3f* vertex;
    vec3f* normal;
    vec2f* texcoord;
    vec3i* index;

    bool hasTexture;
    cudaTextureObject_t texture;
};
}  // namespace chameleon