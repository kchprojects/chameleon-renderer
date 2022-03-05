#pragma once

#include <chameleon_renderer/utils/optix7.h>

#include <chameleon_renderer/utils/math_utils.hpp>

namespace chameleon {
struct TriangleMeshSBTData
{
    glm::vec3 color;
    glm::vec3* vertex;
    glm::vec3* normal;
    glm::vec2* texcoord;
    vec3i* index;

    bool hasTexture;
    cudaTextureObject_t texture;
};
} // namespace chameleon