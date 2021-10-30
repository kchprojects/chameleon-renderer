#pragma once

#include <memory>
#include <vector>

#include <chameleon_renderer/utils/math_utils.hpp>
#include "SceneObject.hpp"
#include <opencv2/opencv.hpp>

namespace chameleon {
// using namespace gdt;

/*! a simple indexed triangle mesh that our sample renderer will
    render */
struct TriangleMesh {
    std::vector<vec3f> vertex;
    std::vector<vec3f> normal;
    std::vector<vec2f> texcoord;
    std::vector<vec3i> index;

    // material data:
    vec3f diffuse;
    int diffuseTextureID{-1};
};

struct Texture {

    cv::Mat image;    
    vec2i resolution{-1};
    int id = -1;

    const void* pixel(){
        return image.data;
    }
};
struct Model{
    std::vector<TriangleMesh> meshes;
    std::vector<Texture> textures;
    //! bounding box of all vertices in the model
    auto& mesh(int id) { return meshes[id]; }
    const auto& mesh(int id) const { return meshes[id]; }
    auto& texture(int id) { return textures[id]; }
    const auto& texture(int id) const { return textures[id]; }
    box3f bounds;

};

Model loadOBJ(const std::string& objFile);
}  // namespace chameleon
