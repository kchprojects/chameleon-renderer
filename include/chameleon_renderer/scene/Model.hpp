#pragma once

#include <chameleon_renderer/utils/math_utils.hpp>
#include <chameleon_renderer/utils/file_utils.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <utility>

#include <vector>

#include "SceneObject.hpp"

namespace chameleon {

/*! a simple indexed triangle mesh that our sample renderer will
    render */
struct TriangleMesh {
    std::vector<glm::vec3> vertex;
    std::vector<glm::vec3> normal;
    std::vector<glm::vec2> texcoord;
    std::vector<vec3i> index;
    // material data:
    glm::vec3 diffuse;
    int diffuseTextureID{-1};

    struct TraversableTriangleMesh {
        struct Vertex;
        struct Edge;
        struct Face;

        struct Face {
            std::array<int, 3> vert_indices;
        };

        struct Edge {
            std::array<int, 2> face_indices={-1,-1};
            struct Hash {
                std::size_t operator()(const std::pair<int, int>& pair) const {
                    return std::hash<int>()(pair.first) ^
                           std::hash<int>()(pair.second);
                }
            };
        };
        struct Vertex {
            std::vector<int> face_indices;
        };

        TraversableTriangleMesh(const TriangleMesh&);

        void export_ttm(const fs::path& path)const;

        const TriangleMesh& base_mesh;
        std::vector<Face> faces;
        std::unordered_map<std::pair<int, int>, Edge, Edge::Hash> edges;
        std::vector<Vertex> vertices;
    };

    const TraversableTriangleMesh& traversable_mesh() const {
        if (!_traversable_mesh.has_value()) {
            _make_traversable();
        }
        return *_traversable_mesh;
    }
    
    TriangleMesh()=default;
    TriangleMesh(const TriangleMesh& other)
        : vertex(other.vertex),
          normal(other.normal),
          texcoord(other.texcoord),
          index(other.index),
          diffuse(other.diffuse),
          diffuseTextureID(other.diffuseTextureID) {}
    
    void swap(TriangleMesh& other){
        using std::swap;
        swap(vertex,vertex);
        swap(normal,normal);
        swap(texcoord,texcoord);
        swap(index,index);
        swap(diffuse,diffuse);
        swap(diffuseTextureID,diffuseTextureID);
    }

    TriangleMesh& operator=(TriangleMesh other){
        //copy and swap
        swap(other);
        return *this;
    }

   private:
    void _make_traversable() const { _traversable_mesh.emplace(*this); }
    mutable std::optional<TraversableTriangleMesh> _traversable_mesh;
};

struct Texture {
    cv::Mat image;
    vec2i resolution{-1};
    int id = -1;

    const void* pixel() { return image.data; }
};
struct Model {
    std::vector<TriangleMesh> meshes;
    std::vector<Texture> textures;
    //! bounding box of all vertices in the model
    auto& mesh(int id) { return meshes[id]; }
    const auto& mesh(int id) const { return meshes[id]; }
    auto& texture(int id) { return textures[id]; }
    const auto& texture(int id) const { return textures[id]; }
    box3f bounds;

    friend Model loadOBJ(const std::string& objFile);
};

Model loadOBJ(const std::string& objFile);
}  // namespace chameleon
