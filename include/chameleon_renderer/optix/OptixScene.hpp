#pragma once
#include <chameleon_renderer/cuda/CUDABuffer.hpp>

#include <chameleon_renderer/optix/OptixContext.hpp>
#include <chameleon_renderer/scene/Model.hpp>
#include <chameleon_renderer/scene/Scene.hpp>
#include <chameleon_renderer/utils/eigen_utils.hpp>

namespace chameleon {
struct SceneModel : public Model
{
    eigen_utils::Mat4<float> obj_mat;

    SceneModel();
    SceneModel(Model m);

    SceneModel(const std::string& model_path);

    void update();

    void transform(const eigen_utils::Mat4<float>& transform_mat);
};

struct OptixStaticScene
{
    struct MeshBuffers
    {
        std::vector<CUDABuffer> vertex;
        std::vector<CUDABuffer> normal;
        std::vector<CUDABuffer> texcoord;
        std::vector<CUDABuffer> index;

        void resize(int numMeshes);

        void upload(int meshID, TriangleMesh& mesh);
        size_t size() const;

        void clear();
    };

    OptixStaticScene();

    void setupAccel();

    void createTextures();

    void add_model(SceneModel sm);

    inline auto& models() { return _models; }
    inline const auto& models() const { return _models; }

    inline auto& mesh_buffers() { return _mesh_buffers; }
    inline const auto& mesh_buffers() const { return _mesh_buffers; }

    inline const auto& meshes() const { return _meshes; }
    inline const auto& texture_objects() const { return textureObjects; }

    int num_meshes;
    OptixTraversableHandle traversable;

protected:
    std::vector<SceneModel> _models;
    std::vector<TriangleMesh> _meshes;
    CUDABuffer asBuffer;

    MeshBuffers _mesh_buffers;
    OptixAccelBuildOptions accelOptions;

    std::vector<cudaArray_t> textureArrays;
    std::vector<cudaTextureObject_t> textureObjects;
};
} // namespace chameleon