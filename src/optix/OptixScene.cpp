#include <chameleon_renderer/optix/OptixScene.hpp>

namespace chameleon {

SceneModel::SceneModel() = default;

SceneModel::SceneModel(Model m)
    : Model(std::move(m))
    , obj_mat(eigen_utils::Mat4<float>::Identity())
{}

SceneModel::SceneModel(const std::string& model_path)
    : Model(loadOBJ(model_path))
    , obj_mat(eigen_utils::Mat4<float>::Identity())
{}

void
SceneModel::update()
{
    for (auto& tm : meshes) {
        for (auto& v : tm.vertex) {
            v = eigen_utils::from_eigen_v4_v3(
                obj_mat * eigen_utils::to_eigen_homogenus(v, true));
        }
        for (auto& n : tm.normal) {
            n = eigen_utils::from_eigen_v4_v3(
                obj_mat * eigen_utils::to_eigen_homogenus(n, false));
        }
    }
    obj_mat = eigen_utils::Mat4<float>::Identity();
}

void
SceneModel::transform(const eigen_utils::Mat4<float>& transform_mat)
{
    obj_mat = transform_mat * obj_mat;
}

void
OptixStaticScene::MeshBuffers::resize(int numMeshes)
{
    vertex.resize(numMeshes);
    normal.resize(numMeshes);
    texcoord.resize(numMeshes);
    index.resize(numMeshes);
}

void
OptixStaticScene::MeshBuffers::upload(int meshID, TriangleMesh& mesh)
{
    vertex[meshID].alloc_and_upload(mesh.vertex);
    index[meshID].alloc_and_upload(mesh.index);
    if (!mesh.normal.empty())
        normal[meshID].alloc_and_upload(mesh.normal);
    if (!mesh.texcoord.empty())
        texcoord[meshID].alloc_and_upload(mesh.texcoord);
}
size_t
OptixStaticScene::MeshBuffers::size() const
{
    return index.size();
}

void
OptixStaticScene::MeshBuffers::clear()
{
    vertex.clear();
    normal.clear();
    texcoord.clear();
    index.clear();
}

OptixStaticScene::OptixStaticScene()
{
    accelOptions.buildFlags =
        OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
}

void
OptixStaticScene::setupAccel()
{
    num_meshes = 0;
    for (auto& model : _models) {
        num_meshes += model.meshes.size();
    }
    _mesh_buffers.resize(num_meshes);
    // ==================================================================
    // triangle inputs
    // ==================================================================
    std::vector<OptixBuildInput> triangleInput(num_meshes);
    std::vector<CUdeviceptr> d_vertices(num_meshes);
    std::vector<CUdeviceptr> d_indices(num_meshes);
    std::vector<uint32_t> triangleInputFlags(num_meshes);

    int mesh_offset = 0;
    for (auto& model : _models) {
        model.update();
        // model.update();
        for (auto meshID = 0u; meshID < model.meshes.size(); meshID++) {
            // upload the model to the device: the builder
            TriangleMesh& mesh = model.meshes[meshID];
            _meshes.push_back(mesh);
            int full_id = mesh_offset + meshID;
            _mesh_buffers.upload(full_id, mesh);

            triangleInput[full_id] = {};
            triangleInput[full_id].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            triangleInputFlags[full_id] = 0;

            // create local variables, because we need a *pointer* to the
            // device pointers
            d_vertices[full_id] = _mesh_buffers.vertex[full_id].d_pointer();
            d_indices[full_id] = _mesh_buffers.index[full_id].d_pointer();

            auto& t_array = triangleInput[full_id].triangleArray;
            t_array.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            t_array.vertexStrideInBytes = sizeof(glm::vec3);
            t_array.numVertices = (int)mesh.vertex.size();
            t_array.vertexBuffers = &d_vertices[full_id];

            t_array.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            t_array.indexStrideInBytes = sizeof(vec3i);
            t_array.numIndexTriplets = (int)mesh.index.size();
            t_array.indexBuffer = d_indices[full_id];

            // in this case we have one SBT entry, and no per-primitive
            // materials:
            t_array.flags = &triangleInputFlags[full_id];
            t_array.numSbtRecords = 1;
            t_array.sbtIndexOffsetBuffer = 0;
            t_array.sbtIndexOffsetSizeInBytes = 0;
            t_array.sbtIndexOffsetStrideInBytes = 0;
        }
        mesh_offset += model.meshes.size();
    }
    // ==================================================================
    // BLAS setup
    // ==================================================================

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(OptixContext::get()->optixContext,
                                             &accelOptions,
                                             triangleInput.data(),
                                             num_meshes, // num_build_inputs
                                             &blasBufferSizes));

    // ==================================================================
    // prepare compaction
    // ==================================================================

    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    // ==================================================================
    // execute build (main stage)
    // ==================================================================

    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(OptixContext::get()->optixContext,
                                /* stream */ 0,
                                &accelOptions,
                                triangleInput.data(),
                                num_meshes,
                                tempBuffer.d_pointer(),
                                tempBuffer.sizeInBytes,

                                outputBuffer.d_pointer(),
                                outputBuffer.sizeInBytes,

                                &traversable,

                                &emitDesc,
                                1););
    CUDA_SYNC_CHECK();

    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(OptixContext::get()->optixContext,
                                  /*stream:*/ 0,
                                  traversable,
                                  asBuffer.d_pointer(),
                                  asBuffer.sizeInBytes,
                                  &traversable));
    CUDA_SYNC_CHECK();

    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();
    createTextures();
}

void
OptixStaticScene::createTextures()
{
    int numTextures = 0;
    for (auto& model : _models) {
        numTextures += model.textures.size();
    }
    textureArrays.resize(numTextures);
    textureObjects.resize(numTextures);
    for (auto& model : _models) {
        for (int textureID = 0; textureID < numTextures; textureID++) {
            auto texture = model.textures[textureID];

            cudaResourceDesc res_desc = {};

            cudaChannelFormatDesc channel_desc;
            int32_t width = texture.resolution.x;
            int32_t height = texture.resolution.y;
            int32_t numComponents = 4;
            int32_t pitch = width * numComponents * sizeof(uint8_t);
            channel_desc = cudaCreateChannelDesc<uchar4>();

            cudaArray_t& pixelArray = textureArrays[textureID];
            CUDA_CHECK(
                cudaMallocArray(&pixelArray, &channel_desc, width, height));

            CUDA_CHECK(cudaMemcpy2DToArray(pixelArray,
                                           /* offset */ 0,
                                           0,
                                           texture.pixel(),
                                           pitch,
                                           pitch,
                                           height,
                                           cudaMemcpyHostToDevice));

            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = pixelArray;

            cudaTextureDesc tex_desc = {};
            tex_desc.addressMode[0] = cudaAddressModeWrap;
            tex_desc.addressMode[1] = cudaAddressModeWrap;
            tex_desc.filterMode = cudaFilterModeLinear;
            tex_desc.readMode = cudaReadModeNormalizedFloat;
            tex_desc.normalizedCoords = 1;
            tex_desc.maxAnisotropy = 1;
            tex_desc.maxMipmapLevelClamp = 99;
            tex_desc.minMipmapLevelClamp = 0;
            tex_desc.mipmapFilterMode = cudaFilterModePoint;
            tex_desc.borderColor[0] = 1.0f;
            tex_desc.sRGB = 0;

            // Create texture object
            cudaTextureObject_t cuda_tex = 0;
            CUDA_CHECK(cudaCreateTextureObject(
                &cuda_tex, &res_desc, &tex_desc, nullptr));
            textureObjects[textureID] = cuda_tex;
        }
    }
}

void
OptixStaticScene::add_model(SceneModel sm)
{
    _models.push_back(std::move(sm));
}
} // namespace chameleon