#pragma once
#include <chameleon_renderer/cuda/TriangleMeshSBTData.h>
#include <cuda_runtime.h>
#include <optix_device.h>
#include <chameleon_renderer/utils/math_utils.hpp>

#include <chameleon_renderer/shader_utils/PerRayData.cuh>

namespace chameleon {

static __forceinline__ __device__ void* unpackPointer(uint32_t i0,
                                                      uint32_t i1) {
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__ void packPointer(void* ptr, uint32_t& i0,
                                                   uint32_t& i1) {
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template <typename T>
static __forceinline__ __device__ T* getPRD() {
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

inline __device__ glm::vec3 transform(const glm::mat3& mat,
                                      const glm::vec3& vec) {
    return mat * vec;
}
inline __device__ float dot(const glm::vec4& a, const glm::vec4& b) {
    return glm::dot(a, b);
}

inline __device__ float distance(const glm::vec3& a, const glm::vec3& b) {
    return sqrt(dot(a-b,a-b));
}

inline __device__ glm::vec4 transform(const glm::mat4& mat,
                                      const glm::vec4& vec) {
    return mat * vec;
}

inline __device__ glm::vec3 transform_vec(const glm::mat4& mat,
                                          const glm::vec3& vec) {
    glm::vec4 vec_4d = {vec.x, vec.y, vec.z, 0.f};
    vec_4d = transform(mat, vec_4d);
    return {vec_4d.x, vec_4d.y, vec_4d.z};
}
inline __device__ glm::vec3 transform_point(const glm::mat4& mat,
                                            const glm::vec3& vec) {
    glm::vec4 vec_4d = {vec.x, vec.y, vec.z, 1.f};
    vec_4d = transform(mat, vec_4d);
    return {vec_4d.x, vec_4d.y, vec_4d.z};
}

inline __device__ void print_mat(const glm::mat4& mat) {
    std::printf("%f\t%f\t%f\t%f\n"
    "%f\t%f\t%f\t%f\n"
    "%f\t%f\t%f\t%f\n"
    "%f\t%f\t%f\t%f\n\n",
    mat[0][0],mat[1][0],mat[2][0],mat[3][0],
    mat[0][1],mat[1][1],mat[2][1],mat[3][1],
    mat[0][2],mat[1][2],mat[2][2],mat[3][2],
    mat[0][3],mat[1][3],mat[2][3],mat[3][3]
    );
}

struct SurfaceInfo {
    glm::vec3 Ns;
    glm::vec3 Ng;
    glm::vec3 surfPos;
    glm::vec3 uv;
    glm::vec3 diffuseColor;
    glm::vec3 bary_coords;
    int primID;
};

inline __device__ SurfaceInfo get_surface_info() {
    const TriangleMeshSBTData& sbtData =
        *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    // ------------------------------------------------------------------
    // gather some basic hit information
    // ------------------------------------------------------------------
    const int primID = optixGetPrimitiveIndex();
    // printf("%d\n",primID);
    const vec3i index = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // compute normal, using either shading normal (if avail), or
    // geometry normal (fallback)
    // ------------------------------------------------------------------
    const glm::vec3& A = sbtData.vertex[index.x];
    const glm::vec3& B = sbtData.vertex[index.y];
    const glm::vec3& C = sbtData.vertex[index.z];
    glm::vec3 Ng = cross(B - A, C - A);

    glm::vec3 Ns;
    Ns = (sbtData.normal)
             ? ((1.f - u - v) * sbtData.normal[index.x] +
                u * sbtData.normal[index.y] + v * sbtData.normal[index.z])
             : Ng;

    // ------------------------------------------------------------------
    // face-forward and normalize normals
    // ------------------------------------------------------------------
    const glm::vec3 rayDir = {optixGetWorldRayDirection().x,
                              optixGetWorldRayDirection().y,
                              optixGetWorldRayDirection().z};

    // if (dot(rayDir, Ng) > 0.f)
    //     Ng = -Ng;
    Ng = normalize(Ng);

    if (dot(Ng, Ns) < 0.f) Ns -= 2.f * dot(Ng, Ns) * Ng;
    Ns = normalize(Ns);

    // ------------------------------------------------------------------
    // compute shadow
    // ------------------------------------------------------------------
    const glm::vec3 bary_coords = {(1.f - u - v),u,v};

    const glm::vec3 surfPos = (1.f - u - v) * sbtData.vertex[index.x] +
                              u * sbtData.vertex[index.y] +
                              v * sbtData.vertex[index.z];

    glm::vec<2, float> tc;
    if (sbtData.texcoord) {
        tc = (1.f - u - v) * sbtData.texcoord[index.x] +
             u * sbtData.texcoord[index.y] + v * sbtData.texcoord[index.z];
    } else {
        tc = {0, 0};
    }

    // ------------------------------------------------------------------
    // diffuse texture
    // ------------------------------------------------------------------

    glm::vec3 diffuseColor = sbtData.color;
    if (sbtData.hasTexture && sbtData.texcoord) {
        float4 fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
        diffuseColor *= glm::vec3(fromTexture.x, fromTexture.y, fromTexture.z);
    }
    // TODO::differentiate meshes
    return {Ns, Ng, surfPos, {tc.x, tc.y, 0}, diffuseColor,bary_coords,primID};
}

}  // namespace chameleon