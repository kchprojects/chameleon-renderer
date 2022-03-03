#pragma once
#include <cuda_runtime.h>
#include <optix_device.h>
#include <chameleon_renderer/cuda/LaunchParams.h>
#include <chameleon_renderer/cuda/TriangleMeshSBTData.h>

#include <chameleon_renderer/utils/math_utils.hpp>

#include "PerRayData.cuh"

namespace chameleon {
/*! launch parameters in constant memory, filled in by optix upon
    optixLaunch (this gets filled in from the buffer we pass to
    optixLaunch) */
extern "C" __constant__ photometry_render::LaunchParams optixLaunchParams;

static __forceinline__ __device__ void* unpackPointer(uint32_t i0,
                                                      uint32_t i1) {
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__ void packPointer(void* ptr,
                                                   uint32_t& i0,
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

inline __device__ vec3f transform(const mat3f& mat, const vec3f& vec) {
    return {dot(mat.r1, vec), dot(mat.r2, vec), dot(mat.r3, vec)};
}
inline __device__ float dot(const vec4f& a, const vec4f& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
inline __device__ vec4f transform(const mat4f& mat, const vec4f& vec) {
    return {dot(mat.r1, vec), dot(mat.r2, vec), dot(mat.r3, vec),
            dot(mat.r4, vec)};
}

inline __device__ vec3f transform_vec(const mat4f& mat, const vec3f& vec) {
    vec4f vec_4d = {vec.x, vec.y, vec.z, 0.f};
    vec_4d = transform(mat, vec_4d);
    return {vec_4d.x, vec_4d.y, vec_4d.z};
}
inline __device__ vec3f transform_point(const mat4f& mat, const vec3f& vec) {
    vec4f vec_4d = {vec.x, vec.y, vec.z, 1.f};
    vec_4d = transform(mat, vec_4d);
    return {vec_4d.x, vec_4d.y, vec_4d.z};
}

inline __device__ vec3f transform_vec(const affine_mat4f& mat,
                                      const vec3f& vec) {
    return transform(mat.R, vec);
}
inline __device__ vec3f transform_point(const affine_mat4f& mat,
                                        const vec3f& vec) {
    return transform_vec(mat, vec) + mat.translation;
}

struct SurfaceInfo {
    vec3f Ns;
    vec3f Ng;
    vec3f surfPos;
    vec3f uv;
    vec3f diffuseColor;
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
    const vec3f& A = sbtData.vertex[index.x];
    const vec3f& B = sbtData.vertex[index.y];
    const vec3f& C = sbtData.vertex[index.z];
    vec3f Ng = cross(B - A, C - A);

    vec3f Ns;
    Ns = (sbtData.normal)
             ? ((1.f - u - v) * sbtData.normal[index.x] +
                u * sbtData.normal[index.y] + v * sbtData.normal[index.z])
             : Ng;

    // ------------------------------------------------------------------
    // face-forward and normalize normals
    // ------------------------------------------------------------------
    const vec3f rayDir = {optixGetWorldRayDirection().x,
                          optixGetWorldRayDirection().y,
                          optixGetWorldRayDirection().z};

    // if (dot(rayDir, Ng) > 0.f)
    //     Ng = -Ng;
    Ng = normalize(Ng);

    if (dot(Ng, Ns) < 0.f)
        Ns -= 2.f * dot(Ng, Ns) * Ng;
    Ns = normalize(Ns);

    // ------------------------------------------------------------------
    // compute shadow
    // ------------------------------------------------------------------
    const vec3f surfPos = (1.f - u - v) * sbtData.vertex[index.x] +
                          u * sbtData.vertex[index.y] +
                          v * sbtData.vertex[index.z];

    vec2f tc;
    if (sbtData.texcoord) {
        tc = (1.f - u - v) * sbtData.texcoord[index.x] +
             u * sbtData.texcoord[index.y] + v * sbtData.texcoord[index.z];
    } else {
        tc = {0, 0};
    }

    // ------------------------------------------------------------------
    // diffuse texture
    // ------------------------------------------------------------------

    vec3f diffuseColor = sbtData.color;
    if (sbtData.hasTexture && sbtData.texcoord) {
        vec4f fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
        diffuseColor *= (vec3f)fromTexture;
    }
    // TODO::differentiate meshes
    return {Ns, Ng, surfPos, {tc.x, tc.y, 0},diffuseColor};
}

}  // namespace chameleon