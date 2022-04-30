#include <chameleon_renderer/cuda/RayType.h>
#include <chameleon_renderer/renderer/barytex/BarytexLaunchParamProvider.cuh>
#include <chameleon_renderer/shader_utils/common.cuh>

namespace chameleon {
using namespace barytex_learn_render;

inline __device__ RadiationRayData
cast_ray(glm::vec3 ray_base, const chameleon::CUDACamera& camera,
         photometry_renderer::ray_t ray_type) {
    // generate ray direction
    ray_base.z = 1;
    glm::vec3 rayDir = camera.camera_mat_inverse * ray_base;

    rayDir = glm::normalize(transform_vec(camera.obj_mat, rayDir));
    rayDir *= -1.f;
    // printf("%f %f %f \n",rayDir.x,rayDir.y,rayDir.z);

    RadiationRayData prd;
    

    uint32_t u0, u1;
    packPointer(&prd, u0, u1);
    optixTrace(
        optixLaunchParams.traversable,
        {camera.pos.x, camera.pos.y, camera.pos.z},
        {rayDir.x, rayDir.y, rayDir.z},
        0.f,    // tmin
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,  // OPTIX_RAY_FLAG_NONE,
        unsigned(ray_type),             // SBT offset
        unsigned(photometry_renderer::ray_t::RAY_TYPE_COUNT),  // SBT stride
        unsigned(ray_type),                                    // missSBTIndex
        u0, u1);
    // rdb.view = rayDir;
    return prd;
}

inline __device__ CUDAArray<RadiationRayData> multisample_square(
    int ix, int iy, const CUDACamera& camera) {
    RadiationRayData rdb[4];

    rdb[0] =
        cast_ray(glm::vec3(ix + .25f, iy + .25f, 1), camera,
                 photometry_renderer::ray_t::RADIANCE);

    rdb[1] =
        cast_ray(glm::vec3(ix + .25f, iy + .75f, 1), camera,
                 photometry_renderer::ray_t::RADIANCE);

    rdb[2] =
        cast_ray(glm::vec3(ix + .75f, iy + .25f, 1), camera,
                 photometry_renderer::ray_t::RADIANCE);

    rdb[3] =
        cast_ray(glm::vec3(ix + .75f, iy + .75f, 1), camera,
                 photometry_renderer::ray_t::RADIANCE);

    return {rdb, 4};
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame() {
    // compute a test pattern based on pixel ID
    
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    
    const auto& camera = optixLaunchParams.camera;
    const uint32_t fbIndex = (ix + iy * camera.res.x);

    // if (ix == camera.res.x / 2 && iy == camera.res.y / 2) {
    //     glm::vec3 ray_base(ix + .5f, iy + .5f, 1);
    //     glm::vec3 rayDir = camera.camera_mat_inverse * ray_base;

    //     rayDir = glm::normalize(transform_vec(camera.obj_mat, rayDir));
    //     rayDir *= -1.f;
    //     // printf("[%f, %f, %f ],\n", rayDir.x, rayDir.y, rayDir.z);
    // }
    
    RadiationRayData rdb =
        cast_ray(glm::vec3(ix + .5f, iy + .5f, 1), camera,
                 photometry_renderer::ray_t::RADIANCE);
    if(rdb.hit.is_valid){
        if(fbIndex > (optixLaunchParams.observation.image.rows * optixLaunchParams.observation.image.cols)){
            printf("%d, %d : %d, %d\n",ix,iy,optixLaunchParams.observation.image.cols, optixLaunchParams.observation.image.rows);
        }else{
            rdb.hit.value = optixLaunchParams.observation.image.data[fbIndex]/255.0f;
        }
    }
    optixLaunchParams.render_data
        .data[fbIndex * optixLaunchParams.sample_multiplier] = rdb.hit;
    // if (rdb.hit.is_valid) {
    if (false) {
        CUDAArray<RadiationRayData> rdb_array =
            multisample_square(ix, iy, camera);

        for (int i = 0; i < rdb_array.size; ++i) {
            optixLaunchParams.render_data
                .data[fbIndex * optixLaunchParams.sample_multiplier + i + 1] =
                rdb_array.data[i].hit;
        }
    }
}

}  // namespace chameleon