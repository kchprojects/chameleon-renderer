#include <chameleon_renderer/cuda/RayType.h>
#include <chameleon_renderer/renderer/PhotometryLaunchParamProvider.cuh>
#include <chameleon_renderer/shader_utils/common.cuh>

namespace chameleon {

inline __device__ photometry_render::RadiationRayDataBase cast_ray(
    glm::vec3 ray_base, const CUDACamera& camera,
    photometry_renderer::ray_t ray_type) {
    // generate ray direction
    ray_base.z = 1; 
    glm::vec3 rayDir = camera.camera_mat_inverse * ray_base;

    rayDir = glm::normalize(transform_vec(camera.obj_mat, rayDir));
    rayDir *= -1.f;
    // printf("%f %f %f \n",rayDir.x,rayDir.y,rayDir.z);

    photometry_render::RadiationRayDataBase rdb;
    photometry_render::RadiationRayData prd = {&rdb.normal, &rdb.uv, &rdb.view,
                                               &rdb.mask};

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
    return rdb;
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

    if (ix == camera.res.x / 2 && iy == camera.res.y / 2) {
        glm::vec3 ray_base(ix + .5f, iy + .5f, 1);
        glm::vec3 rayDir = camera.camera_mat_inverse * ray_base;

    rayDir = glm::normalize(transform_vec(camera.obj_mat, rayDir));
    rayDir *= -1.f;
        // print_mat(camera.obj_mat);
        // printf("pos: %f %f %f \n",camera.pos.x, camera.pos.y, camera.pos.z);
        // printf("dir: %f %f %f \n", rayDir.x, rayDir.y, rayDir.z);
        printf("[%f, %f, %f ],\n", rayDir.x, rayDir.y, rayDir.z);
    }

    photometry_render::RadiationRayDataBase rdb =
        cast_ray(glm::vec3(ix + .5f, iy + .5f, 1), camera,
                 photometry_renderer::ray_t::RADIANCE);

    // if (rdb.mask > 0) {
    if (false) {
        photometry_render::RadiationRayDataBase rdb1 =
            cast_ray(glm::vec3(ix + .25f, iy + .25f, 1), camera,
                     photometry_renderer::ray_t::RADIANCE);

        photometry_render::RadiationRayDataBase rdb2 =
            cast_ray(glm::vec3(ix + .25f, iy + .75f, 1), camera,
                     photometry_renderer::ray_t::RADIANCE);

        photometry_render::RadiationRayDataBase rdb3 =
            cast_ray(glm::vec3(ix + .75f, iy + .25f, 1), camera,
                     photometry_renderer::ray_t::RADIANCE);

        photometry_render::RadiationRayDataBase rdb4 =
            cast_ray(glm::vec3(ix + .75f, iy + .75f, 1), camera,
                     photometry_renderer::ray_t::RADIANCE);

        optixLaunchParams.layers.normal_map[fbIndex] =
            (rdb.normal + rdb1.normal + rdb2.normal + rdb3.normal +
             rdb4.normal) /
            5.f;
        optixLaunchParams.layers.mask[fbIndex] =
            (rdb.mask + rdb1.mask + rdb2.mask + rdb3.mask + rdb4.mask) / 5.f;
        optixLaunchParams.layers.uv_map[fbIndex] =
            (rdb.uv + rdb1.uv + rdb2.uv + rdb3.uv + rdb4.uv) / 5.f;
        optixLaunchParams.layers.view[fbIndex] =
            (rdb.view + rdb1.view + rdb2.view + rdb3.view + rdb4.view) / 5.f;
    } else {
        optixLaunchParams.layers.normal_map[fbIndex] = rdb.normal;
        optixLaunchParams.layers.mask[fbIndex] = rdb.mask;
        optixLaunchParams.layers.uv_map[fbIndex] = rdb.uv;
        optixLaunchParams.layers.view[fbIndex] = rdb.view;
    }
    // rdb.view.z = 0;
    // rdb.view.y = (rdb.view.y + 1.f)/2.f;
    // rdb.view.x = (rdb.view.x + 1.f)/2.f;
    // optixLaunchParams.layers.view[fbIndex] = rdb.view;
    // optixLaunchParams.layers.view[fbIndex] =
    // {ix/float(camera.res.x),iy/float(camera.res.y),0};
}

}  // namespace chameleon