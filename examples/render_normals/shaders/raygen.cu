#include <chameleon_renderer/cuda/RayType.h>
#include <chameleon_renderer/shader_utils/common.cuh>

namespace chameleon {
//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame() {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const auto& camera = optixLaunchParams.camera;

    // generate ray direction
    vec3f rayDir = normalize(
        transform(camera.camera_mat_inverse, vec3f(ix + .5f, iy + .5f, 1)));

    rayDir = normalize(transform_vec(camera.obj_mat, rayDir));
    // pixel index
    const uint32_t fbIndex = (ix + iy * camera.res.x);

    vec3f nml_pix;
    vec3f uv_pix;
    vec3f view_pix;
    uint8_t mask_pix;
    photometry_render::RadiationRayData prd = {&nml_pix, &uv_pix, &view_pix,
                                               &mask_pix, rayDir};

    uint32_t u0, u1;
    packPointer(&prd, u0, u1);
    optixTrace(
        optixLaunchParams.traversable, camera.pos, rayDir,
        0.f,    // tmin
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,                   // OPTIX_RAY_FLAG_NONE,
        unsigned(photometry_renderer::ray_t::RADIANCE),  // SBT offset
        unsigned(photometry_renderer::ray_t::RAY_TYPE_COUNT),  // SBT stride
        unsigned(photometry_renderer::ray_t::RADIANCE),        // missSBTIndex
        u0, u1);

    optixLaunchParams.layers.normal_map[fbIndex] = nml_pix;
    optixLaunchParams.layers.mask[fbIndex] = mask_pix;
    optixLaunchParams.layers.uv_map[fbIndex] = uv_pix;
    optixLaunchParams.layers.view[fbIndex] = view_pix;
}

}  // namespace chameleon