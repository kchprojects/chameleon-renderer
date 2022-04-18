#include <chameleon_renderer/cuda/RayType.h>
#include <chameleon_renderer/renderer/barytex/BarytexLaunchParamProvider.cuh>
#include <chameleon_renderer/shader_utils/common.cuh>
namespace chameleon {
using namespace barytex_learn_render;
extern "C" __global__ void __closesthit__shadow() {}

inline __device__ bool cast_shadow_ray(const SurfaceInfo& si,
                                       const glm::vec3& ray_dir) {
    bool is_visible = false;
    uint32_t u0, u1;
    packPointer(&is_visible, u0, u1);
    glm::vec3 position = si.surfPos + ray_dir * 0.001f;
    optixTrace(
        optixLaunchParams.traversable, {position.x, position.y, position.z},
        {ray_dir.x, ray_dir.y, ray_dir.z},
        0.f,    // tmin
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,                 // OPTIX_RAY_FLAG_NONE,
        unsigned(photometry_renderer::ray_t::SHADOW),  // SBT offset
        unsigned(photometry_renderer::ray_t::RAY_TYPE_COUNT),  // SBT stride
        unsigned(photometry_renderer::ray_t::SHADOW),          // missSBTIndex
        u0, u1);
    return is_visible;
}

inline __device__ glm::vec3 compute_bounce_dir(const SurfaceInfo& si){
    return glm::normalize(optixLaunchParams.observation.light.position - si.surfPos);
}

extern "C" __global__ void __closesthit__radiance() {
    SurfaceInfo si = get_surface_info();
    const glm::vec3 ray_dir = {optixGetWorldRayDirection().x,
                              optixGetWorldRayDirection().y,
                              optixGetWorldRayDirection().z};
    // printf(__FUNCTION__);
    RadiationRayData* prd =
        getPRD<RadiationRayData>();

    glm::vec3 to_light = optixLaunchParams.observation.light.position - si.surfPos;
    glm::vec3 L = compute_bounce_dir(si);
    bool is_visible = cast_shadow_ray(si, L);
    if (!is_visible) {
        prd->hit.is_valid = false;
        return;
    }  

    if (dot(ray_dir, si.Ns) > 0) {
        // TODO:Decide if remove
        si.Ns = si.Ns * -1.f;
    }  // else
    {
        prd->hit.triangle_id = si.primID;
        prd->hit.coordinates = si.bary_coords;
        prd->hit.world_coordinates = si.surfPos;
    
        prd->hit.eye = -ray_dir * distance(si.surfPos, optixLaunchParams.camera.pos);
        prd->hit.light = to_light;

        prd->hit.mesh_normal = si.Ns;
        prd->hit.is_valid = true;
    }
}

}  // namespace chameleon