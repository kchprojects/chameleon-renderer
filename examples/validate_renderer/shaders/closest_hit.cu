#include <chameleon_renderer/shader_utils/common.cuh>
#include <chameleon_renderer/cuda/RayType.h>
#include <chameleon_renderer/renderer/PhotometryLaunchParamProvider.cuh>
namespace chameleon {
extern "C" __global__ void __closesthit__shadow() {

}

// inline __device__ void compute_byte_shadows(const SurfaceInfo& si,
//                                             RadiationRayData* prd) {
//     uint8_t* pixel_stack = (uint8_t*)prd->shadow_data;
//     for (int i = 0; i < optixLaunchParams.light_count; ++i) {
//         uint32_t u0, u1;
//         bool curr_light = false;
//         packPointer(&curr_light, u0, u1);
//         const vec3f lightDir = optixLaunchParams.light_data[i] - si.surfPos;
//         if (dot(lightDir, si.Ns) > 0) {
//             optixTrace(optixLaunchParams.traversable,
//                        si.surfPos + 1e-3f * si.Ng, lightDir,
//                        1e-3f,        // tmin
//                        1.f - 1e-3f,  // tmax
//                        0.0f,         // rayTime
//                        OptixVisibilityMask(255),
//                        // For shadow rays: skip any/closest hit shaders and
//                        // terminate on first intersection with anything. The
//                        // miss shader is used to mark if the light was
//                        visible OPTIX_RAY_FLAG_DISABLE_ANYHIT |
//                            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
//                            OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
//                        SHADOW_RAY_TYPE,  // SBT offset
//                        RAY_TYPE_COUNT,   // SBT stride
//                        SHADOW_RAY_TYPE,  // missSBTIndex
//                        u0, u1);
//         }
//         if (curr_light) {
//             pixel_stack[i] = 255;
//         }
//     }
// }

// inline __device__ void compute_bit_shadows(const SurfaceInfo& si,
//                                            RadiationRayData* prd) {
//     uint8_t* pixel_stack = (uint8_t*)prd->shadow_data;

//     uint8_t curr_byte;
//     bool curr_light;
//     for (uint8_t i = 0; i < prd->shadow_byte_count; ++i) {
//         curr_byte = 0;
//         for (uint8_t j = 0; j < 8; ++j) {
//             curr_light = false;
//             uint32_t u0, u1;
//             packPointer(&curr_light, u0, u1);
//             const vec3f lightDir =
//                 optixLaunchParams.light_data[i * 8 + j] - si.surfPos;
//             if (dot(lightDir, si.Ns) > 0) {
//                 optixTrace(
//                     optixLaunchParams.traversable, si.surfPos + 1e-3f *
//                     si.Ng, lightDir, 1e-3f,        // tmin 1.f - 1e-3f,  //
//                     tmax 0.0f,         // rayTime OptixVisibilityMask(255),
//                     // For shadow rays: skip any/closest hit shaders and
//                     // terminate on first intersection with anything. The
//                     // miss shader is used to mark if the light was visible
//                     OPTIX_RAY_FLAG_DISABLE_ANYHIT |
//                         OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
//                         OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
//                     SHADOW_RAY_TYPE,  // SBT offset
//                     RAY_TYPE_COUNT,   // SBT stride
//                     SHADOW_RAY_TYPE,  // missSBTIndex
//                     u0, u1);
//                 if (curr_light) {
//                     curr_byte |= 1 << j;
//                 }
//             }
//         }
//         pixel_stack[i] = curr_byte;
//     }
// }

inline __device__ glm::vec3 compute_bounce_dir(const SurfaceInfo& si, int light_id){
    if(light_id >= optixLaunchParams.light_data.count){
        return {0.f,0.f,0.f};
    }
    return glm::normalize(optixLaunchParams.light_data.data[light_id].position - si.surfPos);
}

inline __device__ float lambert(const glm::vec3& N, const glm::vec3& L){
    return max(0.f,dot(N, L));
}

inline __device__ float phong(const glm::vec3& E, const glm::vec3& N, const glm::vec3& L, float Ks,float shininess){
    float dot_size = dot(L, N);
    
    glm::vec3 R = 2*dot_size*N - L; 
    float out = Ks*pow(max(dot(E,R),0.f),shininess);
    return out;
}



inline __device__ bool cast_shadow_ray(const SurfaceInfo& si,const  glm::vec3& ray_dir){
    bool is_visible = false;
    uint32_t u0, u1;
    packPointer(&is_visible, u0, u1);
    glm::vec3 position = si.surfPos + ray_dir*0.001f;
    optixTrace(
        optixLaunchParams.traversable, {position.x,position.y,position.z}, {ray_dir.x,ray_dir.y,ray_dir.z},
        0.f,    // tmin
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,  // OPTIX_RAY_FLAG_NONE,
        unsigned(photometry_renderer::ray_t::SHADOW),             // SBT offset
        unsigned(photometry_renderer::ray_t::RAY_TYPE_COUNT),  // SBT stride
        unsigned(photometry_renderer::ray_t::SHADOW),                                    // missSBTIndex
        u0, u1);
    return is_visible;
}

extern "C" __global__ void __closesthit__radiance() {
    const SurfaceInfo si = get_surface_info();
    const glm::vec3 rayDir = {optixGetWorldRayDirection().x,
                          optixGetWorldRayDirection().y,
                          optixGetWorldRayDirection().z};
    // printf(__FUNCTION__);
    photometry_render::RadiationRayData* prd =
        getPRD<photometry_render::RadiationRayData>();
    glm::vec3 L = compute_bounce_dir(si,0);
    bool is_visible = cast_shadow_ray(si,L);
    glm::vec3& norm = *(prd->normal);
    if (dot(rayDir, si.Ns) > 0) {
        norm = {0, 0, 0};
        *(prd->uv) = {0,0,-1};
        *(prd->view) = {0,0,0};
        *(prd->mask) = 0;
    } else {
        norm = si.Ns;
        norm = normalize(
            transform_vec(optixLaunchParams.camera.obj_mat_inverse, norm));
        // correct coord_system
        norm.z *= -1;
        *(prd->uv) = si.uv;
        if(is_visible){
            glm::vec3 light_color = {1,1,1};
            // if(optixLaunchParams.light_data.count > 0){
            //     light_color = optixLaunchParams.light_data.data[0].color;
            // }
            *(prd->view) = si.diffuseColor*lambert(si.Ns,L) + phong(-rayDir,si.Ns,L,0.5f,1.5f)*light_color;
            prd->view->x = min(1.f, (prd->view)->x);
            prd->view->y = min(1.f, (prd->view)->y);
            prd->view->z = min(1.f, (prd->view)->z);
        }else{
            *(prd->view) = {0.f,0.f,0.f};
        }
        *(prd->mask) = 255;
    }
}

}  // namespace chameleon