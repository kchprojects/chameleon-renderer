#include <chameleon_renderer/shader_utils/common.cuh>

namespace chameleon {
extern "C" __global__ void __closesthit__shadow() {
    printf(__FUNCTION__);
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

extern "C" __global__ void __closesthit__radiance() {
    const SurfaceInfo si = get_surface_info();
    const glm::vec3 rayDir = {optixGetWorldRayDirection().x,
                          optixGetWorldRayDirection().y,
                          optixGetWorldRayDirection().z};
    // printf(__FUNCTION__);
    photometry_render::RadiationRayData* prd =
        getPRD<photometry_render::RadiationRayData>();

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
        *(prd->view) = si.diffuseColor*(-dot(rayDir, si.Ns));
        *(prd->mask) = 255;
    }
}

}  // namespace chameleon