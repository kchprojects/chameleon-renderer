#include <chameleon_renderer/shader_utils/common.cuh>
namespace chameleon {
    extern "C" __global__ void __miss__shadow() {
        bool& prd = *(bool*)getPRD<bool>();
        prd = true;
        // printf("__miss__shadow");
    }

    extern "C" __global__ void __miss__radiance() {
        photometry_render::RadiationRayData* prd = getPRD<photometry_render::RadiationRayData>();
        // printf("__miss__radiance");
        *(prd->normal) = {0,0,0};
        *(prd->uv) = {0,0,-1};
        *(prd->view) = {0,0,0};
        *(prd->mask) = 0;
    }
}  // namespace chameleon