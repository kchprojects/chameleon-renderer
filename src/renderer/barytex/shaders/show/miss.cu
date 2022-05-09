#include <chameleon_renderer/shader_utils/common.cuh>
namespace chameleon {
    extern "C" __global__ void __miss__shadow() {
        bool& prd = *(bool*)getPRD<bool>();
        prd = true;
    }

    extern "C" __global__ void __miss__radiance() {
    }
}  // namespace chameleon