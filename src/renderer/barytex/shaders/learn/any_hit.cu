#include <chameleon_renderer/shader_utils/common.cuh>

namespace chameleon {
extern "C" __global__ void __anyhit__shadow() {}
extern "C" __global__ void __anyhit__radiance() {}
}  // namespace chameleon