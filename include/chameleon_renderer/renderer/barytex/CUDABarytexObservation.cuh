#pragma once
#include <chameleon_renderer/cuda/CUDAImage.h>
#include <chameleon_renderer/cuda/CUDALight.h>

namespace chameleon {
    struct CUDABarytexObservation{
        CUDAImage<glm::vec3> image;
        CUDALight light;
    };
} // chameleon