#pragma once
#include <iostream>
#include <map>
#include <memory>

#include <chameleon_renderer/cuda/CUDABuffer.hpp>
#include <chameleon_renderer/utils/optix7.h>

namespace chameleon {
struct OptixContext
{
    CUcontext cudaContext;
    CUstream stream;
    cudaDeviceProp deviceProps;
    OptixDeviceContext optixContext;

    static std::shared_ptr<OptixContext> get(int dev = 0);

    static void initialize();

private:
    OptixContext(int deviceID = 0);

    static bool initialized;
    static std::map<int, std::shared_ptr<OptixContext>> instances;
};
} // namespace chameleon
