#include <chameleon_renderer/optix/OptixContext.hpp>

#include <chameleon_renderer/utils/optix_device_utils.hpp>
#include <chameleon_renderer/utils/terminal_utils.hpp>

namespace chameleon {

std::shared_ptr<OptixContext>
OptixContext::get(int dev)
{
    if (instances.count(dev) == 0) {
        instances[dev] = std::shared_ptr<OptixContext>(new OptixContext(dev));
    }
    return instances[dev];
}

void
OptixContext::initialize()
{
    if (!initialized) {
        std::cout << "#osc: initializing optix..." << std::endl;
        // -------------------------------------------------------
        // check for available optix7 capable devices
        // -------------------------------------------------------
        cudaFree(0);
        int numDevices;
        cudaGetDeviceCount(&numDevices);
        if (numDevices == 0)
            throw std::runtime_error("#osc: no CUDA capable devices found!");
        std::cout << "#osc: found " << numDevices << " CUDA devices"
                  << std::endl;

        // -------------------------------------------------------
        // initialize optix
        // -------------------------------------------------------
        OPTIX_CHECK(optixInit());
        std::cout << "#osc: successfully initialized optix" << std::endl;
        initialized = true;
    }
}

OptixContext::OptixContext(int deviceID)
{
    initialize();
    // for this sample, do everything on one device
    CUDA_CHECK(cudaSetDevice(deviceID));
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaGetDeviceProperties(&deviceProps, deviceID);
    std::cout << "#osc: running on device: " << deviceProps.name << std::endl;
    CUresult cuRes = cuCtxGetCurrent(&cudaContext);
    if (cuRes != CUDA_SUCCESS)
        fprintf(
            stderr, "Error querying current context: error code %d\n", cuRes);
    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(
        optixContext, context_log_cb, nullptr, 4));
    PING
}

bool OptixContext::initialized = false;
std::map<int, std::shared_ptr<OptixContext>> OptixContext::instances = {};
} // namespace chameleon
