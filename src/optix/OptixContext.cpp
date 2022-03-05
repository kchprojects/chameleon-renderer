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
        spdlog::info("initializing optix...");
        // -------------------------------------------------------
        // check for available optix7 capable devices
        // -------------------------------------------------------
        cudaFree(0);
        int numDevices;
        cudaGetDeviceCount(&numDevices);
        if (numDevices == 0){
            spdlog::error("no CUDA capable devices found!");
            throw std::runtime_error("no CUDA capable devices found!");
        }
        spdlog::info("found {} CUDA devices",numDevices);

        // -------------------------------------------------------
        // initialize optix
        // -------------------------------------------------------
        OPTIX_CHECK(optixInit());
        spdlog::info("successfully initialized optix");
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
    spdlog::info("running on device: {}", deviceProps.name);
    CUresult cuRes = cuCtxGetCurrent(&cudaContext);
    if (cuRes != CUDA_SUCCESS){
        spdlog::error("Error querying current context: error code {}n", cuRes);
    }
    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(
        optixContext, context_log_cb, nullptr, 4));
}

bool OptixContext::initialized = false;
std::map<int, std::shared_ptr<OptixContext>> OptixContext::instances = {};
} // namespace chameleon
