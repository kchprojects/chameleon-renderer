#include <chameleon_renderer/optix/OptixModules.hpp>
#include <chameleon_renderer/optix/OptixContext.hpp>
#include <chameleon_renderer/utils/terminal_utils.hpp>

namespace chameleon {

OptixModule base_module_create(
    const std::string& ptx_filename,
    OptixModuleCompileOptions& moduleCompileOptions,
    OptixPipelineCompileOptions& pipelineCompileOptions) {
    OptixModule out;
    const std::string ptxCode = readPTX(ptx_filename);
    char log[2048];
    size_t sizeof_log = sizeof(log);
    PING
    OPTIX_CHECK(optixModuleCreateFromPTX(
        OptixContext::get()->optixContext, &moduleCompileOptions,
        &pipelineCompileOptions, ptxCode.c_str(), ptxCode.size(), log,
        &sizeof_log, &out));

    PING
    if (sizeof_log > 1)
        spdlog::info(log);
    return out;
}

}  // namespace chameleon
