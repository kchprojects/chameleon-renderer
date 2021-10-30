#pragma once
#include <chameleon_renderer/cuda/CUDABuffer.h>
#include <chameleon_renderer/utils/optix7.h>

#include <iostream>
#include <map>
#include <memory>
#include <chameleon_renderer/utils/file_utils.hpp>
#include <chameleon_renderer/utils/optix_device_utils.hpp>

namespace chameleon {

OptixModule base_module_create(
    const std::string& ptx_filename,
    OptixModuleCompileOptions& moduleCompileOptions,
    OptixPipelineCompileOptions& pipelineCompileOptions);

}  // namespace chameleon
