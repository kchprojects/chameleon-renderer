#pragma once
#include <chameleon_renderer/cuda/CUDABuffer.h>
#include <chameleon_renderer/utils/optix7.h>

#include <iostream>
#include <map>
#include <memory>

#include "OptixContext.hpp"
#include "OptixModules.hpp"
#include "OptixScene.hpp"

namespace chameleon {
enum class ProgramType { RAYGEN, MISS, CLOSEST_HIT, ANY_HIT, INTERSECT };
void base_program_create(OptixProgramGroupKind group_kind,
                         const std::string& entryFunctionName,
                         OptixModule module,
                         OptixProgramGroup* PG,
                         OptixProgramGroupOptions pgOptions = {});

struct OptixPrograms {
    std::vector<OptixProgramGroup> raygenPGs;
    CUDABuffer raygenRecordsBuffer;

    std::vector<OptixProgramGroup> missPGs;
    CUDABuffer missRecordsBuffer;

    std::vector<OptixProgramGroup> hitgroupPGs;
    CUDABuffer hitgroupRecordsBuffer;

    OptixShaderBindingTable sbt = {};

    std::map<std::string, int> ray2index;

    OptixPrograms();

    OptixPrograms& add_raygen_program(const std::string& label,
                                      OptixModule raygen_module);

    OptixPrograms& add_ray(const std::string& label,
                           std::optional<OptixModule> miss_module = {},
                           std::optional<OptixModule> any_hit_module = {},
                           std::optional<OptixModule> closest_hit_module = {},
                           std::optional<OptixModule> intersect_module = {});

    OptixPipeline get_pipeline(
        OptixPipelineCompileOptions pipelineCompileOptions = {},
        OptixPipelineLinkOptions pipelineLinkOptions = {});

    void buildSBT(const OptixStaticScene& scene);
};
}  // namespace chameleon
