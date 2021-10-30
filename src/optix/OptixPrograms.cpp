#include <chameleon_renderer/optix/OptixPrograms.hpp>

namespace chameleon {

void
base_program_create(OptixProgramGroupKind group_kind,
                    const std::string& entryFunctionName,
                    OptixModule module,
                    OptixProgramGroup* PG,
                    OptixProgramGroupOptions pgOptions)
{
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = group_kind;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = entryFunctionName.c_str();

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(OptixContext::get()->optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,
                                        &sizeof_log,
                                        PG));
    if (sizeof_log > 1)
        PRINT(log);
}

OptixPrograms::OptixPrograms() = default;

OptixPrograms&
OptixPrograms::add_raygen_program(const std::string& label,
                                  OptixModule raygen_module)
{
    raygenPGs.emplace_back();
    base_program_create(OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
                        "__raygen__" + label,
                        raygen_module,
                        &raygenPGs[0]);
    return *this;
}

OptixPrograms&
OptixPrograms::add_ray(const std::string& label,
                       std::optional<OptixModule> miss_module,
                       std::optional<OptixModule> any_hit_module,
                       std::optional<OptixModule> closest_hit_module,
                       std::optional<OptixModule> intersect_module)
{
    char log[2048];
    size_t sizeof_log = sizeof(log);
    ray2index[label] = missPGs.size();
    missPGs.emplace_back();
    hitgroupPGs.emplace_back();

    if (miss_module.has_value()) {
        base_program_create(OPTIX_PROGRAM_GROUP_KIND_MISS,
                            "__miss__" + label,
                            *miss_module,
                            &missPGs[ray2index[label]]);
    }

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    std::string NameAH = "__anyhit__" + label;
    if (any_hit_module.has_value()) {
        pgDesc.hitgroup.moduleAH = *any_hit_module;
        pgDesc.hitgroup.entryFunctionNameAH = NameAH.c_str();
    }
    std::string NameCH = "__closesthit__" + label;
    if (closest_hit_module.has_value()) {
        pgDesc.hitgroup.moduleCH = *closest_hit_module;

        pgDesc.hitgroup.entryFunctionNameCH = NameCH.c_str();
    }
    std::string NameIS = "__intersect__" + label;
    if (intersect_module.has_value()) {
        pgDesc.hitgroup.moduleIS = *intersect_module;
        pgDesc.hitgroup.entryFunctionNameIS = NameIS.c_str();
    }
    PRINT_VAR(pgDesc.hitgroup.entryFunctionNameCH)
    OPTIX_CHECK(optixProgramGroupCreate(OptixContext::get()->optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,
                                        &sizeof_log,
                                        &hitgroupPGs[ray2index[label]]));

    if (sizeof_log > 1)
        PRINT(log);
    return *this;
}

OptixPipeline
OptixPrograms::get_pipeline(
    OptixPipelineCompileOptions pipelineCompileOptions,
    OptixPipelineLinkOptions pipelineLinkOptions)
{
    OptixPipeline pipeline;

    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
        programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
        programGroups.push_back(pg);
    for (auto pg : missPGs)
        programGroups.push_back(pg);
    char log[2048];
    size_t sizeof_log = sizeof(log);

    OPTIX_CHECK(optixPipelineCreate(OptixContext::get()->optixContext,
                                    &pipelineCompileOptions,
                                    &pipelineLinkOptions,
                                    programGroups.data(),
                                    (int)programGroups.size(),
                                    log,
                                    &sizeof_log,
                                    &pipeline));

    if (sizeof_log > 1)
        PRINT(log);
    OPTIX_CHECK(
        optixPipelineSetStackSize(/* [in] The pipeline to configure the
                                     stack size for */
                                  pipeline,
                                  /* [in] The direct stack size requirement
                                     for
                                     direct callables invoked from IS or AH.
                                   */
                                  2 * 1024,
                                  /* [in] The direct stack size requirement
                                     for direct callables invoked from RG,
                                     MS, or CH.  */
                                  2 * 1024,
                                  /* [in] The continuation stack
                                     requirement. */
                                  2 * 1024,
                                  /* [in] The maximum depth of a traversable
                                     graph passed to trace. */
                                  1));
    if (sizeof_log > 1)
        PRINT(log);
    return pipeline;
}

void
OptixPrograms::buildSBT(const OptixStaticScene& scene)
{
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RaygenRecord> raygenRecords;
    raygenRecords.reserve(raygenPGs.size());
    for (auto& raygenPG : raygenPGs) {
        RaygenRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG, &rec));
        rec.data = nullptr; /* for now ... */
        raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    missRecords.reserve(missPGs.size());
    for (auto& missPG : missPGs) {
        MissRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(missPG, &rec));
        rec.data = nullptr; /* for now ... */
        missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------
    int numObjects = (int)scene.num_meshes;
    std::vector<HitgroupRecord> hitgroupRecords;
    auto& buffers = scene.mesh_buffers();
    const auto& meshes = scene.meshes();
    for (auto meshID = 0u; meshID < buffers.size(); meshID++) {
        auto& mesh = meshes[meshID];
        for (auto& pg : hitgroupPGs) {
            HitgroupRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(pg, &rec));
            rec.data.color = { 1, 1, 1 };
            if (mesh.diffuseTextureID >= 0) {
                rec.data.hasTexture = true;
                rec.data.texture =
                    scene.texture_objects()[mesh.diffuseTextureID];
            } else {
                rec.data.hasTexture = false;
            }
            rec.data.index = (vec3i*)buffers.index[meshID].d_pointer();
            rec.data.vertex = (vec3f*)buffers.vertex[meshID].d_pointer();
            rec.data.normal = (vec3f*)buffers.normal[meshID].d_pointer();
            rec.data.texcoord = (vec2f*)buffers.texcoord[meshID].d_pointer();
            hitgroupRecords.push_back(rec);
        }
    }
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

} // namespace chameleon
