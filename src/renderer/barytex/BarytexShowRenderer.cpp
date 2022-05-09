#include <chameleon_renderer/renderer/barytex/BarytexShowRenderer.hpp>
#include <chameleon_renderer/cuda/CUDABuffer.hpp>

namespace chameleon {
void BarytexShowRenderer::OutputData::resize(int x, int y) {
    mask.resize(x, y);
    view.resize(x, y);
}

void BarytexShowRenderer::OutputData::clear() {
    mask.clear();
    view.clear();
}

BarytexShowRenderer::launch_params_t::Layers
BarytexShowRenderer::get_cuda_output() {
    launch_params_t::Layers out;
    out.size = {int(out_data.view.res_x), int(out_data.view.res_y)};
    out.view = (decltype(out.view))out_data.view.buffer_ptr();
    out.mask = (decltype(out.mask))out_data.mask.buffer_ptr();
    return out;
}

BarytexShowRenderer::BarytexShowRenderer() {
    pipelineCompileOptions.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName =
        "optixLaunchParams";
}

/**
 * @brief
 * @param
 *
 */
void BarytexShowRenderer::setup() {
    raygen_module =
        base_module_create("./barytex_shaders/show/raygen.ptx",
                           moduleCompileOptions, pipelineCompileOptions);
    miss_module =
        base_module_create("./barytex_shaders/show/miss.ptx",
                           moduleCompileOptions, pipelineCompileOptions);
    anyhit_module =
        base_module_create("./barytex_shaders/show/any_hit.ptx",
                           moduleCompileOptions, pipelineCompileOptions);
    closest_hit_module =
        base_module_create("./barytex_shaders/show/closest_hit.ptx",
                           moduleCompileOptions, pipelineCompileOptions);
    programs.add_raygen_program("renderFrame", raygen_module);
    programs.add_ray("radiance", {miss_module}, {anyhit_module},
                     {closest_hit_module});
    programs.add_ray("shadow", {miss_module}, {anyhit_module},
                     {closest_hit_module});
    pipeline =
        programs.get_pipeline(pipelineCompileOptions, pipelineLinkOptions);
    launch_params_buff.alloc(sizeof(launch_params_t));
}

void BarytexShowRenderer::add_camera(const std::string& cam_label,
                                      const PhotometryCamera& cam) {
    photometry_cameras[cam_label] = cam;
}

PhotometryCamera& BarytexShowRenderer::photometry_camera(
    const std::string& cam_label) {
    if (photometry_cameras.count(cam_label) > 0) {
        return photometry_cameras[cam_label];
    }
    throw std::invalid_argument("[" + std::string(__PRETTY_FUNCTION__) +
                                "] unknown camera: " + cam_label);
}

const BarytexShowRenderer::OutputData& BarytexShowRenderer::render(
    const std::string& camera_label) {
    if (photometry_cameras.count(camera_label) <= 0) {
        spdlog::error("[ {} ] unsupported camera: {}", __PRETTY_FUNCTION__,
                      camera_label);
        throw std::invalid_argument("unsupported camera: " + camera_label);
    }

    auto& camera = photometry_cameras[camera_label];
    auto res = camera.resolution();
    out_data.resize(res(0),res(1));
    spdlog::info("Start render");
    TICK;
    launch_params.render_data = get_cuda_output();
    launch_params.camera = camera.get_cuda();
    launch_params.light_data = camera.get_cuda_light_array();
    launch_params.traversable = scene.traversable;
    if(!material_forest.has_value()){
        spdlog::error("no material to render");
        throw std::runtime_error("no material known");
    }
    launch_params.material_forest = *material_forest;
    // _clear_buffers();
    // update_scene();
    launch_params_buff.upload(&launch_params, 1);
    // OPTIX_CHECK(
    optixLaunch(/*! pipeline we're launching launch: */
                pipeline, OptixContext::get()->stream,
                /*! parameters and SBT */
                launch_params_buff.d_pointer(), launch_params_buff.sizeInBytes,
                &programs.sbt,
                /*! dimensions of the launch: */
                res(0), res(1), 1);
    // );
    CUDA_SYNC_CHECK();
    TOCK;
    spdlog::info("Render done");
    return out_data;
}

void BarytexShowRenderer::setup_material(const MaterialLookupForest& mlf){
    if(material_forest.has_value()){
        cudaFree(material_forest->trees);
        material_forest.reset();
    }
    PING;
    material_forest.emplace(mlf.upload_to_cuda());
    PING;

}

}  // namespace chameleon