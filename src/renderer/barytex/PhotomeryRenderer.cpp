#include <chameleon_renderer/renderer/Renderer.hpp>

namespace chameleon {
void
PhotometryRenderer::OutputLayers::resize(size_t x, size_t y)
{
    normal_map.resize(x, y);
    uv_map.resize(x, y);
    mask.resize(x, y);
    view.resize(x, y);
}
void
PhotometryRenderer::OutputLayers::clear()
{
    normal_map.clear();
    uv_map.clear();
    mask.clear();
    view.clear();
}

PhotometryRenderer::launch_params_t::Layers
PhotometryRenderer::OutputLayers::get_cuda()
{
    launch_params_t::Layers out;
    out.size = { int(normal_map.res_x), int(normal_map.res_y) };
    out.normal_map = (decltype(out.normal_map))normal_map.buffer_ptr();

    out.uv_map = (decltype(out.uv_map))uv_map.buffer_ptr();
    out.view = (decltype(out.view))view.buffer_ptr();
    out.mask = (decltype(out.mask))mask.buffer_ptr();
    return out;
}

PhotometryRenderer::PhotometryRenderer()
{
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
void
PhotometryRenderer::setup()
{
    PING raygen_module = base_module_create(
        "./raygen.ptx", moduleCompileOptions, pipelineCompileOptions);
    miss_module = base_module_create(
        "./miss.ptx", moduleCompileOptions, pipelineCompileOptions);
    anyhit_module = base_module_create(
        "./any_hit.ptx", moduleCompileOptions, pipelineCompileOptions);
    closest_hit_module = base_module_create(
        "./closest_hit.ptx", moduleCompileOptions, pipelineCompileOptions);
    PING programs.add_raygen_program("renderFrame", raygen_module);
    programs.add_ray(
        "radiance", { miss_module }, { anyhit_module }, { closest_hit_module });
    // programs.add_ray("shadow", {miss_module}, {anyhit_module},
    //                  {closest_hit_module});
    PING pipeline =
        programs.get_pipeline(pipelineCompileOptions, pipelineLinkOptions);
    PING launch_params_buff.alloc(sizeof(launch_params_t));
}

void
PhotometryRenderer::add_camera(const std::string& cam_label,
                               const PhotometryCamera& cam)
{
    photometry_cameras[cam_label] = cam;
}

PhotometryCamera&
PhotometryRenderer::photometry_camera(const std::string& cam_label)
{
    if (photometry_cameras.count(cam_label) > 0) {
        return photometry_cameras[cam_label];
    }
    throw std::invalid_argument("[" + std::string(__PRETTY_FUNCTION__) +
                                "] unknown camera: " + cam_label);
}

const PhotometryRenderer::OutputLayers&
PhotometryRenderer::render(const std::string& camera_label)
{
    if (photometry_cameras.count(camera_label) > 0) {
        auto& camera = photometry_cameras[camera_label];
        auto res = camera.resolution();
        out_layers.resize(res(0), res(1));
        std::cout << "Start render" << std::endl;
        TICK;

        launch_params.layers = out_layers.get_cuda();
        launch_params.camera = camera.get_cuda();
        launch_params.light_data = camera.get_cuda_light_array();
        launch_params.traversable = scene.traversable;

        // _clear_buffers();
        // update_scene();
        launch_params_buff.upload(&launch_params, 1);
        out_layers.clear();
        TICK;
        // OPTIX_CHECK(
        optixLaunch(/*! pipeline we're launching launch: */
                    pipeline,
                    OptixContext::get()->stream,
                    /*! parameters and SBT */
                    launch_params_buff.d_pointer(),
                    launch_params_buff.sizeInBytes,
                    &programs.sbt,
                    /*! dimensions of the launch: */
                    res(0),
                    res(1),
                    1);
        // );
        CUDA_SYNC_CHECK();
        TOCK;
        std::cout << "Render done" << std::endl;
    } else {
        throw std::invalid_argument("[" + std::string(__PRETTY_FUNCTION__) +
                                    "] unsupported camera: " + camera_label);
    }
    return out_layers;
}

} // namespace chameleon
