#include <chameleon_renderer/renderer/barytex/BarytexLearnRenderer.hpp>

namespace chameleon {
void BarytexLearnRenderer::OutputData::resize(size_t maximum_hitcount) {
    measurements.resize(maximum_hitcount);
}

void BarytexLearnRenderer::OutputData::clear() { measurements.clear(); }

CUDAArray<MeasurementHit>
BarytexLearnRenderer::get_cuda() {
    CUDAArray<MeasurementHit> out;
    out.data =
        reinterpret_cast<MeasurementHit*>(out_data.measurements.buffer_ptr());
    out.size = out_data.measurements.size;
    return out;
}

BarytexLearnRenderer::BarytexLearnRenderer() {
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
void BarytexLearnRenderer::setup() {
    PING raygen_module =
        base_module_create("./barytex_shaders/raygen.ptx", moduleCompileOptions,
                           pipelineCompileOptions);
    miss_module = base_module_create(
        "./barytex_shaders/miss.ptx", moduleCompileOptions, pipelineCompileOptions);
    anyhit_module =
        base_module_create("./barytex_shaders/any_hit.ptx", moduleCompileOptions,
                           pipelineCompileOptions);
    closest_hit_module =
        base_module_create("./barytex_shaders/closest_hit.ptx", moduleCompileOptions,
                           pipelineCompileOptions);
    PING programs.add_raygen_program("renderFrame", raygen_module);
    programs.add_ray("radiance", {miss_module}, {anyhit_module},
                     {closest_hit_module});
    programs.add_ray("shadow", {miss_module}, {anyhit_module},
                     {closest_hit_module});
    PING pipeline =
        programs.get_pipeline(pipelineCompileOptions, pipelineLinkOptions);
    PING launch_params_buff.alloc(sizeof(launch_params_t));
}

void BarytexLearnRenderer::add_camera(const std::string& cam_label,
                                      const PhotometryCamera& cam) {
    photometry_cameras[cam_label] = cam;
}

PhotometryCamera& BarytexLearnRenderer::photometry_camera(
    const std::string& cam_label) {
    if (photometry_cameras.count(cam_label) > 0) {
        return photometry_cameras[cam_label];
    }
    throw std::invalid_argument("[" + std::string(__PRETTY_FUNCTION__) +
                                "] unknown camera: " + cam_label);
}

const BarytexLearnRenderer::OutputData& BarytexLearnRenderer::render(const BarytexObservation& observation) {
    std::string camera_label = observation.cam_label;
    if (photometry_cameras.count(camera_label) <= 0) {
        spdlog::error("[ {} ] unsupported camera: {}", __PRETTY_FUNCTION__,
                      camera_label);
        throw std::invalid_argument("unsupported camera: " + camera_label);
    }

    auto& camera = photometry_cameras[camera_label];
    auto res = camera.resolution();

    if (res(0) != observation.image.cols || res(1) != observation.image.rows) {
        spdlog::error("[ {} ] input image not matching camera resolution",
                      __PRETTY_FUNCTION__);

        throw std::invalid_argument(
            "input image not matching camera resolution");
    }

    int cols =observation.image.cols;
    int rows =observation.image.rows;
    out_data.measurements.resize(cols*rows*launch_params.sample_multiplier);
    spdlog::info("Start render");
    TICK;

    launch_params.render_data = get_cuda();
    launch_params.camera = camera.get_cuda();
    launch_params.observation = observation.get_cuda(camera.object_matrix());

    launch_params.traversable = scene.traversable;

    // _clear_buffers();
    // update_scene();
    launch_params_buff.upload(&launch_params, 1);
    out_data.clear();
    TICK;
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

    CUDA_CHECK(cudaFree(launch_params.observation.light.radial_attenuation.data));
    return out_data;
}

}  // namespace chameleon