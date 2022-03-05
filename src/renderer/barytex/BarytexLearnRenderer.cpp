#include <chameleon_renderer/renderer/barytex/BarytexLearnRenderer.hpp>

namespace chameleon {
void BarytexLearnRenderer::OutputData::resize(size_t maximum_hitcount) {
    measurements.resize(maximum_hitcount);
}

void BarytexLearnRenderer::OutputData::clear() { measurements.clear(); }

void BarytexLearnRenderer::InputData::upload_source_images(
    std::vector<cv::Mat> images) {
    int i = 0;
    width = -1;
    height = -1;
    
    std::vector<void*> img_ptr_array;

    img_ptr_array.reserve(captured_images.size());
    if(images.size() != captured_images.size()){
        captured_images.resize(images.size());
    }
    
    for (const auto& img : images) {
        if (width == -1 && height == -1) {
            width = img.cols;
            height = img.rows;
        } else if (width != img.cols || height != img.rows) {
            throw std::invalid_argument("[" + std::string(__PRETTY_FUNCTION__) +
                                "] image sizes in one capturing must be the same");
        }
        captured_images[i].upload_cv_mat(img);
        img_ptr_array.push_back(captured_images[i].buffer_ptr());
        ++i;
    }

    images_ptr_array.alloc_and_upload<void*>(img_ptr_array);
}

void BarytexLearnRenderer::InputData::clear() { captured_images.clear(); }

BarytexLearnRenderer::launch_params_t::RenderData
BarytexLearnRenderer::get_cuda() {
    launch_params_t::RenderData out;
    out.out_data = reinterpret_cast<MeasurementHit*>(out_data.measurements.buffer_ptr());
    out.out_size = out_data.measurements.size;

    out.captured_images = reinterpret_cast<glm::vec3**>(in_data.images_ptr_array.d_ptr);
    out.height = in_data.height;
    out.img_count = in_data.captured_images.size();
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
        base_module_create("./BarytexLearnRaygen.ptx", moduleCompileOptions,
                           pipelineCompileOptions);
    miss_module = base_module_create(
        "./BarytexLearnMiss.ptx", moduleCompileOptions, pipelineCompileOptions);
    anyhit_module =
        base_module_create("./BarytexLearnAnyHit.ptx", moduleCompileOptions,
                           pipelineCompileOptions);
    closest_hit_module =
        base_module_create("./BarytexLearnClosestHit.ptx", moduleCompileOptions,
                           pipelineCompileOptions);
    PING programs.add_raygen_program("renderFrame", raygen_module);
    programs.add_ray("radiance", {miss_module}, {anyhit_module},
                     {closest_hit_module});
    // programs.add_ray("shadow", {miss_module}, {anyhit_module},
    //                  {closest_hit_module});
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

const BarytexLearnRenderer::OutputData& BarytexLearnRenderer::render(
    const std::string& camera_label) {
    if (photometry_cameras.count(camera_label) > 0) {
        auto& camera = photometry_cameras[camera_label];
        auto res = camera.resolution();
        if (res(0) != in_data.width || res(1) != in_data.height) {
            spdlog::error("[ {} ] input image not matching camera resolution",__PRETTY_FUNCTION__);

            throw std::invalid_argument("input image not matching camera resolution");
        }
        out_data.measurements.resize(res(0)*res(1)*in_data.captured_images.size());
        spdlog::info("Start render");
        TICK;

        launch_params.render_data = get_cuda();
        launch_params.camera = camera.get_cuda();
        launch_params.light_data = camera.get_cuda_light_array();
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
                    launch_params_buff.d_pointer(),
                    launch_params_buff.sizeInBytes, &programs.sbt,
                    /*! dimensions of the launch: */
                    res(0), res(1), 1);
        // );
        CUDA_SYNC_CHECK();
        TOCK;
        spdlog::info("Render done");
    } else {
        spdlog::error("[ {} ] unsupported camera: {}",__PRETTY_FUNCTION__,camera_label);
        throw std::invalid_argument("unsupported camera: " + camera_label);
    }
    return out_data;
}

}  // namespace chameleon