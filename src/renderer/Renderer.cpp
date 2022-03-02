#include <chameleon_renderer/renderer/Renderer.hpp>

namespace chameleon {

void Renderer::add_camera(const std::string& cam_label,
                          const CalibratedCamera& cam) {
    cameras[cam_label] = cam;
}

CalibratedCamera&
Renderer::camera(const std::string& cam_label)
{
    if (cameras.count(cam_label) > 0) {
        return cameras[cam_label];
    }
    throw std::invalid_argument("[" + std::string(__PRETTY_FUNCTION__) +
                                "] unknown camera: " + cam_label);
}


void Renderer::setup_scene(const OptixStaticScene& new_scene) {
    scene = new_scene;
    scene.setupAccel();
    programs.buildSBT(scene);
}
}  // namespace chameleon
