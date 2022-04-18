#include <chameleon_renderer/scene/Light.hpp>
#include <chameleon_renderer/cuda/CUDABuffer.hpp>
#include <chameleon_renderer/utils/eigen_utils.hpp>

namespace chameleon {
eigen_utils::Vec3<float> Light::get_position() {
    throw std::runtime_error(
        std::string("Unimplemented method ") + __PRETTY_FUNCTION__ +
        std::string(" in light_type: ") + std::to_string(int(get_type())));
}
eigen_utils::Vec3<float> Light::get_direction() {
    throw std::runtime_error(
        std::string("Unimplemented method ") + __PRETTY_FUNCTION__ +
        std::string(" in light_type: ") + std::to_string(int(get_type())));
}
void Light::set_position(const eigen_utils::Vec3<float>&) {
    throw std::runtime_error(
        std::string("Unimplemented method ") + __PRETTY_FUNCTION__ +
        std::string(" in light_type: ") + std::to_string(int(get_type())));
}
void Light::set_direction(const eigen_utils::Vec3<float>&) {
    throw std::runtime_error(
        std::string("Unimplemented method ") + __PRETTY_FUNCTION__ +
        std::string(" in light_type: ") + std::to_string(int(get_type())));
}
const LedRadialCharacteristic& Light::get_radial(){
    throw std::runtime_error(
        std::string("Unimplemented method ") + __PRETTY_FUNCTION__ +
        std::string(" in light_type: ") + std::to_string(int(get_type())));
}
void Light::set_radial(LedRadialCharacteristic){
    throw std::runtime_error(
        std::string("Unimplemented method ") + __PRETTY_FUNCTION__ +
        std::string(" in light_type: ") + std::to_string(int(get_type())));
}
void Light::set_intensity(float new_intensity) { intensity = new_intensity; }
void Light::set_color(const glm::vec3& new_color) { color = new_color; }
const glm::vec3& Light::get_color() const { return color; }
float Light::get_intensity() { return intensity; }

CUDALight Light::convert_to_cuda(const eigen_utils::Mat4<float>& object_matrix) {
    using namespace eigen_utils;
    auto l_type = get_type();
    CUDALight out;
    out.color = {color.x, color.y, color.z};
    out.l_type = l_type;
    out.intensity = get_intensity();

    if (l_type == LightType::POINT || l_type == LightType::SPOT || l_type == LightType::LED_LIGHT) {
        out.position = from_eigen_v3(homogenus_transform(
            object_matrix, get_position(), true));
    }
    if (l_type == LightType::DIRECT || l_type == LightType::SPOT || l_type == LightType::LED_LIGHT) {
        out.direction = from_eigen_v3(homogenus_transform(
            object_matrix, get_direction(), false));
    }
    if(l_type == LightType::LED_LIGHT){
        CUDABuffer buff;
        buff.alloc_and_upload(get_radial().ies);
        out.radial_attenuation.data = reinterpret_cast<float*>(buff.d_ptr);
        out.radial_attenuation.size = get_radial().ies.size();
    }
    return out;
}

LightType SpotLight::get_type() { return LightType::SPOT; }

eigen_utils::Vec3<float> SpotLight::get_position() { return position; }
void SpotLight::set_position(const eigen_utils::Vec3<float>& pos) {
    position = pos;
}
eigen_utils::Vec3<float> SpotLight::get_direction() { return direction; }
void SpotLight::set_direction(const eigen_utils::Vec3<float>& dir) {
    direction = dir;
}

PointLight::PointLight(eigen_utils::Vec3<float> position)
    : position(position) {}

LightType PointLight::get_type() { return LightType::POINT; }

eigen_utils::Vec3<float> PointLight::get_position() { return position; }
void PointLight::set_position(const eigen_utils::Vec3<float>& pos) {
    position = pos;
}

DirectionLight::DirectionLight(eigen_utils::Vec3<float> direction)
    : direction(direction) {}

LightType DirectionLight::get_type() { return LightType::DIRECT; }
eigen_utils::Vec3<float> DirectionLight::get_direction() { return direction; }
void DirectionLight::set_direction(const eigen_utils::Vec3<float>& dir) {
    direction = dir;
}


LightType LedLight::get_type() { return LightType::LED_LIGHT; }

void LedLight::set_radial(LedRadialCharacteristic new_ies){
    ies = std::move(new_ies);
}
const LedRadialCharacteristic& LedLight::get_radial(){
    return ies;
}
}  // namespace chameleon