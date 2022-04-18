#pragma once
#include <chameleon_renderer/cuda/CUDALight.h>
#include <chameleon_renderer/utils/eigen_utils.hpp>
#include <chameleon_renderer/utils/json.hpp>
#include <chameleon_renderer/HW/LedRadialCharacteristic.hpp>

namespace chameleon {
struct ILight {
    virtual eigen_utils::Vec3<float> get_position() = 0;
    virtual void set_position(const eigen_utils::Vec3<float>&) = 0;
    virtual eigen_utils::Vec3<float> get_direction() = 0;
    virtual void set_direction(const eigen_utils::Vec3<float>&) = 0;
    virtual float get_intensity() = 0;
    virtual void set_intensity(float) = 0;
    virtual const glm::vec3& get_color() const = 0;
    virtual void set_color(const glm::vec3& color) = 0;

    virtual void set_radial(LedRadialCharacteristic) = 0;
    virtual const LedRadialCharacteristic& get_radial() = 0;
    

    virtual LightType get_type() = 0;
    virtual CUDALight convert_to_cuda(const eigen_utils::Mat4<float>& object_mat= eigen_utils::Mat4<float>::Identity()) = 0;
};

struct Light : ILight {
    eigen_utils::Vec3<float> get_position() override;
    eigen_utils::Vec3<float> get_direction() override;
    void set_position(const eigen_utils::Vec3<float>&) override;
    void set_direction(const eigen_utils::Vec3<float>&) override;
    void set_intensity(float new_intensity) override;
    void set_color(const glm::vec3& new_color) override;

    void set_radial(LedRadialCharacteristic) override;
    const LedRadialCharacteristic& get_radial() override;
    const glm::vec3& get_color() const override;

    float get_intensity() override;

    CUDALight convert_to_cuda(const eigen_utils::Mat4<float>& object_mat = eigen_utils::Mat4<float>::Identity()) override;

   private:
    float intensity = 1;
    glm::vec3 color = {1, 1, 1};
};

class SpotLight : public Light {
    eigen_utils::Vec3<float> position;
    eigen_utils::Vec3<float> direction;

   public:
    LightType get_type() override;

    eigen_utils::Vec3<float> get_position() override;
    void set_position(const eigen_utils::Vec3<float>& pos) override;
    eigen_utils::Vec3<float> get_direction() override;
    void set_direction(const eigen_utils::Vec3<float>& dir);
};

class PointLight : public Light {
    eigen_utils::Vec3<float> position;

   public:
    PointLight() = default;
    PointLight(eigen_utils::Vec3<float> position);

    LightType get_type() override;

    eigen_utils::Vec3<float> get_position() override;
    void set_position(const eigen_utils::Vec3<float>& pos) override;
};

class DirectionLight : public Light {
    eigen_utils::Vec3<float> direction;

   public:
    DirectionLight() = default;
    DirectionLight(eigen_utils::Vec3<float> direction);

    LightType get_type() override;
    eigen_utils::Vec3<float> get_direction() override;
    void set_direction(const eigen_utils::Vec3<float>& dir) override;
};

class LedLight : public SpotLight{
    LedRadialCharacteristic ies;

public:
    LedLight()=default;
    LightType get_type() override;

    void set_radial(LedRadialCharacteristic) override;
    const LedRadialCharacteristic& get_radial() override;

}; 

}  // namespace chameleon