#pragma once
#include <chameleon_renderer/cuda/CudaLight.h>

#include <chameleon_renderer/utils/eigen_utils.hpp>
#include <chameleon_renderer/utils/json.hpp>

namespace chameleon {
struct ILight {
    virtual eigen_utils::Vec3<float> get_position() = 0;
    virtual void set_position(const eigen_utils::Vec3<float>&) = 0;
    virtual eigen_utils::Vec3<float> get_direction() = 0;
    virtual void set_direction(const eigen_utils::Vec3<float>&) = 0;
    virtual float get_intensity() = 0;
    virtual void set_intensity(float) = 0;
    virtual LightType get_type() = 0;
    virtual CudaLight convert_to_cuda()=0;
};

struct Light : ILight {
    float intensity = 1;

    eigen_utils::Vec3<float> get_position() override {
        throw std::runtime_error(
            std::string("Unimplemented method ") + __FUNCTION__ +
            std::string(" in light_type: ") + std::to_string(int(get_type())));
    };
    eigen_utils::Vec3<float> get_direction() override {
        throw std::runtime_error(
            std::string("Unimplemented method ") + __FUNCTION__ +
            std::string(" in light_type: ") + std::to_string(int(get_type())));
    }
    void set_position(const eigen_utils::Vec3<float>&) override {
        throw std::runtime_error(
            std::string("Unimplemented method ") + __FUNCTION__ +
            std::string(" in light_type: ") + std::to_string(int(get_type())));
    }
    void set_direction(const eigen_utils::Vec3<float>&) override {
        throw std::runtime_error(
            std::string("Unimplemented method ") + __FUNCTION__ +
            std::string(" in light_type: ") + std::to_string(int(get_type())));
    }
    void set_intensity(float new_intensity) override {
        intensity = new_intensity;
    }
    float get_intensity() override { return intensity; };

    CudaLight convert_to_cuda() override {
        auto l_type = get_type();
        CudaLight out;
        out.l_type = l_type;
        out.intensity = get_intensity();
        if (l_type == LightType::POINT || l_type == LightType::SPOT) {
            out.position = eigen_utils::from_eigen_v3(get_position());
        }
        if (l_type == LightType::DIRECT || l_type == LightType::SPOT) {
            out.direction = eigen_utils::from_eigen_v3(get_direction());
        }
        return out;
    }
};

class SpotLight : public Light {
    eigen_utils::Vec3<float> position;
    eigen_utils::Vec3<float> direction;

   public:
    LightType get_type() override { return LightType::SPOT; }

    eigen_utils::Vec3<float> get_position() override { return position; };
    void set_position(const eigen_utils::Vec3<float>& pos) override {
        position = pos;
    };
    eigen_utils::Vec3<float> get_direction() override { return direction; };
    void set_direction(const eigen_utils::Vec3<float>& dir) override {
        direction = dir;
    };
};

class PointLight : public Light {
    eigen_utils::Vec3<float> position;

   public:
    PointLight() = default;
    PointLight(eigen_utils::Vec3<float> position) : position(position) {}

    LightType get_type() override { return LightType::POINT; }

    eigen_utils::Vec3<float> get_position() override { return position; };
    void set_position(const eigen_utils::Vec3<float>& pos) override {
        position = pos;
    };
};

class DirectionLight : public Light {
    eigen_utils::Vec3<float> direction;

   public:
    DirectionLight() = default;
    DirectionLight(eigen_utils::Vec3<float> direction) : direction(direction) {}

    LightType get_type() override { return LightType::DIRECT; }
    eigen_utils::Vec3<float> get_direction() override { return direction; };
    void set_direction(const eigen_utils::Vec3<float>& dir) override {
        direction = dir;
    };
};

}  // namespace chameleon