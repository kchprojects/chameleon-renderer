#pragma once
#include <chameleon_renderer/materials/barytex/MeasurementHit.hpp>
#include <chameleon_renderer/utils/file_utils.hpp>
#include <chameleon_renderer/utils/math_utils.hpp>
#include <vector>

namespace chameleon {

enum class Wavelength { Grayscale, Red, Green, Blue };

// TODO c++-20 concepts
template <typename Derived>
struct MaterialModel {
    MaterialModel() {
        // Test interface without virtual methods in compiletime
        float f = static_cast<const Derived*>(this)->get_reflectance(
            glm::vec3(1.f, 0.f, 0.f), glm::vec3(1.f, 0.f, 0.f),
            glm::vec3(1.f, 0.f, 0.f), Wavelength::Grayscale);

        // static_cast<const Derived*>(this)->learn_single();
    }
};

class MaterialModelMock : public MaterialModel<MaterialModelMock> {
   public:
    MaterialModelMock(const glm::vec3& real_position,
                      const std::vector<MeasurementHit>& hits);

    float get_reflectance(const glm::vec3& normal, const glm::vec3& in,
                          const glm::vec3& out,
                          Wavelength w = Wavelength::Grayscale) const;

    void load(const fs::path&);
};

}  // namespace chameleon