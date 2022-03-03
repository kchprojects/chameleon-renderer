#include <chameleon_renderer/materials/barytex/MaterialModel.hpp>

namespace chameleon {

MaterialModelMock::MaterialModelMock(const glm::vec3& real_position,
                                     const std::vector<MeasurementHit>& hits) {
    // TODO: create model from hits
}

float MaterialModelMock::get_reflectance(const glm::vec3& normal,
                                         const glm::vec3& in,
                                         const glm::vec3& out,
                                         Wavelength w) const {
    // for mock only cosine
    return glm::dot(glm::normalize(normal), glm::normalize(in));
}

void MaterialModelMock::load(const fs::path&) {}

}  // namespace chameleon