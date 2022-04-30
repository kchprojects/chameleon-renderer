#pragma once
#include <chameleon_renderer/materials/barytex/MeasurementHit.hpp>

namespace chameleon {

    
struct IsotropicBRDFMeasurement{
    int triangle_id;
    glm::vec3 world_coordinates;
    glm::vec3 value;

    float eye_elevation;
    float light_elevation;
    float light_azimuth;

    IsotropicBRDFMeasurement()=default;
    IsotropicBRDFMeasurement(const MeasurementHit& hit);

    glm::vec3 eye() const;
    glm::vec3 light() const;
};

}  // namespace chameleon