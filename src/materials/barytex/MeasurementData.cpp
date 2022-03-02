#include <chameleon_renderer/materials/barytex/MeasurementData.hpp>

namespace chameleon {

std::unordered_map<int,std::vector<MeasurementHit>>
SingleMeasurementData::compress_hits() const {
    std::unordered_map<int,std::vector<MeasurementHit>> out;

    for (const auto& hit : measurements) {
        if (hit.is_valid) {
            out[hit.triangle_id].push_back(hit);
        }
    }
    return out;
}

void SingleMeasurementData::download(const CUDABuffer& buffer) {
    measurements.clear();
    buffer.download(measurements);
}

}  // namespace chameleon