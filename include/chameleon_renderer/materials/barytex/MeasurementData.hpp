#pragma once
#include <MeasurementHit.hpp>
#include <unordered_set>
#include <chameleon_renderer/cuda/CUDABuffer.hpp>
#include <vector>

namespace chameleon {

struct SingleMeasurementData {
    std::vector<MeasurementHit> measurements;

    std::unordered_set<std::vector<MeasurementHit>> compress_hits() const {
        std::unordered_set<std::vector<MeasurementHit>> out;

        for (const auto& hit : measurements) {
            if (hit.is_valid) {
                out[hit.triangle_id].push_back(out);
            }
        }
        return out;
    }

    void download(const CUDABuffer& buffer){
        measurements.clear();
        buffer.download(measurements);
    }
};

}  // namespace chameleon