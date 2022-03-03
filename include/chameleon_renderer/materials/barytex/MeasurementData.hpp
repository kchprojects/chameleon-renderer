#pragma once
#include <chameleon_renderer/materials/barytex/MeasurementHit.hpp>
#include <chameleon_renderer/cuda/CUDABuffer.hpp>
#include <unordered_map>
#include <vector>

namespace chameleon {

struct SingleMeasurementData {
    std::vector<MeasurementHit> measurements;

    std::unordered_map<int,std::vector<MeasurementHit>> compress_hits() const;

    void download(const CUDABuffer& buffer);
};

}  // namespace chameleon