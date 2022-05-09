#pragma once
#include <chameleon_renderer/materials/barytex/CUDAMaterialLookupForest.hpp>
#include <chameleon_renderer/materials/barytex/MeasurementLookupTree.hpp>
#include <chameleon_renderer/scene/Model.hpp>
#include <chameleon_renderer/utils/file_utils.hpp>

namespace chameleon {
struct MaterialLookupForest {
    std::vector<MeasurementLookupTree> trees;

    MaterialLookupForest(const TriangleMesh& mesh, float minimal_distance,
                         int maximal_tree_depth);
    MaterialLookupForest(const nlohmann::json& j);
    MaterialLookupForest() = default;
    nlohmann::json serialize() const;

    void serialize_bin(const fs::path&) const;
    void load_bin(const fs::path&);

    CUDAMaterialLookupForest upload_to_cuda() const;
};
}  // namespace chameleon
