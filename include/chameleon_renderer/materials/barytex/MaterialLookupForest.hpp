#pragma once
#include <chameleon_renderer/materials/barytex/MeasurementLookupTree.hpp>
#include <chameleon_renderer/materials/barytex/CUDAMaterialLookupForest.hpp>
#include <chameleon_renderer/scene/Model.hpp>

namespace chameleon
{
    struct MaterialLookupForest{
        std::vector<MeasurementLookupTree> trees;

        MaterialLookupForest(const TriangleMesh& mesh, float minimal_distance, int maximal_tree_depth);
        MaterialLookupForest(const nlohmann::json& j);
        nlohmann::json serialize() const;

        CUDAMaterialLookupForest upload_to_cuda()const;
    };
} // namespace chameleon
