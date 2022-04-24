#include <chameleon_renderer/materials/barytex/MaterialLookupForest.hpp>

namespace chameleon {

MaterialLookupForest::MaterialLookupForest(const TriangleMesh& mesh,
                                           float minimal_distance,
                                           int maximal_tree_depth) {
    for (const auto& face : mesh.index) {
        trees.emplace_back(mesh.vertex[face[0]],
                           mesh.vertex[face[1]],
                           mesh.vertex[face[2]], minimal_distance,
                           maximal_tree_depth);
    }
}
MaterialLookupForest::MaterialLookupForest(const nlohmann::json& j){
    for(const auto& tree_j : j){
        trees.emplace_back(tree_j);
    }
}

nlohmann::json MaterialLookupForest::serialize() const {
    nlohmann::json out = nlohmann::json::array();
    for (const auto& tree : trees) {
        out.push_back(tree.serialize());
    }
    return out;
}
}  // namespace chameleon