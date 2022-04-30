#include <chameleon_renderer/cuda/CUDABuffer.hpp>
#include <chameleon_renderer/materials/barytex/MaterialLookupForest.hpp>

namespace chameleon {

MaterialLookupForest::MaterialLookupForest(const TriangleMesh& mesh,
                                           float minimal_distance,
                                           int maximal_tree_depth) {
    int i = 0;
    for (const auto& face : mesh.index) {
        std::cout << float(i++) * 100.f / mesh.index.size()
                  << "%                  \r" << std::flush;
        trees.emplace_back(mesh.vertex[face[0]], mesh.vertex[face[1]],
                           mesh.vertex[face[2]], minimal_distance,
                           maximal_tree_depth);
    }
}
MaterialLookupForest::MaterialLookupForest(const nlohmann::json& j) {
    for (const auto& tree_j : j) {
        trees.emplace_back(tree_j);
    }
}

nlohmann::json MaterialLookupForest::serialize() const {
    nlohmann::json out = nlohmann::json::array();
    int i = 0;
    for (const auto& tree : trees) {
        std::cout << "saving " << float(i) * 100.f / trees.size()
                  << "%                  \r" << std::flush;
        out.push_back(tree.serialize());
        ++i;
    }
    return out;
}

CUDAMaterialLookupForest MaterialLookupForest::upload_to_cuda() const {
    std::vector<CUDAMaterialLookupTree> cuda_forest;
    cuda_forest.reserve(trees.size());
    for (const auto& tree : trees) {
        cuda_forest.push_back(tree.upload_to_cuda());
    }
    CUDABuffer buff;
    buff.alloc_and_upload(cuda_forest);
    return {reinterpret_cast<CUDAMaterialLookupTree*>(buff.d_pointer()),
            trees.size()};
}
}  // namespace chameleon