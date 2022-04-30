#pragma once
#include <chameleon_renderer/materials/barytex/CUDAMaterialLookupTree.hpp>

namespace chameleon {
struct CUDAMaterialLookupForest {
    CUDAMaterialLookupTree* trees;
    size_t tree_count = 0;
};
}  // namespace chameleon