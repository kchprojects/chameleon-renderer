#pragma once
#include <chameleon_renderer/materials/barytex/CUDAMaterial.hpp>

namespace chameleon {
struct CUDAMaterialLookupTree {
    struct MeasurementPoint {
        CUDAGenericMaterial material;
        glm::vec3 position;
    };
    struct Divisor {
        int A_id;
        int B_id;
        int C_id;
        int child_id = -1;
        // cache ?
    };

    struct DivisorNode {
        int parent_id = -1;
        Divisor divisors[4];
        int A, B, C;
    };

    MeasurementPoint* points;
    size_t point_count = 0;

    DivisorNode* divisors;
    size_t divisors_count = 0;
};
}  // namespace chameleon
