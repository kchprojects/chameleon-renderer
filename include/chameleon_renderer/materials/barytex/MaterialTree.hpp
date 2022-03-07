#pragma once
#include <array>
#include <chameleon_renderer/materials/barytex/MaterialModel.hpp>
#include <chameleon_renderer/materials/barytex/MeasurementHit.hpp>
#include <chameleon_renderer/materials/barytex/viz.hpp>
#include <chameleon_renderer/utils/math_utils.hpp>
#include <memory>

namespace chameleon {

// TODO: adjust for general MaterialData from Measurement copy
// template<typename material_model_t>
struct MaterialTree {
    using material_model_t = MaterialModelMock;
    struct GridPoint {
        glm::vec2 position;
        glm::vec3 normal;
        material_model_t material;

        [[nodiscard]] float get_reflectance(const glm::vec3& in,
                                            const glm::vec3& out,
                                            Wavelength wavelength) const;
    };
    using point_array_t = std::vector<GridPoint>;

    struct Node {
        struct BarycentricCache {
            glm::vec2 v0;
            glm::vec2 v1;
            float d00;
            float d01;
            float d11;
            float invDenom;

            BarycentricCache(const glm::vec2& A, const glm::vec2& B,
                             const glm::vec2& C);
        };

        Node* parent = nullptr;
        int A_id = 0;
        int B_id = 1;
        int C_id = 2;
        Triangle2D triangle;

        glm::vec2 MA;
        glm::vec2 MB;
        glm::vec2 MC;
        BarycentricCache bary_cache;

        std::array<std::unique_ptr<Node>, 4> children;
        const point_array_t& points;

        Node(Node* parent, int A_id, int B_id, int C_id,
             const point_array_t& points);

        int get_max_depth();

        glm::vec3 compute_barycentric(const glm::vec2& point) const;

        float compute_reflectance(const glm::vec2& point, const glm::vec3& in,
                                  const glm::vec3& out, Wavelength wavelength,
                                  int depth);
    };

    MaterialTree(const std::vector<MeasurementHit>& measurements, int depth = 2);
    int get_max_depth();

    std::unique_ptr<Node> root;
    int depth;
};

}  // namespace chameleon
