#pragma once
#include <array>
#include <chameleon_renderer/materials/barytex/MaterialModel.hpp>
#include <chameleon_renderer/materials/barytex/MeasurementHit.hpp>
#include <chameleon_renderer/materials/barytex/viz.hpp>
#include <chameleon_renderer/utils/math_utils.hpp>
#include <memory>

namespace chameleon {
static const glm::vec2 base_A = {0.f, 0.f};
static const glm::vec2 base_B = {1.f, 0.f};
static const glm::vec2 base_C = {0.5f, std::sqrt(0.75f)};

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
                                            Wavelength wavelength) const {
            return material.get_reflectance(normal, in, out, wavelength);
        }
    };
    using point_array_t = std::vector<const GridPoint>;

    struct Node {
        struct BarycentricCache {
            glm::vec2 v0;
            glm::vec2 v1;
            float d00;
            float d01;
            float d11;
            float invDenom;

            BarycentricCache(const glm::vec2& A, const glm::vec2& B,
                             const glm::vec2& C)
                : v0(B - A),
                  v1(C - A),
                  d00(glm::dot(v0, v0)),
                  d01(glm::dot(v0, v1)),
                  d11(glm::dot(v1, v1)),
                  invDenom(1.0 / (d00 * d11 - d01 * d01)) {}
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

        Node() = default;
        Node(Node* parent, int A_id, int B_id, int C_id,
             const point_array_t& points)
            : parent(parent),
              A_id(A_id),
              B_id(B_id),
              C_id(C_id),
              points(points) {
            triangle.A = points[A_id].position;
            triangle.B = points[B_id].position;
            triangle.C = points[C_id].position;
            glm::vec2 MA = (triangle.C + triangle.B) / 2.f;
            glm::vec2 MB = (triangle.A + triangle.C) / 2.f;
            glm::vec2 MC = (triangle.A + triangle.B) / 2.f;
        }

        int get_max_depth() {
            int res = 0;
            for (const auto& ch : children) {
                if (ch) {
                    res = std::max(ch->get_max_depth() + 1, res);
                }
            }
            return res;
        }

        glm::vec3 compute_barycentric(const glm::vec2& point) const {
            auto v2 = point - triangle.A;
            float d20 = glm::dot(v2, bary_cache.v0);
            float d21 = glm::dot(v2, bary_cache.v1);
            float v = (bary_cache.d11 * d20 - bary_cache.d01 * d21) * bary_cache.invDenom;
            float w = (bary_cache.d00 * d21 - bary_cache.d01 * d20) * bary_cache.invDenom;
            float u = 1.0f - v - w;
            return {u,v,w};
        }

        float compute_reflectance(const glm::vec2& point, const glm::vec3& in,
                                  const glm::vec3& out, Wavelength wavelength,
                                  int depth) {
            if (depth <= 0) {
                auto A_refl = points[A_id].get_reflectance(in, out, wavelength);
                auto B_refl = points[B_id].get_reflectance(in, out, wavelength);
                auto C_refl = points[C_id].get_reflectance(in, out, wavelength);
                compute_barycentric(point);
                return;
            }
            int pt_size = depth;
            int thickness = depth;

            // TODO: rewrite to determinant -> no 3rd dimension
            glm::vec3 v = {MB.x - MC.x, MB.y - MC.y, 0};
            glm::vec3 tv = {point.x - MC.x, point.y - MC.y, 0};
            glm::vec3 pv = {A.x - MC.x, A.y - MC.y, 0};

            if ((glm::cross(tv, v).z * glm::cross(pv, v).z) > 0) {
                return children[0]->compute_reflectance(point, in,out,wavelength, depth - 1);
            } else {
                v = {MA.x - MC.x, MA.y - MC.y, 0};
                tv = {point.x - MC.x, point.y - MC.y, 0};
                pv = {B.x - MC.x, B.y - MC.y, 0};

                if ((glm::cross(tv, v).z * glm::cross(pv, v).z) > 0) {
                    return children[1]->compute_reflectance(point, in,out,wavelength, depth - 1);
                } else {
                    v = {MA.x - MB.x, MA.y - MB.y, 0};
                    tv = {point.x - MB.x, point.y - MB.y, 0};
                    pv = {C.x - MB.x, C.y - MB.y, 0};
                    if ((glm::cross(tv, v).z * glm::cross(pv, v).z) > 0) {
                        return children[2]->compute_reflectance(point, in,out,wavelength, depth - 1);

                    } else {
                        return children[3]->compute_reflectance(point, in,out,wavelength, depth - 1);
                    }
                }
            }
        }
    };

    MaterialTree(const std::vector<MeasurementHit>& measurements, int depth = 2)
        : root(std::make_unique<Node>()),depth(depth) {
    }
    int get_max_depth() { return depth}

    std::unique_ptr<Node> root;
    int depth;
};

}  // namespace chameleon
