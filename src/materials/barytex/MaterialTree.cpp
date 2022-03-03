#include <array>
#include <chameleon_renderer/materials/barytex/MaterialModel.hpp>
#include <chameleon_renderer/materials/barytex/MaterialTree.hpp>
#include <chameleon_renderer/materials/barytex/MeasurementHit.hpp>
#include <chameleon_renderer/materials/barytex/viz.hpp>
#include <chameleon_renderer/utils/math_utils.hpp>
#include <memory>

namespace chameleon {

[[nodiscard]] float MaterialTree::GridPoint::get_reflectance(
    const glm::vec3& in, const glm::vec3& out, Wavelength wavelength) const {
    return material.get_reflectance(normal, in, out, wavelength);
}

MaterialTree::Node::BarycentricCache::BarycentricCache(const glm::vec2& A,
                                                       const glm::vec2& B,
                                                       const glm::vec2& C)
    : v0(B - A),
      v1(C - A),
      d00(glm::dot(v0, v0)),
      d01(glm::dot(v0, v1)),
      d11(glm::dot(v1, v1)),
      invDenom(1.0 / (d00 * d11 - d01 * d01)) {}

MaterialTree::Node::Node(Node* parent, int A_id, int B_id, int C_id,
                         const point_array_t& points)
    : parent(parent),
      A_id(A_id),
      B_id(B_id),
      C_id(C_id),
      points(points),
      triangle({points.at(A_id).position, points.at(B_id).position,
                points.at(C_id).position}),
      bary_cache(triangle.A, triangle.B, triangle.C) {
    glm::vec2 MA = (triangle.C + triangle.B) / 2.f;
    glm::vec2 MB = (triangle.A + triangle.C) / 2.f;
    glm::vec2 MC = (triangle.A + triangle.B) / 2.f;
}

int MaterialTree::Node::get_max_depth() {
    int res = 0;
    for (const auto& ch : children) {
        if (ch) {
            res = std::max(ch->get_max_depth() + 1, res);
        }
    }
    return res;
}

glm::vec3 MaterialTree::Node::compute_barycentric(
    const glm::vec2& point) const {
    auto v2 = point - triangle.A;
    float d20 = glm::dot(v2, bary_cache.v0);
    float d21 = glm::dot(v2, bary_cache.v1);
    float v =
        (bary_cache.d11 * d20 - bary_cache.d01 * d21) * bary_cache.invDenom;
    float w =
        (bary_cache.d00 * d21 - bary_cache.d01 * d20) * bary_cache.invDenom;
    float u = 1.0f - v - w;
    return {u, v, w};
}

float MaterialTree::Node::compute_reflectance(const glm::vec2& point,
                                              const glm::vec3& in,
                                              const glm::vec3& out,
                                              Wavelength wavelength,
                                              int depth) {
    if (depth <= 0) {
        auto A_refl = points[A_id].get_reflectance(in, out, wavelength);
        auto B_refl = points[B_id].get_reflectance(in, out, wavelength);
        auto C_refl = points[C_id].get_reflectance(in, out, wavelength);
        auto coords = compute_barycentric(point);
        return A_refl*coords.x + B_refl*coords.y + C_refl*coords.z;
    }
    int pt_size = depth;
    int thickness = depth;

    // TODO: rewrite to determinant -> no 3rd dimension
    glm::vec3 v = {MB.x - MC.x, MB.y - MC.y, 0};
    glm::vec3 tv = {point.x - MC.x, point.y - MC.y, 0};
    glm::vec3 pv = {triangle.A.x - MC.x, triangle.A.y - MC.y, 0};

    if ((glm::cross(tv, v).z * glm::cross(pv, v).z) > 0) {
        return children[0]->compute_reflectance(point, in, out, wavelength,
                                                depth - 1);
    } else {
        v = {MA.x - MC.x, MA.y - MC.y, 0};
        tv = {point.x - MC.x, point.y - MC.y, 0};
        pv = {triangle.B.x - MC.x, triangle.B.y - MC.y, 0};

        if ((glm::cross(tv, v).z * glm::cross(pv, v).z) > 0) {
            return children[1]->compute_reflectance(point, in, out, wavelength,
                                                    depth - 1);
        } else {
            v = {MA.x - MB.x, MA.y - MB.y, 0};
            tv = {point.x - MB.x, point.y - MB.y, 0};
            pv = {triangle.C.x - MB.x, triangle.C.y - MB.y, 0};
            if ((glm::cross(tv, v).z * glm::cross(pv, v).z) > 0) {
                return children[2]->compute_reflectance(point, in, out,
                                                        wavelength, depth - 1);

            } else {
                return children[3]->compute_reflectance(point, in, out,
                                                        wavelength, depth - 1);
            }
        }
    }
}

MaterialTree::MaterialTree(const std::vector<MeasurementHit>& measurements,
                           int depth)
    : depth(depth) {}
int MaterialTree::get_max_depth() { return depth; }

}  // namespace chameleon
