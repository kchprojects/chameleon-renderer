
#include <chameleon_renderer/materials/barytex/MeasurementTree.hpp>
#include <chameleon_renderer/materials/barytex/viz.hpp>

namespace chameleon {

MeasurementTree::Node::Node(Node* parent,
                            const glm::vec2& A, const glm::vec2& B,
                            const glm::vec2& C)
    : parent(parent), A(A), B(B), C(C) {}

int MeasurementTree::Node::get_max_depth() {
    int res = 0;
    for (const auto& ch : children) {
        if (ch) {
            res = std::max(ch->get_max_depth() + 1, res);
        }
    }
    return res;
}

void MeasurementTree::Node::emplace_hit(const glm::vec2& point,
                                        const MeasurementHit& hit, int depth) {
    if (depth <= 0) {
        values.push_back({point, hit});
        return;
    }
    int pt_size = depth;
    int thickness = depth;
    glm::vec2 MA = (C + B) / 2.f;
    glm::vec2 MB = (A + C) / 2.f;
    glm::vec2 MC = (A + B) / 2.f;

    // TODO: rewrite to determinant -> no 3rd dimension
    glm::vec3 v = {MB.x - MC.x, MB.y - MC.y, 0};
    glm::vec3 tv = {point.x - MC.x, point.y - MC.y, 0};
    glm::vec3 pv = {A.x - MC.x, A.y - MC.y, 0};

    if ((glm::cross(tv, v).z * glm::cross(pv, v).z) > 0) {
        if (!children[0]) {
            children[0] = std::make_unique<Node>(this, A, MB, MC);
        }
        children[0]->emplace_hit(point, hit, depth - 1);
    } else {
        v = {MA.x - MC.x, MA.y - MC.y, 0};
        tv = {point.x - MC.x, point.y - MC.y, 0};
        pv = {B.x - MC.x, B.y - MC.y, 0};

        if ((glm::cross(tv, v).z * glm::cross(pv, v).z) > 0) {
            if (!children[1]) {
                children[1] = std::make_unique<Node>(this, MC, B, MA);
            }
            children[1]->emplace_hit(point, hit, depth - 1);
        } else {
            v = {MA.x - MB.x, MA.y - MB.y, 0};
            tv = {point.x - MB.x, point.y - MB.y, 0};
            pv = {C.x - MB.x, C.y - MB.y, 0};

            if ((glm::cross(tv, v).z * glm::cross(pv, v).z) > 0) {
                if (!children[2]) {
                    children[2] = std::make_unique<Node>(this, MB, MA, C);
                }
                children[2]->emplace_hit(point, hit, depth - 1);

            } else {
                if (!children[3]) {
                    children[3] = std::make_unique<Node>(this, MB, MA, MC);
                }
                children[3]->emplace_hit(point, hit, depth - 1);
            }
        }
    }
}
void MeasurementTree::Node::draw_subtree(cv::Mat& canvas, int thickness,
                                         int pt_size) const {
    thickness = std::max(thickness, 1);
    pt_size = std::max(pt_size, 1);
    draw_triangle(canvas, A, B, C, pt_size, thickness);
    for (const auto& [point, _] : values) {
        draw_point(canvas,point,5,2);
    }
    for (const auto& ch : children) {
        if (ch) {
            ch->draw_subtree(canvas, thickness - 1, pt_size - 1);
        }
    }
}

MeasurementTree::MeasurementTree(
    const std::vector<MeasurementHit>& measurements, int depth)
    : root(std::make_unique<Node>()) {
    for (const auto& m : measurements) {
        glm::vec2 point = root->A * m.coordinates.x +
                          root->B * m.coordinates.y + root->C * m.coordinates.z;
        root->emplace_hit(point, m, depth);
    }

    cv::Mat viz_canvas =
        cv::Mat::zeros(1000,
                       1000, CV_8UC3);
    root->draw_subtree(viz_canvas, depth, depth);
    cv::imshow("img", viz_canvas);
    cv::waitKey();
}
int MeasurementTree::get_max_depth() {
    return root ? root->get_max_depth() : -1;
}

}  // namespace chameleon
