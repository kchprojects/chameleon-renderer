#pragma once
#include <array>
#include <chameleon_renderer/materials/barytex/MeasurementHit.hpp>
#include <memory>
#include <chameleon_renderer/materials/barytex/viz.hpp>
// for visualization
#include <opencv2/opencv.hpp>

namespace chameleon {
static const glm::vec2 base_A = {0.f, 0.f};
static const glm::vec2 base_B = {1.f, 0.f};
static const glm::vec2 base_C = {0.5f, std::sqrt(0.75f)};


//TODO: adjust for general MaterialData from Measurement copy
struct MaterialTree {
    struct Node {
        Node* parent = nullptr;
        glm::vec2 A = base_A;
        glm::vec2 B = base_B;
        glm::vec2 C = base_C;
        std::array<std::unique_ptr<Node>, 4> children;
        std::vector<std::pair<glm::vec2, MeasurementHit>> values;

        Node() = default;
        Node(Node* parent, const glm::vec2& A, const glm::vec2& B,
             const glm::vec2& C)
            : parent(parent), A(A), B(B), C(C) {}
        int get_max_depth(){
            int res = 0;
            for(const auto& ch : children){
                if(ch){
                    res = std::max(ch->get_max_depth()+1,res);
                }
            }
            return res;
        }

        void emplace_hit(const glm::vec2& point, const MeasurementHit& hit,
                         int depth) {
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
                            children[2] =
                                std::make_unique<Node>(this, MB, MA, C);
                        }
                        children[2]->emplace_hit(point, hit, depth - 1);

                    } else {
                        if (!children[3]) {
                            children[3] =
                                std::make_unique<Node>(this, MB, MA, MC);
                        }
                        children[3]->emplace_hit(point, hit, depth - 1);
                    }
                }
            }
        }
        void draw_subtree(cv::Mat& canvas, int thickness, int pt_size) const {
            thickness = std::max(thickness, 1);
            pt_size = std::max(pt_size, 1);
            draw_triangle(canvas, A, B, C, pt_size, thickness);
            for (const auto& [point, _] : values) {
                cv::Point tp = {int(point.x * viz_edge_size + viz_spacing),
                                int((1 - point.y) * viz_edge_size)};
                cv::drawMarker(canvas, tp, {0, 255, 0}, cv::MARKER_CROSS, 5, 2);
            }
            for (const auto& ch : children) {
                if (ch) {
                    ch->draw_subtree(canvas, thickness - 1, pt_size - 1);
                }
            }
        }
    };

    MaterialTree(const std::vector<MeasurementHit>& measurements,
                    int depth = 2)
        : root(std::make_unique<Node>()) {
        glm::vec2 A = base_A;
        glm::vec2 B = base_B;
        glm::vec2 C = base_C;
        for (const auto& m : measurements) {
            glm::vec2 point =
                A * m.coordinates.x + B * m.coordinates.y + C * m.coordinates.z;
            root->emplace_hit(point, m, depth);
        }

        cv::Mat viz_canvas =
            cv::Mat::zeros(viz_spacing * 2 + viz_edge_size,
                           viz_spacing * 2 + viz_edge_size, CV_8UC3);
        root->draw_subtree(viz_canvas, depth, depth);
        cv::imshow("img", viz_canvas);
        cv::waitKey();
    }
    int get_max_depth() { return root ? root->get_max_depth() : -1; }

    std::unique_ptr<Node> root;
};





}  // namespace chameleon
