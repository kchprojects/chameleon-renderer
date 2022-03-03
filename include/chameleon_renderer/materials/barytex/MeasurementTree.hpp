#pragma once
#include <array>
#include <chameleon_renderer/materials/barytex/MeasurementHit.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <glm/glm.hpp>
#include <memory>

namespace chameleon {

struct MeasurementTree {
    struct Node {
        Node* parent = nullptr;
        glm::vec2 A = {0.f, 0.f};
        glm::vec2 B = {1.f, 0.f};
        glm::vec2 C = {0.5f, std::sqrt(0.75f)};

        std::array<std::unique_ptr<Node>, 4> children;
        std::vector<std::pair<glm::vec2, MeasurementHit>> values;

        Node() = default;
        Node(Node* parent, const glm::vec2& A, const glm::vec2& B,
             const glm::vec2& C);

        int get_max_depth();

        void emplace_hit(const glm::vec2& point, const MeasurementHit& hit,
                         int depth);
        void draw_subtree(cv::Mat& canvas, int thickness, int pt_size) const;
    };

    MeasurementTree(const std::vector<MeasurementHit>& measurements,
                    int depth = 2);
    int get_max_depth();

    std::unique_ptr<Node> root;
};

}  // namespace chameleon
