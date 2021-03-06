#include <chameleon_renderer/materials/barytex/viz.hpp>

namespace chameleon {

void draw_triangle(cv::Mat& canvas, const glm::vec2& A, const glm::vec2& B,
                   const glm::vec2& C, int pt_size, int thickness,
                   int viz_edge_size, int viz_spacing) {
    cv::Point a = {int(A.x * viz_edge_size + viz_spacing),
                   int((1 - A.y) * viz_edge_size)};
    cv::Point b = {int(B.x * viz_edge_size + viz_spacing),
                   int((1 - B.y) * viz_edge_size)};
    cv::Point c = {int(C.x * viz_edge_size + viz_spacing),
                   int((1 - C.y) * viz_edge_size)};

    cv::line(canvas, a, b, {255, 0, 0}, thickness);
    cv::line(canvas, b, c, {255, 0, 0}, thickness);
    cv::line(canvas, c, a, {255, 0, 0}, thickness);

    cv::drawMarker(canvas, a, {0, 0, 255}, cv::MARKER_CROSS, pt_size,
                   thickness);
    cv::drawMarker(canvas, b, {0, 0, 255}, cv::MARKER_CROSS, pt_size,
                   thickness);
    cv::drawMarker(canvas, c, {0, 0, 255}, cv::MARKER_CROSS, pt_size,
                   thickness);
}

void draw_point(cv::Mat& canvas, const glm::vec2& point, int pt_size,
                int thickness, int viz_edge_size, int viz_spacing ) {
    cv::Point tp = {int(point.x * viz_edge_size + viz_spacing),
                    int((1 - point.y) * viz_edge_size)};
    cv::drawMarker(canvas, tp, {0, 255, 0}, cv::MARKER_CROSS, pt_size, thickness);
}

}  // namespace chameleon