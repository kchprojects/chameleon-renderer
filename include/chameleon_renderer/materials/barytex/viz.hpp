#pragma once
#include <chameleon_renderer/utils/math_utils.hpp>
#include <opencv2/opencv.hpp>

namespace chameleon {

void draw_triangle(cv::Mat& canvas, const glm::vec2& A, const glm::vec2& B,
                   const glm::vec2& C, int pt_size, int thickness,
                   int viz_edge_size = 800, int viz_spacing = 100);

void draw_point(cv::Mat& canvas, const glm::vec2& point,int pt_size,int thickness,int viz_edge_size = 800, int viz_spacing = 100);
}  // namespace chameleon