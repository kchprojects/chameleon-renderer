#pragma once

#include <chameleon_renderer/utils/math_utils.hpp>
#include <opencv2/opencv.hpp>

namespace chameleon {
namespace equivalent {
template<typename T>
struct cv2glm;

template<typename T>
using cv2glm_v = typename cv2glm<T>::value;

template<typename T>
struct glm2cv;

template<typename T>
using glm2cv_v = typename glm2cv<T>::value;

template<>
struct cv2glm<cv::Vec3f>
{
    using value = glm::vec3;
};

template<>
struct cv2glm<float>
{
    using value = float;
};

template<>
struct cv2glm<cv::Vec3b>
{
    using value = glm::vec<3,std::uint8_t>;
};
template<>
struct cv2glm<uchar>
{
    using value = std::uint8_t;
};

template<>
struct glm2cv<glm::vec3>
{
    using value = cv::Vec3f;
};

template<>
struct glm2cv<float>
{
    using value = float;
};

template<>
struct glm2cv<glm::vec<3,std::uint8_t>>
{
    using value = cv::Vec3b;
};
template<>
struct glm2cv<std::uint8_t>
{
    using value = uchar;
};

} // namespace equivalent
} // namespace chameleon