#pragma once

#include <chameleon_renderer/utils/math_utils.hpp>
#include <opencv2/opencv.hpp>

namespace chameleon {
namespace equivalent {
template<typename T>
struct cv2gdt;

template<typename T>
using cv2gdt_v = typename cv2gdt<T>::value;

template<typename T>
struct gdt2cv;

template<typename T>
using gdt2cv_v = typename gdt2cv<T>::value;

template<>
struct cv2gdt<cv::Vec3f>
{
    using value = vec3f;
};

template<>
struct cv2gdt<float>
{
    using value = float;
};

template<>
struct cv2gdt<cv::Vec3b>
{
    using value = vec3uc;
};
template<>
struct cv2gdt<uchar>
{
    using value = std::uint8_t;
};

template<>
struct gdt2cv<vec3f>
{
    using value = cv::Vec3f;
};

template<>
struct gdt2cv<float>
{
    using value = float;
};

template<>
struct gdt2cv<vec3uc>
{
    using value = cv::Vec3b;
};
template<>
struct gdt2cv<std::uint8_t>
{
    using value = uchar;
};

} // namespace equivalent
} // namespace chameleon