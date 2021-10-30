#pragma once
#include <chameleon_renderer/cuda/CUDABuffer.h>

#include <chameleon_renderer/utils/cv2gdt.hpp>
#include <opencv2/opencv.hpp>

namespace chameleon {
template<typename IMG_T>
struct ImageLayer
{
    using buffer_t = equivalent::cv2gdt_v<IMG_T>;
    size_t res_x = 0;
    size_t res_y = 0;
    CUDABuffer cuda_buffer;
    std::vector<buffer_t> download_buffer;

    void resize(size_t x, size_t y)
    {
        if (res_x == x and res_y == y)
            return;
        res_x = x;
        res_y = y;
        cuda_buffer.resize(x * y * sizeof(buffer_t));
        download_buffer.resize(x * y);
    }
    auto buffer_ptr() { return cuda_buffer.d_ptr; }

    void clear() { cuda_buffer.clear(); }

    cv::Mat_<IMG_T> get_cv_mat()
    {
        cuda_buffer.download((buffer_t*)download_buffer.data(), res_x * res_y);

        cv::Mat_<IMG_T> out(
            int(res_y), int(res_x), (IMG_T*)download_buffer.data());
        return out.clone();
    }
};
} // namespace chameleon
