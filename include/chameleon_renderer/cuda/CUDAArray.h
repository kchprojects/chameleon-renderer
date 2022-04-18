#pragma once

namespace chameleon {

template <typename T>
struct CUDAArray {
    T* data = nullptr;
    size_t size = 0;
};

}  // namespace chameleon