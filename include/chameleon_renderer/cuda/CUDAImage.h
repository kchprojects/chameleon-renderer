#pragma once
#include <assert.h>

template <typename T>
struct CUDAImage {
    T* data;
    unsigned int rows;
    unsigned int cols;

#ifdef __CUDA_ARCH__
    __device__ T at(unsigned i) {
        assert(i < rows * cols);
        return data[i];
    }
    __device__ T at(unsigned x, unsigned y) { return at(x * cols + y); }
#endif

};