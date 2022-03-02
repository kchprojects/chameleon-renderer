#include <chameleon_renderer/cuda/CUDABuffer.hpp>

namespace chameleon {

inline CUdeviceptr CUDABuffer::d_pointer() const { return (CUdeviceptr)d_ptr; }

void CUDABuffer::resize(size_t size) {
    if (sizeInBytes == size) return;

    if (d_ptr) free();
    alloc(size);
}

void CUDABuffer::alloc(size_t size) {
    assert(d_ptr == nullptr);
    this->sizeInBytes = size;
    CUDA_CHECK(cudaMalloc((void**)&d_ptr, sizeInBytes));
}

void CUDABuffer::clear() {
    assert(d_ptr != nullptr);
    CUDA_CHECK(cudaMemset(d_ptr, 0, this->sizeInBytes));
    // CUDA_CHECK(cudaMalloc((void**)&d_ptr, sizeInBytes));
}

void CUDABuffer::free() {
    CUDA_CHECK(cudaFree(d_ptr));
    d_ptr = nullptr;
    sizeInBytes = 0;
}

CUDABuffer::~CUDABuffer() { free(); }

}  // namespace chameleon
