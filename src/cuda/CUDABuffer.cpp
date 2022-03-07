#include <chameleon_renderer/cuda/CUDABuffer.hpp>
#include <chameleon_renderer/utils/terminal_utils.hpp>

namespace chameleon {

CUdeviceptr CUDABuffer::d_pointer() const { return reinterpret_cast<CUdeviceptr>(d_ptr); }

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
    if(d_ptr != nullptr){
        return;
    }
    CUDA_CHECK(cudaMemset(d_ptr, 0, this->sizeInBytes));
    // CUDA_CHECK(cudaMalloc((void**)&d_ptr, sizeInBytes));
}

void CUDABuffer::free() {
    if(d_ptr == nullptr || sizeInBytes == 0){
        return;
    }
    try{
        CUDA_CHECK(cudaFree(d_ptr));
    }catch(const std::exception& e){
        spdlog::warn("Cannot free data from address {} of size {} with error: {}",d_ptr,sizeInBytes,e.what());
    }
    d_ptr = nullptr;
    sizeInBytes = 0;
}

CUDABuffer::~CUDABuffer() { 
    //TODO: fix memmory leak vs invalid access
    //free(); 
    }

}  // namespace chameleon
