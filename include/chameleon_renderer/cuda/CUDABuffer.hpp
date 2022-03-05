#pragma once
#include <vector>
#include <chameleon_renderer/utils/optix7.h>
#include <assert.h>

namespace chameleon {

/*! simple wrapper for creating, and managing a device-side CUDA
    buffer */
struct CUDABuffer {
    CUdeviceptr d_pointer() const;

    //! re-size buffer to given number of bytes
    void resize(size_t size);

    //! allocate to given number of bytesvector
    void alloc(size_t size);

    void clear();

    //! free allocated memory
    void free();

    template <typename T>
    void alloc_and_upload(const std::vector<T>& vt) {
        alloc(vt.size() * sizeof(T));
        upload((const T*)vt.data(), vt.size());
    }

    template <typename T>
    void upload(const T* t, size_t count) {
        assert(d_ptr != nullptr);
        assert(sizeInBytes == count * sizeof(T));
        CUDA_CHECK(cudaMemcpy(d_ptr, (void*)t, count * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    template <typename T>
    void download(T* t, size_t count) const {
        assert(d_ptr != nullptr);
        assert(sizeInBytes == count * sizeof(T));
        CUDA_CHECK(cudaMemcpy((void*)t, d_ptr, count * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }

    template <typename T>
    void download(std::vector<T>& t) const {
        assert(d_ptr != nullptr);
        unsigned count = sizeInBytes / sizeof(T);
        if (t.size() != count) {
            t.resize(count);
        }

        CUDA_CHECK(cudaMemcpy((void*)t.data(), d_ptr, sizeInBytes,
                              cudaMemcpyDeviceToHost));
    }

    ~CUDABuffer();

    size_t sizeInBytes{0};
    void* d_ptr{nullptr};
};
}  // namespace chameleon
