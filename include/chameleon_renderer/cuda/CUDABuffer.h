#pragma once

#include <chameleon_renderer/utils/optix7.h>
// common std stuff
#include <assert.h>

#include <vector>

namespace chameleon {

/*! simple wrapper for creating, and managing a device-side CUDA
    buffer */
struct CUDABuffer
{
    inline CUdeviceptr d_pointer() const { return (CUdeviceptr)d_ptr; }

    //! re-size buffer to given number of bytes
    void resize(size_t size)
    {
        if (sizeInBytes == size)
            return;

        if (d_ptr)
            free();
        alloc(size);
    }

    //! allocate to given number of bytesvector
    void alloc(size_t size)
    {
        assert(d_ptr == nullptr);
        this->sizeInBytes = size;
        CUDA_CHECK(cudaMalloc((void**)&d_ptr, sizeInBytes));
    }
    void clear()
    {
        assert(d_ptr != nullptr);
        CUDA_CHECK(cudaMemset(d_ptr, 0, this->sizeInBytes));
        // CUDA_CHECK(cudaMalloc((void**)&d_ptr, sizeInBytes));
    }
    //! free allocated memory
    void free()
    {
        CUDA_CHECK(cudaFree(d_ptr));
        d_ptr = nullptr;
        sizeInBytes = 0;
    }

    template<typename T>
    void alloc_and_upload(const std::vector<T>& vt)
    {
        alloc(vt.size() * sizeof(T));
        upload((const T*)vt.data(), vt.size());
    }

    template<typename T>
    void upload(const T* t, size_t count)
    {
        assert(d_ptr != nullptr);
        assert(sizeInBytes == count * sizeof(T));
        CUDA_CHECK(cudaMemcpy(
            d_ptr, (void*)t, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    template<typename T>
    void download(T* t, size_t count) const
    {
        assert(d_ptr != nullptr);
        assert(sizeInBytes == count * sizeof(T));
        CUDA_CHECK(cudaMemcpy(
            (void*)t, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    template<typename T>
    void download(std::vector<T>& t) const
    {
        assert(d_ptr != nullptr);
        unsigned count = sizeInBytes/sizeof(T);
        t.resize(count);

        CUDA_CHECK(cudaMemcpy(
            (void*)t.data(), d_ptr, sizeInBytes, cudaMemcpyDeviceToHost));
    }

    size_t sizeInBytes{ 0 };
    void* d_ptr{ nullptr };
};
} // namespace chameleon
