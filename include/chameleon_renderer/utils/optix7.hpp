#pragma once

// optix 7
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <spdlog/spdlog.h>

#include <sstream>
#include <stdexcept>

#define CUDA_CHECK(call)                                          \
    {                                                             \
        cudaError_t rc = call;                                    \
        if (rc != cudaSuccess) {                                  \
            std::stringstream txt;                                \
            cudaError_t err = rc; /*cudaGetLastError();*/         \
            txt << "CUDA Error " << cudaGetErrorName(err) << " (" \
                << cudaGetErrorString(err) << ")";                \
            spdlog::error(txt.str());                             \
            throw std::runtime_error(txt.str());                  \
        }                                                         \
    }

#define CUDA_CHECK_NOEXCEPT(call) \
    { call; }

#define OPTIX_CHECK(call)                                                    \
    {                                                                        \
        OptixResult res = call;                                              \
        if (res != OPTIX_SUCCESS) {                                          \
            spdlog::error("Optix call ({}) failed with code {} (line {})\n", \
                          #call, res, __LINE__);                             \
            throw std::runtime_error("Optix check fail");                    \
        }                                                                    \
    }

#define CUDA_SYNC_CHECK()                                                      \
    {                                                                          \
        cudaDeviceSynchronize();                                               \
        cudaError_t error = cudaGetLastError();                                \
        if (error != cudaSuccess) {                                            \
            spdlog::error("cuda_sync ({}: line {}): {}\n", __FILE__, __LINE__, \
                          cudaGetErrorString(error));                          \
            throw std::runtime_error("Optix check fail");                      \
        }                                                                      \
    }
