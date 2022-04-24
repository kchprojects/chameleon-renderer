#pragma once 


namespace chameleon
{
    struct CUDAGenericMaterial{
        void* data;
        size_t size;
        MaterialModel model;
    };
} // namespace chameleon
