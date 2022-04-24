#pragma once
#include <chameleon_renderer/materials/barytex/MaterialDataType.hpp>
#include <chameleon_renderer/materials/barytex/CUDAGenericMaterial.hpp>

namespace chameleon
{

    template <MaterialModel model_t>
    struct GenericMaterial;

    struct IGenericMaterial{
        MaterialModel m;

        CUDAGenericMaterial upload_to_cuda(){
            //TODO:
            return {nullptr,0,m};
        }
    };

    template <MaterialModel model_t>
    struct GenericMaterial : public IGenericMaterial{
        using data_t = ModelDataType<model_t>;
        data_t data;
    };
} // namespace chameleon
