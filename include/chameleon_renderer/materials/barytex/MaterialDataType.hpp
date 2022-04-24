#pragma once

namespace chameleon
{
    enum class MaterialModel{
        Lambert = 0//,
        // Phong,
        // NN
    };

    template<MaterialModel m_t>
    struct ModelDataType;

    template<>
    struct ModelDataType<MaterialModel::Lambert>{
        using data_t = float;
    };
} // namespace chameleon
