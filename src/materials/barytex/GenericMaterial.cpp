#include <chameleon_renderer/cuda/CUDABuffer.hpp>
#include <chameleon_renderer/materials/barytex/GenericMaterial.hpp>

namespace chameleon {
CUDAGenericMaterial IGenericMaterial::upload_to_cuda() const {
    static const size_t channel_count = 3;
    CUDABuffer buff;
    size_t param_count;
    if (m == MaterialModel::Lambert) {
        param_count = 1;
        buff.alloc(sizeof(double) * param_count * channel_count);
        auto real_mat =
            dynamic_cast<const GenericMaterial<MaterialModel::Lambert>*>(this);
        buff.upload(reinterpret_cast<const double*>(real_mat->data_rgb.data()),
                                              param_count * channel_count);
    } else if (m == MaterialModel::BlinPhong) {
        param_count = 3;
        buff.alloc(sizeof(double) * param_count * channel_count);
        auto real_mat =
            dynamic_cast<const GenericMaterial<MaterialModel::BlinPhong>*>(this);
        buff.upload(reinterpret_cast<const double*>(real_mat->data_rgb.data()),
                                              param_count * channel_count);
        buff.alloc(sizeof(double) * param_count * channel_count);
    } else if (m == MaterialModel::CookTorrance) {
        param_count = 4;
        buff.alloc(sizeof(double) * param_count * channel_count);
        auto real_mat =
            dynamic_cast<const GenericMaterial<MaterialModel::CookTorrance>*>(this);
        buff.upload(reinterpret_cast<const double*>(real_mat->data_rgb.data()),
                                              param_count * channel_count);

    } else {
        return {nullptr, 0, m};
    }
    return {reinterpret_cast<double*>(buff.d_pointer()), param_count, m};
}
}  // namespace chameleon
