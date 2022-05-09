#include <chameleon_renderer/cuda/CUDABuffer.hpp>
#include <chameleon_renderer/materials/barytex/GenericMaterial.hpp>
#include <chameleon_renderer/utils/terminal_utils.hpp>
#include <iostream>

namespace chameleon {
CUDAGenericMaterial IGenericMaterial::upload_to_cuda() const {
    CUDABuffer buff;
    size_t param_count;
    // PING;
    if (m == MaterialModel::Lambert) {
        // PING;

        param_count = 1;
        buff.alloc(sizeof(double) * param_count * channel_count);
        if (channel_count == 3) {
            auto real_mat =
                dynamic_cast<const GenericMaterial<MaterialModel::Lambert>*>(
                    this);
            buff.upload(
                reinterpret_cast<const double*>(real_mat->data_rgb.data()),
                param_count * channel_count);
        } else if (channel_count == 1) {
            auto real_mat =
                dynamic_cast<const GenericMaterial<MaterialModel::Lambert, 1>*>(
                    this);
            buff.upload(
                reinterpret_cast<const double*>(real_mat->data_rgb.data()),
                param_count * channel_count);
        } else {
            return {nullptr, 0, 0, m};
        }
    } else if (m == MaterialModel::BlinPhong) {
        // PING;
        param_count = 3;
        buff.alloc(sizeof(double) * param_count * channel_count);
        // PING;
        if (channel_count == 3) {
            auto real_mat =
                dynamic_cast<const GenericMaterial<MaterialModel::BlinPhong>*>(
                    this);
            buff.upload(
                reinterpret_cast<const double*>(real_mat->data_rgb.data()),
                param_count * channel_count);
        } else if (channel_count == 1) {
            auto real_mat =
                dynamic_cast<const GenericMaterial<MaterialModel::BlinPhong>*>(
                    this);
            buff.upload(
                reinterpret_cast<const double*>(real_mat->data_rgb.data()),
                param_count * channel_count);
        } else {
            return {nullptr, 0, 0, m};
        }
    } else if (m == MaterialModel::CookTorrance) {
        // PING;

        param_count = 4;
        buff.alloc(sizeof(double) * param_count * channel_count);
        if (channel_count == 3) {
            auto real_mat = dynamic_cast<
                const GenericMaterial<MaterialModel::CookTorrance>*>(this);
            buff.upload(
                reinterpret_cast<const double*>(real_mat->data_rgb.data()),
                param_count * channel_count);
        } else if (channel_count == 1) {
            auto real_mat = dynamic_cast<
                const GenericMaterial<MaterialModel::CookTorrance, 1>*>(this);
            buff.upload(
                reinterpret_cast<const double*>(real_mat->data_rgb.data()),
                param_count * channel_count);
        } else {
            return {nullptr, 0, 0, m};
        }

    } else {
        // throw std::runtime_error("invalid material");
        return {nullptr, 0, 0, m};
    }
    return {reinterpret_cast<double*>(buff.d_pointer()), param_count,
            channel_count, m};
}
}  // namespace chameleon
