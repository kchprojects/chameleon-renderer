#pragma once
#include <chameleon_renderer/materials/barytex/MaterialModel.hpp>

namespace chameleon {
struct CUDAGenericMaterial {
    double* data;
    size_t param_count;
    size_t channel_count;
    MaterialModel model;
};

#ifdef __CUDA_ARCH__
inline __device__ bool get_lambert_material_data(
    const CUDAGenericMaterial& generic_material, int color_index,
    ModelData<MaterialModel::Lambert, double>& out) {
    if (generic_material.model != MaterialModel::Lambert ||
        generic_material.param_count != 1 || color_index > 2) {
        return false;
    }
    const double* data = (const double*)generic_material.data;
    out.albedo = data[generic_material.param_count*color_index];
    return true;
}

inline __device__ bool get_blinphong_material_data(
    const CUDAGenericMaterial& generic_material, int color_index,
    ModelData<MaterialModel::BlinPhong, double>& out) {
    if (generic_material.model != MaterialModel::BlinPhong ||
        generic_material.param_count!= 3 || color_index > 2) {
        return false;
    }
    const double* data = (const double*)generic_material.data;
    out.albedo = data[3*color_index+0];
    out.Ks = data[3*color_index+1];
    out.alpha = data[generic_material.param_count*color_index+2];
    return true;
}

inline __device__ bool get_cook_torrance_material_data(
    const CUDAGenericMaterial& generic_material, int color_index,
    ModelData<MaterialModel::CookTorrance, double>& out) {
    if (generic_material.model != MaterialModel::CookTorrance ||
        generic_material.param_count!= 4 || color_index > 2) {
        return false;
    }
    const double* data = (const double*)generic_material.data;
    out.albedo = data[generic_material.param_count*color_index+0];
    out.Ks = data[3*color_index+1];
    out.F0 = data[3*color_index+2];
    out.m = data[3*color_index+3];
    return true;
}

#endif
}  // namespace chameleon
