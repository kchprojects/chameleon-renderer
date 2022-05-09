#pragma once
#include <chameleon_renderer/materials/barytex/CUDAMaterial.hpp>
#include <chameleon_renderer/materials/barytex/MaterialModel.hpp>
#include <nlohmann/json.hpp>

namespace chameleon {

template <MaterialModel model_t,size_t CH>
struct GenericMaterial;

struct IGenericMaterial {
    MaterialModel m = MaterialModel::NoModel;
    size_t channel_count = 0;

    virtual nlohmann::json to_json() const = 0;

    CUDAGenericMaterial upload_to_cuda() const;
};

template <MaterialModel model_t,size_t CH = 3>
struct GenericMaterial : public IGenericMaterial {
    using data_t = ModelData<model_t, double>;
    using IGenericMaterial::m;
    std::array<data_t, CH> data_rgb;

    GenericMaterial() { m = model_t;channel_count=CH; }
    GenericMaterial(std::array<data_t, CH> data_rgb)
        : data_rgb(std::move(data_rgb)) {
        m = model_t;
        channel_count = CH;
    }

    nlohmann::json to_json() const override {
        nlohmann::json out;
        out["material_model"] = int(model_t);
        out["material_data"] = nlohmann::json::array();
        for (auto i = 0u; i < data_rgb.size(); ++i) {
            nlohmann::json mj;
            if constexpr (model_t == MaterialModel::Lambert) {
                mj["albedo"] = data_rgb[i].albedo;
            }
            if constexpr (model_t == MaterialModel::BlinPhong) {
                mj["albedo"] = data_rgb[i].albedo;
                mj["Ks"] = data_rgb[i].Ks;
                mj["alpha"] = data_rgb[i].alpha;
            }

            if constexpr (model_t == MaterialModel::CookTorrance) {
                mj["albedo"] = data_rgb[i].albedo;
                mj["Ks"] = data_rgb[i].Ks;
                mj["F0"] = data_rgb[i].F0;
                mj["m"] = data_rgb[i].m;
            }
            out["material_data"].push_back(mj);
        }
        return out;
    }
};

template <MaterialModel model_t,size_t channel_count>
inline std::unique_ptr<GenericMaterial<model_t,channel_count>> material_from_json(
    const nlohmann::json& j) {
    auto out = std::make_unique<GenericMaterial<model_t,channel_count>>();
    for (auto i = 0u; i < out->data_rgb.size(); ++i) {
        const nlohmann::json& mj = j.at("material_data")[i];
        if constexpr (model_t == MaterialModel::Lambert) {
            out->data_rgb[i].albedo = mj.at("albedo").get<double>();
        } else {
            if constexpr (model_t == MaterialModel::BlinPhong) {
                out->data_rgb[i].albedo = mj.at("albedo").get<double>();
                out->data_rgb[i].Ks = mj.at("Ks").get<double>();
                out->data_rgb[i].alpha = mj.at("alpha").get<double>();
            } else {
                if constexpr (model_t == MaterialModel::CookTorrance) {
                    out->data_rgb[i].albedo = mj.at("albedo").get<double>();
                    out->data_rgb[i].Ks = mj.at("Ks").get<double>();
                    out->data_rgb[i].F0 = mj.at("F0").get<double>();
                    out->data_rgb[i].m = mj.at("m").get<double>();
                } else {
                    throw std::runtime_error("cannot serialize nomaterial");
                }
            }
        }
    }
    return out;
}

inline std::unique_ptr<IGenericMaterial> generic_material_from_json(
    const nlohmann::json& j) {
    std::unique_ptr<IGenericMaterial> mat;
    size_t channel_count = j.at("material_data").size();
    if(channel_count == 1){
        switch (MaterialModel(j["material_model"].get<int>())) {
            case MaterialModel::Lambert:
                mat = material_from_json<MaterialModel::Lambert,1>(j);
                break;
            case MaterialModel::BlinPhong:
                mat = material_from_json<MaterialModel::BlinPhong,1>(j);
                break;
            case MaterialModel::CookTorrance:
                mat = material_from_json<MaterialModel::CookTorrance,1>(j);
                break;
            default:
                break;
        }
    }else if (channel_count == 3){
        switch (MaterialModel(j["material_model"].get<int>())) {
            case MaterialModel::Lambert:
                mat = material_from_json<MaterialModel::Lambert,3>(j);
                break;
            case MaterialModel::BlinPhong:
                mat = material_from_json<MaterialModel::BlinPhong,3>(j);
                break;
            case MaterialModel::CookTorrance:
                mat = material_from_json<MaterialModel::CookTorrance,3>(j);
                break;
            default:
                break;
        }
    }else{
        throw std::runtime_error("Invalid channel count in general material");
    }
    return mat;
}
}  // namespace chameleon
