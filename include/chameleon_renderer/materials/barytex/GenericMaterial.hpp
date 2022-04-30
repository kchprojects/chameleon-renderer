#pragma once
#include <chameleon_renderer/materials/barytex/CUDAMaterial.hpp>
#include <chameleon_renderer/materials/barytex/MaterialModel.hpp>
#include <nlohmann/json.hpp>

namespace chameleon {

template <MaterialModel model_t>
struct GenericMaterial;

struct IGenericMaterial {
    MaterialModel m = MaterialModel::Lambert;

    virtual nlohmann::json to_json() const = 0;

    CUDAGenericMaterial upload_to_cuda() const;
};

template <MaterialModel model_t>
struct GenericMaterial : public IGenericMaterial {
    using data_t = ModelData<model_t, double>;
    std::array<data_t, 3> data_rgb;

    GenericMaterial() = default;
    GenericMaterial(std::array<data_t, 3> data_rgb)
        : data_rgb(std::move(data_rgb)) {
        m = model_t;
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

template <MaterialModel model_t>
inline std::unique_ptr<GenericMaterial<model_t>> material_from_json(
    const nlohmann::json& j) {
    auto out = std::make_unique<GenericMaterial<model_t>>();
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
                }
            }
        }
    }
    return out;
}

inline std::unique_ptr<IGenericMaterial> generic_material_from_json(
    const nlohmann::json& j) {
    std::unique_ptr<IGenericMaterial> mat;
    switch (MaterialModel(j["material_model"].get<int>())) {
        case MaterialModel::Lambert:
            mat = material_from_json<MaterialModel::Lambert>(j);
            break;
        case MaterialModel::BlinPhong:
            mat = material_from_json<MaterialModel::BlinPhong>(j);
            break;
        case MaterialModel::CookTorrance:
            mat = material_from_json<MaterialModel::CookTorrance>(j);
            break;
        default:
            break;
    }
    return mat;
}
}  // namespace chameleon
