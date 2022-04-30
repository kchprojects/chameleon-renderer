#pragma once
#include <ceres/ceres.h>

#include <chameleon_renderer/materials/barytex/BRDFMeasurement.hpp>
#include <chameleon_renderer/materials/barytex/MaterialModel.hpp>
#include <glm/glm.hpp>

namespace chameleon {
enum class MaterialModel { Lambert = 0, BlinPhong, CookTorrance };

template <MaterialModel m_t, typename T>
struct ModelData;

template <typename T>
struct ModelData<MaterialModel::Lambert, T> {
    T albedo = 0.5;
};

template <typename T>
struct ModelData<MaterialModel::BlinPhong, T> {
    T albedo = 0.5;
    T Ks = 0.5;
    T alpha = 0.5;
};

template <typename T>
struct ModelData<MaterialModel::CookTorrance, T> {
    T albedo = 0.5;
    T Ks = 0.5;
    T F0 = 0.5;
    T m = 0.5;
};

template <MaterialModel m_t, typename T>
struct ModelReflectance;

template <typename T>
struct ModelReflectance<MaterialModel::Lambert, T> {
    static T compute(const glm::vec3& normal, const glm::vec3& light,
                     const glm::vec3& view,
                     const ModelData<MaterialModel::Lambert, T>& data) {
        return T(glm::dot(normal, light)) * data.albedo;
    }
};

template <typename T>
struct ModelReflectance<MaterialModel::BlinPhong, T> {
    static T compute(const glm::vec3& normal, const glm::vec3& light,
                     const glm::vec3& view,
                     const ModelData<MaterialModel::BlinPhong, T>& data) {
        glm::vec3 H = glm::normalize(light + view);
        auto norm_dot = glm::dot(H, normal);
        auto light_dot = glm::dot(normal, light);
        T out = T(light_dot) * data.albedo +
                data.Ks * T(pow(T(norm_dot), data.alpha));
        // if constexpr(std::is_same_v<T,double>){
        // std::cout << out << std::endl;
        // }
        return out;
    }
};

template <typename T>
struct ModelReflectance<MaterialModel::CookTorrance, T> {
    static T D(const T& m, const T& alpha) {
        T out = exp(-pow(tan(alpha) / m, T(2.0))) /
                (T(M_PI) * m * m * pow(cos(alpha), 4));
        // std::cout << "D: " << out << std::endl;
        return out;
    }

    static T F(T F0, T NL) {
        T out = F0 + (T(1) - F0) * (T(1) - NL);
        // std::cout << "F: " << out << std::endl;
        return out;
    }
    static T min(T a, T b) {
        if (a > b) {
            return b;
        }
        return a;
    }
    static T G(T NH, T VH, T NV, T NL) {
        T out = min(T(1), min(T(2) * NH * NV / VH, T(2) * NH * NL / VH));
        // std::cout << "G: " << out << std::endl;
        return out;
    }

    static T compute(const glm::vec3& N, const glm::vec3& L, const glm::vec3& V,
                     const ModelData<MaterialModel::CookTorrance, T>& data) {
        glm::vec3 H = glm::normalize(L + V);
        T NH = T(glm::dot(N, H));
        T VH = T(glm::dot(V, H));
        T NV = T(glm::dot(N, V));
        T NL = T(glm::dot(N, L));

        T alpha = acos(NH);

        T out = NL * data.albedo + data.Ks * D(data.m, alpha) *
                                       G(NH, VH, NV, NL) * F(data.F0, NL) /
                                       (T(4) * NL);
        // std::cout << "data: " << data.albedo << "," << data.Ks << "," << data.F0
        //           << "," << data.m << std::endl;
        // std::cout << "NS: " << NH << "," << VH << "," << NV << "," << NL
        //           << std::endl;
        // std::cout << "out: " << out << std::endl;
        return out;
    }
};

}  // namespace chameleon
