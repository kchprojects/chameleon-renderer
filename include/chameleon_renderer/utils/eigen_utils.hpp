#pragma once

#include <chameleon_renderer/utils/math_utils.hpp>
#include <eigen3/Eigen/Eigen>

namespace chameleon {
namespace eigen_utils {
template <typename T>
using Vec2 = Eigen::Matrix<T, 2, 1>;

template <typename T>
using Vec3 = Eigen::Matrix<T, 3, 1>;

template <typename T>
using Vec4 = Eigen::Matrix<T, 4, 1>;

template <typename T>
using Mat4 = Eigen::Matrix<T, 4, 4>;

template <typename T>
using Mat3 = Eigen::Matrix<T, 3, 3>;

template <typename T>
using Mat = Eigen::Matrix<T, -1, -1>;

inline Vec3<float> to_eigen(const glm::vec3& v) {
    Vec3<float> out;
    out << v.x, v.y, v.z;
    return out;
}
inline Vec4<float> to_eigen_homogenus(const glm::vec3& v, bool is_point) {
    Vec4<float> out;
    if (is_point) {
        out << v.x, v.y, v.z, 1;
    } else {
        out << v.x, v.y, v.z, 0;
    }
    return out;
}
inline glm::vec3 from_eigen_homogenus(const Vec4<float>& v_eigen) {
    return {v_eigen(0), v_eigen(1), v_eigen(2)};
}

inline glm::vec3 from_eigen_v3(const Vec3<float>& v_eigen) {
    return {v_eigen(0), v_eigen(1), v_eigen(2)};
}
inline glm::vec4 from_eigen_v4(const Vec4<float>& v_eigen) {
    return {v_eigen(0), v_eigen(1), v_eigen(2), v_eigen(3)};
}
inline glm::vec3 from_eigen_v4_v3(const Vec4<float>& v_eigen) {
    return {v_eigen(0), v_eigen(1), v_eigen(2)};
}

inline glm::mat3 from_eigen_m3(const Mat3<float>& m_eigen) {
    return glm::transpose(glm::mat3(m_eigen(0, 0), m_eigen(0, 1), m_eigen(0, 2),
            m_eigen(1, 0), m_eigen(1, 1), m_eigen(1, 2),
            m_eigen(2, 0), m_eigen(2, 1), m_eigen(2, 2)));
}
inline glm::mat4 from_eigen_m4(const Mat4<float>& m_eigen) {
    return glm::transpose(glm::mat4(m_eigen(0,0), m_eigen(0,1), m_eigen(0,2), m_eigen(0,3),
            m_eigen(1,0), m_eigen(1,1), m_eigen(1,2), m_eigen(1,3),
            m_eigen(2,0), m_eigen(2,1), m_eigen(2,2), m_eigen(2,3),
            m_eigen(3,0), m_eigen(3,1), m_eigen(3,2), m_eigen(3,3)));
}

inline Vec3<float> homogenus_transform(const Mat4<float>& mat,
                                       const Vec3<float>& vec, bool is_point) {
    Vec4<float> homog;
    if (is_point) {
        homog << vec(0), vec(1), vec(2), 1;
    } else {
        homog << vec(0), vec(1), vec(2), 0;
    }
    auto transf = mat * homog;
    Vec3<float> v;
    v << transf(0), transf(1), transf(2);
    return v;
}

template <typename T, size_t S>
inline Eigen::Matrix<T, S + 1, S + 1> to_homogenus(
    const Eigen::Matrix<T, S, S>& mat) {
    Eigen::Matrix<T, S + 1, S + 1> out =
        Eigen::Matrix<T, S + 1, S + 1>::Identity();
    out.block(0, 0, S, S) = mat;
    return out;
}

template <typename T, size_t S>
inline Eigen::Matrix<T, S + 1, 1> to_homogenus(
    const Eigen::Matrix<T, S, 1>& vec, T w = 0) {
    Eigen::Matrix<T, S + 1, 1> out = Eigen::Matrix<T, S + 1, 1>::Identity();
    out(S, 0) = w;
    return out;
}

inline Eigen::Matrix3f euler_to_mat(float rx, float ry, float rz) {
    Eigen::AngleAxisf pitchAngle(rx, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf yawAngle(ry, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf rollAngle(rz, Eigen::Vector3f::UnitZ());
    Eigen::Quaternion<float> q = pitchAngle * yawAngle * rollAngle;
    return q.matrix();
}

inline Eigen::Matrix3f scale_mat(float sx, float sy, float sz) {
    Eigen::Matrix3f out = Eigen::Matrix3f::Identity();
    out(0, 0) = sx;
    out(1, 1) = sy;
    out(2, 2) = sz;
    return out;
}
inline Eigen::Matrix4f translation_mat(float tx, float ty, float tz) {
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m.block<3, 1>(0, 3) = Vec3<float>(tx, ty, tz);
    return m;
}

inline Eigen::Matrix4f translation_mat(const Vec3<float>& t) {
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m.block<3, 1>(0, 3) = t;
    return m;
}

template <typename T, int C, int R>
class InvertableMatrix {
   public:
    using mat_t = Eigen::Matrix<T, C, R>;
    InvertableMatrix() = default;

    InvertableMatrix(Eigen::Matrix<T, C, R> data_mat)
        : _mat(std::move(data_mat)), _inv(_mat.inverse()) {}

    void set(mat_t m) {
        _mat = std::move(m);
        _inv = _mat.inverse();
    }
    const mat_t& mat() const { return _mat; }

    const mat_t& inv() const { return _inv; }

   private:
    mat_t _mat;
    mat_t _inv;
};

}  // namespace eigen_utils
}  // namespace chameleon
