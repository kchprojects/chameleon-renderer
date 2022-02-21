#pragma once
#include <chameleon_renderer/utils/eigen_utils.hpp>
#include <chameleon_renderer/utils/terminal_utils.hpp>
namespace chameleon {
template<typename T>
struct ModifyGuard
{
    ModifyGuard() = default;
    ModifyGuard(T obj)
        : _obj(std::move(obj))
    {}
    ModifyGuard(const ModifyGuard<T>& other)
        : _obj(other._obj)
    {}
    ModifyGuard(ModifyGuard<T>&& other)
        : _obj(std::move(other._obj))
    {}

    ModifyGuard& operator=(ModifyGuard<T> other)
    {
        _obj = std::move(other._obj);
        modified = true;
        return *this;
    }

    ModifyGuard& operator=(T obj)
    {
        _obj = obj;
        modified = true;
        return *this;
    }
    const T& get() const { return _obj; };
    const T* operator->() const { return &_obj; }

    void set(const T& obj)
    {
        _obj = obj;
        modified = true;
    }
    void set(T&& obj)
    {
        _obj = obj;
        modified = true;
    }

    bool is_modified() const { return modified; }

    void clear_modified() { modified = false; }

protected:
    bool modified = true;
    T _obj;
};

struct SceneObject
{
    using mat_t = typename eigen_utils::InvertableMatrix<float, 4, 4>::mat_t;

    SceneObject() = default;
    SceneObject(mat_t obj_mat)
        : _object_matrix(std::move(obj_mat))
    {}
    SceneObject(const SceneObject&) = default;
    SceneObject& operator=(SceneObject other)
    {
        _object_matrix = std::move(other._object_matrix);
        modified = true;
        return *this;
    }
    void set_object_matrix(mat_t mat) { _object_matrix.set(std::move(mat)); }

    bool is_modified() const { return _object_matrix.is_modified(); }
    const auto& object_matrix() { return _object_matrix->mat(); }
    const auto& inv_object_matrix() { return _object_matrix->inv(); }

protected:
    bool modified = true;
    ModifyGuard<eigen_utils::InvertableMatrix<float, 4, 4>> _object_matrix;
};
} // namespace chameleon
