#pragma once

// GLM cuda compatibility
#ifdef __CUDA_ARCH__
// #define GLM_FORCE_CUDA
#define GLM_FORCE_SINGLE_ONLY
#endif
#include <glm/glm.hpp>

namespace chameleon {
// TODO: remove gdm dependency
struct Triangle3D {
    glm::vec3 A;
    glm::vec3 B;
    glm::vec3 C;
};

struct Triangle2D {
    glm::vec2 A;
    glm::vec2 B;
    glm::vec2 C;
};
template<typename T>
struct box3{
    glm::vec<3,T> min = {0,0,0};
    glm::vec<3,T> max = {0,0,0};

    void extend(const glm::vec<3,T>& vtx){
        if(vtx.x < min.x){
            min.x = vtx.x;
        }
        if(vtx.x > max.x){
            max.x = vtx.x;
        }

        if(vtx.y < min.y){
            min.y = vtx.y;
        }
        if(vtx.y > max.y){
            max.y = vtx.y;
        }

        if(vtx.z < min.z){
            min.z = vtx.z;
        }
        if(vtx.z > max.z){
            max.z = vtx.z;
        }
        
    }

};


using box3f = box3<float>;
using vec2i = glm::vec<2,int>;
using vec3i = glm::vec<3,int>;

}  // namespace chameleon