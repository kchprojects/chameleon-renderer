#pragma once
#include <iostream>


inline void print_vec(glm::vec3 v, std::string name) {
#ifdef DEBUG_MESSAGES
    std::cout << name << (name.empty() ? "" : ":") << "[" << v.x << ", " << v.y
              << ", " << v.z << "]" << std::endl;
#endif
}
template<typename T>
inline void print_var(T var, std::string name){
    #ifdef DEBUG_MESSAGES
    if(name.empty()){
        name = "variable";
    }else{
        std::cout<< name << " : " << var << std::endl;
    }
    #endif
}