#pragma once
#include <glm/glm.hpp>
#include <chameleon_renderer/utils/file_utils.hpp>
#include <chameleon_renderer/materials/barytex/MeasurementHit.hpp>

namespace chameleon {

    enum class Wavelength{
        Grayscale,
        Red,
        Green,
        Blue
    };

    //TODO c++-20 concepts
    template<typename Derived>
    struct MaterialModel{       
        MaterialModel(){
            //Test interface without virtual methods in compiletime
            float f = static_cast<const Derived*>(this)->get_reflectance(glm::vec3(1.f,0.f,0.f),glm::vec3(1.f,0.f,0.f),glm::vec3(1.f,0.f,0.f),Wavelength::Grayscale);

            // static_cast<const Derived*>(this)->learn_single();
        }
    };



    class MaterialModelMock : public MaterialModel<MaterialModelMock>{
        public:

        MaterialModelMock(const glm::vec3& real_position, const std::vector<MeasurementHit>& hits){
            //TODO: create model from hits
        }

        float get_reflectance(const glm::vec3& normal, const glm::vec3& in, const glm::vec3& out, Wavelength w = Wavelength::Grayscale) const{
            // for mock only cosine
            return glm::dot(glm::normalize(normal),glm::normalize(in));
        }

        void load(const fs::path&){}

    };

}