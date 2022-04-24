#pragma once

#include <chameleon_renderer/materials/barytex/MeasurementTree.hpp>
#include <chameleon_renderer/materials/barytex/GenericMaterial.hpp>
#include <nlohmann/json.hpp>

namespace chameleon
{
    struct MeasurementLookupTree{
        struct MeasurementPoint{
            std::unique_ptr<IGenericMaterial> material;
            glm::vec3 position;    
            bool filled = false;        
        };

        struct Divisor{
            int A_id;
            int B_id;
            int C_id;
            int child_id=-1;
            //cache ? 
        };

        struct DivisorNode{
            int parent_id = -1;
            std::array<Divisor,4> divisors;
            int A,B,C;

        };

        MeasurementLookupTree(const glm::vec3& A,const glm::vec3& B,const glm::vec3& C, float min_distance, int max_depth=-1);
        MeasurementLookupTree(const nlohmann::json&);


        std::vector<MeasurementPoint> measurement_points;
        std::vector<DivisorNode> divisors;
        int tree_depth;

        nlohmann::json serialize() const;


        

    };
} // namespace chameleon
