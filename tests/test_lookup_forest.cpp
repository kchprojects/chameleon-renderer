#include <chameleon_renderer/materials/barytex/MaterialLookupForest.hpp>
#include <chameleon_renderer/materials/barytex/viz.hpp>
#include <chameleon_renderer/scene/SceneObject.hpp>
#include <chameleon_renderer/utils/file_utils.hpp>
#include <fstream>
using namespace chameleon;

void export_pcd(const fs::path& path,
                            const MaterialLookupForest& forest) {
    std::ofstream ofs(path);
    for (const auto& tree : forest.trees) {
        for (const auto& point : tree.measurement_points){
            ofs << point.position.x << " " << point.position.y
                << " " << point.position.z <<  "\n";
        }
    }
}

int main(){
    Model m = loadOBJ("/home/karelch/Diplomka/rendering/chameleon-renderer/resources/models/pcb_new.obj");
    MaterialLookupForest mlf(m.meshes[0],0.02,-1);
    MaterialLookupForest mlf2(m.meshes[0],0,3);
    // {
    //     std::ofstream ofs("mlf_obj.json");
    //     ofs << mlf.serialize().dump(4); 
    // }

    export_pcd("forest_sub_dist.txt",mlf);
    export_pcd("forest_sub_uniform.txt",mlf2);

    return 0;
}


// |\
// --
// |\|
