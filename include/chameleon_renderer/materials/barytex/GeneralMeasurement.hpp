#include <chameleon_renderer/materials/barytex/BRDFMeasurement.hpp>
#include <chameleon_renderer/materials/barytex/MaterialLookupForest.hpp>
#include <chameleon_renderer/materials/barytex/MaterialModelSolver.hpp>
#include <chameleon_renderer/materials/barytex/MeasurementHit.hpp>
#define DEBUG_MESSAGES
#include <chameleon_renderer/utils/debug_io.hpp>
#include <chameleon_renderer/scene/Model.hpp>

namespace chameleon {
struct GeneralMeasurement {
    MaterialLookupForest lookup_forest;
    const TriangleMesh::TraversableTriangleMesh& _mesh;
    std::vector<std::vector<IsotropicBRDFMeasurement>> measurements;

    GeneralMeasurement(const TriangleMesh& mesh,
                       const std::vector<MeasurementHit>& all_hits,
                       float minimal_distance, int max_depth);

    GeneralMeasurement(const TriangleMesh& mesh,
                       const std::vector<IsotropicBRDFMeasurement>& all_hits,
                       float minimal_distance, int max_depth);
};

template <MaterialModel model_t>
inline MaterialLookupForest compute_model(const GeneralMeasurement& measurement,
                                          float max_distance = 10000) {
    static const float distance_to_unity = 0.001;
    MaterialLookupForest out = measurement.lookup_forest;
    std::cout << "computing_model" << std::endl;
    for (auto triangle_id = 0u; triangle_id < out.trees.size(); ++triangle_id) {
        std::cout << triangle_id * 100.0 / out.trees.size()
                  << "%                 \r" << std::flush;
        std::unordered_set<int> valid_triangles = {};   
        for (int vertex_id :
             measurement._mesh.faces[triangle_id].vert_indices) {       
            for (int neighbour_id :
                 measurement._mesh.vertices[vertex_id].face_indices) {
                valid_triangles.insert(neighbour_id);
            }
        }
        std::cout<<"valid_triangles: " << valid_triangles.size()<<std::endl;

        for (auto& mes_point : out.trees[triangle_id].measurement_points) {
            ModelSolver<model_t> solver;
            int observation_count = 0;
            for (int searched_triangle : valid_triangles) {
                for (const auto& mes :
                     measurement.measurements[searched_triangle]) {
                    float distance =glm::distance(mes_point.position,mes.world_coordinates);
                    // std::cout<<distance<<std::endl;
                    // // print_vec(mes_point.position,"mespoint");
                    // // print_vec(mes.world_coordinates,"mes");
                    if(distance  <= max_distance){
                        solver.add_observation(
                            mes, 1.0 / std::max(distance, distance_to_unity));
                        ++observation_count;
                    }
                }
            }
            if (observation_count > 5) {
                std::cout<<"solving from: " << observation_count << "obs"<<std::endl;
                solver.solve();
                mes_point.material =
                    std::make_unique<GenericMaterial<model_t>>(solver.res_data);
            }else{
                std::cout<<"cannot solve from: " << observation_count << " observations"<<std::endl;
            }
        }
    }
    return out;
}
}  // namespace chameleon