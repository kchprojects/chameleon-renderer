#include <chameleon_renderer/materials/barytex/BRDFMeasurement.hpp>
#include <chameleon_renderer/materials/barytex/MaterialLookupForest.hpp>
#include <chameleon_renderer/materials/barytex/MaterialModelSolver.hpp>
#include <chameleon_renderer/materials/barytex/MaterialModelSolverGS.hpp>
#include <chameleon_renderer/materials/barytex/MeasurementHit.hpp>
#include <chameleon_renderer/scene/Model.hpp>
#include <chameleon_renderer/utils/debug_io.hpp>

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
                                          float max_distance = 10000,
                                          int max_mes_count = 100) {
    static const float distance_to_unity = 0.001;
    MaterialLookupForest out;
    out.trees.reserve(measurement.lookup_forest.trees.size());
    for (const auto& tree : measurement.lookup_forest.trees) {
        out.trees.emplace_back();
        auto& new_tree = out.trees.back();
        new_tree.tree_depth = tree.tree_depth;
        new_tree.divisors = tree.divisors;
        new_tree.measurement_points.reserve(tree.measurement_points.size());
        for (const auto& point : tree.measurement_points) {
            new_tree.measurement_points.push_back({nullptr, point.position, true});
        }
    }

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
        // std::cout<<"valid_triangles: " << valid_triangles.size()<<std::endl;

        for (auto& mes_point : out.trees[triangle_id].measurement_points) {
            ModelSolver<model_t> solver;
            int observation_count = 0;
            std::map<float, std::vector<IsotropicBRDFMeasurement>>
                valid_measurements;
            for (int searched_triangle : valid_triangles) {
                for (const auto& mes :
                     measurement.measurements[searched_triangle]) {
                    float distance = glm::distance(mes_point.position,
                                                   mes.world_coordinates);
                    // std::cout<<distance<<std::endl;
                    // // print_vec(mes_point.position,"mespoint");
                    // // print_vec(mes.world_coordinates,"mes");
                    if (distance <= max_distance &&
                        (mes.value.x != 0 && mes.value.y != 0 &&
                         mes.value.z != 0) && mes.light().z > 0.001 && mes.eye().z > 0.001) {
                        
                        // print_vec(mes.value,"mes");
                        valid_measurements[distance].push_back(mes);
                        ++observation_count;
                    }
                }
            }
            if (observation_count > 5) {
                int added_count = 0;
                for (const auto& [distance, mes_vec] : valid_measurements) {
                    for (const auto& mes : mes_vec) {
                        solver.add_observation(
                            mes, 1.0 / std::max(distance, distance_to_unity));
                        ++added_count;
                        if (added_count > max_mes_count) break;
                    }
                    if (added_count > max_mes_count) break;
                }
                // std::cout<<"solving from: " << observation_count <<
                // "obs"<<std::endl;
                solver.solve();
                if (std::isnan(solver.res_data[0].albedo) ||
                    std::isnan(solver.res_data[1].albedo) ||
                    std::isnan(solver.res_data[2].albedo)) {
                    mes_point.material = nullptr;
                } else {
                    // std::cout<<"making_mat: " << solver.res_data[0].albedo <<
                    // ", "<< solver.res_data[1].albedo << ", " <<
                    // solver.res_data[2].albedo<<std::endl;
                    mes_point.material =
                        std::make_unique<GenericMaterial<model_t>>(
                            solver.res_data);
                }
            }
            // else{
            //     std::cout<<"cannot solve from: " << observation_count << "
            //     observations"<<std::endl;
            // }
        }
    }
    return out;
}

template <MaterialModel model_t>
inline MaterialLookupForest compute_model_gs(const GeneralMeasurement& measurement,
                                          float max_distance = 10000,
                                          int max_mes_count = 100) {
    static const float distance_to_unity = 0.001;
    MaterialLookupForest out;
    out.trees.reserve(measurement.lookup_forest.trees.size());
    for (const auto& tree : measurement.lookup_forest.trees) {
        out.trees.emplace_back();
        auto& new_tree = out.trees.back();
        new_tree.tree_depth = tree.tree_depth;
        new_tree.divisors = tree.divisors;
        new_tree.measurement_points.reserve(tree.measurement_points.size());
        for (const auto& point : tree.measurement_points) {
            new_tree.measurement_points.push_back({nullptr, point.position, true});
        }
    }

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
        // std::cout<<"valid_triangles: " << valid_triangles.size()<<std::endl;

        for (auto& mes_point : out.trees[triangle_id].measurement_points) {
            ModelSolverGS<model_t> solver;
            int observation_count = 0;
            std::map<float, std::vector<IsotropicBRDFMeasurement>>
                valid_measurements;
            for (int searched_triangle : valid_triangles) {
                for (const auto& mes :
                     measurement.measurements[searched_triangle]) {
                    float distance = glm::distance(mes_point.position,
                                                   mes.world_coordinates);
                    // std::cout<<distance<<std::endl;
                    // // print_vec(mes_point.position,"mespoint");
                    // // print_vec(mes.world_coordinates,"mes");
                    if (distance <= max_distance &&
                        (mes.value.x != 0 && mes.value.y != 0 &&
                         mes.value.z != 0) && mes.light().z > 0.001 && mes.eye().z > 0.001) {
                        
                        // print_vec(mes.value,"mes");
                        valid_measurements[distance].push_back(mes);
                        ++observation_count;
                    }
                }
            }
            if (observation_count > 5) {
                int added_count = 0;
                for (const auto& [distance, mes_vec] : valid_measurements) {
                    for (const auto& mes : mes_vec) {
                        solver.add_observation(
                            mes, 1.0 / std::max(distance, distance_to_unity));
                        ++added_count;
                        if (added_count > max_mes_count) break;
                    }
                    if (added_count > max_mes_count) break;
                }
                // std::cout<<"solving from: " << observation_count <<
                // "obs"<<std::endl;
                solver.solve();
                if (std::isnan(solver.res_data[0].albedo)) {
                    mes_point.material = nullptr;
                } else {
                    // std::cout<<"making_mat: " << solver.res_data[0].albedo <<
                    // ", "<< solver.res_data[1].albedo << ", " <<
                    // solver.res_data[2].albedo<<std::endl;
                    mes_point.material =
                        std::make_unique<GenericMaterial<model_t,1>>(
                            solver.res_data);
                }
            }
            // else{
            //     std::cout<<"cannot solve from: " << observation_count << "
            //     observations"<<std::endl;
            // }
        }
    }
    return out;
}
}  // namespace chameleon