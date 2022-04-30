#include <chameleon_renderer/materials/barytex/GeneralMeasurement.hpp>
#include <chameleon_renderer/materials/barytex/GenericMaterial.hpp>
#include <unordered_set>

namespace chameleon {
GeneralMeasurement::GeneralMeasurement(
    const TriangleMesh& mesh, const std::vector<MeasurementHit>& all_hits,
    float minimal_distance, int max_depth)
    : lookup_forest(mesh, minimal_distance, max_depth),
      _mesh(mesh.traversable_mesh()),
      measurements(lookup_forest.trees.size()) {
    std::cout << "copy_hits" << std::endl;
    for (const auto& hit : all_hits) {
        if (!hit.is_valid) continue;
        measurements[hit.triangle_id].emplace_back(hit);
    }
    std::cout << "done_copy" << std::endl;
}

GeneralMeasurement::GeneralMeasurement(
    const TriangleMesh& mesh,
    const std::vector<IsotropicBRDFMeasurement>& all_hits,
    float minimal_distance, int max_depth)
    : lookup_forest(mesh, minimal_distance, max_depth),
      _mesh(mesh.traversable_mesh()),
      measurements(lookup_forest.trees.size()) {
    std::cout << "copy_hits" << std::endl;
    for (const auto& hit : all_hits) {
        measurements[hit.triangle_id].push_back(hit);
    }
    std::cout << "done_copy" << std::endl;
}

}  // namespace chameleon