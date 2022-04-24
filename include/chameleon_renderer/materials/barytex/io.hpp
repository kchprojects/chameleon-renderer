#pragma once
#include <chameleon_renderer/materials/barytex/MeasurementHit.hpp>
#include <chameleon_renderer/utils/file_utils.hpp>
#include <chameleon_renderer/utils/math_io.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

namespace chameleon {

void to_json(nlohmann::json& out, const MeasurementHit& hit) {
    out["triangle_id"] = to_json(hit.triangle_id);
    out["coordinates"] = to_json(hit.coordinates);
    out["world_coordinates"] = to_json(hit.world_coordinates);
    out["eye"] = to_json(hit.eye);
    out["light"] = to_json(hit.light);
    out["mesh_normal"] = to_json(hit.mesh_normal);
    out["value"] = to_json(hit.value);
    out["valid"] = to_json(hit.is_valid);
}

void from_json(const nlohmann::json& in, MeasurementHit& hit) {
    hit.triangle_id = in.at("triangle_id");
    hit.coordinates = vec_from_json(in.at("coordinates"));
    hit.world_coordinates = vec_from_json(in.at("world_coordinates"));
    hit.eye = vec_from_json(in.at("eye"));
    hit.light = vec_from_json(in.at("light"));
    hit.mesh_normal = vec_from_json(in.at("mesh_normal"));
    hit.value = vec_from_json(in.at("value"));
    hit.is_valid = in.at("valid");
}

void export_measurement_pcd(const fs::path& path,
                            const std::vector<MeasurementHit>& hits) {
    std::ofstream ofs(path);
    for (const auto& hit : hits) {
        if (hit.is_valid) {
            ofs << hit.world_coordinates.x << " " << hit.world_coordinates.y
                << " " << hit.world_coordinates.z << " " << hit.value.x << " "
                << hit.value.y << " " << hit.value.z << "\n";
        }
    }
}

void export_measurement(const std::vector<MeasurementHit>& data,
                        const fs::path& path,bool compress = false) {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& mes : data) {
        if(!compress || mes.is_valid){
            j.emplace_back(mes);
        }
    }
    std::ofstream ofs(path);
    ofs << j;
}
std::vector<MeasurementHit> import_measurement(const fs::path& path, bool only_valid=false) {
    nlohmann::json j;
    std::vector<MeasurementHit> data;
    {
        std::ifstream ifs(path);
        ifs >> j;
    }
    for (const auto& mes_j : j) {
        if( !only_valid || mes_j.at("is_valid").get<bool>()){
            data.emplace_back(mes_j);
        }
    }
    return data;
}
}  // namespace chameleon