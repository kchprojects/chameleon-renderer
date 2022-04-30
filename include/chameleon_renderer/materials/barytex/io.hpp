#pragma once
#include <chameleon_renderer/materials/barytex/MeasurementHit.hpp>
#include <chameleon_renderer/materials/barytex/BRDFMeasurement.hpp>
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

void export_measurement_json(const std::vector<MeasurementHit>& data,
                             const fs::path& path, bool compress = false) {
    nlohmann::json j = nlohmann::json::array();
    std::cout << "exporting measurements to: " << path << std::endl;
    int count = 0;
    for (const auto& mes : data) {
        std::cout << (count++) * 100.0 / data.size() << "%              \r"
                  << std::flush;
        if (!compress || mes.is_valid) {
            j.emplace_back(mes);
        }
    }
    std::cout << std::endl;
    std::ofstream ofs(path);
    ofs << j;
}

template<typename T>
void write_vector_bin(std::ofstream& ofs, const std::vector<T>& vec){
    size_t count = vec.size();
    ofs.write(reinterpret_cast<const char*>(&count),sizeof(count));
    ofs.write(reinterpret_cast<const char*>(vec.data()),
              sizeof(T) * count);
}

template<typename T>
std::vector<T> read_vector_bin(std::ifstream& ifs){
    size_t num_elements = 0;
    ifs.read(reinterpret_cast<char*>(&num_elements), sizeof(num_elements));
    std::vector<T> data(num_elements);
    ifs.read(reinterpret_cast<char*>(data.data()),
             sizeof(T) * num_elements);
    return data;
}

void export_measurement_bin(const std::vector<MeasurementHit>& data,
                            const fs::path& path) {
    std::ofstream ofs(path, std::ios::binary);
    std::cout << "exporting measurements to: " << path << std::endl;
    write_vector_bin(ofs,data);    
}

void export_isotropic_bin(const std::vector<IsotropicBRDFMeasurement>& data,
                            const fs::path& path) {
    std::ofstream ofs(path, std::ios::binary);
    std::cout << "exporting isotropic to: " << path << std::endl;
    write_vector_bin(ofs,data);    
}

std::vector<IsotropicBRDFMeasurement> import_isotropic_bin(const fs::path& path) {
    std::ifstream ifs(path, std::ios::binary);
    return read_vector_bin<IsotropicBRDFMeasurement>(ifs);    
}

std::vector<MeasurementHit> import_measurement_bin(const fs::path& path) {
    std::ifstream ifs(path, std::ios::binary);
    return read_vector_bin<MeasurementHit>(ifs);
}

}  // namespace chameleon