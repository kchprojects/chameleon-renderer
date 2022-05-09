#include <chameleon_renderer/cuda/CUDABuffer.hpp>
#include <chameleon_renderer/materials/barytex/MaterialLookupForest.hpp>
#include <chameleon_renderer/utils/io.hpp>
namespace chameleon {

MaterialLookupForest::MaterialLookupForest(const TriangleMesh& mesh,
                                           float minimal_distance,
                                           int maximal_tree_depth) {
    int i = 0;
    for (const auto& face : mesh.index) {
        std::cout << float(i++) * 100.f / mesh.index.size()
                  << "%                  \r" << std::flush;
        trees.emplace_back(mesh.vertex[face[0]], mesh.vertex[face[1]],
                           mesh.vertex[face[2]], minimal_distance,
                           maximal_tree_depth);
    }
}
MaterialLookupForest::MaterialLookupForest(const nlohmann::json& j) {
    for (const auto& tree_j : j) {
        trees.emplace_back(tree_j);
    }
}

nlohmann::json MaterialLookupForest::serialize() const {
    nlohmann::json out = nlohmann::json::array();
    int i = 0;
    for (const auto& tree : trees) {
        std::cout << "saving " << float(i) * 100.f / trees.size()
                  << "%                  \r" << std::flush;
        out.push_back(tree.serialize());
        ++i;
    }
    return out;
}

CUDAMaterialLookupForest MaterialLookupForest::upload_to_cuda() const {
    std::vector<CUDAMaterialLookupTree> cuda_forest;
    cuda_forest.reserve(trees.size());
    PING;
    for (const auto& tree : trees) {
        cuda_forest.push_back(tree.upload_to_cuda());
    }
    PING;
    CUDABuffer buff;
    buff.alloc_and_upload(cuda_forest);
    return {reinterpret_cast<CUDAMaterialLookupTree*>(buff.d_pointer()),
            trees.size()};
}

void MaterialLookupForest::serialize_bin(const fs::path& path) const {
    std::ofstream ofs(path, std::ios::binary);
    size_t tree_count = trees.size();

    write_bin(ofs, tree_count);
    for (const auto& tree : trees) {
        write_bin(ofs, tree.tree_depth);
        size_t point_count = tree.measurement_points.size();

        write_bin(ofs, point_count);
        for (const auto& point : tree.measurement_points) {
            write_bin(ofs, point.position);
            bool has_material = point.material != nullptr;

            write_bin(ofs, has_material);
            if (has_material) {
                auto model_type = point.material->m;
                auto channel_count = point.material->channel_count;

                write_bin(ofs, model_type);
                write_bin(ofs, channel_count);
                if (channel_count == 1) {
                    switch (model_type) {
                        case MaterialModel::Lambert: {
                            auto& material = dynamic_cast<
                                GenericMaterial<MaterialModel::Lambert, 1>&>(
                                *point.material);

                            write_bin(ofs, material.data_rgb[0]);
                        } break;
                        case MaterialModel::BlinPhong: {
                            auto& material = dynamic_cast<
                                GenericMaterial<MaterialModel::BlinPhong, 1>&>(
                                *point.material);
                            write_bin(ofs, material.data_rgb[0]);
                        } break;
                        case MaterialModel::CookTorrance: {
                            auto& material = dynamic_cast<GenericMaterial<
                                MaterialModel::CookTorrance, 1>&>(
                                *point.material);
                            write_bin(ofs, material.data_rgb[0]);
                        } break;
                        default:
                            throw std::invalid_argument(
                                "unsupported model_type");
                    }
                } else if (channel_count == 3) {
                    switch (model_type) {
                        case MaterialModel::Lambert: {
                            auto& material = dynamic_cast<
                                GenericMaterial<MaterialModel::Lambert>&>(
                                *point.material);

                            write_bin(ofs, material.data_rgb[0]);
                            write_bin(ofs, material.data_rgb[1]);
                            write_bin(ofs, material.data_rgb[2]);
                        } break;
                        case MaterialModel::BlinPhong: {
                            auto& material = dynamic_cast<
                                GenericMaterial<MaterialModel::BlinPhong>&>(
                                *point.material);
                            write_bin(ofs, material.data_rgb[0]);
                            write_bin(ofs, material.data_rgb[1]);
                            write_bin(ofs, material.data_rgb[2]);
                        } break;
                        case MaterialModel::CookTorrance: {
                            auto& material = dynamic_cast<
                                GenericMaterial<MaterialModel::CookTorrance>&>(
                                *point.material);
                            write_bin(ofs, material.data_rgb[0]);
                            write_bin(ofs, material.data_rgb[1]);
                            write_bin(ofs, material.data_rgb[2]);
                        } break;
                        default:
                            throw std::invalid_argument(
                                "unsupported model_type");
                    }
                } else {
                    throw std::runtime_error(
                        "invalid channel count in write tree bin");
                }
            }
        }

        write_vector_bin(ofs, tree.divisors);
    }
}

void MaterialLookupForest::load_bin(const fs::path& path) {
    std::ifstream ifs(path, std::ios::binary);
    size_t tree_count;
    read_bin(ifs, tree_count);
    trees.reserve(tree_count);
    for (auto tree_id = 0u; tree_id < tree_count; ++tree_id) {
        trees.emplace_back();
        auto& tree = trees.back();
        read_bin(ifs, tree.tree_depth);
        size_t point_count;
        read_bin(ifs, point_count);
        tree.measurement_points.reserve(point_count);
        for (auto point_id = 0u; point_id < point_count; ++point_id) {
            tree.measurement_points.emplace_back();
            auto& point = tree.measurement_points.back();
            read_bin(ifs, point.position);
            bool has_material;
            read_bin(ifs, has_material);
            if (has_material) {
                MaterialModel model_type;
                size_t channel_count;
                read_bin(ifs, model_type);
                read_bin(ifs, channel_count);
                if (channel_count == 1) {
                    switch (model_type) {
                        case MaterialModel::Lambert: {
                            auto material = std::make_unique<
                                GenericMaterial<MaterialModel::Lambert, 1>>();
                            read_bin(ifs, material->data_rgb[0]);
                            point.material = std::move(material);
                        } break;
                        case MaterialModel::BlinPhong: {
                            point.material = std::make_unique<
                                GenericMaterial<MaterialModel::BlinPhong, 1>>();
                            auto& material = dynamic_cast<
                                GenericMaterial<MaterialModel::BlinPhong, 1>&>(
                                *point.material);
                            read_bin(ifs, material.data_rgb[0]);
                        } break;
                        case MaterialModel::CookTorrance: {
                            auto material = std::make_unique<GenericMaterial<
                                MaterialModel::CookTorrance, 1>>();
                            read_bin(ifs, material->data_rgb[0]);
                            point.material = std::move(material);
                        } break;
                        default:
                            throw std::invalid_argument(
                                "unsupported model_type");
                    }
                } else if (channel_count == 3) {
                    switch (model_type) {
                        case MaterialModel::Lambert: {
                            auto material = std::make_unique<
                                GenericMaterial<MaterialModel::Lambert, 3>>();
                            read_bin(ifs, material->data_rgb[0]);
                            read_bin(ifs, material->data_rgb[1]);
                            read_bin(ifs, material->data_rgb[2]);
                            point.material = std::move(material);
                        } break;
                        case MaterialModel::BlinPhong: {
                            point.material = std::make_unique<
                                GenericMaterial<MaterialModel::BlinPhong, 3>>();
                            auto& material = dynamic_cast<
                                GenericMaterial<MaterialModel::BlinPhong, 3>&>(
                                *point.material);
                            read_bin(ifs, material.data_rgb[0]);
                            read_bin(ifs, material.data_rgb[1]);
                            read_bin(ifs, material.data_rgb[2]);
                        } break;
                        case MaterialModel::CookTorrance: {
                            auto material = std::make_unique<GenericMaterial<
                                MaterialModel::CookTorrance, 3>>();
                            read_bin(ifs, material->data_rgb[0]);
                            read_bin(ifs, material->data_rgb[1]);
                            read_bin(ifs, material->data_rgb[2]);
                            point.material = std::move(material);
                        } break;
                        default:
                            throw std::invalid_argument(
                                "unsupported model_type");
                    }
                } else {
                    throw std::runtime_error(
                        "invalid channel count in load bin material forest");
                }
            }
        }
        read_vector_bin(ifs, tree.divisors);
    }
}
}  // namespace chameleon