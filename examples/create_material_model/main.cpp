// #include <GL/gl.h>

#include <chameleon_renderer/HW/HWSetup.hpp>
#include <chameleon_renderer/materials/barytex/GeneralMeasurement.hpp>
#include <chameleon_renderer/materials/barytex/io.hpp>
#include <chameleon_renderer/optix/OptixScene.hpp>
#include <chameleon_renderer/renderer/barytex/BarytexShowRenderer.hpp>
#include <chrono>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>

// #define SHOW_SHADOWS
namespace chameleon {
using namespace eigen_utils;
Eigen::Matrix3f rotation3D(float rx, float ry, float rz) {
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX()) *
        Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ());

    return m;
}
Eigen::Matrix4f rotation(float rx, float ry, float rz) {
    Eigen::Matrix4f out = Eigen::Matrix4f::Identity();
    out.block<3, 3>(0, 0) = rotation3D(rx, ry, rz);
    return out;
}

struct Args {
    fs::path obj_path;
    fs::path measurements_path;
    fs::path res_dir;

    struct ComputeModels {
        bool l_g = false;
        bool l_c = false;
        bool b_c = false;
        bool c_c = false;
    };

    ComputeModels comp_models;

    int subdivisions = 3;
    float min_tree_dist = 0.02;
    float min_valid_distance = 0.01;

    Args() = default;
    Args(int argc, char** argv) {
        if (argc > 4) {
            obj_path = argv[1];
            measurements_path = argv[2];
            res_dir = argv[3];
            fs::create_directories(res_dir);
            for (int i = 4; i < argc; ++i) {
                if (std::string(argv[i]) == std::string("lg")) {
                    comp_models.l_g = true;
                } else if (std::string(argv[i]) == std::string("lc")) {
                    comp_models.l_c = true;
                }

                else if (std::string(argv[i]) == std::string("bc")) {
                    comp_models.b_c = true;
                }

                else if (std::string(argv[i]) == std::string("cc")) {
                    comp_models.c_c = true;
                }
            }

        } else {
            throw std::invalid_argument(
                "params [obj_path] [measurement_path] [out_dir] "
                "[model_hashes lg lc bc cc]");
        }
    }

    void store(const fs::path& p) {
        nlohmann::json j;
        j["models"] = {

            {"l_g", comp_models.l_g},
            {"l_c", comp_models.l_c},
            {"b_c", comp_models.b_c},
            {"c_c", comp_models.c_c}};
        j["subdivisions"] = 3;
        j["min_tree_dist"] = 0.02;
        j["min_valid_distance"] = 0.01;
        std::ofstream ofs(p);
        ofs << j.dump(4) << std::endl;
    }
};
extern "C" int main(int argc, char** argv) {
    Args args(argc, argv);

    OptixStaticScene scene;

    auto sm = SceneModel{args.obj_path};

    eigen_utils::Mat4<float> correction = eigen_utils::Mat4<float>::Identity();
    correction(0, 0) = -1;
    correction(1, 1) = 1;
    correction(2, 2) = 1;
    sm.obj_mat = correction * rotation(M_PI / 2, 0, 0);
    sm.update();

    auto all_mes = import_isotropic_bin(args.measurements_path);
    if(all_mes.empty()){
        throw std::invalid_argument("empty measurement file");
    }
    GeneralMeasurement gm(sm.mesh(0), all_mes, args.min_tree_dist,
                          args.subdivisions);
    if (args.comp_models.l_g) {
        compute_model_gs<MaterialModel::Lambert>(gm, args.min_valid_distance)
            .serialize_bin(args.res_dir / "test_gs.lamb_forest");
    }
    if (args.comp_models.l_c) {
        compute_model<MaterialModel::Lambert>(gm, args.min_valid_distance)
            .serialize_bin(args.res_dir / "test_col.lamb_forest");
    }
    if (args.comp_models.b_c) {
        compute_model<MaterialModel::BlinPhong>(gm, args.min_valid_distance)
            .serialize_bin(args.res_dir / "test_col.bp_forest");
    }
    if (args.comp_models.c_c) {
        compute_model<MaterialModel::CookTorrance>(gm, args.min_valid_distance)
            .serialize_bin(args.res_dir / "test_col.ct_forest");
    }
    args.store(args.res_dir / "args.json");
    return 0;
}  // namespace chameleon

}  // namespace chameleon
