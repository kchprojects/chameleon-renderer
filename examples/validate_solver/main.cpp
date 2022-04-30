#include <chameleon_renderer/HW/HWSetup.hpp>
#include <chameleon_renderer/materials/barytex/GeneralMeasurement.hpp>
#include <chameleon_renderer/materials/barytex/io.hpp>
#include <chameleon_renderer/optix/OptixScene.hpp>
#include <chameleon_renderer/renderer/barytex/BarytexLearnRenderer.hpp>
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

Eigen::Matrix4f rotation(const Eigen::Vector3f& r_vec) {
    return rotation(r_vec.x(), r_vec.y(), r_vec.z());
}

struct Args {
    fs::path measurements_path =
        "/home/karelch/Diplomka/rendering/chameleon-renderer/examples/"
        "validate_solver/mesurements/fi_roc_mes.isobrdf";
    fs::path obj_path =
        "/home/karelch/Diplomka/dataset_v1/fi_rock/reconstruction_dummy.obj";

    Args() = default;
    Args(int argc, char** argv) {
        if (argc > 1) {
            measurements_path = argv[1];
        }
        if (argc > 2) {
            obj_path = argv[2];
        }
    }
};

extern "C" int main(int argc, char** argv) {
    Args args(argc, argv);

    auto sm = SceneModel{args.obj_path};

    eigen_utils::Mat4<float> correction = eigen_utils::Mat4<float>::Identity();
    correction(0, 0) = -1;
    correction(1, 1) = 1;
    correction(2, 2) = 1;
    sm.obj_mat = correction * rotation(M_PI / 2, 0, 0);
    sm.update();

    std::cout << "loading" << std::endl;
    auto all_mes = import_isotropic_bin(args.measurements_path);
    std::cout << "loaded" << std::endl;
    GeneralMeasurement gm(sm.mesh(0), all_mes, 0.02, 3);
    auto lambert_forest = compute_model<MaterialModel::BlinPhong>(gm, 0.01);
    auto lf_json = lambert_forest.serialize();
    {
        std::ofstream ofs("fi_rock_bp.json");
        ofs << lf_json << std::endl;
    }
    return 0;
}

}  // namespace chameleon
