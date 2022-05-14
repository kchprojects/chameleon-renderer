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

template <typename Derived>
inline void to_json(nlohmann::json& j,
                    const Eigen::MatrixBase<Derived>& matrix) {
    for (int row = 0; row < matrix.rows(); ++row) {
        nlohmann::json column = nlohmann::json::array();
        for (int col = 0; col < matrix.cols(); ++col) {
            column.push_back(matrix(row, col));
        }
        j.push_back(column);
    }
}

template <typename Derived>
inline void from_json(const nlohmann::json& j,
                      Eigen::MatrixBase<Derived>& matrix) {
    using Scalar = typename Eigen::MatrixBase<Derived>::Scalar;

    for (std::size_t row = 0; row < j.size(); ++row) {
        const auto& jrow = j.at(row);
        for (std::size_t col = 0; col < jrow.size(); ++col) {
            const auto& value = jrow.at(col);
            matrix(row, col) = value.get<Scalar>();
        }
    }
    // std::cout<<matrix<<std::endl;
}

Eigen::Matrix4f translation(float tx, float ty, float tz) {
    Eigen::Matrix4f out = Eigen::Matrix4f::Identity();
    out.block<3, 1>(0, 3) << tx, ty, tz;
    return out;
}
Eigen::Matrix4f translation(const Eigen::Vector3f& t_vec) {
    return translation(t_vec.x(), t_vec.y(), t_vec.z());
}

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

std::map<int, eigen_utils::Mat4<float>> load_mats(
    std::string path, eigen_utils::Mat4<float> correct_mat =
                          eigen_utils::Mat4<float>::Identity()) {
    std::map<int, eigen_utils::Mat4<float>> out;
    std::ifstream ifs(path);
    nlohmann::json j;
    ifs >> j;
    for (auto obj : j) {
        eigen_utils::Mat4<float> new_mat;
        from_json(obj.at("mat"), new_mat);
        new_mat = (new_mat * correct_mat).inverse();
        out[obj.at("id")] = new_mat;
    }
    return out;
}

struct Args {

    // fs::path data_path = "/home/karelch/Diplomka/dataset_v1/fi_rock/";
    // fs::path lights_path =
    //     "/home/karelch/Diplomka/rendering/chameleon-renderer/resources/setups/"
    //     "chameleon/lights.json";
    // std::string extension = ".png";
    // fs::path obj_path =
    //     "/home/karelch/Diplomka/dataset_v1/fi_rock/reconstruction.obj";
    //     fs::path measurements_path =
    //     "/home/karelch/Diplomka/rendering/chameleon-renderer/results/fi_rock/test_mes.isobrdf";

    // fs::path views_path =
    //     "/home/karelch/Diplomka/dataset_v1/fi_rock/cameras.json";
    // fs::path res_dir = "/home/karelch/Diplomka/rendering/chameleon-renderer/results/fi_rock";
    fs::path data_path = "/home/karelch/Diplomka/dataset_v2/mag_box/";
    fs::path lights_path =
        "/home/karelch/Diplomka/rendering/chameleon-renderer/resources/setups/"
        "chameleon/lights.json";
    std::string extension = ".png";
    fs::path obj_path;
    fs::path views_path;
    fs::path forest_path;
    fs::path result_path;
    Args() = default;
    Args(int argc, char** argv) {
        if (argc > 3) {
            data_path = argv[1];
            obj_path = data_path/"reconstruction.obj";
            std::cout<< "Object: " << obj_path.string()<<std::endl;
            std::string extension = ".png";
            views_path =data_path/"cameras.json";
            std::cout<< "Views: " << views_path.string()<<std::endl;

            forest_path = argv[2];
            std::cout<< "Forest: " << forest_path.string()<<std::endl;
            result_path = argv[3];
            std::cout<< "Results: " << result_path.string()<<std::endl;

        }
        else{
            throw std::invalid_argument("parameters [data_folder] [forest_path] ");
        }
    }
};

void setup_views(BarytexShowRenderer& renderer,
                 const fs::path& view_json_path) {
    std::ifstream ifs(view_json_path);
    Json j = Json::parse(ifs);
    for (const auto& [_, view_json] : j.items()) {
        eigen_utils::Mat4<float> view_mat;
        std::string cam_label = view_json.at("img_name");
        from_json(view_json.at("view_mat"), view_mat);
        eigen_utils::Mat3<float> camera_mat;
        auto& cam_mat_jsn = view_json.at("K").at("data");
        camera_mat << cam_mat_jsn[0].get<float>(), cam_mat_jsn[1],
            cam_mat_jsn[2], cam_mat_jsn[3], cam_mat_jsn[4].get<float>(),
            cam_mat_jsn[5], cam_mat_jsn[6], cam_mat_jsn[7], cam_mat_jsn[8];
        eigen_utils::Mat4<float> correction =
            eigen_utils::Mat4<float>::Identity();
        // correction(0,0) = 1;
        // correction(0,2) = 0;

        // correction(1,1) = 0;
        // correction(1,2) = 1;

        // correction(2,2) = 0;
        // correction(2,1) = 1;
        // correction = rotation(0,M_PI,0);
        // std::cout<<correction<<std::endl;

        PhotometryCamera new_view(
            {view_json.at("resolution").at("x").get<int>(),
             view_json.at("resolution").at("y").get<int>()},
            camera_mat, view_mat * correction);
        renderer.add_camera(cam_label, new_view);

        eigen_utils::Vec3<float> pos;
        pos << 0, 0, 0;
        std::vector<std::shared_ptr<ILight>> lights;
        lights.push_back(std::make_shared<PointLight>(pos));
        std::cout << cam_label << std::endl;
        renderer.photometry_camera(cam_label).set_lights(lights);
    }
}
void write_pcd(std::string path, const std::vector<MeasurementHit>& hits) {
    std::ofstream ofs(path);
    for (const auto& hit : hits) {
        if (hit.is_valid) {
            ofs << hit.world_coordinates.x << " " << hit.world_coordinates.y
                << " " << hit.world_coordinates.z << " " << hit.value.x << " "
                << hit.value.y << " " << hit.value.z << "\n";
        }
    }
}
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
    scene.add_model(sm);

    nlohmann::json setup_json;
    {
        std::ifstream ifs(args.lights_path);
        ifs >> setup_json;
    }

    HWSetup hw_setup = {setup_json};

    BarytexShowRenderer renderer;
    renderer.setup();
    renderer.setup_scene(scene);

    setup_views(renderer, args.views_path);
    std::cout << "loading" << std::endl;
    // MaterialLookupForest lamb_forest;
    // lamb_forest.load_bin("test_ser.lamb_forest");

    // auto all_mes = import_isotropic_bin(args.measurements_path);
    // GeneralMeasurement gm(sm.mesh(0), all_mes, 0.02, 3);
    // compute_model_gs<MaterialModel::Lambert>(gm, 0.01).serialize_bin(args.res_dir / "test_gs.lamb_forest");
    // compute_model<MaterialModel::Lambert>(gm, 0.01).serialize_bin(args.res_dir / "test_col.lamb_forest");
    // compute_model<MaterialModel::BlinPhong>(gm, 0.01).serialize_bin(args.res_dir / "test_col.bp_forest");
    // compute_model<MaterialModel::CookTorrance>(gm, 0.01).serialize_bin(args.res_dir / "test_col.ct_forest");
    // auto blin_forest = compute_model<MaterialModel::CookTorrance>(gm, 0.01);
    // blin_forest.serialize_bin("test_ser_mesh.ct_forest");
    // MaterialLookupForest blin_forest;
    // blin_forest.load_bin("test_ser_att.ctor_forest");

    MaterialLookupForest forest;
    forest.load_bin(args.forest_path);
    renderer.setup_material(forest);
    PING;

    // // cv::Mat out;
    cv::namedWindow("view", cv::WINDOW_NORMAL);
    // cv::namedWindow("photo", cv::WINDOW_NORMAL);
    bool should_end = false;
    std::vector<int> light_ids = {4};
    // for (auto& [cam_label, camera] : renderer.photometry_cameras) {
    for (int position = 1; position < 40; position+=5) {
        std::string cam_label = std::to_string(position) + args.extension;

        fs::create_directories(args.result_path/std::to_string(position));
        // should be to 127
        for (int light_id = 0; light_id < 70; ++light_id) {
            // for (int light_id : light_ids) {
            if (hw_setup.lights.count(light_id) > 0) {
                renderer.photometry_camera(cam_label).set_lights(
                    {hw_setup.lights[light_id]});
                auto out = renderer.render(cam_label);
                cv::Mat view = out.view.get_cv_mat();
                view.convertTo(view,CV_8UC1);
                cv::cvtColor(view,view,cv::COLOR_RGB2BGR);
                cv::Mat1b mask = out.mask.get_cv_mat();
                cv::imwrite((args.result_path/std::to_string(position)/(std::to_string(light_id) + ".png")).string(), view);
                cv::imwrite((args.result_path/std::to_string(position)/(std::to_string(light_id) + "_mask.png")).string(), mask);
                // cv::imshow("view", view);
                // cv::imshow("mask", mask);
                // cv::waitKey();
                // break;
            }
        }
    }
    return 0;
}  // namespace chameleon

}  // namespace chameleon
