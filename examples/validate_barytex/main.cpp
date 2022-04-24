// #include <GL/gl.h>

#include <chameleon_renderer/HW/HWSetup.hpp>
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
    fs::path data_path = "/home/karelch/Diplomka/dataset_v1/fi_rock/";
    fs::path lights_path =
        "/home/karelch/Diplomka/rendering/chameleon-renderer/resources/setups/"
        "chameleon/lights.json";
    std::string extension = ".png";
    fs::path obj_path =
        "/home/karelch/Diplomka/dataset_v1/fi_rock/reconstruction.obj";
    fs::path views_path =
        "/home/karelch/Diplomka/dataset_v1/fi_rock/cameras.json";

    Args() = default;
    Args(int argc, char** argv) {
        if (argc > 1) {
            data_path = argv[1];
            obj_path = data_path / "reconstruction.obj";
            views_path = data_path / "cameras.json";
        }
        if (argc > 2) {
            lights_path = argv[2];
        }
    }
};

void setup_views(BarytexLearnRenderer& renderer,
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

    PING;
    OptixStaticScene scene;
    PING;
    auto sm = SceneModel{args.obj_path};
    PING;
    eigen_utils::Mat4<float> correction = eigen_utils::Mat4<float>::Identity();
    correction(0, 0) = -1;
    correction(1, 1) = 1;
    correction(2, 2) = 1;
    sm.obj_mat = correction * rotation(M_PI / 2, 0, 0);
    scene.add_model(sm);
    PING;
    nlohmann::json setup_json;
    {
        std::ifstream ifs(args.lights_path);
        ifs >> setup_json;
    }
    PING;
    HWSetup hw_setup = {setup_json};
    PING;

    BarytexLearnRenderer renderer;
    renderer.setup();
    renderer.setup_scene(scene);
    PING;
    setup_views(renderer, args.views_path);
    PING;

    // // cv::Mat out;
    cv::namedWindow("view", cv::WINDOW_NORMAL);
    // cv::namedWindow("photo", cv::WINDOW_NORMAL);
    bool should_end = false;
    std::vector<int> light_ids = {4};
    // for (auto& [cam_label, camera] : renderer.photometry_cameras) {
    for (int position = 0; position < 50; ++position) {
        std::string cam_label = std::to_string(position) + args.extension;
        // for(int light_id = 0; light_id < 127; ++light_id){
        for (int light_id : light_ids) {
            BarytexObservation observation;
            observation.cam_label = cam_label;
            auto img_path = args.data_path /
                            ("position_" + std::to_string(position)) /
                            (std::to_string(light_id) + args.extension);
            observation.image = cv::imread(img_path.string(), cv::IMREAD_COLOR);
            if (observation.image.empty()) {
                continue;
            }
            cv::cvtColor(observation.image, observation.image,
                         cv::COLOR_BGR2RGB);
            observation.light = hw_setup.lights[light_id];
            auto out = renderer.render(observation);
            auto mes = out.measurements.download();
            export_measurement(mes, "mes.json",true);
            mes = import_measurement("mes.json");
            
            export_measurement_pcd(
                "fi_rock/pcd/" + std::to_string(position) + ".txt", mes);
            // break;
        }
        // break;
    }
    return 0;
}  // namespace chameleon

}  // namespace chameleon
