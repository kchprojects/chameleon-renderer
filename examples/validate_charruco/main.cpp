// #include <GL/gl.h>

#include <chameleon_renderer/optix/OptixScene.hpp>
#include <chameleon_renderer/renderer/PhotometryRenderer.hpp>
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
}

Eigen::Matrix4f translation(float tx, float ty, float tz) {
    Eigen::Matrix4f out = Eigen::Matrix4f::Identity();
    out.block<3, 1>(0, 3) << tx, ty, tz;
    return out;
}
Eigen::Matrix4f translation(const Eigen::Vector3f& t_vec) {
    return translation(t_vec.x(), t_vec.y(), t_vec.z());
}

Eigen::Matrix4f rotation(float rx, float ry, float rz) {
    Eigen::Matrix4f out = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX()) *
        Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ());

    out.block<3, 3>(0, 0) = m;
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

extern "C" int main(int argc, char** argv) {
    std::string obj_path = "/home/karelch/Data/witte/tile.obj";
    std::string calib_path =
        "/home/karelch/Data/witte/render_test/2_6/calib_mono.json";
    const std::string cam_label = "main_camera";
    if (argc > 1) {
        obj_path = argv[1];
    }
    if (argc > 2) {
        calib_path = argv[2];
    }

    OptixStaticScene scene;
    scene.add_model({obj_path});
    std::ifstream ifs(calib_path);

    PhotometryRenderer renderer;
    renderer.setup();
    renderer.setup_scene(scene);
    renderer.add_camera(cam_label,
                        PhotometryCamera({2064, 1544}, Json::parse(ifs)));

    eigen_utils::Mat4<float> coord_correction = rotation(90, 0, 0);
 
    std::vector<std::shared_ptr<ILight>> lights;
    lights.push_back(std::make_shared<PointLight>(pos));
    // std::vector<eigen_utils::Mat4<float>> mats = get_view_matrices();
    // renderer.photometry_camera(cam_label).set_lights(get_lights());
    // cv::Mat out;
    cv::namedWindow("view", cv::WINDOW_NORMAL);
    cv::namedWindow("photo", cv::WINDOW_NORMAL);
    bool should_end = false;
    renderer.photometry_camera(cam_label);
    for (int i = 18; i < 20; ++i) {
        // int i = 2;
        auto mats = load_mats(
            "/home/karelch/Data/witte/render_test/2_6/calib_detections/"
            "detections_" +
                std::to_string(i) + ".json",
            coord_correction);
        cv::Mat photo = cv::imread(
            "/home/karelch/Data/witte/render_test/2_6/undist_calib/" +
                std::to_string(i) + ".png",
            cv::IMREAD_COLOR);
        for (const auto& [_, m] : mats) {
            // auto m = mats[60];
            // renderer.photometry_camera(cam_label).move_to(m);
            // std::cout << m << std::endl;
            auto out = renderer.render(cam_label);
            cv::Mat view = out.view.get_cv_mat();
            cv::normalize(view, view, 0, 255, cv::NORM_MINMAX, CV_8UC3);
            cv::imshow("view", view);
            cv::Mat nml = out.normal_map.get_cv_mat();
            nml.convertTo(nml, CV_8UC3, 127, 127);
            cv::cvtColor(nml, nml, cv::COLOR_BGR2RGB);
            cv::Mat mask = out.mask.get_cv_mat();
            cv::addWeighted(view, 0.5, photo, 0.5, 0, view);
            view.copyTo(photo, mask);
            cv::imshow("photo", photo);

            char k = cv::waitKey() & 0xFF;
            switch (k) {
                case char(27):
                    should_end = true;
                    break;
                default:
                    break;
            }
            if (should_end) {
                break;
            }
            // cv::waitKey();
        }
        cv::imwrite("data_" + std::to_string(i) + ".png", photo);
    }

    return 0;
}  // namespace chameleon

}  // namespace chameleon
