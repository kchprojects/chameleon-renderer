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

Eigen::Matrix3f rotation3D(float rx, float ry, float rz) {
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX()) *
        Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ());

    return m;
}
Eigen::Matrix4f rotation(float rx, float ry, float rz) {
    Eigen::Matrix4f out = Eigen::Matrix4f::Identity();
    out.block<3, 3>(0, 0) = rotation3D(rx,ry,rz);
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
    bool show_movement = true;
    bool show_light_movement = false;
    std::string obj_path =
        "/home/karelch/Diplomka/rendering/chameleon-renderer/resources/models/"
        "monkey.obj";
    std::string calib_path =
        "/home/karelch/Diplomka/rendering/chameleon-renderer/examples/"
        "validate_renderer/cameras/calib1.json";
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
                        PhotometryCamera({4096, 2176}, Json::parse(ifs)));

    
   // // cv::Mat out;
    cv::namedWindow("view", cv::WINDOW_NORMAL);
    // cv::namedWindow("photo", cv::WINDOW_NORMAL);
    bool should_end = false;
    renderer.photometry_camera(cam_label);
    eigen_utils::Mat4<float> view_mat =
        translation(0, 0, 50) * rotation(0, M_PI, 0);
    
    // auto m = mats[60];
    renderer.photometry_camera(cam_label).move_to(view_mat);
    // std::cout << m << std::endl;
    eigen_utils::Vec3<float> pos;
    pos << 50,0,0; 
    if(!show_light_movement){
        std::vector<std::shared_ptr<ILight>> lights;
        lights.push_back(std::make_shared<PointLight>(rotation3D(0, M_PI/4, 0) * pos));
        renderer.photometry_camera(cam_label).set_lights(lights);
    }
    while (!should_end) {
        for (float rot_y = 0; rot_y < 2*M_PI; rot_y += M_PI / 32) {
            if(show_movement){
                renderer.photometry_camera(cam_label).move_by(rotation(0, M_PI / 32, 0));
            }
            if(show_light_movement){
                std::vector<std::shared_ptr<ILight>> lights;
                lights.push_back(std::make_shared<PointLight>(rotation3D(0, rot_y, 0) * pos));
                renderer.photometry_camera(cam_label).set_lights(lights);
            }
    
            
    
            auto out = renderer.render(cam_label);

            cv::Mat view = out.view.get_cv_mat();
            // cv::normalize(view, view, 0, 255, cv::NORM_MINMAX, CV_8UC3);
            // cv::imshow("view", view);
            // cv::Mat nml = out.normal_map.get_cv_mat();
            // nml.convertTo(nml, CV_8UC3, 127, 127);
            // cv::cvtColor(nml, nml, cv::COLOR_BGR2RGB);
            cv::Mat mask = out.mask.get_cv_mat();
            // cv::addWeighted(view, 0.5, photo, 0.5, 0, view);
            // view.copyTo(photo, mask);
            cv::imshow("view", view);

            char k = cv::waitKey(10) & 0xFF;
            switch (k) {
                case char(27):
                    should_end = true;
                    break;
                default:
                    break;
            }
        }
    }
    return 0;
}  // namespace chameleon

}  // namespace chameleon
