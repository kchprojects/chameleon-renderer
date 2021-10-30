// #include <GL/gl.h>

#include <chrono>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <chameleon_renderer/optix/OptixScene.hpp>
#include <chameleon_renderer/renderer/Renderer.hpp>

// #define SHOW_SHADOWS
namespace chameleon {
using namespace eigen_utils;

float to_rad(float deg) {
    return deg * M_PI / 180;
}

Eigen::Matrix4f get_transform_matrix(float x = 0,
                                     float y = 0,
                                     float z = 0,
                                     float rx = 0,
                                     float ry = 0,
                                     float rz = 0) {
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m.block<3, 3>(0, 0) = euler_to_mat(to_rad(rx), to_rad(ry), to_rad(rz));
    Eigen::Matrix4f t = eigen_utils::translation_mat(x, y, z);
    return t * m;
}

std::vector<std::shared_ptr<ILight>> get_lights() {
    std::vector<std::shared_ptr<ILight>> lights;
    for (int i = 0; i < 24; ++i) {
        float x = std::cos((i * 15) * M_PI / 180) * 100.0f;
        float y = std::sin((i * 15) * M_PI / 180) * 100.0f;
        eigen_utils::Vec3<float> pos;
        pos << x, y, 0.0f;
        lights.push_back(std::make_shared<PointLight>(pos));
    }
    return lights;
}

template <typename T>
void project_img_to_uv(cv::Mat_<T> img,
                       cv::Mat_<cv::Vec3f> uv_img,
                       std::map<int, cv::Mat_<T>>& out_map,
                       cv::Mat_<float>& hitcount,
                       int uv_size = 2048) {
    cv::namedWindow("hitcount", cv::WINDOW_NORMAL);
    int position = 0;

    for (cv::Vec3f& val : uv_img) {
        int meshID = int(val(2));
        if (meshID > -1) {
            if (out_map.count(meshID) == 0) {
                out_map[meshID] = cv::Mat_<T>::zeros(uv_size, uv_size);
            }
            float coord_x = (uv_size - 1) * (1 - val(0));
            float coord_y = (uv_size - 1) * (1 - val(1));
            out_map[meshID](std::floor(coord_x), std::floor(coord_y)) =
                img(position);
            hitcount(std::floor(coord_x), std::floor(coord_y)) += 1;

            // out_map[meshID](std::floor(coord_x), std::ceil(coord_y)) =
            //     img(position);
            // hitcount(std::floor(coord_x), std::ceil(coord_y)) += 1;

            // out_map[meshID](std::ceil(coord_x), std::floor(coord_y)) =
            //     img(position);
            // hitcount(std::ceil(coord_x), std::floor(coord_y)) += 1;

            // out_map[meshID](std::ceil(coord_x), std::ceil(coord_y)) =
            //     img(position);
            // hitcount(std::ceil(coord_x), std::ceil(coord_y)) += 1;
        }
        ++position;
    }

    cv::Mat show;
    cv::normalize(hitcount, show, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imshow("hitcount", show);
}

template <typename T>
std::map<int, cv::Mat_<T>> project_img_to_uv(cv::Mat_<T> img,
                                             cv::Mat_<cv::Vec3f> uv_img,
                                             int uv_size = 1024) {
    std::map<int, cv::Mat_<T>> out_map;
    project_img_to_uv(img, uv_img, out_map, uv_size);
    return out_map;
}
std::vector<eigen_utils::Mat4<float>> rotate_around_x(float step = 10,
                                                      float dist = 10) {
    std::vector<eigen_utils::Mat4<float>> mats = {
        get_transform_matrix(0, 0, -dist, 0, 0, 0),
    };
    for (int i = 0; i < 360 / step; ++i) {
        mats.push_back(get_transform_matrix(0, 0, 0, step, 0, 0));
    }
    return mats;
}
std::vector<eigen_utils::Mat4<float>> rotate_full(float step = 10,
                                                  float dist = 10) {
    std::vector<eigen_utils::Mat4<float>> mats = {
        get_transform_matrix(0, 0, -dist, 0, 0, 0),
    };
    for (int i = 0; i < 360 / step; ++i) {
        mats.push_back(get_transform_matrix(0, 0, 0, step, 0, 0));
    }
    // for (int i = 0; i < 360 / step; ++i) {
    //     mats.push_back(get_transform_matrix(0, 0, 0, 0, step, 0));
    // }
    return mats;
}

extern "C" int main(int argc, char** argv) {
    // std::string obj_path = "../data/models/inter_2.obj";
    // std::string obj_path = "../data/models/sponza.obj";
    int tex_size = 2048;
    std::string obj_path = "../data/models/monkey.obj";
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

    std::vector<eigen_utils::Mat4<float>> mats = rotate_full(10, 1);
    // std::vector<eigen_utils::Mat4<float>> mats = get_view_matrices();
    renderer.photometry_camera(cam_label).set_lights(get_lights());
    cv::Mat out;
    cv::namedWindow("nml", cv::WINDOW_NORMAL);
    cv::namedWindow("uv", cv::WINDOW_NORMAL);
    cv::namedWindow("mask", cv::WINDOW_NORMAL);
    cv::namedWindow("view", cv::WINDOW_NORMAL);
    cv::namedWindow("0", cv::WINDOW_NORMAL);
    bool should_end = false;
    renderer.photometry_camera(cam_label);
    std::map<int, cv::Mat_<cv::Vec3b>> tex_buffer;
    cv::Mat_<float> hitcount = cv::Mat_<float>::zeros(tex_size, tex_size);
    cv::Mat photo = cv::imread("mask_prep.png", cv::IMREAD_COLOR);
    for (const auto& m : mats) {
        renderer.photometry_camera(cam_label).move_by(m);
        auto out = renderer.render(cam_label);

        cv::Mat nml = out.normal_map.get_cv_mat();
        nml.convertTo(nml, CV_8UC3, 127, 127);
        cv::cvtColor(nml, nml, cv::COLOR_BGR2RGB);
        cv::imshow("nml", nml);
        cv::imwrite("normal.png", nml);

        cv::Mat view = out.view.get_cv_mat();
        cv::normalize(view, view, 0, 255, cv::NORM_MINMAX, CV_8UC3);
        cv::imshow("view", view);

        auto uv = out.uv_map.get_cv_mat();
        cv::imshow("uv", uv);
        cv::Mat uv_show = uv * 255;
        uv_show.convertTo(uv_show, CV_8UC3);
        cv::imwrite("uv.png", uv_show);
        cv::Mat_<uchar> mask = out.mask.get_cv_mat();
        cv::imshow("mask", mask);
        cv::imwrite("mask.png", mask);
        // project_img_to_uv(mask, uv, tex_buffer,1024);

        // project_img_to_uv<cv::Vec3b>(view, uv, tex_buffer,
        // hitcount,tex_size);
        for (const auto& [lab, tex] : tex_buffer) {
            cv::Mat inp;
            cv::Mat_<uchar> inf_mask = (tex == 0) * 255;
            cv::Mat dist;
            cv::distanceTransform(inf_mask, dist, cv::DIST_L1, 3, CV_32F);
            inf_mask.setTo(0, dist > 15);

            cv::Mat rgb_mat = tex;
            rgb_mat.convertTo(rgb_mat, CV_8UC3);
            // inpaint(rgb_mat,inf_mask,inp,5,cv::INPAINT_TELEA);
            cv::imshow(std::to_string(lab), tex);
            // cv::imshow(std::to_string(lab)+"_inp", inp);
        }

        char k = cv::waitKey(1) & 0xFF;
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
    }
    cv::waitKey();
    cv::Mat show;
    cv::normalize(hitcount, show, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite("hitcount.png", show);
    for (const auto& [lab, tex] : tex_buffer) {
        cv::imwrite(std::to_string(lab) + ".png", tex);
    }

    return 0;
}  // namespace chameleon

}  // namespace chameleon
