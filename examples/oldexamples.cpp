

void run_renderer(PhotometryRenderer& renderer) {
    cv::namedWindow("mat", cv::WINDOW_NORMAL);
    float x = 2;
    float y = 0;
    float z = 0;
    std::vector<cv::Mat> shadows;
    bool should_end = false;
    Eigen::Matrix<float, 3, 1> pos;
    pos << 0, 0, 100;
    float rot_step = M_PI / 2;
    while (!should_end) {
        //{ setup render
        Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
        m.block<3, 3>(0, 0) =
            euler_to_mat(x * rot_step, y * rot_step, z * rot_step);
        m.block<3, 1>(0, 3) = pos;

        renderer.get_scene().camera().set_object_matrix(m);
        //}

        //{
        timed_function<true>([&]() { renderer.render(); });
        //}

        //{ visualize

        cv::Mat out;
#ifdef SHOW_SHADOWS
        cv::Mat baked_shadows = renderer.get_shadow_stack().as_cvMat();
        cv::split(baked_shadows, shadows);
        std::vector<cv::Mat> cols = {shadows[0], shadows[1], shadows[2]};
        cv::merge(cols, out);
#else
        cv::Mat nml = renderer.get_normal_map();
        nml = cv::Mat(nml.mul(cv::Scalar(1, -1, -1)) + cv::Scalar(1, 1, 1))
                  .mul(cv::Scalar(127, 127, 127));
        nml.convertTo(out, CV_8UC3);
        cv::cvtColor(out, out, cv::COLOR_BGR2RGB);
#endif
        cv::imshow("mat", out);
        char k = cv::waitKey();
        switch (k) {
            case 'd':
                x += 1;
                break;
            case 'a':
                x -= 1;
                break;
            case 'w':
                y += 1;
                break;
            case 's':
                y -= 1;
                break;
            case 'e':
                z += 1;
                break;
            case 'q':
                z -= 1;
                break;
            case char(27):
                should_end = true;
                break;
            default:
                break;
        }
        //}
    }
}