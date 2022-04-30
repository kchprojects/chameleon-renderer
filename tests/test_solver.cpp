#include <chameleon_renderer/materials/barytex/MaterialModel.hpp>
#include <chameleon_renderer/materials/barytex/MaterialModelSolver.hpp>
using namespace chameleon;

void test_lambert() {
    double base_albedo = 0.5;
    glm::vec3 normal = {0, 0, 1};
    ModelSolver<MaterialModel::Lambert> solver;
    for (float elevation = 0; elevation < M_PI / 2; elevation += M_PI / 32) {
        IsotropicBRDFMeasurement m;
        m.world_coordinates = {0, 0, 0};
        m.eye_elevation = elevation;
        m.light_elevation = elevation;
        m.light_azimuth = 0;
        m.value = {ModelReflectance<MaterialModel::Lambert, double>::compute(
                       normal, m.light(), m.eye(), {base_albedo}),
                   ModelReflectance<MaterialModel::Lambert, double>::compute(
                       normal, m.light(), m.eye(), {base_albedo}),
                   ModelReflectance<MaterialModel::Lambert, double>::compute(
                       normal, m.light(), m.eye(), {base_albedo})};
        solver.add_observation(m, 1);
    }
    solver.solve();
    std::cout << "lamb.. given: " << base_albedo << std::endl;
    std::cout << "predicted: " << solver.res_data[0].albedo << std::endl;
}

void test_blinphong() {
    double base_albedo = 0.5;
    double Ks = 0.3;
    double alpha = 0.1;
    glm::vec3 normal = {0, 0, 1};
    ModelSolver<MaterialModel::BlinPhong> solver;
    for (float elevation = 0; elevation < M_PI / 2; elevation += M_PI / 32) {
        for (float azimuth = 0; azimuth < M_PI / 2; azimuth += M_PI / 32) {
            IsotropicBRDFMeasurement m;
            m.world_coordinates = {0, 0, 0};
            m.eye_elevation = elevation;
            m.light_elevation = elevation;
            m.light_azimuth = azimuth;
            m.value = {
                ModelReflectance<MaterialModel::BlinPhong, double>::compute(
                    normal, m.light(), m.eye(), {base_albedo, Ks, alpha}),
                ModelReflectance<MaterialModel::BlinPhong, double>::compute(
                    normal, m.light(), m.eye(), {base_albedo, Ks, alpha}),
                ModelReflectance<MaterialModel::BlinPhong, double>::compute(
                    normal, m.light(), m.eye(), {base_albedo, Ks, alpha})};
            solver.add_observation(m, 1);
        }
    }
    solver.solve(true);
    std::cout << "bp.. given: " << base_albedo << " " << Ks << " " << alpha
              << std::endl;
    std::cout << "predicted: " << solver.res_data[0].albedo << " "
              << solver.res_data[0].Ks << " " << solver.res_data[0].alpha
              << std::endl;
}

void test_cooktor() {
    double base_albedo = 0.5;
    double Ks = 0.3;
    double F0 = 0.1;
    double m_base = 0.234;

    glm::vec3 normal = {0, 0, 1};
    ModelSolver<MaterialModel::CookTorrance> solver;
    for (float elevation = M_PI / 32; elevation < M_PI / 2; elevation += M_PI / 32) {
        for (float azimuth = M_PI / 32; azimuth < M_PI / 2; azimuth += M_PI / 32) {
            IsotropicBRDFMeasurement m;
            m.world_coordinates = {0, 0, 0};
            m.eye_elevation = elevation;
            m.light_elevation = elevation;
            m.light_azimuth = azimuth;
            m.value = {
                ModelReflectance<MaterialModel::CookTorrance, double>::compute(
                    normal, m.light(), m.eye(), {base_albedo, Ks, F0, m_base}),
                ModelReflectance<MaterialModel::CookTorrance, double>::compute(
                    normal, m.light(), m.eye(), {base_albedo, Ks, F0, m_base}),
                ModelReflectance<MaterialModel::CookTorrance, double>::compute(
                    normal, m.light(), m.eye(), {base_albedo, Ks, F0, m_base})};
            solver.add_observation(m, 1);
        }
    }
    solver.solve(true);
    std::cout << "ct.. given: " << base_albedo << " " << Ks << " " << F0 << " "
              << m_base << std::endl;
    std::cout << "predicted: " << solver.res_data[0].albedo << " "
              << solver.res_data[0].Ks << " " << solver.res_data[0].F0 << " "
              << solver.res_data[0].m << std::endl;
}

int main() {
    test_lambert();
    test_blinphong();
    test_cooktor();
    return 0;
}

// |\
// --
// |\|
