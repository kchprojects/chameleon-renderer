#pragma once
#include <ceres/ceres.h>

#include <chameleon_renderer/materials/barytex/MaterialModelSolver.hpp>

namespace chameleon {

template <typename DERIVED>
struct ModelSolverBaseGS {
    static constexpr MaterialModel model_t = DERIVED::model_t;
    ceres::Solver::Options options;
    ceres::Problem problem;

    ModelSolverBaseGS() {
        options.max_num_iterations = 50;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = false;
    };

    void solve(bool with_output = false, bool with_report = false) {
        options.minimizer_progress_to_stdout = with_output;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if (with_report) {
            std::cout << summary.BriefReport() << "\n";
        }
    }
};

template <MaterialModel model_t>
struct ModelSolverGS;

template <>
struct ModelSolverGS<MaterialModel::Lambert>
    : public ModelSolverBaseGS<ModelSolverGS<MaterialModel::Lambert>> {
    static constexpr MaterialModel model_t = MaterialModel::Lambert;
    std::array<ModelData<model_t, double>,1> res_data;

    ModelSolverGS() : ModelSolverBaseGS<ModelSolverGS<model_t>>() {
        problem.AddParameterBlock(&res_data[0].albedo, 1);
        problem.SetParameterLowerBound(&res_data[0].albedo, 0, 0.0);
    }
    using Residual = ModelSolver<MaterialModel::Lambert>::Residual;

    void add_observation(const IsotropicBRDFMeasurement& measurement,
                         double weight = 1) {
        IsotropicBRDFMeasurement gs_mes = measurement;
        float mean = (measurement.value.x + measurement.value.y + measurement.value.z)/3.f;
        gs_mes.value = {mean,mean,mean};
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<Residual, 1, 1>(
                new Residual(measurement, measurement.value, weight, 0)),
            nullptr, &res_data[0].albedo);
    }
};

}  // namespace chameleon
