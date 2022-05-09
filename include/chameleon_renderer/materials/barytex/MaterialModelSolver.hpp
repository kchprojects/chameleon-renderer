#pragma once
#include <ceres/ceres.h>

#include <chameleon_renderer/materials/barytex/MaterialModel.hpp>
#include <chameleon_renderer/utils/debug_io.hpp>
namespace chameleon {

template <typename DERIVED>
struct ModelSolverBase {
    static constexpr MaterialModel model_t = DERIVED::model_t;
    ceres::Solver::Options options;
    std::array<ceres::Problem, 3> problems;

    ModelSolverBase() {
        options.max_num_iterations = 50;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = false;
    };

    void solve(bool with_output = false, bool with_report = false) {
        options.minimizer_progress_to_stdout = with_output;
        ceres::Solver::Summary summary;
        for (int i = 0; i < 3; ++i) {
            ceres::Solve(options, &problems[i], &summary);
        }
        if (with_report) {
            std::cout << summary.BriefReport() << "\n";
        }
    }
};

template <MaterialModel model_t>
struct ModelSolver;

template <>
struct ModelSolver<MaterialModel::Lambert>
    : public ModelSolverBase<ModelSolver<MaterialModel::Lambert>> {
    static constexpr MaterialModel model_t = MaterialModel::Lambert;
    std::array<ModelData<model_t, double>, 3> res_data;

    ModelSolver() : ModelSolverBase<ModelSolver<model_t>>() {
        for (int i = 0; i < 3; ++i) {
            problems[i].AddParameterBlock(&res_data[i].albedo, 1);
            problems[i].SetParameterLowerBound(&res_data[i].albedo, 0, 0.0);
        }
    }

    struct Residual {
        Residual(IsotropicBRDFMeasurement measurement, glm::vec3 y_vec,
                 double weight, int index)
            : _measurement(std::move(measurement)),
              _y(y_vec[index]),
              weight(weight),
              index(index) {}

        template <typename T>
        bool operator()(const T* const albedo, T* residual) const {
            // print_vec(_measurement.light(), "light");
            // print_vec(_measurement.eye(), "eye");
            // print_var(*albedo, "albedo");
            // print_var(_y, "y");
            // print_var(weight, "weight");
            auto predicted_y = ModelReflectance<model_t, T>::compute(
                glm::vec3(0, 0, 1), _measurement.light(), _measurement.eye(),
                ModelData<model_t, T>{*albedo});
            // print_var(predicted_y, "predicted");
            *residual = (predicted_y - T(_measurement.value[index])) *
                        (predicted_y - T(_measurement.value[index])) *
                        T(weight);
            return true;
        }

       private:
        const IsotropicBRDFMeasurement _measurement;
        const double _y;
        const double weight;
        const int index;
    };

    void add_observation(const IsotropicBRDFMeasurement& measurement,
                         double weight = 1) {
        for (int i = 0; i < 3; ++i) {
            problems[i].AddResidualBlock(
                new ceres::AutoDiffCostFunction<Residual, 1, 1>(
                    new Residual(measurement, measurement.value, weight, i)),
                nullptr, &res_data[i].albedo);
        }
    }
};

template <>
struct ModelSolver<MaterialModel::BlinPhong>
    : public ModelSolverBase<ModelSolver<MaterialModel::BlinPhong>> {
    static constexpr MaterialModel model_t = MaterialModel::BlinPhong;
    std::array<ModelData<model_t, double>, 3> res_data;

    ModelSolver() : ModelSolverBase<ModelSolver<model_t>>() {
        for (int i = 0; i < 3; ++i) {
            problems[i].AddParameterBlock(&res_data[i].albedo, 1);
            problems[i].AddParameterBlock(&res_data[i].Ks, 1);
            problems[i].AddParameterBlock(&res_data[i].alpha, 1);
            problems[i].SetParameterLowerBound(&res_data[i].albedo, 0, 0.0);
            problems[i].SetParameterLowerBound(&res_data[i].Ks, 0, 0.0);
            problems[i].SetParameterLowerBound(&res_data[i].alpha, 0, 0.0);
            
        }
    }

    struct Residual : ceres::SizedCostFunction<1, 1, 1, 1> {
        Residual(IsotropicBRDFMeasurement measurement, glm::vec3 y_vec,
                 double weight, int index)
            : _measurement(std::move(measurement)),
              _y(y_vec[index]),
              weight(weight),
              index(index) {}

        template <typename T>
        T compute(const T albedo, const T Ks, const T alpha) const {
            return ModelReflectance<model_t, T>::compute(
                glm::vec3(0, 0, 1), _measurement.light(), _measurement.eye(),
                ModelData<model_t, T>{albedo, Ks, alpha});
        }

        template <typename T>
        bool compute_derivatives(const T albedo, const T Ks, const T alpha,
                                 T** jacobians) const {
            jacobians[0][0] = _measurement.light().z;
            glm::vec3 H =
                glm::normalize(_measurement.light() + _measurement.eye());
            auto norm_dot = glm::dot(H, glm::vec3(0, 0, 1));
            if(norm_dot < 0.001){
                return false;
            }
            T spec_pow_alpha = std::pow(norm_dot, alpha);
            jacobians[0][1] = spec_pow_alpha;
            jacobians[0][2] = Ks * spec_pow_alpha * std::log(norm_dot);

            return true;
        }

        bool Evaluate(double const* const* parameters, double* residuals,
                      double** jacobians) const override {
            double predicted_y =
                compute(parameters[0][0], parameters[0][1], parameters[0][2]);

            double res =
                std::pow(predicted_y - _measurement.value[index], 2.0) * weight;
            residuals[0] = res;
            if (jacobians != nullptr && jacobians[0] != nullptr) {
                bool status = compute_derivatives(parameters[0][0], parameters[0][1],
                                    parameters[0][2], jacobians);
                if(! status){
                    return false;
                }
                jacobians[0][0] *= 2 * (predicted_y - _y) * weight;
                jacobians[0][1] *= 2 * (predicted_y - _y) * weight;
                jacobians[0][2] *= 2 * (predicted_y - _y) * weight;
            }
            return true;
        }

       private:
        const IsotropicBRDFMeasurement _measurement;
        const double _y;
        const double weight;
        const int index;
    };

    void add_observation(const IsotropicBRDFMeasurement& measurement,
                         double weight = 1) {
        // for (int i = 0; i < 3; ++i) {
        //     problems[i].AddResidualBlock(
        //         new ceres::AutoDiffCostFunction<Residual, 1, 1, 1, 1>(
        //             new Residual(measurement, measurement.value, weight, i)),
        //         nullptr, &res_data[i].albedo, &res_data[i].Ks,
        //         &res_data[i].alpha);
        // }

        for (int i = 0; i < 3; ++i) {
            problems[i].AddResidualBlock(
                new Residual(measurement, measurement.value, weight, i),
                nullptr, &res_data[i].albedo, &res_data[i].Ks,
                &res_data[i].alpha);
        }
    }
};  // namespace chameleon

template <>
struct ModelSolver<MaterialModel::CookTorrance>
    : public ModelSolverBase<ModelSolver<MaterialModel::CookTorrance>> {
    static constexpr MaterialModel model_t = MaterialModel::CookTorrance;
    std::array<ModelData<model_t, double>, 3> res_data;

    ModelSolver() : ModelSolverBase<ModelSolver<model_t>>() {
        for (int i = 0; i < 3; ++i) {
            problems[i].AddParameterBlock(&res_data[i].albedo, 1);
            problems[i].AddParameterBlock(&res_data[i].Ks, 1);
            problems[i].AddParameterBlock(&res_data[i].F0, 1);
            problems[i].AddParameterBlock(&res_data[i].m, 1);
            problems[i].SetParameterLowerBound(&res_data[i].albedo, 0, 0.0);
            problems[i].SetParameterLowerBound(&res_data[i].F0, 0, 0.0);
            problems[i].SetParameterLowerBound(&res_data[i].Ks, 0, 0.0);
            problems[i].SetParameterLowerBound(&res_data[i].m, 0, 0.0001);
        }
    }

    struct Residual {
        Residual(IsotropicBRDFMeasurement measurement, glm::vec3 y_vec,
                 double weight, int index)
            : _measurement(std::move(measurement)),
              _y(y_vec[index]),
              weight(weight),
              index(index) {}

        template <typename T>
        bool operator()(const T* const albedo, const T* const Ks,
                        const T* const F0, const T* const m,
                        T* residual) const {
            auto predicted_y = ModelReflectance<model_t, T>::compute(
                glm::vec3(0, 0, 1), _measurement.light(), _measurement.eye(),
                ModelData<model_t, T>{*albedo, *Ks, *F0, *m});
            *residual =
                pow(predicted_y - T(_measurement.value[index]), T(2.0)) *
                T(weight);
            return true;
        }

       private:
        const IsotropicBRDFMeasurement _measurement;
        const double _y;
        const double weight;
        const int index;
    };

    void add_observation(const IsotropicBRDFMeasurement& measurement,
                         double weight = 1) {
        for (int i = 0; i < 3; ++i) {
            problems[i].AddResidualBlock(
                new ceres::AutoDiffCostFunction<Residual, 1, 1, 1, 1, 1>(
                    new Residual(measurement, measurement.value, weight, i)),
                nullptr, &res_data[i].albedo, &res_data[i].Ks, &res_data[i].F0,
                &res_data[i].m);
        }
    }
};

}  // namespace chameleon
