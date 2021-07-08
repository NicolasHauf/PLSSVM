#include "plssvm/CSVM.hpp"

#include "plssvm/detail/operators.hpp"

#include "fmt/chrono.h"  // format std::chrono
#include "fmt/core.h"    // fmt::print

#include <cassert>  // assert
#include <chrono>   // std::chrono::stead_clock, std::chrono::duration_cast, std::chrono::milliseconds
#include <cmath>    // std::pow, std::exp
#include <string>   // std::string
#include <vector>   // std::vector

namespace plssvm {

template <typename T>
CSVM<T>::CSVM(parameter<T> &params) :
    CSVM{ params.kernel, params.degree, params.gamma, params.coef0, params.cost, params.epsilon, params.print_info } {}

template <typename T>
CSVM<T>::CSVM(kernel_type kernel, real_type degree, real_type gamma, real_type coef0, real_type cost, real_type epsilon, bool print_info) :
    kernel_{ kernel }, degree_{ degree }, gamma_{ gamma }, coef0_{ coef0 }, cost_{ cost }, epsilon_{ epsilon }, print_info_{ print_info } {}

template <typename T>
void CSVM<T>::learn() {
    auto start_time = std::chrono::steady_clock::now();

    std::vector<real_type> q;
    std::vector<real_type> b = value_;
    #pragma omp parallel sections
    {
        #pragma omp section  // generate q
        {
            q = generate_q();
        }
        #pragma omp section  // generate right-hand side from equation
        {
            b.pop_back();
            b -= value_.back();
        }
        #pragma omp section  // generate bottom right from A
        {
            QA_cost_ = kernel_function(data_.back(), data_.back()) + 1 / cost_;
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    if (print_info_) {
        fmt::print("Setup for solving the optimization problem done in {}.\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }

    start_time = std::chrono::steady_clock::now();

    // solve minimization
    alpha_ = solver_CG(b, num_features_, epsilon_, q);
    alpha_.emplace_back(-sum(alpha_));
    bias_ = value_.back() - QA_cost_ * alpha_.back() - (q * alpha_);

    end_time = std::chrono::steady_clock::now();
    if (print_info_) {
        fmt::print("Solved minimization problem (r = b - Ax) using CG in {}.\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }
}
// TODO: kernel function OpenMP <-> CUDA <-> CSVM
template <typename T>
auto CSVM<T>::kernel_function(const real_type *xi, const real_type *xj, const size_type dim) -> real_type {
    switch (kernel_) {
        case kernel_type::linear:
            return mult(xi, xj, dim);
        case kernel_type::polynomial:
            return std::pow(gamma_ * mult(xi, xj, dim) + coef0_, degree_);
        case kernel_type::rbf: {
            real_type temp = 0;
            for (size_type i = 0; i < dim; ++i) {
                temp += xi[i] - xj[i];
            }
            return std::exp(-gamma_ * temp * temp);
        }
        default:
            throw std::runtime_error{ "Can not decide which kernel!" };  // TODO: change to custom exception?
    }
}

template <typename T>
auto CSVM<T>::kernel_function(const std::vector<real_type> &xi, const std::vector<real_type> &xj) -> real_type {
    assert((xi.size() == xj.size()) && "Sizes in kernel function mismatch!");
    return kernel_function(xi.data(), xj.data(), xi.size());
}

template <typename T>
void CSVM<T>::learn(const std::string &input_filename, const std::string &model_filename) {
    // parse data file
    parse_file(input_filename);

    // setup the data on the device
    setup_data_on_device();

    // learn model
    learn();

    // write results to model file
    write_model(model_filename);

    //    if (true) {  // TODO: check
    //        std::clog << data.size() << ", " << num_features << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(end_parse - begin_parse).count() << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - end_parse).count() << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(end_learn - end_gpu).count() << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(end_write - end_learn).count() << std::endl;
    //    }
}

template <typename T>
auto CSVM<T>::transform_data(const size_type boundary) -> std::vector<real_type> {
    auto start_time = std::chrono::steady_clock::now();

    std::vector<real_type> vec(num_features_ * (num_data_points_ - 1 + boundary));
    #pragma omp parallel for collapse(2)
    for (size_type col = 0; col < num_features_; ++col) {
        for (size_type row = 0; row < num_data_points_ - 1; ++row) {
            vec[col * (num_data_points_ - 1 + boundary) + row] = data_[row][col];
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    if (print_info_) {
        fmt::print("Transformed dataset from 2D AoS to 1D SoA in {}.\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }
    return vec;
}

// explicitly instantiate template class
template class CSVM<float>;
template class CSVM<double>;

}  // namespace plssvm