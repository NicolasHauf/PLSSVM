/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Nicolas Hauf
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenMP/svm_kernel.hpp"

#include "plssvm/constants.hpp"     // plssvm::kernel_index_type, plssvm::OPENMP_BLOCK_SIZE
#include "plssvm/kernel_types.hpp"  // plssvm::kernel_type, plssvm::kernel_function

#include <utility>  // std::forward
#include <vector>   // std::vector

#include <mpi.h> // parallelization using mpi

namespace plssvm::openmp {

namespace detail {

template <kernel_type kernel, typename real_type, typename... Args>
void device_kernel(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const real_type add, const std::vector<std::vector<std::vector<int>>> &bounds, Args &&...args) {
    const auto dept = static_cast<kernel_index_type>(d.size());

    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Datatype mpi_real_type;
    MPI_Type_match_size(MPI_TYPECLASS_REAL, sizeof(real_type), &mpi_real_type);

    // only root rank remembers the right side of the equation
    if (rank != 0) {
        std::fill(ret.begin(), ret.end(), real_type{ 0.0 });
    }

    // each thread computes its box, defined by the bounds vector, as well as small rest bits if the computation can't be devided optimally
    for (int bounds_set = 1; bounds_set < int(bounds[rank].size()); bounds_set++) {
        std::vector<int> current_bounds = bounds[rank][bounds_set];
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (kernel_index_type i = current_bounds[0]; i < current_bounds[1]; i += OPENMP_BLOCK_SIZE) {
            for (kernel_index_type j = current_bounds[2]; j < current_bounds[3]; j += OPENMP_BLOCK_SIZE) {
                for (kernel_index_type ii = 0; ii < OPENMP_BLOCK_SIZE && ii + i < current_bounds[1]; ++ii) {
                    real_type ret_iii = 0.0;
                    if (ii + i < dept) {
                        for (kernel_index_type jj = 0; jj < OPENMP_BLOCK_SIZE && jj + j < std::min(ii + i + 1, current_bounds[3]); ++jj) {
                            if (i + ii >= j + jj && j + jj < dept) {
                                const real_type temp = (kernel_function<kernel>(data[ii + i], data[jj + j], std::forward<Args>(args)...) + QA_cost - q[ii + i] - q[jj + j]) * add;
                                if (ii + i == jj + j) {
                                    ret_iii += (temp + cost * add) * d[ii + i];
                                } else {
                                    ret_iii += temp * d[jj + j];
                                    #pragma omp atomic
                                    ret[jj + j] += temp * d[ii + i];
                                }
                            }
                        }
                        #pragma omp atomic
                        ret[ii + i] += ret_iii;
                    }
                }
            }
        }
    }

    // sending all the results to the root thread, since the entire vector is needed for the next computations in (open mp/gpu)_csvm
    if (rank != 0) {
        MPI_Send(&ret[0], ret.size(), mpi_real_type, 0, 1, MPI_COMM_WORLD);
    } else {
        MPI_Status status;
        std::vector<real_type> temp_ret(ret.size());
        for (int i = 1; i < world_size; ++i) {
            MPI_Recv(&temp_ret[0], temp_ret.size(), mpi_real_type, i, 1, MPI_COMM_WORLD, &status);
            for (typename std::vector<real_type>::size_type j = 0; j < temp_ret.size(); ++j) {
                ret[j] += temp_ret[j];
            }
        }
    }
}

}  // namespace detail

template <typename real_type>
void device_kernel_linear(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const real_type add, const std::vector<std::vector<std::vector<int>>> &bounds) {
    detail::device_kernel<kernel_type::linear>(q, ret, d, data, QA_cost, cost, add, bounds);
}
template void device_kernel_linear(const std::vector<float> &, std::vector<float> &, const std::vector<float> &, const std::vector<std::vector<float>> &, const float, const float, const float, const std::vector<std::vector<std::vector<int>>> &);
template void device_kernel_linear(const std::vector<double> &, std::vector<double> &, const std::vector<double> &, const std::vector<std::vector<double>> &, const double, const double, const double, const std::vector<std::vector<std::vector<int>>> &);

template <typename real_type>
void device_kernel_poly(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const real_type add, const std::vector<std::vector<std::vector<int>>> &bounds, const int degree, const real_type gamma, const real_type coef0) {
    detail::device_kernel<kernel_type::polynomial>(q, ret, d, data, QA_cost, cost, add, bounds, degree, gamma, coef0);
}
template void device_kernel_poly(const std::vector<float> &, std::vector<float> &, const std::vector<float> &, const std::vector<std::vector<float>> &, const float, const float, const float, const std::vector<std::vector<std::vector<int>>> &, const int, const float, const float);
template void device_kernel_poly(const std::vector<double> &, std::vector<double> &, const std::vector<double> &, const std::vector<std::vector<double>> &, const double, const double, const double, const std::vector<std::vector<std::vector<int>>> &, const int, const double, const double);

template <typename real_type>
void device_kernel_radial(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const real_type add, const std::vector<std::vector<std::vector<int>>> &bounds, const real_type gamma) {
    detail::device_kernel<kernel_type::rbf>(q, ret, d, data, QA_cost, cost, add, bounds, gamma);
}
template void device_kernel_radial(const std::vector<float> &, std::vector<float> &, const std::vector<float> &, const std::vector<std::vector<float>> &, const float, const float, const float, const std::vector<std::vector<std::vector<int>>> &, const float);
template void device_kernel_radial(const std::vector<double> &, std::vector<double> &, const std::vector<double> &, const std::vector<std::vector<double>> &, const double, const double, const double, const std::vector<std::vector<std::vector<int>>> &, const double);

}  // namespace plssvm::openmp
