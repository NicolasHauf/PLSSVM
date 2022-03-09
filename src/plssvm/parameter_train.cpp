/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/parameter_train.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::as_lower_case
#include "plssvm/detail/utility.hpp"         // plssvm::detail::to_underlying

#include "cxxopts.hpp"    // cxxopts::Options, cxxopts::value,cxxopts::ParseResult
#include "fmt/core.h"     // fmt::print, fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <cstdio>     // stderr
#include <cstdlib>    // std::exit, EXIT_SUCCESS, EXIT_FAILURE
#include <exception>  // std::exception
#include <string>     // std::string
#include <utility>    // std::move

#include <mpi.h>

namespace plssvm {

void compute_bounds(int n, int world_size, std::vector<std::vector<std::vector<int>>> &ret) {
    
    int u = floor(sqrt(2 * world_size + 0.25) - 0.5) + 1;
    int box_size = ceil(n / static_cast<float>(u)); 

    int cur_thread = 0;
   
    for (int j = 0; j < u - 1; j++) {
        for (int i = j + 1; i < u; i++) {
            ret.push_back(std::vector<std::vector<int>>{
                { box_size * i, box_size * (i + 1), box_size * j, box_size * (j + 1) },
                { box_size * i, box_size * (i + 1), box_size * j, box_size * (j + 1) }
            });
            cur_thread++;
        }
    }

    for (int ij = 0; ij < u; ij += 2) {
        if (cur_thread > world_size - 1) {
            ret[cur_thread % world_size].push_back(std::vector<int>{ box_size * ij, box_size * (ij + 1), box_size * ij, box_size * (ij + 1) });
            ret[cur_thread % world_size].push_back(std::vector<int>{ box_size * (ij + 1), box_size * (ij + 2), box_size * (ij + 1), box_size * (ij + 2) });
            ret.push_back(std::vector<std::vector<int>>{
                { box_size * ij, box_size * (ij + 2), 0, 0 } 
            });
        } else {
            ret.push_back(std::vector<std::vector<int>>{
                { box_size * ij, box_size * (ij + 2), 0, 0 },
                { box_size * ij, box_size * (ij + 1), box_size * ij, box_size * (ij + 1) },
                { box_size * (ij + 1), box_size * (ij + 2), box_size * (ij + 1), box_size * (ij + 2) } });
        }
        cur_thread++;
    }

    for (cur_thread; cur_thread < world_size; cur_thread++) {
        ret.push_back(std::vector<std::vector<int>>{
            { 0, 0, 0, 0 }
        });
    }
}



template <typename T>
parameter_train<T>::parameter_train(std::string p_input_filename) {
    base_type::input_filename = std::move(p_input_filename);
    base_type::model_filename = base_type::model_name_from_input();

    base_type::parse_train_file(input_filename);
}

template <typename T>
parameter_train<T>::parameter_train(int argc, char **argv) {
    cxxopts::Options options(argv[0], "LS-SVM with multiple (GPU-)backends");
    options
        .positional_help("training_set_file [model_file]")
        .show_positional_help();
    options
        .set_width(150)
        .set_tab_expansion()
        // clang-format off
        .add_options()
            ("t,kernel_type", "set type of kernel function. \n\t 0 -- linear: u'*v\n\t 1 -- polynomial: (gamma*u'*v + coef0)^degree \n\t 2 -- radial basis function: exp(-gamma*|u-v|^2)", cxxopts::value<decltype(kernel)>()->default_value(fmt::format("{}", detail::to_underlying(kernel))))
            ("d,degree", "set degree in kernel function", cxxopts::value<decltype(degree)>()->default_value(fmt::format("{}", degree)))
            ("g,gamma", "set gamma in kernel function (default: 1 / num_features)", cxxopts::value<decltype(gamma)>())
            ("r,coef0", "set coef0 in kernel function", cxxopts::value<decltype(coef0)>()->default_value(fmt::format("{}", coef0)))
            ("c,cost", "set the parameter C", cxxopts::value<decltype(cost)>()->default_value(fmt::format("{}", cost)))
            ("e,epsilon", "set the tolerance of termination criterion", cxxopts::value<decltype(epsilon)>()->default_value(fmt::format("{}", epsilon)))
            ("b,backend", "choose the backend: openmp|cuda|opencl|sycl", cxxopts::value<decltype(backend)>()->default_value(detail::as_lower_case(fmt::format("{}", backend))))
            ("p,target_platform", "choose the target platform: automatic|cpu|gpu_nvidia|gpu_amd|gpu_intel", cxxopts::value<decltype(target)>()->default_value(detail::as_lower_case(fmt::format("{}", target))))
            ("q,quiet", "quiet mode (no outputs)", cxxopts::value<bool>(print_info)->default_value(fmt::format("{}", !print_info)))
            ("h,help", "print this helper message", cxxopts::value<bool>())
            ("input", "", cxxopts::value<decltype(input_filename)>(), "training_set_file")
            ("model", "", cxxopts::value<decltype(model_filename)>(), "model_file");
    // clang-format on

    // parse command line options
    cxxopts::ParseResult result;
    try {
        options.parse_positional({ "input", "model" });
        result = options.parse(argc, argv);
    } catch (const std::exception &e) {
        fmt::print("{}\n{}\n", e.what(), options.help());
        std::exit(EXIT_FAILURE);
    }

    // print help message and exit
    if (result.count("help")) {
        fmt::print("{}", options.help());
        std::exit(EXIT_SUCCESS);
    }

    // parse kernel_type and cast the value to the respective enum
    kernel = result["kernel_type"].as<decltype(kernel)>();

    // parse degree
    degree = result["degree"].as<decltype(degree)>();

    // parse gamma
    if (result.count("gamma")) {
        gamma = result["gamma"].as<decltype(gamma)>();
        if (gamma == decltype(gamma){ 0.0 }) {
            fmt::print(stderr, "gamma = 0.0 is not allowed, it doesnt make any sense!\n");
            fmt::print("{}", options.help());
            std::exit(EXIT_FAILURE);
        }
    } else {
        gamma = decltype(gamma){ 0.0 };
    }

    // parse coef0
    coef0 = result["coef0"].as<decltype(coef0)>();

    // parse cost
    cost = result["cost"].as<decltype(cost)>();

    // parse epsilon
    epsilon = result["epsilon"].as<decltype(epsilon)>();

    // parse backend_type and cast the value to the respective enum
    backend = result["backend"].as<decltype(backend)>();

    // parse target_platform and cast the value to the respective enum
    target = result["target_platform"].as<decltype(target)>();

    // parse print info
    print_info = !print_info;

    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    print_info = print_info && (rank == 0);

    // parse input data filename
    if (!result.count("input")) {
        fmt::print(stderr, "Error missing input file!");
        fmt::print("{}", options.help());
        std::exit(EXIT_FAILURE);
    }
    input_filename = result["input"].as<decltype(input_filename)>();

    // parse output model filename
    if (result.count("model")) {
        model_filename = result["model"].as<decltype(model_filename)>();
    } else {
        model_filename = base_type::model_name_from_input();
    }

    int n = 0;

    if (rank == 0) {
        base_type::parse_train_file(input_filename);
        n = data_ptr->size() - 1;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    std::vector<std::vector<std::vector<int>>> bounds;

    compute_bounds(n, world_size, bounds);

    bounds_ptr = std::make_shared<const std::vector<std::vector<std::vector<int>>>>(std::move(bounds));
}

// explicitly instantiate template class
template class parameter_train<float>;
template class parameter_train<double>;

}  // namespace plssvm