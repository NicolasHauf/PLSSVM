/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Main function compiled to the `svm-train` executable used for training a C-SVM model.
 */

#include "plssvm/core.hpp"

#include <exception>  // std::exception
#include <iostream>   // std::cerr, std::endl

#include <mpi.h>

// perform calculations in single precision if requested
#ifdef PLSSVM_EXECUTABLES_USE_SINGLE_PRECISION
using real_type = float;
#else
using real_type = double;
#endif

int main(int argc, char *argv[]) {
    try {

        MPI_Init(&argc, &argv);

        int rank, world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);


        // parse SVM parameter from command line
        plssvm::parameter_train<real_type> params{ argc, argv };

        // create SVM
        auto svm = plssvm::make_csvm(params);
        

        MPI_Barrier(MPI_COMM_WORLD);
        // learn
        svm->learn();

        if (rank == 0) {
            // save model file
            svm->write_model(params.model_filename);
        }
        MPI_Finalize();

    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}