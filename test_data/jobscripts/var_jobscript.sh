#!/bin/bash
#SBATCH --job-name=PLSSVM
#SBATCH --output=results/improve100_1000/100_1000.out
#SBATCH --time=2-23:00:00
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=13

module load openmpi/4.0.4-gcc-10.2

srun -n 100 ./../build/svm-train --input ../data/100k/1000f/data_file.libsvm -e 1

