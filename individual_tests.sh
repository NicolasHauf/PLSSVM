#!/bin/bash
cd build
for i in {1..48}
do
    mpirun -n 1 ctest -I $i,$i
done
cd ..
