# run this application before running the jobscript via sbatch

import os
import math

# creates a jobscript for the given parameter(s)
def create_js_for_index(i):
    if i > 15*8 or i < 1:
        print("index too large or too small(>15*8, <1)")
        return
    f = open("var_jobscript.sh", "w")
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --job-name=PLSSVM\n") # change the job name if needed
    f.write("#SBATCH --output=results/improve50_new/{}_10k.out\n".format(i)) # change output folder and file name to fit the request
    f.write("#SBATCH --time=05:00:00\n")
    f.write("#SBATCH --exclusive")

    # change overall tasks and tasks per node to fit the request
    f.write("#SBATCH --ntasks-per-node=8\n") 
    f.write("#SBATCH --nodes={}\n\n".format(math.ceil(i/8))) 
    f.write("module load openmpi/4.0.4-gcc-10.2\n\n")
    f.write("srun -n {} ./../build/svm-train --input ../data/50k/data_file.libsvm -e 0.000001\n\n".format(i))
    f.write("../svm-predict --model ../data_file.libsvm.model --test ../data/10k/data_file_test.libsvm")
    f.close()


# wait for input of parameters and save the name of the output file and later job ID to easily retrieve the slurm data after the job is finished
# inputs: x,y -> sbatch files for x to y-1 (x and y should be integers)
#         x   -> sbatch file for x
#         'q' -> quit the application
times_id = dict()
s = ""
while s != "q":        
    s = input()
    if len(s.split(",")) != 2 and len(s.split(",")) != 1:
        print("false input")
    if s == "q":
        pass
    elif len(s.split(",")) == 2:
        for i in range(int(s.split(",")[0]), int(s.split(",")[1])):
            create_js_for_index(i)
            os.system("echo {}".format(i))
            os.system("echo {}_10k.out >> current_job_id".format(i))
    else:
        create_js_for_index(int(s))
        os.system("echo {}_10k.out >> current_job_id".format(int(s)))

