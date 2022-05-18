# run this application after the job is done

import os

line_adress, cur_adress = True, ''
# open the log file to read the job id and test result file
# append the sacct output into the test result file
# output the required statistics -> job ID, RSS, runtime, virtual machine size
f_jobs = open("current_job_id")
for line in f_jobs.readlines():
    if line_adress:
        cur_adress = line
        line_adress = False
    else:
        os.system("sacct --format=\"JobID,AveRSS,MaxRSS,Elapsed,MaxRSSTask,AveVMSize,MaxVMSize\" -j {} >> results/improve50_new/{}".format(line.split(" ")[3][:-1], cur_adress))
        line_adress = True
f_jobs.close()

# clean the log file once the job is done
f_jobs = open("current_job_id", "w")
f_jobs.write('')
f_jobs.close()

