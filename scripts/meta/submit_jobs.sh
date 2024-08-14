#!/bin/bash/

# for jobname in $(ls /work/dlclarge1/mallik-scaling/trial_runs/configs/ | grep train | sed 's/train_//g' | sed 's/.yaml//g')

for jobname in scale_width_base \
               scale_width_target \
               scale_depth_base \
               scale_depth_target \
               scale_compound_base \
               scale_compound_target \
               scale_width_target2 \
               scale_depth_target2 \
               scale_compound_target2
do
    jobID=$(
        sbatch /work/dlclarge1/mallik-warmstarting/warmstarting_exps/meta/scripts/run_slurm.sh ${jobname} <<< y \
        | awk '{print $4}'
    )
    echo $jobname": "$jobID
    echo 
done
