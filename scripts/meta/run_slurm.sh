#!/bin/bash

#SBATCH --job-name=scaling
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=bosch_gpu-rtx2080
#SBATCH --output=/work/dlclarge1/mallik-warmstarting/warmstarting_exps/slurm_logs/%j.out
#SBATCH --error=/work/dlclarge1/mallik-warmstarting/warmstarting_exps/slurm_logs/%j.err
#SBATCH --time=0-12:00:00

source /work/dlclarge1/mallik-warmstarting/misc/setup_script.sh

# run your script/command here

# end of file