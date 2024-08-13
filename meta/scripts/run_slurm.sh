#!/bin/bash

#SBATCH --job-name=scaling
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=bosch_gpu-rtx2080
#SBATCH --output=/work/dlclarge1/mallik-scaling/scaling_exps/slurm_logs/%j.out
#SBATCH --error=/work/dlclarge1/mallik-scaling/scaling_exps/slurm_logs/%j.err
#SBATCH --time=0-12:00:00

source $HOME/scaling.sh

cd /work/dlclarge1/mallik-scaling/scaling_all_the_way

# example usage: sbatch run_slurm.sh scale_width_base
bash /work/dlclarge1/mallik-scaling/scaling_exps/meta/scripts/simple_run_wikitext.sh $1

# end of file