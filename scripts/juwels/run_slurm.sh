#!/bin/bash

#SBATCH --account=projectnucleus
#SBATCH --partition=booster
#SBATCH --job-name=booster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/p/project1/projectnucleus/mallik2/warmstarting/warmstarting_exps/slurm_logs/booster%j.out
#SBATCH --error=/p/project1/projectnucleus/mallik2/warmstarting/warmstarting_exps/slurm_logs/booster%j.err
#SBATCH --time=00:50:00


source /p/project1/projectnucleus/mallik2/warmstarting/misc/setup_script.sh nyoy-env

# run your script/command here
my_script() {
    bash /p/project1/projectnucleus/mallik2/warmstarting/warmstarting_exps/scripts/simple_run_wikitext.sh
}

time my_script
# end of file