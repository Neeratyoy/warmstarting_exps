#!/bin/bash

#SBATCH --account=projectnucleus
#SBATCH --partition=develbooster
#SBATCH --job-name=devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/p/project1/projectnucleus/mallik2/warmstarting/warmstarting_exps/slurm_logs/devel%j.out
#SBATCH --error=/p/project1/projectnucleus/mallik2/warmstarting/warmstarting_exps/slurm_logs/devel%j.err
#SBATCH --time=00:50:00


source /p/project1/projectnucleus/mallik2/warmstarting/misc/setup_script.sh

# run your script/command here
bash /p/project1/projectnucleus/mallik2/warmstarting/warmstarting_exps/scripts/simple_run_wikitext.sh

# end of file