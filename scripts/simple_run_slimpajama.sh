#!/bin/bash

dataset="slimpajama"
name="neeratyoy"  # load the relevant canvas settings
_exp_tree="test_run/estimates"  # change the experiment subdirectory name

source /work/dlclarge1/mallik-warmstarting/misc/setup_script.sh nyoy_env /work/dlclarge1/mallik-warmstarting/envs/

target_scale_prefix="width-num-heads_block=1024_depth=8"
target_scale=4
exp_tree=${_exp_tree}"/"${target_scale}

# check path for cwd
cd /work/dlclarge1/mallik-warmstarting/misc/neeratyoy/code/warmstarting_exps

python "./warms/run_mod_template.py" \
    --canvas_access $name \
    --output_tree $exp_tree \
    --dataset $dataset \
    --warmstart \
    --warmstart_base_path "/work/dlclarge1/mallik-warmstarting/warmstarting_exps/results/icml/base/s0/seed=444" \
    --ddp
# end of file