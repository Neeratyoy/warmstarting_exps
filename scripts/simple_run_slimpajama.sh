#!/bin/bash

base_path=$(dirname "$0")  # reads the file's parent directory

dataset="slimpajama"
name="neeratyoy"  # load the relevant canvas settings
exp_tree="test_run/slimpajama/ddp"  # change the experiment subdirectory name

python ${base_path}"warms/run_mod_template.py" \
    --canvas_access $name \
    --output_tree $exp_tree \
    --dataset $dataset \
    --warmstart \
    --warmstart_base_path "/work/dlclarge1/mallik-warmstarting/warmstarting_exps/results/icml/base/s0/seed=444" \
    --target_scale "/work/dlclarge1/mallik-warmstarting/misc/neeratyoy/code/warmstarting_exps/configs/width_num_heads/dev/width-num-heads_block=1024_depth=8_scale4.yaml" \
    --base_lr 0.003 \
    --micro_batch_size 64 \
    --slurm_partition l40 \
    --mup_base "/work/dlclarge1/mallik-warmstarting/misc/neeratyoy/code/warmstarting_exps/configs/width_num_heads/dev/mup/width-num-heads_block=1024_depth=8_scale0.bsh" #\
    # --ddp /
    # --ddp_strategy "ddp"
# end of file