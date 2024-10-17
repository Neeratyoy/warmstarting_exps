#!/bin/bash

base_path=$(dirname "$0")  # reads the file's parent directory

dataset="wikitext"
name="timur"  # load the relevant canvas settings
exp_tree="test_run/wikitext/warm_snp"  # change the experiment subdirectory name

python ${base_path}"/../../warms/run_main.py" \
    --canvas_access $name \
    --output_tree $exp_tree \
    --dataset $dataset \
    # add the following line to override canvas.train_template
    # --train_config_path $1
# end of file