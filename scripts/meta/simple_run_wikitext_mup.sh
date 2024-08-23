#!/bin/bash

base_path=$(dirname "$0")  # reads the file's parent directory

dataset="wikitext"
name="global"  # load the relevant canvas settings
exp_tree="test_run/mup"  # change the experiment subdirectory name

# edit the train_config_path here to run a mup-enabled training configuration
train_config_path=${base_path}"/../../configs/train_template_mup.yaml" 

python ${base_path}"/../../warms/run_main.py" \
    --canvas_access $name \
    --output_tree $exp_tree \
    --dataset $dataset \
    --train_config_path $train_config_path  # override canvas.train_template
# end of file]