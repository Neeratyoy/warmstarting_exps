cd /work/dlclarge1/mallik-warmstarting/scale_and_warmstart/

python examples/run_configs.py \
    --data_config_path /work/dlclarge1/mallik-warmstarting/warmstarting_exps/configs/data_handlers/wikitext_data_handler.yaml \
    --train_config_path /work/dlclarge1/mallik-warmstarting/warmstarting_exps/configs/train_template_mup.yaml \
    --output_dir /work/dlclarge1/mallik-warmstarting/warmstarting_exps/results/test_run_mup/
# end of file