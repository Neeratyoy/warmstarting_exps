python /work/dlclarge1/mallik-warmstarting/scale_and_warmstart/saws/plot_results.py --results_dirs \
    /work/dlclarge1/mallik-warmstarting/warmstarting_exps/results/shitty_draft/slimpajama/block=1024_depth=6/baseline/mup/constLR/scale0_scale4 \
    /work/dlclarge1/mallik-warmstarting/warmstarting_exps/results/shitty_draft/slimpajama/block=1024_depth=6/baseline/sp/constLR/scale4 \
    --run_names mup-0->4 sp-0->4 \
    --x_axes step flops \
    --smoothing \
    --output_dir /work/dlclarge1/mallik-warmstarting/warmstarting_exps/results/shitty_draft/slimpajama/block=1024_depth=6/figures/test/
