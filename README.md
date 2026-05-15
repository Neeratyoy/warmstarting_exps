# warmstarting_exps

### Overview

This repo contains the experiment wrappers for the paper [When is Warmstarting Effective for Scaling Language Models?](https://arxiv.org/pdf/2605.13405) that jointly interface the following supporting code: 

[Warmstarting and LLM code](https://github.com/automl/scale_and_warmstart/tree/main) | [Synthetic MLP code](https://github.com/JohannesHog/mlp_scaling/tree/main) | [Scaling Law code](https://github.com/Neeratyoy/EffingScaling/tree/warmstarting26)

### Installations

For **environment setup**:
```bash
# run `conda deactivate` if conda is part of ~/.bashrc
git clone https://github.com/automl/venv_templates.git
cd venv_templates/
git checkout meta
cat README.md
# Follow the instructions in README.md
# summary example: replace `my_env` with your environment name
# NOTE: update L19 in `setup.py` to be python3/python3.10/or whatever is available/or desired
bash setup.sh my_env ../envs
source activate.sh my_env ../envs

# test
which ipython   # expected: ../envs/bin/ipython
which python  # expected: ../envs/bin/python
which python3  # expected: ../envs/bin/python3
which python3.10  # expected: ../envs/bin/python3.10
cd ..
```
**TIP**: use a `setup_script.sh` to generalize these commands and use the _environment name_ as argument


For **dependencies**:
```bash
git clone https://github.com/automl-private/scale_and_warmstart.git
cd scale_and_warmstart
pip install -e .  # or `poetry install`
cd ..

# test
python -c "import saws"
```

For **the experiment repo**:
```bash
git clone https://github.com/Neeratyoy/warmstarting_exps.git
cd warmstarting_exps
pip install -e .  # or `poetry install`
cd ..

# test
python -c "import warms"
```

### Quick run

```bash
cd warmstarting_exps/scripts/meta/
bash simple_run_wikitext.sh
```
OR
```bash
cd warmstarting_exps/

python warms/run_main.py --output_tree "test_run/sp"
```

### Quick run with muParam

```bash
cd warmstarting_exps/scripts/meta/
bash simple_run_wikitext_mup.sh
```
OR
```bash
cd warmstarting_exps/

python warms/run_main.py \
    --output_tree "test_run/mup" \
    --train_config_path "configs/train_template_mup.yaml"
```

### Quick run with warmstart

Set `warmstart_config.activate: true` in the [template](configs/train_template_mup.yaml) and run [the above](#quick-run-with-muparam) (with other suitable changes such as output path, etc.).


### For customized runs

Refer to [template for training configuration](configs/train_template.yaml) which can be loaded 
using `saws.TrainConfig()` and customized to run the training of choice.

For new datasets, add a configuration for the dataset similar to the [existing data configurations](configs/data_handlers/). 
Add the suitable key in the [dataset map](warms/__init__.py) for this experiment repo.
To train using this dataset, simply [run the training](warms/run_main.py) with the new dataset key. 
This will trigger the download and preprocessing (assuming _huggingface_ data).

### For customized experiment setup

Update the `.toml` file in the format of [`*_exp_canvas.toml`](configs/meta_exp_canvas.toml).
Note that any sub-user can only redefine variables that have been defined in `global`.
In order to enforce _Path()_ type-casting of path variables, the corresponding key should have at least one of _root_, _dir_, or _path_ in its name.
