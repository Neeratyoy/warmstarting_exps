# warmstarting_exps


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

**NOTE**: These examples represent an atomic run that should not break with any commit