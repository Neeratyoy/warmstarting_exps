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
pip install -e .
cd ..

# test
python -c "import saws"
```
