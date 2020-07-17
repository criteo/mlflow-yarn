#!/bin/bash

export MLFLOW_CONDA_HOME=~/miniconda/
rm -rf ~/venv
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install -U pip setuptools
pip install -e .
pip install git+https://github.com/criteo-forks/mlflow
pip install pytest

pushd mlflow-yarn
    # pytest -s tests
popd
