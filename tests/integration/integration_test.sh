#!/bin/bash

set -e

pushd mlflow-yarn
    export MLFLOW_CONDA_HOME=~/miniconda/
    rm -rf ~/venv
    python3 -m venv ~/venv
    source ~/venv/bin/activate
    pip install -U pip setuptools
    pip install -e .
    pip install git+https://github.com/criteo-forks/mlflow.git@criteo-master
    pip install pytest

    pytest -m integration -s --rootdir=tests --log-cli-level=INFO
popd
