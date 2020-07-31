#!/bin/bash

set -e

pushd mlflow-yarn
    export MLFLOW_YARN_TESTS=1
    # if MLFLOW_YARN_TESTS=0 you need to configure the tracking server
    # export MLFLOW_TRACKING_URI=..
    rm -rf ~/venv
    python3.6 -m venv ~/venv
    source ~/venv/bin/activate
    pip install -U pip setuptools
    pip install .
    pip install pytest
    pytest -m integration -s tests --log-cli-level=INFO
popd
