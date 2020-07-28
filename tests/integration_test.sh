#!/bin/bash

set -e

pushd mlflow-yarn
    rm -rf ~/venv
    python3.6 -m venv ~/venv
    source ~/venv/bin/activate
    pip install -U pip setuptools
    pip install cluster-pack@git+https://github.com/fhoering/cluster-pack.git@build_from_spec
    pip install .
    pip install pytest
    pytest -m integration -s tests --log-cli-level=INFO
popd
