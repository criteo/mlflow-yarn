# mlflow-yarn

Backend implementation for running [MLFlow projects](https://www.mlflow.org/docs/latest/projects.html) against a Hadoop/YARN backend).

To activate just do 'pip install mlflow-yarn' along your mlflow environment. It will register the [plugin](https://www.mlflow.org/docs/latest/plugins.html) as an entrypoint with the YARN backend. 

```bash
$ pip install mlflow-yarn
```

mlflow-yarn only supports Python ≥3.6.

## Developement

Install from source

```bash
$ git clone https://github.com/criteo/cluster-pack
$ cd cluster-pack
$ pip install .
```

## Examples

# Example with [pip only project](https://github.com/criteo/mlflow-yarn/tree/master/tests/resources/pip_project)

- Dependencies are pulled from requirements.txt

```bash
$ git clone https://github.com/criteo/mlflow-yarn
$ pip install mlflow
mlflow run tests/resources/pip_project -e compute_intersection -P size=10000 --backend yarn
```

# Example with [conda project](https://github.com/criteo/mlflow-yarn/tree/master/tests/resources/conda_project)

- Dependencies are pulled from conda.yaml

```bash
$ git clone https://github.com/criteo/mlflow-yarn
$ pip install mlflow
$ mlflow run tests/resources/conda_project -e compute_intersection -P size=10000 --backend yarn
```

[More infos](https://www.mlflow.org/docs/latest/projects.html#mlproject-file) on setting up a project file.

Docker container environment is currently not supported.