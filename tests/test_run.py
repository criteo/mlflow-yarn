import os
from mlflow import projects


def test_simple_run():
    projects.run(
        os.path.join(os.path.dirname(__file__), "example_yarn_project"), 
        backend="yarn", entry_point="greeter", parameters={"name":"test_name"})
