import os
import pytest
from mlflow import projects

pytestmark = pytest.mark.integration


def test_simple_run():
    projects.run(
        os.path.join(os.path.dirname(__file__), "..", "resources", "example_yarn_project"),
        backend="yarn", entry_point="greeter", parameters={"name": "test_name"})
