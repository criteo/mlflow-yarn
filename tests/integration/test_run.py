import logging
import os
import pytest
import skein
from cluster_pack.skein import skein_helper
from mlflow import projects

_logger = logging.getLogger(__name__)


pytestmark = pytest.mark.integration


def test_simple_run():
    run = projects.run(
        os.path.join(os.path.dirname(__file__), "..", "resources", "example_yarn_project"),
        backend="yarn", entry_point="greeter", parameters={"greeting": "Hello", "name": "world"})

    with skein.Client() as client:
        logs = skein_helper.get_application_logs(client, run.skein_app_id, 2)
        merged_logs = ""
        for key, value in logs.items():
            merged_logs += value
            _logger.info(f"logs:{key} {value}")
        assert "Hello world" in merged_logs
