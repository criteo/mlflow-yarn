import logging
import os
import pytest
import skein
from cluster_pack.skein import skein_helper
from mlflow import projects

_logger = logging.getLogger(__name__)


pytestmark = pytest.mark.integration


def test_simple_run_conda():
    run = projects.run(
        os.path.join(os.path.dirname(__file__), "resources", "conda_project"),
        backend="yarn", entry_point="compute_intersection",
        parameters={"size": 10000},
        synchronous=True)

    _check_merged_logs(run.skein_app_id, "Time taken in secs:")


def test_simple_run_pip():
    run = projects.run(
        os.path.join(os.path.dirname(__file__), "resources", "pip_project"),
        backend="yarn", entry_point="compute_intersection",
        parameters={"size": 10000},
        synchronous=True)

    _check_merged_logs(run.skein_app_id, "Time taken in secs:")


def _check_merged_logs(app_id, key_word):
    with skein.Client() as client:
        logs = skein_helper.get_application_logs(client, app_id, 2)
        merged_logs = ""
        for key, value in logs.items():
            merged_logs += value
            _logger.info(f"logs:{key} {value}")
        assert key_word in merged_logs
