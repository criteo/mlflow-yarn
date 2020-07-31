import logging
import os
import pytest
import skein
from cluster_pack.skein import skein_launcher
from mlflow import projects

_logger = logging.getLogger(__name__)


pytestmark = pytest.mark.integration


def test_simple_run_conda():
    submitted_run = projects.run(
        os.path.join(os.path.dirname(__file__), "resources", "conda_project"),
        backend="yarn", entry_point="compute_intersection",
        parameters={"size": 10001},
        synchronous=False)
    result_status = submitted_run.wait()

    _check_merged_logs(
        submitted_run.skein_app_id,
        "Time taken in secs:",
        result_status)


def test_simple_run_pip():
    submitted_run = projects.run(
        os.path.join(os.path.dirname(__file__), "resources", "pip_project"),
        backend="yarn", entry_point="compute_intersection",
        parameters={"size": 10002},
        synchronous=False)
    result_status = submitted_run.wait()

    _check_merged_logs(
        submitted_run.skein_app_id,
        "Time taken in secs:",
        result_status)


def test_simple_run_pip_synchronous():
    submitted_run = projects.run(
        os.path.join(os.path.dirname(__file__), "resources", "pip_project"),
        backend="yarn", entry_point="compute_intersection",
        parameters={"size": 10003},
        synchronous=True)

    # failure launches ExecutionException without the infos of the active run
    _check_merged_logs(
        submitted_run.skein_app_id,
        "Time taken in secs:",
        True)


def _check_merged_logs(app_id, key_word, result_status):
    with skein.Client() as client:
        logs = skein_launcher.get_application_logs(client, app_id, 2)
        merged_logs = ""
        for key, value in logs.items():
            merged_logs += value
            _logger.info(f"logs:{key} {value}")
        assert result_status
        assert key_word in merged_logs
