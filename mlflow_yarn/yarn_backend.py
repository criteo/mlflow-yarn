import logging
import os
import tempfile
import shlex
import time

import conda_pack
import skein
import cluster_pack
from cluster_pack.skein import skein_config_builder, skein_helper
from cluster_pack import packaging


from mlflow.entities import RunStatus
from mlflow.projects.utils import fetch_and_validate_project, get_or_create_run
from mlflow.projects.backend.abstract_backend import AbstractBackend
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.projects import load_project
from mlflow.exceptions import ExecutionException
from mlflow.tracking import MlflowClient

from typing import Tuple, List, Dict


_logger = logging.getLogger(__name__)

_skein_client: skein.Client = None


def yarn_backend_builder() -> AbstractBackend:
    global _skein_client
    if not _skein_client:
        _skein_client = skein.Client()
    return YarnProjectBackend(_skein_client)


class YarnSubmittedRun(SubmittedRun):
    """Instance of SubmittedRun
       corresponding to a Yarn Job launched through skein to run an MLflow
       project.

    :param skein_app_id: ID of the submitted Skein Application.
    :param mlflow_run_id: ID of the MLflow project run.
    """
    def __init__(self, client: skein.Client, skein_app_id: str, mlflow_run_id: str) -> None:
        super().__init__()
        self._skein_client = client
        self.skein_app_id = skein_app_id
        self._mlflow_run_id = mlflow_run_id

    @property
    def run_id(self) -> str:
        return self._mlflow_run_id

    def wait(self) -> bool:
        return skein_helper.wait_for_finished(self._skein_client, self.skein_app_id)

    def cancel(self) -> None:
        self._skein_client.kill_application(self.skein_app_id)

    def get_status(self) -> RunStatus:
        app_report = self._skein_client.application_report(self.skein_app_id)
        return self._translate_to_runstate(app_report.state)

    def _translate_to_runstate(self, app_state: str) -> RunStatus:
        if app_state == skein.model.ApplicationState.FINISHED:
            return RunStatus.FINISHED
        elif app_state == skein.model.ApplicationState.KILLED:
            return RunStatus.KILLED
        elif app_state == skein.model.ApplicationState.FAILED:
            return RunStatus.FAILED
        elif (app_state == skein.model.ApplicationState.NEW_SAVING or
              app_state == skein.model.ApplicationState.ACCEPTED or
              app_state == skein.model.ApplicationState.SUBMITTED):
            return RunStatus.SCHEDULED
        elif app_state == skein.model.ApplicationState.RUNNING:
            return RunStatus.RUNNING

        raise ExecutionException(f"YARN Application {self._skein_app_id}"
                                 f" has invalid status: {app_state}")


class YarnProjectBackend(AbstractBackend):

    """Implementation of AbstractBackend running the job on YARN"""
    def __init__(self, client: skein.Client):
        super().__init__()
        self._skein_client = client

    def run(self, project_uri: str, entry_point: str, params: Dict,
            version: str, backend_config: str, tracking_uri: str, experiment_id: str
    ) -> SubmittedRun:
        _logger.info('using yarn backend')
        _logger.info(locals())
        work_dir = fetch_and_validate_project(project_uri, version, entry_point, params)
        active_run = get_or_create_run(None, project_uri, experiment_id, work_dir, version,
                                       entry_point, params)
        _logger.info(f"run_id={active_run.info.run_id}")
        _logger.info(f"work_dir={work_dir}")
        project = load_project(work_dir)

        with tempfile.TemporaryDirectory() as tempdir:
            entry_point_command = project.get_entry_point(entry_point)\
                .compute_command(params, tempdir)

            _logger.info(f"entry_point_command={entry_point_command}")

            if project.conda_env_path:
                spec_file = project.conda_env_path
            else:
                spec_file = os.path.join(work_dir, "requirements.txt")
                if not os.path.exists(spec_file):
                    raise ValueError

            package_path = cluster_pack.upload_spec(spec_file)

            additional_files = []
            for file in os.listdir(work_dir):
                full_path = os.path.join(work_dir, file)
                if os.path.isfile(full_path):
                    additional_files.append(full_path)

            _logger.info(package_path)

            entry_point, args = try_split_cmd(entry_point_command)

            _logger.info(f"args {entry_point} {args}")

            skein_config = skein_config_builder.build(
                    module_name=entry_point,
                    args=args,
                    package_path=package_path,
                    additional_files=additional_files,
                    tmp_dir=tempdir)

            env = {
                "MLFLOW_RUN_ID": active_run.info.run_id,
                "MLFLOW_TRACKING_URI": mlflow.get_tracking_uri()
            }

            service = skein.Service(
                resources=skein.model.Resources("1 GiB", 1),
                files=skein_config.files,
                script=skein_config.script,
                env=env
            )
            spec = skein.ApplicationSpec(services={"service": service})
            app_id = self._skein_client.submit(spec)
            MlflowClient().set_tag(active_run.info.run_id, "skein_application_id", app_id)
            return YarnSubmittedRun(self._skein_client, app_id, active_run.info.run_id)


def try_split_cmd(cmd: str) -> Tuple[str, List[str]]:
    parts = []
    found_python = False
    for part in shlex.split(cmd):
        if part == "-m":
            continue
        elif not found_python and part.startswith("python"):
            found_python = True
            continue
        parts.append(part)
    entry_point = ""
    args = []
    if len(parts) > 0:
        entry_point = parts[0]
    if len(parts) > 1:
        args = parts[1:]
    return entry_point, args
