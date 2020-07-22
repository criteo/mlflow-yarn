import logging
import os
import tempfile
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
from mlflow.projects import _get_or_create_conda_env, _get_entry_point_command, load_project
from mlflow.exceptions import ExecutionException


_logger = logging.getLogger(__name__)

_skein_client: skein.Client = None


def yarn_backend_builder():
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
    def __init__(self, client, skein_app_id, mlflow_run_id):
        super().__init__()
        self._skein_client = client
        self._skein_app_id = skein_app_id
        self._mlflow_run_id = mlflow_run_id

    @property
    def run_id(self):
        return self._mlflow_run_id

    def wait(self):
        return skein_helper.wait_for_finished(self._skein_client, self._skein_app_id)

    def cancel(self):
        self._skein_client.kill_application(self._skein_app_id)

    def get_status(self):
        app_report = self._skein_client.application_report(self._skein_app_id)
        return self._translate_to_runstate(app_report.state)

    def _translate_to_runstate(self, app_state):
        if app_state == "SUCCEEDED":
            return RunStatus.FINISHED
        elif app_state == "KILLED":
            return RunStatus.KILLED
        elif app_state == "FAILED":
            return RunStatus.FAILED
        elif app_state == "UNDEFINED":
            return RunStatus.RUNNING

        raise ExecutionException("YARN Application {self._skein_app_id}"
                                 " has invalid status: {app_state}")


class YarnProjectBackend(AbstractBackend):

    """Implementation of AbstractBackend running the job on YARN"""
    def __init__(self, client):
        super().__init__()
        self._skein_client = client

    def run(self, project_uri, entry_point, params,
            version, backend_config, tracking_uri, experiment_id):
        _logger.info('using yarn backend')
        _logger.info(locals())
        work_dir = fetch_and_validate_project(project_uri, version, entry_point, params)
        active_run = get_or_create_run(None, project_uri, experiment_id, work_dir, version,
                                       entry_point, params)

        _logger.info(f"work_dir={work_dir}")
        project = load_project(work_dir)

        with tempfile.TemporaryDirectory() as tempdir:
            entry_point_command = project.get_entry_point(entry_point)\
                .compute_command(params, tempdir)

            _logger.info(f"entry_point_command={entry_point_command}")

            conda_env_name = _get_or_create_conda_env(project.conda_env_path)
            conda_path = os.path.join(os.environ["MLFLOW_CONDA_HOME"], "envs", conda_env_name)
            package_path = _pack_conda_env(conda_path)

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

            service = skein.Service(
                resources=skein.model.Resources("1 GiB", 1),
                files=skein_config.files,
                script=skein_config.script
            )
            spec = skein.ApplicationSpec(services={"service": service})
            app_id = self._skein_client.submit(spec)
            return YarnSubmittedRun(self._skein_client, app_id, active_run.info.run_id)


def try_split_cmd(cmd):
    parts = [p for p in cmd.split(" ")
             if p != "-m" and p != "python" and p != "python3"]
    entry_point = ""
    args = []
    if len(parts) > 0:
        entry_point = parts[0]
    if len(parts) > 1:
        args = parts[1:]
    return entry_point, args


def _pack_conda_env(conda_env_name):
    temp_tarfile_dir = tempfile.mkdtemp()
    conda_pack_filename = os.path.join(temp_tarfile_dir, "conda_env.tar.gz")
    conda_pack.pack(prefix=conda_env_name, output=conda_pack_filename)
    return conda_pack_filename
