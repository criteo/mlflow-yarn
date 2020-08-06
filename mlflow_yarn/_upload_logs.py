import getpass
import os
import mlflow
import traceback


def _extract_skein_container_name(container_id: str) -> str:
    parts = container_id.rsplit("_", 1)
    if len(parts) > 0:
        return parts[0]
    return ""


def _log_url() -> str:
    # from https://github.com/apache/hadoop/blob/f02b0e19940dc6fc1e19258a40db37d1eed89d21
    # /hadoop-yarn-project/hadoop-yarn/hadoop-yarn-common
    # /src/main/java/org/apache/hadoop/yarn/webapp/util/WebAppUtils.java
    if (
      not os.getenv("SKEIN_CONTAINER_ID", "") or
      not os.getenv("NM_HOST", "") or
      not os.getenv("NM_HTTP_PORT", "") or
      not os.getenv("CONTAINER_ID", "")
    ):
        return ""

    user = getpass.getuser()
    name = _extract_skein_container_name(os.environ["SKEIN_CONTAINER_ID"])
    if not name:
        return ""

    return (f'http://{os.environ["NM_HOST"]}:{os.environ["NM_HTTP_PORT"]}'
            f'/node/containerlogs/{os.environ["CONTAINER_ID"]}/{user}/{name}.log?start=-4096')


def _upload_logs(local_log_path: str) -> None:
    try:
        mlflow.set_tag("log_url", _log_url())
        mlflow.log_artifact(local_log_path)
    except Exception:
        print("failed to upload logs to mlflow")
        traceback.print_exc()
