import logging
import os
import pytest
import skein
from cluster_pack.skein import skein_helper
from mlflow import projects

from mlflow_yarn import yarn_backend

_logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "cmd, expected_entry_point, expected_args",
    [
        ("python -m my_package.my_module --param1 'Hello python' b",
         "my_package.my_module",
         ["--param1", "Hello python", "b"]),
        ("python3 myscript.py arg",
         "myscript.py",
         ["arg"]),
        ("python3.6 myscript.py arg1 arg2",
         "myscript.py",
         ["arg1", "arg2"])
    ]
)
def test_try_split_cmd(cmd, expected_entry_point, expected_args):
    entry_point, args = yarn_backend.try_split_cmd(cmd)
    assert entry_point == expected_entry_point
    assert args == expected_args
