import os
import setuptools
import versioneer

here = os.path.abspath(os.path.dirname(__file__))

DESCRIPTION = "Backend implementation for running MLFlow projects on Hadoop/YARN"

try:
    LONG_DESCRIPTION = open(os.path.join(here, "README.md"), encoding="utf-8").read()
except Exception:
    LONG_DESCRIPTION = ""


def _read_reqs(relpath):
    fullpath = os.path.join(os.path.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [s.strip() for s in f.readlines()
                if (s.strip() and not s.startswith("#"))]


REQUIREMENTS = _read_reqs("requirements.txt")
TESTS_REQUIREMENTS = _read_reqs("tests-requirements.txt")

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Operating System :: OS Independent",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Topic :: Software Development :: Libraries"
]


setuptools.setup(
    name="mlflow-yarn",
    packages=setuptools.find_packages(),
    version=versioneer.get_version(),
    install_requires=REQUIREMENTS,
    tests_require=TESTS_REQUIREMENTS,
    python_requires=">=3.6",
    maintainer="Criteo",
    maintainer_email="github@criteo.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    classifiers=CLASSIFIERS,
    keywords="mlflow",
    url="https://github.com/criteo/mlflow-yarn",
    entry_points={
        "mlflow.project_backend":
            "yarn=mlflow_yarn.yarn_backend:yarn_backend_builder",
    },
)
