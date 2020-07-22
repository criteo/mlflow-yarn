# you need to install python3, docker & docker-compose
# run from root dir mlflow-yarn scripts/integration_test.ps1
pip install hadoop-test-cluster
htcluster startup --image cdh5 --mount .:mlflow-yarn
htcluster exec -- /home/testuser/mlflow-yarn/tests/integration/integration_test.sh