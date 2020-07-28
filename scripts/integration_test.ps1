# you need to install python3, docker & docker-compose
# run from root dir mlflow-yarn scripts/integration_test.ps1
pip install hadoop-test-cluster
htcluster startup --image cdh5 --mount .:mlflow-yarn

# to avoid sharing files on the worker node we copy the python3.6 install script via hdfs to worker /tmp folder
htcluster exec -u root -s edge -- chown -R testuser /home/testuser
htcluster exec -u root -s edge -- /home/testuser/mlflow-yarn/tests/install_python.sh
htcluster exec -u root -s edge -- hdfs dfs -put /home/testuser/mlflow-yarn/tests/install_python.sh hdfs:///tmp
htcluster exec -u root -s worker -- hdfs dfs -get hdfs:///tmp/install_python.sh ~
htcluster exec -u root -s worker -- ./install_python.sh
htcluster exec -- /home/testuser/mlflow-yarn/tests/integration_test.sh