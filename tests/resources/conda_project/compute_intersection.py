"""
Example program helping verify functionality for passing parameters other than those required in
the MLproject file.
"""
import argparse
import time
import mlflow
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("size", help="Array size", type=int)
    args = parser.parse_args()

    a = np.random.randint(0, args.size * 2, args.size)
    b = np.random.randint(0, args.size * 2, args.size)

    start_time = time.perf_counter()
    np.intersect1d(a, b)
    elapsed_time = time.perf_counter() - start_time
    print(f"Time taken in secs: {elapsed_time}")
    mlflow.log_metric("Time taken in secs", elapsed_time)
