from __future__ import annotations
import sys
import argparse
import time

from postprocessing.postproc import run_post

def run_post_cmd():

    # parse aguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--run_id", type=int)
    parser.add_argument("--batch_id", type=str)

    # read arguments
    args = parser.parse_args()
    config_path = args.config
    run_id = args.run_id
    batch_id = args.batch_id

    run_post(run_id, batch_id, config_path)

if __name__ == "__main__":
    tstart = time.time()

    # can be used to override arguments in case we want to run in interpreter
    sys.argv = [
        "run_postproc.py",
        "--config",
        r"\\sample\path\configs\postprocessing.ini",
        "--run_id",
        "280",
        "--batch_id",
        "testing"
    ]

    run_post_cmd()
    print("Done postproc ---------------------")
    elapsed = time.time() - tstart
    print(f"post took {elapsed}")
