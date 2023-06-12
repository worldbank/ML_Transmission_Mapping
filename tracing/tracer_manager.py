import argparse
import json
import logging
import multiprocessing as mp
import os
import queue
import random
import time
from abc import ABC
from enum import Enum
from multiprocessing import Queue
from typing import Optional, Union
from uuid import uuid4

import psycopg2
import psycopg2.extras

from database.database_utils import get_db_config
from tracing.tracer_fakepredict import Tracer_FakePredict
from tracing.tracer_real_and_fake_predict import Tracer_RealandFakePredict
from tracing.tracer_realpredict import Tracer_RealPredict
from tracing.tracer_utils import TracerTask
from utils.dataclasses.point import Point

# call it in any place of your program
# before working with UUID objects in PostgreSQL
psycopg2.extras.register_uuid()


PredictionMode = Enum("PredictionMode", ["Fake", "Real", "Real_Fake"])

def tracer_spawner(
    runid: int, task_queue, config, prediction_mode: PredictionMode, log_file: str
):
    """
    This function spawns a single (blocking) tracer, whenever it exists, it starts a new one with the next task in
    the queue, until the queue is empty.

    :param runid: The run id to which the tracers should write their results.
    :param task_queue: The queue with tasks
    :param config: The config with the parameters to use for tracing
    :param prediction_mode: Whether to use only predictions, only OSM points, or both.
    :param log_file: Which file to log to
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logging.info(f"Started spawnpool : {os.getpid()}")
    task: TracerTask = task_queue.get(block=True, timeout=5)

    pred_function = {
        PredictionMode.Fake: Tracer_FakePredict,
        PredictionMode.Real: Tracer_RealPredict,
        PredictionMode.Real_Fake: Tracer_RealandFakePredict,
    }[prediction_mode]

    while not task_queue.empty(): # While there are still new tasks:
        logging.info(f"Tracing from: {task}...")
        pred_function(
            runid,
            task_queue,
            costmap_path=config["costmap_path"],
            storage_config=config["storage_config"],
            costmap=None,
            topleft=None,
            tracertask=task,
            model_weight_path=config["model_weight_path"],
            database_config_path=config["database_config_path"],
            score_threshold=float(config["tracing_constants"]["score_threshold"]),
            mock_detection_geojson=config.get("mock_detection_geojson"),
            mock_images_folder=config.get("mock_images_folder"),
            api_key = config["mapbox_api"]["api_key"]
        ).trace() # Create a new instance of a tracer, and set it off.

        logging.info("Tracer died, looking for new one!")
        if task_queue.empty() is not None:
            try:
                task: TracerTask = task_queue.get(block=True, timeout=600)
            except queue.Empty:
                pass
        else:
            # If the queue is empty, sleep for 10 minutes, if no new tasks have been added in the meantime,
            # kill the worker
            logging.info(
                f"Oh no the queue was empty! Napping for 10 minutes before really terminating, time {time.time()}"
            )
            time.sleep(600)
    logging.info(f"No new task found, spawnpool closing {time.time()}: {os.getpid()}")


class TracerManager(ABC):
    """
    Abstract tracer manager.
    Currently, the only abstract feature is something that acts as the task queue.
    Could be implemented to work in the cloud.
    """

    task_queue: Queue

    def __init__(
        self,
        config_path: Optional[str] = None,
    ):
        assert os.path.exists(config_path), "config file cannot be found {}".format(config_path)
        with open(config_path, "r", encoding="utf8") as f:
            config = json.load(f)

        self.config = config

        self.default_radius = int(config["tracing_constants"]["default_radius"])
        self.default_budget = float(config["tracing_constants"].get("default_budget", 25.0))

        self.num_workers = int(config["num_workers"])

        self.run_description = config["run_description"]
        self.run_id = config.get("continue_from_run_id")
        self.mock_detection_geojson = config.get("mock_detection_geojson")
        self.costmap_path = config["costmap_path"]

    def start_run(self, prediction_mode: PredictionMode, log_file: str):
        """
        Starts a full run of smarttracing

        :param prediction_mode: Determines whether to use the deeplearning model, fetch points from OSM,
        or use OSM points combined with deeplearning results
        :param log_file: Path to file to log to
        """
        run_description = self.run_description or str(uuid4())
        self.run_id = self.run_id or self._make_run(run_description=run_description)

        print(f"============\n Starting with run: {self.run_id}\n============")

        # Start workers until task_queue is emtpy
        print(self.num_workers)
        if self.num_workers > 0:
            print("Creating pool")
            with mp.get_context("spawn").Pool(self.num_workers) as pool:
                processes = [
                    pool.apply_async(

                        tracer_spawner,
                        args=(self.run_id, self.task_queue, self.config, prediction_mode, log_file),
                    )
                    for _ in range(self.num_workers)
                ]
                _ = [p.get() for p in processes]
        else:
            print("Running single-threaded!!")
            tracer_spawner(
                self.run_id, self.task_queue, self.config, prediction_mode=prediction_mode, log_file=log_file
            )

    def _make_run(self, run_description: str):
        """
        Logs a new run_id to the database, with the given description. This will help you track the results of
        various runs. As a result, it gives you a new unique run_id to use for logging the results to.
        :param run_description: Description of this smartracing run
        :return: the new run id
        """
        connection = psycopg2.connect(**get_db_config(self.config.get("database_config_path")))
        try:
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO run (description) VALUES (%s) RETURNING id", (run_description,)
                    )
                    runid = cursor.fetchone().id
        finally:
            connection.close()
        return runid

    def load_startingpoints(self):
        """
        Loads the starting points for this task, as defined in the config, from the database
        """
        connection = psycopg2.connect(**get_db_config(self.config.get("database_config_path")))
        try:
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT id, file_path, ST_X(geom) as lng, ST_Y(geom) as lat "
                        "FROM startingpoints "
                        "WHERE set_name=(%s)",
                        (self.config.get("set_name"),),
                    )
                    startingpoints = cursor.fetchall()
        finally:
            connection.close()

        random.shuffle(startingpoints) # shuffle the startingpoints, sometimes there are multiple close together

        for starting_point in startingpoints: # Fill the queue with tasks based on the startingpoints
            point = Point(starting_point.lng, starting_point.lat)
            self.task_queue.put(
                TracerTask(
                    point,
                    direction=None,
                    radius=self.default_radius,
                    budget=self.default_budget,
                    starting_point_id=starting_point.id,
                    line_id=None,
                )
            )


class TracerManagerLocal(TracerManager):
    """
    Implements the TracerManager with a local task_queue and multiprocessing manager.
    """
    def __init__(
        self,
        config_path: Optional[str] = None,
    ):
        super(TracerManagerLocal, self).__init__(config_path)
        self.multiprocessing_manager = mp.Manager()
        self.task_queue = self.multiprocessing_manager.Queue()



def tracer_manager_constructor(config_path):
    """
    Constructs a tracer manager from a config file, results in either a
    """
    assert os.path.exists(config_path), "config file cannot be found {}".format(config_path)
    with open(config_path, "r", encoding="utf8") as f:
        config = json.load(f)
    mode = config["mode"]
    if mode == "AWS":
        raise NotImplementedError("Smarttracing on AWS is not implemented in this repo")
    elif mode == "LOCAL":
        return TracerManagerLocal(config_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    t0 = time.time()
    print("start time:", t0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--log_file", type=str)
    args = parser.parse_args()
    TM = tracer_manager_constructor(args.config)
    TM.load_startingpoints()

    try:
        TM.start_run(PredictionMode.Real_Fake, args.log_file)
        print(f"Total processing took: {time.time() - t0}")

    except Exception as e:
        print("Failed", e)
        print(f"Total processing took: {time.time() - t0}")
