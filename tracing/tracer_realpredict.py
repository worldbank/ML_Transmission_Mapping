import os
from multiprocessing import Queue
from typing import Optional, Tuple

import mercantile
import numpy as np

from model.centernet import centernet
from model.centernet_utils import preprocess_image
from tracing.tracer import Tracer
from tracing.tracer_utils import TracerTask
from utils.dataclasses.point import Point
from utils.dataclasses.tower import Tower
from utils.tileDownloader.tileRequester_mapbox import TileRequester_mapbox

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "True"


class Tracer_RealPredict(Tracer):
    """
    Starts from a task in the task_queue (startpoint, radius, direction) and start tracing until completed
    """

    def __init__(
        self,
        run_id: int,
        queue: Optional[Queue],
        costmap: Optional[np.ndarray],
        topleft: Optional[Tuple[int, int]],
        costmap_path: Optional[str],
        tracertask: TracerTask,
        model_weight_path: str,
        storage_config: dict,
        database_config_path: Optional[str] = None,
        score_threshold=0.15,
        api_key: Optional[str] = None,
        **_
    ):
        super().__init__(
            run_id,
            queue,
            costmap,
            topleft,
            costmap_path,
            tracertask,
            database_config_path=database_config_path,
        )
        assert storage_config["location"] in ["AWS", "LOCAL"]
        self.tile_requester = TileRequester_mapbox(
            api_key=api_key,
            storage_config=storage_config,
            database_config=database_config_path
            or r"\\knippa\d\Projects\A191_powergrid\amazon.ini",
            run_id=run_id,
        )

        classes_dict = {"powertower": 0}

        self.prediction_model, self.debug_model = centernet(
            backbone="resnet101",
            num_classes=len(classes_dict),
            max_objects=45,
            score_threshold=0.1,
            nms=False,
        )
        self.prediction_model.load_weights(model_weight_path)
        self.score_threshold = score_threshold
        self.RETRY_CONFIGS.append(
            {
                "costmap_config": {
                    "next_angle_limit_overwrite": None,
                    "radius_overwrite": self.radius,
                    "gridsize_overwrite": 11,
                    "include_used_tiles": True,
                },
                "budget_overwrite": None,
                "score_threshold": 0.18,
                "reuse_predictions": False,
            }
        )

    def _predict_on_tile(self, tile_image, tile_coord, score_threshold=None, **kwargs):
        if score_threshold is not None:
            pass
        score_threshold = score_threshold or self.score_threshold
        try:
            prepped_image = preprocess_image(np.copy(tile_image))
            pred = self.prediction_model.predict_on_batch(prepped_image[np.newaxis, ...])[0]

            bounds = mercantile.bounds(mercantile.Tile(*tile_coord, z=18))
            tile_lng, tile_lat = (bounds.west, bounds.north)
            tile_width, tile_height = (
                abs(bounds.east - bounds.west),
                abs(bounds.south - bounds.north),
            )
            image_width, image_height = tile_image.shape[:2]
            towers = []
            for predicted_tower in [pr for pr in pred if pr[2] > score_threshold]:
                within_tile_x, within_tile_y = predicted_tower[:2]
                if (
                    within_tile_y < 0
                    or within_tile_y >= image_height
                    or within_tile_x < 0
                    or within_tile_x >= image_width
                ):
                    continue
                if within_tile_y == 0:
                    within_tile_y += 1
                if within_tile_x == 0:
                    within_tile_x += 1
                score = predicted_tower[2]
                tower_lng = tile_lng + tile_width * (within_tile_x / image_width)
                tower_lat = tile_lat - tile_height * (within_tile_y / image_height)
                towers.append(
                    Tower(Point(float(tower_lng), float(tower_lat)), float(score), new_tower=True)
                )
            return towers
        except Exception as e:
            print(e)
            return []


if __name__ == "__main__":
    tracer = Tracer_RealPredict(
        83,
        None,
        None,
        None,
        r"\\sample\path\05_ProbabilityMap\01_India\ALL\03_final_costs.tif",
        TracerTask(
            startpoint=Point(86.872894, 23.877843),
            direction=45,
            radius=280,
        ),
        r"\\path\to\model.hdf5",
    )

    tracer.trace()
    print("")
