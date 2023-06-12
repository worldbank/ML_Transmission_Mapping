import json
import os
from bisect import bisect_left, bisect_right
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "True"


class Tracer_RealandFakePredict(Tracer):
    """
    This implementation of the abstract tracer class allows you to combine real predictions by the DL model on the
    image, with "fake" predictions, or towers that are already known in OSM.
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
        score_threshold=0.16,
        api_key: Optional[str] = None,
        mock_detection_geojson: str = None,
        ** _
    ):
        """
        Creates the tracer that combines real predictions with pre-known tower locations.

        :param run_id: Which run id to log the results to
        :param queue: The queue with tasks, the tracer will occasionally write new tasks to it
        :param costmap: The costmap, if provided, uses the costmap from memory, otherwise loads it from the costmap_path
        :param topleft: If costmap is passed, this defines the location of the costmap array
        :param costmap_path: Path to load the costmap from, mutually exclusive with costmap and topleft
        :param tracertask: dataclass defining the parameters for the current tracer run
        :param database_config_path: The location of the config for the database connection
        :param model_weight_path: The path to the weights of the model to use
        :param storage_config: All downloaded tiles are backupped for reuse; this is the location to which they are
            saved
        :param score_threshold: The minumum confidence to be considered a tower
        :param api_key: The API key with which to request tiles
        :param mock_detection_geojson: The pre-known towers to mix in with the predictions.
        """
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
            database_config=database_config_path,
            run_id=run_id,
        )

        with open(
            mock_detection_geojson,
            "r",
        ) as f:
            self.osm_points = json.load(f)

        self.osm_points = [
            f
            for f in self.osm_points["features"]
            if f.get("geometry") is not None
            and f.get("geometry", {}).get("coordinates") is not None
        ]
        self.osm_points = sorted(
            self.osm_points, key=lambda x: x.get("geometry", {}).get("coordinates")
        )
        self.osm_points_xs, self.osm_points_ys = zip(
            *[p.get("geometry").get("coordinates") for p in self.osm_points]
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

        # adding an additional retry scenario with a lower score threshold
        self.RETRY_CONFIGS.append(
            {
                "costmap_config": {
                    "next_angle_limit_overwrite": None,
                    "radius_overwrite": self.radius,
                    "gridsize_overwrite": 11,
                    "include_used_tiles": True,
                },
                "budget_overwrite": None,
                "score_threshold": 0.17,
                "reuse_predictions": False,
            }
        )

    def _fake_predict_on_tile(self, tile_image, tile_coord, **kwargs):
        """
        Getting the known tower locations for the tile_coord

        :param tile_image: Only for consistency, not used
        :param tile_coord: The (x,y) location of the tile
        :return: a list of towers with a confidence of 1
        """

        tile_bounds = mercantile.bounds(mercantile.Tile(tile_coord[0], tile_coord[1], self.zoom))
        left = bisect_left(self.osm_points_xs, tile_bounds.west)
        right = bisect_right(self.osm_points_xs, tile_bounds.east)
        ys = self.osm_points_ys[left:right]
        y_filter = [y >= tile_bounds.south and y <= tile_bounds.north for y in ys]
        points = [p for p, filtered in zip(self.osm_points[left:right], y_filter) if filtered]
        scores = [1.0] * len(points)

        return [
            Tower(Point(*point.get("geometry").get("coordinates")), score, new_tower=True)
            for point, score in zip(points, scores)
        ]

    def _predict_on_tile(self, tile_image, tile_coord, score_threshold=None, **kwargs):
        """
        Implements the prediction function.
        This call first performs the real prediction on the tile with the DL model.
        Followed by a "fake predict" extending the list of found towers with those already present in, for example, OSM

        :param tile_image: The image for this tile to predict on
        :param tile_coord: The location of the tile predicting on
        :param score_threshold: The score threshold of the prediction to count
        :param kwargs: Only for subclassing, not used.
        :return:
        """
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

                fake_towers = self._fake_predict_on_tile(tile_image, tile_coord)
                towers.extend(fake_towers)

            return towers
        except Exception as e:
            print(e)
            return []
