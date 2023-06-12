import json
from bisect import bisect_left, bisect_right
from multiprocessing import Queue
from typing import Optional, Tuple

import mercantile
import numpy as np

from tracing.tracer import Tracer
from tracing.tracer_utils import TracerTask
from utils.dataclasses.point import Point
from utils.dataclasses.tower import Tower
from utils.tileDownloader.tileRequester_mock import TileRequester_mock


class Tracer_FakePredict(Tracer):
    """
    This implementation of the abstract tracer class allows you to fake the predictions, possibly iterating a little faster when tuning hyperparameters.
    Instead of actually predicting, it just returns any points in OSM that are within the tile
    """

    def __init__(
        self,
        run_id: int,
        queue: Optional[Queue],
        costmap: Optional[np.ndarray],
        topleft: Optional[Tuple[int, int]],
        costmap_path: Optional[str],
        tracertask: TracerTask,
        mock_detection_geojson: str,
        mock_images_folder: str,
        **_
    ):
        """
        Initialises the tracer

        :param run_id: Which run id to log the results to
        :param queue: The queue with tasks, the tracer will occasionally write new tasks to it
        :param costmap: The costmap, if provided, uses the costmap from memory, otherwise loads it from the costmap_path
        :param topleft: If costmap is passed, this defines the location of the costmap array
        :param costmap_path: Path to load the costmap from, mutually exclusive with costmap and topleft
        :param tracertask: dataclass defining the parameters for the current tracer run
        :param mock_detection_geojson: A geojson with the points to serve as predictions
        :param mock_images_folder: A folder with arbitrary images, to test out some image operations without requesting
        actual data.
        """
        super().__init__(run_id, queue, costmap, topleft, costmap_path, tracertask)
        self.tile_requester = TileRequester_mock("42", mock_images_folder, None, self.run_id)
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

    def _predict_on_tile(self, tile_image, tile_coord):
        """
        For this mock, we "predict" by getting the point from OSM

        :param tile_image: The tile image to "predict" on, not really used
        :param tile_coord: The coord of the tile, used to find intersecting points in the mock data
        :return: a list of towers found within the given tile, all with a max confidence of 1
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
