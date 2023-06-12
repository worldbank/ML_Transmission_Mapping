import time
import warnings
from abc import ABC, abstractmethod
from multiprocessing import Queue
from typing import List, Optional, Tuple
from uuid import UUID, uuid4

import mercantile
import numpy as np
import psycopg2
import psycopg2.extras
from geopy.distance import distance
from psycopg2.errors import UniqueViolation

from database.database_utils import get_db_config
from tracing.tracer_utils import TracerTask
from utils.costMapInterface_fromfile import CostMapInterface_from_file
from utils.costMapInterface_frommemory import CostMapInterface_from_memory
from utils.dataclasses.point import Point
from utils.dataclasses.tower import Tower
from utils.other import filter_towers_closer_than_x
from utils.tileDownloader.tileRequester import TileRequester
from utils.tracer_tools import get_search_kernel


class Tracer(ABC):
    """
    Starts from a task in the task_queue (startpoint, radius, direction) and start tracing until completed
    """

    tile_requester: TileRequester

    def __init__(
        self,
        run_id: int,
        queue: Queue,
        costmap: Optional[np.ndarray],
        topleft: Optional[Tuple[int, int]],
        costmap_path: Optional[str],
        tracertask: TracerTask,
        database_config_path: Optional[str] = None,
    ):
        """
        :param run_id: Which run id to log the results to
        :param queue: The queue with tasks, the tracer will occasionally write new tasks to it
        :param costmap: The costmap, if provided, uses the costmap from memory, otherwise loads it from the costmap_path
        :param topleft: If costmap is passed, this defines the location of the costmap array
        :param costmap_path: Path to load the costmap from, mutually exclusive with costmap and topleft
        :param tracertask: dataclass defining the parameters for the current tracer run
        :param database_config_path: The location of the config for the database connection
        """
        psycopg2.extras.register_uuid()
        self.direction_consistency_threshold = 10
        assert (costmap is None and topleft is None and costmap_path is not None) or (
            costmap is not None and topleft is not None and costmap_path is None
        )
        self.run_id = run_id
        self._connection = psycopg2.connect(**get_db_config(database_config_path))
        self.repeated_towers_threshold = 3 # How many towers in a row can already be known to a tracer before it
        # terminates
        self.queue = queue
        self.location = tracertask.startpoint
        self.distances = [tracertask.radius] # This is a list of the distances between all found towers, the median of
        # this will be the next radius for the kernel.
        self.initial_direction = tracertask.direction # The direction in which the towers are to be expected
        # towers than this have been found, it stops looking. In practise; always infinite
        # (so it is never the reason why the tracers stops looking for more towers)
        self.minimum_tiles_bought = 3 # No matter how fast the tracer finds a tower, it always downloads at least
        # this many tiles at each step. Setting this to 3, means it will always look for powertowers in the 3 most
        # likely tiles.

        # Instantiate a class that interacts with the costmap data
        if costmap is not None:
            self.costMapInterface = CostMapInterface_from_memory(
                cost_map=costmap, topleft_xy=topleft
            )
        else:
            self.costMapInterface = CostMapInterface_from_file(costmap_path)

        self.gridsize = 11 # Size of the grid to look within
        self.zoom = 18 # Zoom level of the tiles
        self.budget = tracertask.budget # The tracer "buys" tiles for the price defined in the local costmap until
        # this budget has been spent.
        self.next_angle_limit = tracertask.next_angle_limit # When the angle between the next tile and the current
        # tower, and the last known direction is larger than this, the cost becomes much higher.

        self.previous_location = None
        self.last_confident_location = None
        self.previous_direction = None
        self.last_confident_direction = None
        self.bought_tiles_counts = []

        # I assume a tracer follows a single line, so let's give it a new uuid, connecting single towers to a trace
        self.line_id = tracertask.line_id

        self.last_tower_uuid = tracertask.last_tower_uuid
        self.starting_point_id = tracertask.starting_point_id

        # The tracer has a list of retry_configs, whenever it gets stuck these define things to try.

        self.found_towers_on_last_try = 0 # When the tracer finds a tower in the last retry scenario,
        # this number is increased by 1
        self.FOUND_TOWERS_ON_LAST_TRY_LIMIT = 3 # When the tracer has found the last 3 towers only on the very last
        # attempt, it is very unlikely that it follows a proper powerline, and the tracer terminates.

        self.RETRY_CONFIGS = [
            {  # Regular settings
                "costmap_config": {
                    "next_angle_limit_overwrite": None,
                    "radius_overwrite": None,
                    "gridsize_overwrite": None,
                },
                "budget_overwrite": None,
            },
            {  # Sometimes we just want to check a liiiiitle bit further
                "costmap_config": {
                    "next_angle_limit_overwrite": None,
                    "radius_overwrite": self.radius * 1.2,
                    "gridsize_overwrite": 13,
                },
                "budget_overwrite": self.budget / 4,
            },
            {  # Looking 2x the radius
                "costmap_config": {
                    "next_angle_limit_overwrite": None,
                    "radius_overwrite": self.radius * 2,
                    "gridsize_overwrite": 21,
                },
                "budget_overwrite": None,
            },
            {  # Maybe we made a weird angle
                "costmap_config": {
                    "next_angle_limit_overwrite": 120,
                    "radius_overwrite": None,
                    "gridsize_overwrite": None,
                },
                "budget_overwrite": self.budget / 2,
            },
            {  # Reuse towers we found before to cross crossings
                "costmap_config": {
                    "next_angle_limit_overwrite": None,
                    "radius_overwrite": self.radius,
                    "gridsize_overwrite": 11,
                    "include_used_tiles": True,
                },
                "budget_overwrite": None,
            },
        ]

    @property
    def location_tile(self):
        """
        :return: The mercantile tile in which the current location is in.
        """
        return mercantile.tile(lng=self.location.lng, lat=self.location.lat, zoom=self.zoom)

    @property
    def radius(self):
        """
        :return: The expected distance to the next tower
        """
        return np.median(self.distances)

    @staticmethod
    def get_direction_between_points(point_a: Point, point_b: Point):
        """
        Little helper function to get the direction between two points

        :param point_a: One point
        :param point_b: The next point
        :return: The travel direction between the two points in degrees
        """
        if point_b.lng >= point_a.lng:
            quadrant = 90
            if point_b.lat >= point_a.lat:
                quadrant *= -1
        else:
            quadrant = 270
            if point_b.lat < point_a.lat:
                quadrant *= -1

        adjacent = distance(
            (point_a.lat, point_a.lng),
            (point_a.lat, point_b.lng),
        ).meters
        distance_to_b = (
            distance(
                (point_a.lat, point_a.lng),
                (point_b.lat, point_b.lng),
            ).meters
            + 1e-9
        )

        return int(
            abs(np.round(np.rad2deg(np.arccos(adjacent / distance_to_b))).astype(int) + quadrant)
        )

    @property
    def direction(self):
        """
        :return: The direction the tracer is heading in (the direction between the last tower, and the one before)
        """
        if self.previous_location is None:
            return self.initial_direction
        return self.get_direction_between_points(self.previous_location, self.location)

    def _make_previous_tiles_infinite_expensive(self, cost_map, x, y, grid_size):
        """
        Get's the location of any tiles that have been predicted on previously and makes their cost infinite,
        so they will never be bought again.

        :param cost_map: The costmap
        :param x: The x coord of the center point around which to buy the costmap
        :param y: The y coord of the center point around which to buy the costmap
        :param grid_size: The size of the
        :return: The costmap, in which all tiles previously bought have their cost set to infinite
        """

        # Determine the bounds of the grid the tracer will be looking at for this step.
        xmin = int(x - np.floor(grid_size / 2))
        xmax = int(x + np.ceil(grid_size / 2))
        ymin = int(y - np.floor(grid_size / 2))
        ymax = int(y + np.ceil(grid_size / 2))

        # Fetch all previously bought tiles *in this run*
        with self._connection:
            with self._connection.cursor() as cursor:
                cursor.execute(
                    "SELECT tile.x, tile.y "
                    "FROM predicted LEFT JOIN tile "
                    f"ON tile.id=predicted.tile_id "
                    f"WHERE tile.x >= {xmin} AND tile.x < {xmax} AND tile.y >= {ymin} AND tile.y < {ymax} "
                    f"AND predicted.run_id = {self.run_id} "
                )
                previous_tiles = cursor.fetchall()

        # Set the cost of each of these tiles to infinite
        for tile in previous_tiles:
            cost_map[tile.y - ymin, tile.x - xmin] = np.inf

        return cost_map

    def get_local_cost_map(
        self,
        next_angle_limit,
        radius,
        gridsize=11,
        include_used_tiles=False,
        debug=False,
    ):
        """
        This function generates a local costmap, this is a piece of the costmap of size <gridsize> centered around the
        last known powertower/startingpoint.
        This piece is then combined with a search kernel (radius + direction) and all previous tiles are made
        infinitely expensive.

        :param next_angle_limit: The limit of the expected direction
        :param radius: The expected distance to the next tower
        :param gridsize: The size of the grid to consider
        :param include_used_tiles: Whether to allow re-using towers in previously analysed tiles
        :param debug: In debug mode, the individual kernels are returned.
        :return: The local costmap; or, if debug=True, the local costmap, the base costmap, and the search kernel
        """

        tile = mercantile.tile(lng=self.location.lng, lat=self.location.lat, zoom=self.zoom)
        base_costmap = self.costMapInterface.get_cost_grid(tile.x, tile.y, gridsize)

        if base_costmap.shape != (gridsize, gridsize):
            return None  # local map outside global map

        search_kernel = get_search_kernel(
            center_lon=self.location.lng,
            center_lat=self.location.lat,
            radius=radius,
            gridsize=gridsize,
            direction=self.direction,
            min_weight=0.0,
            max_weight=0.9,
            next_angle_limit=next_angle_limit,
            debug=debug,
        )
        base_costmap_weight = 1
        search_kernel_weight = 4

        base_costmap /= np.max(
            base_costmap
        )  # Normalise so that a tile can cost at most base_costmap_weight + search_kernel_weight

        local_cost_map = base_costmap_weight * base_costmap + search_kernel_weight * (
            1 - search_kernel
        )

        if not include_used_tiles:
            local_cost_map = self._make_previous_tiles_infinite_expensive(
                local_cost_map, tile.x, tile.y, gridsize
            )

        if debug:
            return local_cost_map, base_costmap, search_kernel
        return local_cost_map

    def get_tile(self, tile_coord: Tuple[int, int], **kwargs) -> Tuple[np.ndarray, List[Tower]]:
        """
        Get's the imagery for a given tile, and either loads the previously found towers from the db, or predicts on
        it straight away.

        :param tile_coord: The tile for which to get the image and the towers
        :param kwargs: Depending on the predict mode, other params can be passed
        :return: np.array with the image for this tile, and a list of towers found within
        """
        tile_image, towers = self.tile_requester.get_tile(*tile_coord)
        if not kwargs.get("reuse_predictions", True):
            towers = None
        if towers is None:
            towers = self._predict_on_tile(tile_image, tile_coord, **kwargs)
            self._log_predicted(tile_coord)
        return tile_image, towers

    @abstractmethod
    def _predict_on_tile(
        self, tile_image: np.ndarray, tile_coord: Tuple[int, int], **kwargs
    ) -> List[Tower]:
        """
        Abstract method, implementation should return a list of towers present on this tile.

        :param tile_image: The image of the tile
        :param tile_coord: The position of the tile, in x,y slippy coords
        :param kwargs: Possible arguments required for the specific implementation
        :return: A list of the towers within this tile
        """
        raise NotImplementedError()

    def _log_predicted(self, tile_coord: Tuple[int, int]):
        """
        Logs that this tile was looked at for this run
        :param tile_coord: The x,y of the tile, in slippy coords
        """
        with self._connection:
            with self._connection.cursor() as cursor:
                try:
                    cursor.execute(
                        "INSERT INTO predicted (tile_id, run_id) VALUES (%s, %s)",
                        (f"{tile_coord[0]}_{tile_coord[1]}", self.run_id),
                    )
                except UniqueViolation as e:
                    print(e)
                    # This tile was already predicted on, probably a race condition?
                    print(f"unique violation for prediction! {tile_coord}")
                    pass

    def convert_local_tile_to_global_tile(self, y, x, gridsize):  # Numpy does rows, columns
        """
        Takes a tile in the costmap and converts it to a tile coord in the real world

        :param y: The y coords within the costmap
        :param x: The x coords within the costmap
        :param gridsize: The gridsize used to get the costmap
        :return: The x,y coord of the tile in the real world
        """
        topleft = (
            self.location_tile.x - gridsize // 2,
            self.location_tile.y - gridsize // 2,
        )
        return (x + topleft[0], y + topleft[1])

    def log_towers(self, towers, bought_tiles, spent_budget):
        """
        Logs towers found within one tile to the database

        :param towers: A list of towers to be logged
        :param bought_tiles: How many tiles have been bought before this tile was hit
        :param spent_budget: How much budget was spent to get to these found towers
        """
        if len(towers) > 0:
            for tower in towers:
                if not tower.new_tower:
                    print(f"tower was not new {time.time()}")
                    continue

                tower_line_id = tower.new_line_id or self.line_id

                with self._connection:
                    with self._connection.cursor() as cursor:
                        tile_id = f"{towers[0].location.x}_{towers[0].location.y}"
                        if tower.new_tower:
                            query_params = dict(
                                tile_id=tile_id,
                                score=tower.confidence,
                                bought_tiles=bought_tiles,
                                spent_budget=spent_budget,
                                run_id=self.run_id,
                                tmp_line_id=tower_line_id,
                                tower_uuid=tower.uuid,
                            )
                            geomstring = f"POINT( {tower.location.lng} {tower.location.lat} )"
                            query = (
                                f"INSERT INTO powertower ({','.join(query_params.keys())},geom) "
                                f"VALUES ("
                                f"{','.join(['%s'] * len(query_params.keys()))}, ST_GeomFromText('{geomstring}', 4326)"
                                f")"
                            )
                            try:
                                cursor.execute(query, list(query_params.values()))
                                print(f"Inserted at {time.asctime( time.localtime(time.time()) )}")
                            except UniqueViolation as e:
                                print(e)
                                # This tile already had towers?
                                print(
                                    f"unique violation! {tower.location.lng}, {tower.location.lat}"
                                )

                                # set tower uuid to be that of the existing tower

                                query = """SELECT tower_uuid FROM powertower where geom like ST_GeomFromText('%s')
                                    and run_id=%s""" % (
                                    geomstring,
                                    self.run_id,
                                )

                                try:
                                    cursor.execute(query)
                                    result = cursor.fetchall()
                                    tower.uuid = result[0][0]

                                    print(f"Inserted tower at {time.time()}")
                                except Exception:
                                    print("Cannot fetch tower uuid")

    def terminate(self, cause: str="200"):
        """
        Kill the tracer; the cause is just a message for logging
        :param cause: A string that is logged as the reason for termination.
        """
        termination_string = f"{cause}: Traced {len(self.distances)} towers over a distance of: {sum(self.distances)}"
        print(termination_string)
        return 9001

    def trace(self):
        """
        Calls the main trace function in a loop, trying a slightly different config each time (the retry scenarios)
        """
        retry = 0
        while retry <= len(self.RETRY_CONFIGS):
            retry = self._trace(retry)

    def _trace(self, retry:int=0):
        """
        The main trace call

        :param retry: How many times this trace has been retried from this location
        :return: The next retry value to use.
        """
        if retry > len(self.RETRY_CONFIGS): # If we are out of retry scenarios, terminate
            return self.terminate("404")
        elif retry == len(self.RETRY_CONFIGS):
            # This implements the final desperate attempt; trace back to the last high-confidence point and search for
            # other towers.
            retry_config = self.RETRY_CONFIGS[0]
            retry_config["budget_overwrite"] = (
                self.budget * 1.5
            )  # Give more budget to look really hard?
            if self.last_confident_location is None:
                return self.terminate("404")
            self.location = self.last_confident_location
            self.previous_location = (
                None  # Setting this to None, so we can set the starting direction manually
            )
            self.last_confident_location = (
                None  # Setting this to None, so we can set the starting direction manually
            )
            self.initial_direction = self.last_confident_direction

        else:
            retry_config = self.RETRY_CONFIGS[retry]

        use_this_gridsize = retry_config["costmap_config"]["gridsize_overwrite"] or self.gridsize

        local_cost_map = self.get_local_cost_map(
            radius=retry_config["costmap_config"]["radius_overwrite"] or self.radius,
            next_angle_limit=retry_config["costmap_config"]["next_angle_limit_overwrite"]
            or self.next_angle_limit,
            gridsize=use_this_gridsize,
            include_used_tiles=retry_config["costmap_config"].get("include_used_tiles", False),
            debug=False,
        )
        if local_cost_map is None:  # Costmap is outside base cost map
            return self.terminate(cause="off the edge")

        found_towers = []
        current_budget = retry_config["budget_overwrite"] or self.budget
        tiles_bought = []

        only_repeated_towers = 0

        # Repeat until budget runs out or we found a tower, AND always buy at least 3 tiles
        while (
            current_budget > 0
            or len(tiles_bought) < self.minimum_tiles_bought
        ):
            # "Buy" tile
            cheapest_tile_index = np.unravel_index(local_cost_map.argmin(), local_cost_map.shape)
            cheapest_tile_cost = local_cost_map[cheapest_tile_index]
            current_budget -= cheapest_tile_cost
            tiles_bought.append((cheapest_tile_index, cheapest_tile_cost))

            local_cost_map[
                cheapest_tile_index
            ] = np.Inf  # Make this tile super expensive, so we won't buy again
            # Check tile for tower
            tile_coord = self.convert_local_tile_to_global_tile(
                *cheapest_tile_index, gridsize=use_this_gridsize
            )
            tile, towers = self.get_tile(
                tile_coord,
                score_threshold=retry_config.get("score_threshold"),
                reuse_predictions=retry_config.get("reuse_predictions", True),
            )

            found_towers.extend(towers)

        if len(found_towers) > 1:
            found_towers, filtered_out = filter_towers_closer_than_x(found_towers)
            if filtered_out:
                print("Filtered towers that were too close!")

        if len(found_towers) > 1:  # We found a fork!
            # (This check happens again later, but this one is before logging, and the other one after logging the
            # towers to the DB, that's important

            # make these towers have new uuids
            current_direction = self.direction
            if current_direction is not None and self.line_id is not None:
                # If line_id is None, we're not continuing the line (because we have not really started one)
                direction_consistency = [
                    (
                        tower,
                        abs(
                            self.get_direction_between_points(self.location, tower.location)
                            - current_direction
                        ),
                    )
                    for tower in found_towers
                ]
                tower, consistency = min(direction_consistency, key=lambda x: x[1])
                if consistency < self.direction_consistency_threshold:
                    tower.new_line_id = self.line_id

            for tow in found_towers:
                if tow.new_line_id is None:
                    tow.new_line_id = self.get_new_line_id(
                        split_from=self.last_tower_uuid,
                        starting_tower=tow.uuid,
                        starting_point_id=self.starting_point_id,
                    )
            self.starting_point_id = None

        if len(found_towers) == 1 and self.line_id is None:
            # finding 1 tower from a startpoint - make a line
            self.line_id = self.get_new_line_id(
                starting_tower=found_towers[0].uuid,
                starting_point_id=self.starting_point_id,
                split_from=self.last_tower_uuid,
            )
            self.starting_point_id = None

        # Send discovered towers to database
        self.log_towers(found_towers, len(tiles_bought), self.budget - current_budget)

        if len(found_towers) > 0 and len([t for t in found_towers if not t.new_tower]) == len(
            found_towers
        ):
            only_repeated_towers += 1
        elif len(found_towers) > 0:
            only_repeated_towers = 0

        if only_repeated_towers > self.repeated_towers_threshold:
            warnings.warn(
                f"Found only known towers {only_repeated_towers} times in a row, so the line we're tracing was likely"
                " already known\n"
                "Terminating!"
            )
            return self.terminate("304 - known towers")

        # Update location, direction and radius
        # If no towers are found within the allocated budget, warn, then terminate
        if len(found_towers) == 0:
            # Search again, with adjuster params according to self.RETRY_CONFIGS
            self.terminate("504 - 404 but looked harder!")
            return retry + 1

        elif len(found_towers) > 1:
            # if len(set([tower.location.tile for tower in found_towers])) > 1:
            warnings.warn(
                f"Number of towers changed at {self.location}! So I'm splitting up!.\n"
                f"Terminating!"
            )
            moving_on_with_tower = None
            for tower in found_towers:
                if tower.new_line_id == self.line_id:
                    moving_on_with_tower = tower
                elif tower.new_tower:  # If the line_id is the same, we're continuing
                    assert tower.new_line_id is not None

                    if (
                        not self.last_tower_uuid
                    ):  # to handle case where subline splits up the first time it finds towers
                        self.last_tower_uuid = tower.uuid
                    self.queue.put(
                        TracerTask(
                            startpoint=tower.location,
                            direction=self.get_direction_between_points(
                                self.location, tower.location
                            ),
                            # weigh in the original radius to keep the radius from growing too much
                            radius=int(np.mean([self.radius, self.distances[0]])),
                            budget=self.budget,
                            line_id=tower.new_line_id,
                            last_tower_uuid=self.last_tower_uuid,
                        )
                    )

                    if self.line_id:  # Not in the case of startpoints
                        # I'm assuming we found a crossing? So let's also start looking at these points in the opposite
                        # direction
                        self.queue.put(
                            TracerTask(
                                startpoint=tower.location,
                                direction=self.get_direction_between_points(
                                    tower.location, self.location
                                ),
                                # weigh in the original radius to keep the radius from growing too much
                                radius=int(np.mean([self.radius, self.distances[0]])),
                                budget=self.budget,
                                line_id=self.get_new_line_id(  # Assign the other direction a new line id
                                    starting_tower=tower.uuid,
                                    split_from=self.last_tower_uuid,
                                    starting_point_id=self.starting_point_id,
                                ),
                                last_tower_uuid=self.last_tower_uuid,
                            )
                        )
            if moving_on_with_tower is None:
                return self.terminate("300 - fork")
            else:
                found_towers = [moving_on_with_tower]

        latest_distance = distance(
            (self.location.lat, self.location.lng),
            (found_towers[0].location.lat, found_towers[0].location.lng),
        ).meters
        radius_overwrite = retry_config["costmap_config"]["radius_overwrite"]
        if radius_overwrite is not None:
            self.distances.append(latest_distance / (radius_overwrite / self.radius))
        self.previous_direction = self.direction
        self.previous_location = self.location
        self.location = found_towers[0].location
        self.last_tower_uuid = found_towers[0].uuid

        if retry >= len(self.RETRY_CONFIGS) - 1:
            # The last real attempt that's not backtracking, up the tricky detection count
            self.found_towers_on_last_try += 1
            if self.found_towers_on_last_try > self.FOUND_TOWERS_ON_LAST_TRY_LIMIT:
                return self.terminate("418 - This might be a teapot")
        else:
            # Found a new tower with reasonable effort, so this line is probably OK
            self.found_towers_on_last_try = 0
            self.last_confident_direction = self.direction
            self.last_confident_location = self.location

        # trace again, from the next tower
        return 0

    def get_new_line_id(
        self,
        *,
        starting_tower: UUID,
        split_from: Optional[UUID] = None,
        starting_point_id: Optional[int] = None,
    ) -> UUID:
        """
        Get a new line id to log to (each single trace logs to a unique line-id) and logs this to the DB

        :param starting_tower: The current tower, from which the new line starts
        :param split_from: The previous tower, the last tower of a connecting line segment from which this line
            originated
        :param starting_point_id: The id of the first tower
        :return: The uuid for the new line
        """
        new_line_id = uuid4()

        print(
            "\n".join(
                [
                    str(v)
                    for v in [
                        new_line_id,
                        self.line_id,
                        split_from,
                        starting_tower,
                        starting_point_id,
                        self.run_id,
                    ]
                ]
            )
        )

        with self._connection:
            with self._connection.cursor() as cursor:
                try:
                    cursor.execute(
                        "INSERT INTO sublines (id, parent_line, split_from, starting_tower, startingpoint, run_id)"
                        " VALUES (%s, %s, %s, %s, %s, %s)",
                        (
                            new_line_id,
                            self.line_id,
                            split_from,
                            starting_tower,
                            starting_point_id,
                            self.run_id,
                        ),
                    )
                except Exception as e:
                    print("Couldn't make a subline!")
                    print(e)
        return new_line_id
