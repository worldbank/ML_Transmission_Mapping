from typing import Tuple

import mercantile
import numpy as np
from osgeo import gdal
from pyproj import Transformer


class CostMapInterface_from_memory:
    """
    Little class to interact with a costmap that is entirely loaded in memory
    """

    def __init__(self, cost_map: np.ndarray, topleft_xy: Tuple[int, int]):
        """
        :param cost_map: The entire costmap as an array
        :param topleft_xy: The coord of the topleft tile
        """
        self.cost_map = cost_map
        self.topleft = topleft_xy

    def get_cost(self, x, y):
        """
        Get the cost at this precise location
        """
        rx, ry = (x - self.topleft[0], y - self.topleft[1])
        return self.cost_map[rx, ry]

    def get_cost_grid(self, x, y, grid_size=11):
        """
        Gets the subsection of the costmap around x,y to be used for the local cost map

        :param x: Center x
        :param y: Center y
        :param grid_size: The size of the grid with x,y at it's center
        :return: A grid_size x grid_size array with costs
        """
        rx, ry = (x - self.topleft[0], y - self.topleft[1])
        cost = self.cost_map[
            int(rx - np.floor(grid_size / 2)) : int(rx + np.ceil(grid_size / 2)),
            int(ry - np.floor(grid_size / 2)) : int(ry + np.ceil(grid_size / 2)),
        ]
        return cost
