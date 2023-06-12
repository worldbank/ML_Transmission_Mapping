import time
from pathlib import Path

import mercantile
import numpy as np
from osgeo import gdal
from pyproj import Transformer


class CostMapInterface_from_file:
    """
    Little class to interact with a costmap file, without loading the whole thing in memory
    """
    def __init__(self, cost_map_path: Path):
        """
        :param cost_map_path: Path to the costmap raster
        """
        self.cost_map_path = cost_map_path
        ulx, xres, xskew, uly, yskew, yres = gdal.Open(str(self.cost_map_path)).GetGeoTransform()
        transformer = Transformer.from_crs("epsg:3857", "epsg:4326")
        ulx, uly = transformer.transform(ulx, uly)
        topleft_tile = mercantile.tile(lng=uly, lat=ulx, zoom=18)
        self.topleft = (topleft_tile.x, topleft_tile.y)

    def get_cost(self, x, y):
        """
        Get the costmap value at this position
        :param x:
        :param y:
        """
        rx, ry = (x - self.topleft[0], y - self.topleft[1])
        ds = gdal.Open(str(self.cost_map_path))
        costmap = ds.GetRasterBand(1)
        cost = costmap.ReadAsArray(rx, ry, 1, 1)
        return cost[0][0]

    def get_cost_grid(self, x, y, grid_size=11):
        """
        Get's the subsection of the costmap around x,y to be used for the local cost map

        :param x: Center x
        :param y: Center y
        :param grid_size: The size of the grid with x,y at it's center
        :return: A grid_size x grid_size array with costs
        """
        rx, ry = (x - self.topleft[0], y - self.topleft[1])
        for retry in range(20):
            ds = gdal.Open(str(self.cost_map_path))
            if ds is not None:
                break
            else:
                time.sleep(1)
        costmap = ds.GetRasterBand(1)
        cost = costmap.ReadAsArray(
            max(int(rx - np.floor(grid_size / 2)), 0),
            max(int(ry - np.floor(grid_size / 2)), 0),
            min((ds.RasterXSize - int(rx - np.floor(grid_size / 2))), grid_size),
            min((ds.RasterYSize - int(ry - np.floor(grid_size / 2))), grid_size),
        )
        return cost
