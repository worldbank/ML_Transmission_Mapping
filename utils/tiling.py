import dataclasses
from collections import namedtuple
from typing import List

import mercantile
from osgeo import ogr, osr

from utils.data_utils import createBoundingPoly

BasicPoint = namedtuple("Point", ["lng", "lat"])


@dataclasses.dataclass
class PatchInfo:
    """
    Dataclass to hold information about the patches to be requested from mapbox
    """

    lngmin: float
    latmin: float
    lngmax: float
    latmax: float
    nlng: int
    nlat: int
    srs_wkt: str

    @property
    def tile_index(self):
        return str(round(self.lngmin, 14)) + "_" + str(round(self.latmax, 14))

    @property
    def width(self):
        return abs(self.lngmax - self.lngmin)

    @property
    def height(self):
        return abs(self.latmax - self.latmin)

    @property
    def pixel_size(self):
        PixelSize = namedtuple("PixelSize", ["width", "height"])
        return PixelSize(self.width / self.nlng, self.height / self.nlat)

    @property
    def center_point(self):
        """
        To make sure we get the tile we want, we reference the middle of the box, rather than a shared edge
        :return:
        """
        return BasicPoint(self.lngmin + self.width / 2, self.latmax - self.height / 2)

    @property
    def geotransform(self):
        return [
            self.lngmin,
            self.pixel_size.width,
            0,
            self.latmax,
            0,
            -self.pixel_size.height,  # Let's assume we're north-up
        ]

    def __eq__(self, other):
        return self.tile_index == other.tile_index

    def __hash__(self):
        return self.tile_index.__hash__()


def create_patch_extent_list(
    *,
    aoi_geoms_wkt: List,
    tile_size: int,
    zoom: int,
    aoi_epsg: int,
    aoi_relation: str = "intersect",
    verbose: int = 0,
):
    """
    Function that creates a list of patches that can be created within the passed aoi's
    This function only accepts named keyword arguments

    :param aoi_geoms_wkt: List of aoi geometries in wkt format
    :param tile_size: Number of pixels in a tile
    :param pixel_size: Real-world size of a pixel
    :param zoom: Zoom level https://wiki.openstreetmap.org/wiki/Zoom_levels
    :param aoi_epsg: The epsg of the aoi
    :param aoi_relation: Intersect, Within, or Contains, how to check what tiles should be included
    :return: List of PatchInfo objects
    """

    assert aoi_relation in [
        "intersect",
        "within",
        "contains",
    ], "aoi_relation has to be 'intersect' or 'within' or 'contains'"

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(aoi_epsg)
    srs_wkt = srs.ExportToWkt()

    # conver to geometry
    aoi_geoms = []
    if aoi_geoms_wkt is not None:

        for aoi_geom_wkt in aoi_geoms_wkt:
            aoi_geom = ogr.CreateGeometryFromWkt(aoi_geom_wkt).Buffer(0)
            aoi_geoms.append(aoi_geom)

    patch_extent_list = []

    for aoi_geom in aoi_geoms:

        if aoi_geom is None:
            print("corrupt intersected aoi geom!!")
            continue

        lngmin_geom, lngmax_geom, latmin_geom, latmax_geom = aoi_geom.GetEnvelope()
        if verbose:
            print(
                "aoi geom has envelope {} ".format(
                    str([lngmin_geom, lngmax_geom, latmin_geom, latmax_geom])
                )
            )

        # find lngmin, latmax of top left patch to be generatated
        tile = mercantile.tile(lng=lngmin_geom, lat=latmax_geom, zoom=zoom)
        bounds = mercantile.bounds(tile)
        # add small eps to avoid fetching neighbours

        lngmin = bounds.west
        if verbose:
            print(
                "bounds of top left patch to be created {}".format(
                    str([bounds.north, bounds.south])
                )
            )

        # initialize stuff for the while loop
        lngmin_patch = bounds.west
        latmax_patch = bounds.north

        while True:
            tile = mercantile.tile(lng=lngmin_patch, lat=latmax_patch, zoom=zoom)
            bounds = mercantile.bounds(tile)

            patch_info = PatchInfo(
                lngmin=bounds.west,
                lngmax=bounds.east,
                latmin=bounds.south,
                latmax=bounds.north,
                nlat=tile_size,
                nlng=tile_size,
                srs_wkt=srs_wkt,
            )

            patch_poly = createBoundingPoly(
                xmin=bounds.west, xmax=bounds.east, ymin=bounds.north, ymax=bounds.south
            )

            insert = False

            # check if patch poly intersects with aoi geom
            if aoi_relation == "intersect":
                if patch_poly.Intersects(aoi_geom):
                    insert = True

            # check if patch_poly is within aoi geom
            elif aoi_relation == "within":
                if patch_poly.Within(aoi_geom):
                    insert = True

            # check if aoi_geom is within patch_poly
            elif aoi_relation == "contains":
                if patch_poly.Contains(aoi_geom):
                    insert = True

            if insert:
                patch_extent_list.append(patch_info)
            else:
                pass

            # update lngmin of subset for correct GeoTransform
            lngmin_patch = bounds.east

            # if pixel number of e_end is more than number of columns
            if lngmin_patch > lngmax_geom:
                # update lngmin and latmax of subset for correct GeoTransform
                lngmin_patch = lngmin
                latmax_patch = bounds.south

            if latmax_patch < latmin_geom:
                break

    # remove duplicates from list
    patch_extent_list_clean = set(patch_extent_list)
    return patch_extent_list_clean
