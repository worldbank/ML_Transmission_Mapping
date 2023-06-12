import contextlib
import dataclasses
import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import mercantile
import numpy as np
from osgeo import gdal, ogr, osr


@contextlib.contextmanager
def gdalOpen(filepath):
    gdalFile = gdal.Open(filepath)
    yield gdalFile
    # gdalFile = None # When the context closes, gdalFile is automatically garbage collected


def rebin(a, shape):
    """
    function that does average resampling,
    only works if new shape is multiple of old shape and if shape has 2 dimensions

    :param a: input image as array
    :param shape: The desired shape
    :return: A resized array
    """

    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def read_gdal_file(
    file_path: Union[str, Path],
    shape: Tuple[int, int, int],
    dtype,
    bands: Optional[List[int]] = None,
    load_geo_transforms: bool = False,
):
    """
    Reads a gdal file. Label files or input files.
    If it is a label file, it will also read the mask bands, and create a mask_array, if mask_params is passed

    Args:
        file_path: path to the file to be loaded
        shape: the shape of the output arrays
        dtype: the dtype of the output array
        bands: Optional, a list of integers denoting which bands to read in what order
        load_geo_transforms: whether to obtain the geo transformations or not
        mask_params: The parameters used to create the mask_array

    Returns:
        a dict with at least "img" and "corrupt" and optionally "geo_transforms" and "mask_array"
    """

    img = np.ones(shape, dtype=dtype)
    geo_transforms = []
    bands = bands or list(
        range(shape[-1])
    )  # If bands is None, bands follows desired output shape (num_classes)
    total_bands = bands

    with gdalOpen(file_path) as gdal_dataset:
        if load_geo_transforms:
            geo_transform = gdal_dataset.GetGeoTransform()
            geo_transforms.append(geo_transform)

        corrupt = False

        for band_index, band in enumerate(total_bands):
            try:
                raster_band = gdal_dataset.GetRasterBand(band)
            except Exception:
                warnings.warn("could not open: " + file_path)
                corrupt = True
            if not corrupt:
                try:
                    band_data = raster_band.ReadAsArray().astype(dtype)
                    if not band_data.shape[0] == shape[0]:
                        if band_data.shape[0] % shape[0] == 0:
                            print("rebin")
                            band_data = rebin(
                                band_data,
                                shape[:2],
                            )
                        else:
                            diff_size = round((band_data.shape[0] - shape[0]) / 2)
                            band_data = band_data[
                                diff_size : shape[0] + diff_size,
                                diff_size : shape[0] + diff_size,
                            ]

                    img[:, :, band_index] = band_data
                except Exception:
                    print("patch does not have band {}".format(str(band)))
                    corrupt = True

    return_vals = {"img": img, "corrupt": corrupt}
    if load_geo_transforms:
        return_vals["geo_transforms"] = geo_transforms
    return return_vals


def read_image_from_tif(path, patch_size, bands_list, dtype=np.float32):
    """
    Reads a tif file

    :param path: Path to the image
    :param patch_size: Size of the image
    :param bands_list: which bands to read
    :param dtype: as which datatype to return the data
    :return: The image, if reading succeeded, and the geotransform of the image
    """
    if isinstance(patch_size, int):
        shape_X = (
            patch_size,
            patch_size,
            len(bands_list),
        )
    else:
        shape_X = (*patch_size, len(bands_list))
    results = read_gdal_file(
        file_path=str(path),
        shape=shape_X,
        dtype=dtype,
        load_geo_transforms=True,
        bands=bands_list,
    )
    imgX, this_corrupt, geotransform = results["img"], results["corrupt"], results["geo_transforms"]
    return imgX, this_corrupt, geotransform


def createBoundingPoly(xmin, ymax, xmax, ymin):
    """Creates ogr polygon geometry from bounding box"""
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(xmin, ymax)
    ring.AddPoint(xmax, ymax)
    ring.AddPoint(xmax, ymin)
    ring.AddPoint(xmin, ymin)
    ring.AddPoint(xmin, ymax)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly


def WKTlist_to_shp(wkts, outfile, EPSG, layername="", overwrite=False, geom_type=ogr.wkbPolygon):
    """
    convert a list of wkt to shapfile (only for polygons)

    :param wkts: a wkt list of wkt
    :param outfile: output shapefile location
    :param EPSG: output EPSG
    :param layername: Name of the layer
    :param overwrite: Overwrite or not
    :param geom_type: The type of geometry of the layer
    """
    output_path = Path(outfile)
    if output_path.exists():
        if not overwrite:
            raise ValueError(f"{output_path} already exists!")
        else:
            for file in output_path.parent.glob(output_path.with_suffix(".*").name):
                os.remove(file)

    # layer Definition
    srs = osr.SpatialReference()  # create srs
    srs.ImportFromEPSG(EPSG)

    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(str(output_path))
    lyr = ds.CreateLayer(layername, srs, geom_type)
    fdefn = lyr.GetLayerDefn()

    if isinstance(wkts, str):
        wkts = [wkts]

    for a_wkt in wkts:
        # create Geomentry from wkt
        geom = ogr.CreateGeometryFromWkt(a_wkt)
        # create shapefiles
        feature = ogr.Feature(fdefn)
        feature.SetGeometry(geom)
        lyr.CreateFeature(feature)

        # Destroy the feature to free resources
        feature.Destroy()

    # Destroy the data source to free resources
    ds.Destroy()


def get_geom_wkt_list(shp, intersection_wkt=None, attribute_name=None, attribute_value=None):
    """function that reads all the geometries in a shapefile into list of wkt geoms

    shp <str>:                      path of shapefile
    intersection_wkt <wkt>:         only get geoms hat intersect with this wkt
    attribute_name <str>:           allows to set filter on cerain attribute name
    attribute_value:                filter only geoms where attribute_name has this value
    """

    # assertions
    if attribute_name:
        assert (
            attribute_value
        ), "if an attribute name is supplied, an attribute_value to filter on is also required"

    if intersection_wkt is not None:
        intersection_geom = ogr.CreateGeometryFromWkt(intersection_wkt)

    geoms_wkt = []
    ds = ogr.Open(shp)
    assert ds is not None, "could not open aoi shapefile"
    layer = ds.GetLayer(0)
    for feat in layer:
        geom = feat.GetGeometryRef()

        if geom is None:
            print("feature has no geometry, will be skipped")
            continue

        if attribute_name:
            attribute = feat.GetField(attribute_name)
            if attribute != attribute_value:
                continue

        if intersection_wkt is not None:
            if not geom.Intersects(intersection_geom):
                continue

        wkt = geom.ExportToWkt()

        geoms_wkt.append(wkt)

    ds = None

    print("total of {} geoms".format(len(geoms_wkt)))
    return geoms_wkt


def stretch_to_byt(patch_data):
    """stretches 0-1 in float to 1-255"""

    # stretch patch_date to 1-255
    patch_data_str = (patch_data - 0) / 1 * (255 - 1) + 1
    patch_data_str = np.array(patch_data_str, dtype=np.uint8)

    return patch_data_str


@dataclasses.dataclass
class SlippyPoint:
    """
    data container to hold information on a point detection
    This implementation is used for the powertower project. A more generic baseclass could be defined when we go into
    other usecases.
    """

    lng: float
    lat: float
    zoom_level: int = 18
    tile_size: int = 512
    label: str = "powertower"  # Will fix this when we get different classnames

    def __post_init__(self):
        """
        Set additional properties based on the provided data
        """
        self.tile = mercantile.tile(lat=self.lat, lng=self.lng, zoom=self.zoom_level)
        bounds = mercantile.bounds(self.tile)
        self.tile_x = self.tile.x
        self.tile_y = self.tile.y
        self.x = np.round((self.lng - bounds.west) / (bounds.east - bounds.west) * self.tile_size)
        self.y = np.round(
            (self.lat - bounds.north) / (bounds.south - bounds.north) * self.tile_size
        )

    @classmethod
    def from_list(cls, lst, zoom_level, tile_size):
        """
        Construct multiple points from a single list containing point data
        Args:
            lst: The list of points
            zoom_level: The zoom level at which they were detected (needed to determine the slippytile properties)
            tile_size: The tile size in pixels
        Returns:
        A list of SlippyPoint objects
        """
        if len(lst) == 0:
            return None
        if len(lst) == 1 and isinstance(lst[0], list) and len(lst[0]) == 2:
            return cls(*lst[0], zoom_level=zoom_level, tile_size=tile_size)
        if len(lst) == 2 and not isinstance(lst[0], list):
            return cls(*lst, zoom_level=zoom_level, tile_size=tile_size)
        else:
            return [cls.from_list(p, zoom_level=zoom_level, tile_size=tile_size) for p in lst]


@dataclasses.dataclass
class Annotation:
    """
    A container holding information on a groundtruth point of data
    Currently this just supports points, as this is the only current usecase
    """

    visibility: str
    point: SlippyPoint
    set: str

    def to_array(self):
        return [self.point.x, self.point.y]

    @classmethod
    def from_dict(cls, dct, zoom_level, tile_size):
        properties = dct.get("properties", {})
        visibility = properties.get("visibility", "")
        if visibility is None:  # Yes, I don't know how this happens either, but somehow it does?
            visibility = ""
        return cls(
            visibility=visibility.lower(),
            point=SlippyPoint.from_list(
                dct.get("geometry", {}).get("coordinates", []),
                zoom_level=zoom_level,
                tile_size=tile_size,
            ),
            set=properties.get("set", "trainval") or "trainval",
        )

    @property
    def tilename(self):
        return f"{self.point.tile_x}_{self.point.tile_y}"

    def __repr__(self):
        return f"{self.tilename} - {self.point.lng}_{self.point.lat}"

    @property
    def visualisation_colour(self):
        return {"": (0, 255, 0), "poor": (100, 200, 0), "very poor": (60, 150, 90)}[self.visibility]

    def visualise(self, image):
        cv2.circle(image, [int(self.point.x), int(self.point.y)], 10, self.visualisation_colour, 2)
        return image
