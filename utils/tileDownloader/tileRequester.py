import json
import os
import tempfile
from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List, Optional, Tuple

import boto3
import mercantile
import numpy as np
import psycopg2
from osgeo import gdal, gdal_array, ogr, osr
from psycopg2.errors import UniqueViolation
from tqdm import tqdm

from database.database_utils import get_db_config
from utils.data_utils import (
    Annotation,
    WKTlist_to_shp,
    get_geom_wkt_list,
    read_image_from_tif,
    stretch_to_byt,
)
from utils.dataclasses.dataParams import DataParams
from utils.dataclasses.point import Point
from utils.dataclasses.tower import Tower
from utils.tiling import create_patch_extent_list
from utils.tracer_tools import mercantile_to_WKT


class TileRequester(ABC):
    """
    The abstract baseclass of the tile requester. Implementations of this class can request images from a tileserver,
    caching them to disk in case of soon reuse.
    """
    def __init__(
        self,
        api_key: str,
        storage_config: dict,
        database_config: Optional[str],
        run_id: Optional[int],
    ):
        """
        :param api_key: The key for the api
        :param storage_config: A config defining where the images should be cached to
        :param database_config: The database config, to this database, it is logged which images are downloaded,
        and possibly previously predicted towers are re-used.
        :param run_id: The run_id of the current run
        """
        self.api_key = api_key
        self.zoom = 18
        self._connection = psycopg2.connect(**get_db_config(database_config))
        self.database_config = database_config
        self.epsg = 4326
        self.higher_dpi = True  # I don't think we would ever not want this?

        self.storage_config = storage_config

        self.run_id = run_id
        self.mock_mode = False

        assert storage_config["location"] in ["AWS", "LOCAL"]
        self.storage_location = storage_config["location"]
        self.output_folder = Path(storage_config.get("output_folder"))

        if self.storage_location == "AWS":
            self.aws_access_key_id = storage_config["aws_access_key_id"]
            self.aws_secret_access_key = storage_config["aws_secret_access_key"]
            self.s3_bucket_name = storage_config["s3_bucket_name"]

            self.s3 = boto3.client(
                "s3",
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
            )

            aws_config = {
                "AWS_SECRET_ACCESS_KEY": self.aws_secret_access_key,
                "AWS_ACCESS_KEY_ID": self.aws_access_key_id,
            }
            for k, v in aws_config.items():
                gdal.SetConfigOption(k, v)

    @property
    @abstractmethod
    def tile_size(self):
        raise NotImplementedError()

    def get_tiles_from_slippy_list(
        self, slippy_list, output_dir, workers=0, skip_existing=True, dry_run_dir=None
    ):
        """
        Given a list of slippy tiles, download all these images (useful to gather a training dataset)

        :param slippy_list: A list of slippytiles
        :param output_dir: The folder to store them to
        :param workers: How many to do in parallel
        :param skip_existing: Whether to re-download existing tiles
        :param dry_run_dir: When this is given, only create a little shape file with the footprint of the tile and
        place it in this location.
        """
        self.output_folder = Path(output_dir)
        if dry_run_dir is not None and workers > 0:
            raise ValueError("Number of workers should be 0 when doing a dry-run")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Storing patches to: {output_path}")

        if workers == 0:
            if dry_run_dir is not None:
                # Store the footprint of the tile to a shp file.
                polies = []
                for tile in tqdm(slippy_list):
                    bounds = mercantile.bounds(tile)
                    polies.append(
                        [
                            [bounds.west, bounds.north],
                            [bounds.west, bounds.south],
                            [bounds.east, bounds.south],
                            [bounds.east, bounds.north],
                            [bounds.west, bounds.north],
                        ]
                    )
                Path(dry_run_dir).mkdir(exist_ok=True)
                geom = self._polygons_to_geometry(polies)
                WKTlist_to_shp(
                    [geom.ExportToWkt()],
                    outfile=str(Path(dry_run_dir) / "tiles.shp"),
                    EPSG=self.epsg,
                    geom_type=ogr.wkbMultiPolygon,
                )
            else:
                # Actually donload the images
                for tile_spot in slippy_list:
                    tile_image, _ = self.get_tile(tile_spot.x, tile_spot.y)
                    image_path = output_path / (f"{tile_spot.x}_{tile_spot.y}.tif")
                    if skip_existing and image_path.is_file():
                        continue
                    image_array = np.asarray(tile_image)[..., :3]
                    self.save_patch_to_file(
                        image_array,
                        image_path,
                        DataParams(**{"patchSize": self.tile_size, "buffer": 0}),
                        clipBufferBool=False,
                        gt=self._geotransform_from_tile(tile_spot),
                        epsg=self.epsg,
                        bands_first=False,
                    )
        else:
            def _download_patch(tile_spot):
                """
                Function for a single worker to execute that downloads a tile
                :param tile_spot: The tile to download
                """
                try:
                    tile_image, _ = self.get_tile(x=tile_spot.x, y=tile_spot.y, new_connection=True)
                    image_path = output_path / (f"{tile_spot.x}_{tile_spot.y}.tif")
                    if skip_existing and image_path.is_file():
                        return
                    image_array = np.asarray(tile_image)
                    self.save_patch_to_file(
                        image_array,
                        image_path,
                        DataParams(**{"patchSize": self.tile_size, "buffer": 0}),
                        clipBufferBool=False,
                        gt=self._geotransform_from_tile(tile_spot),
                        epsg=self.epsg,
                        bands_first=False,
                    )
                except Exception as e:
                    print(e)

            with ThreadPool(workers) as p:
                list(
                    tqdm(
                        p.imap_unordered(_download_patch, slippy_list, chunksize=10),
                        total=len(slippy_list),
                    )
                )

    def _geotransform_from_tile(self, tile):
        """
        Creates a geotransfrom tuple from a tile location
        :param tile: (x,y) for the slippy tile
        :return: The tuple defining the geotransform
        """
        bounds = mercantile.bounds(tile)
        return [
            bounds.west,
            (bounds.east - bounds.west) / self.tile_size,  # pixelsize
            0,
            bounds.north,
            0,
            -abs(bounds.north - bounds.south) / self.tile_size,  # Let's assume we're north-up
        ]

    def get_tiles_from_aoi(
        self,
        aoi_shp_path,
        output_dir,
        workers=0,
        dry_run_dir: Optional[str] = None,
        skip_existing=True,
    ):
        """
        Get and save all mapbox tiles contained in an AOI. When a "dry_run_dir" is passed, the tiles are not downloaded,
        but instead a shape file is created, which shows the position of all tiles on the map.
        :param aoi_shp_path: The AOI to download the tiles for
        :param output_dir: The directory to which to store the images
        :param dry_run_dir: The directory where the dryrun shape file should be stored
        """
        assert Path(aoi_shp_path).is_file(), f"{aoi_shp_path} does not exists"
        if dry_run_dir is not None:
            assert workers == 0, "Workers should be 0 when doing a dryrun"

        # check if there is attribute_name and value to take into account
        aoi_geoms_wkt = get_geom_wkt_list(aoi_shp_path)

        patch_list = create_patch_extent_list(
            aoi_geoms_wkt=aoi_geoms_wkt,
            tile_size=self.tile_size,
            aoi_epsg=self.epsg,
            aoi_relation="intersect",
            zoom=self.zoom,
        )
        print(f"Creating {len(patch_list)} patches from aoi: {aoi_shp_path}")

        slippy_tile_list = []
        for patch in patch_list:
            point = patch.center_point
            slippy_tile_list.append(mercantile.tile(lng=point.lng, lat=point.lat, zoom=self.zoom))

        self.get_tiles_from_slippy_list(
            slippy_tile_list, output_dir, workers, skip_existing, dry_run_dir
        )

    def get_tiles_from_geojson(
        self, geojson_path, output_dir, workers=0, skip_existing=True, dry_run_dir=None
    ):
        """
        wraps get_tiles_from_slippy_list by creating such a list with tiles that contain a point in the geojson

        :param geojson_path: Path to a geojson file with towers
        :param output_dir: Location to which the tiles should be stored
        :param workers: How many tiles to download in parallel
        :param skip_existing: Whether or not to overwrite existing tiles
        :param dry_run_dir: When this is provided, a shape file of all the outlines is created instead of downloading
        the actual images.
        :return:
        """
        geojson_path = Path(geojson_path)
        with geojson_path.open("r") as f:
            labels = json.load(f)
        full_label_list = [
            Annotation.from_dict(lbl, zoom_level=self.zoom, tile_size=self.tile_size)
            for lbl in labels.get("features", [])
            if lbl.get("properties", {}).get("power", "tower") == "tower"
            and lbl.get("geometry") is not None
        ]

        xs_ys = [(label.point.tile_x, label.point.tile_y) for label in full_label_list]

        xmin = min(xs_ys, key=lambda x: x[0])[0]
        ymin = min(xs_ys, key=lambda x: x[1])[1]
        xmax = max(xs_ys, key=lambda x: x[0])[0]
        ymax = max(xs_ys, key=lambda x: x[1])[1]

        with self._connection:
            with self._connection.cursor(cursor_factory=None) as cursor:
                cursor.execute(
                    "SELECT id "
                    "FROM tile "
                    f"WHERE x>={xmin} AND x<={xmax} AND y>={ymin} AND y <= {ymax}"
                )
                tile_ids = cursor.fetchall()

        tile_ids = [tile_id.id for tile_id in tile_ids]

        slippy_tile_list = []
        for x, y in xs_ys:
            if f"{x}_{y}" not in tile_ids:
                slippy_tile_list.append(mercantile.Tile(x, y, self.zoom))
        print(
            f"Skipped {len(full_label_list)-len(slippy_tile_list)} tiles, because they were already in the database"
        )

        self.get_tiles_from_slippy_list(
            slippy_tile_list[::-1], output_dir, workers, skip_existing, dry_run_dir
        )

    def get_tile(self, x, y, new_connection=None):
        """
        x and y are slippy map tilenames:
        https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames

        Gets the tile, but checks the cache first
        :param x:
        :param y:
        :param new_connection: Used for workers to refresh the connection
        :return: Pillow image of tile at x,y

        Also returns towers, either a list, or None if no prediction happened yet
        towers can be an empty list, this means prediction was done, but no towers were found
        """
        key = f"{x}_{y}"
        img, towers = self.check_database(key, new_connection=new_connection)
        if img is None:
            img = self._request_tile(x, y)
            if img is not None and img.shape[0] == 3:
                img = img.transpose([1, 2, 0])
            self.insert_tile_to_database(img, x, y, new_connection=new_connection)
        if img is not None and img.shape[0] == 3:
            img = img.transpose([1, 2, 0])
        return img, towers

    def insert_tile_to_database(self, image, x, y, new_connection=False):
        """
        Whenever we have downloaded a new tile, we store the image, and update the database with its location
        :param image: The image that was downloaded
        :param x:
        :param y:
        :param new_connection: Used for workers to refresh the connection
        :return:
        """
        output_path = self.output_folder / f"{x}_{y}.tif"
        if self.storage_location == "AWS":
            dbpath = "s3://" + self.s3_bucket_name + "/" + str(output_path)
        else:
            dbpath = str(output_path)
        tile_spot = mercantile.Tile(x=x, y=y, z=self.zoom)
        if not self.mock_mode:
            self.save_patch_to_file(
                np.asarray(image),
                output_path,
                DataParams(**{"patchSize": self.tile_size, "buffer": 0}),
                clipBufferBool=False,
                gt=self._geotransform_from_tile(tile_spot),
                epsg=self.epsg,
                bands_first=False,
            )
        geomstring = mercantile_to_WKT(tile_spot)
        query_params = {"id": f"{x}_{y}", "x": int(x), "y": int(y), "file_path": str(dbpath)}
        query = (
            f"INSERT INTO tile ({','.join(query_params.keys())}, geom) "
            f"VALUES ({','.join(['%s'] * len(query_params.keys()))}, ST_GeomFromText('{geomstring}', 4326))"
        )
        connection = self._connection
        if new_connection:
            connection = psycopg2.connect(**get_db_config(self.database_config))
        try:
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute(query, tuple(query_params.values()))
        except UniqueViolation:
            pass

    def check_database(self, key: str, new_connection=False):
        """
        Check if a certain tile_id already exists in the database, and gets the result + possible towers contained
        within

        :param key: The tile_id (f"{x}_{y}")
        :param new_connection: Used for workers to refresh the connection
        :return: The image of the tile, and possible towers contained within
        """
        connection = self._connection
        if new_connection:
            connection = psycopg2.connect(**get_db_config(self.database_config))
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT predicted.run_id, tile.file_path "
                    "FROM tile LEFT OUTER JOIN predicted ON tile.id = predicted.tile_id "
                    "WHERE tile.id = (%s)",
                    (key,),
                )
                predictions = cursor.fetchall()
        if len(predictions) == 0:
            return None, None # This tile has not been previously downloaded
        else:
            file_path = predictions[0].file_path
            tile_image = self._load_file(file_path)

        if self.run_id not in [pred.run_id for pred in predictions]:
            return tile_image, None # The tile has been downloaded, but has not been predicted on *in this run*
        else:
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT ST_X(geom) as lng, ST_Y(geom) as lat, score, tower_uuid "
                        "FROM powertower "
                        "WHERE tile_id=(%s) AND run_id=(%s)",
                        (key, self.run_id),
                    )
                    towers = cursor.fetchall()

            found_towers = []
            for tower in [t for t in towers if t.lng is not None]:
                found_towers.append(
                    Tower(
                        location=Point(lng=tower.lng, lat=tower.lat),
                        confidence=tower.score,
                        new_tower=False,
                        uuid=tower.tower_uuid,
                    )
                )
            return tile_image, found_towers # The tile has been downloaded before, and we predicted on it in this run!

    @abstractmethod
    def _request_tile(self, x, y):
        """
        Actually requests the tile, without checking the cache

        :param x:
        :param y:
        :return: Pillow Image
        """
        raise NotImplementedError()

    def get_tile_from_coords(self, *, long, lat):
        """
        Gets the tile at latitude and longitude
        :param long:
        :param lat:
        :return: Pillow image of tile at coordinate
        """
        tile = mercantile.tile(lng=long, lat=lat, zoom=self.zoom)
        return self.get_tile(tile.x, tile.y)


    @staticmethod
    def _polygons_to_geometry(inputs: List[List[List[int]]]):
        """
        Converts a list of lists of x,y coords to a gdal geometry

        :param inputs: The list of list of x,y coords
        :return: A gdal geometry
        """
        multi_shape = ogr.Geometry(ogr.wkbMultiPolygon)

        for polygon in inputs:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for point in polygon:
                ring.AddPoint(*point)
            poly_geom = ogr.Geometry(ogr.wkbPolygon)
            poly_geom.AddGeometry(ring)
            multi_shape.AddGeometry(poly_geom)

        return multi_shape

    def _load_file(self, file_path):
        """
        Reads a file as an array at file_path
        :param file_path: The path to the file, when self.storage_location is AWS, it should be an s3 path, and
        otherwise it should be local.
        :return: An array with the image
        """
        loaded_file = None
        if self.storage_location == "AWS":
            loaded_file = self._load_file_s3(file_path)
        elif self.storage_location == "LOCAL" or loaded_file is None:
            loaded_file = self._load_file_local(file_path)
        return loaded_file

    def save_patch_to_file(
        self,
        output_tile_array,
        outputfilename,
        dataParams: DataParams,
        clipBufferBool: bool,
        gt: Tuple,
        epsg: int,
        bands_first: bool=True,
        verbose: int=0,
    ):
        """
        Function that stores an image to a tiff file.

        :param output_tile_array: The image array to save
        :param outputfilename: The location to put it in
        :param dataParams: an instance of DataParams, defining the resolution, buffer and total image size.
        :param clipBufferBool: If true, the buffer is clipped from the result
        :param gt: The geotransform tuple
        :param epsg: The projection
        :param bands_first: If true; bands first, otherwise bands last
        :param verbose: How much to print
        """
        try:
            if self.storage_location == "AWS":
                return self._save_patch_to_file_s3(
                    output_tile_array,
                    outputfilename,
                    dataParams,
                    clipBufferBool,
                    gt,
                    epsg,
                    bands_first,
                    verbose,
                )
            elif self.storage_location == "LOCAL":
                return self._save_patch_to_file_local(
                    output_tile_array,
                    outputfilename,
                    dataParams,
                    clipBufferBool,
                    gt,
                    epsg,
                    bands_first,
                    verbose,
                )
        except Exception as e:
            print(e)
            # Tracing should never crash on this

    def _save_patch_to_file_local(
        self,
        output_tile_array,
        outputfilename,
        dataParams: DataParams,
        clipBufferBool,
        gt,
        epsg,
        bands_first=True,
        verbose=0,
    ):
        """
        Implements save_patch_to_file for local storing
        """
        _output_tile_array = np.copy(output_tile_array)
        if len(output_tile_array.shape) == 4:  # includes batch dimension
            assert (
                output_tile_array.shape[0] == 1
            ), "output_tile_array has a batch dimension larger than 1, please pass a single image"
            _output_tile_array = _output_tile_array[0, ...]

        # stretch patch_date to 1-255
        if verbose:
            print(_output_tile_array.shape)
        if _output_tile_array.dtype != np.uint8:  # Only scale when needed
            _output_tile_array = stretch_to_byt(_output_tile_array)

        # numbands
        if verbose:
            print(_output_tile_array.shape)
        if bands_first:
            numBands = _output_tile_array.shape[0]
        else:
            numBands = _output_tile_array.shape[-1]

        if clipBufferBool is True:
            patchsize = dataParams.patchSize - dataParams.buffer
            gt = [
                gt[0] + (dataParams.buffer * dataParams.resolution) / 2,
                gt[1],
                gt[2],
                gt[3] - (dataParams.buffer * dataParams.resolution) / 2,
                gt[4],
                gt[5],
            ]
            # write to aray
            halfBuf = int(dataParams.buffer / 2)
            _output_tile_array = _output_tile_array[
                :,
                halfBuf : dataParams.patchSize - halfBuf,
                halfBuf : dataParams.patchSize - halfBuf,
            ]
        else:
            # write to aray
            if verbose:
                print("writing")
            patchsize = dataParams.patchSize

        try:
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(epsg)
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(
                str(outputfilename),
                patchsize,
                patchsize,
                numBands,
                gdal.GDT_Byte,
                options=["COMPRESS=DEFLATE"],
            )

            if srs.ExportToWkt() == "":
                raise Exception("osr module not working properly, cannot set projection from EPSG")

            ds.SetProjection(srs.ExportToWkt())
            ds.SetGeoTransform(gt)
            for ii, band in enumerate(range(1, numBands + 1)):
                outband = ds.GetRasterBand(band)
                if bands_first:
                    outband.WriteArray(_output_tile_array[ii, :, :])
                else:
                    outband.WriteArray(_output_tile_array[:, :, ii])
            ds = None
        except Exception as e:
            print(e)
            print(f"Failed to save image at: {outputfilename}")

    def _load_file_local(self, file_path):
        """
        Load an image array from local disk

        :param file_path: The path of the image to load
        :return: The image, or None if something went wrong
        """
        try:
            image, *_ = read_image_from_tif(
                file_path, (self.tile_size, self.tile_size), [1, 2, 3], dtype=np.uint8
            )
            return image
        except Exception as e:
            print(e)
            return None

    def _save_patch_to_file_s3(
        self,
        output_tile_array,
        outputfilename,
        dataParams: DataParams,
        clipBufferBool,
        gt,
        epsg,
        bands_first=True,
        verbose=0,
    ):
        """
        Implements save_patch_to_file for AWS storing
        """
        _output_tile_array = np.copy(output_tile_array)
        if len(output_tile_array.shape) == 4:  # includes batch dimension
            assert (
                output_tile_array.shape[0] == 1
            ), "output_tile_array has a batch dimension larger than 1, please pass a single image"
            _output_tile_array = _output_tile_array[0, ...]

        # stretch patch_date to 1-255
        if verbose:
            print(_output_tile_array.shape)
        if _output_tile_array.dtype != np.uint8:  # Only scale when needed
            _output_tile_array = stretch_to_byt(_output_tile_array)

        # numbands
        if verbose:
            print(_output_tile_array.shape)
        if bands_first:
            numBands = _output_tile_array.shape[0]
        else:
            numBands = _output_tile_array.shape[-1]

        if clipBufferBool is True:
            patchsize = dataParams.patchSize - dataParams.buffer
            gt = [
                gt[0] + (dataParams.buffer * dataParams.resolution) / 2,
                gt[1],
                gt[2],
                gt[3] - (dataParams.buffer * dataParams.resolution) / 2,
                gt[4],
                gt[5],
            ]
            # write to aray
            halfBuf = int(dataParams.buffer / 2)
            _output_tile_array = _output_tile_array[
                :,
                halfBuf : dataParams.patchSize - halfBuf,
                halfBuf : dataParams.patchSize - halfBuf,
            ]
        else:
            # write to aray
            if verbose:
                print("writing")
            patchsize = dataParams.patchSize

        try:
            tmpfile = tempfile.NamedTemporaryFile("wb", delete=False)
            tmpfile.close()

            srs = osr.SpatialReference()
            srs.ImportFromEPSG(epsg)
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(
                tmpfile.name,
                patchsize,
                patchsize,
                numBands,
                gdal.GDT_Byte,
                options=["COMPRESS=DEFLATE"],
            )

            if srs.ExportToWkt() == "":
                raise Exception("osr module not working properly, cannot set projection from EPSG")

            ds.SetProjection(srs.ExportToWkt())
            ds.SetGeoTransform(gt)
            for ii, band in enumerate(range(1, numBands + 1)):
                outband = ds.GetRasterBand(band)
                if bands_first:
                    outband.WriteArray(_output_tile_array[ii, :, :])
                else:
                    outband.WriteArray(_output_tile_array[:, :, ii])
            ds = None

            outputfilename = str(outputfilename).replace("\\", "/")
            with open(tmpfile.name, "rb") as f:
                self.s3.upload_fileobj(f, self.s3_bucket_name, str(outputfilename))
            os.remove(tmpfile.name)

        except Exception as e:
            print(e)
            print(f"Failed to save image at: {outputfilename}")

    def _load_file_s3(self, file_path):
        """
        Load an image from s3

        :param file_path: Path to the image on s3
        :return: The image array, or None if something went wrong
        """
        try:
            if file_path.startswith("\\\\"):
                file_path = os.path.join(*file_path.split(os.sep)[-3:])
                aws_path = os.path.join("/vsis3", self.s3_bucket_name, file_path)
                aws_path = aws_path.replace("\\", "/")
            else:
                aws_path = file_path
                aws_path = aws_path.replace("\\", "/")
                aws_path = aws_path.replace(
                    "//", "/"
                )  # Some paths were stored as s3://P~ and others as s3:/P~
                aws_path = aws_path.replace("s3:", "/vsis3")
            return gdal_array.LoadFile(aws_path)
        except Exception as e:
            print(f"tileRequester.py:715 - {e}")
            return None
