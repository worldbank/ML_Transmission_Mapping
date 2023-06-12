"""
calculating targets
"""

import json
import os
import time
from math import sqrt

import cv2
import fiona
import geopandas as gpd
import numpy as np
import rasterio
from osgeo import gdal, osr
from rasterio import Affine
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.warp import Resampling, reproject
from scipy import signal

from .utils import cell_to_poly, read_image_subset, save_raster


class Targets:
    def create(
        self,
        folder_ntl_in,
        aoi_in,
        pt_targets,
        pop_in,
        ntl_threshold,
        upsample_by,
        percentile,
        min_pop_value,
        targets_complete_out,
        temp_dir,
    ):
        """Creates grid 'targets'

        Args:
            folder_ntl_in (str): Path to directory with input nighttime imagery
            aoi_in (str): Path to aoi shapefile
            pt_targets (str): Path to additional point targets vector file
            pop_in (str): Path to input population map
            ntl_threshold (float): Nighttime imagery threshold (values above are considered electrified)
            upsample_by (int): The factor by which to upsample the nighttime imagery input raster, applied to both axes.
            percentile (int): Percentile used when merging monthly nighttime imagery,
                            to define when a cell can be part of a 'target'
            min_pop_value (int): Minimum population a cell should have to be part of a 'target'
            targets_complete_out (str): Path to output targets map
            temp_dir (str): Path to directory to store temporary outputs
        """

        start = time.time()
        folder_ntl_out = os.path.join(temp_dir, "01_ntl_clipped")
        raster_merged_out = os.path.join(temp_dir, "ntl_merged.tif")
        if not os.path.exists(raster_merged_out):
            print("Clip and merge monthly rasters")
            self.clip_rasters(folder_ntl_in, folder_ntl_out, aoi_in)
            raster_merged, affine = self.merge_rasters(folder_ntl_out, percentile=percentile)
            save_raster(raster_merged_out, raster_merged, affine)

        print(
            "Clip, filter and resample NTL, need to redo this even if it already exists to get affine"
        )
        el_targets_out = os.path.join(temp_dir, "ntl_targets.tif")
        ntl_thresh, affine = self.prepare_ntl(
            raster_merged_out,
            aoi_in,
            threshold=ntl_threshold,
            upsample_by=upsample_by,
        )
        save_raster(el_targets_out, ntl_thresh, affine)

        targets_clean_out = os.path.join(temp_dir, "cleaned_ntl_targets.tif")
        if not os.path.exists(targets_clean_out):
            print("Remove target areas with no underlying population")
            targets_clean = self.drop_zero_pop(
                el_targets_out, pop_in, aoi_in, min_pop_value=min_pop_value
            )
            save_raster(targets_clean_out, targets_clean, affine)

        targets_with_google_out = os.path.join(temp_dir, "targets_with_google.tif")
        if not os.path.exists(targets_with_google_out):
            print("add point targets (from google API)")
            targets_complete = self.add_point_targets(targets_clean_out, pt_targets, aoi_in)
            save_raster(targets_with_google_out, targets_complete, affine)
        else:
            print("point targets form google API already added, reading it in memmory")
            targets_complete_rd = rasterio.open(targets_with_google_out)
            targets_complete = targets_complete_rd.read(1)

        if not os.path.exists(targets_complete_out):
            print("morphological closing of targets")
            iterations = 2
            kernel_size = 3
            kernel = np.ones((kernel_size, kernel_size, 1), np.uint8)
            targets_complete_morph = cv2.morphologyEx(
                targets_complete, cv2.MORPH_CLOSE, kernel, iterations=iterations
            )
            save_raster(targets_complete_out, targets_complete_morph, affine)

        end = time.time()
        print("creating targets file is done, it took {}s".format(str(end - start)))

    def clip_rasters(self, folder_in, folder_out, aoi_in, debug=False):
        """Read continental rasters one at a time, clip to AOI and save

        Args:
            folder_in (str): Path to directory containing rasters.
            folder_out (str): Path to directory to save clipped rasters.
            aoi_in (str):  Path to an AOI file (readable by Fiona) to use for clipping.
        """

        aoi = gpd.read_file(aoi_in)
        coords = [json.loads(aoi.to_json())["features"][0]["geometry"]]

        for file_path in os.listdir(folder_in):
            if file_path.endswith(".tif"):
                if debug:
                    print(f"Doing {file_path}")
                ntl_rd = rasterio.open(os.path.join(folder_in, file_path))
                ntl, affine = mask(dataset=ntl_rd, shapes=coords, crop=True, nodata=0)

                if ntl.ndim == 3:
                    ntl = ntl[0]

                save_raster(os.path.join(folder_out, file_path), ntl, affine)

    def merge_rasters(self, folder, percentile=70):
        """Merge a set of monthly rasters keeping the nth percentile value.

        Used to remove transient features from time-series data.

        Args:
            folder (str): Folder containing rasters to be merged.
            percentile (int): optional (default 70.)
                Percentile value to use when merging using np.nanpercentile.
                Lower values will result in lower values/brightness.

        Returns:
            raster_merged (numpy array):  The merged array.
            affine (affine.Affine):  The affine transformation for the merged raster.
        """

        affine = None
        rasters = []

        for file in os.listdir(folder):
            if file.endswith(".tif"):
                ntl_rd = rasterio.open(os.path.join(folder, file))
                rasters.append(ntl_rd.read(1))

                if not affine:
                    affine = ntl_rd.transform

        raster_arr = np.array(rasters)
        raster_merged = np.percentile(raster_arr, percentile, axis=0)

        return raster_merged, affine

    def prepare_ntl(self, ntl_in, aoi_in, threshold=0.1, upsample_by=2):
        """Convert the supplied NTL raster and output an array of electrified cells
        as targets for the algorithm.

        Args:
            ntl_in (str):  Path to an NTL raster file.
            aoi_in (str): Path to a Fiona-readable AOI file.
            threshold (float):  optional (default 0.1.)
                The threshold to apply after filtering, values above
                are considered electrified.
            upsample_by (int): optional (default 2.)
                The factor by which to upsample the input raster, applied to both axes
                (so a value of 2 results in a raster 4 times bigger). This is to
                allow the roads detail to be captured in higher resolution.

        Returns
            ntl_thresh (numpy array):  Array of cells of value 0 (not electrified) or 1 (electrified).
            newaff (affine.Affine):  Affine raster transformation for the returned array.
        """

        def filter_func(i, j):
            """Function used in creating raster filter."""

            d_rows = abs(i - 20)
            d_cols = abs(j - 20)
            d = sqrt(d_rows**2 + d_cols**2)

            if d == 0:
                return 0.0
            else:
                return 1 / (1 + d / 2) ** 3

        aoi = gpd.read_file(aoi_in)

        # create filter
        vec_filter_func = np.vectorize(filter_func)
        ntl_filter = np.fromfunction(vec_filter_func, (41, 41), dtype=float)
        ntl_filter = ntl_filter / ntl_filter.sum()

        ntl_big = rasterio.open(ntl_in)
        coords = [json.loads(aoi.to_json())["features"][0]["geometry"]]
        ntl, affine = mask(dataset=ntl_big, shapes=coords, crop=True, nodata=0)

        if ntl.ndim == 3:
            ntl = ntl[0]

        ntl_convolved = signal.convolve2d(ntl, ntl_filter, mode="same")
        ntl_filtered = ntl - ntl_convolved

        ntl_interp = np.empty(
            shape=(
                1,  # same number of bands
                round(ntl.shape[0] * upsample_by),
                round(ntl.shape[1] * upsample_by),
            )
        )

        # adjust the new affine transform based on the 'upsample_by' argument
        newaff = Affine(
            affine.a / upsample_by,
            affine.b,
            affine.c,
            affine.d,
            affine.e / upsample_by,
            affine.f,
        )
        with fiona.Env():
            with rasterio.Env():
                reproject(
                    ntl_filtered,
                    ntl_interp,
                    src_transform=affine,
                    dst_transform=newaff,
                    src_crs={"init": "epsg:4326"},
                    dst_crs={"init": "epsg:4326"},
                    resampling=Resampling.bilinear,
                )

        ntl_interp = ntl_interp[0]

        ntl_thresh = np.empty_like(ntl_interp).astype("uint8")
        ntl_thresh[:] = ntl_interp[:]
        ntl_thresh[ntl_thresh < threshold] = 0
        ntl_thresh[ntl_thresh >= threshold] = 1

        return ntl_thresh, newaff

    def drop_zero_pop(self, targets_in, pop_in, aoi, min_pop_value=1):
        """Drop electrified cells with no other evidence of human activity.
        Loop through all target pixels, read in underlying pop pixels

        Args:
            targets_in (str): Path to output from prepare_ntl()
            pop_in (str): Path to a population raster such as GHS or HRSL.
            aoi (str):  An AOI to use to clip the population raster.

        Returns:
            targets (numpy array):  Array with zero population sites dropped.
        """

        start = time.time()

        # get info of pop
        ds_pop = gdal.Open(str(pop_in))
        prj_pop = ds_pop.GetProjection()
        del ds_pop

        # get info of targets
        ds_targets = gdal.Open(str(targets_in))
        gt_targets = ds_targets.GetGeoTransform()
        targets = ds_targets.ReadAsArray()
        xmax = ds_targets.RasterXSize  # ncolumns
        ymax = ds_targets.RasterYSize  # nrow
        del ds_targets

        print("creating geom list for raster with size {} {}".format(xmax, ymax))
        geom_list = []
        for row in range(ymax - 1):
            for col in range(xmax - 1):
                if targets[row][col] == 1:
                    poly = cell_to_poly(gt_targets, col, row)

                    # reproject and add to tuple list
                    source = osr.SpatialReference()
                    source.ImportFromEPSG(4326)
                    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                    target = osr.SpatialReference()
                    target.ImportFromWkt(prj_pop)
                    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                    transform = osr.CoordinateTransformation(source, target)
                    poly.Transform(transform)
                    geom_list.append(("{}_{}".format(row, col), poly))

        print("reading the {} values".format(len(geom_list)))
        result_dic = read_image_subset(
            str(pop_in),
            geom_list,
            [1],
            pixel_mode="centroid",
            padding_value=None,
            return_mask=False,
        )

        print("filtering targets")
        for geom_id in result_dic:
            if np.mean(result_dic[geom_id]["observations"]) <= min_pop_value:

                id_split = geom_id.split("_")
                i = int(id_split[0])
                j = int(id_split[1])

                targets[i][j] = 0

        end = time.time()
        print("all done, took {}s".format(end - start))

        return targets.astype(np.byte)

    def add_point_targets(self, targets, point_targets, aoi_in):
        """Add point vector targets (e.g. from google API)

        Args:
            targets (str): Path to targets
            point_targets (str):  Path to point targets to add
            aoi_in (str): Path to AOI

        Returns:
            targets  (numpy array): Slope as a raster array with the value being the cost of traversing.
        """

        aoi = gpd.read_file(aoi_in)
        pt_targets_masked = gpd.read_file(point_targets, mask=aoi)
        pt_targets = gpd.sjoin(pt_targets_masked, aoi, how="inner", op="intersects")
        pt_targets = pt_targets[pt_targets_masked.columns]
        pts_for_raster = [(row.geometry, 1) for _, row in pt_targets.iterrows()]

        targets_rd = rasterio.open(targets)
        targets = targets_rd.read(1)
        shape = targets.shape
        affine = targets_rd.transform

        point_targets_raster = rasterize(
            pts_for_raster,
            out_shape=shape,
            fill=0,
            default_value=0,
            all_touched=True,
            transform=affine,
        )

        # add the points targets raster to the orignal targets
        combined_targets = point_targets_raster + targets
        combined_targets[combined_targets > 0] = 1

        return combined_targets


if __name__ == "__main__":
    pass
