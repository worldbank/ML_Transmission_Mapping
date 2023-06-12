"""
calculating cost map
"""

import os
import time
import attr

import geopandas as gpd
import mercantile
import numpy as np
import rasterio
from osgeo import gdal, ogr
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

from .pathfinding import PathFinder
from .utils import (
    clip_raster,
    clip_shp,
    convert_wgs_to_utm,
    linear_rescale,
    save_raster,
)


class GridCosts:
    def create_final_cost_map(
        self,
        targets_out,
        prelim_cost_out,
        final_cost_out,
        temp_dir,
        rescaling_params,
        final_cost_weights,
        animate_out=None,
    ):
        """Combines the paths from pathfinding algorithm with initial cost maps to create final cost map
        with values from 0 (certainly a tower?) to 1 (very high cost for tower) and resamples this to EPSG 3857

        Args:
            targets_out (str): Path of targets map
            prelim_cost_out (str): Path of preliminary cost map
            final_cost_out (str): Path to final cost map to create
            temp_dir (str): Path to directory for storing temp files
            rescaling_params (list): Parameters to rescale distance to grid TO cost
            final_cost_weights (list): List of wieghts to use when combining cost layers
            animate_out (bool):  Whether to make animation of path finding
        """

        dist_out, affine = PathFinder().create_dist_map(
            targets_out, prelim_cost_out, temp_dir, animate_out
        )

        print("combine dist (rescaled) with prelim cost layer")
        combined_cost_out = os.path.join(temp_dir, "combined_cost_out.tif")
        prelim_cost_rd = rasterio.open(prelim_cost_out)
        prelim_cost = prelim_cost_rd.read(1)

        dist_rd = rasterio.open(dist_out)
        dist = dist_rd.read(1)
        dist_rescaled = linear_rescale(dist, rescaling_params)
        combined_cost = self.combine_cost_layers([dist_rescaled, prelim_cost], final_cost_weights)
        save_raster(combined_cost_out, combined_cost, affine)

        if not os.path.exists(final_cost_out):
            print("allign to webmercator tiles")
            self.allign_to_webmercator_tiles(combined_cost_out, final_cost_out)

    def create_prelim_cost_map(
        self,
        weights,
        roads_in,
        aoi_in,
        land_use_in,
        slope_in,
        targets_out,
        temp_dir,
        final_cost_out,
        osm_water,
        osm_transmission_lines_in=None,
    ):
        """Creates preliminary cost map

        Args:
            weights (list): DESCRIPTION.
            roads_in (str): Path to roads
            aoi_in (str): Path to aoi
            land_use_in (str): Path to land use map
            slope_in (str): Path to slope map
            targets_out (str): Path of targetsof targets map
            temp_dir (str): Path to directory for temporary outputs
            final_cost_out (str): Path to preliminary cost map to create
            osm_water (str): Path to water shape
            osm_transmission_lines_in (str, optional): Path to OSM transmission lines shape. Defaults to None.
        """

        start = time.time()

        # get affine from  targets_out
        ntl_rd = rasterio.open(targets_out)
        affine = ntl_rd.transform
        ntl_rd = None

        roads_costs_out = os.path.join(temp_dir, "road_cost.tif")
        if not os.path.exists(roads_costs_out):
            print("Roads: assign values, clip and rasterize")
            self.prepare_roads(roads_in, aoi_in, targets_out, temp_dir, roads_costs_out)

        land_use_costs_out = os.path.join(temp_dir, "land_use_cost.tif")
        if not os.path.exists(land_use_costs_out):
            print("land use cost")
            land_use_costs = self.prepare_land_use(land_use_in, aoi_in, targets_out, osm_water)
            save_raster(land_use_costs_out, land_use_costs, affine, nodata=-1)
        else:
            land_use_rd = rasterio.open(land_use_costs_out)
            land_use_costs = land_use_rd.read(1)

        slope_costs_out = os.path.join(temp_dir, "slope_cost.tif")
        if not os.path.exists(slope_costs_out):
            print("slope cost")
            slope_costs = self.prepare_slope(slope_in, aoi_in, targets_out)
            save_raster(slope_costs_out, slope_costs, affine, nodata=-1)
        else:
            slope_rd = rasterio.open(slope_costs_out)
            slope_costs = slope_rd.read(1)

        cost_raster_out = os.path.join(temp_dir, "combined_costs.tif")
        if not os.path.exists(cost_raster_out):
            print("combine costs using weights")
            roads_rd = rasterio.open(roads_costs_out)
            roads_costs = roads_rd.read(1)
            cost_raster = self.combine_cost_layers(
                [land_use_costs, roads_costs, slope_costs], weights
            )
            save_raster(cost_raster_out, cost_raster, affine, nodata=-1)

        if osm_transmission_lines_in:
            print("existing transmission lines to be set to 0")
            final_cost = self.set_grid_zero(osm_transmission_lines_in, aoi_in, cost_raster_out)
            save_raster(final_cost_out, final_cost, affine, nodata=-1)
        else:
            print("no existing transmission lines found")
            cost_raster_rd = rasterio.open(cost_raster_out)
            cost_raster = cost_raster_rd.read(1)
            save_raster(final_cost_out, cost_raster, affine, nodata=-1)

        end = time.time()
        print("done, it took {}s".format(str(end - start)))

    def combine_cost_layers(self, cost_rasters, weights):
        """Combine cost layers.

        Args:
            cost_rasters (list): Paths to cost rasters
            weights (list): weights to give each cost layer

        Returns:
            cost_raster (numpy array): Combined cost raster
        """

        assert len(cost_rasters) == len(weights), "length has to be the same"

        final_cost = np.zeros_like(cost_rasters[0])
        for cost_raster, weight in zip(cost_rasters, weights):
            final_cost = final_cost + (cost_raster * weight)

        # standardise
        print("divide by sum of weights {}".format(sum(weights)))
        final_cost = final_cost / sum(weights)

        return final_cost

    def set_grid_zero(self, roads_in, aoi_in, cost_raster_path):
        """Set pixels covering grid to 0 within current raster cost map

        Args:
            roads_in (str):  Path to roads input
            aoi_in (str) : Path to aoi shape
            cost_raster_path (str) :  Path of current cost raster
        """
        # clip roads lyr to aoi
        _, file_extension = os.path.splitext(str(roads_in))
        roads_in_clipped = str(cost_raster_path).replace(".tif", file_extension)
        if not os.path.exists(roads_in_clipped):
            clip_shp(str(roads_in), str(aoi_in), roads_in_clipped)

        # find the correct utm zone
        ntl_ds = gdal.Open(str(cost_raster_path), 1)
        ntl_gt = ntl_ds.GetGeoTransform()
        ulx, xres, xskew, uly, yskew, yres = ntl_gt

        # get roads layer and its info
        roads_in_clipped_ds = ogr.Open(roads_in_clipped)
        roads_lyr = roads_in_clipped_ds.GetLayer(0)
        spatialRef = roads_lyr.GetSpatialRef()

        road_weights = [("power", "line"), ("power", "minor_line")]

        for attribute_name, class_name in road_weights:

            # create mem vector
            driver_label = ogr.GetDriverByName("Memory")
            mem_ds_label = driver_label.CreateDataSource("Fake")

            # create layer
            print("create label layer")
            mem_layer_label = mem_ds_label.CreateLayer(
                class_name, spatialRef, ogr.wkbMultiLineString
            )
            defn = mem_layer_label.GetLayerDefn()
            roads_lyr.ResetReading()
            for feat in roads_lyr:
                if feat.GetField(attribute_name) == class_name:
                    label_feat = ogr.Feature(defn)
                    label_feat.SetGeometry(feat.GetGeometryRef())
                    mem_layer_label.CreateFeature(label_feat)

            num_features = mem_layer_label.GetFeatureCount()
            print("rasterize {} features".format(num_features))
            gdal.RasterizeLayer(ntl_ds, [1], mem_layer_label, burn_values=[0])

        cost_raster = ntl_ds.ReadAsArray()
        ntl_ds = None

        return cost_raster

    def prepare_land_use(self, land_use_in, aoi_in, roads_in, osm_water_in):
        """Prepare a land use costlayer for use in algorithm.

        Args:
            land_use_in (str): Path to a land use raster layer
            aoi_in (str): AOI to clip roads.
            osm_water (str): shp with water to add to land use map
            roads_in (str):   Path to a raster file, used for correct shape and
                affine of roads raster as well as as set weights of
                roads to 0.

        Returns:
            land_use_raster (numpy array): Land use as a raster array with the value being the cost of traversing.
            affine (affine.Affine):  Affine raster transformation for the new raster (same as ntl_in).
        """

        aoi = gpd.read_file(aoi_in)

        # Clip land use layer to AOI
        clipped, affine, crs = clip_raster(land_use_in, aoi)

        # We need to warp the population layer to exactly overlap with targets
        # First get array, affine and crs from targets (which is what we)
        roads_rd = rasterio.open(roads_in)
        roads = roads_rd.read(1)
        shape = roads.shape
        land_use_proj = np.empty_like(roads).astype("float32")
        dest_affine = roads_rd.transform
        dest_crs = roads_rd.crs

        # Then use reproject
        with rasterio.Env():
            reproject(
                source=clipped,
                destination=land_use_proj,
                src_transform=affine,
                dst_transform=dest_affine,
                src_crs=crs,
                dst_crs=dest_crs,
                resampling=Resampling.nearest,
            )

        # burn water from osm
        osm_water = gpd.read_file(osm_water_in)
        osm_water_features = [(row.geometry, 1) for _, row in osm_water.iterrows()]

        osm_water_raster = rasterize(
            osm_water_features,
            out_shape=shape,
            fill=0,
            default_value=0,
            all_touched=True,
            transform=dest_affine,
        )

        # classify
        land_use_costs = np.empty_like(land_use_proj)
        land_use_costs[:] = land_use_proj[:]

        land_use_costs[
            (land_use_costs == 0)
            | (land_use_costs == 60)
            | (land_use_costs == 40)
            | (land_use_costs == 50)
        ] = (1 / 6)
        land_use_costs[(land_use_costs == 20) | (land_use_costs == 30)] = 2 / 6
        land_use_costs[
            (land_use_costs == 121)
            | (land_use_costs == 122)
            | (land_use_costs == 123)
            | (land_use_costs == 124)
            | (land_use_costs == 125)
            | (land_use_costs == 126)
        ] = (3 / 6)
        land_use_costs[
            (land_use_costs == 111)
            | (land_use_costs == 112)
            | (land_use_costs == 113)
            | (land_use_costs == 114)
            | (land_use_costs == 115)
            | (land_use_costs == 116)
        ] = (4 / 6)
        land_use_costs[(land_use_costs == 90) | (land_use_costs == 100)] = 5 / 6
        land_use_costs[
            (land_use_costs == 70) | (land_use_costs == 80) | (land_use_costs == 200)
        ] = 1
        land_use_costs[osm_water_raster == 1] = 1  # water from osm

        return land_use_costs

    def prepare_slope(self, slope_in, aoi_in, roads_in):
        """Prepare a slope cost layer for use in algorithm.

        Args:
            slope_in (str): Path to a slope raster layer.
            aoi_in (str): AOI to clip roads.
            roads_in (str) :  Path to a raster file, used for correct shape and
                affine of roads raster as well as to set weights of
                roads to 0.

        Returns:
            slope_raster (numpy array): Slope as a raster array with the value being the cost of traversing.
            affine (affine.Affine): Affine raster transformation for the new raster (same as ntl_in).
        """

        # Clip population layer to AOI
        aoi = gpd.read_file(aoi_in)
        clipped, affine, crs = clip_raster(slope_in, aoi)

        # We need to warp the population layer to exactly overlap with targets
        # First get array, affine and crs from targets (which is what we)
        roads_rd = rasterio.open(roads_in)
        roads = roads_rd.read(1)
        slope_proj = np.empty_like(roads).astype("float32")
        dest_affine = roads_rd.transform
        dest_crs = roads_rd.crs

        # Then use reproject
        with rasterio.Env():
            reproject(
                source=clipped,
                destination=slope_proj,
                src_transform=affine,
                dst_transform=dest_affine,
                src_crs=crs,
                dst_crs=dest_crs,
                resampling=Resampling.average,
            )

        slope_cost = np.empty_like(slope_proj)
        slope_cost[:] = slope_proj[:]
        slope_cost[slope_cost < 20] = 1 / 5
        slope_cost[(slope_cost >= 20) & (slope_cost < 30)] = 2 / 5
        slope_cost[(slope_cost >= 30) & (slope_cost < 40)] = 3 / 5
        slope_cost[slope_cost >= 40] = 1

        return slope_cost

    def prepare_roads(self, roads_in, aoi_in, ntl_in, temp_dir, roads_costs_out):
        """Prepare a roads feature layer for use in algorithm.

        Args:
            roads_in (str): Path to a roads feature layer. This implementation is specific to
                OSM data and won't assign proper weights to other data inputs.
            aoi_in (str): AOI to clip roads.
            ntl_in (str): Path to a raster file, only used for correct shape and
                affine of roads raster.

        Returns:
            roads_raster (numpy array): Roads as a raster array with the value being the cost of traversing.
            affine (affine.Affine): Affine raster transformation for the new raster (same as ntl_in).
        """

        # clip roads lyr to aoi
        ext = os.path.splitext(roads_in)[1]
        roads_in_clipped = os.path.join(
            temp_dir, os.path.basename(roads_in).replace(ext, "_clipped" + ext)
        )
        if not os.path.exists(roads_in_clipped):
            print("clipping roads in to {}".format(roads_in_clipped))
            clip_shp(str(roads_in), str(aoi_in), roads_in_clipped)

        # find the correct utm zone and reproject
        ntl_ds = gdal.Open(str(ntl_in))
        ntl_out = os.path.join(temp_dir, "reprojected.tif")
        ntl_gt = ntl_ds.GetGeoTransform()
        ulx, xres, xskew, uly, yskew, yres = ntl_gt
        lrx = ulx + (ntl_ds.RasterXSize * xres)
        lry = uly + (ntl_ds.RasterYSize * yres)
        utm_epsg = convert_wgs_to_utm(ntl_gt[0], ntl_gt[3])
        if not os.path.exists(ntl_out):
            print("reprojecting targets to {}".format(ntl_out))
            warpopts = gdal.WarpOptions(
                dstSRS=utm_epsg,
                creationOptions=["COMPRESS=DEFLATE", "SPARSE_OK=TRUE", "PREDICTOR=2", "TILED=YES"],
            )
            gdal.Warp(ntl_out, ntl_ds, options=warpopts)
        ntl_ds = gdal.Open(str(ntl_out))
        gt = ntl_ds.GetGeoTransform()

        # roads are in epsg4326, transform to UTM
        roads_utm = roads_in_clipped.replace(ext, "_UTM" + ext)
        if not os.path.exists(roads_utm):
            print("roads are in epsg4326, transform to UTM")
            ds_roads_utm = gdal.VectorTranslate(roads_utm, str(roads_in_clipped), dstSRS=utm_epsg)
        else:
            ds_roads_utm = ogr.Open(roads_utm)

        # get roads layer and its info
        roads_lyr = ds_roads_utm.GetLayer(0)
        spatialRef = roads_lyr.GetSpatialRef()

        road_weights = [
            ("highway", "motorway", 0.100),
            ("highway", "trunk", 0.111),
            ("highway", "primary", 0.125),
            ("highway", "secondary", 0.142),
            ("highway", "tertiary", 0.167),
            ("highway", "unclassified", 0.200),
            ("highway", "residential", 0.25),
            ("highway", "service", 0.333),
            ("highway", "rail", 1 / 8),
        ]

        # create output raster for distances per road type in same projection and
        driver = gdal.GetDriverByName("GTiff")

        def _scale(to_write, min_src, max_src, min_dst, max_dst):
            # scales from min and max values in dictionary to values between 1 and 255 (0 is kept for nodata)

            to_write_scaled = min_dst + ((to_write - min_src) * (max_dst - min_dst)) / (
                max_src - min_src
            )

            to_write_scaled = min(max_dst, to_write_scaled)
            to_write_scaled = max(min_dst, to_write_scaled)

            return to_write_scaled

        # for each road type, rasterize
        idx = 0
        cost_array_list = []
        costs_utm = os.path.join(temp_dir, "costs_utm.tif")
        for attribute_name, class_name, weight in road_weights:
            idx += 1

            roads_out = os.path.join(temp_dir, "road_raster_{}.tif".format(str(weight)))

            # create mem vector
            driver_label = ogr.GetDriverByName("Memory")
            mem_ds_label = driver_label.CreateDataSource("Fake")

            # create layer
            mem_layer_label = mem_ds_label.CreateLayer(
                class_name, spatialRef, ogr.wkbMultiLineString
            )
            defn = mem_layer_label.GetLayerDefn()
            roads_lyr.ResetReading()
            print(attribute_name)
            for feat in roads_lyr:
                if feat.GetField(attribute_name) == class_name:
                    label_feat = ogr.Feature(defn)
                    label_feat.SetGeometry(feat.GetGeometryRef())
                    mem_layer_label.CreateFeature(label_feat)

            num_features = mem_layer_label.GetFeatureCount()

            if not os.path.exists(roads_out) and num_features > 0:

                # rasterize that road type
                roads_ds = driver.Create(
                    str(roads_out),
                    ntl_ds.RasterXSize,
                    ntl_ds.RasterYSize,
                    1,
                    gdal.GDT_Float32,
                    options=["COMPRESS=DEFLATE", "SPARSE_OK=TRUE", "TILED=YES"],
                )
                roads_ds.SetGeoTransform(gt)
                roads_ds.SetProjection(spatialRef.ExportToWkt())

                print("rasterize {} features".format(num_features))
                gdal.RasterizeLayer(roads_ds, [1], mem_layer_label, burn_values=[1])
                roads_ds = None

            # distance file
            roads_distance_out = os.path.join(temp_dir, "road_distances_{}.tif".format(str(weight)))

            if not os.path.exists(roads_distance_out) and num_features > 0:
                distance_ds = driver.Create(
                    str(roads_distance_out),
                    ntl_ds.RasterXSize,
                    ntl_ds.RasterYSize,
                    1,
                    gdal.GDT_Float32,
                    options=["COMPRESS=DEFLATE", "SPARSE_OK=TRUE", "TILED=YES"],
                )
                distance_ds.SetGeoTransform(gt)
                distance_ds.SetProjection(spatialRef.ExportToWkt())

                # calc dist
                roads_ds = gdal.Open(str(roads_out))
                roadsband = roads_ds.GetRasterBand(1)
                dstband = distance_ds.GetRasterBand(1)
                alg_options = ["VALUES=1", "MAXDIST=2000", "DISTUNITS=GEO"]
                gdal.ComputeProximity(roadsband, dstband, alg_options)
                dstband.FlushCache()
                distance_ds = None
                roads_ds = None

            if os.path.exists(roads_distance_out) and not os.path.exists(costs_utm):
                print("calculate cost")
                distance_ds = gdal.Open(str(roads_distance_out))
                dstband = distance_ds.GetRasterBand(1)
                dist_array = dstband.ReadAsArray()
                cost_array = linear_rescale(dist_array, [200, 2000, weight, 1])

                # write it to file
                cost_out = os.path.join(temp_dir, "road_cost_{}.tif".format(str(weight)))
                if not os.path.exists(cost_out):
                    cost_ds = driver.Create(
                        str(cost_out),
                        ntl_ds.RasterXSize,
                        ntl_ds.RasterYSize,
                        1,
                        gdal.GDT_Float32,
                        options=["COMPRESS=DEFLATE", "SPARSE_OK=TRUE", "TILED=YES"],
                    )
                    cost_ds.SetGeoTransform(gt)
                    costband = cost_ds.GetRasterBand(1)
                    costband.WriteArray(cost_array)
                    cost_ds = None

                # print(cost_array.dtype)
                cost_array_list.append(cost_array)
                distance_ds = None

        if not os.path.exists(costs_utm):
            # stack and find max
            stacked_costs = np.dstack(cost_array_list)
            max_costs = np.amin(stacked_costs, 2)

            # write it to file
            cost_ds = driver.Create(
                str(costs_utm),
                ntl_ds.RasterXSize,
                ntl_ds.RasterYSize,
                1,
                gdal.GDT_Float32,
                options=["COMPRESS=DEFLATE", "SPARSE_OK=TRUE", "TILED=YES"],
            )
            cost_ds.SetGeoTransform(gt)
            costband = cost_ds.GetRasterBand(1)
            costband.WriteArray(max_costs)
            cost_ds = None

        print("warp to {}".format(roads_costs_out))
        gdal.UseExceptions()
        ds = gdal.Open(costs_utm)

        warpopts = gdal.WarpOptions(
            outputBounds=[ulx, lry, lrx, uly],
            xRes=xres,
            yRes=yres,
            srcSRS=utm_epsg,
            dstSRS="EPSG:4326",
            creationOptions=["COMPRESS=DEFLATE", "SPARSE_OK=TRUE", "PREDICTOR=2", "TILED=YES"],
        )
        ds_out = gdal.Warp(str(roads_costs_out), ds, options=warpopts)
        del ds_out

        return

    def allign_to_webmercator_tiles(self, in_file, out_file, resampleAlg=gdal.GRA_Bilinear):
        """Warps input raster to be alligned with web mercator tiles with zoomlevel 18

        Args:
            in_file (str): Input raster file to allign
            out_file (str): Output alligned raster filte to create
            resampleAlg: Resampling algorithm to use
        """

        # get the bounds
        ds = gdal.Open(str(in_file))
        gt = ds.GetGeoTransform()
        ulx, xres, xskew, uly, yskew, yres = gt
        lrx = ulx + (ds.RasterXSize * xres)
        lry = uly + (ds.RasterYSize * yres)

        # get mercantile tiles for each of the 4 corners
        lr_bounds = mercantile.xy_bounds(mercantile.tile(lrx, lry, 18))
        ul_bounds = mercantile.xy_bounds(mercantile.tile(ulx, uly, 18))
        x_res = lr_bounds.right - lr_bounds.left
        y_res = lr_bounds.top - lr_bounds.bottom

        # warp it to allign to mercantile
        ds_out = gdal.Warp(
            str(out_file),
            ds,
            resampleAlg=resampleAlg,
            outputBounds=[ul_bounds.left, lr_bounds.bottom, lr_bounds.right, ul_bounds.top],
            xRes=x_res,
            yRes=y_res,
            srcSRS="EPSG:4326",
            dstSRS="EPSG:3857",
            options=["COMPRESS=DEFLATE", "SPARSE_OK=TRUE", "PREDICTOR=2", "TILED=YES"],
        )
        del ds_out


if __name__ == "__main__":
    pass
