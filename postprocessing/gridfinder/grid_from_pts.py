import argparse
import json
import os
import sys
from configparser import ConfigParser

import geopandas as gpd
import gridfinder as gf
import rasterio
from geopandas import GeoDataFrame
from rasterio.enums import Resampling
from rasterio.features import rasterize

from grid_costs.grid_cost_tools.pathfinding import PathFinder
from grid_costs.grid_cost_tools.utils import save_raster
from postprocessing.network import TowerNetwork


def calc_network_from_cmd():

    # parse aguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--aoi_name", type=str)
    parser.add_argument("--batch_id", type=str)
    parser.add_argument("--run_id", type=str)

    # read arguments
    args = parser.parse_args()
    config_path = args.config
    aoi_name = args.aoi_name
    batch_id = args.batch_id
    run_id = args.run_id

    calc_network(config_path, aoi_name, batch_id, run_id)


def calc_network(config_path: str, aoi_name: str, batch_id: str, run_id: int):
    """Make preliminary gridfinder network from points (for intermediate sampling)

    Args:
        config_path (str): path to config file
        aoi_name (str): aoi name (for config)
        batch_id (str): processing batch id
        run_id (int): smart tracing run id
    """
    # read confs
    config = ConfigParser()
    config.read(config_path, encoding="utf-8")

    # some params
    cutoff = float(config[aoi_name]["cutoff"])

    # required configs
    db_config = os.path.join(os.getcwd(), "..\\..\\database\\amazon.ini")
    run_config = config[aoi_name]["run_config"]
    folder_out = config[aoi_name]["folder_out"]
    aoi_in = config[aoi_name]["aoi_in"]
    set_name = config[aoi_name]["starting_points"]
    costmap = config[aoi_name]["costmap"]

    # output locations
    parent_folder = os.path.join(folder_out, aoi_name, batch_id)
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
    targets_json = os.path.join(parent_folder, f"targets_used_{batch_id}.json")
    targets_as_raster_loc = os.path.join(parent_folder, f"targets_as_raster_{batch_id}.tif")
    skeletonised_raster_loc = os.path.join(parent_folder, f"skeletonised_raster_{batch_id}.tif")
    line_network_loc = os.path.join(parent_folder, f"line_network_{batch_id}.shp")
    upsampled_cm_loc = os.path.join(parent_folder, f"upsampled_cm_{batch_id}.tif")

    # make tower network
    tn = TowerNetwork(run_config, db_config, run_id)
    points_for_network = merge_towers_starting_points(tn, set_name, batch_id)

    # save the points
    with open(targets_json, "w") as outfile:
        outfile.write(points_for_network)

    aoi = gpd.read_file(aoi_in)
    upsample_raster(costmap, 2, upsampled_cm_loc)

    raster_from_points(targets_json, aoi, upsampled_cm_loc, targets_as_raster_loc)

    print(f"Starting optimisation for {batch_id}")

    dist, affine = PathFinder().create_dist_map(
        targets_out=targets_as_raster_loc,
        prelim_cost_out=upsampled_cm_loc,
        temp_dir=parent_folder,
        animate_out=False,
    )
    dists_with_cutoff, _ = gf.threshold(dist, cutoff=cutoff)

    skeletonised = gf.thin(dists_with_cutoff)
    save_raster(skeletonised_raster_loc, skeletonised, affine)

    lines_gdf = gf.raster_to_lines(skeletonised_raster_loc)
    lines_gdf.to_file(line_network_loc, driver="GPKG")

    print(f"Completed network for batch id {batch_id}")


def merge_towers_starting_points(tn: TowerNetwork, set_name: str, batch_id: str):
    """Merge towers and starting points

    Args:
        tn (TowerNetwork): Network object
        set_name (str): name of starting point set
        batch_id (str): post proc batch id

    Returns:
        str: serialized json of merged result
    """
    tn.fetch_data()
    tn.fetch_accepted_towers(batch_id)

    tn.fetch_starting_points(set_name=set_name)
    tn.accepted_geojson["features"].extend(tn.startpoints_geojson["features"])

    # Serializing json
    json_object = json.dumps(tn.accepted_geojson)

    return json_object


def upsample_raster(
    raster_path: str, upsample_factor: int, out_loc: str, interp_method=Resampling.bilinear
):
    """Upsample a raster given an upsampling factor

    Args:
        raster_path (str): path to raster
        upsample_factor (int): factor to upsample by
        out_loc (str): output location
        interp_method (optional): rasterio interpolation method. Defaults to Resampling.bilinear.
    """

    with rasterio.open(raster_path) as dataset:

        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upsample_factor),
                int(dataset.width * upsample_factor),
            ),
            resampling=interp_method,
        )

        print("Shape before resample:", dataset.shape)
        print("Shape after resample:", data.shape[1:])

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]), (dataset.height / data.shape[-2])
        )
        print("Transform before resample:\n", dataset.transform, "\n")
        print("Transform after resample:\n", transform)

        save_raster(out_loc, data[0], transform, dataset.crs)


def raster_from_points(
    targets_json: str, aoi: GeoDataFrame, template_raster: str, targets_as_raster_loc: str
):
    """Burn in raster with point locations

    Args:
        targets_json (str): path to json containing point targets
        aoi (GeoDataFrame): gpd containing aoi polygon
        template_raster (str): template raster of desired shape
        targets_as_raster_loc (str): output location
    """
    pt_targets_masked = gpd.read_file(targets_json, mask=aoi)
    pts_for_raster = [(row.geometry, 1) for _, row in pt_targets_masked.iterrows()]

    dest_crs = pt_targets_masked.crs

    # read a raster file with desired dimensions
    targets_rd = rasterio.open(template_raster)
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

    save_raster(
        path=targets_as_raster_loc,
        raster=point_targets_raster,
        affine=affine,
        crs=dest_crs,
    )


if __name__ == "__main__":

    # can be used to override arguments in case we want to run in interpreter
    sys.argv = [
        "grid_frm_pts.py",
        "--config",
        r"path\electricity-network-mapping\postprocessing\gridfinder\gf_params.ini",
        "--aoi_name",
        "Bangladesh_south",
        "--batch_id",
        "245_south_b4_post",
        "--run_id",
        "245",
    ]

    calc_network_from_cmd()
