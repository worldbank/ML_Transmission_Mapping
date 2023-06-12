"""
Entry point for calculating grid cost map
"""

import argparse
import os
import sys
from configparser import ConfigParser

from grid_costs.grid_cost_tools.costs import GridCosts
from grid_costs.grid_cost_tools.targets import Targets


def calc_grid_costs_cmd():
    """Entrypoint to call calc_grid_costs through cmd"""

    # parse aguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--aoi_name", type=str)

    # read arguments
    args = parser.parse_args()
    config_path = args.config
    aoi_name = args.aoi_name

    calc_grid_costs(config_path, aoi_name)


def calc_grid_costs(config_path, aoi_name):
    """Main function that creates the grid costs map

    Args:
        config_path (str): Path to config fie with all required settings
        aoi_name (str): For which AOI to run the calc_grid_costs function,
                        this allows the config file to have settings for multiple AOIs in it.
    """

    # read confs
    config = ConfigParser()
    config.read(config_path, encoding="utf-8")
    folder_ntl_in = config[aoi_name]["folder_ntl_in"]
    land_use_in = config[aoi_name]["land_use_in"]
    osm_transmission_lines_in = config[aoi_name]["osm_transmission_lines_in"]
    if osm_transmission_lines_in == "":
        osm_transmission_lines_in = None
    else:
        osm_transmission_lines_in = osm_transmission_lines_in
    aoi_in = config[aoi_name]["aoi_in"]
    roads_in = config[aoi_name]["roads_in"]
    osm_water = config[aoi_name]["osm_water"]
    pop_in = config[aoi_name]["pop_in"]
    folder_out = config[aoi_name]["folder_out"]
    pt_targets = config[aoi_name]["pt_targets"]
    slope_in = config[aoi_name]["slope_in"]

    # generic params
    percentile = float(
        config[aoi_name]["percentile"]
    )  # percentile value to use when merging monthly NTL rasters
    ntl_threshold = float(
        config[aoi_name]["ntl_threshold"]
    )  # threshold when converting filtered NTL to binary (probably shouldn't change)
    upsample_by = float(
        config[aoi_name]["upsample_by"]
    )  # factor by which to upsample before processing roads (both dimensions are scaled by this)
    min_pop_value = float(
        config[aoi_name]["min_pop_value"]
    )  # minimum average population value for electrified area to be able to be really electrified

    # fixed params
    prelim_cost_weights = [1, 1, 1]
    dist_rescaling_params = [0, 10, 0, 1]
    final_cost_weights = [1, 1]

    # ---------------- Start processing ----------------------

    # intermediate and output file locations
    targets_tempdir = os.path.join(folder_out, "01_targets")
    prelim_cost_tempdir = os.path.join(folder_out, "02_prelim_costs")
    final_cost_tempdir = os.path.join(folder_out, "03_final_costs")
    targets_out = os.path.join(folder_out, "01_targets.tif")
    prelim_cost_out = os.path.join(folder_out, "02_prelim_costs.tif")
    final_cost_out = os.path.join(folder_out, "03_final_costs.tif")

    for dir_path in [folder_out, targets_tempdir, prelim_cost_tempdir, final_cost_tempdir]:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    # start process to create targets
    if not os.path.exists(targets_out):
        Targets().create(
            folder_ntl_in,
            aoi_in,
            pt_targets,
            pop_in,
            ntl_threshold,
            upsample_by,
            percentile,
            min_pop_value,
            targets_out,
            targets_tempdir,
        )
    else:
        print("{} already exists".format(targets_out))

    # create base cost map
    gc = GridCosts()
    if not os.path.exists(prelim_cost_out):
        gc.create_prelim_cost_map(
            prelim_cost_weights,
            roads_in,
            aoi_in,
            land_use_in,
            slope_in,
            targets_out,
            prelim_cost_tempdir,
            prelim_cost_out,
            osm_water,
            osm_transmission_lines_in=osm_transmission_lines_in,
        )
    else:
        print("{} already exists".format(prelim_cost_out))

    # create final cost map (combining base costs and predicted paths)
    if not os.path.exists(final_cost_out):
        gc.create_final_cost_map(
            targets_out,
            prelim_cost_out,
            final_cost_out,
            final_cost_tempdir,
            dist_rescaling_params,
            final_cost_weights,
        )


if __name__ == "__main__":

    # can be used to override arguments in case we want to run in interpreter
    sys.argv = [
        "grid_costs.py",
        "--config",
        r"path\electricity-network-mapping\grid_costs\params.ini",
        "--aoi_name",
        "DomRep"
    ]

    calc_grid_costs_cmd()
