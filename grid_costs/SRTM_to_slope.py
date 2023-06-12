"""
downloads SRTM tiles, corresponding to AOI, merges them together and calculate slope
"""

import argparse
import os
import sys
import zipfile

import requests
from grid_cost_tools.utils import (
    convert_wgs_to_utm,
    create_poly,
    get_geom_wkt_list,
    vrt_from_folder,
)
from osgeo import gdal, ogr


def download_srtm_tiles(tilenames, output_dir: str):
    """Function that downloads srt tiles given

    Args:
        tilenames (list): List of tilenames to download
        output_dir (str): Directory into which to download
    """

    for tilename in tilenames:
        url = f"http://step.esa.int/auxdata/dem/SRTMGL1/{tilename}.SRTMGL1.hgt.zip"
        filename = os.path.join(output_dir, f"{tilename}.zip")
        if not os.path.exists(filename):
            r = requests.get(url)
            print("downloading {}".format(url))
            if r.status_code == 200:
                with open(filename, "wb") as out:
                    for bits in r.iter_content():
                        out.write(bits)
            else:
                print(r.content)


def do_unzip(input_folder, output_folder):
    """Function that unzips files

    Args:
        input_folder (str): Input directory with zips to unzip
        output_folder (str): Output directory for unzipped files
    """

    for afile in os.listdir(input_folder):
        if afile.endswith(".zip"):
            in_path = os.path.join(input_folder, afile)
            out_path = os.path.join(output_folder, afile.replace(".zip", ""))
            if not os.path.exists(out_path):
                print("Unzipping ", afile)
                with zipfile.ZipFile(in_path, "r") as z:
                    z.extractall(out_path)


def convert_to_utm_tif(in_dir, out_dir, extension):
    """Converts to tif and reproject to correct UTM zone

    Args:
        in_dir (str): directory with input files
        out_dir (str): directory for output
        extension (str): extension of files to look for in in_dir
    """

    # convert geotiff
    for afile in os.listdir(in_dir):
        if afile.endswith(extension):
            input_path = os.path.join(in_dir, afile)

            # open and get UTM to reproject to
            ds = gdal.Open(input_path)
            gt = ds.GetGeoTransform()
            utm_epsg = convert_wgs_to_utm(gt[0], gt[3])

            # warp
            output_path = os.path.join(out_dir, afile.replace(extension, ".tif"))
            if not os.path.isfile(output_path):
                print("convert to tif, reproject to {}".format(utm_epsg))

                warpopts = gdal.WarpOptions(
                    dstSRS=utm_epsg,
                    creationOptions=[
                        "COMPRESS=DEFLATE",
                        "SPARSE_OK=TRUE",
                        "PREDICTOR=2",
                        "TILED=YES",
                    ],
                )
                gdal.Warp(output_path, ds, options=warpopts)
                ds = None

            else:
                print("file already exists.\n")


def get_SRTM_tile_names(aoi_path):
    """Checks online which SRTM tilenames intersect with the AOI

    Args:
        aoi_path (TYPE): Path to aoi shapefile

    Returns:
        tile_names (ilst): List of tilenames intersecting with AOI
    """

    # open shapefile and get first geom as wkt
    aoi = ogr.CreateGeometryFromWkt(get_geom_wkt_list(aoi_path)[0])

    # loop through all possible grids and check if they intersect with the aoi
    # these are lower left corners
    N_range = 59
    S_range = 56
    E_range = 179
    W_range = 180

    xDist = 1
    yDist = -1

    tile_names = []

    # eastern tiles
    for E in range(E_range + 1):

        for S in range(S_range + 1):
            poly = create_poly(E, -S, xDist, yDist)
            if poly.Intersects(aoi):
                tile_name = f"S{S:02d}E{E:03d}"
                tile_names.append(tile_name)

        for N in range(N_range + 1):
            poly = create_poly(E, N, xDist, yDist)
            if poly.Intersects(aoi):
                tile_name = f"N{N:02d}E{E:03d}"
                tile_names.append(tile_name)

    # western tiles
    for W in range(W_range + 1):

        for S in range(S_range + 1):
            poly = create_poly(-W, -S, xDist, yDist)
            if poly.Intersects(aoi):
                tile_name = f"S{S:02d}W{W:03d}"
                tile_names.append(tile_name)

        for N in range(N_range + 1):
            poly = create_poly(-W, N, xDist, yDist)
            if poly.Intersects(aoi):
                tile_name = f"N{N:02d}W{W:03d}"
                tile_names.append(tile_name)

    print("{} files to download".format(len(tile_names)))
    return tile_names


def create_slope(in_dir, out_dir):
    """Creates slope map from DEM

    Args:
        in_dir (str): Dir with input DEMs
        out_dir (TYPE): Dir for output slope maps
    """

    for file_name in os.listdir(in_dir):
        print("creating slope map")
        in_file = os.path.join(in_dir, file_name)
        out_file = os.path.join(out_dir, file_name)
        ds = gdal.Open(in_file)
        gdal.DEMProcessing(out_file, ds, "slope")
        del ds


def SRTM_to_slope(output_dir, aoi_name, aoi_path):
    """Main function that downloads SRTM and converts to slop

    Args:
        output_dir (str): Directory to write outputs to
        aoi_name (str): Name of aoi to give to output files
        aoi_path (str): Path to aoi shape for which to create slope map
    """

    assert os.path.exists(output_dir), "output_dir {} does not exist".format(output_dir)

    # create subdirs
    output_dir = os.path.join(output_dir, aoi_name)
    download_dir = os.path.join(output_dir, "01_downloaded")
    unzip_dir = os.path.join(output_dir, "02_unzip")
    merge_dir = os.path.join(output_dir, "03_merged")
    tif_dir = os.path.join(output_dir, "04_convert_to_UTM_tif")
    slope_dir = os.path.join(output_dir, "05_slope")

    for dir_name in [output_dir, download_dir, unzip_dir, tif_dir, merge_dir, slope_dir]:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

    # find which tile nubers to download
    tile_names = get_SRTM_tile_names(aoi_path)

    # download
    download_srtm_tiles(tile_names, download_dir)

    # unzip
    do_unzip(download_dir, unzip_dir)

    # create vrt from these files
    vrt_file = os.path.join(merge_dir, aoi_name + "_SRTM30m_fullcov.vrt")
    in_dirs = [os.path.join(unzip_dir, f) for f in os.listdir(unzip_dir)]
    vrt_from_folder(vrt_file, in_dirs, [".hgt"])

    # convert to tiff and reproject to utm
    convert_to_utm_tif(merge_dir, tif_dir, ".vrt")

    # slope
    create_slope(tif_dir, slope_dir)


def SRTM_to_slope_cmd():
    """Entrypoint to call SRTM_to_slope through cmd"""

    # parse aguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--aoi_name", type=str)
    parser.add_argument("--aoi_path", type=str)
    args = parser.parse_args()

    # get arguments
    output_dir = args.output_dir
    aoi_name = args.aoi_name
    aoi_path = args.aoi_path

    SRTM_to_slope(output_dir, aoi_name, aoi_path)


if __name__ == "__main__":

    # can be used to override arguments in case we want to run in interpreter
    sys.argv = [
        "SRTM_to_slope.py",
        "--aoi_path",
        r"\\sample\path\10_GADM\gadm40_LBR_shp\gadm40_LBR_0.shp",
        "--aoi_name",
        "India",
        "--output_dir",
        r"\\sample\out",
    ]

    SRTM_to_slope_cmd()
