"""
Utility module used internally.
"""

import json
import math
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from osgeo import gdal, ogr
from rasterio.mask import mask


def clip_shp(in_shp, clip_shp, out_shp):
    """Clip shape with another shape

    Args:
        in_shp (str): shape to clip
        clip_shp (str): shape that is used as clip
        out_shp (str): shape to create
    """

    # open shape to be clipped
    if in_shp.endswith(".shp"):
        driverName = "ESRI Shapefile"
    elif in_shp.endswith(".gpkg"):
        driverName = "GPKG"
    else:
        raise Exception("unknown extension {}".format(in_shp))
    driver = ogr.GetDriverByName(driverName)
    inDataSource = ogr.Open(in_shp, 0)
    if not inDataSource:
        print("could not open {}".format(in_shp))
    inLayer = inDataSource.GetLayer()

    # shape to clip with
    inClipSource = ogr.Open(clip_shp, 0)
    inClipLayer = inClipSource.GetLayer()

    # perform clipping
    outDataSource = driver.CreateDataSource(out_shp)
    outLayer = outDataSource.CreateLayer(
        "FINAL", inLayer.GetSpatialRef(), geom_type=ogr.wkbMultiLineString
    )

    outLayer.StartTransaction()
    ogr.Layer.Clip(inLayer, inClipLayer, outLayer)
    outLayer.CommitTransaction()

    # cleanup
    inDataSource.Destroy()
    inClipSource.Destroy()
    outDataSource.Destroy()


def world2Pixel(gt, x, y):
    """Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate

    Args:
        gt (list): GDAL geomatrix
        x (float): X coordinate to transform
        y (float): Y coordinate to transform

    Returns:
        pixel (int): X position of pixel within image
        line (int): Y position of pixel within image
    """

    ulX = gt[0]
    ulY = gt[3]
    xDist = gt[1]
    yDist = gt[5]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / yDist)

    return (pixel, line)


def pixel2World(gt, x, y):
    """Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the geospatial coordinate from a pixel location

    Args:
        gt (list): GDAL geomatrix
        x (float): X position of pixel within image
        y (float): Y position of pixel within image

    Returns:
        x_world (float): x coordinate
        y_world (float): y coordinate
    """

    ulX = gt[0]
    ulY = gt[3]
    xDist = gt[1]
    yDist = gt[5]

    x_world = ulX + (x * xDist)
    y_world = ulY + (y * yDist)  # if y is negative, wil be subtracted

    return (x_world, y_world)


def cell_to_poly(geoMatrix, x, y):

    ulX, ulY = pixel2World(geoMatrix, x, y)

    xDist = geoMatrix[1]
    yDist = geoMatrix[5]

    poly = create_poly(ulX, ulY, xDist, yDist)

    return poly


def create_poly(ulX, ulY, xDist, yDist):
    """Creates ogr polygon geometry for a cell

    Args:
        ulX (float): Left X coordinate of cell
        ulY (float): Upper Y coordinate of cell
        xDist (float): X length of cell
        yDist (float): Y length of cell

    Returns:
        poly (ogr.geometry): polygon of cell
    """

    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(ulX, ulY)
    ring.AddPoint(ulX + xDist, ulY)
    ring.AddPoint(ulX + xDist, ulY + yDist)
    ring.AddPoint(ulX, ulY + yDist)
    ring.AddPoint(ulX, ulY)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    return poly


def gdalReadArrayOffsets(
    rbs, x_start_in, y_start_in, clip_nx, clip_ny, img_nx, img_ny, dt_NP=float, padding_value=0
):
    """function that reads in array with gdal with offsets, checks if offset falls outisde extent of the image,
    if it does, it pads it.

    Args:
        rbs (list): list of raster bands to read from
        x_start_in (int): x_offset of patch in relation to img
        y_start_in (int): y_offset of patch in relation to img
        clip_nx (int): x size of patch to be created
        clip_ny (int): y size of patch to be created
        img_nx (int): x size of input image
        img_ny (int): y size of input image

    Returns:
        arr_in (np.array): np.array of with shape [clip_nx, clip_ny,n_bnds]
    """

    assert (
        x_start_in <= img_nx
    ), "x_start_in  {} already falls outside image_nx {}, how is that possible".format(
        x_start_in, img_nx
    )

    x_start_in = round(x_start_in)
    y_start_in = round(y_start_in)

    n_bnds = len(rbs)

    # initialize input array
    if type(padding_value) == float:
        dt_NP = float
    arr_in = np.zeros(
        (clip_ny, clip_nx, n_bnds), dtype=dt_NP
    )  # in numpy shape first is y then is x
    if padding_value:
        arr_in[:] = padding_value

    # end of tile in offset
    x_end_in = x_start_in + clip_nx
    y_end_in = y_start_in + clip_ny

    clip_ny_or = clip_ny
    clip_nx_or = clip_nx

    # readAsArray only possible within image, so check if part to read falls within array
    if x_start_in < 0:
        x_start_tile = 0 - x_start_in
        x_start_in = 0
        clip_nx = clip_nx - x_start_tile
    else:
        x_start_tile = 0

    if y_start_in < 0:
        y_start_tile = 0 - y_start_in
        y_start_in = 0
        clip_ny = clip_ny - y_start_tile
    else:
        y_start_tile = 0

    if x_end_in > img_nx:
        x_end_in = img_nx
        clip_nx = x_end_in - x_start_in
        x_end_tile = x_start_tile + clip_nx
    else:
        x_end_tile = clip_nx_or

    if y_end_in > img_ny:
        y_end_in = img_ny
        clip_ny = y_end_in - y_start_in
        y_end_tile = y_start_tile + clip_ny
    else:
        y_end_tile = clip_ny_or

    for x, rb in enumerate(rbs):

        arr_in[
            int(y_start_tile) : int(y_end_tile), int(x_start_tile) : int(x_end_tile), x
        ] = rb.ReadAsArray(int(x_start_in), int(y_start_in), int(clip_nx), int(clip_ny))

    return arr_in


def read_image_subset(
    in_path, geom_list, band_numbs, pixel_mode="centroid", padding_value=None, return_mask=True
):
    """Read the values from the image and provide an object mask
    it was tested that reating the object_mask is a matter of miliseconds so can always be done

    Args:
        in_path (str): imagefile path to get array subset from
        geom_list (list): list of tuples (geom_id, osgeo geometry)
        band_numbs (list): list with band numbs to extract (provide either this or band_names)
        pixel_mode (str): either 'centroid' or 'within' or 'touching'
        return_mask (bool): whether the mask band (with the object geom) is included
        padding_value (float):  if none, there is no padding (wil skip if geom not within img),
                        else will pad with whatever is given

    Returns:
        result_dict (dict): {'object_mask':xx,'observations':xx} where xx is a np.array and
            False if part of the array to read falls outside the extent of the image
    """

    # assertions
    assert pixel_mode in (
        "within",
        "centroid",
        "touching",
    ), "pixel mode {} is not allowed, has to be within/touching/centroid".format(pixel_mode)
    assert type(band_numbs) == list, "band_numbs has to be a list"

    # open image and get bands
    ds = gdal.Open(in_path)
    gt = ds.GetGeoTransform()
    total_num_bands = ds.RasterCount
    bands = []

    if band_numbs == ["ALL_BANDS"]:
        for band_num in range(ds.RasterCount):
            rb = ds.GetRasterBand(band_num + 1)
            bands.append(rb)
    else:
        for band_numb in band_numbs:
            if band_numb <= total_num_bands:
                rb = ds.GetRasterBand(band_numb)
                bands.append(rb)
            else:
                print(
                    "band number {} is not present in file with {} bands".format(
                        str(band_numb), str(total_num_bands)
                    )
                )

    # translate parcel properties to image subset location
    n_bands = len(bands)

    result_dict = {}
    if n_bands == 0:
        return result_dict

    for geom_id, geom in geom_list:
        result = clip(
            geom,
            bands,
            gt,
            pixel_mode=pixel_mode,
            return_mask=return_mask,
            padding_value=padding_value,
        )

        # put in dictionary
        if result:
            image_values, object_mask, gt_result = result
            result = {}
            result["object_mask"] = object_mask
            result["observations"] = image_values
            result_dict[geom_id] = result

    # close
    ds = None

    return result_dict


def convert_wgs_to_utm(lon: float, lat: float):
    """Based on lat and lng, return best utm epsg-code

    Args:
        lon (float): Longiude
        lat (float): Latitude

    Returns:
       epsg_code (str): EPSG code
    """

    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = "0" + utm_band
    if lat >= 0:
        epsg_code = "EPSG:326" + utm_band
        return epsg_code
    epsg_code = "EPSG:327" + utm_band

    return epsg_code


def linear_rescale(in_raster, rescaling_params):
    """applies linear rescaling to numpy array

    Args:
        in_raster (np.array): input numpy array
        rescaling_params (list): [min_val_in, max_val_in, min_val_out, max_val_out  ]

    Returns:
        in_raster_scaled (np.array): rescaled numpy array

    """
    min_val_in, max_val_in, min_val_out, max_val_out = rescaling_params

    in_raster[in_raster < min_val_in] = min_val_in
    in_raster[in_raster > max_val_in] = max_val_in

    in_raster_scaled = min_val_out + ((in_raster - min_val_in) * (max_val_out - min_val_out)) / (
        max_val_in - min_val_in
    )

    return in_raster_scaled


def clip(geom, bands, gt, pixel_mode="centroid", return_mask=True, padding_value=False):
    """Read the values from the image and provide an object mask

    Args:
        geom (ogr.geom): ogr polygon for which to get raster pixel values
        gt (list):  gdal geotransform of image from which to read the pixels
        pixel_mode (str): one of ('centroid','touching','within') to decide which pixels to return
        return_mask (bool): whether to return a mask (boolean array) of the geom
        padding_value (float):  if none, there is no padding (wil skip if geom not within img),
                        else will pad with whatever is given

    Returns:
        image_values (np.array): array of imag epixels
        object_mask (np.array): boolean array of geom
        gt_result (list): gdal geotransform of output array

    """

    # get np datatype
    dt_GDAL = bands[0].DataType
    dt_trans = {
        1: np.int8,
        2: np.uint16,
        3: np.int16,
        4: np.uint32,
        5: np.int32,
        6: np.float32,
        7: np.float64,
        10: np.complex64,
        11: np.complex128,
    }
    dt_NP = dt_trans[dt_GDAL]

    nX = bands[0].XSize
    nY = bands[0].YSize

    # translate parcel properties to image subset location
    n_bands = len(bands)
    xmin, xmax, ymin, ymax = geom.GetEnvelope()

    px = (xmin - gt[0]) / gt[1]
    py = (ymax - gt[3]) / gt[5]
    nx = (xmax - gt[0]) / gt[1] - px
    ny = (ymin - gt[3]) / gt[5] - py
    px, py, nx, ny = int(px), int(py), int(nx), int(ny)

    # check that it is a valid part of the image
    is_in_image = True
    if nx == 0 or ny == 0:
        is_in_image = False
    if px < 0 or py < 0:
        is_in_image = False
    elif px + nx >= nX:
        is_in_image = False
    elif py + ny >= nY:
        is_in_image = False

    if not is_in_image:
        print(geom.ExportToWkt())
    if padding_value is None and not is_in_image:
        print("geom does not fall within the image")
        return False

    # prepare result geotransform
    gt_result = (
        gt[0] + px * gt[1],
        gt[1],
        0,
        gt[3] + py * gt[5],
        0,
        gt[5],
    )

    # init arrays
    image_values = np.zeros((ny, nx, n_bands), dtype=float)
    object_mask = np.zeros((ny, nx), dtype=bool)

    # read image_values
    if padding_value and not is_in_image:
        image_values = gdalReadArrayOffsets(
            bands, px, py, nx, ny, nX, nY, dt_NP=dt_NP, padding_value=padding_value
        )
    else:
        # start reading
        image_values = np.zeros((ny, nx, n_bands), dtype=dt_NP)
        for x, band in enumerate(bands):
            image_values[:, :, x] = band.ReadAsArray(px, py, nx, ny)

    if return_mask:
        # get parcel_mask
        rtempdriver = gdal.GetDriverByName("MEM")
        stempdriver = ogr.GetDriverByName("MEMORY")

        # create a temporary OGR layer with one feature
        source = stempdriver.CreateDataSource("memData")
        tlyr = source.CreateLayer("", None, ogr.wkbMultiPolygon)
        tdefn = tlyr.GetLayerDefn()
        tfeat = ogr.Feature(tdefn)
        tfeat.SetGeometry(geom)
        tlyr.CreateFeature(tfeat)
        tlyr.SyncToDisk()
        source.SyncToDisk()

        # create a temporary raster with the image subset
        target_ds = rtempdriver.Create("", int(nx), int(ny), 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform(gt_result)
        if pixel_mode == "touching":
            options = ["ALL_TOUCHED=TRUE"]
        else:
            options = []
        gdal.RasterizeLayer(target_ds, [1], tlyr, burn_values=[1], options=options)

        if pixel_mode == "within":
            tlyr2 = source.CreateLayer("border", None, ogr.wkbMultiLineString)
            tdefn2 = tlyr2.GetLayerDefn()
            tfeat2 = ogr.Feature(tdefn2)
            tfeat2.SetGeometry(geom.GetBoundary())
            tlyr2.CreateFeature(tfeat2)
            tlyr2.SyncToDisk()
            source.SyncToDisk()
            gdal.RasterizeLayer(
                target_ds, [1], tlyr2, burn_values=[0], options=["ALL_TOUCHED=TRUE"]
            )

        tband = target_ds.GetRasterBand(1)
        object_mask[:] = np.array(tband.ReadAsArray())

        source.Destroy()
        target_ds = None

    return image_values, object_mask, gt_result


def save_raster(path, raster, affine, crs=None, nodata=0):
    """Save a raster to the specified file.

    Args:
        file (str): Output file path
        raster (numpy.array): 2D numpy array containing raster values
        affine (affine.Affine): Affine transformation for the raster
        crs (str, proj.Pro,): optional (default EPSG4326) CRS for the raster
    """

    path = Path(path)
    if not path.parents[0].exists():
        path.parents[0].mkdir(parents=True, exist_ok=True)

    if not crs:
        crs = "+proj=latlong"

    filtered_out = rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=raster.shape[0],
        width=raster.shape[1],
        count=1,
        dtype=raster.dtype,
        compress="deflate",
        sparse_ok=True,
        crs=crs,
        transform=affine,
        nodata=nodata,
    )
    filtered_out.write(raster, 1)
    filtered_out.close()


def vrt_from_folder(vrt_file, in_folder_list, ext_list, extra_options=[]):
    """function that creates vrt, from a list of folders for files with a list of endings

    Args:
        vrt_file (str): full path of output vrt file
        in_folder_list (list): list of folders in which to lok for files to add to vrt,
                            folders listed last will have priority in the vrt in case of overlap
        ext_list (list): list of file endings which files can have to be added to vrt
        options (list): extra gdal vrt command options can be added, e,g, '-seperate' to create seperate bands
    """

    assert all(
        isinstance(v, list) for v in [in_folder_list, ext_list, extra_options]
    ), "in_folders and in ext_list have to be lists"

    # create list
    file_list = []
    for in_folder in in_folder_list:
        for f in os.listdir(in_folder):
            for ext in ext_list:
                if f.endswith(ext.lower()) or f.endswith(ext.upper()) or f.endswith(ext):
                    file_list.append(os.path.join(in_folder, f))

    # assert some files were found

    vrt_from_list(vrt_file, file_list, extra_options=extra_options)


def vrt_from_list(vrt_file, file_list, extra_options=[]):
    """function that creates vrt, from a list of paths

    Args:
        vrt_file (str): full path of output vrt file
        file_list (list): list of files
        options (list):  extra gdal vrt command options can be added, e,g, '-seperate' to create seperate bands
    """

    # write list to text file
    txt_file = vrt_file[:-4] + ".txt"
    with open(txt_file, "w") as F:
        for file_path in file_list:
            F.write(file_path + "\n")

    vrt_options = gdal.BuildVRTOptions(extra_options)
    gdal.BuildVRT(vrt_file, file_list, options=vrt_options)
    print("vrt created")


# clip_raster is copied from openelec.clustering
def clip_raster(raster, boundary, boundary_layer=None):
    """Clip the raster to the given administrative boundary.

    Args:
        raster (str or rasterio.io.DataSetReader): Location of or already opened raster.
        boundary (str or geopandas.GeoDataFrame) : The polygon by which to clip the raster.
        boundary_layer (str) :  optional: for multi-layer files (like GeoPackage), specify the layer to be used.

    Returns:
        tuple
            Three elements:
                clipped : numpy.ndarray
                    Contents of clipped raster.
                affine : affine.Affine()
                    Information for mapping pixel coordinates
                    to a coordinate system.
                crs : dict
                    Dict of the form {'init': 'epsg:4326'} defining the coordinate
                    reference system of the raster.

    """

    if isinstance(raster, Path):
        raster = str(raster)
    if isinstance(raster, str):
        raster = rasterio.open(raster)

    if isinstance(boundary, Path):
        boundary = str(boundary)
    if isinstance(boundary, str):
        if ".gpkg" in boundary:
            driver = "GPKG"
        else:
            driver = None  # default to shapefile
            boundary_layer = ""  # because shapefiles have no layers

        boundary = gpd.read_file(boundary, layer=boundary_layer, driver=driver)

    if not (boundary.crs == raster.crs or boundary.crs == raster.crs.data):
        boundary = boundary.to_crs(crs=raster.crs)
    coords = [json.loads(boundary.to_json())["features"][0]["geometry"]]

    # mask/clip the raster using rasterio.mask
    clipped, affine = mask(dataset=raster, shapes=coords, crop=True)

    if len(clipped.shape) >= 3:
        clipped = clipped[0]

    return clipped, affine, raster.crs


def get_geom_wkt_list(shp, intersection_wkt=None, attribute_name=None, attribute_value=None):
    """function that reads all the geometries in a shapefile into list of wkt geoms

    Args:
        shp (str): path of shapefile
        intersection_wkt (str): only get geoms hat intersect with this wkt
        attribute_name (str): allows to set filter on cerain attribute name
        attribute_value (str): filter only geoms where attribute_name has this value
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


if __name__ == "__main__":

    import time

    start_time = time.time()

    in_shp = r"\\sample\path\01_OpenData\Yemen\03_OSM\01_download\highway_road_power.shp"
    clip_shps = (
        r"\\sample\path\03_PilotAOI\ProbabilityMapAOI\Yemen_ProbMap_AOI.shp"
    )
    out_shp = r"\\sample\path\01_OpenData\Yemen\03_OSM\01_download\highway_road_power_clipped3.shp"

    clip_shp(in_shp, clip_shps, out_shp)

    print("--- %s seconds ---" % (time.time() - start_time))
