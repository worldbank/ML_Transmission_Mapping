import os

import geopandas as gpd
import pandas as pd


from postprocessing.network import TowerNetwork
from postprocessing.points_to_path import (
    add_nearest_neighbors,
    db_scan_clustering,
    points_to_lines,
)

def join_lines_from_db(db_config: str, run_id: int, batch_id: str, max_dist: int, max_nn_dist: int, max_points_per_linestring: int, outfile: str):
    batch_id_postproc = "181_final"
    run_id = 181

    query = f"""SELECT json_build_object(
                    'type', 'FeatureCollection',
                    'features', json_agg(ST_AsGeoJSON(t.*)::json)
                    )
                    FROM
                    (select id, tower_uuid, ST_Transform(geom,3857) from powertower
            where
            tower_uuid in
            (select tower_uuid from subsublines_towers where accepted=True and batch_id='{batch_id_postproc}')
            and
            powertower.tower_uuid not in
            (select tower_uuid from accepted_towers where batch_id='181_testing_osm_1')) as t;
            """

    dbconfig = os.path.join(os.getcwd(), "..\\database\\amazon.ini")
    runconfig_path = os.path.join(
        os.getcwd(), "..\\configs\\electricity_network_mappping_local_yemen.json"
    )

    tn = TowerNetwork(runconfig_path, dbconfig, run_id, in_post=True)

    with tn.connection:
        with tn.connection.cursor() as curs:
            curs.execute(query)
            geojson = curs.fetchall()[0][0]
    df = gpd.GeoDataFrame.from_features(geojson["features"])
    df.loc[df.tower_uuid.isna(), "tower_uuid"] = pd._testing.rands_array(
        12, sum(df.tower_uuid.isnull())
    )
    df_proj = df.to_crs("EPSG:3857")
    df_proj = df_proj[df_proj.geometry.is_valid]  # handle invalid geometries
    result_df = db_scan_clustering(df_proj, max_dist)
    with_nbs = add_nearest_neighbors(result_df, max_nn_dist)
    line_dict = points_to_lines(with_nbs, max_points_per_linestring)

    print("done")

    print(line_dict)

    mydf = pd.DataFrame.from_dict(line_dict, orient="index")
    final = gpd.GeoDataFrame(mydf, geometry=mydf.geometry)

    df3857 = final.set_crs(3857)
    df3857.to_crs(4326).to_file(
        outfile
    )

def join_lines_from_shp(path_to_shp: str, max_dist: int,max_nn_dist: int, max_points_per_linestring: int, outfile: str):
    df = gpd.read_file(path_to_shp)
    df.loc[df.tower_uuid.isna(), "tower_uuid"] = pd._testing.rands_array(
    12, sum(df.tower_uuid.isnull())
)
    df_proj = df.to_crs("EPSG:3857")
    df_proj = df_proj[df_proj.geometry.is_valid]  # handle invalid geometries
    result_df = db_scan_clustering(df_proj, max_dist)
    with_nbs = add_nearest_neighbors(result_df, max_nn_dist)
    line_dict = points_to_lines(with_nbs, max_points_per_linestring)

    print("done")
    print(line_dict)

    mydf = pd.DataFrame.from_dict(line_dict, orient="index")
    final = gpd.GeoDataFrame(mydf, geometry=mydf.geometry)

    df3857 = final.set_crs(3857)
    df3857.to_crs(4326).to_file(
        outfile
    )
    

if __name__ == "__main__":

    max_dist = 2000 # maximum distance between clusters in DB scan algorithm
    max_nn_dist  =  3000 #maximum distance for nearest neighbours calc
    max_points_per_linestring = 1000 # maximum vertices in joined network segments

    join_lines_from_shp(r"sample\path\india_all.shp", max_dist, max_nn_dist, max_points_per_linestring, r"\\sample\path\lines_2.shp")