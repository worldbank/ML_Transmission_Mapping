from __future__ import annotations

import json
import os
from configparser import ConfigParser
from dataclasses import asdict
from typing import Optional
from uuid import uuid4

import geopandas as gpd
import pandas as pd
import pandas.io.sql as sqlio
import psycopg2
import shapely
from geoalchemy2 import Geometry
from shapely.geometry import LineString
from sqlalchemy import create_engine
from tqdm import tqdm
import time
   

from database.database_utils import get_db_config
from utils.costMapInterface_fromfile import CostMapInterface_from_file
from utils.sublines import (
    add_geometry_to_sublines,
    compare_direction,
    fetch_a_subline,
    split_subline_by_direction,
    subline_scores_and_num_towers,
    write_cost_to_powertower,
    write_numtowers_to_db,
    write_subline_scores_to_db,
)


class TowerNetwork:
    def __init__(
        self,
        run_config_path: Optional[str] = None,
        db_config_path: Optional[str] = None,
        run_id: int = None,
        in_post: bool = False,
    ):
        """Class that can fetch results of smart-tracing and perform postprocessing

        Args:
            run_config_path (Optional[str], optional): Run config file used in smart-tracing. Defaults to None.
            db_config_path (Optional[str], optional): DB config file. Defaults to None.
            run_id (int, optional): Run ID in database. Defaults to None.
            in_post: whether or not this class is being used in postproc or not (default False)
        """

        assert os.path.exists(db_config_path), f"db config file cannot be found {db_config_path}"
        assert os.path.exists(run_config_path), "run config file cannot be found {}".format(
            run_config_path
        )
        self.db_config_path = db_config_path
        self.connection = psycopg2.connect(**get_db_config(self.db_config_path))

        with open(run_config_path, "r", encoding="utf8") as f:
            run_config = json.load(f)

        self.run_config = run_config

        if not in_post:
            self.costmap_path = run_config["costmap_path"]
            self.costmap_interface = CostMapInterface_from_file(self.costmap_path)

        self.run_id = run_id
        self.towers_gdf = gpd.GeoDataFrame()
        self.sublines_gdf = gpd.GeoDataFrame()
        self.startpoints_gdf = gpd.GeoDataFrame()

    def select_distinct_tmp_line_ids(
        self, multiple_towers: bool = False
    ):
        """Select distinct tmp_line_ids from powertowers table for a certain run

        Args:
            multiple_towers (bool, optional): Only selecting tmp_line_ids that have towers. Defaults to False.

        Returns:
            List(str): list of unique tmp_line_ids
        """

        query = """SELECT DISTINCT tmp_line_id from powertower where run_id = %s;"""

        if multiple_towers:
            query = """select distinct tmp_line_id from powertower
            where run_id=%s
            and tmp_line_id in (select id from sublines where num_towers >1);"""

        else:
            query = """select distinct tmp_line_id from powertower
            where run_id=%s
            and tmp_line_id in (select id from sublines);"""

        params = [self.run_id]

        with self.connection:
            with self.connection.cursor() as curs:
                curs.execute(query, params)
                result = curs.fetchall()

        self.distinct_tmp_line_ids = [record[0] for record in result]

        return self.distinct_tmp_line_ids

    def make_subsublines(self, direction_cutoff: int = 40, batch_id: str = None):
        """Make subsublines (straight line segments) from sublines

        Args:
            direction_cutoff (int, optional): Direction change (degrees) to
                indicate new line segment. Defaults to 40.
            batch_id (str): Batch ID of the postprocessing run
        """

        # set a batch_id for the database if it doesnt exist
        if not batch_id:
            batch_id = str(uuid4())
            print(batch_id)

        print("Making subsublines from straight line segments")
        for subline_id in tqdm(self.distinct_tmp_line_ids):
            # only sublines with points should go in here
            subline = fetch_a_subline(self.db_config_path, subline_id, only_good_towers=False)
            subsublines, _ = split_subline_by_direction(subline, direction_cutoff)

            if subsublines:

                as_gdf = pd.concat([pd.json_normalize(asdict(i)) for i in subsublines])

                # 1A: update subsublines temp table
                crs = "epsg:4326"
                for_db = gpd.GeoDataFrame(
                    as_gdf.drop(
                        columns=[
                            "tower_ids",
                            "tower_costs",
                            "tower_coords",
                            "tower_scores",
                            "split_from",
                            "startingpoint",
                            "tower_spent_budget",
                        ]
                    ),
                    crs=crs,
                    geometry="geometry",
                )
                for_db["run_id"] = self.run_id
                for_db["batch_id"] = batch_id
                for_db = for_db.rename(columns={"my_id": "id", "parent_line": "parent_subline"})
                dsn_params = self.connection.get_dsn_parameters()
                db_url = (
                    "postgresql://{user}:{pw}@{host}:{port}/{database}".format(dsn_params['user'], dsn_params['password'],dsn_params['host'],dsn_params['port'],dsn_params['databse'])
                )

                engine = create_engine(db_url)
                with self.connection:
                    try:
                        for_db.to_postgis(
                            "subsublines",
                            engine,
                            if_exists="append",
                            dtype={"geometry": Geometry("LINESTRING", srid="4326")},
                        )

                    except Exception as e:
                        print("Problem inserting subsublines", e)

                # 1B Update subsub/powertower table
                try:
                    with self.connection:
                        with self.connection.cursor() as curs:
                            # first for the subsublines
                            for _, row in as_gdf.iterrows():

                                for tower_id in row["tower_ids"]:
                                    query = """INSERT INTO subsublines_towers(tower_uuid,subsubline, batch_id)
                                        VALUES ('%s','%s','%s')""" % (
                                        tower_id,
                                        row["my_id"],
                                        batch_id,
                                    )
                                    curs.execute(query)

                except Exception as e:
                    print(str(e))

    def get_min_distance_in_a_gdf(self, df: gpd.GeoDataFrame, source: LineString):
        """Calculate the minimum distance between a LineString object and a geodataframe of other Linestrings

        Args:
            df (gpd.GeoDataFrame): GeoDataFrame containing linestring objects
            source (LineString): Linestring geometry of interest

        Returns:
            pd.Series: Series of tuples (min distance and its argument)
        """
        distances = df.distance(source)
        return distances.min(), distances.argmin()

    def accept_OSM_towers(self, batch_id: str, path_to_osm: str, buff_dist: int):
        """Accept towers that lie within a certain distance of OSM lines.

        Args:
            batch_id (str): Postproc batch id
            path_to_osm (str): Path to OSM lines shapefile/geojson
            buff_dist (int): Acceptance distance (m)
        """

        osm = gpd.read_file(path_to_osm)
        osm = osm.to_crs("EPSG:3857")

        # sort by length
        osm["length"] = osm.geometry.length
        osm.sort_values(by=["length"], ascending=False, inplace=True)

        towers = tn.towers_gdf.set_crs("EPSG:4326")
        towers = towers.to_crs("EPSG: 3857")

        gdf_accepted = gpd.GeoDataFrame()

        for j, poly in osm.iterrows():

            k = towers.within(poly.geometry.buffer(buff_dist))
            if len(k.value_counts().values) > 1:
                towers["osm_id"] = str(j)

                gdf_accepted = pd.concat([gdf_accepted, towers[k]])

        # drop duplicates keeping only the longest osm line
        gdf_accepted.drop_duplicates(subset="tower_uuid", keep="first", inplace=True)

        print(f"we have accepted {len(gdf_accepted)} towers within {buff_dist}m of OSM lines ")
        try:
            for _, row in gdf_accepted.iterrows():

                query = """INSERT INTO public.accepted_towers (tower_uuid, run_id, batch_id, cluster_id)
                    values ('%s', %i, '%s', '%s') """ % (
                    row.tower_uuid,
                    self.run_id,
                    batch_id,
                    "temp",
                )
                with self.connection:
                    with self.connection.cursor() as curs:
                        curs.execute(query)

        except Exception as e:
            print(e)

    def filter_subsubline_level(
        self,
        batch_id: str,
        cost_thresh: float,
        score_thresh: float,
        angle_cutoff: int = 30,
        distance_cutoff: int = 1000,
    ):
        """Filter subsublines.

        Args:
            batch_id (str): ID of batch postprocessing
            cost_thresh (float): Cost threshold
            score_thresh (float): Score threshold
            angle_cutoff (int, optional): Cutoff angle for nearby lines. Defaults to 30.
            distance_cutoff (int, optional): Distance cutoff for nearby lines (m). Defaults to 1000m.

        Returns:
            List(str): Accepted subsubline ids
        """
        accepted_ssl_ids = []

        # get the subsublines
        self.fetch_subsublines(batch_id)

        # first round of filtering
        accepted_ssls = self.subsublines_gdf[
            (self.subsublines_gdf["mean_score"] >= score_thresh)
            & (self.subsublines_gdf["mean_cost"] <= cost_thresh)
        ]
        rejected_ssls = self.subsublines_gdf[
            ~(
                (self.subsublines_gdf["mean_score"] >= score_thresh)
                & (self.subsublines_gdf["mean_cost"] <= cost_thresh)
            )
        ]

        for _, row in accepted_ssls.iterrows():
            accepted_ssl_ids.append(row["id"])

        # convert the crs for distance measurements
        rejected_ssls = rejected_ssls.to_crs(3857)
        accepted_ssls = accepted_ssls.to_crs(3857)

        distances = rejected_ssls.geometry.apply(
            lambda x: self.get_min_distance_in_a_gdf(accepted_ssls, x)
        )
        rejected_ssls["distances"], rejected_ssls["distancesargs"] = distances.str
        
        
        # second round filtering
        # accept subsublines with a similar direction within a certain distance of an accepted ssls
        print("Checking distance between subsublines")

        for i, row_rej in tqdm(rejected_ssls.iterrows()):
            # check against all accepted
            if row_rej["distances"] < distance_cutoff:
                angle_diff = compare_direction(
                    row_rej["mean_direction"],
                    accepted_ssls.iloc[[row_rej["distancesargs"]]]["mean_direction"].values,
                )
                if angle_diff > angle_cutoff:
                    continue

                accepted_ssl_ids.append(row_rej["id"])

        # write the results
        print(
            f"we have accepted {len(accepted_ssl_ids)} ssls out of a possible {len(self.subsublines_gdf)}"
        )
        try:
            for ssl_id in tqdm(accepted_ssl_ids):

                query = (
                    """UPDATE public.subsublines_towers SET accepted = True WHERE subsubline = '%s' and batch_id='%s'"""
                    % (
                        ssl_id,
                        batch_id,
                    )
                )
                with self.connection:
                    with self.connection.cursor() as curs:
                        curs.execute(query)

        except Exception as e:
            print(e)

        return accepted_ssl_ids

    def filter_tower_level(self, batch_id: str):
        """Tower level filtering based on nearby subsublines.

        Args:
            batch_id (str): ID of postprocessing batch
        """
        print("Performing tower-level filtering for batch id %s" % batch_id)

        # filter the accepted subsublines and accept all towers that lie within buffer
        # buffer sublines by distance to previous line
        query = (
            """SELECT id, st_astext(ST_Buffer(st_extend(subsublines.geometry,3,0,3,0), 0.0002))
            from subsublines where subsublines.id in
            (SELECT subsubline from subsublines_towers
            where accepted=True and batch_id='%s');"""
            % batch_id
        )

        try:
            with self.connection:
                gdf = gpd.GeoDataFrame(sqlio.read_sql_query(query, self.connection))
                gdf["geometry"] = gdf["st_astext"].apply(shapely.wkt.loads)

                # overrwrite geom
                gdf.set_geometry("geometry")
                gdf = gdf.set_crs(4326)
                self.buffered_lines = gdf

        except Exception as e:
            print(e)

        # get the not yet accepted towers
        if len(self.towers_gdf) == 0:
            self.fetch_data()
        towers_unaccepted = self.fetch_not_yet_accepted(batch_id)

        try:
            with self.connection:
                with self.connection.cursor() as curs:

                    for _, row in tqdm(towers_unaccepted.iterrows()):
                        accepted = False

                        row = towers_unaccepted[towers_unaccepted["tower_uuid"] == row.tower_uuid]

                        # first very low cost towers are automatically accepted accepted
                        if row["cost"].values[0] < 0.09:
                            accepted = True
                            contained_by = "00000000-0000-0000-0000-000000000000" 
                            # placeholder

                        if not accepted:
                            # for each polygon check if the unaccepted tower falls inside it
                            for _, poly in self.buffered_lines.iterrows():
                                
                                polygon_for_shapely = self.buffered_lines[
                                    self.buffered_lines["id"] == poly.id
                                ]
                                if polygon_for_shapely.geometry.values.contains(
                                    row.geometry.set_crs(4326).values
                                ):
                                    contained_by = poly["id"]
                                    accepted = True
                                    break

                        if accepted:
                            query = f"""INSERT INTO subsublines_towers(tower_uuid, subsubline, batch_id, accepted)
                                VALUES ('{row["tower_uuid"].values[0]}','{contained_by}','{batch_id}', True)"""

                            try:
                                curs.execute(query)

                            except Exception as e:
                                print("Couldn't accept a tower in DB", e)
                                continue

        except Exception as e:
            print("Problem with geometry check for buffered subsublines", str(e))

    def fetch_not_yet_accepted(self, batch_id: str):
        """Fetch towers that have not been accepted yet.

        Args:
            batch_id (str): Batch ID of the postprocessing run

        Returns:
            GeoDataFrame: Dataframe containing towers and attributes
        """

        query_geojson = """SELECT json_build_object(
                        'type', 'FeatureCollection',
                        'features', json_agg(ST_AsGeoJSON(t.*)::json)
                        )
                        FROM (
                            SELECT geom, powertower.tower_uuid, public.powertower.id, tile_id, score, bought_tiles,
                            spent_budget, object_type, tmp_line_id, cost
                            FROM public.powertower
                            RIGHT JOIN public.run ON run_id = public.run.id
                            WHERE run_id = {} and powertower.tower_uuid NOT IN
                            (
                                SELECT tower_uuid from subsublines_towers where batch_id = '{}'
                            )
                            ) as t;""".format(
            self.run_id, batch_id
        )

        with self.connection:
            with self.connection.cursor() as curs:
                curs.execute(query_geojson)
                geojson = curs.fetchall()[0][0]

        return gpd.GeoDataFrame.from_features(geojson["features"])

    def fetch_data(self, limit: int = None):
        """Fetch raw results of smart-tracing (only towers)

        Args:
            limit (int, optional): Limit rows (for debugging). Defaults to None.
        """

        if limit is not None:
            limit_string = "LIMIT {}".format(limit)

        else:
            limit_string = ""

        query_geojson = """SELECT json_build_object(
                        'type', 'FeatureCollection',
                        'features', json_agg(ST_AsGeoJSON(t.*)::json)
                        )
                        FROM (
                            SELECT geom, powertower.tower_uuid, public.powertower.id, tile_id, score, bought_tiles,
                            spent_budget, object_type, tmp_line_id
                            FROM public.powertower
                            RIGHT JOIN public.run ON run_id = public.run.id
                            WHERE run_id = {} {}
                            ) as t;""".format(
            self.run_id, limit_string
        )

        with self.connection:
            with self.connection.cursor() as curs:
                curs.execute(query_geojson)
                geojson = curs.fetchall()[0][0]

        self.towers_gdf = gpd.GeoDataFrame.from_features(geojson["features"])

    def fetch_starting_points(self, set_name: str, type: str = "DB"):
        """Fetch the starting points used for smart-tracing.

        Args:
            set_name (str): Name of starting point set (column in table startingpoints)
            type (str, optional): From "File" or "DB". Defaults to "DB".
        """
        if type == "File":
            starting_point_files = self.run_config["_initial_startingpoints"]

            for starting_points_file in starting_point_files:
                with open(starting_points_file, "r", encoding="utf8") as f:
                    points = json.load(f)
                if "powerplant" in starting_points_file:
                    self.powerplants_starting = points["features"]
                elif "substation" in starting_points_file:
                    self.substation_starting = points["features"]
                else:
                    self.other_starting = points["features"]

        elif type == "DB":

            query_geojson = """SELECT json_build_object(
                            'type', 'FeatureCollection',
                            'features', json_agg(ST_AsGeoJSON(t.*)::json)
                            )
                            FROM (
                                SELECT *
                                FROM public.startingpoints
                                WHERE set_name LIKE %s
                                ) as t;"""
            params = [set_name]

            with self.connection:
                with self.connection.cursor() as curs:
                    curs.execute(query_geojson, params)
                    geojson = curs.fetchall()[0][0]

            self.startpoints_geojson = geojson
            self.startpoints_gdf = gpd.GeoDataFrame.from_features(geojson["features"])

    def fetch_sublines(self, tmp_line: bool = True):
        """Get the information from the sublines table and populate a geodataframe

        Args:
            tmp_line (bool, optional): Whether or not we only look for sublines with a
            corresponding tmp_line_id. Defaults to True.


        """

        query = """SELECT * from public.sublines WHERE run_id= %i;"""

        if tmp_line:
            query = """SELECT * from public.sublines where sublines.id in
            (select tmp_line_id from powertower) and run_id = %i;"""

        with self.connection:
            self.sublines_gdf = sqlio.read_sql_query(query % self.run_id, self.connection)

    def fetch_subsublines(self, batch_id: str):
        """Get the information from the subsublines table and populate a geodataframe

        Args:
            batch_id: str of the filtering batch
        """

        query = """SELECT *, ST_AsText(geometry) from public.subsublines WHERE batch_id= '%s';"""

        query_ssl = """SELECT * from public.subsublines_towers WHERE batch_id= '%s';"""

        try:
            with self.connection:
                gdf = gpd.GeoDataFrame(sqlio.read_sql_query(query % batch_id, self.connection))
                gdf.geometry = gpd.GeoSeries.from_wkt(gdf["st_astext"])  # overrwrite geom
                gdf.set_geometry("geometry")
                gdf = gdf.set_crs(4326)
                self.subsublines_gdf = gdf

                self.subsublines_towers_rel = sqlio.read_sql_query(
                    query_ssl % batch_id, self.connection
                )

        except Exception as e:
            print(e)

    def fetch_accepted_towers(self, batch_id: str):
        """Fetch towers that have been accepted.

        Args:
            batch_id (str): Batch ID of the post-processing/subsubline generation.
        """
        query = f"""SELECT json_build_object(
                    'type', 'FeatureCollection',
                    'features', json_agg(ST_AsGeoJSON(tab.*)::json)
                    )
                    FROM
                    (   select t.*, cluster_id from
                            (
                                select powertower.* from powertower
                                INNER JOIN accepted_towers ON ((powertower.tower_uuid = accepted_towers.tower_uuid)
                                and batch_id='{batch_id}')
                                where powertower.tower_uuid in
                                (SELECT tower_uuid from public.accepted_towers where batch_id='{batch_id}')
                                UNION
                                select powertower.* from powertower where tower_uuid in
                                (select tower_uuid from subsublines_towers where accepted=True and
                                  batch_id='{batch_id}')
                            ) as t
                        LEFT JOIN
                        accepted_towers ON ((t.tower_uuid = accepted_towers.tower_uuid) and batch_id='{batch_id}'))
                    as tab

                """

        with self.connection:
            with self.connection.cursor() as curs:
                curs.execute(query)
                geojson = curs.fetchall()[0][0]

        self.accepted_geojson = geojson

    def close_db_connection(self):
        self.connection.close()