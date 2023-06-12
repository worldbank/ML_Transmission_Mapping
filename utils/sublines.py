import json
from typing import Any, List
from uuid import uuid4

import geopandas as gpd
import matplotlib.pyplot as plt
import mercantile
import numpy as np
import psycopg2
from osgeo import ogr
from shapely.geometry import LineString
from tqdm import tqdm

from database.database_utils import get_db_config
from postprocessing.line_object import Line
from tracing.tracer import Tracer
from utils.costMapInterface_fromfile import CostMapInterface_from_file
from utils.dataclasses.point import Point

# Utility functions used in the making of sublines in postprocessing


def fetch_a_subline(db_config_path: str, tmp_line_id: str):
    """Fetch a subline from the database and make a Line object instance.

    Args:
        db_config_path (str): DB config path
        tmp_line_id (str): Line id of the subline in the powertower table

    Returns:
        Line: Line object containing subline attributes
    """
    where_str = "where (tmp_line_id = '%s') ORDER BY id ASC" % (tmp_line_id)

    query_powtow = (
        """SELECT tower_uuid, score,cost,spent_budget, ST_x(geom), ST_y(geom), geom
        FROM powertower """
        + where_str
    )

    query_sublines = """select * FROM sublines where id = '%s'""" % (tmp_line_id)

    connection = psycopg2.connect(**get_db_config(db_config_path))

    try:
        with connection.cursor() as curs:
            curs.execute(query_powtow)
            result_powtow = curs.fetchall()
            curs.execute(query_sublines)
            result_sublines = curs.fetchall()[0]

    finally:
        connection.close()

    line = Line(
        my_id=tmp_line_id,
        parent_line=result_sublines.parent_line,
        split_from=result_sublines.split_from,
        starting_tower=result_sublines.starting_tower,
        startingpoint=result_sublines.startingpoint,
        num_towers=len([i.tower_uuid for i in result_powtow]),
        tower_ids=[i.tower_uuid for i in result_powtow],
        tower_coords=[(i.st_x, i.st_y) for i in result_powtow],
        tower_scores=[i.score for i in result_powtow],
        tower_costs=[i.cost for i in result_powtow],
        tower_spent_budget=[i.spent_budget for i in result_powtow],
        geometry=result_sublines.geom,
        mean_score=None,
        mean_cost=None,
        mean_direction=None,
    )

    return line

def add_geometry_to_sublines(db_config_path: str, subline_ids: List[str]):
    """Add a Linestring geom to subline based on its towers

    Args:
        db_config_path (str): DB config path
        subline_ids (List): 1d list of subline ids
    """

    # first get all sublines

    connection = psycopg2.connect(**get_db_config(db_config_path))
    print("Found %i subline ids to update" % (len(subline_ids)))

    try:
        for id in tqdm(subline_ids):
            query = """UPDATE sublines
                SET
                geom = (select ST_MakeLine(
                    ARRAY( SELECT geom from powertower where tmp_line_id='%s' ORDER BY powertower.id ASC)))
                        where id= '%s'; """ % (
                id,
                id,
            )

            with connection.cursor() as curs:
                curs.execute(query)
    finally:
        connection.commit()
        connection.close()

def get_towers_in_a_subline(db_config_path: str, subline_id: str):
    """Get towers in a subline from db.

    Args:
        db_config_path (str): DB config path.
        subline_id (str): Subline UUID.

    Returns:
        List[Records]: List of tower records for the supploied subline.
    """

    query = """SELECT *, ST_X(geom), ST_Y(geom)
                    FROM public.powertower WHERE tmp_line_id = %s"""

    params = [subline_id]

    connection = psycopg2.connect(**get_db_config(db_config_path))

    try:
        with connection.cursor() as curs:
            curs.execute(query, params)
            result = curs.fetchall()
    finally:
        connection.close()

    return result

def extract_cost_at_point(x: float, y: float, cm_interface: CostMapInterface_from_file):
    """Get cost at point

    Args:
        x (_type_): Lon
        y (_type_): Lat

    Returns:
        cost (float): Cost at point from costmap
    """
    tile = mercantile.tile(x, y, zoom=18)
    cost = cm_interface.get_cost(tile.x, tile.y)

    return cost

def write_cost_to_powertower(db_config_path: str, run_id: int, costmap_path: str):
    """Calculate and write costs for each tower to db.

    Args:
        db_config_path (str): DB config path
        run_id (int): Run ID
        costmap_path (str): Costmap path
    """

    costmap_interface = CostMapInterface_from_file(costmap_path)

    # get all the towers
    query = """SELECT tower_uuid, ST_X(geom), ST_Y(geom)
    FROM public.powertower WHERE run_id = %s"""

    params = [str(run_id)]

    connection = psycopg2.connect(**get_db_config(db_config_path))

    print("Getting and setting costs for run_id %s" % run_id)

    try:
        with connection.cursor() as curs:
            curs.execute(query, params)
            result = curs.fetchall()

            for i, record in tqdm(enumerate(result)):
                # get cost
                cost = extract_cost_at_point(record.st_x, record.st_y, costmap_interface)

                query = """update powertower set cost = %s where tower_uuid='%s'""" % (
                    cost,
                    record.tower_uuid,
                )

                curs.execute(query)

    except Exception as e:
        print("cannot update cost", e)

    finally:
        connection.commit()
        connection.close()

def subline_scores_and_num_towers(db_config_path: str, subline_ids: List[str]):
    """Get sublines scores and number of constituent towers.

    Args:
        db_config_path (str): Path to db config file.
        subline_ids (List[str]): List of subline IDs to calculate for

    Returns:
        List, List: List of scores and number of towers
    """
    scores = []
    numtowers = []

    for _, id in tqdm(enumerate(subline_ids)):
        record = get_towers_in_a_subline(db_config_path, id)

        if record == []:
            # No towers exist in the subline

            scores.append(0)
            numtowers.append(0)
            continue

        df = gpd.GeoDataFrame(record)

        scores.append(df["score"].mean())
        numtowers.append(len(record))

    return scores, numtowers

def write_subline_scores_to_db(db_config_path: str, ids: np.ndarray, scores: np.ndarray):
    """Write scores to db for each subline

    Args:
        db_config_path (str): Path to DB config file.
        ids (List[str]): subline IDs
        scores (List[float]): List of scores
    """

    connection = psycopg2.connect(**get_db_config(db_config_path))

    print("Writing mean scores for %i sublines to DB" % (len(ids)))

    try:
        for i, id in tqdm(enumerate(ids)):
            query = """UPDATE public.sublines SET mean_score = %f WHERE id = '%s';""" % (
                scores[i],
                id,
            )
            with connection.cursor() as curs:
                curs.execute(query)
    finally:
        connection.commit()
        connection.close()

def write_numtowers_to_db(db_config_path: str, ids: np.ndarray, numtowers: np.ndarray):
    """Write number of towers to the sublines table

    Args:
        array (ndarray): [num towers, id]
    """
    connection = psycopg2.connect(**get_db_config(db_config_path))

    print("Writing num towers for %i sublines to DB" % (len(ids)))
    try:
        for i, id in tqdm(enumerate(ids)):
            query = """UPDATE public.sublines SET num_towers = %i WHERE id = '%s';""" % (
                numtowers[i],
                id,
            )
            with connection.cursor() as curs:
                curs.execute(query)
    finally:
        connection.commit()
        connection.close()

def split_subline_by_direction(line: Line, cutoff_dir_change: int):
    """Make straight line segments (subsublines) from a subline and writes the

    Args:
        line (Line): Subline as Line object
        cutoff_dir_change (int): Direction change (degrees) to indicate new line segment.

    Returns:
        List[Line]: List of subsubline Line objects
        List[str]: List of lonely/ floating tower UUIDs
    """

    # first get the directions
    breakidx = [0]
    restart = False
    for i in range(0, len(line.tower_ids) - 1):
        direction = Tracer.get_direction_between_points(
            Point(line.tower_coords[i][0], line.tower_coords[i][1]),
            Point(line.tower_coords[i + 1][0], line.tower_coords[i + 1][1]),
        )
        if i == 0 or restart:
            last_one = direction
            restart = False
            continue

        if abs(direction - last_one) > cutoff_dir_change:
            # get all the towers up to this point
            breakidx.append(i)
            restart = True

        last_one = direction
    breakidx.append(-1)

    # now we have the break indices, can make the new line objects
    subsublines = []
    lonelypoints = []

    for i in range(0, len(breakidx) - 1):

        tow_coords = line.tower_coords[breakidx[i] : breakidx[i + 1]]
        tow_scores = line.tower_scores[breakidx[i] : breakidx[i + 1]]
        tow_ids = line.tower_ids[breakidx[i] : breakidx[i + 1]]
        tow_costs = line.tower_costs[breakidx[i] : breakidx[i + 1]]
        tower_spent_budget = line.tower_spent_budget[breakidx[i] : breakidx[i + 1]]

        if len(tow_coords) < 3:
            if len(line.tower_ids) == 1:
                tow_coords = line.tower_coords[0]
                tow_scores = line.tower_scores[0]
                tow_ids = line.tower_ids[0]
                tow_costs = line.tower_costs[0]
                tower_spent_budget = line.tower_spent_budget[0]

            lonelypoints.append(tow_ids[0])

        else:
            geom = LineString(tow_coords)

            # make a new line
            subline = Line(
                my_id=str(uuid4()),
                parent_line=line.my_id,
                starting_tower=tow_ids[0],
                num_towers=len(tow_ids),
                tower_ids=tow_ids,
                tower_coords=tow_coords,
                tower_scores=tow_scores,
                tower_costs=tow_costs,
                tower_spent_budget=tower_spent_budget,
                geometry=geom,
                split_from=None,
                startingpoint=None,
                mean_cost=None,
                mean_score=None,
                mean_direction=None,
            )

            subsublines.append(subline)

    return subsublines, lonelypoints

def compare_direction(a: float, b: float):
    """Basic direction comparison

    Args:
        a (float): Angle one
        b (float): Angle two

    Returns:
        Float: Relative angular difference between a and b
    """
    c = np.abs(a - b)
    if c > 180:
        c -= 360
        c = np.abs(c)

    return c
