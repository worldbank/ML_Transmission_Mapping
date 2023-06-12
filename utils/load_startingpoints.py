import json
import warnings

import psycopg2
from psycopg2.errors import UniqueViolation

from database.database_utils import get_db_config
from utils.dataclasses.point import Point


def load_startingpoints(file_path, id_column, db_config_path, set_name):
    """
    Takes the startingpoints in a geojson and loads them into the database

    :param file_path: The path to the file with startingpoints to load
    :param id_column: The name of the column that should be used as ID
    :param db_config_path: The path to the config defining the database connection
    :param set_name: The set to place the startingpoints in (train or test?)
    """
    connection = psycopg2.connect(**get_db_config(db_config_path))
    with open(file_path, "r", encoding="utf8") as f:
        points = json.load(f)
    for idx, point in enumerate(points["features"]):
        coords = point["geometry"]["coordinates"]
        if len(coords) == 1:
            coords = coords[0]
        insert_query = (
            f"INSERT INTO startingpoints (id_column, object_id, file_path, set_name, geom) "
            f"VALUES (%s, %s, %s, %s, ST_GeomFromText('{Point(lng=coords[0], lat=coords[1]).to_WKT()}',4326))"
        )
        if id_column == "idx":
            obj_id = idx
        else:
            obj_id = point["properties"].get(id_column)
            if obj_id is None:
                warnings.warn(f"This object has no id! {point}")
                continue
        with connection:
            with connection.cursor() as cursor:
                try:
                    cursor.execute(insert_query, (id_column, obj_id, file_path, set_name))
                except UniqueViolation:
                    pass