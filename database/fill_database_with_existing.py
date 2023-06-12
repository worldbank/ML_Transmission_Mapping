from functools import partial
from multiprocessing import Pool
from pathlib import Path

import psycopg2
from mercantile import Tile
from psycopg2.errors import UniqueViolation
from tqdm import tqdm

from database.database_utils import get_db_config
from utils.tracer_tools import mercantile_to_WKT


def fill_database_with_existing_tiles(folder, db_config_file=None):
    _connection = psycopg2.connect(**get_db_config(db_config_file))

    folder = Path(folder)
    for img in tqdm(list(folder.glob("*.tif"))):
        x, y = img.stem.split("_")
        x, y = int(x), int(y)
        geomstring = mercantile_to_WKT(Tile(x, y, z=18))
        imgpath = "/".join(["s3:/powergridmapping", *img.parts[2:]])
        query = (
            f"INSERT INTO tile (id, x, y, file_path, geom) "
            f"VALUES (%s, %s, %s, %s, ST_GeomFromText('{geomstring}'))"
            f"ON CONFLICT (id) DO UPDATE"
            f"  SET file_path = %s"
        )
        with _connection:
            with _connection.cursor() as cursor:
                try:
                    cursor.execute(query, (img.stem, x, y, imgpath, imgpath))
                except UniqueViolation:
                    pass


if __name__ == "__main__":
    folders = [
        r"\\path\powertowertiles"
    ]

    Pool(len(folders)).map(
        partial(
            fill_database_with_existing_tiles,
            db_config_file=r"D:\development\electricity-network-mapping\database\amazon.ini",
        ),
        folders,
    )
