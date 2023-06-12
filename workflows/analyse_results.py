import geopandas as gpd
import pandas.io.sql as sqlio
from sqlalchemy_utils.types.pg_composite import psycopg2
from database.database_utils import get_db_config


def analyse_run(run_id: int, db_config_path: str):
    """Analyse a run of smart-tracing

    Args:
        run_id (int): Run ID in the database
        db_config_path (str): Path to db config file
    """    

    query = (
        """
    SELECT * FROM powertower WHERE run_id = %s
    """
        % run_id
    )

    connection = psycopg2.connect(**get_db_config(db_config_path))
    df = gpd.GeoDataFrame(sqlio.read_sql_query(query, connection))
    print(f"{ df.bought_tiles.mean() =}")
    print(f"{ df.bought_tiles.median() =}")
    print(f"{ df.bought_tiles.std() =}")
    print(f"{ df.bought_tiles.max() =}")
    print(f"{ df.bought_tiles.sum() =}")
    print(f"{ sum(df.bought_tiles == 3) =}")
    tower_count = len(df)
    print(f"{ tower_count =}")

    query = (
        """
        SELECT distinct tile_id FROM predicted WHERE run_id = %s
        """
        % run_id
    )

    df = gpd.GeoDataFrame(sqlio.read_sql_query(query, connection))
    all_tiles = len(df)
    print(f"{ all_tiles=}")
    print(f"{ all_tiles/tower_count =}")

def extract_postprocessed_results(batch_id: str, db_config_path: str, save_to_filepath: str):
    """Extract postprocessed results

    Args:
        batch_id (str): Batch ID of postprocessing
        db_config_path (str): Path to database config file
        save_to_filepath (str): Path to save output geojson
    """    
    query = f"""select t.*, cluster_id from 
                (
                    select powertower.* from powertower INNER JOIN
                    accepted_towers ON ((powertower.tower_uuid =
                    accepted_towers.tower_uuid) and batch_id='{batch_id}') where
                    powertower.tower_uuid in (SELECT tower_uuid from
                    public.accepted_towers where batch_id='{batch_id}') UNION
                    select powertower.* from powertower where tower_uuid in
                    (select tower_uuid from subsublines_towers where
                    accepted=True and batch_id='{batch_id}')
                ) as t LEFT JOIN accepted_towers ON ((t.tower_uuid =
                accepted_towers.tower_uuid) and batch_id='{batch_id}')"""
    connection = psycopg2.connect(**get_db_config(db_config_path))

    df = gpd.GeoDataFrame.from_postgis(query, connection)

    print(f"Postprocessing batch id {batch_id} results in {len(df)} towers")

    df.to_file(save_to_filepath, driver="GeoJSON")

if __name__ == "__main__":
    extract_postprocessed_results(
        "257_extra_normal", r"\\sample\path\database\amazon.ini", r"sample\path\TEST_POSTPROC.geojson"
    )
