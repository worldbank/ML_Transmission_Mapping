from utils.load_startingpoints import load_startingpoints

def populate_startingpoints(path_to_db_config: str, set_name: str, path_to_geojson: str, id_column: str):
    """Populate the database with starting points for smart tracing

    Args:
        path_to_db_config (str): Path to database config file
        set_name (str): Name to assign this group of start points
        path_to_geojson (str): Path to geojson file containing start points (with EPSG: 4326 and a unique identifying column)
        id_column (str): Name of unique identifying column in geojson file
    """    
    startingpoints = [
        (
            path_to_geojson,
            id_column,
        )
    ]

    for file_path, id_column in startingpoints:
        load_startingpoints(file_path, id_column, path_to_db_config, set_name)

if __name__ == "__main__":
   
    populate_startingpoints(path_to_db_config=r"\\sample\path\database\local_db.ini", set_name="LiberiaTest", path_to_geojson=r"\\sample\path\sample_data\smart_tracing\Powerplants_Substations_Liberia.json", id_column="place_id")
