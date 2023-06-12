from __future__ import annotations

import json
from configparser import ConfigParser


from postprocessing.network import TowerNetwork
from utils.sublines import (
    add_geometry_to_sublines,
    subline_scores_and_num_towers,
    write_cost_to_powertower,
    write_numtowers_to_db,
    write_subline_scores_to_db,
)

def run_post(run_id: int, batch_id: str, config_path: str):
    # Set up configurations
    config = ConfigParser()
    config.read(config_path, encoding="utf-8")

    subline_direction_cutoff = config["subline_direction_cutoff"] 
    osm_buffer_dist = config["osm_buffer_dist"] 
    cost_thresh_segment = config["cost_thresh_segment"] 
    score_thresh_segment = config["score_thresh_segment"] 
    angle_cutoff_segments = config["angle_cutoff_segments"] 
    distance_cutoff_segments = config["distance_cutoff_segments"] 
    path_to_osm = config["path_to_osm_lines"]
    costmap_path = config["path_to_costmap"]
    dbconfig = config["db_config"]
    runconfig_path =  config["run_config"]

    with open(runconfig_path, "r", encoding="utf8") as f:
        runconfig = json.load(f)

    tn = TowerNetwork(runconfig_path, dbconfig, run_id, in_post=True)

    # add attributes to DB
    tn.fetch_sublines()
    sb_ids = tn.sublines_gdf["id"].values
    add_geometry_to_sublines(dbconfig, sb_ids)
    scores, ntowers = subline_scores_and_num_towers(dbconfig, sb_ids)
    write_subline_scores_to_db(dbconfig, sb_ids, scores)
    write_numtowers_to_db(dbconfig, sb_ids, ntowers)
    write_cost_to_powertower(dbconfig, run_id, costmap_path)

    # make sublines
    tn.select_distinct_tmp_line_ids(
        multiple_towers=True, with_temp_accepted=False
    )  # multipletowers true and temp accepted false means we exclude sublines with only one tower or less
    tn.make_subsublines(subline_direction_cutoff, batch_id=batch_id)
    tn.fetch_subsublines(batch_id)
    print("Finished making sub-sublines for batch ID", batch_id)

    # Performing tower filtering
    tn.fetch_data()
    tn.accept_OSM_towers(batch_id, path_to_osm, osm_buffer_dist)
    tn.fetch_subsublines(batch_id)
    tn.filter_subsubline_level(
        batch_id=batch_id,
        cost_thresh=cost_thresh_segment,
        score_thresh=score_thresh_segment,
        angle_cutoff=angle_cutoff_segments,
        distance_cutoff=distance_cutoff_segments,
    )

    tn.filter_tower_level(batch_id)
    tn.close_db_connection()
