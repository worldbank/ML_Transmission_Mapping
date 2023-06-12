# A191 WB electricity network mapping
## Dependencies
It is recommended to use this package in an isolated Python environment running Python 3.9.

GDAL is required - the most robust installation method is to directly download the Python 3.9 wheels here: 
https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal GDAL-3.4.3-cp39-cp39-win_amd64.whl 

and install by running:
`python -m pip install \\path\GDAL-3.4.3-cp39-cp39-win_amd64.whl`

Following the installation of GDAL the rest of the dependencies can be installed by executing the following command in the repository root:

`pip install -r requirements.txt`

finally the repository itself can be installed by running
`pip install .`.

## Requirements
### PostGIS enabled PostgreSQL database
Created with database\setup_db.sql

### Database connection config
Example seen in database\local_db.ini

## Workflow
### Create costmap
Create a costmap for your region of interest by using the workflow seen in workflows\run_costmap.py

An example config can be found in configs\costmap.ini

The final output is stored in 03_final_costs.tif
### Load start points in DB
Before running smart tracing we need to populate the startingpoints table in the database with points to begin tracing from. These points can be e.g. powerplants and substations from OSM datasets. 

These points should be collected in GeoJSON format with a unique ID and geometry (in EPSG: 4326)

An example of inserting these start points can be seen in workflows/populate_startingpoints.py
### Run smart tracing
Smart tracing can be started via the command line inside the initialised Conda environment. 

`python ./tracing/tracer_manager.py --config /path/to/config.ini --log_file /path/to/logfile.out`

An example config file can be seen in configs\test_smartracing_config.json. Parameters that can be varied are:

costmap_path: Path to the costmap generated for the region of interest.
set_name:  The name of the batch of start points loaded in the DB see workflows/populate_startingpoints.py
num_workers: Number of workers per run
run_description: Description of the run to be stored in DB run table
model_weight_path: Path to the weights of the trained DL model
database_config_path: Path to the db config as mentioned above
mock_detection_geojson: Path to a GeoJSON file containing known towers in the region of interest 
storage_config: Configuration for tile downloading + storage
tracing_constants: Smart tracing parameters (see tracing\tracer_real_and_fake_predict.py)
Mapbox API key and secret: Needed for tile downloading

### Post processing
##### Run postprocessing 
Filter out unlikely towers from raw smart tracing results.

First define parameters (can be seen in sample config configs\postprocessing.ini)
run_id: run ID of smart tracing to post process
batch_id:  batch ID to give this post processing run
subline_direction_cutoff: Angular cut-off parameter for straight-line segment generation (degrees)
osm_buffer_dist:  automatically accepting towers within this distance of an OSM tower (m)
cost_thresh_segment: ost threshold for straight-line acceptance
score_thresh_segment: Score threshold for straight-line acceptance
angle_cutoff_segments: Maximum allowed angular difference for neighbouring subline to be accepted (deg)
distance_cutoff_segments: Neighbouring line segment search distance (m)

##### Join lines
Line formation workflow can be used on a shapefile (exported post-processed results from DB, see workflows\analyse_results.py) or directly from the database. This workflow identifies clusters of points and attempts to create sub-network regions.

Example workflow in workflows\run_lineformation.py

##### Run modified gridfinder
A gridfinder approach to generating the network from a set of points can be undertaken with the workflow in postprocessing\gridfinder\grid_from_pts.py and example config in postprocessing\gridfinder\gf_params.ini. 

(Note: The 02_prelim_costs.tif output from the costmap generation step should be used for the costmap input)
