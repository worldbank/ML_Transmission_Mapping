from grid_costs.grid_cost_cmd import calc_grid_costs

if __name__ == "__main__":
    config_path = r"\\sample\path\configs\costmap.ini"
    aoi_name = 'Liberia' # should be present in config file

    calc_grid_costs(config_path, aoi_name)