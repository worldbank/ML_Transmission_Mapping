from typing import List

import numpy as np
from geopandas import GeoDataFrame
from scipy import spatial
from shapely.geometry import LineString
from sklearn.cluster import DBSCAN


def db_scan_clustering(gdf: GeoDataFrame, max_dist: int, min_samples: int = 2):
    """Label points based on their membership of a spatial cluster using DBSCAN.

    Args:
        gdf (GeoDataFrame): DataFrame containing point geometries in projected co-ordinates
        max_dist (int): Maximum distance between members of a cluster
        min_samples (int, optional): Minimum number of points per cluster. Defaults to 2.

    Returns:
        GeoDataFrame: Input df extended with label column
    """

    x = gdf.geometry.x.values
    y = gdf.geometry.y.values

    dbscan = DBSCAN(eps=max_dist, min_samples=min_samples, algorithm="auto")
    result = dbscan.fit([x for x in zip(x, y)])

    gdf["label"] = result.labels_

    return gdf


def add_nearest_neighbors(df: GeoDataFrame, max_dist: int):
    """This functions adds a column to the input df containing the ordered
        nearest neighbors within a maximum distance belonging to the same group/label.

    Args:
        df (GeoDataFrame): DataFrame containing point geometries and labels
        max_dist (int): Maximum distance for nearest neighbors

    Returns:
        GeoDataFrame: Input dataframe extended with column containing neighboring tower uuids
    """
    # get list of points
    points = df["geometry"].apply(lambda g: [g.x, g.y]).tolist()

    # make a spatial index by using a tree
    kdtree = spatial.KDTree(points)

    # calculates the number of nearest neighbors per point
    num = kdtree.query_ball_point(points, r=max_dist, p=2, return_length=True)

    neighbor_list = []
    for i, numbr in enumerate(num):
        # order the neighbors by distance
        _, neighs = kdtree.query(x=points[i], k=numbr)
        if not type(neighs) == List:
            neighs = [neighs]

        # convert indices to tower_uuids
        ids = [df.iloc[j].tower_uuid for j in neighs]
        array = np.delete(ids, np.where(ids == df.iloc[i].tower_uuid))
        neighbor_list.append(array)

    df["neighbors"] = neighbor_list

    return df


def points_to_lines(df: GeoDataFrame, max_points: int):
    """Join point geometries to make linestrings per point cluster.

    Args:
        df (GeoDataFrame): Dataframe containing points, their neighbors, and tower_uuids
        max_points (int): Maximum vertices/points per linestring

    Returns:
        Dict: Dictionary containing linestring geometries, num_points and labels
    """
    points_skip = []
    line_dict = {}

    num_points = len(df)
    currentgroup = 1

    for _, source_point in df.iterrows():

        if source_point.tower_uuid in points_skip:
            continue

        new_geom = [source_point.geometry]
        search_from_point = source_point
        no_matches = False

        group = source_point.label

        points_skip.append(source_point.tower_uuid)

        for i in range(0, num_points + 1):

            if len(new_geom) >= max_points:
                print("met max points", i)
                break

            if no_matches:
                break

            nearest_neighbors = search_from_point.neighbors

            for neighbor_id in nearest_neighbors:

                if neighbor_id in points_skip:
                    continue

                if not group == df[df["tower_uuid"] == neighbor_id].label.values[0]:
                    continue

                nb_point = df[df["tower_uuid"] == neighbor_id].iloc[0]

                new_geom.append(nb_point.geometry)
                points_skip.append(neighbor_id)
                search_from_point = nb_point
                break
            else:
                no_matches = True

        if len(new_geom) < 2:  # avoid invalid geometries
            continue

        line_geom = LineString(new_geom)
        line_dict[str(currentgroup)] = {
            "geometry": line_geom,
            "num_points": len(new_geom),
            "label": source_point.label,
        }
        currentgroup += 1
    return line_dict
