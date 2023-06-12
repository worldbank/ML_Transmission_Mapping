from collections import defaultdict

import geopy
import matplotlib.pyplot as plt
import mercantile
import numpy as np
from geopy.distance import distance


def extend_circle(grid, circle_tiles, shrink_factor, direction):
    """
    This function takes a circle, in grid tiles, and then makes it larger

    Args:
        grid: The search kernel
        circle_tiles: The tiles forming the circle
        shrink_factor: The factor with which the weights are reduced
        direction: "in" or "out"

    Returns: A new grid with a larger circle marked
    """
    assert direction in ["in", "out"]
    direction = {"in": +1, "out": -1}[direction]
    gridsize = grid.shape[0]

    finished = False
    while not finished:
        circle_stepped = defaultdict(list)
        for tile_spot in circle_tiles:
            tile_weight = grid[tile_spot[0], tile_spot[1]]
            if tile_spot[0] < gridsize / 2: # If the new tile is left of the center
                next_spot = (tile_spot[0] + direction, tile_spot[1]) # Take a step in the direction
                if next_spot[0] < 0 or next_spot[0] >= gridsize: # If it is outside the grid, skip
                    continue
                if grid[next_spot[0], next_spot[1]] == 0:
                    circle_stepped[next_spot].append(tile_weight) # Make it part of the circle
            if tile_spot[0] > gridsize / 2:
                next_spot = (tile_spot[0] - direction, tile_spot[1])
                if next_spot[0] < 0 or next_spot[0] >= gridsize:
                    continue
                if grid[next_spot[0], next_spot[1]] == 0:
                    circle_stepped[next_spot].append(tile_weight)
            if tile_spot[1] < gridsize / 2:
                next_spot = (tile_spot[0], tile_spot[1] + direction)
                if next_spot[1] < 0 or next_spot[1] >= gridsize:
                    continue
                if grid[next_spot[0], next_spot[1]] == 0:
                    circle_stepped[next_spot].append(tile_weight)
            if tile_spot[1] > gridsize / 2:
                next_spot = (tile_spot[0], tile_spot[1] - direction)
                if next_spot[1] < 0 or next_spot[1] >= gridsize:
                    continue
                if grid[next_spot[0], next_spot[1]] == 0:
                    circle_stepped[next_spot].append(tile_weight)
        circle_stepped = [(spot, max(weights)) for spot, weights in circle_stepped.items()]
        if len(circle_stepped) == 0:
            finished = True
        for tile_spot, tile_weight in circle_stepped:
            grid[tile_spot[0], tile_spot[1]] = tile_weight * shrink_factor
        circle_tiles = [c for c, _ in circle_stepped]
    return grid


def get_angle_weights(direction: int, next_angle_limit: int = 70, min_weight: float = 0.2):
    """
    Create a list of 360 length (one for each direction) with the weight for that direction as value
    :param direction: The direction we should look in (in degrees from up)
    :param next_angle_limit: The max expected angle deviation from the direction
    :param min_weight: The weight will not be lower than this
    :return: The list of a weight per angle 0-360
    """
    direction = int(direction)
    weight_list = [min_weight] * 360
    high_weight_decay = np.sin(np.linspace(0.3, 1, num=next_angle_limit) * (np.pi / 2))
    if direction - next_angle_limit < 0:
        weight_list[:direction] = high_weight_decay[next_angle_limit - direction :]
        weight_list[360 - (next_angle_limit - direction) :] = high_weight_decay[
            : next_angle_limit - direction
        ]
    else:
        weight_list[direction - next_angle_limit : direction] = high_weight_decay

    if direction + next_angle_limit >= 360:
        overshoot = (direction + next_angle_limit) - 360
        weight_list[direction:] = high_weight_decay[overshoot:][::-1]
        weight_list[:overshoot] = high_weight_decay[:overshoot][::-1]
    else:
        weight_list[direction : direction + next_angle_limit] = high_weight_decay[::-1]

    return weight_list


def get_direction_mask(center_lon, center_lat, direction, next_angle_limit=80, gridsize=11):
    """
    Create a mask that has a high probability for tiles in the correct direction, lowering the probability as the
    direction is further off.

    Args:
        center_lon: Center location longitude
        center_lat: Center location latititude
        direction: 0 == north
        next_angle_limit: Max deviation from direction in degrees
        gridsize: Size of the grid to generate

    Returns:
        The direction mask for the local search kernel
    """
    # Normally 0 is west, and the direction goes counterclockwise, but north and clockwise seems better
    direction = (360 - 90 - direction) % 360

    angle_weights = get_angle_weights(direction, next_angle_limit)

    center_tile = mercantile.tile(center_lon, center_lat, 18)
    radius_mask = np.zeros((gridsize, gridsize), dtype=np.float32)
    for angle in range(0, 360, 3):
        angle_weight = angle_weights[angle]
        for dist in range(0, (gridsize // 2 + 1) * 200, 100):
            point = geopy.distance.distance(meters=dist).destination(
                geopy.Point(longitude=center_lon, latitude=center_lat), bearing=angle
            )
            tile = mercantile.tile(lat=point.latitude, lng=point.longitude, zoom=18)
            local_tile_coord = (
                tile.x - center_tile.x + gridsize // 2,
                tile.y - center_tile.y + gridsize // 2,
            )
            if (
                local_tile_coord[0] < 0
                or local_tile_coord[0] >= gridsize
                or local_tile_coord[1] < 0
                or local_tile_coord[1] >= gridsize
            ):
                # This tile is too far away, so the next one will also be too far away
                break
            radius_mask[local_tile_coord[0], local_tile_coord[1]] = max(
                angle_weight, radius_mask[local_tile_coord[0], local_tile_coord[1]]
            )
    return radius_mask


def get_search_kernel_radius(
    center_lon,
    center_lat,
    radius,
    gridsize=11,
    min_weight=0.1,
    max_weight=1.0,
):
    """
    Creates a search kernel mask with a high probability at the expected distance from the centerpoint

    :param center_lon: Center location longitude
    :param center_lat: Center location latititude
    :param radius: The expected distance to the next tower
    :param gridsize: The size of the local search kernel to return
    :param min_weight: The minimum weight of the least likely tile
    :param max_weight: The maximum weight of the most likely tile
    :return: The local search kernel mask for distance as an array
    """
    search_kernel = np.zeros((gridsize, gridsize), dtype=np.float32)
    dists_lng_lat = np.repeat(
        np.stack(
            [
                np.repeat(np.array(list(range(1, gridsize + 1)))[:, np.newaxis], gridsize, 1),
                np.repeat(np.array(list(range(1, gridsize + 1)))[np.newaxis, :], gridsize, 0),
            ],
            axis=2,
        ).astype(np.float32)[np.newaxis, ...],
        5,
        axis=0,
    )

    tile = mercantile.tile(lat=center_lat, lng=center_lon, zoom=18)
    bounds = mercantile.bounds(tile)
    tile_width, tile_height = (abs(bounds.east - bounds.west), abs(bounds.south - bounds.north))
    mid_lng, mid_lat = (bounds.west + (tile_width) / 2, bounds.south + (tile_height) / 2)
    top_left_lng, top_left_lat = (
        mid_lng - (gridsize // 2 + 1) * tile_width,
        mid_lat - (gridsize // 2 + 1) * tile_height,
    )
    dists_lng_lat *= (tile_width, tile_height)
    # This looks a bit magic, but it's all the offsets for the: center, topleft, topright, bottomright, bottomleft
    # per tile:
    dists_lng_lat += [
        [[(top_left_lng, top_left_lat)]],
        [[(top_left_lng - tile_width / 2, top_left_lat + tile_height / 2)]],
        [[(top_left_lng + tile_width / 2, top_left_lat + tile_height / 2)]],
        [[(top_left_lng + tile_width / 2, top_left_lat - tile_height / 2)]],
        [[(top_left_lng - tile_width / 2, top_left_lat - tile_height / 2)]],
    ]

    for r in range(gridsize):
        for c in range(gridsize):
            distance_to_radius = []
            for d in range(len(dists_lng_lat)):
                distance_to_radius.append(
                    abs(
                        distance(
                            (dists_lng_lat[d, r, c, 1], dists_lng_lat[d, r, c, 0]),
                            (center_lat, center_lon),
                        ).meters
                        - radius
                    )
                )
            search_kernel[r, c] = min(distance_to_radius)
    search_kernel = search_kernel**1.2  # Making it flatter
    search_kernel /= np.max(search_kernel)
    return 1 - search_kernel


def get_search_kernel(
    center_lon,
    center_lat,
    radius,
    direction,
    gridsize=11,
    min_weight=0.0,
    max_weight=0.9,
    next_angle_limit=100,
    debug=False,
):
    """
    Get's the local search kernel mask

    :param center_lon: The longitude of the center
    :param center_lat: The latitude of the center
    :param radius: The expected distance to the next tower
    :param direction: The expected direction of the next tower (in degrees, 0 is up)
    :param gridsize: The size of the output grid
    :param min_weight: The minumum weight of the least likely tile
    :param max_weight: The maximum weight of the most likely tile
    :param next_angle_limit: The maximum deviation in angle from the direction
    :param debug: When true, output some intermediate results.
    :return: The local search kernel mask, i.e. the weights based on distance and direction, without looking at the
    costmap
    """
    search_kernel = get_search_kernel_radius(
        center_lon,
        center_lat,
        radius,
        gridsize=gridsize,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    if debug:
        plt.figure()
        plt.title("Search kernel")
        plt.imshow(search_kernel)
    if direction is not None:
        direction_mask = get_direction_mask(
            center_lon, center_lat, direction, next_angle_limit=next_angle_limit, gridsize=gridsize
        )
        search_kernel *= direction_mask
        if debug:
            plt.figure()
            plt.title("Direction mask")
            plt.imshow(direction_mask)
    search_kernel[gridsize // 2, gridsize // 2] = 0

    if debug:
        plt.figure()
        plt.title("Combined kernel")
        plt.imshow(search_kernel)

    return search_kernel


def mercantile_to_WKT(tile: mercantile.Tile):
    bounds = mercantile.bounds(tile)
    WKT = (
        f"POLYGON(("
        f"{bounds.west} {bounds.south},"
        f"{bounds.east} {bounds.south},"
        f"{bounds.east} {bounds.north},"
        f"{bounds.west} {bounds.north},"
        f"{bounds.west} {bounds.south}"
        f"))"
    )
    return WKT
