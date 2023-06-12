from typing import List, Tuple

from geopy.distance import distance

from utils.dataclasses.tower import Tower


def filter_towers_closer_than_x(
    towers: List[Tower], min_distance: float = 0.03
) -> Tuple[List[Tower], bool]:
    """
    Takes a list of towers, and returns only the highest confidence tower in case of multiple towers within min_distance
    of each other.

    takes list of towers and min_distance in km

    Returns the filtered towers, and whether filtering has happened
    """
    # This solution does not work with a line when the middle one is encountered last... but that's ok?
    grouped_towers = []
    for new_tower in towers:
        did_group = False
        for towergroup in grouped_towers:
            for tower in towergroup:
                dist = distance(
                    (tower.location.lat, tower.location.lng),
                    (new_tower.location.lat, new_tower.location.lng),
                )
                if dist <= min_distance:
                    towergroup.append(new_tower)
                    did_group = True
                    break
            if did_group:
                break
        else:
            grouped_towers.append([new_tower])

    return_towers = [max(towergroup, key=lambda t: t.confidence) for towergroup in grouped_towers]
    return return_towers, len(return_towers) != len(towers)
