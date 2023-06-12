from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from tracing.tracer import Tracer
from utils.dataclasses.point import Point


@dataclass
class Line:
    """Dataclass that holds a line object - can represent sublines or subsublines."""

    my_id: str
    parent_line: str
    split_from: str
    starting_tower: str
    startingpoint: int
    num_towers: int
    tower_ids: List[str]
    tower_coords: List[tuple]
    tower_scores: List[float]
    tower_costs: List[float]
    tower_spent_budget: List[float]
    geometry: str
    mean_score: float
    mean_cost: float
    mean_direction: float

    def __post_init__(self):
        """Setting some variables after initialisation"""
        self.mean_score: float = self.calc_mean_score()
        self.mean_cost: float = self.calc_mean_cost()
        self.mean_direction: float = self.calc_subline_directionality()

    def calc_mean_score(self):
        """Get the mean score based on towers in the line.

        Returns:
            float: mean score
        """
        try:
            return np.mean(self.tower_scores)
        except Exception as e:
            print("problem calculating subsubline mean score", e)
            return 0

    def calc_mean_cost(self):
        """Get the mean cost based on towers in the line.

        Returns:
            float: mean cost
        """
        try:
            return np.mean(self.tower_costs)
        except Exception as e:
            print("problem calculating subsubline mean cost", e)
            return 1

    def calc_subline_directionality(self):
        """Get the mean direction based on towers in the line.

        Returns:
            float: mean direction
        """
        try:
            directions = []
            for i in range(0, len(self.tower_coords) - 1):
                direction = Tracer.get_direction_between_points(
                    Point(self.tower_coords[i][0], self.tower_coords[i][1]),
                    Point(self.tower_coords[i + 1][0], self.tower_coords[i + 1][1]),
                )

                directions.append(direction)

            mean = np.mean(directions)
            return mean

        except Exception as e:
            print("problem calculating subsubline mean direction", e)
            return 0
