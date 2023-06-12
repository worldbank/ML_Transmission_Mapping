from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from utils.dataclasses.point import Point


@dataclass
class TracerTask:
    """
    Dataclass defining a task for a single tracer
    """

    startpoint: Point # The point from which to start tracing (a tower or a substation/powerplant)
    direction: Optional[float] # The direction in which to start looking, None to look all around
    radius: float # The radius within which to search (the distance between towers)
    line_id: Optional[UUID] # The line id to log the towers to
    budget: float = 25.0 # The budget for this tracer, the higher this value the more tiles it will look at
    next_angle_limit: int = 80 # A limit to how far the next tower is expected to deviate from the direction
    last_tower_uuid: Optional[UUID] = None # The last tower in the line this new line sprouted from
    starting_point_id: Optional[int] = None # The id of the first tower of this new line

    def to_dict(self):
        """
        Converts the dataclass to a dict for serialisation purposes
        :return: The dict with all the properties
        """
        return {
            "startpoint": self.startpoint.to_dict(),
            "direction": self.direction,
            "radius": self.radius,
            "budget": self.budget,
            "next_angle_limit": self.next_angle_limit,
            "line_id": self.line_id,
            "last_tower_uuid": self.last_tower_uuid,
            "starting_point_id": self.starting_point_id,
        }

    @classmethod
    def from_dict(cls, dict):
        """
        re-creates the dataclass from a dict.

        :param dict: The dict to parse.
        :return: An instance of this dataclass with the data from the dict
        """
        return cls(
            startpoint=Point.from_dict(dict["startpoint"]),
            direction=dict["direction"],
            radius=dict["radius"],
            budget=dict["budget"],
            next_angle_limit=dict["next_angle_limit"],
            line_id=dict["line_id"],
            last_tower_uuid=dict["last_tower_uuid"],
            starting_point_id=dict["starting_point_id"],
        )
