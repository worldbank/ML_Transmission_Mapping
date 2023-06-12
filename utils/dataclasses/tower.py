from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID, uuid4

from utils.dataclasses.point import Point


@dataclass
class Tower:
    location: Point
    confidence: float
    new_tower: bool
    uuid: UUID = field(default_factory=uuid4)
    new_line_id: Optional[UUID] = None
