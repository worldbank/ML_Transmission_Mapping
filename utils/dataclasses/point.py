from dataclasses import dataclass

import mercantile


@dataclass
class Point:
    lng: float
    lat: float

    def __post_init__(self):
        self.tile = mercantile.tile(lng=self.lng, lat=self.lat, zoom=18)
        self.x = self.tile.x
        self.y = self.tile.y

    def to_WKT(self):
        return f"POINT ({self.lng} {self.lat})"

    def __repr__(self):
        return self.to_WKT()

    def to_dict(self):
        return {"lng": self.lng, "lat": self.lat}

    @classmethod
    def from_dict(cls, dict):
        return cls(dict["lng"], dict["lat"])
