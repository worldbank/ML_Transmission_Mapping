import dataclasses


@dataclasses.dataclass
class DataParams:
    patchSize: int # The size of the image
    buffer: int = 0 # The size of the buffer to overlap with
    resolution: float = 0.5 # The resolution of the image