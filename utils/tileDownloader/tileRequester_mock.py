import random
from pathlib import Path
from typing import Optional

import PIL

from utils.tileDownloader.tileRequester import TileRequester


class TileRequester_mock(TileRequester):
    """
    Implementation of the abstract TileRequester to get an image tile from nowhere
    """
    def __init__(
        self,
        api_key: str,
        output_folder: str,
        database_config: Optional[str],
        run_id: Optional[int],
    ):
        """
         The interface is kept consistent, but the only argument used is the output folder

        :param output_folder: This folder is instead used to get images, this mocked tile requester just returns a
        random image from this folder
        """
        super().__init__(api_key, output_folder, database_config, run_id)
        self.compression_type = ".jpg"  # Could also be .jpg{70/80/90} for higher jpg compression
        self.mock_directory = output_folder
        self.mock_images = list(Path(output_folder).glob(f"*{self.compression_type}"))
        self.mock_mode = True

    @property
    def tile_size(self):
        return 512 if self.higher_dpi else 256

    def _request_tile(self, x, y):
        """
        Randomly samples a tile from the output folder and returns it

        :param x:
        :param y:
        :return: Pillow Image
        """
        image_path = random.sample(self.mock_images, 1)[0]
        image = PIL.Image.open(image_path)
        image = image.resize((self.tile_size, self.tile_size))
        return image
