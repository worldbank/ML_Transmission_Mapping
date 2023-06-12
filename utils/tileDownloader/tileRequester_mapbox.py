import time
from typing import Dict, Optional

import numpy as np
import requests
from PIL import Image

from utils.tileDownloader.tileRequester import TileRequester


class TileRequester_mapbox(TileRequester):
    """
    Implementation of the abstract TileRequester to get an image tile from the mapbox API
    """

    def __init__(
        self,
        api_key: str,
        storage_config: Dict,
        database_config: Optional[str],
        run_id: Optional[int],
    ):
        """
        :param api_key: The api key for the mapbox API
        :param storage_config: Where to store the data to
        :param database_config: Where to log the data to
        :param run_id: The ID for the current run
        """
        super().__init__(api_key, storage_config, database_config, run_id)
        self.compression_type = ".jpg"  # Could also be .jpg{70/80/90} for higher jpg compression

    @property
    def tile_size(self):
        return 512 if self.higher_dpi else 256

    def _request_tile(self, x, y):
        """
        Actually requests the tile, without checking the cache

        :param x:
        :param y:
        :return: Pillow Image
        """
        request_url = (
            f"https://api.mapbox.com/v4/mapbox.satellite/"
            f"{self.zoom}/"
            f"{x}/{y}"
            f"{['','@2x'][self.higher_dpi]}"
            f"{self.compression_type}"
            f"?access_token={self.api_key}"
        )

        resp = None
        for retry_count in range(5):  # retry up to 5 times
            try:
                resp = requests.get(request_url, stream=True)
                if resp.status_code != 200:
                    print(resp.content)
                    print(resp.__dict__)
                    print("failed request, trying again?")
                else:
                    break
            except Exception as e:
                print(e)
                pass
            if retry_count > 2:
                time.sleep(60)  # Rate limit for mapbox resets after a minute

        if resp is None or resp.status_code != 200:
            return

        return np.array(Image.open(resp.raw))
