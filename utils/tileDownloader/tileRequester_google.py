import base64
import hashlib
import hmac
import urllib.parse as urlparse
from typing import Optional, Tuple

import mercantile
import numpy as np
import PIL.Image
import requests
from PIL import Image

from utils.tileDownloader.tileRequester import TileRequester


class TileRequester_google(TileRequester):
    """
    Implementation of the abstract TileRequester to get an image tile from the google maps API
    """

    def __init__(
        self,
        api_key: Optional[str],
        storage_config: dict,
        database_config: Optional[str],
        signing_secret: Optional[str],
        run_id: Optional[int],
    ):
        """
        :param api_key: The API key for google
        :param storage_config: Where to store the images to
        :param database_config: Definition of the database
        :param signing_secret: With large quantities; you need a signing secret
        :param run_id: The id for this run
        """
        super().__init__(api_key, storage_config, database_config, run_id)
        self.compression_type = "png32"
        self.signing_secret = signing_secret

    @property
    def tile_size(self):
        return 512

    def get_center_point_from_slippy_tile(self, x, y) -> Tuple[float, float]:
        """
        Google's imagery is not accessed per tile, but rather as a width/height around a centerpoint

        :param x: The x of the tile
        :param y: The y of the tile
        :return: The longitude, latitude of the center of the tile
        """
        tile = mercantile.Tile(x, y, self.zoom)
        lnglatbox = mercantile.bounds(tile)
        lngmid = lnglatbox.west + (lnglatbox.east - lnglatbox.west) / 2
        latmid = lnglatbox.north + (lnglatbox.south - lnglatbox.north) / 2

        # We request a tile that is 50 pixels larger, it will get 25 more pixels above, and 25 below
        # In order to only request redundant pixels towards the bottom, we shift the center 25 pixels down?
        latmid += (lnglatbox.south - lnglatbox.north) / 512 * 25

        return lngmid, latmid

    def _request_tile(self, x, y, clip_watermark=True):
        """
        Actually requests the tile, without checking the cache

        :param x:
        :param y:
        :return: Pillow Image
        """

        # Re-calculate the centerpoint from the x,y tile coordinates
        lng, lat = self.get_center_point_from_slippy_tile(x, y)

        tile_size_scaled = int(self.tile_size / ([1, 2][self.higher_dpi]))

        request_url = [
            "https://maps.googleapis.com/maps/api/staticmap?maptype=satellite",
            f"format={self.compression_type}",
            f"zoom={self.zoom}",
            f"center={lat},{lng}",
            f"scale={[1,2][self.higher_dpi]}",
            f"size={tile_size_scaled}x{tile_size_scaled+25}",
            f"key={self.api_key}",
        ]
        request_url = "&".join(request_url)

        for retry_count in range(5):  # retry up to 5 times
            resp = requests.get(self.sign_url(request_url, self.signing_secret), stream=True)
            if resp.status_code != 200:
                print(resp.content)
                print(resp.__dict__)
                print("failed request, trying again?")
            else:
                break
        if resp.status_code != 200:
            return
        img = Image.open(resp.raw)
        if clip_watermark:
            image_array = np.array(img)
            h = image_array.shape[0]
            top_image = image_array[: h - 50, ...]
            return PIL.Image.fromarray(top_image)
        return img

    @staticmethod
    def sign_url(input_url=None, secret=None):
        """
        Sign a request URL with a URL signing secret.
        https://developers.google.com/maps/documentation/maps-static/digital-signature

        :param input_url: The url to sign
        :param secret: The secret ot sign it with
        :return: The signed url
        """

        if not input_url or not secret:
            raise Exception("Both input_url and secret are required")

        url = urlparse.urlparse(input_url)

        # We only need to sign the path+query part of the string
        url_to_sign = url.path + "?" + url.query

        # Decode the private key into its binary format
        # We need to decode the URL-encoded private key
        decoded_key = base64.urlsafe_b64decode(secret)

        # Create a signature using the private key and the URL-encoded
        # string using HMAC SHA1. This signature will be binary.
        signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)

        # Encode the binary signature into base64 for use within a URL
        encoded_signature = base64.urlsafe_b64encode(signature.digest())

        original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

        # Return signed URL
        return original_url + "&signature=" + encoded_signature.decode()
