import json
from pathlib import Path
from typing import Dict

from jsonschema import validate


def load_config(config_path: str, config_schema: Dict):
    train_config_path = Path(config_path)
    with train_config_path.open("r") as f:
        train_config = json.load(f)
    validated_config = validate(train_config, config_schema)
    return validated_config
