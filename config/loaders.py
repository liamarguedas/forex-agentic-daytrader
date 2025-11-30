from pydantic import BaseModel
from pathlib import Path
import yaml

PATH = Path(__file__).parents[1] / "config" / "file"

PAIR_CONFIG = PATH / "pair.yaml"
MODEL_CONFIG = PATH / "model.yaml"


class Pair(BaseModel):
    FROM: str
    TO: str

    @classmethod
    def load(cls):
        with open(PAIR_CONFIG, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data["PAIR"])


class ModelConfig(BaseModel):

    SEQUENCE: int

    @classmethod
    def load(cls):
        with open(MODEL_CONFIG, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data["MODEL"])
