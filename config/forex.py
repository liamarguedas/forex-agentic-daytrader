from pydantic import BaseModel
from pathlib import Path
import yaml

PATH = Path(__file__).parents[1] / "config" / "pair.yaml"


class Pair(BaseModel):
    FROM: str
    TO: str

    @classmethod
    def load(cls):
        with open(PATH, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data["PAIR"])
