from datetime import datetime
from config import Pair
from data import UtilityPipelines
from pathlib import Path
import re
from tensorflow.keras.models import load_model

PATH = Path(__file__).parents[1]
SAVE_PATH = PATH / "model" / "keras"
pair = Pair.load()  # ← load the actual config


def save_model(model):

    now = datetime.now()

    file_name = f"GRU_{pair.FROM}{pair.TO}_SEQUENTIAL_FORECASTER_" + now.strftime(
        "%Y%m%d_%H%M%S"
    )

    model.save(SAVE_PATH / f"{file_name}.keras")


def load_latest_model():

    latest_file = UtilityPipelines.last_modified_file(SAVE_PATH)

    if not latest_file.name.endswith(".keras"):
        raise ValueError(f"Latest file is not a Keras model: {latest_file.name}")

    return load_model(latest_file)
