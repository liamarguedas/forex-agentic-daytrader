from datetime import datetime
from config import Pair
from data import UtilityPipelines
from pathlib import Path
import string, random
import joblib
from tensorflow.keras.models import load_model

PATH = Path(__file__).parents[1]
KERAS_PATH = PATH / "model" / "keras"
SCALER_PATH = PATH / "model" / "scaler"
pair = Pair.load()  # load the actual config


def load_scaler(path):
    scaler = joblib.load(path)
    return scaler


def save_model(model):

    now = datetime.now()

    file_name = f"GRU_{pair.FROM}{pair.TO}_SEQUENTIAL_FORECASTER_" + now.strftime(
        "%Y%m%d_%H%M%S"
    )

    model.save(KERAS_PATH / f"{file_name}.keras")


def save_scaler(scaler):
    now = datetime.now()

    file_name = f"SCALER_{pair.FROM}{pair.TO}_SEQUENTIAL_FORECASTER_" + now.strftime(
        "%Y%m%d_%H%M%S"
    )

    joblib.dump(scaler, KERAS_PATH / f"{file_name}.pkl")


def load_latest(load: str):

    if load == "model":

        latest_file = UtilityPipelines.last_modified_file(KERAS_PATH)

        if not latest_file.name.endswith(".keras"):
            raise ValueError(f"Latest file is not a Keras model: {latest_file.name}")

        return load_model(latest_file)

    if load == "scaler":
        latest_file = UtilityPipelines.last_modified_file(SCALER_PATH)

        if not latest_file.name.endswith(".pkl"):
            raise ValueError(f"Latest file is not a pickle scaler: {latest_file.name}")

        return load_scaler(latest_file)


def create_string_id(size=8):
    return "".join(
        [
            random.choice(string.ascii_lowercase + string.ascii_uppercase)
            for _ in range(size)
        ]
    )
