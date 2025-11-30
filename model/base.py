from pydantic import BaseModel, PrivateAttr
from pathlib import Path
from datetime import datetime
from config import Pair
from .util import create_string_id, load_latest
from .transformer import DataTransformer
from data import UtilityPipelines
import pandas as pd
import numpy as np
from config import ModelConfig

ROOT = Path(__file__).parents[1]
pair = Pair.load()
model_config = ModelConfig.load()  # load the actual config


class Model(BaseModel):

    _pair: str = PrivateAttr(pair.FROM + pair.TO)
    _rule_id: str = PrivateAttr(
        create_string_id()
    )  # Creating a ID for each prediction made by the model for governance

    @staticmethod
    def fetch_to_predict_data():
        pipeline = UtilityPipelines()
        data = pipeline.get_lastest_data(train=False, fetch_new_data=True)
        return data

    @staticmethod
    def track_prediction(date, id, pair):
        file = ROOT / "model" / "rule" / "governance.csv"
        data = pd.read_csv(file, parse_dates=["date"])
        new_row = pd.DataFrame(
            {"date": [pd.to_datetime(date)], "id": [id], "pair": [pair]}
        )
        data = pd.concat([data, new_row], ignore_index=True)
        data.to_csv(file, index=False)

    def _save_rule_prediction(self, content: pd.DataFrame):
        now = datetime.now()
        file_name = f'{self._rule_id}-{now.strftime("%d-%m-%Y")}-{self._pair}.csv'
        file_path = ROOT / "model" / "rule" / "predict" / file_name
        self.track_prediction(now, self._rule_id, self._pair)
        content.to_csv(file_path)

    def predict_next_month(self, days=30) -> pd.DataFrame:

        data = self.fetch_to_predict_data()

        model = load_latest("model")
        scaler = load_latest("scaler")
        transformer = DataTransformer(sequence_length=model_config.SEQUENCE)
        transformer.set_fitted_scaler(scaler)

        data_with_time_features = transformer.add_time_features(data)

        data_scaled = transformer.transform(data_with_time_features)

        current_sequence = data_scaled[-model_config.SEQUENCE :].copy()

        predictions_scaled, predictions_dates = [], []

        next_date = pd.to_datetime(data["date"]).max() + pd.offsets.BDay(1)

        for _ in range(days):

            X_input = current_sequence.reshape(1, model_config.SEQUENCE, 1)

            next_scaled_value = model.predict(X_input)[0][0]

            predictions_scaled.append(next_scaled_value)
            predictions_dates.append(next_date)

            current_sequence = np.append(current_sequence[1:], next_scaled_value)

            next_date += pd.offsets.BDay(1)

        predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
        final_predictions = transformer.inverse_transform(predictions_scaled)

        prediction_df = transformer.create_predictions_dataframe(
            final_predictions, predictions_dates
        )

        self._save_rule_prediction(prediction_df)
        print(prediction_df)

        return prediction_df
