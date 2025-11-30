from pydantic import BaseModel, PrivateAttr
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataTransformer(BaseModel):

    sequence_length: int = 50
    _scaler: MinMaxScaler = PrivateAttr(default_factory=MinMaxScaler)

    @staticmethod
    def create_predictions_dataframe(prediction: np.ndarray, date):
        next_date = date.max() + pd.offsets.BDay(1)
        prediction_df = pd.DataFrame(
            {"date": [next_date], "predicted_close": prediction.flatten()}
        )
        return prediction_df

    def add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add year, month, day, and day_of_week (as string name) columns to the DataFrame.
        Assumes a 'date' column is present as string or datetime.
        """
        data = data.copy()
        data["year"] = pd.to_datetime(data["date"]).dt.year
        data["month"] = pd.to_datetime(data["date"]).dt.month
        data["day"] = pd.to_datetime(data["date"]).dt.day
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        data["day_of_week"] = pd.to_datetime(data["date"]).dt.dayofweek.map(
            lambda k: days[k]
        )
        data["date"] = pd.to_datetime(data["date"])
        data = data.set_index("date")
        data = data.sort_index()
        return data

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Fit the scaler to close prices and transform them. Returns 2D array (for LSTM/GRU input).
        """
        close_prices = data["close"].to_numpy().reshape(-1, 1)
        return self._scaler.fit_transform(close_prices)

    def create_sequences(self, data: np.ndarray) -> tuple:
        """
        Convert a time series into (X, y) sequences for training recurrent networks.
        Returns:
            - X: np.ndarray with shape (num_samples, sequence_length, 1)
            - y: np.ndarray with shape (num_samples, 1)
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i : i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform production data using the *already fitted* scaler.
        """
        close_prices = data["close"].to_numpy().reshape(-1, 1)
        return self._scaler.transform(close_prices)

    def get_fitted_scaler(self):
        return self._scaler

    def set_fitted_scaler(self, scaler):
        self._scaler = scaler

    def create_latest_sequence(self, data: np.ndarray) -> np.ndarray:
        """
        Create the last sequence for inference.
        Returns shape (1, sequence_length, 1)
        """
        if len(data) < self.sequence_length:
            raise ValueError(
                f"Not enough data to create a sequence. "
                f"Need {self.sequence_length}, got {len(data)}"
            )

        seq = data[-self.sequence_length :]  # last N points
        return seq.reshape(1, self.sequence_length, 1)

    def inverse_transform(self, value: np.ndarray) -> np.ndarray:
        """
        Convert model output back to real price scale.
        """
        return self._scaler.inverse_transform(value)
