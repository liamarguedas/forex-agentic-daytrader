from pathlib import Path
from pydantic import BaseModel
from .base import AlphaVantage
from ..config import Pair
import pandas

PATH = Path(__file__).parents[1]


class UtilityPipelines(BaseModel):

    @property
    def training_data_path(self):
        return PATH / "data" / "training"

    @property
    def production_data_path(self):
        return PATH / "data" / "production"

    @staticmethod
    def fetch_new_data():
        data_instance = AlphaVantage(
            from_pair=Pair.FROM, to_pair=Pair.TO, training_data=False
        )
        data_instance.get()

    @staticmethod
    def load_csv(csv) -> pandas.DataFrame:

        try:
            temp = pandas.read_csv(csv)
            temp.rename(
                columns={
                    "Unnamed: 0": "date",
                    "1. open": "open",
                    "2. high": "high",
                    "3. low": "low",
                    "4. close": "close",
                },
                inplace=True,
            )

            print(f"Data range: {temp.iloc[-1]["date"]} to {temp.iloc[0]["date"]}")

        except Exception as e:
            print(e)

        return temp

    @staticmethod
    def last_modified_file(folder_path: Path):

        folder_files = list(folder_path.iterdir())

        if not folder_files:
            raise FileNotFoundError(f"No files found in {folder_path}.")

        folder_files.sort(key=lambda file: file.stat().st_mtime, reverse=True)

        return folder_files[0]

    def get_lastest_data(self, train=False, fetch_new_data=False):
        if fetch_new_data:
            self.fetch_new_data()

        path = self.training_data_path if train else self.production_data_path
        last_file = self.last_modified_file(path / "csv")
        return self.load_csv(last_file)
