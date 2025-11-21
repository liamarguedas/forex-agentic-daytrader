from pathlib import Path
from pydantic import BaseModel
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

        if folder_files is None:
            raise FileNotFoundError(f"No files found in {folder_path}.")

        folder_files.sort(key=lambda file: file.stat().st_mtime, reverse=True)

        return folder_files[0]

    def get_lastest_train_data(self):
        last_file = self.last_modified_file(self.training_data_path / "csv")
        return self.load_csv(last_file)

    def get_lastest_production_data(self):
        last_file = self.last_modified_file(self.production_data_path / "csv")
        return self.load_csv(last_file)
