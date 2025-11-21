from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import json
import requests
import pandas

PATH = Path(__file__).parent


class AlphaVantage(BaseModel):

    from_pair: str
    to_pair: str
    timeline: str = "FX_DAILY"
    training_data: bool = False

    @property
    def outputsize(self):
        return "full" if self.training_data else "compact"

    @property
    def ALPHAVANTAGE_API_KEY(self):
        load_dotenv()
        api = os.getenv("ALPHAVANTAGE_API_KEY")
        if api:
            return api

    @staticmethod
    def export_to_csv(json_file, name):

        Path(name).parent.mkdir(parents=True, exist_ok=True)

        data = pandas.read_json(json_file).T
        data.to_csv(name)

    @staticmethod
    def log_metadata(source_metadata: dict, path: Path):

        path.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        file_name = now.strftime("%d-%m-%Y-%H%M%S") + ".txt"

        with open(path / file_name, "w", encoding="utf-8") as log_file:
            for key in source_metadata.keys():
                log_file.write(f'{key.split(".")[1][1:]}: {source_metadata[key]}\n')

    def retrieve_data(self):
        try:
            function_call = f"function={self.timeline}"
            from_symbol_call = f"&from_symbol={self.from_pair}"
            to_symbol_call = f"&to_symbol={self.to_pair}"
            api_key_call = f"&apikey={self.ALPHAVANTAGE_API_KEY}"
            data_amount = f"&outputsize={self.outputsize}"
            api_call = "https://www.alphavantage.co/query?"
            response = requests.get(
                api_call
                + function_call
                + from_symbol_call
                + to_symbol_call
                + api_key_call
                + data_amount,
                timeout=10,
            )
            return response.json()

        except Exception as e:
            print(response.status_code)
            return None

    def create_data(self):
        data = self.retrieve_data()
        if data:
            now = datetime.now()

            file_name = f"{self.from_pair}{self.to_pair}" + now.strftime(
                "%d-%m-%Y %H-%M-%S"
            )

            save_path = PATH / "training" if self.training_data else PATH / "production"

            json_file = file_name + ".json"
            csv_file = file_name + ".csv"

            self.log_metadata(data["Meta Data"], save_path / "logs")

            (save_path / "json").mkdir(parents=True, exist_ok=True)

            json_path = save_path / "json" / json_file

            with open(json_path, "w", encoding="utf-8") as file:
                json.dump(data["Time Series FX (Daily)"], file)

            self.export_to_csv(json_path, save_path / "csv" / csv_file)

    def get(self):
        self.create_data()
