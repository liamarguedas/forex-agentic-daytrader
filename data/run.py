from base import AlphaVantage
from config import Pair


def main():

    # training_data:
    # True: Will get 20 years of daily data. Intended to retrain the model
    # False: Will get most recent data. Intended to make predicitons.

    pair = Pair.load()  # ← load the actual config
    data_instance = AlphaVantage(
        from_pair=pair.FROM, to_pair=pair.TO, training_data=False
    )
    data_instance.get()


if __name__ == "__main__":
    main()
