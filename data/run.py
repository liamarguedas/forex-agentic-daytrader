from base import AlphaVantage


def main():

    # training_data:
    # True: Will get 20 years of daily data. Intended to retrain the model
    # False: Will get most recent data. Intended to make predicitons.

    data_instance = AlphaVantage(from_pair="EUR", to_pair="USD", training_data=False)
    data_instance.get()


if __name__ == "__main__":
    main()
