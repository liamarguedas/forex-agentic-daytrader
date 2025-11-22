from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from data import UtilityPipelines
from .transformer import DataTransformer
from .util import save_model


def train():

    pipeline = UtilityPipelines()
    data = pipeline.get_lastest_data(train=True, fetch_new_data=False)

    transformer = DataTransformer(sequence_length=50)

    data_with_time_features = transformer.add_time_features(data)

    data_transformed = transformer.fit_transform(data_with_time_features)

    X, y = transformer.create_sequences(data_transformed)

    model = Sequential(
        [
            GRU(
                64,
                return_sequences=False,
                input_shape=(X.shape[1], X.shape[2]),
            ),
            Dense(32, activation="relu"),
            Dense(1),  # Predict the next price
        ]
    )

    model.compile(optimizer="adam", loss="mean_squared_error")

    history = model.fit(
        X, y, epochs=300, batch_size=31, validation_split=0.2, verbose=2
    )

    save_model(model)

    for key, values in history.history.items():
        print(f"{key}: {values[-1]}")


if __name__ == "__main__":
    main()
