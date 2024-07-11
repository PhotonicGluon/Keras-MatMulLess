"""
Runs the time-series forecasting training code.

Adapted from https://keras-matmulless.readthedocs.io/en/stable/examples/time-series-forecasting.html.
"""

# Imports
import keras
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import keras_mml

# Download the dataset
downloaded = keras.utils.get_file(
    fname="airline-passengers.csv",
    origin="https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
)

df = pd.read_csv(downloaded, usecols=["Passengers"])

# Ensure that all the values in the dataset are expressed as floats (and not integers).
dataset = df.values
dataset = dataset.astype("float32")

# Normalize the dataset to be in the interval [0, 1].
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# We will use 80% of the data for training and the other 20% for testing.
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size : len(dataset), :]

print(len(train), len(test))

# We want to provide the model some past data (specifically, LOOK_BACK days worth of data) and then ask it to predict
# the following day's passenger count.
LOOK_BACK = 3


def create_dataset(dataset, look_back=1):
    x, y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i : i + look_back, 0]
        x.append(a)
        y.append(dataset[i + look_back, 0])
    return np.array(x), np.array(y)


train_X, train_Y = create_dataset(train, LOOK_BACK)
test_X, test_Y = create_dataset(test, LOOK_BACK)

# Keras RNNs require the input to be of the shape (batch, timesteps, features), so we will reshape the input to match
# this. Since we are just using the passenger counts, we only have one feature to include.
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

# Gated Recurrent Unit (GRU) Model
model = keras.Sequential()
model.add(keras.Input((LOOK_BACK, 1)))
model.add(keras_mml.layers.GRUMML(4))
model.add(keras.layers.Dense(1))

model.compile(loss="mse", optimizer="adam", metrics=["mae"])


# We define a callback to print the training output once every 10 epochs. This is to reduce clutter on the screen.
class print_training_results_Callback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if int(epoch) % 10 == 0:
            print(
                f"Epoch: {epoch:>3}"
                + f" | loss: {logs['loss']:.5f}"
                + f" | mae: {logs['mae']:.5f}"
                + f" | val_loss: {logs['val_loss']:.5f}"
                + f" | val_mae: {logs['val_mae']:.5f}"
            )


# With all that setup, we can train the model!
model.fit(
    train_X,
    train_Y,
    epochs=200,
    batch_size=1,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, min_delta=1e-4, verbose=1),
        print_training_results_Callback(),
    ],
    verbose=0,
)

# Make predictions
train_pred = model.predict(train_X)
test_pred = model.predict(test_X)

# Un-normalize the predictions
train_pred_orig = scaler.inverse_transform(train_pred)
train_Y_orig = scaler.inverse_transform([train_Y])
test_pred_orig = scaler.inverse_transform(test_pred)
test_y_orig = scaler.inverse_transform([test_Y])

# Then evaluate the MSE
train_score = np.sqrt(mean_squared_error(train_Y_orig[0], train_pred_orig[:, 0]))
print(f"Train MSE: {train_score:.2f}")
test_score = np.sqrt(mean_squared_error(test_y_orig[0], test_pred_orig[:, 0]))
print(f"Test MSE:  {test_score:.2f}")

# Assert that the test score is not too bad
assert test_score < 75

# Linear Recurrent Unit (LRU) Model
model = keras.Sequential()
model.add(keras.Input((LOOK_BACK, 1)))
model.add(keras_mml.layers.LRUMML(4, 8))
model.add(keras.layers.Dense(1))

model.compile(loss="mse", optimizer="adam", metrics=["mae"])

model.fit(
    train_X,
    train_Y,
    epochs=200,
    batch_size=1,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, min_delta=1e-4, verbose=1),
        print_training_results_Callback(),
    ],
    verbose=0,
)

# Make predictions
train_pred = model.predict(train_X)
test_pred = model.predict(test_X)

# Un-normalize the predictions
train_pred_orig = scaler.inverse_transform(train_pred)
train_Y_orig = scaler.inverse_transform([train_Y])
test_pred_orig = scaler.inverse_transform(test_pred)
test_y_orig = scaler.inverse_transform([test_Y])

# Then evaluate the MSE
train_score = np.sqrt(mean_squared_error(train_Y_orig[0], train_pred_orig[:, 0]))
print(f"Train MSE: {train_score:.2f}")
test_score = np.sqrt(mean_squared_error(test_y_orig[0], test_pred_orig[:, 0]))
print(f"Test MSE:  {test_score:.2f}")

# Assert that the test score is not too bad
assert test_score < 75
