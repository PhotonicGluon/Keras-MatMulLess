"""
Runs the gated linear units training code.

Adapted from https://keras-matmulless.readthedocs.io/en/stable/examples/gated-linear-unit.html.
"""

# Imports
import keras
import numpy as np

import keras_mml

np.random.seed(42)  # For reproducibility

# We create a dataset of 10000 entries.
X = np.random.uniform(low=0, size=(10000, 5))

y = np.zeros_like(X)
condition = X[:, 0] + X[:, 2] >= 1
indices = np.where(condition)
y[indices] = X[indices] * 0.5

print(X[:5])
print(y[:5])

# We use a train-test split of 9:1.
train_data = X[:9000], y[:9000]
test_data = X[9000:], y[9000:]

# The model that we will use is the traditional GLU model with the sigmoid activation function.
model = keras.Sequential(
    layers=[keras.layers.Input(shape=(5,)), keras_mml.layers.GLUMML(32, activation="sigmoid"), keras.layers.Dense(5)],
    name="GLU-Model",
)
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


# Now we can train the model. We train it for 100 epochs using a batch size of 256.
NUM_EPOCHS = 100
BATCH_SIZE = 256

model.fit(
    train_data[0],
    train_data[1],
    validation_data=test_data,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0,
    callbacks=[print_training_results_Callback()],
)

# Evaluate
test_mse, test_mae = model.evaluate(test_data[0], test_data[1])
print("Test MSE:", test_mse)
print("Test MAE:", test_mae)

# Check that the validation score isn't that bad
assert test_mse < 0.025
assert test_mae < 0.105
