"""
Runs the MNIST classifier training code.

Adapted from https://keras-matmulless.readthedocs.io/en/stable/examples/mnist-classifier.html.
"""

# First, let's prepare the imports.
import keras
import training_setup

import keras_mml

# Define constants relating to the data.
NUM_CLASSES = 10  # 10 distinct classes, 0 to 9
INPUT_SHAPE = (28, 28)  # 28 x 28 greyscale images

# Load the data from the `mnist` dataset.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Perform some preprocessing.
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

# Define the `Sequential` model.
model = keras.Sequential(
    [
        keras.Input(shape=INPUT_SHAPE),
        keras.layers.Flatten(),
        keras_mml.layers.DenseMML(256),
        keras_mml.layers.DenseMML(256),
        keras_mml.layers.DenseMML(256),
        keras.layers.Dense(
            NUM_CLASSES, activation="softmax"
        ),  # The last layer needs to be `Dense` for the output to work
    ],
    name="Classifier-MML",
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# We can now train the model.
model.fit(x_train, y_train, batch_size=128, epochs=20, validation_split=0.1)

# Once the model is trained, let's evaluate it.
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Check that accuracy is at least 90%
assert score[1] >= 0.9
