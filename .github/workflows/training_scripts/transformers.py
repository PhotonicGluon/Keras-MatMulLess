"""
Runs the transformers training code.

Adapted from https://keras-matmulless.readthedocs.io/en/stable/examples/transformers.html.
"""

# Imports
import keras
import training_setup

import keras_mml

# For our purposes, we will consider only the top 5000 words. This will be our vocabulary size (VOCAB_SIZE).
VOCAB_SIZE = 5000

(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)

# How many sequences did we load?
print(len(x_train), "training sequences")
print(len(x_val), "validation sequences")

# We will limit each sequence to a length of 100 (MAX_LEN). This means that words beyond the MAX_LEN mark will be
# removed, while sequences that are not long enough will be padded to MAX_LEN.
MAX_LEN = 100

x_train = keras.utils.pad_sequences(x_train, maxlen=MAX_LEN)
x_val = keras.utils.pad_sequences(x_val, maxlen=MAX_LEN)

# For this example we elect to choose small hyperparameters.

EMBEDDING_DIM = 16
NUM_HEADS = 2
FFN_DIM = 16

# We will add some dropout in the final fully-connected network to act as regularization and reduce overfitting.
model = keras.models.Sequential(
    layers=[
        keras.layers.Input(shape=(MAX_LEN,)),
        keras_mml.layers.TokenEmbedding(MAX_LEN, VOCAB_SIZE, EMBEDDING_DIM, with_positions=True),
        keras_mml.layers.TransformerBlockMML(EMBEDDING_DIM, FFN_DIM, NUM_HEADS),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(2, activation="softmax"),
    ]
)

model.summary()

# We will train the model to minimize the categorical crossentropy of the model, where we output the accuracy of the
# model as a metric for us to monitor.
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Let's train the model!
model.fit(x_train, y_train, batch_size=16, epochs=3, validation_data=(x_val, y_val))

# How well did the model do?

val_loss, val_acc = model.evaluate(x_val, y_val)
print(f"Validation loss:     {val_loss:.5f}")
print(f"Validation accuracy: {val_acc * 100:.2f}%")

# Check that the accuracy isn't too bad
assert val_loss < 0.5
assert val_acc > 0.8
