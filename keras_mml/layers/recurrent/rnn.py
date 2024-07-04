"""
Custom common RNN layers.
"""

from typing import Any, List, Optional

import keras
from keras import ops, random


class DropoutRNNCell:
    """
    Object that holds dropout-related functionality for RNN cells.

    This class is **not** a standalone RNN cell.

    Read docstring of this object at
    https://github.com/keras-team/keras/blob/v3.3.3/keras/src/layers/rnn/dropout_rnn_cell.py.
    """

    def __init__(self):
        """
        Initializes a blank dropout RNN cell.
        """

        self.dropout = 0.0
        self.recurrent_dropout = 0.0

        self.seed_generator = None

        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def get_dropout_mask(self, step_input):
        """
        Gets the dropout mask for the current step's input.
        """

        if self._dropout_mask is None and self.dropout > 0:
            ones = ops.ones_like(step_input)
            self._dropout_mask = random.dropout(ones, rate=self.dropout, seed=self.seed_generator)
        return self._dropout_mask

    def get_recurrent_dropout_mask(self, step_input):
        """
        Gets the recurrent dropout mask for the current step's input.
        """

        if self._recurrent_dropout_mask is None and self.recurrent_dropout > 0:
            ones = ops.ones_like(step_input)
            self._recurrent_dropout_mask = random.dropout(ones, rate=self.recurrent_dropout, seed=self.seed_generator)
        return self._recurrent_dropout_mask

    def reset_dropout_mask(self):
        """
        Reset the cached dropout mask if any.
        """

        self._dropout_mask = None

    def reset_recurrent_dropout_mask(self):
        """
        Reset the cached recurrent dropout mask if any.
        """

        self._recurrent_dropout_mask = None


class RNN(keras.layers.RNN):
    """
    Custom RNN layer that implements custom overrides of common methods.
    """

    def call(self, sequences, initial_state: Optional[List] = None, mask: Optional[Any] = None, training: bool = False):
        """
        Calling method of the layer.

        Args:
            inputs: Inputs into the layer.
            initial_state: List of initial state tensors to be passed to the first call of the cell.
                If not provided, will cause creation of zero-filled initial state tensors.
            mask: Binary tensor indicating whether a given timestep should be masked. An individual
                True entry indicates that the corresponding timestep should be utilized, while a
                False entry indicates that the corresponding timestep should be ignored.
            training: Indicates whether the layer should behave in training mode or in inference
                mode. This argument is passed to the cell when calling it.

        Returns:
            Transformed inputs.
        """

        output = super().call(sequences, initial_state=initial_state, mask=mask, training=training)

        if keras.config.backend() == "jax" and mask is not None:
            # I have no idea why, but when using the Jax backend along with masking, a second copy of the outputs is
            # returned along with the first. The following code just takes the first output and ignores the rest.

            output = output[0]

        return output

    def inner_loop(self, sequences, initial_state, mask, training: bool = False):
        """
        Handles the execution of the recurrent loop of the recurrent neural network.

        Args:
            sequences: Inputs into the layer.
            initial_state: List of initial state tensors to be passed to the first call of the cell.
                If not provided, will cause creation of zero-filled initial state tensors.
            mask: Binary tensor indicating whether a given timestep should be masked. An individual
                True entry indicates that the corresponding timestep should be utilized, while a
                False entry indicates that the corresponding timestep should be ignored.
            training: Indicates whether the layer should behave in training mode or in inference
                mode. This argument is passed to the cell when calling it.

        Returns:
            Transformed inputs.
        """

        if keras.tree.is_nested(initial_state):
            initial_state = initial_state[0]
        if keras.tree.is_nested(mask):
            mask = mask[0]
        return super().inner_loop(sequences, initial_state, mask=mask, training=training)
