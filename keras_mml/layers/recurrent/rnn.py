"""
Custom common RNN layers.
"""

from typing import Any, List, Optional

import keras


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
