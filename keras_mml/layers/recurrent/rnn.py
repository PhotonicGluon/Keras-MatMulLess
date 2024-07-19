"""
Custom common RNN layers.
"""

import keras
import numpy as np
from jaxtyping import Float


class RNN(keras.layers.RNN):
    """
    Custom RNN layer that implements custom overrides of common methods.
    """

    def inner_loop(
        self,
        sequences: Float[np.ndarray, "batch_size timesteps features"],
        initial_state: Float[np.ndarray, "*state_dims"],
        mask: Float[np.ndarray, "*mask_dims"],
        training: bool = False,
    ):
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
