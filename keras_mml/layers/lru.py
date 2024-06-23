"""
Implements a matmul-less Linear Recurrent Unit (LRU) layer.
"""

from typing import Any, Dict, Optional

import keras


@keras.saving.register_keras_serializable(package="keras_mml")
class LRUCellMML(keras.Layer):
    """
    TODO: ADD
    """


@keras.saving.register_keras_serializable(package="keras_mml")
class LRUMML(keras.layers.RNN):
    """
    TODO: ADD DOCS
    """

    def __init__(self, **kwargs):  # TODO: ADD MORE
        """
        TODO: ADD DOCS
        """

        cell = LRUCellMML(name="lrumml_cell")
        super().__init__(cell, **kwargs)

        self.input_spec = keras.layers.InputSpec(ndim=3)

    # Public methods
    def call(self, sequences, initial_state: Optional[Any] = None, mask: Optional[Any] = None, training: bool = False):
        """
        TODO: ADD DOCS
        """
        return super().call(sequences, initial_state=initial_state, mask=mask, training=training)

    def inner_loop(self, sequences, initial_state, mask, training: bool = False):
        """
        TODO: ADD DOCS
        """

        if keras.tree.is_nested(initial_state):
            initial_state = initial_state[0]
        if keras.tree.is_nested(mask):
            mask = mask[0]
        return super().inner_loop(sequences, initial_state, mask=mask, training=training)

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration for the layer.

        Returns:
            Layer configuration.
        """

        config = {}
        base_config = super().get_config()
        del base_config["cell"]

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LRUMML":
        """
        Creates the layer from the given configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            Created :py:class:`~LRUMML` instance.
        """

        return cls(**config)
