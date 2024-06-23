"""
Implements a matmul-less Linear Recurrent Unit (LRU) layer.
"""

from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
from keras import ops


@keras.saving.register_keras_serializable(package="keras_mml")
class _GammaLogInitializer(keras.initializers.Initializer):
    def __init__(self, nu_log: np.ndarray):
        self.nu_log = nu_log

    def __call__(self, **kwargs):
        lambda_mod = ops.exp(ops.exp(self.nu_log))
        gamma_log = ops.log(ops.sqrt(ops.ones_like(lambda_mod) - ops.square(lambda_mod)))
        return gamma_log

    def get_config(self):
        return {"nu_log": self.nu_log}


@keras.saving.register_keras_serializable(package="keras_mml")
class LRUCellMML(keras.Layer):
    """
    TODO: ADD

    References:
    - https://arxiv.org/pdf/2303.06349
    - https://github.com/Gothos/LRU-pytorch/blob/main/LRU_pytorch/LRU.py
    - https://github.com/sustcsonglin/flash-linear-rnn/blob/master/linear_rnn/layers/lru.py
    """

    def __init__(
        self, units: int, state_size: int, r_min: float = 0, r_max: float = 1, max_phase: float = 6.283, **kwargs
    ):
        """
        TODO: ADD
        """

        if units <= 0:
            raise ValueError(f"Invalid number of units. Expected a positive integer, got {units}.")
        if state_size <= 0:
            raise ValueError(f"Invalid state size. Expected a positive integer, got {state_size}.")

        super().__init__(**kwargs)

        self.input_spec = keras.layers.InputSpec(ndim=2)

        self.units = units
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase

        self.state_size = state_size
        self.output_size = units

    # Helper methods
    def _nu_log_init(self, shape: Tuple[int, ...], dtype: str = None) -> np.ndarray:
        """
        Initializer for the ``nu_log`` weight.

        Args:
            shape: Shape of the weight.
            dtype: Data type of the weight. Defaults to None.

        Returns:
            Weight values.
        """

        uniform = keras.random.uniform(shape, dtype=dtype)
        nu_log = ops.log(-0.5 * ops.log(uniform * (self.r_max**2 - self.r_min**2) + self.r_min**2))
        return nu_log

    def _theta_log_init(self, shape: Tuple[int, ...], dtype: str = None) -> np.ndarray:
        """
        Initializer for the ``theta_log`` weight.

        Args:
            shape: Shape of the weight.
            dtype: Data type of the weight. Defaults to None.

        Returns:
            Weight values.
        """

        uniform = keras.random.uniform(shape, dtype=dtype)
        theta_log = ops.log(self.max_phase * uniform)
        return theta_log

    def _B_init(self, shape: Tuple[int, int], dtype: str = None) -> np.ndarray:
        """
        Initializer for the $B$ matrix weights.

        Args:
            shape: Shape of the weight.
            dtype: Data type of the weight. Defaults to None.

        Returns:
            Weight values.
        """

        _, units = shape
        return keras.random.normal(shape, dtype=dtype) / ops.sqrt(2 * units)

    def _C_init(self, shape: Tuple[int, int], dtype: str = None) -> np.ndarray:
        """
        Initializer for the $C$ matrix weights.

        Args:
            shape: Shape of the weight.
            dtype: Data type of the weight. Defaults to None.

        Returns:
            Weight values.
        """

        state_size, _ = shape
        return keras.random.normal(shape, dtype=dtype) / ops.sqrt(state_size)

    # Public methods
    def build(self, input_shape: Tuple[int, int]):
        """
        TODO: Add
        """

        super().build(input_shape)

        # Initialization of Lambda is complex valued distributed uniformly on a ring between r_min and r_max, with the
        # phase in the interval $[0, max_phase]$ (See lemma 3.2 for initialization details)
        self.nu_log = self.add_weight(name="nu_log", shape=(self.state_size,), initializer=self._nu_log_init)
        self.theta_log = self.add_weight(name="theta_log", shape=(self.state_size,), initializer=self._theta_log_init)

        # Glorot initialized Input/Output projection matrices
        self.B_re = self.add_weight(name="B_re", shape=(self.state_size, self.units), initializer=self._B_init)
        self.B_im = self.add_weight(name="B_im", shape=(self.state_size, self.units), initializer=self._B_init)
        self.C_re = self.add_weight(name="C_re", shape=(self.state_size, self.units), initializer=self._C_init)
        self.C_im = self.add_weight(name="C_im", shape=(self.state_size, self.units), initializer=self._C_init)
        self.D = self.add_weight(name="D", shape=(self.units,), initializer="glorot_normal")

        # Normalization factor
        self.gamma_log = self.add_weight(
            name="gamma_log", shape=(self.state_size), initializer=_GammaLogInitializer(self.nu_log)
        )

    def get_initial_state(self, batch_size: Optional[int] = None) -> List[np.ndarray]:
        """
        Gets the initial states.

        Args:
            batch_size: Batch size for the cell. Defaults to None.

        Returns:
            Initial states.
        """

        # TODO: Check if we need to make this work with complex
        return [ops.zeros((batch_size, self.state_size), dtype=self.compute_dtype)]


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
