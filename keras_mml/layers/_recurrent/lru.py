"""
Implements an almost matmul-less Linear Recurrent Unit (LRU) layer.
"""

from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
from keras import ops

from keras_mml.layers.core import DenseMML


@keras.saving.register_keras_serializable(package="keras_mml")
class _GammaLogInitializer(keras.initializers.Initializer):
    def __init__(self, nu_log: np.ndarray):
        self.nu_log = nu_log

    def __call__(self, *args, **kwargs):
        lambda_mod = ops.exp(-ops.exp(self.nu_log))
        gamma_log = ops.log(ops.sqrt(ops.ones_like(lambda_mod) - ops.square(lambda_mod)))
        return gamma_log


@keras.saving.register_keras_serializable(package="keras_mml")
class LRUCellMML(keras.Layer):
    """
    Cell class for the :py:class:`~LRUMML` layer.

    This class processes one step within the whole time sequence input, whereas :py:class:`~LRUMML`
    processes the whole sequence.

    Attributes:
        units: Dimensionality of the output space.
        state_dim: Dimensionality of the internal state space.
        fully_mml: Whether to use matmul-free operations for all the layers.
        r_min: Minimum modulus of the complex weights in :math:`\\mathbf{\\Lambda}`.
        r_max: Maximum modulus of the complex weights in :math:`\\mathbf{\\Lambda}`.
        max_phase: Maximum phase of the complex weights in :math:`\\mathbf{\\Lambda}`.
    """

    def __init__(
        self,
        units: int,
        state_dim: int,
        fully_mml: bool = False,
        r_min: float = 0,
        r_max: float = 1,
        max_phase: float = 6.283,
        **kwargs,
    ):
        """
        Initializes a new instance of the layer.

        Args:
            units: Dimensionality of the output space.
            state_dim: Dimensionality of the internal state space.
            fully_mml: Whether to use matmul-free operations for all the layers.
            r_min: Minimum modulus of the complex weights in :math:`\\mathbf{\\Lambda}`.
            r_max: Maximum modulus of the complex weights in :math:`\\mathbf{\\Lambda}`.
            max_phase: Maximum phase of the complex weights in :math:`\\mathbf{\\Lambda}`.
            **kwargs: Keyword arguments for :py:class:`keras.Layer`.

        Raises:
            ValueError: If the units provided is not a positive integer.
            ValueError: If the state dimensionality provided is not a positive integer.
        """

        if units <= 0:
            raise ValueError(f"Invalid number of units. Expected a positive integer, got {units}.")
        if state_dim <= 0:
            raise ValueError(f"Invalid state dimensionality. Expected a positive integer, got {state_dim}.")

        super().__init__(**kwargs)

        self.input_spec = keras.layers.InputSpec(ndim=2)

        self.units = units
        self.state_dim = state_dim
        self.fully_mml = fully_mml

        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase

        self.state_size = state_dim
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

        input_dim, _ = shape
        values = keras.random.normal(shape, dtype=dtype) / ops.sqrt(2 * input_dim)
        return values

    def _C_init(self, shape: Tuple[int, int], dtype: str = None) -> np.ndarray:
        """
        Initializer for the $C$ matrix weights.

        Args:
            shape: Shape of the weight.
            dtype: Data type of the weight. Defaults to None.

        Returns:
            Weight values.
        """

        state_dim, _ = shape
        values = keras.random.normal(shape, dtype=dtype) / ops.sqrt(state_dim)
        return values

    # Public methods
    def build(self, input_shape: Tuple[int, int]):
        """
        Create layer weights.

        Args:
            input_shape: Shape of the input.
        """

        super().build(input_shape)

        # Decide what layer class to use for the output-adjacent layers
        if self.fully_mml:
            output_layer_class = DenseMML
        else:
            output_layer_class = keras.layers.Dense

        # Initialization of Lambda is complex valued distributed uniformly on a ring between r_min and r_max, with the
        # phase in the interval $[0, max_phase]$ (See lemma 3.2 for initialization details)
        self.nu_log = self.add_weight(name="nu_log", shape=(self.state_dim,), initializer=self._nu_log_init)
        self.theta_log = self.add_weight(name="theta_log", shape=(self.state_dim,), initializer=self._theta_log_init)

        # Glorot initialized Input/Output projection matrices
        self.B_re = DenseMML(self.state_dim, kernel_initializer=self._B_init, name="B_re")
        self.B_im = DenseMML(self.state_dim, kernel_initializer=self._B_init, name="B_im")
        self.C_re = output_layer_class(self.units, kernel_initializer=self._C_init, name="C_re")
        self.C_im = output_layer_class(self.units, kernel_initializer=self._C_init, name="C_im")
        self.D = output_layer_class(self.units, kernel_initializer="glorot_normal", name="D")

        self.B_re.build(input_shape)
        self.B_im.build(input_shape)
        self.C_re.build((None, self.state_dim))
        self.C_im.build((None, self.state_dim))
        self.D.build(input_shape)

        # Normalization factor
        self.gamma_log = self.add_weight(
            name="gamma_log", shape=(self.state_dim,), initializer=_GammaLogInitializer(self.nu_log)
        )

    def call(self, inputs, states, training=False):
        """
        Calling method of the cell.

        Args:
            inputs: Inputs into the layer. Has shape ``(batch, features)``.
            states: State(s) from the previous timestep. Has shape ``(batch, units)``.
            training: Whether the layer should behave in training mode or in inference mode.

        Returns:
            Transformed inputs.
        """

        # Get previous state
        state = states[0] if keras.tree.is_nested(states) else states
        state_re, state_im = ops.split(state, 2, axis=1)
        state_re = ops.squeeze(state_re, axis=1)
        state_im = ops.squeeze(state_im, axis=1)

        # Compute real and imaginary parts of the diagonal lambda matrix
        lambda_mod = ops.exp(-ops.exp(self.nu_log))
        lambda_re = lambda_mod * ops.cos(ops.exp(self.theta_log))
        lambda_im = lambda_mod * ops.sin(ops.exp(self.theta_log))

        # Get the normalization factor, gamma
        gamma = ops.exp(self.gamma_log)

        # Compute new state
        new_state_re = lambda_re * state_re - lambda_im * state_im + gamma * self.B_re(inputs)
        new_state_im = lambda_re * state_im + lambda_im * state_re + gamma * self.B_im(inputs)

        new_state = ops.stack([new_state_re, new_state_im], axis=1)
        new_state = [new_state] if keras.tree.is_nested(states) else new_state

        # Compute output
        output = self.C_re(new_state_re) - self.C_im(new_state_im) + self.D(inputs)
        return output, new_state

    def get_config(self):
        """
        Gets the configuration for the layer.

        Returns:
            Layer configuration.
        """

        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "state_dim": self.state_dim,
                "fully_mml": self.fully_mml,
                "r_min": self.r_min,
                "r_max": self.r_max,
                "max_phase": self.max_phase,
            }
        )
        return config

    def get_initial_state(self, batch_size: Optional[int] = None) -> List[np.ndarray]:
        """
        Gets the initial states.

        Args:
            batch_size: Batch size for the cell.

        Returns:
            Initial states.
        """

        return [ops.zeros((batch_size, 2, self.state_size), dtype=self.compute_dtype)]


@keras.saving.register_keras_serializable(package="keras_mml")
class LRUMML(keras.layers.RNN):
    """
    Linear Recurrent Unit (LRU) layer, mostly without matrix multiplications.

    The core algorithm of this layer mostly follows the implementation in |LinearRU|_ (see Appendix
    A), with a few modifications from |LRU-PyTorch|_. We replace some matrix multiplications with
    ternary weights, although the option to use it as fully matrix multiplication free is available
    using the :py:attr:`~fully_mml` attribute.

    Attributes:
        units: Dimensionality of the output space.
        state_dim: Dimensionality of the internal state space.
        fully_mml: Whether to use matmul-free operations for all the layers.
        r_min: Minimum modulus of the complex weights in :math:`\\mathbf{\\Lambda}`.
        r_max: Maximum modulus of the complex weights in :math:`\\mathbf{\\Lambda}`.
        max_phase: Maximum phase of the complex weights in :math:`\\mathbf{\\Lambda}`.

    .. |LinearRU| replace:: *Resurrecting Recurrent Neural Networks for Long Sequences*
    .. _LinearRU: https://arxiv.org/pdf/2303.06349v1
    .. |LRU-PyTorch| replace:: the ``LRU-pytorch`` GitHub repository
    .. _LRU-PyTorch: https://github.com/Gothos/LRU-pytorch/blob/e250d5a5/LRU_pytorch/LRU.py
    """

    def __init__(
        self,
        units: int,
        state_dim: int,
        fully_mml: bool = False,
        r_min: float = 0,
        r_max: float = 1,
        max_phase: float = 6.283,
        **kwargs,
    ):
        """
        Initializes a new instance of the layer.

        Args:
            units: Dimensionality of the output space.
            state_dim: Dimensionality of the internal state space.
            fully_mml: Whether to use matmul-free operations for all the layers.
            r_min: Minimum modulus of the complex weights in :math:`\\mathbf{\\Lambda}`.
            r_max: Maximum modulus of the complex weights in :math:`\\mathbf{\\Lambda}`.
            max_phase: Maximum phase of the complex weights in :math:`\\mathbf{\\Lambda}`.
            **kwargs: Keyword arguments for :py:class:`keras.Layer`.

        Raises:
            ValueError: If the units provided is not a positive integer.
            ValueError: If the state dimensionality provided is not a positive integer.
        """

        cell = LRUCellMML(
            name="lrumml_cell",
            units=units,
            state_dim=state_dim,
            fully_mml=fully_mml,
            r_min=r_min,
            r_max=r_max,
            max_phase=max_phase,
        )
        super().__init__(cell, **kwargs)

        self.input_spec = keras.layers.InputSpec(ndim=3)

    # Properties
    @property
    def units(self):
        """
        :meta private:
        """
        return self.cell.units

    @property
    def state_dim(self):
        """
        :meta private:
        """
        return self.cell.state_dim

    @property
    def fully_mml(self):
        """
        :meta private:
        """
        return self.cell.fully_mml

    @property
    def r_min(self):
        """
        :meta private:
        """
        return self.cell.r_min

    @property
    def r_max(self):
        """
        :meta private:
        """
        return self.cell.r_max

    @property
    def max_phase(self):
        """
        :meta private:
        """
        return self.cell.max_phase

    # Public methods
    def call(self, sequences, initial_state: Optional[List] = None, mask: Optional[Any] = None, training: bool = False):
        """
        Calling method of the layer.

        Args:
            inputs: Inputs into the layer, with shape ``(batch, timesteps, features)``.
            initial_state: List of initial state tensors to be passed to the first call of the cell.
                If not provided, will cause creation of zero-filled initial state tensors.
            mask: Binary tensor of shape ``(samples, timesteps)`` indicating whether a given
                timestep should be masked. An individual True entry indicates that the corresponding
                timestep should be utilized, while a False entry indicates that the corresponding
                timestep should be ignored.
            training: Indicates whether the layer should behave in training mode or in inference
                mode. This argument is passed to the cell when calling it.

        Returns:
            Transformed inputs.
        """

        output = super().call(sequences, initial_state=initial_state, mask=mask, training=training)

        if keras.config.backend() == "jax" and mask is not None:
            # FIXME:
            #   I have no idea why, but when using the Jax backend along with masking, a second copy of the outputs is
            #   returned along with the first. The following code just takes the first output and ignores the rest.

            output = output[0]

        return output

    def inner_loop(self, sequences, initial_state, mask, training: bool = False):
        """
        Handles the execution of the recurrent loop of the recurrent neural network.

        Args:
            sequences: Inputs into the layer, with shape ``(batch, timesteps, features)``.
            initial_state: List of initial state tensors to be passed to the first call of the cell.
                If not provided, will cause creation of zero-filled initial state tensors.
            mask: Binary tensor of shape ``(samples, timesteps)`` indicating whether a given
                timestep should be masked. An individual True entry indicates that the corresponding
                timestep should be utilized, while a False entry indicates that the corresponding
                timestep should be ignored.
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

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration for the layer.

        Returns:
            Layer configuration.
        """

        config = {
            "units": self.units,
            "state_dim": self.state_dim,
            "fully_mml": self.fully_mml,
            "r_min": self.r_min,
            "r_max": self.r_max,
            "max_phase": self.max_phase,
        }
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
