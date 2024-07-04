"""
Implements an almost matmul-less Linear Recurrent Unit (LRU) layer.
"""

from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
from keras import ops, random

from keras_mml.layers.core import DenseMML
from keras_mml.layers.recurrent.rnn import RNN, DropoutRNNCell


@keras.saving.register_keras_serializable(package="keras_mml")
class _GammaLogInitializer(keras.initializers.Initializer):
    def __init__(self, nu_log: np.ndarray):
        self.nu_log = nu_log

    def __call__(self, *args, **kwargs):
        lambda_mod = ops.exp(-ops.exp(self.nu_log))
        gamma_log = ops.log(ops.sqrt(ops.ones_like(lambda_mod) - ops.square(lambda_mod)))
        return gamma_log


@keras.saving.register_keras_serializable(package="keras_mml")
class LRUCellMML(keras.Layer, DropoutRNNCell):
    """
    Cell class for the :py:class:`~LRUMML` layer.

    This class processes one step within the whole time sequence input, whereas :py:class:`~LRUMML`
    processes the whole sequence.

    .. admonition:: Calling Convention
        :class: tip

        - **Input Shape**: 2D tensor of shape ``(batch_size, features)``
        - **Output Shape**: ``(batch_size, units)``

    Attributes:
        units: Dimensionality of the output space.
        state_dim: Dimensionality of the internal state space.
        fully_mml: Whether to use matmul-free operations for all the layers.
        r_min: Minimum modulus of the complex weights in :math:`\\mathbf{\\Lambda}`.
        r_max: Maximum modulus of the complex weights in :math:`\\mathbf{\\Lambda}`.
        max_phase: Maximum phase of the complex weights in :math:`\\mathbf{\\Lambda}`.
        use_bias: Whether to use a bias vector for the layer.
        dropout: Fraction of the units to drop for the linear transformation of the inputs. Should
            be a float between 0 and 1.
        recurrent_dropout: Fraction of the units to drop for the linear transformation of the
            recurrent state. Should be a float between 0 and 1.
        seed: Random seed for dropout.
        state_size: Size of the recurrent state.
        output_size: Size of the output vector.
    """

    def __init__(
        self,
        units: int,
        state_dim: int,
        fully_mml: bool = False,
        r_min: float = 0,
        r_max: float = 1,
        max_phase: float = np.pi * 2,
        use_bias: bool = False,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        seed: Optional[int] = None,
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
            use_bias: Whether to use a bias vector for the layer.
            dropout: Fraction of the units to drop for the linear transformation of the inputs.
                Should be a float between 0 and 1.
            recurrent_dropout: Fraction of the units to drop for the linear transformation of the
                recurrent state. Should be a float between 0 and 1.
            seed: Random seed for dropout.
            **kwargs: Keyword arguments for :py:class:`keras.Layer`.

        Raises:
            ValueError: If the units provided is not a positive integer.
            ValueError: If the state dimensionality provided is not a positive integer.
        """

        if units <= 0:
            raise ValueError(f"Invalid number of units. Expected a positive integer, got {units}.")

        if state_dim <= 0:
            raise ValueError(f"Invalid state dimensionality. Expected a positive integer, got {state_dim}.")

        keras.layers.Layer.__init__(self, **kwargs)
        DropoutRNNCell.__init__(self)

        self.input_spec = keras.layers.InputSpec(ndim=2)

        self.state_size = state_dim
        self.output_size = units

        # Main attributes
        self.units = units
        self.state_dim = state_dim
        self.fully_mml = fully_mml
        self.use_bias = use_bias

        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))

        self.seed = seed
        self.seed_generator = random.SeedGenerator(seed=seed)

        # Hidden weights/layers
        self._nu_log = None
        self._theta_log = None
        self._gamma_log = None

        self._b_gate_re = None
        self._b_gate_im = None
        self._c_gate_re = None
        self._c_gate_im = None
        self._d_gate = None

    # Helper methods
    def _init_nu_log(self, shape: Tuple[int, ...], dtype: str = None) -> np.ndarray:
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

    def _init_theta_log(self, shape: Tuple[int, ...], dtype: str = None) -> np.ndarray:
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

    def _init_b_matrix(self, shape: Tuple[int, int], dtype: str = None) -> np.ndarray:
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

    def _init_c_matrix(self, shape: Tuple[int, int], dtype: str = None) -> np.ndarray:
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
        self._nu_log = self.add_weight(name="nu_log", shape=(self.state_dim,), initializer=self._init_nu_log)
        self._theta_log = self.add_weight(name="theta_log", shape=(self.state_dim,), initializer=self._init_theta_log)

        # Glorot initialized Input/Output projection matrices
        self._b_gate_re = DenseMML(
            self.state_dim, use_bias=self.use_bias, kernel_initializer=self._init_b_matrix, name="B_re"
        )
        self._b_gate_im = DenseMML(
            self.state_dim, use_bias=self.use_bias, kernel_initializer=self._init_b_matrix, name="B_im"
        )
        self._c_gate_re = output_layer_class(
            self.units, use_bias=self.use_bias, kernel_initializer=self._init_c_matrix, name="C_re"
        )
        self._c_gate_im = output_layer_class(
            self.units, use_bias=self.use_bias, kernel_initializer=self._init_c_matrix, name="C_im"
        )
        self._d_gate = output_layer_class(
            self.units, use_bias=self.use_bias, kernel_initializer="glorot_normal", name="D"
        )

        self._b_gate_re.build(input_shape)
        self._b_gate_im.build(input_shape)
        self._c_gate_re.build((None, self.state_dim))
        self._c_gate_im.build((None, self.state_dim))
        self._d_gate.build(input_shape)

        # Normalization factor
        self._gamma_log = self.add_weight(
            name="gamma_log", shape=(self.state_dim,), initializer=_GammaLogInitializer(self._nu_log)
        )

    def call(self, inputs, states, training=False):
        """
        Calling method of the cell.

        Args:
            inputs: Inputs into the layer.
            states: State(s) from the previous timestep.
            training: Whether the layer should behave in training mode or in inference mode.

        Returns:
            Transformed inputs.
        """

        # Get previous state
        state = states[0] if keras.tree.is_nested(states) else states
        state_re, state_im = ops.split(state, 2, axis=1)
        state_re = ops.squeeze(state_re, axis=1)
        state_im = ops.squeeze(state_im, axis=1)

        # Handle dropping out
        dp_mask = self.get_dropout_mask(inputs)
        recurrent_dp_mask = self.get_recurrent_dropout_mask(inputs)

        if training and 0.0 < self.dropout < 1.0:
            inputs = inputs * dp_mask
        if training and 0.0 < self.recurrent_dropout < 1.0:
            state_re = state_re * recurrent_dp_mask
            state_im = state_im * recurrent_dp_mask

        # Compute real and imaginary parts of the diagonal lambda matrix
        lambda_mod = ops.exp(-ops.exp(self._nu_log))
        lambda_re = lambda_mod * ops.cos(ops.exp(self._theta_log))
        lambda_im = lambda_mod * ops.sin(ops.exp(self._theta_log))

        # Get the normalization factor, gamma
        gamma = ops.exp(self._gamma_log)

        # Compute new state
        new_state_re = lambda_re * state_re - lambda_im * state_im + gamma * self._b_gate_re(inputs)
        new_state_im = lambda_re * state_im + lambda_im * state_re + gamma * self._b_gate_im(inputs)

        new_state = ops.stack([new_state_re, new_state_im], axis=1)
        new_state = [new_state] if keras.tree.is_nested(states) else new_state

        # Compute output
        output = self._c_gate_re(new_state_re) - self._c_gate_im(new_state_im) + self._d_gate(inputs)
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
                "use_bias": self.use_bias,
                "dropout": self.dropout,
                "recurrent_dropout": self.recurrent_dropout,
                "seed": self.seed,
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
class LRUMML(RNN):
    """
    Linear Recurrent Unit (LRU) layer, mostly without matrix multiplications.

    The core algorithm of this layer mostly follows the implementation in |LinearRU|_ (see Appendix
    A), with a few modifications from |LRU-PyTorch|_. We replace some matrix multiplications with
    ternary weights, although the option to use it as fully matrix multiplication free is available
    using the :py:attr:`~fully_mml` attribute.

    .. admonition:: Calling Convention
        :class: tip

        - **Input Shape**: 3D tensor of shape ``(batch_size, timesteps, features)``
            - Takes an optional mask of shape ``(batch_size, timesteps)``
        - **Output Shape**: ``(batch_size, units)``

    Attributes:
        units: Dimensionality of the output space.
        state_dim: Dimensionality of the internal state space.
        fully_mml: Whether to use matmul-free operations for all the layers.
        r_min: Minimum modulus of the complex weights in :math:`\\mathbf{\\Lambda}`.
        r_max: Maximum modulus of the complex weights in :math:`\\mathbf{\\Lambda}`.
        max_phase: Maximum phase of the complex weights in :math:`\\mathbf{\\Lambda}`.
        use_bias: Whether to use a bias vector for the layer.
        dropout: Fraction of the units to drop for the linear transformation of the inputs. Should
            be a float between 0 and 1.
        recurrent_dropout: Fraction of the units to drop for the linear transformation of the
            recurrent state. Should be a float between 0 and 1.
        seed: Random seed for dropout.

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
        max_phase: float = np.pi * 2,
        use_bias: bool = False,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        seed: Optional[int] = None,
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
            use_bias: Whether to use a bias vector for the layer.
            dropout: Fraction of the units to drop for the linear transformation of the inputs.
                Should be a float between 0 and 1.
            recurrent_dropout: Fraction of the units to drop for the linear transformation of the
                recurrent state. Should be a float between 0 and 1.
            seed: Random seed for dropout.
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
            use_bias=use_bias,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            seed=seed,
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

    @property
    def use_bias(self):
        """
        :meta private:
        """
        return self.cell.use_bias

    @property
    def dropout(self):
        """
        :meta private:
        """
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        """
        :meta private:
        """
        return self.cell.recurrent_dropout

    @property
    def seed(self):
        """
        :meta private:
        """
        return self.cell.seed

    # Public methods
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
            "use_bias": self.use_bias,
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "seed": self.seed,
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
            Created instance.
        """

        return cls(**config)
