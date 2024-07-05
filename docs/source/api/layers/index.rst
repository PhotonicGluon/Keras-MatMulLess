Layers
======

This page lists modules containing Keras-MML layers.

It should be noted that it is not necessary to import these modules individually; just accessing the ``layers`` module is good enough to access the layers.

For example, you can access :py:class:`~keras_mml.layers.core.DenseMML` by using the following code.

.. code-block:: python

    import keras_mml

    keras_mml.layers.DenseMML(32)

.. currentmodule:: keras_mml.layers

.. rubric:: Modules

.. autosummary::
    :toctree: generated
    :template: custom-module-template.rst
    :recursive:

    activations
    core
    normalizations
    recurrent
    transformer
    misc
