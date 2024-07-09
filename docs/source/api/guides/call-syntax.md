# Understanding the Call Syntax

This guide will help you understand the calling syntax of the layers.

## Why?

We use the [`jaxtyping`](https://github.com/patrick-kidger/jaxtyping) package to help write type annotations for arrays. Specifically, these [type annotations](https://docs.kidger.site/jaxtyping/api/array/) specify the shape and type of the arrays (actually *tensors*) that these layers accept.

## How To Interpret

Consider the following call signature.

::::{grid}
:outline:

:::{grid-item}

```{eval-rst}
.. method:: SomeLayer.call(self, inputs)

   Calling method of the layer.

   :param inputs: Inputs into the layer.
   :type inputs: :class:`Float[Array, 'batch_size *dims last_dim']`
   :returns: :class:`Float[Array, 'batch_size *dims units']` – Transformed inputs.
```

:::

::::

Let us unpack what the calling syntax `Float[Array, 'batch_size *dims last_dim']` means.

- The `Float` means that the array (tensor) is supposed to contain floating point values.
- `Array` indicates that we are using an array.
- `'batch_size *dims last_dim'` specifies the shape of the array.
  - The shape is a string of space-separated symbols, such as `'a b c d'`. Each symbol represents a separate axis.
  - An axis that is prepended with `*` (like `*dims` in the above example) means that it can be used to match multiple axes (or no axis). Thus `*dims` matches any intermediate axes.
  
  So the shape specified by `'batch_size *dims last_dim'` is one that
  - takes a variable `batch_size` as the first axis;
  - accepts any number of variable axes as the intermediate `dims`; and
  - accepts a variable `last_dim` as the last axis.

Now, referring to the return type `Float[Array, 'batch_size *dims units']`,

- the `Float` means that a tensor of floats will be returned;
- `Array` indicates that we are returning an array; and
- `'batch_size *dims units'` indicates that the shape of the array is almost the same as the input array, except that the last dimension is changed to be the value of `units`.
