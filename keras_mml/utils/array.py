"""
Utilities for handing numpy arrays.
"""

import os
from typing import Any, Tuple

import numpy as np

from keras_mml.utils.misc import int_to_bin


def as_numpy(x: Any) -> np.ndarray:
    """
    Converts the given object into a numpy array.

    This function will first attempt to call the ``.numpy()`` method of the object and return it.
    Failing to do so, it will revert to the :external:py:func:`numpy.array()` function.

    Args:
        x: Object to convert into a numpy array.

    Returns:
        Numpy array of the given object.
    """

    try:
        return x.numpy()
    except RuntimeError:
        if os.environ["KERAS_BACKEND"] == "torch":
            return as_numpy(x.detach())
    except AttributeError:
        pass

    return np.array(x)


def encode_ternary_array(x: np.ndarray) -> Tuple[Tuple[int, int], bytes]:
    """
    Encodes a ternary array into a more space efficient format.
    
    For a given ternary matrix, we convert the individual elements into bit sequences. Specifically,
    
    - ``0`` becomes ``0``;
    - ``1`` becomes ``10``; and
    - ``-1`` becomes ``11``.
    
    We then convert these bit sequences into bytes. These are the bytes that are returned.

    Args:
        x: Ternary array to encode.

    Raises:
        ValueError: If the input array is not 2-dimensional.

    Returns:
        A tuple. The first element is the shape of the array. The second element is the encoded
        representation of the array.
    
    Examples:
        >>> x = np.array([[0, 1, -1], [-1, 0, 1]])
        >>> shape, encoded = encode_ternary_array(x)
        >>> shape
        (2, 3)
        >>> encoded
        b'^\\x80'
    """

    shape = x.shape
    if len(shape) != 2:
        raise ValueError("Input array must be 2-dimensional")

    flattened = x.flatten()

    output = bytearray()
    temp_bits = ""
    remainder = ""
    for elem in flattened:
        if remainder != "" and temp_bits == "":
            temp_bits = remainder
            remainder = ""

        # Encode 0 as "0", 1 as "10", and -1 as "11" in bits
        if elem == 0:
            part = "0"
        elif elem == 1:
            part = "10"
        else:
            part = "11"

        temp_bits += part

        # Encode every 8 bits as a byte
        if len(temp_bits) >= 8:
            main = temp_bits[:8]
            remainder = temp_bits[8:]

            byte = int(main, 2)
            output.append(byte)

            temp_bits = ""

    if len(temp_bits) != 0:
        temp_bits = temp_bits + "0" * (8 - len(temp_bits))
        byte = int(temp_bits, 2)
        output.append(byte)

    return shape, bytes(output)


def decode_ternary_array(shape: Tuple[int, int], encoded: bytes) -> np.ndarray:
    """
    Decodes the ternary array generated by :py:func:`~encode_ternary_array`.

    Args:
        shape: Shape of the original array.
        encoded: Encoded ternary array.

    Raises:
        ValueError: If the shape is not of rank 2.
        ValueError: If the encoded byte string is empty.

    Returns:
        The decoded ternary array.
    
    Examples:
        >>> decode_ternary_array((2, 3), b"^\\x80")
        array([[ 0,  1, -1],
               [-1,  0,  1]])
    """
    
    if len(shape) != 2:
        raise ValueError("Shape must be of rank 2")

    if len(encoded) == 0:
        raise ValueError("Cannot decode empty encoded array")
    
    total = shape[0] * shape[1]
    flattened = np.zeros((total,), dtype=int)
    
    i = -1  # We'll increment this right at the start
    byte_num = 0
    remaining = list(int_to_bin(encoded[0], pad_len=8))  # Trim off the first byte
    while True:
        if len(remaining) == 0 or (len(remaining) == 1 and remaining[0] == "1"):  # Can't decode lone "1"
            byte_num += 1
            remaining += list(int_to_bin(encoded[byte_num], pad_len=8))

        i += 1
        
        if i == total:
            break
        
        if remaining.pop(0) == "0":
            # Should be a zero, but since the array is already all zeroes, we can skip
            continue
        
        if remaining.pop(0) == "0":
            # The full sequence is "10" -> 1
            flattened[i] = 1
        else:
            # The full sequence is "11" -> -1
            flattened[i] = -1
           
    return flattened.reshape(shape)
