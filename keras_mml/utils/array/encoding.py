"""
Utilities for encoding special types of arrays.
"""

from collections import Counter
from typing import List, Tuple

import numpy as np

from keras_mml.utils.misc.number import int_to_bin

# fmt: off
BITS_MAPPING = {
    0: "0",   # Most common
    1: "10",  # Second most common
    2: "11"   # Least common
}
# fmt: on


def encode_ternary_array(x: np.ndarray) -> Tuple[Tuple[int, ...], bytes]:
    """
    Encodes a ternary array into a more space efficient format.

    For a given ternary matrix, which consists of only the elements ``0``, ``1``, and ``-1``, we
    convert the individual elements into bit sequences. Specifically,

    - the most common element becomes ``0``;
    - the second most common element becomes ``10``; and
    - the least common element becomes ``11``.

    We then convert these bit sequences into bytes. The first three to four bits describes which
    elements were converted to ``0`` and ``10`` respectively.

    - ``0`` will be represented by ``0``;
    - ``1`` will be represented by ``10``; and
    - ``-1`` will be represented by ``11``.

    This function will prepend this information in front of the bit sequences, and then return them
    as bytes.

    Args:
        x: Ternary array to encode.

    Returns:
        A tuple. The first element is the shape of the array. The second element is the encoded
        representation of the array.

    Examples:
        >>> x = np.array([1, -1, 1, 0, -1, 1])
        >>> shape, encoded = encode_ternary_array(x)
        >>> shape
        (6,)
        >>> encoded
        b'\\xb4\\xe0'

        >>> x = np.array([[0, 1, -1], [-1, 0, 0]])
        >>> shape, encoded = encode_ternary_array(x)
        >>> shape
        (2, 3)
        >>> encoded
        b'n\\x80'

        >>> x = np.array([[[0, 1, -1], [-1, 1, 0]], [[1, 0, 0], [0, -1, 1]]])
        >>> shape, encoded = encode_ternary_array(x)
        >>> shape
        (2, 2, 3)
        >>> encoded
        b'K\\xe48'
    """

    shape = x.shape
    flattened = x.flatten()

    # Get the frequency of each of the elements
    counts = Counter(flattened)
    elem_order = [x[0] for x in counts.most_common()]

    # Record down what the top 2 most frequent elements are
    temp_bits = ""
    temp_bits += BITS_MAPPING[[0, 1, -1].index(elem_order[0])]
    temp_bits += BITS_MAPPING[[0, 1, -1].index(elem_order[1])]

    # Then perform the encoding
    output = bytearray()
    remainder = ""
    for elem in flattened:
        # Get the bits that represent the current element
        if remainder != "" and temp_bits == "":
            temp_bits = remainder
            remainder = ""
        temp_bits += BITS_MAPPING[elem_order.index(elem)]

        # Encode every 8 bits as a byte
        if len(temp_bits) >= 8:
            main = temp_bits[:8]
            remainder = temp_bits[8:]

            byte = int(main, 2)
            output.append(byte)

            temp_bits = ""

    if len(temp_bits) != 0 or len(remainder) != 0:
        temp_bits = temp_bits + remainder
        temp_bits = temp_bits + "0" * (8 - len(temp_bits))
        byte = int(temp_bits, 2)
        output.append(byte)

    return shape, bytes(output)


def _retrieve_bit_encoding(encoded: bytes) -> Tuple[List[int], List[str]]:
    """
    Retrieves the bit encoding map from the encoded representation.

    Args:
        encoded: Encoded ternary array.

    Returns:
        A tuple. The first element is the ordering of the distinct elements, from most common to
        least common. The second element is the remaining unconsumed bit buffer.
    """

    start_index = 0
    retrieved_count = 0
    elem_order = []
    remaining = {0, 1, -1}
    buffer = list(int_to_bin(encoded[0], pad_len=8))  # Encoding information is 3 to 4 bits, which is within first byte

    while retrieved_count < 2:
        if buffer.pop(0) == "0":
            # Full sequence is "0" -> 0
            elem_order.append(0)
            remaining.remove(0)
            retrieved_count += 1
        elif buffer.pop(0) == "0":
            # Full sequence is "10" -> 1
            elem_order.append(1)
            remaining.remove(1)
            retrieved_count += 1
        else:
            # Full sequence is "11" -> -1
            elem_order.append(-1)
            remaining.remove(-1)
            retrieved_count += 1

        start_index += 1

    elem_order += remaining

    return elem_order, buffer


def decode_ternary_array(shape: Tuple[int, ...], encoded: bytes) -> np.ndarray:
    """
    Decodes the ternary array generated by :py:func:`~encode_ternary_array`.

    Args:
        shape: Shape of the original array.
        encoded: Encoded ternary array.

    Raises:
        ValueError: If the encoded byte string is empty.

    Returns:
        The decoded ternary array.

    Examples:
        >>> decode_ternary_array((6,), b"\\xb4\\xe0")
        array([ 1, -1,  1,  0, -1,  1])
        >>> decode_ternary_array((2, 3), b"n\\x80")
        array([[ 0,  1, -1],
               [-1,  0,  0]])
        >>> decode_ternary_array((2, 2, 3), b"K\\xe48")
        array([[[ 0,  1, -1],
                [-1,  1,  0]],
        <BLANKLINE>
               [[ 1,  0,  0],
                [ 0, -1,  1]]])
    """

    if len(encoded) == 0:
        raise ValueError("Cannot decode empty encoded array")

    # Allocate memory for output
    total = shape[0]
    for i in range(1, len(shape)):
        total *= shape[i]

    flattened = np.zeros((total,), dtype=int)

    # Retrieve bit encoding
    elem_order, buffer = _retrieve_bit_encoding(encoded)

    # Decode array
    i = -1  # We'll increment this right at the start
    byte_num = 0
    while True:
        i += 1
        if i == total:
            break

        if len(buffer) == 0 or (len(buffer) == 1 and buffer[0] == "1"):  # Can't decode lone "1"
            byte_num += 1
            buffer += list(int_to_bin(encoded[byte_num], pad_len=8))

        if buffer.pop(0) == "0":
            # The full sequence is "0" -> get most common element
            flattened[i] = elem_order[0]
        elif buffer.pop(0) == "0":
            # The full sequence is "10" -> get second most common element
            flattened[i] = elem_order[1]
        else:
            # The full sequence is "11" -> get least common element
            flattened[i] = elem_order[2]

    return flattened.reshape(shape)
