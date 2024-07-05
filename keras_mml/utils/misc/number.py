"""
Miscellaneous number utilities.
"""


def int_to_bin(x: int, pad_len: int = 8) -> str:
    """
    Converts an integer into its binary representation.

    Args:
        x: Integer to convert.
        pad_len: Length to pad to. If the length of the original binary representation is longer
            than the pad length, this will be ignored.

    Returns:
        Binary representation of the integer.

    Examples:
        >>> int_to_bin(42, pad_len=4)
        '101010'
        >>> int_to_bin(42, pad_len=8)
        '00101010'
    """

    out = bin(x)[2:]  # Trim the "0b"
    if pad_len > len(out):
        out = "0" * (pad_len - len(out)) + out
    return out
