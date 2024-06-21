from keras_mml.utils.misc import int_to_bin


def test_int_to_bin():
    assert int_to_bin(42, pad_len=4) == "101010"
    assert int_to_bin(42, pad_len=8) == "00101010"
