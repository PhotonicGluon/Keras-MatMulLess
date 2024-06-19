import pytest

from keras_mml.utils.validation import ensure_is_rank_2


def test_ensure_is_rank_2():
    with pytest.raises(ValueError):
        ensure_is_rank_2((1,))

    ensure_is_rank_2((1, 2))

    with pytest.raises(ValueError):
        ensure_is_rank_2((1, 2, 3))
