import pytest
from nd import NdArrayD


def test_ctor():
    arr = NdArrayD([12, 12, 12])
    arr.set_values([i for i in range(12 * 12 * 12)])


def test_setter_raises():

    arr = NdArrayD([12, 12, 12])
    with pytest.raises(ValueError):
        arr.set_values([23])
