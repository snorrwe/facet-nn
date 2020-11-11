import pytest
from nd import NdArrayD, make_ndf64


def test_ctor():
    arr = NdArrayD([12, 12, 12])
    arr.set_values([i for i in range(12 * 12 * 12)])


def test_factory():
    _arr = make_ndf64([2, 4, 3], [i for i in range(2 * 4 * 3)])


def test_setter_raises():
    arr = NdArrayD([12, 12, 12])
    with pytest.raises(ValueError):
        arr.set_values([23])
