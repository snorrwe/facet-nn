import pytest
from nd import NdArrayD


def test_ctor():
    """
    smoke test
    """
    arr = NdArrayD([12, 12, 12], [i for i in range(12 * 12 * 12)])
    arr.set_values([i * 2 for i in range(12 * 12 * 12)])


def test_setter_raises():
    arr = NdArrayD([12, 12, 12])
    with pytest.raises(ValueError):
        arr.set_values([23])


def test_mat_mul():
    a = NdArrayD([2, 3], [1, -2, 1, 2, 1, 3])
    b = NdArrayD([3, 2], [2, 1, 3, 2, 1, 1])

    exp = NdArrayD([2, 2], [-3, -2, 10, 7])
    c = a @ b
    assert (c == exp).all()
