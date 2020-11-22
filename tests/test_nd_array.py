import pytest
from nd import NdArrayD, array


def test_factory():
    arr = array([[[2], [2], [2]], [[1], [1], [1]]])
    assert arr.shape == [2, 3, 1]
    exp = NdArrayD([2, 3, 1], [2, 2, 2, 1, 1, 1])
    assert (arr == exp).all()


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
    d = a.matmul(b)
    assert (c == exp).all()
    assert (c == d).all()


def test_mat_mul_many():
    """
    take 8 of each 'template' matrix then apply matmul

    the result should be the same as `test_mat_mul` 8 times
    """
    a_template = [1, -2, 1, 2, 1, 3]
    b_template = [2, 1, 3, 2, 1, 1]
    a = NdArrayD([2, 4, 2, 3], a_template * 8)
    b = NdArrayD([8, 3, 2], b_template * 8)

    exp = NdArrayD([8, 2, 2], [-3, -2, 10, 7] * 8)
    c = a @ b
    d = a.matmul(b)
    assert (c == exp).all()
    assert (c == d).all()


def test_adding_scalar():
    a = NdArrayD([0], [69])
    b = NdArrayD([8, 8, 3], [0 for _ in range(8 * 8 * 3)])

    c = a + b
    d = b + a

    assert (c == d).all()

    exp = NdArrayD([8, 8, 3], [69 for _ in range(8 * 8 * 3)])

    assert (c == exp).all()


def test_adding_vectors():
    a = NdArrayD([4], [i for i in range(4)])
    b = NdArrayD([3, 4])

    c = a + b
    d = b + a

    assert (c == d).all()

    exp = NdArrayD([3, 4], [i for i in range(4)] * 3)

    assert (c == exp).all()


def test_adding_matrix_to_tensor():
    a_template = [1, -2, 1, 2, 1, 3]
    b_template = [2, 1, 3, 2, 1, 1]

    a = NdArrayD([2, 3], a_template)
    b = NdArrayD([8, 2, 3], b_template * 8)

    c = a + b
    d = b + a

    assert (c == d).all()

    exp = NdArrayD([8, 2, 3], [3, -1, 4, 4, 2, 4] * 8)

    assert (c == exp).all()

    # convert back by subtraction...
    e = c - a
    assert (e == b).all()
