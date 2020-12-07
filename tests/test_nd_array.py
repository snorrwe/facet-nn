import pytest
import pydu
from pydu import NdArrayD, NdArrayI, array


def test_factory():
    arr = array([[[2], [2], [2]], [[1], [1], [1]]])
    assert arr.shape == [2, 3, 1]
    exp = NdArrayD([2, 3, 1], [2, 2, 2, 1, 1, 1])

    res = arr == exp
    assert res.all()

    barray = array([[[True] * 3], [[True] * 3]])
    assert res == barray


def test_sum_mat():
    arr = [[2, 4], [2, 1], [2, 3]] * 2

    res = pydu.sum(arr)

    print(res)

    assert res.shape == [6]
    assert (res == pydu.array([6, 3, 5, 6, 3, 5])).all()


def test_sum_vec():
    arr = [2, 4, 2, 1, 2, 3]

    res = pydu.sum(arr)

    print(res)

    assert len(res.shape) == 0
    assert (res == pydu.array([14])).all()


def test_sum_scalar():
    res = pydu.sum(69)
    assert (res == pydu.scalar(69)).all()


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
    c = a.matmul(b)
    assert (c == exp).all()

    # as out param
    d = NdArrayD([], [69])
    assert (d != c).all()
    e = a.matmul(b, d)

    assert (c == e).all()
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
    c = a.matmul(b)
    assert (c == exp).all()


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


def test_argmax():
    a = array([[1, 2, 3], [4, 2, 3], [1, 5, 3]])

    res = pydu.argmax(a)

    assert res.shape == [3]
    assert (res == NdArrayI([3], [2, 0, 1])).all()


def test_argmax_2():
    values = [1, 5, 3, 4]
    res = pydu.argmax(values)
    assert res.shape == []
    assert res[0] == 1, res


def test_argmax_3():
    values = [[1, 0, 5, 4]] * 3
    res = pydu.argmax(values)
    assert res.shape == [3]
    assert res[0] == 2, res
    assert res[1] == 2, res
    assert res[2] == 2, res


def test_iter():
    arr = pydu.array([[[1, 2, 3, 4]] * 4] * 4)
    assert arr.shape == [4, 4, 4]
    flat = list(arr)
    assert flat == [1, 2, 3, 4] * 16


def test_replace_where():
    arr = pydu.array([[[1, 2, 3, 4]] * 4] * 4)

    arr.replace_where(lambda _i, x: 69 if x % 2 == 0 else None)

    flat = list(arr)
    assert flat == [1, 69, 3, 69] * 16
