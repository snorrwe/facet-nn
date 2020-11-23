import random
import sys

import pytest

from pydu import NdArrayD, softmax


def flatten2d(lst):
    return [a for l in lst for a in l]


def test_softmax_rand():
    """
    output has to have the same shape as input
    each column of ouput should have a sum of 1.0 and the same ordering as the input
    """
    M = 10_000
    N = 4
    inp = [[random.randrange(-69, 42) for _ in range(N)] for _ in range(M)]
    inp = NdArrayD([M, N], flatten2d(inp))

    out = softmax(inp)

    assert out.shape == [M, N]

    for (out, inp) in zip(out.iter_cols(), inp.iter_cols()):
        s = sum(out)
        assert abs(s - 1) <= sys.float_info.epsilon * 8  # add some tolerance

        # max indices
        maxi = max(range(len(inp)), key=inp.__getitem__)
        maxo = max(range(len(out)), key=out.__getitem__)

        try:
            assert maxi == maxo
        except AssertionError:
            # if they differ check if the inputs were close...
            in_err = abs(inp[maxi] - inp[maxo])
            assert (
                in_err < sys.float_info.epsilon
            ), f"Max in: {maxi} max out: {maxo}\n in, out:\n{inp}\n{out}"


def test_softmax_batch():
    # 3 3d matrices
    inp = NdArrayD([3, 3], flatten2d([[1, -2, 1], [2, 1, 3], [1, 2, 3],]))
    out = softmax(inp)
    exp = NdArrayD(
        [3, 3],
        flatten2d(
            [
                [0.4878555511603684, 0.024288897679263212, 0.4878555511603684],
                [0.24472847105479764, 0.09003057317038046, 0.6652409557748218],
                [0.09003057317038046, 0.24472847105479764, 0.6652409557748218],
            ]
        ),
    )
    assert (out == exp).all(), out


def test_softmax_single():
    # 1 3d matrices
    inp = NdArrayD([3], [1, 2, 3])
    out = softmax(inp)
    print(out)
    assert (
        out
        # example taken from the NNFS book
        == NdArrayD([3], [0.09003057317038046, 0.24472847105479764, 0.6652409557748218])
    ).all()


def test_softmax_scalar():
    # 1 3d matrices
    inp = NdArrayD([0], [1123])
    out = softmax(inp)
    assert (out == NdArrayD([0], [1])).all()
