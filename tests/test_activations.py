import pytest
import nd

from nd import NdArrayD, softmax


def flatten2d(lst):
    return [a for l in lst for a in l]


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
    assert (out == exp).all()


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
