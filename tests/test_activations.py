import random
import sys

import pytest

import pyfacet
from pyfacet import NdArrayD, softmax


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


def test_diagflat():
    softmax_output = pyfacet.array([0.7, 0.1, 0.2]).reshape([3, 1])
    foo = pyfacet.diagflat([a for l in softmax_output.iter_cols() for a in l])
    for i in range(3):
        for j in range(3):
            if i == j:
                assert foo[[i, j]] == softmax_output[i]
            else:
                assert foo[[i, j]] == 0
