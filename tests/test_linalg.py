"""
testing generic linalg functions
"""
from pyfacet import NdArrayD, normalize_vectors

import random
from math import sqrt


def test_normalize_vectors():
    out = normalize_vectors(
        [[[random.randrange(-123, 123) for _ in range(10)]] * 10] * 5
    )

    for vec in out.iter_rows():
        l = sqrt(sum(x * x for x in vec))
        diff = abs(1.0 - l)
        assert diff < 0.02
