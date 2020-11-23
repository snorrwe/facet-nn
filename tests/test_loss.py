import sys
from pydu import categorical_cross_entropy, array


def test_cce_simple():
    x = array([[0.7, 0.1, 0.2], [0.00001, 0.99999, 0.00001]])
    y = array([[1, 0, 0]] * 2)

    res = categorical_cross_entropy(x, y)

    assert len(res) == 2
    assert abs(res[0] - 0.35667494393873245) < sys.float_info.epsilon
    assert abs(res[1] - 11.512925464970229) < sys.float_info.epsilon
