from pyfacet import categorical_cross_entropy, array
from pyfacet.loss import MeanSquaredError


def test_cce_simple():
    x = array([[0.7, 0.1, 0.2], [0.00001, 0.99999, 0.00001]])
    y = array([[1, 0, 0]] * 2)

    res = categorical_cross_entropy(x, y)

    assert len(res) == 2
    assert abs(res[0] - 0.35667) < 0.01
    assert abs(res[1] - 11.51293) < 0.01


def test_mse_simple():
    x = array([[0.7, 0.1, 0.2]] * 2)
    y = array([[1, 0, 0]] * 2)

    loss = MeanSquaredError()

    res = loss.forward(x, y)

    exp = (0.3 ** 2 + 0.1 ** 2 + 0.2 ** 2) / 3.0

    for l in res:
        assert abs(exp - l) < 0.01

    loss.backward(array([[1] * 3] * 2), y)

    # just make sure dinputs is defined
    print(loss.dinputs)
    assert loss.dinputs.shape == [2, 3]
