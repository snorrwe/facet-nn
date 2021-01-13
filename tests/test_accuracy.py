from pyfacet.accuracy import Accuracy_Regression
import pyfacet as pf


def test_accuracy():

    y = pf.array([[1, 0, 0]] * 3)
    pred = pf.array([[1, 1, 1]] * 3)

    acc = Accuracy_Regression()
    acc.init(y)

    res = acc.calculate(pred, y)

    assert 0.0 < res < 0.12
