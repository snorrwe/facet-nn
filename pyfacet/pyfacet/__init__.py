from operator import mul
import random
import string

from .pyfacet import *


def labels_to_once_hot(labels):
    """
    convert a list of labels to a one-hot dense target matrix
    """
    lset = set(labels)
    label_pos = {label: i for (i, label) in enumerate(lset)}
    n_classes = len(lset)

    res = []
    for label in labels:
        v = [0.0] * n_classes
        v[label_pos[label]] = 1.0
        res.append(v)

    return pyfacet.array(res)


def accuracy(pred, target):
    """
    How often the prediction matches the target, on average
    """
    # collapse dense data into sparse data
    if len(target.shape) == 2:
        target = pyfacet.argmax(target)
    pred = pyfacet.argmax(pred)
    diff = pred == target
    return pyfacet.mean(diff.as_f64())


def first_n(n, it):
    """
    :param it: iterable
    :return: the first n items in it
    """

    try:
        for _ in range(n):
            yield next(it)
    except StopIteration:
        return


def reduce(op, arr, init):
    for x in arr:
        init = op(init, x)
    return init


def cce_backward(z, y):
    samples = reduce(mul, z.shape[:-1], 1)
    gradient = (y / z) * pyfacet.scalar(-1.0)
    # Normalize gradient
    return gradient / pyfacet.scalar(samples)
