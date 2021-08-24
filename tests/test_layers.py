import pyfacet as pf


def test_dense_layer_ctor():
    _ = pf.DenseLayer(16, 8)
    _ = pf.DenseLayer(16, 8, weight_regularizer_l1=0.5)
    _ = pf.DenseLayer(16, 8, weight_regularizer_l2=0.5)
    _ = pf.DenseLayer(16, 8, bias_regularizer_l1=0.5)
    _ = pf.DenseLayer(16, 8, bias_regularizer_l2=0.5)


def test_dense_forward():
    layer = pf.DenseLayer(1, 8)

    X = pf.array([[12]] * 128)

    layer.forward(X)

    assert layer.output.shape == [128, 8]
