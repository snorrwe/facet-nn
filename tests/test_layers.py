import pydu


def test_dense_layer_ctor():
    _layer = pydu.DenseLayer(16, 8)
    _layer = pydu.DenseLayer(16, 8, weight_regularizer_l1=0.5)
    _layer = pydu.DenseLayer(16, 8, weight_regularizer_l2=0.5)
    _layer = pydu.DenseLayer(16, 8, bias_regularizer_l1=0.5)
    _layer = pydu.DenseLayer(16, 8, bias_regularizer_l2=0.5)