import unittest

from src.ml.model import get_cnn_model


class TestCnnModel(unittest.TestCase):
    def test_something(self):
        cnn = get_cnn_model((11, 11, 5), n_outputs=5, convolutions=2, optimizer='sgd')
        self.assertEqual(2, len(cnn.loss), "Two losses should be computed")
        self.assertEqual(5, cnn.outputs[1].shape[1])
