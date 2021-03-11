import unittest

from src.ml.model import CNN


class TestCNNModel(unittest.TestCase):
    def test_cnn_model_init(self):
        cnn = CNN(input_shape=(31, 31, 9), n_outputs=4, convolutions=3)
        self.assertEqual(13, len(cnn.layers))
