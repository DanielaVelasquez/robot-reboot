import unittest

from src.ml.model import CNN


class TestModel(unittest.TestCase):
    def test_model_init(self):
        cnn = CNN(input_shape=(31, 31, 9), n_outputs=4)
        print(cnn.summary())