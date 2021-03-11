import unittest
import numpy as np
from src.ml.model import CNN


class TestModel(unittest.TestCase):
    def test_model_init(self):
        cnn = CNN(input_shape=(31, 31, 9), n_outputs=4)
        # cnn.build(np.arange(31*31*9).reshape(31,31,9,1))
        cnn.compile()
        print(cnn.summary())
