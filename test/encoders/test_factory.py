import unittest

from src.encoders.factory import RobotRebootEncoderFactory


class TestRobotRebootEncoderFactory(unittest.TestCase):

    def test_get_by_name_returns_encoder_default_loads(self):
        factory = RobotRebootEncoderFactory(load_default=True)
        name = 'maze-and-two-planes-per-robot'
        encoder = factory.get_by_name(name, 4, (31, 31))
        self.assertEqual(name, encoder.name())
        self.assertEqual((31, 31, 9), encoder.shape(),
                         'Encoder was no created with the right arguments (rows, cols, 2 * robot + 1)')
