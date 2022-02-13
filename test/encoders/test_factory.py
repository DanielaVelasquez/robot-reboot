import unittest

from src.encoders.factory import RobotRebootEncoderFactory
from src.encoders.maze_and_two_planes_per_robot import MazeAndTwoPlanesPerRobotBuilder
from test.robot_reboot.util import get_robot_reboot_game

TWO_PLANES_PER_ROBOT_ENCODER_NAME = 'maze-and-two-planes-per-robot'
POSITIONING_ENCODER_NAME = 'maze-and-robot-positioning-encoder'


class TestRobotRebootEncoderFactory(unittest.TestCase):

    def test_get_by_name_returns_encoder_default_loads(self):
        factory = RobotRebootEncoderFactory(load_default=True)
        game = get_robot_reboot_game(n_robots=4, maze_size=31)
        self.__assert_encoders_31_x_31_maze_and_4_robots(factory, TWO_PLANES_PER_ROBOT_ENCODER_NAME, game, (31, 31, 9))
        self.__assert_encoders_31_x_31_maze_and_4_robots(factory, POSITIONING_ENCODER_NAME, game, (31, 31, 13))

    def __assert_encoders_31_x_31_maze_and_4_robots(self, factory, name, game, expected_shape):
        encoder = factory.get_by_name(name, game)
        self.assertEqual(name, encoder.name())
        self.assertEqual(expected_shape, encoder.shape(),
                         'Encoder was no created with the right arguments (rows, cols, 2 * robot + 1)')

    def test_get_by_name_throws_value_error_encoder_not_found(self):
        factory = RobotRebootEncoderFactory(load_default=False)
        name = 'maze-and-two-planes-per-robot'
        self.assertRaises(ValueError, lambda: factory.get_by_name(name, get_robot_reboot_game()))

    def test_register_encoder(self):
        factory = RobotRebootEncoderFactory(load_default=False)
        factory.register_encoder(TWO_PLANES_PER_ROBOT_ENCODER_NAME, MazeAndTwoPlanesPerRobotBuilder())
        encoder = factory.get_by_name(TWO_PLANES_PER_ROBOT_ENCODER_NAME, get_robot_reboot_game())
        self.assertEqual(TWO_PLANES_PER_ROBOT_ENCODER_NAME, encoder.name())
        self.assertEqual((31, 31, 9), encoder.shape(),
                         'Encoder was no created with the right arguments (rows, cols, 2 * robot + 1)')
