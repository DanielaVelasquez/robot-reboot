import unittest

from exceptions.robot_reboot.model import NeuralNetworkOutputNotMatchingGameActions
from src.ml.model import get_cnn_model
from src.robot_reboot.factory import RobotRebootFactory
from src.robot_reboot.model import RobotRebootModel


class TestRobotRebootModel(unittest.TestCase):
    def test_init(self):
        f = RobotRebootFactory(seed=26)
        game, state, _ = f.create(11)
        cnn = get_cnn_model((11, 11, 5), n_outputs=len(game.actions), convolutions=2, optimizer='adam')
        model = RobotRebootModel(game, cnn)
        self.assertEqual(cnn, model.network)
        self.assertEqual(game, model.game)

    def test_init_fails_when_len_p_in_cnn_not_matching_game_number_actions(self):
        f = RobotRebootFactory(seed=26)
        game, state, _ = f.create(11)
        cnn = get_cnn_model((11, 11, 5), n_outputs=2, convolutions=2, optimizer='adam')
        self.assertRaises(NeuralNetworkOutputNotMatchingGameActions, lambda: RobotRebootModel(game, cnn))
