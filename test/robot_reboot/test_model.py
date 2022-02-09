import unittest
import numpy as np
from src.exceptions.robot_reboot.model import NeuralNetworkOutputNotMatchingGameActions
from src.ml.model import get_model
from src.robot_reboot.factory import RobotRebootFactory
from src.robot_reboot.model import RobotRebootModel


class TestRobotRebootModel(unittest.TestCase):
    def test_init(self):
        f = RobotRebootFactory()
        game, state, _ = f.create(31)
        cnn = get_model()
        model = RobotRebootModel(game, cnn)
        self.assertEqual(cnn, model.network)
        self.assertEqual(game, model.game)

    def test_init_fails_when_len_p_in_cnn_not_matching_game_number_actions(self):
        np.random.seed(26)
        f = RobotRebootFactory()
        game, state, _ = f.create(11)
        cnn = get_model()
        self.assertRaises(NeuralNetworkOutputNotMatchingGameActions, lambda: RobotRebootModel(game, cnn))
