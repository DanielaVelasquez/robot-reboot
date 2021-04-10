import unittest

from src.robot_reboot.factory import RobotRebootFactory
from src.uct.model import UctModel


class TestUctModel(unittest.TestCase):
    def test_predict_returns_1_for_each_action(self):
        game, state, quadrants_ids = RobotRebootFactory().create(31, locate_robot_close_goal=True, max_movements=4)
        uctModel = UctModel(game)
        v, p = uctModel.predict(state)
        self.assertIsNone(v)
        for i in p:
            self.assertEqual(i, 1)
