import unittest

import numpy as np

from src.encoders.action_encoder import get_map_actions_to_index_and_actions_list
from src.robot_reboot.game import RobotRebootGame
from src.robot_reboot.goal_house import RobotRebootGoalHouse


class TestActionEncoder(unittest.TestCase):
    def test_get_map_actions_to_index_and_actions_list_with_robot_reboot_game(self):
        house = RobotRebootGoalHouse(0, (2, 2))
        maze = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]
                         ])
        game = RobotRebootGame(4, maze, house)
        map_actions_to_index, actions_list = get_map_actions_to_index_and_actions_list(game)
        self.assertEqual(len(map_actions_to_index), 16)
        self.assertEqual(len(actions_list), 16)
        for action in map_actions_to_index:
            index = map_actions_to_index[action]
            self.assertEqual(actions_list[index], action)
