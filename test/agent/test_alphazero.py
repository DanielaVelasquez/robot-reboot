import unittest

import numpy as np
from keras.optimizer_v2.gradient_descent import SGD

from src.agent.alphazero import AlphaZeroAgent
from src.encoders.maze_and_robot_positioning_encoder import MazeAndRobotPositioningEncoder
from src.experience.alphazero_experience import AlphaZeroExperienceCollector
from src.ml.model import get_model_v2
from src.robot_reboot.classic_robot_reboot_hash import ClassicRobotRebootZobristHash
from src.robot_reboot.game import RobotRebootGame
from src.robot_reboot.goal_house import RobotRebootGoalHouse
from src.robot_reboot.state import RobotRebootState


def setup_and_get_robot_reboot_game():
    goal_house_pos = (0, 0)
    house = RobotRebootGoalHouse(1, goal_house_pos)
    maze = np.array([[0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0]
                     ])
    game = RobotRebootGame(4, maze, house)
    robot_1 = (2, 2)
    robot_2 = (4, 0)
    robot_3 = (4, 2)
    robot_4 = (4, 4)
    game_state = RobotRebootState(game, [robot_1, robot_2, robot_3, robot_4],
                                  zobrist_hash_generator=ClassicRobotRebootZobristHash())
    encoder = MazeAndRobotPositioningEncoder(game)
    model = get_model_v2(encoder.shape(), len(game.actions))
    model.compile(
        SGD(lr=0.01),
        loss=['categorical_crossentropy', 'mse'])
    return encoder, game_state, model


class TestAlphaZeroAgent(unittest.TestCase):

    def test_select_action_without_collector(self):
        encoder, game_state, model = setup_and_get_robot_reboot_game()
        alphazero_agent = AlphaZeroAgent(model, encoder, rounds_per_action=1)
        action = alphazero_agent.select_action(game_state)
        self.assertIsNotNone(action)
        next_state = game_state.apply_action(action)
        self.assertNotEqual(next_state, game_state)

    def test_select_action_with_collector(self):
        encoder, game_state, model = setup_and_get_robot_reboot_game()
        collector = AlphaZeroExperienceCollector()
        collector.begin_episode()
        alphazero_agent = AlphaZeroAgent(model, encoder, rounds_per_action=1, collector=collector)
        action = alphazero_agent.select_action(game_state)
        self.assertIsNotNone(action)
        next_state = game_state.apply_action(action)
        collector.complete_episode(game_state.get_value())
        self.assertNotEqual(next_state, game_state)
