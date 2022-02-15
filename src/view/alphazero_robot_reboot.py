import numpy as np
from keras.optimizer_v2.gradient_descent import SGD

from src.agent.alphazero import AlphaZeroAgent
from src.agent.human import HumanAgent
from src.encoders.maze_and_robot_positioning_encoder import MazeAndRobotPositioningEncoder
from src.ml.model import get_model_v2
from src.robot_reboot.classic_robot_reboot_hash import ClassicRobotRebootZobristHash
from src.robot_reboot.factory import RobotRebootFactory
from src.robot_reboot.game import RobotRebootGame
from src.robot_reboot.goal_house import RobotRebootGoalHouse
from src.robot_reboot.state import RobotRebootState
from src.ui.robot_reboot_game_view import RobotRebootGameView
from src.view.base import simulate_game


def main():
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

    print(game_state.robots_positions)
    print(game.goal_house)
    alphazero_agent = AlphaZeroAgent(model, encoder)
    rrView = RobotRebootGameView(game)
    simulate_game(alphazero_agent, game, game_state, rrView)


if __name__ == "__main__":
    main()
