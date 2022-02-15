import numpy as np
from keras.optimizer_v2.gradient_descent import SGD

from src.agent.alphazero import AlphaZeroAgent
from src.encoders.maze_and_robot_positioning_encoder import MazeAndRobotPositioningEncoder
from src.ml.model import get_model_v2
from src.robot_reboot.classic_robot_reboot_hash import ClassicRobotRebootZobristHash
from src.robot_reboot.factory import RobotRebootFactory
from src.ui.robot_reboot_game_view import RobotRebootGameView
from src.view.base import simulate_game


def main():
    np.random.seed(26)
    factory = RobotRebootFactory()
    game, game_state, selected_quadrants = factory.create(31, locate_robot_close_goal=True, max_movements=1,
                                                          zobrist_hash_generator=ClassicRobotRebootZobristHash())
    encoder = MazeAndRobotPositioningEncoder(game)

    model = get_model_v2(encoder.shape(), len(game.actions))
    model.compile(
        SGD(lr=0.01),
        loss=['categorical_crossentropy', 'mse'])

    print(game_state.robots_positions)
    print(game.goal_house)
    alphazero_agent = AlphaZeroAgent(model, encoder, rounds_per_action=30)
    rrView = RobotRebootGameView(game)
    simulate_game(alphazero_agent, game, game_state, rrView)


if __name__ == "__main__":
    main()
