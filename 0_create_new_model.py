import argparse
import os

from keras.optimizer_v2.gradient_descent import SGD

from src.encoders.maze_and_robot_positioning_encoder import MazeAndRobotPositioningEncoder
from src.robot_reboot.classic_robot_reboot_hash import ClassicRobotRebootZobristHash
from src.robot_reboot.factory import RobotRebootFactory
from src.robot_reboot.model import get_model_v2


def create_model(model_path, model_name):
    assert os.path.isdir(model_path)
    factory = RobotRebootFactory()
    # This configuration does not matter much at this stage. It is only to generate the encoder correctly
    game, game_state, selected_quadrants = factory.create(31, locate_robot_close_goal=True,
                                                          n_movements=20,
                                                          zobrist_hash_generator=ClassicRobotRebootZobristHash(),
                                                          move_all_robots=True)
    encoder = MazeAndRobotPositioningEncoder(game)
    model = get_model_v2(encoder.shape(), len(game.actions))
    model.compile(
        SGD(learning_rate=0.01),
        loss=['categorical_crossentropy', 'mse'])
    model.save(f'{model_path}/{model_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        required=False,
        default='models/',
        help='Models name'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=False,
        default='model_0',
        help='Models name'
    )
    args = parser.parse_args()
    create_model(args.model_path, args.model_name)
