import argparse
import logging
import os
import sys
import uuid

import h5py
import keras.models
import numpy as np

from src.agent.alphazero import AlphaZeroAgent
from src.encoders.maze_and_robot_positioning_encoder import MazeAndRobotPositioningEncoder
from src.experience.alphazero_experience import AlphaZeroExperienceCollector
from src.game_simulator.base import simulate_game
from src.robot_reboot.classic_robot_reboot_hash import ClassicRobotRebootZobristHash
from src.robot_reboot.factory import RobotRebootFactory

logging.getLogger().setLevel(logging.INFO)


def self_play(path_to_model, model_name, seed, number_games, rounds_per_action, locate_robot_close_goal, max_movements,
              max_actions_per_game):
    experience_directory = f'experiences/{model_name}'
    assert os.path.isdir(experience_directory)
    assert os.path.isdir(path_to_model)
    logging.info('Loading model ' + path_to_model)
    model = keras.models.load_model(path_to_model)

    np.random.seed(seed)
    factory = RobotRebootFactory()
    n_movement_choices = [i for i in range(1, max_movements + 1)]
    self_play_id = str(uuid.uuid4().time)[:8]
    logging.info(f'Self play id {self_play_id}')
    for i in range(number_games):
        logging.info('Starting game ' + str(i + 1) + '/' + str(number_games))
        n_movements = np.random.choice(n_movement_choices)
        game, game_state, selected_quadrants = factory.create(31, locate_robot_close_goal=locate_robot_close_goal,
                                                              n_movements=n_movements,
                                                              zobrist_hash_generator=ClassicRobotRebootZobristHash(),
                                                              move_all_robots=True)
        encoder = MazeAndRobotPositioningEncoder(game)
        collector = AlphaZeroExperienceCollector()
        alphazero_agent = AlphaZeroAgent(model, encoder, rounds_per_action=rounds_per_action, collector=collector)
        final_state = simulate_game(game_state, alphazero_agent, collector, max_actions=max_actions_per_game)

        value = final_state.get_value()
        total_actions = final_state.sequence_i
        logging.info('Finished game ' + str(i + 1) + '/' + str(number_games) +
                     '\nValue= ' + str(value) +
                     '\nTotal actions = ' + str(total_actions))
        buffer = collector.to_buffer()
        experience_file_name = f'{experience_directory}/experience-{i}-{self_play_id}.hd5f'
        with h5py.File(experience_file_name, 'w') as experience_out:
            buffer.serialize(experience_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_model',
        type=str,
        required=True,
        help='Path to the model'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Name of the model to use for self-play'
    )
    parser.add_argument(
        '--number_games',
        type=int,
        required=True,
        help='Number of games to simulate'
    )
    parser.add_argument(
        '--rounds_per_action',
        type=int,
        required=True,
        help='Number of states explore before picking an action'
    )
    parser.add_argument(
        '--max_movements',
        type=int,
        required=False,
        default=None,
        help='Number of actions the robot is away from its target'
    )
    parser.add_argument(
        '--seed',
        type=int,
        required=False,
        default=26,
        help='Value used to generate the games, it allows reproducible scenarios'
    )
    parser.add_argument(
        '--max_actions_per_game',
        type=int,
        default=sys.maxsize,
        help='Maximum number of actions one single game can have'
    )

    args = parser.parse_args()
    locate_robot_close_goal = args.max_movements is not None

    self_play(args.path_to_model, args.model_name, args.seed, args.number_games, args.rounds_per_action,
              locate_robot_close_goal, args.max_movements, args.max_actions_per_game)

    # self_play('models/model_0', 'model_0', 26, 1, 50, True, 1, 2)
