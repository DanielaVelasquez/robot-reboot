import argparse
import logging
import os
import sys
import uuid

import h5py
import keras.models
import numpy as np
import pandas as pd

from src.agent.alphazero import AlphaZeroAgent
from src.encoders.maze_and_robot_positioning_encoder import MazeAndRobotPositioningEncoder
from src.experience.alphazero_experience import AlphaZeroExperienceCollector
from src.game_simulator.base import simulate_game
from src.robot_reboot.classic_robot_reboot_hash import ClassicRobotRebootZobristHash
from src.robot_reboot.factory import RobotRebootFactory

logging.getLogger().setLevel(logging.INFO)


def self_play(path_to_models, model_names, path_to_results, seed, number_games, rounds_per_action,
              locate_robot_close_goal, max_movements, max_actions_per_game):
    logging.info('path to models ' + str(path_to_models))
    logging.info('model_names ' + str(model_names))
    assert len(path_to_models) == len(model_names)
    assert os.path.isdir(path_to_results)
    logging.info('path to models ' + str(path_to_models))
    for path_to_model in path_to_models:
        assert os.path.isdir(path_to_model)

    logging.info('Loading models')
    models = [keras.models.load_model(path_to_model) for path_to_model in path_to_models]

    np.random.seed(seed)
    factory = RobotRebootFactory()
    if max_movements:
        n_movement_choices = [i for i in range(1, max_movements + 1)]

    collectors = [AlphaZeroExperienceCollector() for _ in path_to_models]
    wins = [0 for _ in path_to_models]

    results = list()

    competition_id = str(uuid.uuid4().time)[:8]
    logging.info(f'Competition {competition_id}')

    for i in range(number_games):
        logging.info('Starting game ' + str(i + 1) + '/' + str(number_games))
        n_movements = get_n_movements(max_movements, n_movement_choices)

        game, game_state, selected_quadrants = factory.create(31, locate_robot_close_goal=locate_robot_close_goal,
                                                              n_movements=n_movements,
                                                              zobrist_hash_generator=ClassicRobotRebootZobristHash(),
                                                              move_all_robots=True)
        encoder = MazeAndRobotPositioningEncoder(game)
        for j, model in enumerate(models):
            collector = collectors[j]
            alphazero_agent = AlphaZeroAgent(model, encoder, rounds_per_action=rounds_per_action, collector=collector)
            final_state = simulate_game(game_state, alphazero_agent, collector, max_actions=max_actions_per_game)
            value = final_state.get_value()
            total_actions = final_state.sequence_i
            result = {
                'seed': seed,
                'number_games': number_games,
                'rounds_per_action': rounds_per_action,
                'locate_robot_close_goal': locate_robot_close_goal,
                'n_movements': n_movements,
                'max_actions_per_game': max_actions_per_game,
                'game': i + 1,
                'model': model_names[j],
                'value': value,
                'total_actions': total_actions,
            }
            results.append(result)
            wins[j] += value
            logging.info('Model ' + model_names[j] + ' finished game ' + str(i + 1) + '/' + str(number_games))
            logging.info("{value = " + str(value) + "# actions = " + str(total_actions) + "}")

    for i, collector in enumerate(collectors):
        buffer_1 = collector.to_buffer()
        experience_file_name = f'{path_to_results}/experiences-{model_names[i]}-{competition_id}.hd5f'
        with h5py.File(experience_file_name, 'w') as experience_out:
            buffer_1.serialize(experience_out)

    df = pd.DataFrame(results)
    df.to_csv(f'{path_to_results}/results-{competition_id}.csv')
    logging.info('Competition' + competition_id + ' concluded')
    for i, win in enumerate(wins):
        logging.info('Model ' + model_names[i] + ' won ' + str(win))


def get_n_movements(max_movements, n_movement_choices):
    if max_movements:
        n_movements = np.random.choice(n_movement_choices)
    else:
        n_movements = None
    return n_movements


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_models',
        type=list,
        nargs='+',
        required=True,
        help='Paths to the models'
    )
    parser.add_argument(
        '--model_names',
        type=list,
        nargs='+',
        required=True,
        help='Names of the models'
    )
    parser.add_argument(
        '--path_to_results',
        type=str,
        required=True,
        help='Path to save results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        required=False,
        default=26,
        help='Value used to generate the games, it allows reproducible scenarios'
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
        '--max_actions_per_game',
        type=int,
        default=sys.maxsize,
        help='Maximum number of actions one single game can have'
    )

    args = parser.parse_args()
    locate_robot_close_goal = args.max_movements is not None

    self_play([''.join(path_model) for path_model in args.path_to_models],
              [''.join(model_name) for model_name in args.model_names],
              args.path_to_results, args.seed, args.number_games, args.rounds_per_action,
              locate_robot_close_goal, args.max_movements, args.max_actions_per_game)
