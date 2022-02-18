import argparse
import logging
import sys
import time
import uuid

import h5py
import numpy as np
import pandas as pd
from keras.optimizer_v2.gradient_descent import SGD

from src.agent.alphazero import AlphaZeroAgent
from src.encoders.maze_and_robot_positioning_encoder import MazeAndRobotPositioningEncoder
from src.experience.alphazero_experience import AlphaZeroExperienceCollector
from src.game_simulator.base import simulate_game
from src.robot_reboot.model import get_model_v2
from src.robot_reboot.classic_robot_reboot_hash import ClassicRobotRebootZobristHash
from src.robot_reboot.factory import RobotRebootFactory

logging.getLogger().setLevel(logging.INFO)


def experiment(seed, number_games, rounds_per_action, locate_robot_close_goal, max_movements, move_all_robots,
               max_actions_per_game):
    np.random.seed(seed)
    factory = RobotRebootFactory()
    results = list()
    experiment_id = str(uuid.uuid4())
    logging.info('Starting experiment' + experiment_id)
    for i in range(number_games):
        logging.info('Starting simulation' + str(i + 1) + '/' + str(number_games))
        start = time.time()
        game, game_state, selected_quadrants = factory.create(31, locate_robot_close_goal=locate_robot_close_goal,
                                                              max_movements=max_movements,
                                                              zobrist_hash_generator=ClassicRobotRebootZobristHash(),
                                                              move_all_robots=move_all_robots)
        encoder = MazeAndRobotPositioningEncoder(game)
        model = get_model_v2(encoder.shape(), len(game.actions))
        model.compile(
            SGD(learning_rate=0.01),
            loss=['categorical_crossentropy', 'mse'])
        alphazero_agent = AlphaZeroAgent(model, encoder, rounds_per_action=rounds_per_action)
        collector = AlphaZeroExperienceCollector()
        final_state = simulate_game(game_state, alphazero_agent, collector, max_actions=max_actions_per_game)
        total_time_seconds = time.time() - start
        logging.info('Simulation' + str(i + 1) + ' completed')
        value = final_state.get_value()
        total_actions = final_state.sequence_i

        result = {
            'experiment_id': experiment_id,
            'seed': seed,
            'number_games': number_games,
            'rounds_per_action': rounds_per_action,
            'locate_robot_close_goal': locate_robot_close_goal,
            'max_movements': max_movements,
            'move_all_robots': move_all_robots,
            'max_actions_per_game': max_actions_per_game,
            'time_sec': total_time_seconds,
            'value': value,
            'total_actions': total_actions,
        }
        results.append(result)
    logging.info('Saving experience data for experiment' + experiment_id)
    # buffer = collector.to_buffer()
    # with h5py.File(f'{experiment_id}-experience.hdf5', 'w') as experience_outf:
    #     buffer.serialize(experience_outf)

    logging.info('Saving results for experiment' + experiment_id)
    df = pd.DataFrame(results)
    df.to_csv(f'{experiment_id}-results.csv')
    logging.info('Experiment' + experiment_id + ' concluded')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
        '--move_all_robots',
        dest='move_all_robots',
        action='store_true',
        help='Determines if all the robots are moved when placing the robots away. This is only taken into '
             'consideration if a max_movements parameter is provided'
    )
    parser.set_defaults(move_all_robots=False)
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

    experiment(args.seed, args.number_games, args.rounds_per_action, locate_robot_close_goal, args.max_movements,
               args.move_all_robots, args.max_actions_per_game)
    # experiment(26, 30, 30, True, 1, False, 100)
