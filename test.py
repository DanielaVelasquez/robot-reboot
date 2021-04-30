import tensorflow as tf
import numpy as np
from src.ml.model import get_model
import pandas as pd

from src.alphazero.game_player import GamePlayer

from src.robot_reboot.game import get_game_from_matrix
from src.alphazero.game_player import GamePlayer
from src.robot_reboot.model import RobotRebootModel
from src.ml.util import get_test_data

states = get_test_data(['robot_reboot_data/test.tfrecords'])


def evaluate(cnn):
    history = {
        'history': {
            'score': list(),
            'outcome': list()
        }
    }

    MAX_DEPTH = 20
    for s in states:
        game, state = get_game_from_matrix(s)
        model = RobotRebootModel(game, cnn)
        game_player = GamePlayer(model, game)
        final_state = game_player.play(state, MAX_DEPTH)
        history['history']['outcome'].append(game.get_value(final_state))
        history['history']['score'].append(game.get_score(final_state))
    return history


def summary(history):
    score = history['history']['score']
    outcome = history['history']['outcome']
    return sum(outcome), sum(score) / len(score)


def get_df(histories):
    data = list()
    i = 0
    for h in histories:
        wins, avg_score = summary(h)
        data.append({'Model': i, 'Wins': wins, 'Average Score': avg_score})
        i += 1
    return pd.DataFrame(data=data)


model_2 = get_model()
model_2.load_weights('robot_reboot_model/model_2/')

from src.alphazero.alphazero import AlphaZero

alphazero_history = {
    'history': {
        'score': list(),
        'outcome': list()
    }
}
def main():
    MAX_DEPTH = 20
    i = 0
    for s in states:
        print(f'{i}-th state')
        game, state = get_game_from_matrix(s)
        model = RobotRebootModel(game, model_2)
        game_player_alphazero = GamePlayer(model, game)
        alphazero = AlphaZero(MAX_DEPTH, game_player_alphazero, playouts=50)
        game_player = GamePlayer(alphazero, game)

        final_state = game_player.play(state, MAX_DEPTH)
        alphazero_history['history']['outcome'].append(game.get_value(final_state))
        alphazero_history['history']['score'].append(game.get_score(final_state))
        i += 1


if __name__ == '__main__':
    main()