import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.alphazero.game import Game
from src.alphazero.monte_carlo_tree_search import MonteCarloTreeSearch


class DeepHeuristic(MonteCarloTreeSearch):

    def __init__(self,  input_shape: tuple, n_outputs: int, model_name="my_model.h5", invalid_action=-1):
        MonteCarloTreeSearch.__init__(self, invalid_action)
        self.model_name = model_name
        _input = tf.keras.layers.Input(input_shape, name='input')
        x = tf.keras.layers.Conv2D(64, 4, activation='relu', padding='same')(_input)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        win_output = tf.keras.layers.Dense(1, activation='softmax', name='win_output')(x)
        actions_output = tf.keras.layers.Dense(n_outputs, activation='softmax', name='actions_output')(x)

        self.model = tf.keras.Model(inputs=_input, outputs=[win_output, actions_output])
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam')
        self.checkpoint_cb = keras.callbacks.ModelCheckpoint(self.model_name)

    def record(self, game: Game, score):
        state = game.state()
        rows, cols, layers = state.shape
        dataset = state.reshape(1, rows, cols, layers)
        probabilities = self.get_actions_probabilities_mtcs_based(game, training=False)
        probabilities = [probabilities[i] for i in probabilities]
        self.model.fit(
            {'input': dataset},
            {
                'win_output': np.array([[score]]),
                'actions_output': np.array([probabilities])
            },
            callbacks=[self.checkpoint_cb]
        )

    def predict(self, game):
        state = game.state()
        rows, cols, layers = state.shape
        dataset = state.reshape(1, rows, cols, layers)
        return self.model.predict(dataset)

    def heuristic_value(self, game: Game):
        predict = self.predict(game)
        return predict[0][0][0]

    def action_probabilities(self, game):
        predict = self.predict(game)
        probabilities = predict[1][0]
        all_actions = game.get_all_actions()
        actions = {all_actions[i]: probabilities[i] for i in range(len(all_actions))}
        return actions

    def save_model(self, ):
        self.model.save(self.model_name)

    def load_model(self):
        self.model = keras.models.load_model(self.model_name)
