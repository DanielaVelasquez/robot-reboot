import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.alphazero.game_legacy import GameLegacy
from src.alphazero.neural_network import NeuralNetwork


class DeepHeuristic(NeuralNetwork):

    def __init__(self,  input_shape: tuple, n_outputs: int, model_name="my_model.h5"):
        NeuralNetwork.__init__(self)
        self.model_name = model_name
        _input = tf.keras.layers.Input(input_shape, name='input')
        x = tf.keras.layers.Conv2D(64, 4, activation='relu', padding='same')(_input)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Conv2D(64, 4, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Conv2D(64, 4, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        value_output = tf.keras.layers.Dense(1, activation='sigmoid', name='value_output')(x)
        # THIS CAN't BE SOFTMAX, We need sigmad or somerhing
        # THIS CALCULATES THE PROBABILITY OF WINNING By taking an action
        ## -1 loss, 0 draw, 1 win
        probabilities_output = tf.keras.layers.Dense(n_outputs, activation='softmax', name='probabilities_output')(x)

        self.model = tf.keras.Model(inputs=_input, outputs=[value_output, probabilities_output])
        losses = {
            "value_output": 'mean_squared_error',
            "probabilities_output": tf.keras.losses.BinaryCrossentropy()
        }
        self.model.compile(loss=losses, optimizer='adam')
        self.checkpoint_cb = keras.callbacks.ModelCheckpoint(self.model_name)

    def record(self, game: GameLegacy, score):
        state = game.observation()
        rows, cols, layers = state.shape
        dataset = state.reshape(1, rows, cols, layers)
        probabilities = self.get_actions_probabilities(game, training=False)
        probabilities = [probabilities[i] for i in probabilities]
        self.model.fit(
            {'input': dataset},
            {
                'value_output': np.array([[score]]),
                'probabilities_output': np.array([probabilities])
            },
            callbacks=[self.checkpoint_cb]
        )

    def predict(self, game):
        state = game.observation()
        rows, cols, layers = state.shape
        dataset = state.reshape(1, rows, cols, layers)
        return self.model.predict(dataset)

    def heuristic_value(self, game: GameLegacy):
        predict = self.predict(game)
        return predict[0][0][0]

    def action_probabilities(self, game):
        predict = self.predict(game)
        probabilities = predict[1][0]
        all_actions = game.get_all_actions()
        probabilities = {all_actions[i]: probabilities[i] for i in range(len(all_actions))}
        return probabilities

    def save_model(self, ):
        self.model.save(self.model_name)

    def load_model(self):
        self.model = keras.models.load_model(self.model_name)
