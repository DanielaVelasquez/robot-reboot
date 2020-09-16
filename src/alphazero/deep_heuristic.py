from tensorflow import keras
import tensorflow as tf
import numpy as np

from src.alphazero.game import Game
from src.alphazero.monte_carlo_tree_search import MonteCarloTreeSearch


class DeepHeuristic(MonteCarloTreeSearch):

    def __init__(self,  input_shape: tuple, n_outputs: int):
        self.model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=input_shape),
            keras.layers.Conv2D(64, 4, activation='relu', padding='same'),
            keras.layers.MaxPooling2D(2),
            # keras.layers.Conv2D(64, 4, activation='relu', padding='same'),
            # keras.layers.MaxPooling2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(n_outputs)
        ])
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.checkpoint_cb = keras.callbacks.ModelCheckpoint("my_model.h5")

    def record(self, game: Game, score):
        state = game.state()
        rows, cols, layers = state.shape
        dataset = tf.data.Dataset.from_tensor_slices(([state.reshape(1, rows, cols, layers)], np.array([[score]])))
        self.model.fit(dataset, callbacks=[self.checkpoint_cb])

    def heuristic_value(self, game: Game):
        state = game.state()
        rows, cols, layers = state.shape
        dataset = state.reshape(1, rows, cols, layers)
        predict = self.model.predict(dataset)
        return np.max(predict)

    def save_model(self, model_name="my_model.h5"):
        self.model.save(model_name)

    def load_model(self, model_name="my_model.h5"):
        self.model = keras.models.load_model(model_name)
