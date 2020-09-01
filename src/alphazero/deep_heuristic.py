from tensorflow import keras

from src.alphazero.game import Game
from src.alphazero.monte_carlo_tree_search import MonteCarloTreeSearch


class DeepHeuristic(MonteCarloTreeSearch):

    def __init__(self, game: Game, input_shape: tuple, n_outputs: tuple):
        MonteCarloTreeSearch.__init__(game)
        self.model = keras.models.Sequential([
            keras.layers.Conv2D(64, 4, activation='relu', padding='same', input_shape=input_shape),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(64, 4, activation='relu', padding='same', input_shape=input_shape),
            keras.layers.MaxPooling2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(n_outputs, activation='softmax')
        ])

    def record(self, game: Game, score):
        dataset = [{'input': game.state(), 'target': score}]
        self.model.fit(dataset, batch_size=1)

    def heuristic_value(self, game: Game):
        dataset = [{'input': game.state(), 'target': None}]
        return self.model.predict(dataset)
