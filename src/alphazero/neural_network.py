from abc import ABC, abstractmethod
from .game_legacy import GameLegacy


class NeuralNetwork(ABC):
    @abstractmethod
    def record(self, game: GameLegacy, score):
        """Stores data to learn heuristic values

        Args:
             game: Current game
             score: Game score based on current state
        """
        pass

    @abstractmethod
    def heuristic_value(self, game: GameLegacy):
        """Computes a heuristic value that determines how worthy a state is

        Args:
            game: Current game to evaluate state on

        Returns::
            number: Approximation on how good is the game's current state
        """
        pass

    @abstractmethod
    def action_probabilities(self, game):
        """Computes probability to execute each action on current game

        Args:
            game: Current game to evaluate state on

        Returns::
            list: Probability of winning for each action
        """
        pass
