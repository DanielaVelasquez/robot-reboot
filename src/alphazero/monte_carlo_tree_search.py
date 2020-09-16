from abc import ABC, abstractmethod
import numpy as np

from .game import Game


class MonteCarloTreeSearch(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def record(self, game: Game, score):
        """Stores data to learn heuristic values

        Args:
             game: Current game
             score: Game score based on current state
        """
        pass

    @abstractmethod
    def heuristic_value(self, game: Game):
        """Computes a heuristic value that determines how worthy a state is

        Args:
            game: Current game to evaluate state on

        Returns::
            number: Approximation on how good is the game's current state
        """
        pass

    def playout_value(self, game: Game):
        """Final game result after a set of actions taken on the current game

        Args:
            game: starting game

        Returns:
            number: Determines possible game outcome from its current state

        """
        if game.is_over():
            self.record(game, game.score())
            return game.score()

        actions = {}
        actions_heuristic_value = {}
        for action in game.get_valid_actions():
            game.move(action)
            actions[action] = action
            actions_heuristic_value[action] = self.heuristic_value(game)
            game.undo_move()
        best_action = actions[max(actions_heuristic_value, key=actions_heuristic_value.get)]
        game.move(best_action)
        value = self.playout_value(game)
        game.undo_move()
        self.record(game, value)
        return value

    def monte_carlo_value(self, game: Game, n=100):
        """Estimated  value of a state based on multiple playouts

        Args:
            game: Game to evaluate
            n: number of playouts
        Returns:
            number: Estimated value on how good current game is after multiple random games

        """
        return np.mean([self.playout_value(game) for i in range(n)])

    def best_action(self, game: Game):
        """Best action to take based on current game status

        Args:
            game: Game to evaluate what is the best course of action from its current state

        Returns:
            Best action to take on current game state based on executions

        """
        actions = {}
        for action in game.get_valid_actions():
            game.move(action)
            actions[action] = self.monte_carlo_value(game)
            game.undo_move()
        return max(actions, key=actions.get)

    def best_action_eval(self, game: Game):
        """Best action to take based on current game status evaluating, not training anything

        Args:
            game: Game to evaluate what is the best course of action from its current state

        Returns:
            Best action to take on current game state based on executions

        """
        actions = {}
        for action in game.get_valid_actions():
            game.move(action)
            actions[action] = self.heuristic_value(game)
            game.undo_move()
        return max(actions, key=actions.get)
