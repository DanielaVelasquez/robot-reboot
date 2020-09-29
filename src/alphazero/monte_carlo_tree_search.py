from abc import ABC, abstractmethod
import numpy as np

from .game import Game

TREE_VISIT_COUNT = {}


class MonteCarloTreeSearch(ABC):

    def __init__(self, invalid_action=-1):
        self.invalid_action = invalid_action
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

    @abstractmethod
    def action_probabilities(self, game):
        """Computes probability to execute each action on current game

        Args:
            game: Current game to evaluate state on

        Returns::
            list: Probability of winning for each action
        """
        pass

    def playout_value(self, game: Game, training):
        """Final game result after a set of actions taken on the current game

        Args:
            game: starting game
            training: Determines if the tree is being trained
        Returns:
            number: Determines possible game outcome from its current state

        """
        results = list()
        if hash(str(game.state())) in TREE_VISIT_COUNT:
            TREE_VISIT_COUNT[hash(str(game.state()))] += 1
        else:
            TREE_VISIT_COUNT[hash(str(game.state()))] = 1

        if game.is_over():
            if training:
                self.record(game, game.score())
            return game.score()

        actions = {}
        actions_heuristic_value = {}
        action_visit_count = {}
        for action in game.get_valid_actions():
            game.move(action)
            actions[action] = action
            actions_heuristic_value[action] = self.heuristic_value(game)
            state_after_action = hash(str(game.state()))
            if state_after_action in TREE_VISIT_COUNT:
                action_visit_count[action] = TREE_VISIT_COUNT[state_after_action]
            else:
                action_visit_count[action] = 0

            game.undo_move()

        best_action_heuristic_value = actions[max(actions_heuristic_value, key=actions_heuristic_value.get)]
        results.append(self.execute_action(game, best_action_heuristic_value, training))

        del action_visit_count[best_action_heuristic_value]
        best_action_visit_count = actions[min(action_visit_count, key=action_visit_count.get)]
        results.append(self.execute_action(game, best_action_visit_count, training))

        if training:
            actions_mcts = self.get_actions_probabilities_mtcs_based(game, training=False)
        else:
            actions_mcts = self.action_probabilities(game)
        actions_mcts = dict(filter(lambda a: a[1] != self.invalid_action and a[0] != best_action_heuristic_value and a[
            0] != best_action_visit_count, actions_mcts.items()))
        if actions_mcts:
            best_action_mcts = max(actions_mcts, key=actions_mcts.get)
            results.append(self.execute_action(game, best_action_mcts, training))

        return np.mean(results)

    def execute_action(self, game, action, training):
        game.move(action)
        value = self.playout_value(game, training)
        game.undo_move()
        if training:
            self.record(game, value)
        return value

    def monte_carlo_value(self, game: Game, training=True):
        """Estimated  value of a state based on multiple playouts

        Args:
            game: Game to evaluate
            n: number of playouts
            training: Determines if the tree is being trained
        Returns:
            number: Estimated value on how good current game is after multiple random games

        """
        # return np.mean([self.playout_value(game, training) for i in range(n)])
        return self.playout_value(game, training)

    def get_actions_probabilities_mtcs_based(self, game: Game, training=True):
        """Probability of winning when executing each action based on current game status

        Args:
            game: Game to evaluate what is the best course of action from its current state
            training: Determines if the tree is being trained
        Returns:
            dict: Each action and its chances to dealing toa win based on current game status

        """
        actions = {}
        valid_actions = game.get_valid_actions()
        for action in game.get_all_actions():
            if action in valid_actions:
                game.move(action)
                actions[action] = self.monte_carlo_value(game, training)
                game.undo_move()
            else:
                actions[action] = -1

        return actions

    def best_action(self, game, training=True):
        actions = self.get_actions_probabilities_mtcs_based(game, training)
        return max(actions, key=actions.get)
