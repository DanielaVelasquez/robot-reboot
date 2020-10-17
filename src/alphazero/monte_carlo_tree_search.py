import numpy as np

from .game import Game
from .neural_network import NeuralNetwork

TREE_VISIT_COUNT = {}


class MonteCarloTreeSearch:

    def __init__(self, neural_network: NeuralNetwork, exploratory_parameter: int, max_tree_depth=10):
        self.neural_network = neural_network
        self.exploratory_parameter = exploratory_parameter
        self.max_tree_depth = max_tree_depth

    def playout_value(self, game: Game, depth: int):
        """Final game result after a set of actions taken on the current game

        Args:
            game: starting game
            depth: current tree depth exploration
        Returns:
            number: Determines possible game outcome from its current state

        """
        observation = game.observation()
        if hash(str(observation)) in TREE_VISIT_COUNT:
            TREE_VISIT_COUNT[hash(str(observation))] += 1
        else:
            TREE_VISIT_COUNT[hash(str(observation))] = 1

        # Leaf
        if game.is_over() or depth >= self.max_tree_depth:
            return game.score()

        actions_probabilities = self.neural_network.action_probabilities(game)
        best_action = None
        best_action_value = -1

        for action in game.get_all_actions():
            if action in game.get_valid_actions():
                # Action probability
                p = actions_probabilities[action]

                game.move(action)
                # State value after executing this action
                v = self.neural_network.heuristic_value(game)

                state_after_action = hash(str(observation))
                if state_after_action in TREE_VISIT_COUNT:
                    visit_count = TREE_VISIT_COUNT[state_after_action]
                else:
                    visit_count = 0

                action_value = v + self.exploratory_parameter * (visit_count + p)

                if best_action is None or best_action_value < action_value:
                    best_action = action
                    best_action_value = action_value

                game.undo_move()

        v = self.execute_action(game, best_action, depth)
        return v

    def execute_action(self, game, action, depth):
        game.move(action)
        value = self.playout_value(game, depth + 1)
        game.undo_move()
        return value

    def monte_carlo_value(self, game: Game, n):
        """Estimated  value of a state based on multiple playouts

        Args:
            game: Game to evaluate
            n: number of playouts
        Returns:
            number: Estimated value on how good current game is after multiple random games

        """
        return np.mean([self.playout_value(game, depth=0) for i in range(n)])

    def get_actions_probabilities(self, game: Game, playout_executions=30):
        """Probability of winning when executing each action based on current game status

        Args:
            game: Game to evaluate what is the best course of action from its current state
            playout_executions: number of playouts to evaluate a state
        Returns:
            dict: Each action and its chances to dealing toa win based on current game status
        """
        actions_probability = {}
        actions_v = {}
        valid_actions = game.get_valid_actions()
        for action in game.get_all_actions():
            if action in valid_actions:
                game.move(action)
                actions_probability[action] = self.monte_carlo_value(game, playout_executions)
                actions_v[action] = game.score()
                game.undo_move()
            else:
                actions_probability[action] = 0
                actions_v[action] = -1

        return actions_probability, actions_v

    def best_action(self, game):
        actions, values = self.get_actions_probabilities(game)
        return max(actions, key=actions.get)
