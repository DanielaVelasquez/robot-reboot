import numpy as np

from .gamelegacy import GameLegacy
from .neural_network import NeuralNetwork
from .edge import Edge


class MonteCarloTreeSearch:

    def __init__(self, neural_network: NeuralNetwork, exploratory_parameter: int, max_tree_depth=10):
        self.neural_network = neural_network
        self.exploratory_parameter = exploratory_parameter
        self.max_tree_depth = max_tree_depth
        self.edges = {}

    def playout_value(self, game: GameLegacy, depth: int):
        """Final game result after a set of actions taken on the current game

        Args:
            game: starting game
            depth: current tree depth exploration
        Returns:
            number: determines if the game was a win or a loss

        """
        # Leaf
        if game.is_over() or depth >= self.max_tree_depth:
            value = self.neural_network.heuristic_value(game)
            return game.score(), value

        best_action = self.get_best_action(game)

        if best_action is None:
            value = self.neural_network.heuristic_value(game)
            return game.score(), value

        game_score, value = self.execute_action(game, best_action, depth)
        return game_score, value

    def get_best_action(self, game):
        actions_probabilities = self.neural_network.action_probabilities(game)
        best_action = None
        best_action_value = -1
        current_state = game.observation()
        for action in game.get_all_actions():
            edge = self.__get_edge(action, current_state)
            p = actions_probabilities[action]
            if action not in game.get_valid_actions():
                p = 0

            action_value = edge.q + self.__get_exploration_value(p, edge.n)

            if best_action is None or best_action_value < action_value:
                best_action = action
                best_action_value = action_value

        return best_action

    def execute_action(self, game, action, depth):
        observation = game.observation()
        edge = self.__get_edge(action, observation)

        game.move(action)
        edge.count_visit()

        game_score, value = self.playout_value(game, depth + 1)
        edge.update_next_state_value(value)

        game.undo_move()
        return game_score, value

    def monte_carlo_value(self, game: GameLegacy, n):
        """Estimated  value of a state based on multiple playouts

        Args:
            game: Game to evaluate
            n: number of playouts
        Returns:
            number: Estimated value on how good current game is after multiple random games

        """
        results = [self.playout_value(game, depth=0) for i in range(n)]
        return np.mean([results[i][0] for i in range(n)]), np.mean([results[i][1] for i in range(n)])

    def get_actions_probabilities(self, game: GameLegacy, playout_executions=30):
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

        observation = game.observation()

        for action in game.get_all_actions():
            if action in valid_actions:
                game.move(action)

                actions_probability[action], avg_value = self.monte_carlo_value(game, playout_executions)
                self.__update_edge(observation, action, avg_value)
                edge = self.__get_edge(action, observation)
                edge.update_probability(actions_probability[action])
                actions_v[action] = edge.q
                game.undo_move()
            else:
                actions_probability[action] = -1
                actions_v[action] = -1

        return actions_probability, actions_v

    def best_action(self, game):
        actions_probabilities, values = self.get_actions_probabilities(game)
        best_action = None
        best_action_value = -1
        for action in game.get_all_actions():
            p = actions_probabilities[action]
            v = values[action]
            if p == -1:
                action_value = -1
            else:
                action_value = v + self.exploratory_parameter * p

            if best_action is None or best_action_value < action_value:
                best_action = action
                best_action_value = action_value

        return best_action

    def __register_visit(self, state, action):
        edge = self.__get_edge(action, state)
        edge.count_visit()

    def __update_edge(self, state, action, value):
        edge = self.__get_edge(action, state)
        edge.count_visit()
        edge.update_next_state_value(value)

    def __get_edge(self, action, state):
        state_edges = self.__get_state_edges(state)
        if action not in state_edges:
            state_edges[action] = Edge()
        return state_edges[action]

    def __get_state_edges(self, state):
        if hash(str(state)) not in self.edges:
            self.edges[hash(str(state))] = {}
        state_edges = self.edges[hash(str(state))]
        return state_edges

    def __get_exploration_value(self, p, n):
        """ Function that increases if an edge hasn't been explored much

        Args:
             p: edge probability
             n: number of visits for an edge
        Return:
            number: value how much exploration is needed
        """
        if n == 0:
            n = 0.01
        return self.exploratory_parameter * (p / n)
