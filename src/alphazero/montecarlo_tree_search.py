import numpy as np

from src.alphazero.state import State
from .game_player import GamePlayer


class MonteCarloTreeSearch:
    """ Applies monte carlo tree search using a game player to find
    the best possible actions to take based on a state
    Attributes:
        exploratory_parameter (number):     value [0,1] determines how important it is to explore new edges in the tree
        max_depth             (number):     maximum depth for the tree while searching
        game_player           (GamePlayer): player for the game to optimize moves
        playouts              (number):     number of playouts per simulation
        edges                 (dict):       dictionary mapping visited states to a dictionary of actions taken from that
                                            state
                                            {
                                                s0: {
                                                    a1: edge
                                                    a2: edge
                                                },
                                                s1: {
                                                    a2: edge
                                                }
                                            }
    """

    def __init__(self, exploratory_parameter, max_depth, game_player: GamePlayer, playouts=100):
        """
        Initializes a MonteCarlo tree search
        Args:
            exploratory_parameter (number):     value [0,1] determines how important it is to explore new edges in the tree
            max_depth             (number):     maximum depth for the tree while searching
            game_player           (GamePlayer): player for the game to optimize moves
            playouts              (number):     number of playouts per simulation (default 100)
        """
        assert 0 <= exploratory_parameter <= 1, "exploratory parameter must be between [0,1]"
        assert max_depth > 0, "Tree depth must be greater than 0"
        assert game_player is not None, "game_player must be provided"
        assert playouts > 0, "playouts must be greater than 0"

        self.__exploratory_parameter = exploratory_parameter
        self.__max_depth = max_depth
        self.__game_player = game_player
        self.__game = self.__game_player.game
        self.__playouts = playouts
        self.__edges = {}

    @property
    def exploratory_parameter(self):
        return self.__exploratory_parameter

    @property
    def max_depth(self):
        return self.__max_depth

    @property
    def game_player(self):
        return self.__game_player

    @property
    def playouts(self):
        return self.__playouts

    def search(self, state: State):
        p = np.empty(len(self.__game.actions), dtype=float)
        for i in range(len(self.__game.actions)):
            a = self.__game.actions[i]
            next_state = self.__game_player.game.apply(a, state)
            p[i] = self.__simulations(next_state)
        return p

    def __simulations(self, state: State, n):
        v = np.array([self.__playout(state) for _ in range(n)])
        return v.mean()

    def __playout(self, state: State, depth=0):
        if self.__game.is_over(state) or depth > self.__max_depth:
            return self.__game.get_value(state)
        p, v = self.__game_player.predict(state)

        heuristic_values = self.__fn_heurictic_value(p, v)
        a = best(
            heuristic_values)  # If we make this a list then getting the righ action might get tricky, could use a map between index: action
        next_state = self.__game.apply(a, state)
        v = self.__playouts(next_state, depth + 1)
        self.__update_edge(state, a, v)
        return v

    def __fn_heurictic_value(self, p, v):
        """
        returns a dictionary
        @param p:
        @param v:
        @return:
        """
        pass
        # TODO: pass this funciton as a parameter that can take p, v and q?

    def __update_edge(self, state, action, v):
        if state not in self.__edges:
            self.__edges[state] = {}

        if action in self.__edges[state]:
            self.__edges[state][action].w += v
            self.__edges[state][action].n += 1
        else:
            self.__edges[state][action].w = v
            self.__edges[state][action].n = 1
