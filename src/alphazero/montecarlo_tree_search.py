from .game_player import GamePlayer
# from src.alphazero.state import State


class MonteCarloTreeSearch:
    """ Applies monte carlo tree search using a game player to find
    the best possible actions to take based on a state
    Attributes:
        exploratory_parameter (number):     value [0,1] determines how important it is to explore new edges in the tree
        max_depth             (number):     maximum depth for the tree while searching
        game_player           (GamePlayer): player for the game to optimize moves
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

    def __init__(self, exploratory_parameter, max_depth, game_player: GamePlayer):
        """
        Initializes a MonteCarlo tree search
        Args:
            exploratory_parameter (number):     value [0,1] determines how important it is to explore new edges in the tree
            max_depth             (number):     maximum depth for the tree while searching
            game_player           (GamePlayer): player for the game to optimize moves
        """
        assert 0 <= exploratory_parameter <= 1, "exploratory parameter must be between [0,1]"
        assert max_depth > 0, "Tree depth must be greater than 0"
        assert game_player is not None, "game_player must be provided"

        self.__exploratory_parameter = exploratory_parameter
        self.__max_depth = max_depth
        self.__game_player = game_player
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

    # def search(self, state: State):
    #     p = {}
