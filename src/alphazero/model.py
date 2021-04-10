from abc import ABC, abstractmethod
from .state import State
from .game import Game


class Model(ABC):
    """Model of a function that predicts a probability distribution among actions
    in the states of a game

    Attributes:
        model_name (string): name of the model
        game       (Game):   game where the model make predictions on
    """

    def __init__(self, model_name, game: Game):
        """ Initializes a model
        Args:
            model_name (string): game for the model
            game       (Game)  : game where the model makes predictions on

        """
        self.__model_name = model_name
        self.__game = game

    @property
    def model_name(self):
        return self.__model_name

    @property
    def game(self):
        return self.__game

    @abstractmethod
    def predict(self, state: State):
        """Makes a probability distribution based on the state over the actions

        Args:
            state (State): where we want to predict the action with the best outcome
        Returns:
            p (list):   probability of best possible action from the current state
            v (number): 1 if the game is predicted to win, 0 if it's a draw, -1 if its a lost
        """
        pass
