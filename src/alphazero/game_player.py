from abc import ABC, abstractmethod
from .state import State
from .model import Model
from .game import Game


class GamePlayer(ABC):
    """ Player for a game, it knows how to interpreted the model results and apply them
    to a game.

    Attributes:
        model (Model): Function to predict best possible actions on the game
        game  (Game):  Game to play

    """

    def __init__(self, model: Model, game: Game):
        """ Initializes a game
        Args:
            model (Model): Function to predict best possible action on the game
            game  (Game):  Game to play

        """
        self.__model = model
        self.__game = game

    @property
    def model(self):
        return self.__model

    @property
    def game(self):
        return self.__game

    @abstractmethod
    def play(self, state: State):
        """
        Plays the game starting from an initial state until the game is over based on the predictions
        made by the model.
        Args:
             state (State): initial state of the game
        Returns:
            state   (State): final state of the game
            actions (List):  List of actions taken from the initial state to the end state
        """
        pass

    @abstractmethod
    def predict(self, state):
        """
        Predicts the probability distribution and value of a state based on the model respond
        Args:
            state (State): state to evaluate
        Returns
            p (list):   Predicted probability distribution over each action
            v (number): Predicted value
        """
        pass
