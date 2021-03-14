import numpy as np

from exceptions.exceptions import RequiredValueException
from exceptions.util import assertOrThrow
from src.alphazero.game import Game
from src.alphazero.model import Model
from src.alphazero.state import State


class GamePlayer:
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
        assertOrThrow(model is not None, RequiredValueException("model"))
        assertOrThrow(game is not None, RequiredValueException("game"))
        self.__model = model
        self.__game = game

    @property
    def model(self):
        return self.__model

    @property
    def game(self):
        return self.__game

    def play(self, state: State, max_actions=100):
        """
        Plays the game starting from an initial state until the game is over based on the predictions
        made by the model or a max number of actions are performed without success
        Args:
             state        (State): initial state of the game
             max_actions  (int):   max number of actions to perform in the game
        Returns:
            state   (State): final state of the game
        """
        return self.__play(state, 0, max_actions)

    def __play(self, state: State, actions_count, max_actions):
        """
        Plays a game recursively until the game is finished or the max number of actions are performed on the game
        Args:
            state         (State): state to apply an action
            actions_count (int):   number of actions performed
            max_actions   (int):   max number of actions to play
        Returns:
            state   (State): final state of the game
        """
        if self.__game.is_over(state) or actions_count >= max_actions:
            return state
        p, v = self.predict(state)
        i_best = np.argsort(p)[::-1][0]
        action = self.__game.actions[i_best]

        next_state = self.__game.apply(action, state)
        return self.__play(next_state, actions_count + 1, max_actions)

    def predict(self, state: State):
        """
        Predicts the probability distribution and value of a state based on the model respond
        Args:
            state (State): state to evaluate
        Returns
            v (number):     Predicted value
            p (np array):   Predicted probability distribution over each action
        """
        assertOrThrow(state is not None, RequiredValueException("state"))
        return self.__model.predict(state)
