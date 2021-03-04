import numpy as np
from src.alphazero.action import Action
from src.alphazero.game import Game
from src.alphazero.model import Model
from src.alphazero.state import State


class FakeAction(Action):

    def __init__(self, value):
        Action.__init__(self)
        self.__value = value

    @property
    def value(self):
        return self.__value

    def __str__(self):
        return f'a{self.__value}'


class FakeState(State):

    def __init__(self, game, value, sequence_i=0):
        State.__init__(self, game, sequence_i)
        self.__value = value

    @property
    def value(self):
        return self.__value

    def __str__(self):
        return f's{self.__value}'


class FakeModel(Model):

    def __init__(self, fn_predict_probability):
        Model.__init__(self, "fake_model", FakeGame())
        self.__fn_predict_probability = fn_predict_probability

    def predict(self, state: State):
        return self.__fn_predict_probability(len(self.game.actions), state), np.nan

    def train(self, train_x, train_y, test_x, test_y):
        pass


class FakeGame(Game):

    def __init__(self):
        Game.__init__(self, [FakeAction(i + 1) for i in range(4)])
        self.valid_actions_dict = None

    def get_value(self, state: FakeState):
        if state.value == 3:
            return 1
        elif state.value == 1:
            return 0
        else:
            return -1

    def is_over(self, state: FakeState):
        return state.value == 3

    def get_score(self, state: FakeState):
        return state.sequence_i

    def apply(self, action: FakeAction, state: FakeState):
        """It goes to the state with the same action value
        E.g if action.value = 1 then it'll go to state.value =1 regardless of the starting state
        """
        return FakeState(self, action.value, state.sequence_i + 1)

    def get_valid_actions(self, state: FakeState):
        if self.valid_actions_dict is not None and state.value in self.valid_actions_dict:
            return self.valid_actions_dict[state.value]
        return self.actions


def fn_predict_probability_1_for_next_action(actions_size, state: FakeState):
    """" Fake predict probabilities of winning over each action. Probabilities are set to 0 except for one action.
    E.g state = 1, actions_size = 3 then p = [0, 1, 0]
    E.g state = 3, actions_size = 3 then p = [1, 0, 0]
    (Function used in the FakeModel)
    Args:
        actions_size (int):   number of actions
        state        (State): state to calculate the probabilities of winning per actions
    Returns:
        p (np array): probability over actions. All actions with 0 except for one action with probability of 1
    """
    p = np.zeros(actions_size, dtype=float)
    p[state.value % actions_size] = 1
    return p
