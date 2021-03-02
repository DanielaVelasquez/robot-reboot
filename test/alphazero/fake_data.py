from src.alphazero.action import Action
from src.alphazero.game import Game
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

    def __init__(self, sequence_i, game, value):
        State.__init__(self, game, sequence_i)
        self.__value = value

    @property
    def value(self):
        return self.__value

    def __str__(self):
        return f's{self.__value}'


class FakeGame(Game):
    def __init__(self):
        Game.__init__(self, [FakeAction(i + 1) for i in range(4)])

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
        return FakeState(state.sequence_i + 1, self, action.value)